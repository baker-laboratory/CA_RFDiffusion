from __future__ import annotations
import copy
import torch
import numpy as np 

from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.scoring import *

import rf2aa.kinematics
import rf2aa.util
import networkx as nx 

def any_a_in_b(a: list, b: list): 
    # check if any element in a is in b
    return any([x in b for x in a])

def get_sm_lengths(is_sm: np.array, is_same_chain: np.array):
    """
    Returns the lengths of the small molecules. 

    Args:
        is_sm: true if token is small molecule token 
        is_sam_chain: 2d array designating same chain or not
    """
    where_is_sm = np.where(is_sm)[0]

    G = nx.from_numpy_array(is_same_chain)

    # make mask denoting which cliques are SM cliques? 
    cliques = list(nx.find_cliques(G))
    is_sm_clique = [any_a_in_b(a, where_is_sm) for a in cliques]
    sm_cliques = [c for c, is_sm_c in zip(cliques, is_sm_clique) if is_sm_c]
    return [len(c) for c in sm_cliques]

def generate_Cbeta(N,Ca,C):
    # recreate Cb given N,Ca,C
    b = Ca - N 
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    #Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fd: below matches sidechain generator (=Rosetta params)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb

def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

init_N = ChemData().init_N
init_C = ChemData().init_C
norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N*norm_C, dim=-1) # cosine of ideal N-CA-C bond angle

# More complicated version splits error in CA-N and CA-C (giving more accurate CB position)
# It returns the rigid transformation from local frame to global frame
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.sum(e1*v2, dim=-1) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca

def get_tor_mask(seq, torsion_indices, mask_in=None):
    B,L = seq.shape[:2]
    tors_mask = torch.ones((B,L,10), dtype=torch.bool, device=seq.device)
    tors_mask[...,3:7] = torsion_indices[seq,:,-1] > 0
    tors_mask[:,0,1] = False
    tors_mask[:,-1,0] = False

    # mask for additional angles
    tors_mask[:,:,7] = seq!=ChemData().aa2num['GLY']
    tors_mask[:,:,8] = seq!=ChemData().aa2num['GLY']
    tors_mask[:,:,9] = torch.logical_and( seq!=ChemData().aa2num['GLY'], seq!=ChemData().aa2num['ALA'] )
    tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=ChemData().aa2num['UNK'] )
    tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=ChemData().aa2num['MAS'] )

    if mask_in is not None:
        # mask for missing atoms
        # chis
        ti0 = torch.gather(mask_in,2,torsion_indices[seq,:,0])
        ti1 = torch.gather(mask_in,2,torsion_indices[seq,:,1])
        ti2 = torch.gather(mask_in,2,torsion_indices[seq,:,2])
        ti3 = torch.gather(mask_in,2,torsion_indices[seq,:,3])
        is_valid = torch.stack((ti0, ti1, ti2, ti3), dim=-2).all(dim=-1)
        tors_mask[...,3:7] = torch.logical_and(tors_mask[...,3:7], is_valid)
        tors_mask[:,:,7] = torch.logical_and(tors_mask[:,:,7], mask_in[:,:,4]) # CB exist?
        tors_mask[:,:,8] = torch.logical_and(tors_mask[:,:,8], mask_in[:,:,4]) # CB exist?
        tors_mask[:,:,9] = torch.logical_and(tors_mask[:,:,9], mask_in[:,:,5]) # XG exist?

    return tors_mask

def get_torsions(xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, mask_in=None):
    B,L = xyz_in.shape[:2]
    
    tors_mask = get_tor_mask(seq, torsion_indices, mask_in)
    
    # torsions to restrain to 0 or 180degree
    tors_planar = torch.zeros((B, L, 10), dtype=torch.bool, device=xyz_in.device)
    tors_planar[:,:,5] = seq == ChemData().aa2num['TYR'] # TYR chi 3 should be planar

    # idealize given xyz coordinates before computing torsion angles
    xyz = xyz_in.clone()
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:])
    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)
    xyz[...,0,:] = torch.einsum('brij,j->bri', Rs, Nideal) + Ts
    xyz[...,2,:] = torch.einsum('brij,j->bri', Rs, Cideal) + Ts

    torsions = torch.zeros( (B,L,10,2), device=xyz.device )
    # avoid undefined angles for H generation
    torsions[:,0,1,0] = 1.0
    torsions[:,-1,0,0] = 1.0

    # omega
    torsions[:,:-1,0,:] = th_dih(xyz[:,:-1,1,:],xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:])
    # phi
    torsions[:,1:,1,:] = th_dih(xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:],xyz[:,1:,2,:])
    # psi
    torsions[:,:,2,:] = -1 * th_dih(xyz[:,:,0,:],xyz[:,:,1,:],xyz[:,:,2,:],xyz[:,:,3,:])

    # chis
    ti0 = torch.gather(xyz,2,torsion_indices[seq,:,0,None].repeat(1,1,1,3))
    ti1 = torch.gather(xyz,2,torsion_indices[seq,:,1,None].repeat(1,1,1,3))
    ti2 = torch.gather(xyz,2,torsion_indices[seq,:,2,None].repeat(1,1,1,3))
    ti3 = torch.gather(xyz,2,torsion_indices[seq,:,3,None].repeat(1,1,1,3))
    torsions[:,:,3:7,:] = th_dih(ti0,ti1,ti2,ti3)
    
    # CB bend
    NC = 0.5*( xyz[:,:,0,:3] + xyz[:,:,2,:3] )
    CA = xyz[:,:,1,:3]
    CB = xyz[:,:,4,:3]
    t = th_ang_v(CB-CA,NC-CA)
    t0 = ref_angles[seq][...,0,:]
    torsions[:,:,7,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    # CB twist
    NCCA = NC-CA
    NCp = xyz[:,:,2,:3] - xyz[:,:,0,:3]
    NCpp = NCp - torch.sum(NCp*NCCA, dim=-1, keepdim=True)/ torch.sum(NCCA*NCCA, dim=-1, keepdim=True) * NCCA
    t = th_ang_v(CB-CA,NCpp)
    t0 = ref_angles[seq][...,1,:]
    torsions[:,:,8,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )

    # CG bend
    CG = xyz[:,:,5,:3]
    t = th_ang_v(CG-CB,CA-CB)
    t0 = ref_angles[seq][...,2,:]
    torsions[:,:,9,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    mask0 = torch.isnan(torsions[...,0]).nonzero()
    mask1 = torch.isnan(torsions[...,1]).nonzero()
    torsions[mask0[:,0],mask0[:,1],mask0[:,2],0] = 1.0
    torsions[mask1[:,0],mask1[:,1],mask1[:,2],1] = 0.0

    # alt chis
    torsions_alt = torsions.clone()
    torsions_alt[torsion_can_flip[seq,:]] *= -1

    return torsions, torsions_alt, tors_mask, tors_planar

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    mask = ~(torch.isnan(xyz_tips[:,:,0]))
    if torch.isnan(xyz_tips).any(): # replace NaN tip atom with virtual Cb atom
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

        xyz_tips = torch.where(torch.isnan(xyz_tips), Cb, xyz_tips)
    return xyz_tips, mask

# process ideal frames
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)

    return torch.stack((Xn,Yn,Zn), dim=-1)

def cross_product_matrix(u):
    B, L = u.shape[:2]
    matrix = torch.zeros((B, L, 3, 3), device=u.device)
    matrix[:,:,0,1] = -u[...,2]
    matrix[:,:,0,2] = u[...,1]
    matrix[:,:,1,0] = u[...,2]
    matrix[:,:,1,2] = -u[...,0]
    matrix[:,:,2,0] = -u[...,1]
    matrix[:,:,2,1] = u[...,0]
    return matrix

# writepdb
def writepdb(filename, atoms, seq, binderlen=None, idx_pdb=None, bfacts=None, chain_idx=None):
    f = open(filename,"w")
    ctr = 1
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    for i,s in enumerate(scpu):
        if chain_idx is None:
            if binderlen is not None:
                if i < binderlen:
                    chain = 'A'
                else:
                    chain = 'B'
            else:
                chain = 'A'
        else:
            chain = chain_idx[i]
        if (len(atomscpu.shape)==2):
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, " CA ", ChemData().num2aa[s],
                    chain, idx_pdb[i], atomscpu[i,0], atomscpu[i,1], atomscpu[i,2],
                    1.0, Bfacts[i] ) )
            ctr += 1
        elif atomscpu.shape[1]==3:
            for j,atm_j in enumerate([" N  "," CA "," C  "]):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, ChemData().num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                ctr += 1
        else:
            natoms = atomscpu.shape[1]
            if (natoms!=14 and natoms!=27):
                assert False, f'bad size: {atoms.shape}, must be [L, 14|27,...]'
            atms = ChemData().aa2long[s]
            # his prot hack
            if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
                atms = (
                    " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                      None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                    " HD1",  None,  None,  None,  None,  None,  None) # his_d

            for j,atm_j in enumerate(atms):
                if (j<natoms and atm_j is not None): # and not torch.isnan(atomscpu[i,j,:]).any()):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, ChemData().num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                    ctr += 1

"""
# resolve tip atom indices
tip_indices = torch.full((22,), 0)
for i in range(22):
    tip_atm = ChemData().aa2tip[i]
    atm_long = ChemData().aa2long[i]
    tip_indices[i] = atm_long.index(tip_atm)

# resolve torsion indices
torsion_indices = torch.full((22,4,4),0)
torsion_can_flip = torch.full((22,10),False,dtype=torch.bool)
for i in range(22):
    i_l, i_a = ChemData().aa2long[i], ChemData().aa2longalt[i]
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i,j,k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i,3+j] = True ##bb tors never flip
# HIS is a special case
torsion_can_flip[8,4]=False
"""

def get_mu_xt_x0(xt, px0, t, schedule, alphabar_schedule, eps=1e-6):
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1

    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*schedule[t_idx])/(1-alphabar_schedule[t_idx]))*px0_ca
    b = ((torch.sqrt(1-schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))*xt_ca
    mu = a + b

    return mu, sigma

def get_t2d(
        xyz_t: torch.Tensor, 
        is_sm: torch.Tensor, 
        atom_frames: torch.Tensor, 
        use_cb: bool = False, 
        mask_t_2d = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the t2d features for given backbone (+cb) coordinates.

    Args:
        xyz_t (torch.Tensor): Atom coordinates of shape [T, L, 36, 3], where T is the number of
            templates, L is the sequence length, and 36 is the number of atoms per residue.
        is_sm (torch.Tensor): Boolean tensor of shape [L] indicating which residues are part of the
            structural motif.
        atom_frames (torch.Tensor): Atom frames of shape [F, 3, 2], where F is the number of frames.
        use_cb (bool, optional): Whether to use CB atoms in t2d computation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - t2d (torch.Tensor): The computed t2d features of shape [L, L, C], where C is the
                number of t2d channels.
                The t2d channels contain:
                - distance bins
                - angle sin/cos
                - mask values
            - mask_t_2d (torch.Tensor): A dummy mask of shape [1, L, L], set to all True 
                for rf diffusion.

    Note:
        The 'mask_t_2d' is set to all True because non-existing atoms are removed from the structure
        upstream of this function.
    """
    L = xyz_t.shape[1]

    # TODO(Look into atom_frames)
    xyz_t_frames = rf2aa.util.xyz_t_to_frame_xyz_sm_mask(xyz_t[None], is_sm, atom_frames[None])

    if mask_t_2d is None:
        mask_t_2d = torch.ones(1,L,L).bool().to(xyz_t_frames.device)
    else: 
        pass 

    kinematics_params = copy.deepcopy(rf2aa.kinematics.PARAMS)
    kinematics_params['USE_CB'] = use_cb
    t2d = rf2aa.kinematics.xyz_to_t2d(xyz_t_frames, mask_t_2d[None], params=kinematics_params)
    
    # Strip batch dimension
    t2d = t2d[0]
    
    return t2d, mask_t_2d

def find_contiguous_true_indices(arr):
    """
    Find the (start, stop) indices of each contiguous set of True values in a boolean array.

    Args:
    arr (numpy.ndarray): Input boolean array.

    Returns:
    list of tuples: List of (start, stop) index pairs for each contiguous True segment.
    """
    if torch.is_tensor(arr):
        arr = arr.numpy()
    assert arr.dtype == bool

    # Find the indices where True values start and stop
    stops = np.where(arr[:-1] & ~arr[1:])[0] + 1
    starts = np.where(~arr[:-1] & arr[1:])[0] + 1

    # Handle the case where the array starts with True
    if arr[0]:
        starts = np.insert(starts, 0, 0)

    # Handle the case where the array ends with True
    if arr[-1]:
        stops = np.append(stops, len(arr))

    # Pair the start and stop indices
    indices = list(zip(starts, stops))

    return indices


def mask_sequence_chunks(is_masked, p=0.5):
    """ 
    Take some regions that are unmasked and mask them with probability p, in a contiguous/chunked fashion.
    """
    is_masked = is_masked.copy()
    is_revealed = ~is_masked

    revealed_indices = find_contiguous_true_indices(is_revealed)

    # print('revealed indices are: ', revealed_indices)
    for start,stop in revealed_indices:
    
        if np.random.rand() < p:
            n_mask      = np.random.randint(1, stop - start+1)             # how many within here to mask? 
            mask_start  = np.random.randint(start, stop - n_mask+1)
            mask_stop   = mask_start + n_mask
            # print('masking from ', mask_start, ' to ', mask_stop, ' out of ', start, ' to ', stop)
            is_masked[mask_start:mask_stop+1] = True # mask it 

    return is_masked
