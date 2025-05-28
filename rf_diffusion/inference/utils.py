import numpy as np
import os
from omegaconf import DictConfig
from rf_diffusion.kinematics import xyz_to_t2d
import torch
import torch.nn.functional as nn
from rf_diffusion.util import get_torsions
from scipy.spatial.transform import Rotation as scipy_R
from rf_diffusion.util import rigid_from_3_points
import random
import string 
import rf2aa.chemical
import rf2aa.tensor_util
import rf_diffusion.parsers
from rf_diffusion.chemical import ChemicalData as ChemData
import pandas as pd
from rf_diffusion.silent_files import format_scores

###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################

# These functions behave exactly the same as before but now do not rely on class fields from the Denoiser

def slerp_update(r_t, r_0, t, mask=0):
    """slerp_update uses SLERP to update the frames at time t to the
    predicted frame for t=0

    Args:
        R_t, R_0: rotation matrices of shape [3, 3]
        t: time step
        mask: set to 1 / True to skip update.

    Returns:
        slerped rotation for time t-1 of shape [3, 3]
    """
    # interpolate FRAMES between one and next 
    if not mask:
        key_rots = scipy_R.from_matrix(np.stack([r_t, r_0], axis=0))
    else:
        key_rots = scipy_R.from_matrix(np.stack([r_t, r_t], axis=0))

    key_times = [0,1]

    interpolator = Slerp(key_times, key_rots)
    alpha = np.array([1/t])
    
    # grab the interpolated FRAME 
    interp_frame  = interpolator(alpha)
    
    # constructed rotation matrix which when applied YIELDS interpolated frame 
    interp_rot = (interp_frame.as_matrix().squeeze() @ np.linalg.inv(r_t.squeeze()) )[None,...]

    return interp_rot

def get_next_frames(xt, px0, t, diffuser, so3_type, diffusion_mask, noise_scale=1.):
    """get_next_frames gets updated frames using either SLERP or the IGSO(3) + score_based reverse diffusion.
    

    based on self.so3_type use slerp or score based update.

    SLERP xt frames towards px0, by factor 1/t
    Rather than generating random rotations (as occurs during forward process), calculate rotation between xt and px0
   
    Args:
        xt: noised coordinates of shape [L, 14, 3]
        px0: prediction of coordinates at t=0, of shape [L, 14, 3]
        t: integer time step
        diffuser: Diffuser object for reverse igSO3 sampling
        so3_type: The type of SO3 noising being used ('igso3', or 'slerp')
        diffusion_mask: of shape [L] of type bool, True means not to be
            updated (e.g. mask is true for motif residues)
        noise_scale: scale factor for the noise added (IGSO3 only)
    
    Returns:
        backbone coordinates for step x_t-1 of shape [L, 3, 3]
    """
    N_0  = px0[None,:,0,:]
    Ca_0 = px0[None,:,1,:]
    C_0  = px0[None,:,2,:]

    R_0, Ca_0 = rigid_from_3_points(N_0, Ca_0, C_0)

    N_t  = xt[None, :, 0, :]
    Ca_t = xt[None, :, 1, :]
    C_t  = xt[None, :, 2, :]

    R_t, Ca_t = rigid_from_3_points(N_t, Ca_t, C_t)
    R_0 = scipy_R.from_matrix(rf2aa.tensor_util.assert_squeeze(R_0).numpy())
    R_t = scipy_R.from_matrix(rf2aa.tensor_util.assert_squeeze(R_t).numpy())

    # Sample next frame for each residue
    all_rot_transitions = []
    for i in range(len(xt)):
        r_0 = R_0[i].as_matrix()
        r_t = R_t[i].as_matrix()
        mask_i = diffusion_mask[i]

        if so3_type == "igso3":
            r_t_next = diffuser.so3_diffuser.reverse_sample(r_t, r_0, t,
                    mask=mask_i, noise_level=noise_scale)[None,...]
            interp_rot =  r_t_next @ (r_t.T)
        elif so3_type == "slerp":
            interp_rot = slerp_update(r_t, r_0, t, diffusion_mask[i])
        else:
            assert False, "so3 diffusion type %s not implemented"%so3_type

        all_rot_transitions.append(interp_rot)

    all_rot_transitions = np.stack(all_rot_transitions, axis=0)

    # Apply the interpolated rotation matrices to the coordinates
    next_crds   = np.einsum('lrij,laj->lrai', all_rot_transitions, xt[:,:3,:] - Ca_t.squeeze(0)[:,None,...].numpy()) + Ca_t.squeeze(0)[:,None,None,...].numpy()

    # (L,3,3) set of backbone coordinates with slight rotation 
    return next_crds.squeeze(1)

def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1
    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*beta_schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*beta_schedule[t_idx])/(1-alphabar_schedule[t_idx]))*px0_ca
    b = ((torch.sqrt(1-beta_schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))*xt_ca

    mu = a + b

    return mu, sigma

def get_next_ca(xt, px0, t, diffusion_mask, crd_scale, beta_schedule, alphabar_schedule, noise_scale=1.):
    """
    Given full atom x0 prediction (xyz coordinates), diffuse to x(t-1)
    
    Parameters:
        
        xt (L, 14/27, 3) set of coordinates
        
        px0 (L, 14/27, 3) set of coordinates

        t: time step. Note this is zero-index current time step, so are generating t-1    

        logits_aa (L x 20 ) amino acid probabilities at each position

        seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t

        diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused 

        noise_scale: scale factor for the noise being added

    """

    # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale 
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    # get mu(xt, x0)
    mu, sigma = get_mu_xt_x0(xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule)

    sampled_crds = torch.normal(mu, torch.sqrt(sigma*noise_scale))
    delta = sampled_crds - xt[:,1,:] #check sign of this is correct

    if diffusion_mask is not None:
        # calculate the mean displacement between the current motif and where 
        # RoseTTAFold thinks it should go 
        # print('Got motif delta')
        # motif_delta = (px0[diffusion_mask,:3,...] - xt[diffusion_mask,:3,...]).mean(0).mean(0)

        delta[diffusion_mask,...] = 0
        # delta[diffusion_mask,...] = motif_delta

    out_crds = xt + delta[:, None, :]

    return out_crds/crd_scale, delta/crd_scale

class DecodeSchedule():
    """
    Class for managing AA decoding schedule stuff
    """

    def __init__(self, L, visible, aa_decode_steps=40, mode='distance_based'):

        # only distance based for now
        assert mode in ['distance_based']# , 'uniform_linear', 'ar_fixbb']
        self.mode = mode

        self.visible = visible

        # start as all high - only matters when a residue is being decoded
        # at which point we will know the true T
        self.T = torch.full((L,), 999)

        # number of residues being decoded on each step
        if aa_decode_steps > 0:
            tmp = np.array(list(range((~self.visible).sum())))
            np.random.shuffle(tmp)
            ndecode_per_step = np.array_split(tmp, aa_decode_steps)
            np.random.shuffle(ndecode_per_step)
            self.ndecode_per_step = [len(a) for a in ndecode_per_step]


    def get_next_idx(self, cur_indices, dmap):
        """
        Given indices being currently sampled and a distance map, return one more index which is allowed to
        be sampled at the same time as cur indices

        Parameters:

            cur_indices (list, required): indices of residues also being decoded this step

            dmap (torch.tensor, required): (L,L) distance map of CA's
        """
        L = dmap.shape[0]
        options = torch.arange(L)[~self.visible] # indices of non-decoded residues

        # find the index with the largest average distance from all decoded residues
        #mean_distances = dmap[cur_indices,options]
        d_rows    = dmap[cur_indices]
        d_columns = d_rows[:,options]
        mean_distances = d_columns.mean(dim=0)


        #mean_distances = mean_distances.mean(dim=0)


        best_idx_local  = torch.argmax(mean_distances) # index within options tensor
        best_idx_global = options[best_idx_local]      # the option itself

        return best_idx_global


    def get_decode_positions(self, t_idx, px0):
        """
        Returns the next (0-indexed) positions to decode for this timestep
        """
        L = px0.shape[0]
        assert t_idx < len( self.ndecode_per_step ) # taken care of outside this class in sampling loop

        N = self.ndecode_per_step[t_idx]
        decode_list = []

        if self.mode == 'distance_based':
            # perform dynamic distance based sampling
            ca   = px0[:,1,:]
            dmap = torch.sqrt( (ca[None,:] - ca[:,None]).square().sum(dim=-1) + 1e-6 )

            for i in range(N):
                if i == 0:
                    # sample a random residue which hasn't been decoded yet
                    first_idx = np.random.choice(torch.arange(L)[~self.visible])
                    decode_list.append(int(first_idx))
                    self.visible[first_idx] = True
                    self.T[first_idx] = t_idx + 1
                    continue

                # given already sampled indices, get another
                decode_idx = self.get_next_idx(decode_list,dmap)

                decode_list.append(int(decode_idx))
                self.visible[decode_idx] = True # set this now because get_next_idx depends on it
                self.T[decode_idx] = t_idx+1    # now that we know this residue is decoded, set its big T value

        return decode_list


    @property
    def idx2steps(self):
        return self.T

def preprocess(seq, xyz_t, t, T, ppi_design, binderlen, target_res, device):
    """
    Function to prepare inputs to diffusion model
        
        seq (torch.tensor, required): (L) integer sequence 

        msa_masked (torch.tensor, required): (1,1,L,48)

        msa_full  (torch,.tensor, required): (1,1, L,25)
        
        xyz_t (torch,tensor): (L,14,3) template crds (diffused) 
        
        t1d (torch.tensor, required): (1,L,22) this is the t1d before tacking on the chi angles. Last plane is 1/t (conf hacked as timestep)
    """
    L = seq.shape[-1]
    ### msa_masked ###
    ##################
    msa_masked = torch.zeros((1,1,L,48))
    msa_masked[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]
    msa_masked[:,:,:,22:44] = nn.one_hot(seq, num_classes=22)[None, None]

    ### msa_full ###
    ################
    msa_full = torch.zeros((1,1,L,25))
    msa_full[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.zeros((1,1,L,21))
    t1d[:,:,:,:21] = nn.one_hot(torch.where(seq == 21, 20, seq), num_classes=21)[None,None]
    
    """
    DJ -  next line (commented out) was an attempt to see if model more readily 
          moved the motif if they were all set to same confidence 
          in order to alleveate chainbreaks at motif. Maybe sorta worked? definitely not full fix
    """
    # conf = conf = torch.where(seq == 21, 1-t/T, 1-t/T)[None,None,...,None]
    conf = torch.where(seq == 21, 1-t/T, 1.)[None,None,...,None]
    t1d = torch.cat((t1d, conf), dim=-1)

    # NRB: Adding in dimension for target hotspot residues
    target_residue_feat = torch.zeros_like(t1d[...,0])[...,None]
    if ppi_design and target_res is not None:
        absolute_idx = [resi+binderlen for resi in target_res]
        target_residue_feat[...,absolute_idx,:] = 1

    t1d = torch.cat((t1d, target_residue_feat), dim=-1)

    ### xyz_t ###
    #############
    xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
    xyz_t=xyz_t[None, None]
    xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)
    
    ### t2d ###
    ###########
    t2d = xyz_to_t2d(xyz_t)
  
    ### idx ###
    ###########
    idx = torch.arange(L)[None]
    if ppi_design:
        idx[:,binderlen:] += 200

    ### alpha_t ###
    ###############
    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
    alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1,-1,L,10,2)
    alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
    
    #put tensors on device
    msa_masked = msa_masked.to(device)
    msa_full = msa_full.to(device)
    seq = seq.to(device)
    xyz_t = xyz_t.to(device)
    idx = idx.to(device)
    t1d = t1d.to(device)
    t2d = t2d.to(device)
    alpha_t = alpha_t.to(device)
    return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t

def get_pdb_lines_traj(path):
    with open(path, 'r') as fh:
        s = fh.read()
        models = s.strip().split('ENDMDL')
        models = [m for m in models if m]
        models = [m.split('\n') for m in models]
    return models

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return rf_diffusion.parsers.parse_pdb_lines_target(lines, **kwargs)

def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)

    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins



def process_target(pdb_path, parse_hetatom=False, center=True):

    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct['xyz'][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct['xyz'] - ca_center)
    seq_orig = torch.from_numpy(target_struct['seq'])
    atom_mask = torch.from_numpy(target_struct['mask'])
    seq_len = len(xyz)

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    xyz_27[:, :ChemData().NHEAVY, :] = xyz[:, :ChemData().NHEAVY, :]

    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :ChemData().NHEAVY] = atom_mask[:, :ChemData().NHEAVY]
    out = {
           'xyz_27': xyz_27,
            'mask_27': mask_27,
            'seq': seq_orig,
            'pdb_idx': target_struct['pdb_idx']
            } 
    if parse_hetatom:
        out['xyz_het'] = target_struct['xyz_het']
        out['info_het'] = target_struct['info_het']
    return out
    

def recycle_schedule(T, rec_sched=None, num_designs=1):
    """  
    Function to convert input recycle schedule into a list of recycles.
    Input:
        - T: Max T
        - rec_sched: timestep:num_recycles|timestep:num_recycles
            e.g. T=200, rec_sched = 50:2/25:4/2:10. At timestep 50, start having 2 recycles, then 4 at t=25, 2=10
    """
    if rec_sched is not None:
        schedule = np.ones(T, dtype='int')
        if "/" in rec_sched:
            parts = rec_sched.split("/")
        else:
            parts = [rec_sched]
        indices=[int(i.split(":")[0]) for i in parts]
        assert all(indices[i] > indices[i+1] for i in range(len(indices) - 1)), "Recycle schedule indices must be in decreasing order"
        for part in parts:
            idx, num = part.split(":")
            schedule[:int(idx)] = int(num)
    else:
        schedule = np.ones(T, dtype='int') * int(num_designs)
    return schedule
 
class BlockAdjacency():
    """
    Class for handling PPI design inference with ss/block_adj inputs.
    Basic idea is to provide a list of scaffolds, and to output ss and adjacency
    matrices based off of these, while sampling additional lengths.
    Inputs:
        - scaffold_list: list of scaffolds (e.g. ['2kl8','1cif']). Can also be a .txt file.
        - scaffold dir: directory where scaffold ss and adj are precalculated
        - sampled_insertion: how many additional residues do you want to add to each loop segment? Randomly sampled 0-this number
        - sampled_N: randomly sample up to this number of additional residues at N-term
        - sampled_C: randomly sample up to this number of additional residues at C-term
        - ss_mask: how many residues do you want to mask at either end of a ss (H or E) block. Fixed value
        - num_designs: how many designs are you wanting to generate? Currently only used for bookkeeping
        - systematic: do you want to systematically work through the list of scaffolds, or randomly sample (default)
        - num_designs_per_input: Not really implemented yet. Maybe not necessary
    Outputs:
        - L: new length of chain to be diffused
        - ss: all loops and insertions, and ends of ss blocks (up to ss_mask) set to mask token (3). Onehot encoded. (L,4)
        - adj: block adjacency with equivalent masking as ss (L,L)     
    """

    def __init__(self, conf, num_designs):
        """
        Parameters:
          inputs:
             conf.scaffold_list as conf
             conf.inference.num_designs for sanity checking
        """

        # either list or path to .txt file with list of scaffolds
        if isnstance(conf.scaffold_list, list):
            self.scaffold_list = scaffold_list
        elif conf.scaffold_list[-4:] == '.txt':
            #txt file with list of ids
            list_from_file = []
            with open(conf.scaffold_list,'r') as f:
                for line in f:
                    list_from_file.append(line.strip())
            self.scaffold_list = list_from_file
        else:
            raise NotImplementedError

        # path to directory with scaffolds, ss files and block_adjacency files
        self.scaffold_dir = conf.scaffold_dir

        # maximum sampled insertion in each loop segment
        self.sampled_insertion = conf.sampled_insertion

        # maximum sampled insertion at N- and C-terminus
        if '-' in str(conf.sampled_N):
            self.sampled_N = [int(str(conf.sampled_N).split("_")[0]), int(str(conf.sampled_N).split("-")[1])]
        else:
            self.sampled_N = [0, int(conf.sampled_N)]
        if '-' in str(conf.sampled_C):
            self.sampled_C = [int(str(conf.sampled_C).split("_")[0]), int(str(conf.sampled_C).split("-")[1])]
        else:
            self.sampled_C = [0, int(conf.sampled_C)]

        # number of residues to mask ss identity of in H/E regions (from junction)
        # e.g. if ss_mask = 2, L,L,L,H,H,H,H,H,H,H,L,L,E,E,E,E,E,E,L,L,L,L,L,L would become\
        # M,M,M,M,M,H,H,H,M,M,M,M,M,M,E,E,M,M,M,M,M,M,M,M where M is mask
        self.ss_mask = conf.ss_mask

        # whether or not to work systematically through the list
        self.systematic = conf.systematic

        self.num_designs = num_designs

        if len(self.scaffold_list) > self.num_designs:
            print("WARNING: Scaffold set is bigger than num_designs, so not every scaffold type will be sampled")


        # for tracking number of designs
        self.num_completed = 0
        if self.systematic:
            self.item_n = 0

    def get_ss_adj(self, item):
        """
        Given at item, get the ss tensor and block adjacency matrix for that item
        """
        ss = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_ss.pt'), weights_only=False)
        adj = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_adj.pt'), weights_only=False)

        return ss, adj

    def mask_to_segments(self, mask):
        """
        Takes a mask of True (loop) and False (non-loop), and outputs list of tuples (loop or not, length of element)
        """
        segments = []
        begin=-1
        end=-1
        for i in range(mask.shape[0]):
            # Starting edge case
            if i == 0:
                begin = 0
                continue

            if not mask[i] == mask[i-1]:
                end=i
                if mask[i-1].item() is True:
                    segments.append(('loop', end-begin))
                else:
                    segments.append(('ss', end-begin))
                begin = i

        # Ending edge case: last segment is length one
        if not end == mask.shape[0]:
            if mask[i].item() is True:
                segments.append(('loop', mask.shape[0]-begin))
            else:
                segments.append(('ss', mask.shape[0]-begin))
        return segments

    def expand_mask(self, mask, segments):
        """
        Function to generate a new mask with dilated loops and N and C terminal additions
        """
        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])

        output = N_add * [False]
        for ss, length in segments:
            if ss == 'ss':
                output.extend(length*[True])
            else:
                # randomly sample insertion length
                ins = random.randint(0, self.sampled_insertion)
                output.extend((length + ins)*[False])
        output.extend(C_add*[False])
        assert torch.sum(torch.tensor(output)) == torch.sum(~mask)
        return torch.tensor(output)

    def expand_ss(self, ss, adj, mask, expanded_mask):
        """
        Given an expanded mask, populate a new ss and adj based on this
        """
        ss_out = torch.ones(expanded_mask.shape[0])*3 #set to mask token
        adj_out = torch.full((expanded_mask.shape[0], expanded_mask.shape[0]), 0.)

        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.full(adj_out.shape, True)
        #mask out loops/insertions, which is ~expanded_mask
        expanded_mask_2d[~expanded_mask, :] = False
        expanded_mask_2d[:,~expanded_mask] = False

        mask_2d = torch.full(adj.shape, True)
        # mask out loops. This mask is True=loop
        mask_2d[mask, :] = False
        mask_2d[:,mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        adj_out = adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

        return ss_out, adj_out


    def mask_ss_adj(self, ss, adj, expanded_mask):
        """
        Given an expanded ss and adj, mask some number of residues at either end of non-loop ss
        """
        original_mask = torch.clone(expanded_mask)
        if self.ss_mask > 0:
            for i in range(1, self.ss_mask+1):
                expanded_mask[i:] *= original_mask[:-i]
                expanded_mask[:-i] *= original_mask[i:]


        ss[~expanded_mask] = 3
        adj[~expanded_mask,:] = 0
        adj[:,~expanded_mask] = 0

        return ss, adj

    def get_scaffold(self):
        """
        Wrapper method for pulling an item from the list, and preparing ss and block adj features
        """
        if self.systematic:
            # reset if num designs > num_scaffolds
            if self.item_n >= len(self.scaffold_list):
                self.item_n = 0
            item = self.scaffold_list[self.item_n]
            self.item_n += 1
        else:
            item = random.choice(self.scaffold_list)
        print("Scaffold constrained based on file: ", item)
        # load files
        ss, adj = self.get_ss_adj(item)
        # separate into segments (loop or not)
        mask = torch.where(ss == 2, 1, 0).bool()
        segments = self.mask_to_segments(mask)

        # insert into loops to generate new mask
        expanded_mask = self.expand_mask(mask, segments)

        # expand ss and adj
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)

        # finally, mask some proportion of the ss at either end of the non-loop ss blocks
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)

        # and then update num_completed
        self.num_completed += 1

        return ss.shape[0], torch.nn.functional.one_hot(ss.long(), num_classes=4), adj

class Target():
    """
    Class to handle targets (fixed chains).
    Inputs:
        - path to pdb file
        - hotspot residues, in the form B10,B12,B60 etc
        - whether or not to crop, and with which method
    Outputs:
        - Dictionary of xyz coordinates, indices, pdb_indices, pdb mask
    """

    def __init__(self, conf: DictConfig):

        self.pdb = parse_pdb(conf.target_path)
        if conf.hotspots:
            hotspots = list(conf.hotspots)
            self.hotspots = [(i[0], int(i[1:])) for i in hotspots]
        else:
            self.hotspots = []
        # sanity check
        if conf.radial_crop is not None and conf.contig_crop is not None:
            raise ValueError("Cannot have both radial and contig cropping")

        # add hotspots
        self.add_hotspots()

        if conf.radial_crop:
#             self.pdb = self.radial_crop(radial_crop)
            raise NotImplementedError("Haven't implemented radial cropping yet")

        elif conf.contig_crop:
            self.pdb = self.contig_crop(conf.contig_crop)
    
    def parse_contig(self, contig_crop):
        """
        Takes contig input and parses
        """
        contig_list = []
        for contig in contig_crop.split(" "):
            subcon=[]
            for crop in contig.split(","):
                if crop[0].isalpha():
                    subcon.extend([(crop[0], p) for p in np.arange(int(crop.split("-")[0][1:]), int(crop.split("-")[1])+1)])
            contig_list.append(subcon)

        return contig_list

    def contig_crop(self, contig_crop, residue_offset=200):
        """
        Method to take a contig string referring to the receptor and output a pdb dictionary with just this crop
        NB there are two ways to provide inputs:
            - 1) e.g. B1-30,0 B50-60,0. This will add a residue offset between each chunk
            - 2) e.g. B1-30,B50-60,B80-100. This will keep the original indexing of the pdb file. 
        Can handle the target being on multiple chains
        """

        # add residue offset between chains if multiple chains in receptor file
        for idx, val in enumerate(self.pdb['pdb_idx']):
            if idx != 0 and val != self.pdb['pdb_idx'][idx-1]:
                self.pdb['idx'][idx:] += (residue_offset + idx)


        # convert contig to mask
        contig_list = self.parse_contig(contig_crop)

        # add residue offset to different parts of contig_list
        for contig in contig_list[1:]:
            start = int(contig[0][1])
            self.pdb['idx'][start:] += residue_offset

        contig_list = np.array(contig_list).flatten()

        mask = np.array([True if i in contig_list else False for i in self.pdb['pdb_idx']])

        # sanity check
        assert np.sum(self.pdb['hotspots']) == np.sum(self.pdb['hotspots'][mask]), "Supplied hotspot residues are missing from the target contig!"

        for key, val in self.pdb:
            self.pdb[key] = val[mask]

    def centre_pdb(self):
        self.pdb['xyz'] = self.pdb['xyz'] - self.pdb['xyz'][:,:1,:].mean(axis=0)

    def add_hotspots(self):
        hotspots = np.array([1. if i in self.hotspots else 0. for i in self.pdb['pdb_idx']])
        self.pdb['hotspots'] = hotspots

    def radial_crop(self, radial_crop):
        #TODO
        pass

    def get_target(self):
        return self.pdb

def assemble_config_from_chk(conf: DictConfig):
    """
    Function for loading model config from checkpoint directly.
    
    Takes:
        - config file
    
    Actions:
        - Loads model checkpoint and looks for "Config"
        - Replaces all -model and -diffuser items
        - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
    
    """
    
    # TODO change this so we don't have to load the model twice
    pass


def append_run_to_scorefile(run_prefix, tag, scores, delimiter):
    '''
    Called from run_inference.py

    Append the results of this run to the scorefile

    Args:
        run_prefix (str): The run prefix from run_inference
        tag (tag): The name of the design
        scores (dict[str,?]): The dictionary of scores to output
        delimiter (str): The delimiter for the scorefile. Typically either ',' or ' '. ',' implies .csv ' ' implies .sc
    '''

    suffix = '.sc' if delimiter == ' ' else '.csv'
    score_file_name = run_prefix + '_out' + suffix

    append_to_scorefile(score_file_name, tag, scores, delimiter)

def append_to_scorefile(fname, tag, scores, delimiter):
    '''
    Write these scores to a scorefile. Appending to it if it exists

    Args:
        fname (str): The name of the file
        tag (str): The name of the design
        scores (dict[str,?]): The scores to write
        delimeter (str): The delimiter for the scorefile
    '''
    scores['description'] = tag

    write_header = not os.path.exists(fname)

    string_scores = format_scores(scores)

    df = pd.DataFrame([string_scores])
    df.to_csv(fname, sep=delimiter, header=write_header, index=None, mode='a')

def conf_select_px0(model_out, px0_source='atom37'):
    """
    Allows users to select which px0 they want to return.
    Can use the backbone coords generated by openfold, or rfo preds from rf2aa.
    Openfold only gives protein atoms, rf2aa gives full atom predictions, including nucleics.
    Args:
        model_out:   (dict) containing processed RoseTTAfold preds.
        px0_source:  (str)  specified choice of px0 source, controlled by the `inference.px0_source` arg in config spec.
            format: `px0_source` string, or substring preceding a '.' delimiter must be a key of the `model_out` dict to pull out a specified item, 
                     then a substring following the '.' delimiter can access a specific attribute from that item.
            options:
                * 'atom37'             (default) : representation from OpenFold post-processing of rfo.
                * 'rfo.xyz_allatom' (alternative): accesses the 'xyz_allatom' attribute of 'rfo' object.
    Returns:
        px0: (torch.Tensor[long]): predicted x0 coordinates [L, n_atoms, 3]
    """

    px0_source_spec = px0_source.split('.') # break up for potential access of rfo attributes

    if len(px0_source_spec)==2:
        model_key, model_attr = px0_source_spec
        assert model_key in model_out, f"specified conf.inference.px0_source starts with '{model_key}', but must start with one of: {model_out.keys()}."
        assert hasattr(model_out[model_key], model_attr), f"specified attribute '{model_attr}' not found in model_out['{model_key}']"
        px0 = getattr(model_out[model_key], model_attr)
    elif len(px0_source_spec)==1:
        model_key = px0_source_spec[0]
        if model_key != 'refine':
            assert model_key in model_out, f"specified conf.inference.px0_source is '{model_key}', but must be one of: {model_out.keys()}."
            px0 = model_out[model_key]
        else:
            assert len(model_out['rfo'].xyz_allatom.shape) == 4
            L = model_out['rfo'].xyz_allatom.shape[1]
            px0 = torch.zeros(L,36,3)
            px0[:,:5,:] = model_out['rfo'].xyz_allatom[0,:,:5,:]
    else:
        px0 = model_out['atom37'][-1, -1]
        
    # index first batch dim and last iter dim
    if len(px0.shape)==5:
        return px0[-1, -1]
    elif len(px0.shape)==4:
        return px0[-1]
    else:
        return px0
