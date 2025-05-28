import torch
from rf2aa.chemical import ChemicalData as ChemData
from openfold.utils import rigid_utils
from rf2aa.util_module import XYZConverter
from rf_diffusion.frame_diffusion.data import all_atom



def generate_H(N, CA, C):
    '''
    Returns the H atom of proteins for hbonds and similar

    Using exactly Rosetta's definition:
        - H is always in the plane of C-N-CA
        - H is 1.01 A away from N
        - H is 119.150002 degrees away from CA

    The H immediately following a chainbreak still follows these rules but will be oriented a bit randomly because of that

    Args:
        N (torch.Tensor): The N atoms [L,3]
        CA (torch.Tensor): The CA atoms [L,3]
        C (torch.Tensor): The C atoms [L,3]

    Returns:
        H (torch.Tensor): The location of the H atom
    '''
    if len(N) == 0:
        return N.clone()

    # Constants from Rosetta. The final constant is psi = 180 degrees but that's covered by our planarity definition
    H_bond_length = 1.01
    H_CA_N_angle = torch.deg2rad(torch.tensor(119.150002))

    # Build a fake C at the origin for the first aa
    first_C = torch.zeros((1, 3), dtype=C.dtype)
    C_w_extra = torch.cat((first_C, C))

    # Build the two unit vectors coming from N
    C_from_N = C_w_extra[:-1] - N
    C_from_N = C_from_N / torch.linalg.norm( C_from_N, axis=-1 )[:,None]
    CA_from_N = CA - N
    CA_from_N = CA_from_N / torch.linalg.norm(CA_from_N, axis=-1)[:,None]

    # The rotation unit for the C_N_CA plane
    C_N_CA_plane_norm = torch.cross( C_from_N, CA_from_N, dim=-1 )
    C_N_CA_plane_norm /= torch.linalg.norm( C_N_CA_plane_norm, axis=-1)[:,None]

    # Handle the case where C, N, and CA are colinear (could happen at a chainbreak)
    # bad_mask == C_N_CA are co-linear
    bad_mask = torch.isnan(C_N_CA_plane_norm[:,0])
    if bad_mask.sum() > 0:
        # Instead of crossing with C_from_N, cross with the x-axis
        C_N_CA_plane_norm[bad_mask] = torch.cross( torch.tensor([[1.0, 0.0, 0.0]]) , CA_from_N[bad_mask], dim=-1 )
        C_N_CA_plane_norm[bad_mask] /= torch.linalg.norm(C_N_CA_plane_norm[bad_mask], axis=-1)[:,None]

        # super rare, but bad_mask == C_N_CA are co-linear with x-axis
        bad_mask = torch.isnan(C_N_CA_plane_norm[:,0])
        if bad_mask.sum() > 0:
            # Instead of crossing with C_from_N, cross with the y-axis
            C_N_CA_plane_norm[bad_mask] = torch.cross( torch.tensor([[0.0, 1.0, 0.0]]) , CA_from_N[bad_mask], dim=-1 )
            C_N_CA_plane_norm[bad_mask] /= torch.linalg.norm(C_N_CA_plane_norm[bad_mask], axis=-1)[:,None]

    # The y-component of our sin calculation
    in_plane_towards_H = torch.cross( C_N_CA_plane_norm, CA_from_N, dim=-1 )

    # H is in the C_N_CA plane, 1.01A away from N, and 119.150002 degrees away from CA
    H = N + H_bond_length * in_plane_towards_H * torch.sin(H_CA_N_angle) + H_bond_length * CA_from_N * torch.cos(H_CA_N_angle)

    return H



# Chi angles to use if you just need something valid to give an amino acid
#  These come from the default residues that Rosetta builds
ideal_chi_angles = {
    'ALA':[0.0, 0.0, 0.0, 0.0],
    'CYS':[63.7, 60.0, 0.0, 0.0],
    'ASP':[59.4, 31.1, 0.0, 0.0],
    'GLU':[63.4, -177.2, -0.6, 0.0],
    'PHE':[60.7, 68.4, 0.0, 0.0],
    'GLY':[0.0, 0.0, 0.0, 0.0],
    'HIS':[62.3, -79.3, 0.0, 0.0],
    'ILE':[-172.7, 166.5, 0.0, 0.0],
    'LYS':[62.6, -178.4, -179.5, -180.0],
    'LEU':[70.7, 165.7, 0.0, 0.0],
    'MET':[63.9, -172.2, 72.0, 0.0],
    'ASN':[58.8, 46.7, 0.0, 0.0],
    'PRO':[30.0, -33.9, 24.8, 0.0],
    'GLN':[63.2, -177.4, -80.9, 0.0],
    'ARG':[62.5, 176.8, 176.4, 85.5],
    'SER':[68.0, 100.0, 0.0, 0.0],
    'THR':[-170.7, 140.0, 0.0, 0.0],
    'VAL':[64.6, 0.0, 0.0, 0.0],
    'TRP':[60.9, 89.3, 0.0, 0.0],
    'TYR':[-177.7, 79.1, 180.0, 0.0],
}

def get_sc_ideal_torsions(seq):
    '''
    A function for generating sidechain torsions from scratch

    If you are trying to build sidechains from nothing and you need torsions, this will
      get you values that generate good-looking sidechains

    They might clash with other residues, but at least they won't clash with themselves...

    Args:
        seq (torch.Tensor[int]): Sequence, must all be protein [L]

    Returns:
        torsions (torch.Tensor[float]): Torsions to give to rf2aa [L,ChemData().NTOTALDOFS,2]
    '''

    if len(seq) == 0:
        return torch.zeros((0, ChemData().NTOTALDOFS, 2))

    assert (seq < 20).all(), f'get_ideal_torsions was passed a non-protein residue {seq}'

    # Setting all the torsions to angle 0 does a pretty good job of hitting all the important stuff
    # In particular, these ones *really* need to be 0
    # 7 CB Bend
    # 8 CB Twist
    # 9 CG Bend
    torsions = torch.zeros((len(seq),ChemData().NTOTALDOFS,2))
    torsions[:,:,0] = 1.0 # cos(0) == 1

    my_chis = torch.tensor([ideal_chi_angles[ChemData().num2aa[s]] for s in seq])
    my_chis = torch.deg2rad(my_chis)

    torsions[:,3:7,0] = torch.cos(my_chis)
    torsions[:,3:7,1] = torch.sin(my_chis)

    return torsions




def build_ideal_sidechains(xyz, seq):
    '''
    A function that builds ideal sidechains from scratch

    Use this if you must generate sidechain residues from truly nothing besides a backbone

    Args:
        xyz (torch.Tensor[float]): The backbone that we will be using to generate sidechains [L,>=4,3]
        seq (torch.Tensor[int]): The sequence of the sidechains that we will be building

    Returns:
        xyz_ideal (torch.Tensor[float]): The input backbone but with ideal sidechains [L,36,3]
    '''

    if len(xyz) == 0:
        return torch.zeros((0,ChemData().NTOTAL,3))

    assert xyz.shape[1] >= 4, "This function builds sidechains but it's up to you to build the N, CA, C, O first"

    torsions = get_sc_ideal_torsions(seq)

    xyz_converter = XYZConverter()
    atom_mask, xyz_ideal = xyz_converter.compute_all_atom(seq[None], xyz[None], torsions[None])
    xyz_ideal = xyz_ideal[0]
    xyz_ideal[:,:4] = xyz[:,:4]

    return xyz_ideal


# Code to generate those RTs

# conf = test_utils.construct_conf()
# prepare_pyrosetta(conf)

# pose = pyro().pose_from_sequence("AA")
# pose.dump_pdb('extended.pdb')

# indep = aa_model.make_indep('extended.pdb')
# rigids = du.rigid_frames_from_atom_14(indep.xyz)
# rots = rigids.get_rots()
# trans = rigids.get_trans()
# rot_mats = rots.get_rot_mats()

# my_frames = torch.zeros((2, 4, 4))
# my_frames[:,:3,:3] = rot_mats
# my_frames[:,:3,3] = trans
# my_frames[:,3,3] = 1

# RT_rosetta_res1 = my_frames[0]
# RT_dihedral_180 = torch.linalg.inv(my_frames[0]) @ my_frames[1]


# From Rosetta. This is the Relative Transform of adjacent backbone frames in an extended peptide
RT_dihedral_180 = torch.tensor([[ 2.7071300149e-01, -9.6266013384e-01,  0.0000000000e+00,  3.5619554520e+00],
                                [-9.6266007423e-01, -2.7071303129e-01,  0.0000000000e+00, -1.3316509724e+00],
                                [ 0.0000000000e+00,  0.0000000000e+00, -1.0000000000e+00,  0.0000000000e+00],
                                [ 0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00]],
                                dtype=torch.float64 )

# From Rosetta. The location of the first AA
RT_rosetta_res1 = torch.tensor([[ 0.3617492318, -0.9322754741,  0.0000000000,  1.4579999447],
                                [ 0.9322754741,  0.3617492616,  0.0000000000,  0.0000000000],
                                [ 0.0000000000,  0.0000000000,  1.0000000000,  0.0000000000],
                                [ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]],
                                dtype=torch.float64)


def build_extended_backbone(length):
    '''
    Builds a protein backbone with all phi, psi, omega set to 180 degrees

    Returned backbone has N, CA, C, O, CB

    Args:
        length (int): Number of aas in new protein

    Returns:
        backbone_xyz (torch.Tensor[int]): N, CA, C, O, CB of new protein [L,5,3]

    '''

    # This function uses float64 because errors accumulate with float32

    # Build the homogeneous-transform style frames
    my_frames = torch.zeros((length, 4, 4), dtype=torch.float64)
    my_frames[0] = RT_rosetta_res1

    # Build each additional backbone frame from the previous
    for i in range(length-1):
        my_frames[i+1] = my_frames[i] @ RT_dihedral_180

    # back to float32
    my_frames = my_frames.float()

    # Create the openfold structures
    rotations = rigid_utils.Rotation(rot_mats=my_frames[:,:3,:3])
    translations = my_frames[:,:3,3]

    rigids = rigid_utils.Rigid(rotations, translations)

    psi_180 = torch.zeros((length, 2))
    psi_180[:,1] = -1.0 # cos(180) == -1 # I think openfold has their cos and sin backwards

    backbone_xyz = all_atom.compute_backbone(rigids, psi_180)[0][:,:5]

    return backbone_xyz

def extended_ideal_xyz_from_seq(seq, include_hydrogens=False):
    '''
    Builds a protein from sequence with extended backbone (phi, psi, omega = 180)

    Args:
        seq (torch.Tensor[int]): Numeric sequence of protein
        include_hydrogens (bool): Also return hydrogens

    Returns:
        xyz: (torch.Tensor[float]): The xyz coordinates of the protein [L,3]
        atom_mask: (torch.tensor[float]): Which atoms are present? [L,NHEAVY or NTOTAL]
    '''

    NATOMS = ChemData().NTOTAL if include_hydrogens else ChemData().NHEAVY

    L = len(seq)
    xyz = torch.zeros((L, NATOMS, 3))
    xyz[:,:5] = build_extended_backbone(len(seq))
    xyz[:,4:] = build_ideal_sidechains(xyz, seq)[:,4:NATOMS]

    if include_hydrogens:
        # All the other hydrogens should be good to go except H (because the bb torsions are wrong in build_ideal_sidechains())
        H_idx = ChemData().aa2long[0].index(' H  ')
        xyz[:,H_idx] = generate_H(xyz[:,0], xyz[:,1], xyz[:,2])

    atom_mask = ChemData().allatom_mask[seq]

    return xyz, atom_mask
    
