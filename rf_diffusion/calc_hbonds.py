import torch
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.chemical import get_residue_bond_feats

from rf_diffusion.nucleic_compatibility_utils import get_resi_type_mask

import scipy.spatial.transform


MAX_ORBITALS = 8 # the sp3 0 connect generates an octahedron

# Not every call to hb_map uses these. So don't swap them!
HBMAP_OTHER_IDX0 = 0
HBMAP_OTHER_IATOM = 1
HBMAP_OUR_IATOM = 2
HBMAP_WE_ARE_DONOR = 3
HBMAP_N_FIELDS = 4


def identify_first_last_bb_atoms(seq):
    '''
    Find the indices of the first and last backbone atoms for each token in seq

    Args:
        seq (torch.Tensor[int]): The sequence [L]

    Returns:
        frame_first_bb (torch.Tensor[int]): The index of the first backbone atom [L]
        frame_last_bb (torch.Tensor[int]): The index of the last backbone atom [L]
    '''

    frame_is_nucl = get_resi_type_mask(seq, 'na')

    # Protein backbone is from N -> C
    ALA = ChemData().num2aa.index('ALA')
    N_idx = ChemData().aa2long[ALA].index(' N  ')
    C_idx = ChemData().aa2long[ALA].index(' C  ')

    # DNA backbone is from P -> O3'
    DA = ChemData().num2aa.index(' DA')
    P_idx = ChemData().aa2long[DA].index(' P  ')
    O3p_idx = ChemData().aa2long[DA].index(" O3'")

    frame_first_bb = torch.full((len(seq),), N_idx, dtype=int)
    frame_last_bb = torch.full((len(seq),), C_idx, dtype=int)

    frame_first_bb[frame_is_nucl] = P_idx
    frame_last_bb[frame_is_nucl] = O3p_idx

    return frame_first_bb, frame_last_bb


def identify_polymer_atom_connected_indices(frame_seq):
    '''
    Determine the indices of the connected atoms for each atom in polymer residues
    Also returns a "spare index" which is the index of another atom attached to one of the connected atoms (so CA if the main atom is O)
        for sp2 h-bond plane calculations

    Args:
        frame_seq (torch.Tensor[int]): The sequence of the polymer residues [L]

    Returns:
        connect_idx (torch.Tensor[int]): The indices of the connected atoms. -1s denote "no more connections" [L,NHEAVY,4]
        frame_n_connect (torch.Tensor[int]): The number of connected atoms for each atom [L,NHEAVY]
        spare_idx (torch.Tensor[int]): The neighbor of a neighbor atom [L,NHEAVY]
    '''

    residue_bond_feats = get_residue_bond_feats()
    frame_bond_feats = torch.stack([residue_bond_feats[aa] for aa in frame_seq], axis=0)
    frame_masks = torch.stack([ChemData().allatom_mask[aa][:ChemData().NHEAVY] for aa in frame_seq], axis=0)

    N_frame = len(frame_bond_feats)
    frame_range = torch.arange(N_frame, dtype=int)

    # Get the indices of the other connected atoms
    connect_idx = torch.full((N_frame, ChemData().NHEAVY, 4), -1, dtype=int)
    frame_n_connect = torch.zeros((N_frame, ChemData().NHEAVY), dtype=int)

    # This can be vectorized with a giant cumsum index matrix but I'm not sure that's faster
    wh_heavy_bonds = torch.where(frame_bond_feats[:,:ChemData().NHEAVY,:ChemData().NHEAVY])
    for iframe, ileft, iright in zip(*wh_heavy_bonds):
        connect_idx[iframe,ileft,frame_n_connect[iframe,ileft]] = iright
        frame_n_connect[iframe,ileft] += 1

    # For SP2 atoms with 1 frame_n_connect, get a connected atom of our only connection
    partner_idx = connect_idx[:,:,0]
    partner_idx[~frame_masks[:,:ChemData().NHEAVY]] = 0 # for atoms not present in this residue

    # Get the 4 connect idx of our partner atom
    partner_connect_idx = connect_idx[frame_range[:,None], partner_idx]

    # Try the first atom
    spare_idx = partner_connect_idx[:,:,0]
    # retry if the spare idx points back to us
    retry_mask = spare_idx == torch.arange(ChemData().NHEAVY, dtype=int)[None]
    spare_idx[retry_mask] = partner_connect_idx[:,:,1][retry_mask]

    return connect_idx, frame_n_connect, spare_idx


def identify_polymer_atom_connected_xyz(frame_xyz, frame_seq, missing_atom_loc=None):
    '''
    Determine the xyz coords of the connected atoms for each atom in polymer residues
    Also returns a "spare xyz" which is the xyz of another atom attached to one of the connected atoms (so CA if the main atom is O)
        for sp2 h-bond plane calculations

    Args:
        frame_xyz (torch.Tensor[float]): The xyz of the polymer residue atoms [L,NHEAVY]
        frame_seq (torch.Tensor[int]): The sequence of the polymer residues [L]
        missing_atom_loc (torch.Tensor[float] or None): The location to use for the first and last residue's missing connection [3]

    Returns:
        frame_connect_xyz (torch.Tensor[float]): The xyz of the connected atoms. nan denotes "no more connections" [L,NHEAVY,4,3]
        frame_n_connect (torch.Tensor[int]): The number of connected atoms for each atom [L,NHEAVY]
        frame_spare_xyz (torch.Tensor[float]): The neighbor of a neighbor atom [L,NHEAVY,3]
        connect_idx (torch.Tensor[int]): The indices of the connected atoms. -1s denote "no more connections" [L,NHEAVY,4]
    '''

    if missing_atom_loc is None:
        missing_atom_loc = torch.zeros(3, dtype=frame_xyz.dtype)

    # Identify backbone atoms
    frame_first_bb, frame_last_bb = identify_first_last_bb_atoms(frame_seq)

    N_frame = len(frame_seq)
    frame_range = torch.arange(N_frame, dtype=int)

    # Get the indices of the neighboring atoms
    frame_connect_idx, frame_n_connect, frame_spare_xyz = identify_polymer_atom_connected_indices(frame_seq)

    # Get the xyz of those atoms
    frame_connect_xyz = frame_xyz[frame_range[:,None,None], frame_connect_idx]
    frame_connect_xyz[frame_connect_idx == -1] = torch.nan
    frame_spare_xyz = frame_xyz[frame_range[:,None],frame_spare_xyz]

    # The first and last polymer residues can't have a connect atom so we build one at the origin
    frame_first_bb, frame_last_bb = identify_first_last_bb_atoms(frame_seq)

    prev_poly_connect_xyz = torch.cat( [missing_atom_loc[None], frame_xyz[frame_range,frame_last_bb][:-1]])
    next_poly_connect_xyz = torch.cat( [frame_xyz[frame_range,frame_first_bb][1:], missing_atom_loc[None]])

    # Fix first and last atom of each polymer residue that are missing a connection
    frame_connect_xyz[frame_range,frame_first_bb,frame_n_connect[frame_range,frame_first_bb]] = prev_poly_connect_xyz
    frame_n_connect[frame_range,frame_first_bb] += 1
    frame_connect_xyz[frame_range,frame_last_bb,frame_n_connect[frame_range,frame_last_bb]] = next_poly_connect_xyz
    frame_n_connect[frame_range,frame_last_bb] += 1

    return frame_connect_xyz, frame_n_connect, frame_spare_xyz, frame_connect_idx


def find_polymer_polar_atoms(indep):
    '''
    Parse the polymer atoms of an indep into the required fields for h-bond calculations

    Args:
        indep (indep): Indep

    Returns:
        dict:
            polar_idx0 (torch.Tensor[int]): The indep idx0 of the polar atom [N]
            polar_iatom (torch.Tensor[int]): The atom number of the polar atom [N]
            is_sp2 (torch.Tensor[bool]): Whether this atom is sp2 (vs sp3) [N]
            is_donor (torch.Tensor[bool]): Whether this atom is a donor [N]
            is_acceptor (torch.Tensor[bool]): Whether this atom is an acceptor [N]
            is_carbonyl_O (torch.Tensor[bool]): Whether this atom is a carbonyl O [N]
            polar_xyz (torch.Tensor[float]): The xyz of this atom [N,3]
            connect_xyz (torch.Tensor[float]): The xyz of the connected atoms [N,4,3]
            n_connect (torch.Tensor[int]): The number of connected atoms [N]
            spare_xyz (torch.Tensor[float]): The xyz of a connected atom of a connected atom (for sp2 plane determination) [N,3]
            N_to_satisfy (torch.Tensor[int]): The number of h-bonds needed to satisfy this atom [N]

    '''

    residue_bond_feats = get_residue_bond_feats()

    # Initialize some helper variables
    wh_frame = torch.where(~indep.is_sm)[0]
    N_frame = len(wh_frame)
    frame_range = torch.arange(N_frame, dtype=int)

    frame_seq = indep.seq[wh_frame]
    frame_is_nucl = get_resi_type_mask(frame_seq, 'na')
    frame_is_prot = ~frame_is_nucl
    assert frame_seq.max() < len(residue_bond_feats)

    # Get (L_frame, NHEAVY) data for each polymer residue
    frame_xyz = indep.xyz[wh_frame,:ChemData().NHEAVY]
    frame_elements = torch.stack([torch.tensor([ord(elem) if elem is not None else 0 for elem in ChemData().aa2elt[aa][:ChemData().NHEAVY]], dtype=int) for aa in frame_seq], axis=0)
    frame_bond_feats = torch.stack([residue_bond_feats[aa][:ChemData().NHEAVY] for aa in frame_seq], axis=0)

    frame_first_bb, frame_last_bb = identify_first_last_bb_atoms(frame_seq)

    # Apparently missing DNA backbone atoms have the origin as their placement...
    locally_missing = (frame_xyz == 0).all(axis=-1)

    frame_connect_xyz, frame_n_connect, frame_spare_xyz, frame_connect_idx = identify_polymer_atom_connected_xyz(frame_xyz, frame_seq)

    # find polars, donors, and acceptors

    # First we determine where the SP2 atoms are

    # Figure out elements
    frame_is_N = frame_elements == ord('N')
    frame_is_O = frame_elements == ord('O')
    frame_is_polar = (frame_is_N | frame_is_O ) & ~locally_missing
    frame_num_H = frame_bond_feats[:,:,ChemData().NHEAVY:].sum(axis=-1)

    # Figure out bonding situation
    frame_is_aro = (frame_bond_feats == ChemData().AROMATIC_BOND).any(axis=-1)
    frame_has_double = frame_is_aro | (frame_bond_feats == ChemData().DOUBLE_BOND).any(axis=-1)

    # Deal with resonance structures (double bonds flipping between neighboring atoms)
    this_neighbor_has_double = frame_has_double[frame_range[:,None,None], frame_connect_idx] & (frame_connect_idx > -1)
    a_neighbor_has_double = this_neighbor_has_double.any(axis=-1)
    frame_is_resonance = frame_is_polar & a_neighbor_has_double & ~frame_is_aro & ~frame_has_double

    # Assign SP2
    frame_is_sp2 = frame_has_double | frame_is_resonance
    frame_is_sp2[frame_is_prot,frame_first_bb[frame_is_prot]] = True # protein N is sp2 but the double-bond isn't shown
    frame_is_carbonyl_O = frame_is_sp2 & frame_is_O

    # Tyrosine isn't actually a carbonyl (oh pKa...)
    TYR = ChemData().num2aa.index('TYR')
    frame_is_TYR = frame_seq == TYR
    OH = ChemData().aa2long[TYR].index(' OH ')
    frame_is_carbonyl_O[frame_is_TYR,OH] = False


    # Now we figure out the satisfaction situation
    frame_N_total_want = torch.full((N_frame,ChemData().NHEAVY), 4, dtype=int)
    frame_N_total_want[frame_is_sp2] = 3

    frame_N_to_satisfy = frame_N_total_want - frame_n_connect

    # Donors and acceptors can be calculated now that we know the satisfaction
    frame_is_acceptor = frame_is_polar & (frame_num_H < frame_N_to_satisfy)
    frame_is_donor = frame_is_polar & (frame_num_H > 0)

    # Special case HIS. Both N can be both donor and acceptor
    HIS = ChemData().num2aa.index('HIS')
    frame_is_HIS = frame_seq == HIS
    ND1 = ChemData().aa2long[HIS].index(' ND1')
    NE2 = ChemData().aa2long[HIS].index(' NE2')

    frame_is_acceptor[frame_is_HIS,ND1] = ~locally_missing[frame_is_HIS,ND1]
    frame_is_acceptor[frame_is_HIS,NE2] = ~locally_missing[frame_is_HIS,ND1]
    frame_is_donor[frame_is_HIS,ND1] = ~locally_missing[frame_is_HIS,ND1]
    frame_is_donor[frame_is_HIS,NE2] = ~locally_missing[frame_is_HIS,ND1]


    frame_final_polar = frame_is_acceptor | frame_is_donor

    ret = {}
    frame_idx, polar_iatom = torch.where(frame_final_polar)
    ret['polar_idx0'] = wh_frame[frame_idx]
    ret['polar_iatom'] = polar_iatom
    ret['is_sp2'] = frame_is_sp2[frame_final_polar]
    ret['is_donor'] = frame_is_donor[frame_final_polar]
    ret['is_acceptor'] = frame_is_acceptor[frame_final_polar]
    ret['is_carbonyl_O'] = frame_is_carbonyl_O[frame_final_polar]
    ret['polar_xyz'] = frame_xyz[frame_final_polar]
    ret['connect_xyz'] = frame_connect_xyz[frame_final_polar]
    ret['n_connect'] = frame_n_connect[frame_final_polar]
    ret['spare_xyz'] = frame_spare_xyz[frame_final_polar]
    ret['N_to_satisfy'] = frame_N_to_satisfy[frame_final_polar]

    return ret


def identify_sm_connected_indices(sm_bond_feats):
    '''
    Determine the indices of the connected atoms for each atom
    Also returns a "spare index" which is the index of another atom attached to one of the connected atoms (so CA if the main atom is O)
        for sp2 h-bond plane calculations

    Args:
        sm_bond_feats (torch.Tensor[int]): A subset of the indep's bond feats [L_sm,L_sm]

    Returns:
        sm_connect_idx (torch.Tensor[int]): The indices of the connected atoms. -1s denote "no more connections" [L_sm,12]
        sm_n_connect (torch.Tensor[int]): The number of connected atoms [L_sm]
        sm_spare_idx (torch.Tensor[int]): The neighbor of a neighbor atom [L_sm]
    '''

    N_sm = len(sm_bond_feats)

    # Get the indices of the other connected atoms
    sm_connect_idx = torch.full((N_sm, 12), -1, dtype=int)
    sm_n_connect = torch.zeros((N_sm,), dtype=int)

    # This can be vectorized with a cumsum index matrix but I'm not sure it's faster
    wh_sm_bonds = torch.where(sm_bond_feats)
    for ileft, iright in zip(*wh_sm_bonds):
        try:
            sm_connect_idx[ileft,sm_n_connect[ileft]] = iright
        except IndexError:
            assert False, 'More than 12 atoms bonded to 1 atom. Aborting h-bond calculation'
        sm_n_connect[ileft] += 1

    # For SP2 atoms with 1 frame_n_connect, get a connected atom of our only connection
    sm_partner_idx = sm_connect_idx[:,0]

    # Get the 12 connect idx of our partner atom
    sm_partner_connect_idx = sm_connect_idx[sm_partner_idx]

    # Try the first atom
    sm_spare_idx = sm_partner_connect_idx[:,0]
    # retry if the spare idx points back to us
    sm_retry_mask = sm_spare_idx == torch.arange(N_sm, dtype=int)
    sm_spare_idx[sm_retry_mask] = sm_partner_connect_idx[:,1][sm_retry_mask]

    return sm_connect_idx, sm_n_connect, sm_spare_idx


def identify_sm_connected_xyz(sm_xyz, sm_bond_feats):
    '''
    Determine the xyz coords of the connected atoms for each atom in polymer residues
    Also returns a "spare xyz" which is the xyz of another atom attached to one of the connected residues (so CA if the main atom is O)
        for sp2 h-bond plane calculations

    Args:
        sm_xyz (torch.Tensor[float]): The xyz of the atoms [L_sm]
        sm_bond_feats (torch.Tensor[int]): A subset of the indep's bond feats [L_sm,L_sm]

    Returns:
        sm_connect_xyz (torch.Tensor[float]): The xyz of the connected atoms. nan denotes "no more connections" [L_sm,4,3]
        sm_n_connect (torch.Tensor[int]): The number of connected atoms [L_sm]
        sm_spare_xyz (torch.Tensor[float]): The neighbor of a neighbor atom [L_sm,3]
        sm_connect_idx (torch.Tensor[int]): The indices of the connected atoms. -1s denote "no more connections" [L_sm,12]
    '''

    sm_connect_idx, sm_n_connect, sm_spare_idx = identify_sm_connected_indices(sm_bond_feats)

    sm_connect_xyz = sm_xyz[sm_connect_idx]
    sm_connect_xyz[sm_connect_idx == -1] = torch.nan
    sm_spare_xyz = sm_xyz[sm_spare_idx]

    return sm_connect_xyz, sm_n_connect, sm_spare_xyz, sm_connect_idx


def find_sm_polar_atoms(indep):
    '''
    Parse the small molecule atoms of an indep into the required fields for h-bond calculations

    Args:
        indep (indep): Indep

    Returns:
        dict:
            polar_idx0 (torch.Tensor[int]): The indep idx0 of the polar atom [N]
            polar_iatom (torch.Tensor[int]): The atom number of the polar atom [N]
            is_sp2 (torch.Tensor[bool]): Whether this atom is sp2 (vs sp3) [N]
            is_donor (torch.Tensor[bool]): Whether this atom is a donor [N]
            is_acceptor (torch.Tensor[bool]): Whether this atom is an acceptor [N]
            is_carbonyl_O (torch.Tensor[bool]): Whether this atom is a carbonyl O [N]
            polar_xyz (torch.Tensor[float]): The xyz of this atom [N,3]
            connect_xyz (torch.Tensor[float]): The xyz of the connected atoms [N,4,3]
            n_connect (torch.Tensor[int]): The number of connected atoms [N]
            spare_xyz (torch.Tensor[float]): The xyz of a connected atom of a connected atom (for sp2 plane determination) [N,3]
            N_to_satisfy (torch.Tensor[int]): The number of h-bonds needed to satisfy this atom [N]

    '''

    # Helper variables
    wh_sm = torch.where(indep.is_sm)[0]
    N_sm = len(wh_sm)

    # Gather metadata about the small molecule elements
    sm_seq = indep.seq[indep.is_sm]
    sm_elements = torch.tensor([ord(ChemData().num2aa[aa]) if len(ChemData().num2aa[aa]) == 1 else 0 for aa in sm_seq])
    sm_xyz = indep.xyz[indep.is_sm,1]
    sm_bond_feats = indep.bond_feats[indep.is_sm][:,indep.is_sm]

    # Get the location of the connected atoms
    sm_connect_xyz, sm_n_connect, sm_spare_xyz, sm_connect_idx = identify_sm_connected_xyz(sm_xyz, sm_bond_feats)

    # Identify the elements
    sm_is_N = sm_elements == ord('N')
    sm_is_O = sm_elements == ord('O')
    sm_is_polar = sm_is_N | sm_is_O #| (sm_elements == ord('S'))


    # Without the hydrogens, we can only really guess at the pKa
    sm_is_aro = (sm_bond_feats == ChemData().AROMATIC_BOND).any(axis=-1)
    sm_has_double = sm_is_aro | (sm_bond_feats == ChemData().DOUBLE_BOND).any(axis=-1)

    assert not (sm_bond_feats == ChemData().TRIPLE_BOND).any(), 'Triple bonds not implemented yet'

    # Resonance structures
    sm_this_neighbor_has_double = sm_has_double[sm_connect_idx] & (sm_connect_idx > -1)
    sm_a_neighbor_has_double = sm_this_neighbor_has_double.any(axis=-1)
    sm_is_resonance = sm_is_polar & sm_a_neighbor_has_double & ~sm_is_aro & ~sm_has_double
    sm_is_sp2 = sm_has_double | sm_is_resonance

    sm_is_carbonyl_O = sm_is_O & sm_is_sp2

    sm_bond_order = sm_bond_feats.sum(axis=-1)
    sm_bond_order[sm_is_aro] = sm_n_connect[sm_is_aro] + 1 # The aromatic ring counts as a double bond, then it's num-neighbors
    sm_bond_order[sm_is_resonance] += 1 # The resonance guys picked up an extra bond, pka guess

    # Figure out satisfaction
    sm_N_total_want = torch.full((N_sm,), 4, dtype=int)
    sm_N_total_want[sm_is_sp2] = 3
    sm_N_to_satisfy = sm_N_total_want - sm_n_connect

    # To continue being polar, you have to need to be satisfied
    sm_mid_polar = sm_is_polar & (sm_N_to_satisfy > 0)

    sm_is_acceptor = torch.zeros(N_sm, dtype=bool)
    sm_is_donor = torch.zeros(N_sm, dtype=bool)

    # This is an *ok* way to assign donors and acceptors but it leaves a lot to be desired
    sm_is_acceptor[sm_mid_polar & (sm_bond_order == 0)] = True # water, ammonia
    sm_is_donor[sm_mid_polar & (sm_bond_order == 0)] = True # water, ammonia

    sm_is_acceptor[sm_mid_polar & (sm_bond_order == 1)] = True # hydroxyl, some primary amines
    sm_is_donor[sm_mid_polar & (sm_bond_order == 1)] = True # NH3, hydroxyl

    sm_is_acceptor[sm_mid_polar & (sm_bond_order == 2)] = True # carbonyl, ether, some secondary amines
    sm_is_donor[sm_mid_polar & (sm_bond_order == 2)] = sm_is_N[sm_mid_polar & (sm_bond_order == 2)] # most secondary amines, terminal double-bond N

    sm_is_acceptor[sm_mid_polar & (sm_bond_order == 3)] = True # tertiary amine, aromatic N, O in a ring
    sm_is_donor[sm_mid_polar & (sm_bond_order == 3)] = sm_is_N[sm_mid_polar & (sm_bond_order == 3)]  # tertiary amine, aromatic N

    # You are polar if you are a donor or an acceptor
    sm_final_polar = sm_is_acceptor | sm_is_donor


    # Assign output variables
    ret = {}
    ret['polar_idx0'] = wh_sm[torch.where(sm_final_polar)[0]]
    ret['polar_iatom'] = torch.ones(sm_final_polar.sum(), dtype=int)
    ret['is_sp2'] = sm_is_sp2[sm_final_polar]
    ret['is_donor'] = sm_is_donor[sm_final_polar]
    ret['is_acceptor'] = sm_is_acceptor[sm_final_polar]
    ret['is_carbonyl_O'] = sm_is_carbonyl_O[sm_final_polar]
    ret['polar_xyz'] = sm_xyz[sm_final_polar]
    ret['connect_xyz'] = sm_connect_xyz[sm_final_polar][:,:4] # trim down from 12 to 4 to match polymers
    ret['n_connect'] = sm_n_connect[sm_final_polar]
    ret['spare_xyz'] = sm_spare_xyz[sm_final_polar]
    ret['N_to_satisfy'] = sm_N_to_satisfy[sm_final_polar]

    return ret


def perp_vec(vec, tol=0.001):
    '''
    Returns a vector perpendicular to the input vector(s)

    Args:
        vec (torch.Tensor[float]): The input vector [...,3]
        tol (float): How close to 0 the dot of the test-vector is allowed to be

    Returns:
        perp_vecs (torch.Tensor[float]): Vectors that are normalized and perpendicular to the input [...,3]
    '''

    n_unsqueeze = len(vec.shape)-1
    perp = torch.cross(vec, torch.tensor([1, 0, 0], dtype=vec.dtype)[(None,)*n_unsqueeze], axis=-1)
    alt_perps = torch.cross(vec, torch.tensor([0, 1, 0], dtype=vec.dtype)[(None,)*n_unsqueeze], axis=-1)

    parallel = torch.sum(torch.abs(perp)) < tol

    return normalized(torch.where(parallel, alt_perps, perp))


def normalized(vectors):
    '''
    Normalizes input vectors

    Args:
        vectors (torch.Tensor[float]): The input vectors [...,3]

    Returns:
        normalized_vectors (torch.Tensor[float]): The input vectors but normalized to length 1 [...,3]
    '''
    norms = torch.linalg.norm(vectors, axis=-1)
    assert (norms > 0).all()
    normed = vectors / norms[...,None]
    return normed


def get_orbitals_sp2_1_connection(pol_xyz, connect_xyz, spare_xyz, is_carbonyl_O):
    '''
    Build orbitals for an sp2 atom with 1 connection

    Two orbitals at 120 and 240 degrees from the connected atom will be built in the plane defined by pol-connect-spare
    Carbonyls get an additional orbital at 180 to facilitate secondary structures

    Args:
        pol_xyz (torch.Tensor[float]): The xyz of the polar atoms [N,3]
        connect_xyz (torch.Tensor[float]): The xyz of the connected atom [N,3]
        spare_xyz (torch.Tensor[float]): The xyz of an atom connected to connect_xyz [N,3]
        is_carbonyl_O (torch.Tensor[bool]): Whether or not this is a carbonyl O [N]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''
    store_orbital_units = torch.full((len(pol_xyz), MAX_ORBITALS, 3), torch.nan, dtype=pol_xyz.dtype)
    if len(pol_xyz) == 0:
        return store_orbital_units 

    connect_unit = normalized( connect_xyz - pol_xyz )
    spare_xyz = normalized( spare_xyz - connect_xyz )
    sp2_plane = normalized( torch.cross( connect_unit, spare_xyz, axis=-1) )

    one_twenty_rotation_vec = (sp2_plane * torch.pi * 2 / 3).numpy()

    rotation_matrices = torch.tensor(scipy.spatial.transform.Rotation.from_rotvec(one_twenty_rotation_vec).as_matrix()).float()

    # Rotate twice by 120 degrees to get the orbitals
    orb_unit0 = torch.einsum('bij,bj->bi', rotation_matrices, connect_unit) # row-wise matrix multiplication
    orb_unit1 = torch.einsum('bij,bj->bi', rotation_matrices, orb_unit0) # row-wise matrix multiplication

    store_orbital_units[:,0] = orb_unit0
    store_orbital_units[:,1] = orb_unit1

    # The extra O is just the mirror of the connect atom (used for helices and sheets mostly)
    store_orbital_units[is_carbonyl_O,2] = -connect_unit[is_carbonyl_O]

    return store_orbital_units

# sp2 2 connect
# opposite connects, plane from connects

def get_orbitals_sp2_2_connections(pol_xyz, connect_xyzs):
    '''
    Build orbitals for an sp2 atom with 2 connection

    A single orbital is built opposite the center of angle formed by the two connected atoms

    Args:
        pol_xyz (torch.Tensor[float]): The xyz of the polar atoms [N,3]
        connect_xyzs (torch.Tensor[float]): The xyzs of the connected two atoms [N,2,3]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''
    store_orbital_units = torch.full((len(pol_xyz), MAX_ORBITALS, 3), torch.nan, dtype=pol_xyz.dtype)
    if len(pol_xyz) == 0:
        return store_orbital_units 

    connect_unit0 = normalized( connect_xyzs[:,0] - pol_xyz )
    connect_unit1 = normalized( connect_xyzs[:,1] - pol_xyz )
    connect_mid_unit = normalized( connect_unit0 + connect_unit1 )

    store_orbital_units[:,0] = -connect_mid_unit

    return store_orbital_units

def get_orbitals_sp3_0_connections(pol_xyz):
    '''
    Build orbitals for an sp3 atom with 0 connections (like water)

    8 orbitals are built in an octahedron (think corners of a cube)

    Args:
        pol_xyz (torch.Tensor[float]): The xyz of the polar atoms [N,3]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''
    store_orbital_units = torch.full((len(pol_xyz), MAX_ORBITALS, 3), torch.nan, dtype=pol_xyz.dtype)
    if len(pol_xyz) == 0:
        return store_orbital_units 

    invsqrt3 = torch.sqrt(torch.tensor(1/3))
    store_orbital_units[:,0] = torch.tensor([ invsqrt3,  invsqrt3,  invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,1] = torch.tensor([ invsqrt3,  invsqrt3, -invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,2] = torch.tensor([ invsqrt3, -invsqrt3,  invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,3] = torch.tensor([ invsqrt3, -invsqrt3, -invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,4] = torch.tensor([-invsqrt3,  invsqrt3,  invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,5] = torch.tensor([-invsqrt3,  invsqrt3, -invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,6] = torch.tensor([-invsqrt3, -invsqrt3,  invsqrt3], dtype=pol_xyz.dtype)
    store_orbital_units[:,7] = torch.tensor([-invsqrt3, -invsqrt3, -invsqrt3], dtype=pol_xyz.dtype)

    return store_orbital_units


def get_orbitals_sp3_1_connection(pol_xyz, connect_xyz):
    '''
    Build orbitals for an sp3 atom with 1 connection (like a hydroxyl)

    6 orbitals are built 109.5 degrees away from the connect atom in a circle 60 degrees apart
    Diffusion can't see hydrogens, so we just allow all hydrogen placements

    Args:
        pol_xyz (torch.Tensor[float]): The xyz of the polar atoms [N,3]
        connect_xyz (torch.Tensor[float]): The xyz of the connected atom [N,3]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''
    store_orbital_units = torch.full((len(pol_xyz), MAX_ORBITALS, 3), torch.nan, dtype=pol_xyz.dtype)
    if len(pol_xyz) == 0:
        return store_orbital_units 

    connect_unit = normalized( connect_xyz - pol_xyz )
    perp_unit = perp_vec(connect_unit)

    tetrahedral_rotation_vec = (perp_unit * torch.deg2rad(torch.tensor(109.5))).numpy()
    tet_rotation_matrices = torch.tensor(scipy.spatial.transform.Rotation.from_rotvec(tetrahedral_rotation_vec).as_matrix()).float()

    # Get the first element up in the tetrahedral circle
    orb_unit0 = torch.einsum('bij,bj->bi', tet_rotation_matrices, connect_unit) # row-wise matrix multiplication
    store_orbital_units[:,0] = orb_unit0

    # Make a matrix to rotate the hydroxyl
    hsweep_rotation_vec = (connect_unit * torch.pi / 3 ).numpy()
    hsweep_rotation_matrix = torch.tensor(scipy.spatial.transform.Rotation.from_rotvec(hsweep_rotation_vec).as_matrix()).float()

    for i in range(5):
        store_orbital_units[:,i+1] = torch.einsum('bij,bj->bi', hsweep_rotation_matrix, store_orbital_units[:,i]) # row-wise matrix multiplication

    return store_orbital_units


def get_orbitals_sp3_2_connections(pol_xyz, connect_xyzs):
    '''
    Build orbitals for an sp3 atom with 2 connections

    Two orbitals are built in the clasic tetrahedral geometry, 109.5 degrees apart from each other at the mid-plane between
        the two connect xyzs

    Args:
        pol_xyz (torch.Tensor[float]): The xyz of the polar atoms [N,3]
        connect_xyzs (torch.Tensor[float]): The xyzs of the connected atoms [N,2,3]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''
    store_orbital_units = torch.full((len(pol_xyz), MAX_ORBITALS, 3), torch.nan, dtype=pol_xyz.dtype)
    if len(pol_xyz) == 0:
        return store_orbital_units 

    connect_unit0 = normalized( connect_xyzs[:,0] - pol_xyz )
    connect_unit1 = normalized( connect_xyzs[:,1] - pol_xyz )
    connect_mid_unit = normalized( connect_unit0 + connect_unit1 )

    sp3_axis = normalized( connect_unit1 - connect_unit0 )

    first_rotation_vec = (sp3_axis * torch.deg2rad(torch.tensor( (360 - 109.5) / 2 ))).numpy()
    first_rotation_matrices = torch.tensor(scipy.spatial.transform.Rotation.from_rotvec(first_rotation_vec).as_matrix()).float()

    tetrahedral_rotation_vec = (sp3_axis * torch.deg2rad(torch.tensor(109.5))).numpy()
    tet_rotation_matrices = torch.tensor(scipy.spatial.transform.Rotation.from_rotvec(tetrahedral_rotation_vec).as_matrix()).float()

    # Orbitals are around the axis we calculated
    orb_unit0 = torch.einsum('bij,bj->bi', first_rotation_matrices, connect_mid_unit) # row-wise matrix multiplication
    orb_unit1 = torch.einsum('bij,bj->bi', tet_rotation_matrices, orb_unit0) # row-wise matrix multiplication

    store_orbital_units[:,0] = orb_unit0
    store_orbital_units[:,1] = orb_unit1

    return store_orbital_units

def get_orbitals_sp3_3_connections(pol_xyz, connect_xyzs):
    '''
    Build orbitals for an sp3 atom with 3 connections

    A single orbital is built opposite the center of the angle formed by the 3 connected atoms

    Args:
        pol_xyz (torch.Tensor[float]): The xyz of the polar atoms [N,3]
        connect_xyzs (torch.Tensor[float]): The xyzs of the connected atoms [N,3,3]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''
    store_orbital_units = torch.full((len(pol_xyz), MAX_ORBITALS, 3), torch.nan, dtype=pol_xyz.dtype)
    if len(pol_xyz) == 0:
        return store_orbital_units

    connect_unit0 = normalized( connect_xyzs[:,0] - pol_xyz )
    connect_unit1 = normalized( connect_xyzs[:,1] - pol_xyz )
    connect_unit2 = normalized( connect_xyzs[:,2] - pol_xyz )
    connect_mid_unit = normalized( connect_unit0 + connect_unit1 + connect_unit2 )

    store_orbital_units[:,0] = - connect_mid_unit

    return store_orbital_units


def generate_polar_orbitals(polar_xyz, connect_xyz, n_connect, spare_xyz, is_sp2, is_carbonyl_O, **kwargs):
    '''
    Generate the orbitals used for h-bond calculations

    The orbitals are unit vectors pointing to where hydrogens would be (or lone pairs)

    Args:
        polar_xyz (torch.Tensor[float]): The xyz of this atom [N,3]
        connect_xyz (torch.Tensor[float]): The xyz of the connected atoms [N,4,3]
        n_connect (torch.Tensor[int]): The number of connected atoms [N]
        spare_xyz (torch.Tensor[float]): The xyz of a connected atom of a connected atom (for sp2 plane determination) [N,3]
        is_sp2 (torch.Tensor[bool]): Whether this atom is sp2 (vs sp3) [N]
        is_carbonyl_O (torch.Tensor[bool]): Whether this atom is a carbonyl O [N]

    Returns:
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''

    N_polar = len(polar_xyz)

    orbital_units = torch.full((N_polar, MAX_ORBITALS, 3), torch.nan, dtype=polar_xyz.dtype)

    mask = is_sp2 & (n_connect == 1)
    orbital_units[mask] = get_orbitals_sp2_1_connection(polar_xyz[mask], connect_xyz[mask,0], spare_xyz[mask], is_carbonyl_O[mask])

    mask = is_sp2 & (n_connect == 2)
    orbital_units[mask] = get_orbitals_sp2_2_connections(polar_xyz[mask], connect_xyz[mask,:2])

    mask = ~is_sp2 & (n_connect == 0)
    orbital_units[mask] = get_orbitals_sp3_0_connections(polar_xyz[mask])

    mask = ~is_sp2 & (n_connect == 1)
    orbital_units[mask] = get_orbitals_sp3_1_connection(polar_xyz[mask], connect_xyz[mask,0])

    mask = ~is_sp2 & (n_connect == 2)
    orbital_units[mask] = get_orbitals_sp3_2_connections(polar_xyz[mask], connect_xyz[mask,:2])

    mask = ~is_sp2 & (n_connect == 3)
    orbital_units[mask] = get_orbitals_sp3_3_connections(polar_xyz[mask], connect_xyz[mask,:3])

    assert not torch.isnan(orbital_units[:,0]).any(), "A polar atom didn't generate any orbitals!" 

    return orbital_units




def rifdock_hbond(donor_xyz, donor_units, acceptor_xyz, acceptor_units, max_hbond=-2, H_dist=1.01):
    '''
    Rifdock's hbond function. 0.9 pearson-r to Rosetta

    Originally written by Will Sheffler. Then optimized for accuracy by Longxing and Brian

    This function uses unit vectors pointing away from the heavy-atom towards where either the hydrogens would be or the lone pairs
    This function is very fast and surprisingly accurate given how simple it is

    Args:
        donor_xyz (torch.Tensor[float]): The xyz coordinates of the donor heavy atoms [...,3]
        donor_units (torch.Tensor[float]): The unit vectors denoting the direction of the hydrogens [...,3]
        acceptor_xyz (torch.Tensor[float]): The xyz coordinates of the acceptor heavy atoms [...,3]
        acceptor_units (torch.Tensor[float]): The unit vectors denoting the direction of the orbitals [...,3]
        max_hbond (torch.Tensor[float]): The best value a h-bond can achieve. Classicaly set to -2
        H_dist (torch.Tensor[float]): The bond-length for hydrogens

    Returns:
        hbond_score (torch.Tensor[float]): The score of the hbond from 0 to max_hbond [...,3]
    '''

    donor_h = donor_xyz + donor_units * H_dist

    h_to_a = acceptor_xyz - donor_h
    h_to_a_len = torch.linalg.norm( h_to_a, axis=-1 )
    h_to_a /= h_to_a_len[...,None]

    h_dirscore = torch.sum( donor_units * h_to_a, axis=-1).clip(0, 1)
    a_dirscore = torch.sum( -acceptor_units * h_to_a, axis=-1).clip(0, 1)

    diff = h_to_a_len - 2.00
    diff[diff < 0] *= 1.5 # increase dis pen if too close

    max_diff = 0.8
    diff_oob = (diff >= max_diff) | (diff <= -max_diff)

    score = torch.square( 1.0 - torch.square( diff / max_diff ) ) * -1
    score[diff_oob] = 0

    dirscore = h_dirscore * h_dirscore * a_dirscore

    return score * dirscore * -max_hbond


def get_lowest_hbond_score_per_atom_pair(N_polar, wh_donors, wh_acceptors, scores_don_acc, hbond_threshold=-0.01):
    '''
    An internal function used to tally orbital scores into per-atom scores

    Args:
        N_polar (int): Number of polars
        wh_donors (tuple(torch.Tensor[int],torch.Tensor[int])): ipolar and iorbital for the donor orbitals
        wh_acceptors (tuple(torch.Tensor[int],torch.Tensor[int])): ipolar and iorbital for the acceptor orbitals
        score_don_acc (torch.Tensor[float]): The all-by-all scores for donors vs acceptors
        hbond_threshold (float): The hbond threshold

    Returns:
        hbond_scores_don_acc (Torch.Tensor[float]): The best donor-acceptor h-bond scores [N_polar,N_polar]
    '''
    assert hbond_threshold < 0

    hbond_scores_don_acc = torch.zeros((N_polar, N_polar), dtype=scores_don_acc.dtype)

    wh_hbonds = torch.where(scores_don_acc < hbond_threshold)
    which_don = wh_donors[wh_hbonds[0]]
    which_acc = wh_acceptors[wh_hbonds[1]]
    which_score = scores_don_acc[wh_hbonds]

    # Slow
    # for idon, iacc, score in zip(which_don, which_acc, which_score):
    #     hbond_scores_don_acc[idon, iacc] = torch.min(hbond_scores_don_acc[idon, iacc], which_score)

    # Some chat gpt magic
    indices = which_don * N_polar + which_acc
    flat_scores = hbond_scores_don_acc.view(-1)

    flat_scores.scatter_reduce_(dim=0, index=indices, src=which_score, reduce='amin')
    hbond_scores_don_acc = flat_scores.view(N_polar, N_polar)

    return hbond_scores_don_acc



def all_by_polar_atom_hbonds(polar_ret, indep=None, hbond_threshold=-0.01):
    '''
    Calculate the all-by-all hbond matrix for all of the polar atoms

    Args:
        polar_ret (dict): The return value from find_polymer_polar_atoms and find_sm_polar_atoms

    Returns:
        hbond_scores_don_acc (torch.Tensor[float]): The best h-bond found when atom_i is acting as a donor making an h-bond to acceptor atom_j [N,N]
        orbital_units (torch.Tensor[float]): The unit vectors for the calculated orbitals. Nan for unused fields [N,MAX_ORBITALS,3]
    '''

    is_donor = polar_ret['is_donor']
    is_acceptor = polar_ret['is_acceptor']
    polar_xyz = polar_ret['polar_xyz']

    # Get the orbitals
    orbital_units = generate_polar_orbitals(**polar_ret, indep=indep)

    # Break them up into donor and acceptor orbitals
    valid_orbitals = ~(torch.isnan(orbital_units).any(axis=-1))
    valid_donor_orbitals = valid_orbitals & is_donor[:,None]
    valid_acceptor_orbitals = valid_orbitals & is_acceptor[:,None]

    wh_donors = torch.where(valid_donor_orbitals)
    wh_acceptors = torch.where(valid_acceptor_orbitals)

    # Prepare the xyz information
    donor_xyz = polar_xyz[wh_donors[0]]
    donor_units = orbital_units[valid_donor_orbitals]

    acceptor_xyz = polar_xyz[wh_acceptors[0]]
    acceptor_units = orbital_units[valid_acceptor_orbitals]

    # Calculate all-by-all hbonds
    hbond_scores_orb_don_acc = rifdock_hbond(donor_xyz[:,None], donor_units[:,None], acceptor_xyz[None,:], acceptor_units[None,:])

    # Reduce to per-atom level
    hbond_scores_don_acc = get_lowest_hbond_score_per_atom_pair(len(polar_xyz), wh_donors[0], wh_acceptors[0], hbond_scores_orb_don_acc, hbond_threshold=hbond_threshold)

    return hbond_scores_don_acc, orbital_units



def fill_hbond_map(best_atom_hbond_don_acc, indep_len, polar_idx0, polar_iatom, hbond_threshold=-0.01, **kwargs):
    '''
    A helper function to fill the hbond map with best_atom_hbond_don_acc

    Args:
        best_atom_hbond_don_acc (torch.Tensor[float]): The best h-bond found when atom_i is acting as a donor making an h-bond to acceptor atom_j [N,N]
        indep_len (int): indep.length()
        polar_idx0 (torch.Tensor[int]): The indep idx0 of the polar atom [N]
        polar_iatom (torch.Tensor[int]): The atom number of the polar atom [N]
        hbond_threshold (float): The hbond threshold
        **kwargs: Unused

    Returns:
        hbond_map (torch.Tensor[int]): Ragged array of h-bonds that exist. -1 denotes empty field. (HBMAP_OTHER_IDX0, HBMAP_OTHER_IATOM, HBMAP_OUR_IATOM, HBMAP_WE_ARE_DONOR) [L,?,4]
        hbond_scores (torch.Tensor[float]): Ragged array of h-bond scores. nan denotes empty field. [L,?]
    '''

    # other_idx0, other_iatom, our_iatom, we_are_donor

    hbond_exists = best_atom_hbond_don_acc < hbond_threshold

    if hbond_exists.sum() == 0:
        max_residue_hbonds = 0
        hbond_map = torch.full((indep_len, max_residue_hbonds, HBMAP_N_FIELDS), -1, dtype=int)
        hbond_scores = torch.full((indep_len, max_residue_hbonds), torch.nan, dtype=best_atom_hbond_don_acc.dtype)
        return hbond_map, hbond_scores


    better_than_other_way = best_atom_hbond_don_acc <= torch.transpose(best_atom_hbond_don_acc, 1, 0)
    hbonds_to_store = better_than_other_way & hbond_exists

    wh_to_store = torch.where(hbonds_to_store)

    N_hbonds = len(wh_to_store[0])
    don_idx0 = polar_idx0[wh_to_store[0]]
    don_iatom = polar_iatom[wh_to_store[0]]
    acc_idx0 = polar_idx0[wh_to_store[1]]
    acc_iatom = polar_iatom[wh_to_store[1]]
    score_to_store = best_atom_hbond_don_acc[wh_to_store]

    # Prepare both directions for storage
    our_idx0 = torch.cat((don_idx0, acc_idx0))
    our_iatom = torch.cat((don_iatom, acc_iatom))
    other_idx0 = torch.cat((acc_idx0, don_idx0))
    other_iatom = torch.cat((acc_iatom, don_iatom))
    full_scores = torch.cat((score_to_store, score_to_store))
    we_are_donor = torch.cat((torch.ones(N_hbonds, dtype=int), torch.zeros(N_hbonds, dtype=int)))

    # Fancy logic to store this in a vectorized way

    # First make it such that our_idx0 is ascending
    argsort = torch.argsort(our_idx0)

    our_idx0 = our_idx0[argsort]
    our_iatom = our_iatom[argsort]
    other_idx0 = other_idx0[argsort]
    other_iatom = other_iatom[argsort]
    full_scores = full_scores[argsort]
    we_are_donor = we_are_donor[argsort]

    # Now get the counts for each our_idx0
    _, counts = torch.unique_consecutive(our_idx0, return_counts=True)

    # Prepare the output structures
    max_residue_hbonds = counts.max()
    hbond_map = torch.full((indep_len, max_residue_hbonds, HBMAP_N_FIELDS), -1, dtype=int)
    hbond_scores = torch.full((indep_len, max_residue_hbonds), torch.nan, dtype=best_atom_hbond_don_acc.dtype)

    # Prepare the secondary indices within each our_idx0
    internal_indices = torch.cat([torch.arange(count) for count in counts])

    # Store it all
    hbond_map[our_idx0,internal_indices,HBMAP_OTHER_IDX0] = other_idx0
    hbond_map[our_idx0,internal_indices,HBMAP_OTHER_IATOM] = other_iatom
    hbond_map[our_idx0,internal_indices,HBMAP_OUR_IATOM] = our_iatom
    hbond_map[our_idx0,internal_indices,HBMAP_WE_ARE_DONOR] = we_are_donor
    hbond_scores[our_idx0,internal_indices] = full_scores

    return hbond_map, hbond_scores

