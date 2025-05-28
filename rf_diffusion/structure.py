from __future__ import annotations  # Fake Import for type hinting, must be at beginning of file

"""
Adapted from PyDSSP for ss conditioning
"""
from einops import repeat, rearrange
import torch
import numpy as np
from typing import Union, Tuple
from typing import Literal
from rf2aa.chemical import ChemicalData as ChemData
from rf_diffusion.build_coords import generate_H
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rf_diffusion.aa_model import Indep

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
# DEFAULT_MARGIN = 1.0
atomnum = {' N  ':0, ' CA ': 1, ' C  ': 2, ' O  ': 3}

C3_ALPHABET = np.array(['H', 'E', 'L', '?'])

HELIX = 0
STRAND = 1
LOOP = 2
ELSE = 3


def _check_input(coord):
    # Validates input
    org_shape = coord.shape
    assert (len(org_shape)==3) or (len(org_shape)==4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = repeat(coord, '... -> b ...', b=1) if len(org_shape)==3 else coord
    return coord, org_shape


# Don't use this function. It doesn't handle chainbreaks well at all
# def _get_hydrogen_atom_position(coord: torch.Tensor) -> torch.Tensor:
#     # A little bit lazy (but should be OK) definition of H position here.
#     vec_cn = coord[:,1:,0] - coord[:,:-1,2]
#     vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
#     vec_can = coord[:,1:,0] - coord[:,1:,1]
#     vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
#     vec_nh = vec_cn + vec_can
#     vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
#     return coord[:,1:,0] + 1.01 * vec_nh


def get_hbond_map(
    coord: torch.Tensor,
    cutoff: float=DEFAULT_CUTOFF,
    # margin: float=DEFAULT_MARGIN,
    return_e: bool=False
    # final_cut: float=0.00001
    ) -> torch.Tensor:
    """
    Calculate the hydrogen bond map based on the given coordinates.

    Args:
        coord (torch.Tensor): The input coordinates
        cutoff (float, optional): The cutoff distance for hydrogen bond interactions.
        margin (float, optional): The margin value for the hydrogen bond map.
        return_e (bool, optional): Whether to return the electrostatic interaction energy.
        final_cut (float, optional): bcov added this. Some very "not-hbonds" were getting counted

    Returns:
        hbond_map (torch.Tensor[bool]): hbond_map[n,o] Whether the N atom of residue n h-bonds to the O atom of residue o [L,L]

    Raises:
        AssertionError: If the number of atoms is not 5 (N, CA, C, O, H).

    """
    # check input
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a==5), "Number of atoms should 5 (N,CA,C,O,H)"
    h = coord[:,1:,4]
    # distance matrix
    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)    # [b,(l-1),(l-1),c] where [,,:,] is degenerate and coords[0] is missing
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)                # [b,(l-1),(l-1),c] where [,,:,] is degenerate and coords[0] is missing
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)  # [b,(l-1),(l-1),c] where [,:,,] is degenerate and coords[-1] is missing
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)  # [b,(l-1),(l-1),c] where [,:,,] is degenerate and coords[-1] is missing
    d_on = torch.linalg.norm(omap - nmap, dim=-1)                  # [b,(l-1),(l-1)]  [,n,o] where o[-1] and n[0] are missing
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)                  # [b,(l-1),(l-1)]  [,h,c] where c[-1] and h[0] are missing
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)                  # [b,(l-1),(l-1)]  [,h,o] where o[-1] and h[0] are missing
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)                  # [b,(l-1),(l-1)]  [,n,c] where c[-1] and n[0] are missing
    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [0,1,1,0]) # [b,l,l] [,n,o] h-bond energy
    if return_e: return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool)
    local_mask *= ~torch.diag(torch.ones(l-1, dtype=bool), diagonal=-1)
    local_mask *= ~torch.diag(torch.ones(l-1, dtype=bool), diagonal=+1) # pydssp was missing this and as such called strange turns strands
    # local_mask *= ~torch.diag(torch.ones(l-2, dtype=bool), diagonal=-2) # this is a hack in pydssp to prevent 1-residue self anti-parallel strands
    # pyDSSP, why did you have to make this so complicated? - yours truly bcov
    # hydrogen bond map (continuous value extension of original definition)
    # hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    # hbond_map = (torch.sin(hbond_map/margin*torch.pi/2)+1.)/2
    hbond_map = e * repeat(local_mask.to(e.device), 'l1 l2 -> b l1 l2', b=b)
    # return h-bond map
    hbond_map = hbond_map.squeeze(0) if len(org_shape)==3 else hbond_map
    return hbond_map < DEFAULT_CUTOFF


def assign_torch(coord: torch.Tensor, compute_pairs: bool=True, is_proline: torch.Tensor=None) -> Tuple[torch.Tensor, List[Tuple[Tuple[int]]]]:
    """
    Assigns secondary structure elements (SSEs) to a given coordinate tensor.

    Args:
        coord (torch.Tensor): The input coordinate tensor.
        compute_pairs (bool): Also compute strand pairs and correctly identify beta bulges
        is_proline (torch.Tensor[bool] or None): Which residues are amino acid proline?

    Returns:
        torch.Tensor: The tensor representing the assigned SSEs.
        List[Tuple[Tuple[int]]]]: Strand pairs ((i_start,i_last),(j_start,j_last)). i_start < j_start

    """
    if len(coord) == 0:
        return torch.zeros((0, 3), dtype=int), []

    # check input
    coord, org_shape = _check_input(coord)
    # get hydrogen bond map
    hbmap = get_hbond_map(coord)
    # Proline can't h-bond with its N
    if is_proline is not None:
        assert hbmap.shape[0] == 1
        hbmap[:,is_proline,:] = False
    hbmap = rearrange(hbmap, '... l1 l2 -> ... l2 l1') # convert into "i:C=O, j:N-H" form [b,l,l]
    # identify turn 3, 4, 5
    turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3) > 0. # [b,(l-3)] o[i] --> n[i+3], missing o[-3:] and n[:3]
    turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4) > 0. # [b,(l-3)] o[i] --> n[i+4], missing o[-4:] and n[:4]
    turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5) > 0. # [b,(l-3)] o[i] --> n[i+5], missing o[-5:] and n[:5]
    # assignment of helical sses
    h3 = torch.nn.functional.pad(turn3[:,:-1] * turn3[:,1:], [1,3])
    h4 = torch.nn.functional.pad(turn4[:,:-1] * turn4[:,1:], [1,4])
    h5 = torch.nn.functional.pad(turn5[:,:-1] * turn5[:,1:], [1,5])
    # helix4 first
    helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
    # Again, Rosetta doesn't do this. If you remove any part of a h3 that's part of a h4 you miss the ends of helices sometimes
    # h3 = h3 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    # h5 = h5 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
    helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1) > 0 # [b,(l-2),(l-2),3,3] basically a 3x3 view around each element of hbmap
    unfoldmap_rev = unfoldmap.transpose(-4,-3)              # [b,(l-2),(l-2),3,3] but the (l-2) dimensions are swapped
                                                                                  # the final dims are still [...,o,n] but i and j are backwards
    #              o:i-1 --> n:j               o:j --> n:i+1               o:j-1 --> n:i           o:i --> n:j+1
    p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
    p_bridge = torch.nn.functional.pad(p_bridge, [1,1,1,1])
    #              o:i --> n:j               o:j --> n:i                 o:i-1 --> n:j+1           o:j-1 --> n:i+1
    a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
    a_bridge = torch.nn.functional.pad(a_bridge, [1,1,1,1])
    a_bridge *= ~torch.eye(coord.shape[1], dtype=bool)[None] # Now that the hack is removed from hbmap we have to prevent self-pairing via i-1 --> i+1
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0

    if compute_pairs:
        # if N - C > 2.5A it's a new chain
        start_of_new_chain = torch.nn.functional.pad(torch.linalg.norm(coord[:,1:,0] - coord[:,:-1,2], axis=-1) > 2.5, [1,0], value=True)
        ladder, pairs = do_compute_pairs(p_bridge, a_bridge, start_of_new_chain)
    else:
        pairs = []

    # pydssp also got this slightly wrong: helix4 > strand > (helix3 | helix5)
    strand = ladder
    helix3 *= ~strand # Strand takes precidence over helix3
    helix5 *= ~strand # Strand takes precidence over helix5
    strand *= ~helix4 # Helix4 takes precidence over strand
    helix = (helix3 + helix4 + helix5) > 0
    loop = (~helix * ~strand)
    onehot = torch.stack([helix, strand, loop], dim=-1) # modified from pydssp
    onehot = onehot.squeeze(0) if len(org_shape)==3 else onehot
    return onehot, pairs


def flip_a_pair(a_pair, L):
    '''
    Flip an anti-parallel pair back into the real numbering scheme

    Args:
        a_pair (tuple[tuple[int]]): The pair where the second part is inverted
        L (int): Length of protein

    Returns:
        pair (tuple[tuple[int]]): The pair with the secondar part inverted
    '''
    ((a, b), (c, d)) = a_pair
    c = L-1 - c
    d = L-1 - d
    p1 = (a, b) if a < b else (b, a)
    p2 = (d, c) if c < d else (c, d) # pair 2 is reversed because anti-parallel

    return (p1, p2) if p1[0] < p2[0] else (p2, p1)

def do_compute_pairs(p_bridge, a_bridge, start_of_new_chain):
    '''
    Compute strand pairing from p_bridge and a_bridge

    Args:
        p_bridge (torch.Tensor[bool]): H-bond map for the parallel direction [1, L, L]
        a_bridge (torch.Tensor[bool]): H-bond map for the anti-parallel direction [1, L, L]
        start_of_new_chain (torch.Tensor[bool]): This is the first residue of a new chain [1, L]

    Returns:
        ladder (torch.Tensor[bool]): The new strand assignment
        pairs (List[Tuple[Tuple[int]]]]): Strand pairs ((i_start,i_last),(j_start,j_last)). i_start < j_start
    '''

    assert p_bridge.shape[0] == 1, "This doesn't have to be this way but this code never sees batch > 1"
    L = p_bridge.shape[1]

    p_pairs, p_in_pair = find_parallel_pairs(p_bridge[0], start_of_new_chain)

    # You have to flip the j dimension for antiparallel
    r_a_pairs, r_a_in_pair = find_parallel_pairs(torch.flip(a_bridge[0], [1]), start_of_new_chain, antiparallel=True)
    a_pairs = [flip_a_pair(pair, L) for pair in r_a_pairs]
    a_in_pair = torch.flip(r_a_in_pair, [1])

    pairs = list(set(p_pairs) | set(a_pairs))
    ladder = (p_in_pair | a_in_pair).any(axis=-1)[None]

    return ladder, pairs


def find_parallel_pairs(p_bridge, start_of_new_chain, small_gap=1, big_gap=4, antiparallel=False):
    '''
    Find strand pairs using the dssp rules
    The rules (at least according to rosetta) are that both strands are allowed to have gaps (bulges) in
      their pairing, and if a gap exists, one side may have up to 4 skipped residues and the other max 1

    Inputs:
        p_bridge (torch.Tensor[bool]): Whether or not these strands are making a parallel beta-strand connection [L,L]
        start_of_new_chain (torch.Tensor[bool]): This is the first residue of a new chain [1, L]
        small_gap (int): Max gap on the small side. Don't change this
        big_gap (int): Max gap on the big side. Don't change this
        antiparallel (bool): We are working with the antiparallel side and j has been flipped

    Returns:
        pairs (list[tuple[tuple[int]]]): The strand pairs found ((i_start,i_last),(j_start,j_last)). i_start < j_start
        in_pair (torch.Tensor[bool]): Whether or not this residue is part of a strand pair [L,L]
    '''
    L = p_bridge.shape[0]
    pairs = []
    in_pair = torch.zeros(p_bridge.shape, dtype=bool)

    for base_i in range(L):
        for base_j in range(L):
            if in_pair[base_i, base_j]:
                continue

            if p_bridge[base_i, base_j]:

                i = base_i
                j = base_j
                valid_extension = True

                while valid_extension:
                    valid_extension = False

                    # one can have a gap of size big_gap but the other gap must be <= small_gap

                    nextt = torch.where(p_bridge[i:i+big_gap+2,j:j+big_gap+2])

                    # next[:,0] is just this bridge
                    # next[:,1] will be the lowest i index then against the lowest j index
                    if len(nextt[0]) > 1:
                        next_bridge_i = nextt[0][1] + i
                        next_bridge_j = nextt[1][1] + j

                        # If the start of the next chain is also h-bonding to the same strand it totally doesn't count
                        if start_of_new_chain[0,i+1:next_bridge_i+1].any():
                            continue
                        if not antiparallel:
                            if start_of_new_chain[0,j+1:next_bridge_j+1].any():
                                continue
                        else:
                            lb = L-1 - next_bridge_j
                            ub = L-1 - (j+1)
                            if start_of_new_chain[0,lb:ub+1].any():
                                continue

                        gap_i = next_bridge_i - i - 1
                        gap_j = next_bridge_j - j - 1

                        valid_extension = (gap_i <= small_gap and gap_j <= big_gap) or (gap_i <= big_gap and gap_j <= small_gap)

                        if antiparallel and valid_extension:
                            # in antiparallel, i must be less than real-j otherwise hairpins can be turned into single strands pairing with themselves
                            valid_extension = next_bridge_i < L-1-next_bridge_j

                        if valid_extension:
                            i = next_bridge_i
                            j = next_bridge_j

                pairs.append(((int(base_i), int(i)), (int(base_j), int(j))))
                assert i >= base_i
                assert j >= base_j
                in_pair[base_i:i+1,base_j:j+1] = True
                if antiparallel:
                    # You have to reflect over the line y = L-1 - x
                    new_base_i = L-1 - j
                    new_i = L-1 - base_j
                    new_base_j = L-1 - i
                    new_j = L-1 - base_i
                    in_pair[new_base_i:new_i+1,new_base_j:new_j+1] = True
                else:
                    in_pair[base_j:j+1,base_i:i+1] = True

    assert in_pair[p_bridge].all()

    return pairs, in_pair



def read_pdbtext_with_checking(pdbstring: str):
    """
    Reads the coordinates from a PDB string and returns them as a numpy array.
    Only takes C, CA, N, O atoms. Attempts to validate inputs 

    Args:
        pdbstring (str): The PDB string containing the coordinates.

    Returns:
        numpy.ndarray: The coordinates extracted from the PDB string.

    """    
    lines = pdbstring.split("\n")
    coords, atoms, resid_old, check = [], None, None, []
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16], None)
            resid = l[21:26]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                    check.append(atom_check)
                atoms, resid_old, atom_check = [], resid, []
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
                atom_check.append(iatom)
    if atoms is not None:
        coords.append(atoms)
        check.append(atom_check)
    coords = np.array(coords)
    # check
    assert len(coords.shape) == 3, "Some required atoms [N,CA,C,O] are missing in the input PDB file"
    check = np.array(check)
    assert np.all(check[:,0]==0), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,1]==1), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,2]==2), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,3]==3), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    # output
    return coords


def assign(
    coord: Union[torch.Tensor, np.ndarray],
    out_type: Literal['onehot', 'index', 'c3'] = 'index',
    compute_pairs: bool=True,
    is_proline: torch.Tensor=None
    ) -> Tuple[torch.Tensor, List[Tuple[Tuple[int]]]]:
    """
    Assigns secondary structure labels to a given set of coordinates.


    Args:
        coord (Union[torch.Tensor, np.ndarray]): The input coordinates.
        out_type (Literal['onehot', 'index', 'c3'], optional): The type of output to return. 
            Defaults to 'c3'.
        compute_pairs (bool): Also compute strand pairs and correctly identify beta bulges
        is_proline (torch.Tensor[bool] or None): Which residues are amino acid proline?

    Returns:
        np.ndarray: The assigned secondary structure labels.
        List[Tuple[Tuple[int]]]]: Strand pairs ((i_start,i_last),(j_start,j_last)). i_start < j_start

    Raises:
        AssertionError: If the input type is not torch.Tensor or np.ndarray.
        AssertionError: If the output type is not 'onehot', 'index', or 'c3'.
    """
    assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    assert out_type in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"
    # main calculation
    onehot, pairs = assign_torch(coord, compute_pairs=compute_pairs, is_proline=is_proline)
    # output one-hot
    if out_type == 'onehot':
        return onehot, pairs
    # output index
    index = torch.argmax(onehot.to(torch.long), dim=-1)
    if out_type == 'index':
        return index, pairs
    # output c3
    c3 = C3_ALPHABET[index.cpu().numpy()]
    return c3, pairs

def read_pdbtext_no_checking(pdbstring: str):
    """
    Reads the coordinates from a PDB string and returns them as a numpy array.
    Only takes C, CA, N, O atoms

    Args:
        pdbstring (str): The PDB string containing the coordinates.

    Returns:
        numpy.ndarray: The coordinates extracted from the PDB string.

    """
    lines = pdbstring.split("\n")
    coords, atoms, resid_old = [], None, None
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16], None)
            resid = l[21:26]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                atoms, resid_old = [], resid
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
    if atoms is not None:
        coords.append(atoms)
    coords = np.array(coords)
    return coords


def get_bb_pydssp(indep: Indep) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearranges indep.xyz into a format for PyDSSP.

    Args:
        indep (Indep): The input object containing the data.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The rearranged data in the format required by PyDSSP. [L, 4, 3]
        torch.Tensor: Which residues were actually considered
        torch.Tensor: Which residues have amino acid proline?
    """
    is_prot = (indep.seq <= 21) * ~indep.is_gp * ~indep.is_sm
    # is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot') * ~is_gp * ~indep.is_sm
    N_idx = ChemData().aa2long[0].index(" N  ")
    CA_idx = ChemData().aa2long[0].index(" CA ")
    C_idx = ChemData().aa2long[0].index(" C  ")
    O_idx = ChemData().aa2long[0].index(" O  ")
    N = indep.xyz[is_prot, N_idx]
    CA = indep.xyz[is_prot, CA_idx]
    C = indep.xyz[is_prot, C_idx]
    O = indep.xyz[is_prot, O_idx]
    H = generate_H(N, CA, C)
    bb = torch.stack([N, CA, C, O, H], dim=0)
    bb_pydssp = torch.transpose(bb, 0, 1)
    is_proline = indep.seq[is_prot] == ChemData().one_letter.index('P')
    return bb_pydssp, is_prot, is_proline



def get_dssp_string(dssp_output: torch.Tensor) -> str:
    '''
    Convert the output from structure.get_dssp() to a human readable string

    Args:
        dssp_output (Tensor[long]): the output tensor from structure.get_dssp()

    Returns:
        str: The human readable dssp string (ex: 'LLLHHHHHLLLLEEEELLLL')
    '''
    return ''.join(C3_ALPHABET[dssp_output])


def get_dssp(indep: Indep, compute_pairs: bool=True) -> Tuple[torch.Tensor, List[Tuple[Tuple[int]]]]:
    '''
    Get the DSSP assignemt of indep using PyDSSP.

    Note! PyDSSP labels beta bulges as loops unless compute_pairs is True

    After a lot of work, this function (with compute_pairs=True) exactly matches pyrosetta (well, like 99.99%+)

    structure.HELIX = 0
    structure.STRAND = 1
    structure.LOOP = 2
    structure.ELSE = 3

    Args:
        indep (Indep): The input object containing the data.
        compute_pairs (bool): Also compute strand pairs and correctly identify beta bulges

    Returns:
        torch.Tensor: The DSSP assignment [L]

    '''

    bb_pydssp, is_prot, is_proline = get_bb_pydssp( indep )
    dssp = torch.full((indep.length(),), ELSE, dtype=int)
    dssp[is_prot], pairs = assign(bb_pydssp, out_type='index', compute_pairs=compute_pairs, is_proline=is_proline)

    # Re-index the pairs onto the full indep
    if not is_prot.all() and len(pairs) > 0:
        wh = torch.where(is_prot)[0]
        pairs = [((wh[a],wh[b]),(wh[c],wh[d])) for ((a,b),(c,d)) in pairs]

    return dssp, pairs











