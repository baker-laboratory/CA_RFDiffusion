from __future__ import annotations  # Fake import for type hinting, must be at beginning of file

import numpy as np
import torch
import copy
import rf_diffusion.aa_model as aa_model
from rf_diffusion.chemical import ChemicalData as ChemData
import random

# Imports for typing only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rf_diffusion.aa_model import Indep
    from typing import Tuple, Union


# Specific imports for kinematics updates for NA
# from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.kinematics import get_dih
import warnings
from rf2aa.util import get_atom_frames
import networkx as nx

# For trasmutate code
from torch.distributions import MultivariateNormal

dna_base_complement = {
    22: 25,
    25: 22,
    24: 23,
    23: 24,
    26: 26
}

rna_base_complement = {
    22: 25,
    25: 22,
    24: 23,
    23: 24,
    26: 26
}

# Mol class management code: can possibly move to ChemData() if people agree on the logic.
mask_ind_by_class = {
                    'protein': ChemData().num2aa.index('MAS'), # prot mask is MAS
                    'dna':     ChemData().num2aa.index(' DX'), # dna mask is  DX
                    'rna':     ChemData().num2aa.index(' RX'), # rna mask is  RX
                    'unknown': ChemData().num2aa.index('UNK'), # unknown mask is UNK, (used for hybrid chain class assumption)
                    'atom':    ChemData().num2aa.index('ATM'), # if we want to mask small mol or atom (currently unused, and probably undesireable)
                    }

# Define mol_class_indices via 3-letter code specification and reference index used in ChemData().
mol_class_3letter = {
    'protein': ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO', # prot resis,
                'SER','THR','TRP','TYR','VAL','UNK','MAS','MEN','HIS_D'], # unk, mask, N-methyl asparagine, his_D
    'dna':     [' DA',' DC',' DG',' DT',' DX'], # DNA bases and unk_DNA
    'rna':     [' RA',' RC',' RG',' RU',' RX'], # RNA bases and unk_RNA
    'atom':   ['Al', 'As', 'Au', 'B','Be', 'Br', 'C', 'Ca', 'Cl','Co', 'Cr', 'Cu', 'F', 'Fe','Hg', 'I', # atoms and unk_ATM
                'Ir', 'K', 'Li', 'Mg','Mn', 'Mo', 'N', 'Ni', 'O','Os', 'P', 'Pb', 'Pd', 'Pr','Pt', 'Re', 
                'Rh', 'Ru', 'S','Sb', 'Se', 'Si', 'Sn', 'Tb','Te', 'U', 'W', 'V', 'Y', 'Zn', 'ATM'], 
    }
# Get all token indices for a given molecule class
mol_class_inds = {mol_class: [ChemData().aa2num[aa_code] for aa_code in mol_class_3letter[mol_class]] for mol_class in mol_class_3letter.keys()}

# Given a token ind, see what mol class it belongs to:
inds_to_mol_class = {ind: mol_class for mol_class in mol_class_inds.keys() for ind in mol_class_inds[mol_class]}

# Given a token ind, return the associated mask for that mol_class
inds_to_mol_class_mask = {ind: mask_ind_by_class[mol_class] for mol_class in mol_class_inds.keys() for ind in mol_class_inds[mol_class]}




def get_full_mask_seq(seq):
    """
    input:
        seq: 1d-tensor of length (L), containing sequence tokens in integer form
    returns:
        mask_seq: 1d-tensor of length (L), containing the mask token for the mol_class at input positions.
    """
    mask_seq = torch.full(seq.shape, ChemData().MASKINDEX, device=seq.device)
    for i,s_i in enumerate(seq):
        mask_seq[i] = mask_ind_by_class[inds_to_mol_class[int(s_i)]]
    return mask_seq

def find_protein_dna_chains(idx_pdb, seq):
    """
    idx_pdb: list of length seq of tuples (pdb index, residue mask)
    seq: list of residue ids

    Returns:
    Ls: list of lengths of chains
    is_protein: boolean array of length sum(Ls) indicating whether each residue is protein (false implies DNA)
    is_protein_chain: boolean array of length len(Ls) indicating whether each chain is protein (false implies DNA)
    """
    protein_chains = []
    dna_chains = []
    token_ub = 31
    
    for i, token in enumerate(seq):
        if token >= 22 and token <= token_ub:
            dna_chains.append(idx_pdb[i][0])
        elif token < 20:
            protein_chains.append(idx_pdb[i][0])

    dna_chains = list(set(dna_chains))
    protein_chains = list(set(protein_chains))

    order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    is_protein = []
    is_protein_chain = []
    Ls = []
    for chain_ID in order:
        if chain_ID in dna_chains:
            Ls.append(len([x for x in idx_pdb if x[0] == chain_ID]))
            is_protein.extend([False] * Ls[-1])
            is_protein_chain.append(False)
        elif chain_ID in protein_chains:
            Ls.append(len([x for x in idx_pdb if x[0] == chain_ID]))
            is_protein.extend([True] * Ls[-1])
            is_protein_chain.append(True)

    return Ls, np.array(is_protein), np.array(is_protein_chain)

def get_default_mask_seq(indep, contig_map, inf_conf,
    mol_classes = ['protein','rna','dna']
    ):
    """
    Generates a vector containing the default output tokens for the full sequence, 
    given the polymer/molecule class in a chain.
    Example (standard) 3-letter codes placed in diffused regions of a sequence when saving pdb files:
     * diffused prot -> ALA
     * diffused rna  ->  RX
     * diffused dna  ->  DX
    This vector is then used to determine what to replace the mask tokens with 
    when saving as a pdb file.
    Args:
        indep      (indep):     used to check seq positions and their corresponding chains (and therefore mol_class)
        contig_map (ContigMap): contains info about mol_class identity (if computed) and assign appropriate mask labels.
        inf_conf   (OmegaConf): allows users to control which output 3letter codes each diffused seq token should map to.
            Example:
             * inf_conf.diffused_mask_codes=['UNK','MAS',..] -> detect these tokens in indep.seq
             * inf_conf.output_mask_codes.protein='ALA', etc -> save diffused-prot positions as 'ALA'

    Output:
        default_seq (torch.Tensor[int]): the seq to use at each diffused position when writing output file [L]
        mask_aas    (torch.Tensor[int]): tensor containing all mask aa tokens to replace in output seq [num_mask_tokens]
    """
    letter2ind = {alpha_i : i for i, alpha_i in enumerate(aa_model.alphabet)}
    # Initialize mask tokens based on given config:
    if 'output_mask_codes' in inf_conf.keys():
        default_3letter_by_class = {
            # check for config spec of default codes, otherwise default back to ALA.
            # also add leading spaces if dict values have less than 3 characters 
            #    (artifact of argument parsing for NA codes)
            mc: inf_conf['output_mask_codes'].get(mc,'ALA').rjust(3)
            for mc in mol_classes
        }
    else:
        default_3letter_by_class = {mc: 'ALA' for mc in mol_classes}

    # Get list of diffused seq mask codes:
    if 'diffused_mask_codes' in inf_conf.keys():
        diffused_mask_codes = inf_conf.diffused_mask_codes
    else:
        diffused_mask_codes = ['UNK','MAS']

    mask_aas = torch.tensor([ChemData().aa2num[code] for code in diffused_mask_codes])
    default_ind_by_class     = {mc: ChemData().aa2num[aa] for mc,aa in default_3letter_by_class.items()}
    default_seq = torch.zeros_like(indep.seq)
    
    # iterate through contig_map data and update sequence and mask according to mol class:
    if hasattr(contig_map, 'mol_classes'):
        for i, chn_i in enumerate(indep.chains()):
            # Only update default seq for specific mol class if we have mol_class data for that chain in the contigs
            if (letter2ind[chn_i] < len(contig_map.mol_classes)) and (not indep.is_sm[i]):
                default_seq[i] = default_ind_by_class[contig_map.mol_classes[letter2ind[chn_i]]]

    return default_seq, mask_aas



def get_resi_type_mask(seq: Union[np.array, torch.Tensor, int, float], nuc_type: str) -> Union[np.array, torch.Tensor, bool]:
    """
    seq: list of residue ids (any shape)
    nuc_type: 'dna', 'rna', or 'na', 'prot', 'prot_and_mask'

    WARNING: incompatibility potential with nucleic acid diffusion depending on diffusion tokenization

    Returns:
    nucl_residues: boolean array of same dimension as input seq indicating whether each residue is DNA
    """
    if nuc_type == 'dna':
        lb = 22
        ub = 26
    elif nuc_type == 'rna':
        lb = 27
        ub = 31
    elif nuc_type == 'na':
        lb = 22
        ub = 31
    elif nuc_type == 'prot':
        lb = 0
        ub = 20
    elif nuc_type == 'prot_and_mask':
        lb = 0
        ub = 21
    if isinstance(seq, torch.Tensor):
        return torch.logical_and((seq >= lb), (seq <= ub))
    elif isinstance(seq, np.ndarray):
        return np.logical_and((seq >= lb), (seq <= ub))
    elif isinstance(seq, (int, float)):
        return (seq >= lb) and (seq <= ub)


def get_dna_residues(seq: torch.Tensor) -> torch.Tensor:
    """
    seq: list of residue ids (any shape)

    Returns:
    dna_residues: boolean array of length len(seq) indicating whether each residue is DNA
    """
    return torch.logical_and((seq >= 22), (seq <= 26))


def get_rna_residues(seq):
    """
    seq: list of residue ids (any shape)

    Returns:
    rna_residues: boolean array of length len(seq) indicating whether each residue is RNA
    """
    return torch.logical_and((seq >= 27), (seq <= 31))


def convert_boolean_mask_to_indices(mask):
    return torch.cumsum(get_dna_residues(mask), 0) - 1


def find_complementary_dna_base(seq, xyz, index):
    # A: N1 close to N3 on T -> seq = 22, atom = 15, seq = 25, atom = 14
    # T: N3 close to N1 on A -> seq = 25, atom = 14, seq = 22, atom = 15
    # G: N1 close to N3 on C -> seq = 24, atom = 15, seq = 23, atom = 14
    # C: N3 close to N1 on G -> seq = 23, atom = 14, seq = 24, atom = 15

    if seq[index] == 22: # A
        target = xyz[index:index+1, 15]
        candidates = xyz[:, 14]
        distances = torch.cdist(target, candidates).flatten()
        distances[seq != 25] = 999999
        distances[distances == 0] = 999999
        return distances.argmin()

    elif seq[index] == 25: # T
        target = xyz[index:index+1, 14]
        candidates = xyz[:, 15]
        distances = torch.cdist(target, candidates).flatten()
        distances[seq != 22] = 999999
        distances[distances == 0] = 999999
        return distances.argmin()

    elif seq[index] == 24: # G
        target = xyz[index:index+1, 15]
        candidates = xyz[:, 14]
        distances = torch.cdist(target, candidates).flatten()
        distances[seq != 23] = 999999
        distances[distances == 0] = 999999
        return distances.argmin()

    elif seq[index] == 23: # C
        target = xyz[index:index+1, 14]
        candidates = xyz[:, 15]
        distances = torch.cdist(target, candidates).flatten()
        distances[seq != 24] = 999999
        distances[distances == 0] = 999999
        return distances.argmin()

    else:
        return 'failed'


def dna_complementarity_direction(seq, xyz, Ls): # TODO better way is to match the substring
    assert len(Ls) == 3
    na_one_start = Ls[0]
    na_two_start = Ls[0] + Ls[1]
    seq_one = seq[na_one_start:na_two_start]
    seq_two = seq[na_two_start:]
    seq_one_complement = torch.tensor([dna_base_complement[res.item()] for res in seq_one])

    if len(seq_one) != len(seq_two):
        return 'reverse' # defaulting to reverse
    
    flip = seq_two.flip(0)
    
    if similar(seq_one_complement, flip) or similar(seq_one_complement[1:], flip[:-1]) or similar(seq_one_complement[:-1], flip[1:]):
        return 'reverse'
    if similar(seq_one_complement, seq_two) or similar(seq_one_complement[1:], seq_two[:-1]) or similar(seq_one_complement[:-1], seq_two[1:]):
        return 'same'
    else: # failed to find complementarity
        return 'reverse'


class NucleicAcid_Interface_Preserving_Crop:
    def __init__(self,
                 contact_type: str, # "protein_dna", "protein_rna", "protein_na"
                 closest_k: int = 2,
                 distance_ball_around_contact_angstroms_min: int = 30,
                 distance_ball_around_contact_angstroms_max: int = 85,
                 chain_search_angstroms: float = 10,
                 max_gap_to_add: int = 5,
                 contact_offcenter_var: float = 3.,
                 min_island_size: int = 10,
                 min_island_size_na: int = 5,
                 max_size: int = 256,
                 min_na_size: int = 4   ,
                 min_prot_size: int = 8,

                 ) -> None:
        self.contact_type = contact_type  
        self.closest_k = closest_k
        self.distance_ball_around_contact_angstroms_min = distance_ball_around_contact_angstroms_min
        self.distance_ball_around_contact_angstroms_max = distance_ball_around_contact_angstroms_max
        self.chain_search_angstroms = chain_search_angstroms
        self.max_gap_to_add = max_gap_to_add
        self.min_island_size = min_island_size
        self.min_island_size_na = min_island_size_na
        self.max_size = max_size
        self.min_na_size = min_na_size
        self.min_prot_size = min_prot_size

        self.contact_offcenter_var = contact_offcenter_var

    # def __call__(self, indep: Indep, atom_mask: torch.Tensor, **kwargs) -> dict:
    #     """
    #     Applies the cropping function to indep

    #     Args:
    #         indep (torch.Tensor): The input data.
    #         **kwargs: Additional keyword arguments.

    #     Returns:
    #         dict: A dictionary containing the modified input data, including indep
    #     """
    #     if not (torch.logical_and(indep.seq >= 22, indep.seq <= 31)).any():
    #         return dict(
    #             indep=indep,
    #             **kwargs
    #         )
    #     indep = indep.clone()
    #     indep.xyz[~atom_mask[:, :ChemData().NHEAVY]] = torch.nan # This line may be causing a nan issue downstream

    #     na_contacts, prot_contacts, prot_chain_indices, dna_chain_indices, rna_chain_indices, distances = self._get_contacts_and_chains(indep)
    #     # detect contacts
    #         # get protein and nucleic residues
    #         # initial implementation will only get them based on indep.seq and then traversing same_chain
        
    #     # pick a contact
    #     selection = np.random.randint(len(na_contacts))
    #     prot_contact_ind, distance = prot_contacts[selection], distances[selection]

    #     prot_contact = torch.nanmean(indep.xyz[prot_contact_ind], dim=0)

    #     prot_contact = prot_contact + (torch.randn_like(prot_contact) * self.contact_offcenter_var)

    #     # get all residues within distance ball
        
    #     # nearby_chain_indices = self._get_all_chains_within_chain_search_angstroms(indep, prot_contact, prot_chain_indices, dna_chain_indices, rna_chain_indices, distance)
    #     crop_indices = self._crop_indices_within_distance(indep, prot_contact, distance)

    #     crop_indices, all_chains_inorder = self._connect_disconnected(indep, crop_indices, prot_chain_indices, dna_chain_indices, rna_chain_indices)

    #     crop_indices = self._drop_islands(indep, crop_indices, all_chains_inorder)


    #     # traverse each chain and find all residues within a distance ball from the contact, and within a backbone atom distance from each other
    #     # might not work well with atomization, would need to figure out how atomization works exactly to confirm
    #     # crop_indices = self._traverse_chains_get_crop_indices(indep, nearby_chain_indices, prot_contact)
    #     pop_mask = torch.zeros_like(indep.seq)
    #     pop_mask[crop_indices] = 1
    #     pop_mask = pop_mask.bool()

    #     # Uncrop tiny proteins or nucleic acids using a standard range cropper
    #     pop_mask = self._uncrop_tiny(indep, pop_mask)

    #     # Pop all masks
    #     aa_model.pop_mask(indep, pop_mask)

    #     indep.xyz = torch.nan_to_num(indep.xyz, nan=0.0)  # To potentially alleviate nan bug, can see later if we can remove

    #     return dict(
    #         indep=indep,
    #         **kwargs
    #     )

    def _uncrop_tiny(self, indep: Indep, pop_mask: torch.Tensor):
        """
        Uncrops small proteins or nucleic acids from the pop mask.
        Logic can be improved to attempt to expand current crop to be larger rather than
        redoing the crop, which could reduce contacts. This function mainly serves
        as a failsafe for poor crops and will be rarely called

        Args:
            indep (Indep): The holy indep
            pop_mask (torch.Tensor): The pop mask.

        Returns:
            torch.Tensor: The updated pop mask.
        """
        is_prot = get_resi_type_mask(indep.seq, 'prot_and_mask')
        is_nucl = get_resi_type_mask(indep.seq, 'na')

        prot_too_small = torch.sum(is_prot[pop_mask]) < self.min_prot_size
        na_too_small = torch.sum(is_nucl[pop_mask]) < self.min_na_size
        if prot_too_small and not na_too_small:
            # Delete small proteins and keep it nucleic acid (convert to RNA/DNA dataset)
            pop_mask[is_prot] = False
        elif na_too_small and not prot_too_small:
            # If nucleic acid is too small, keep current protein and do basic crop using indices of nucleic acid
            pop_mask[is_nucl] = False
            nonzero_indices = self._get_simple_crop_indices(is_nucl, self.max_size - torch.sum(pop_mask[is_prot]).item())
            pop_mask[nonzero_indices] = True
        elif prot_too_small:
            # Remove the smaller of the two and do a random crop, converting the datapoint into an RNA/DNA or pdb_aa datapoint 
            pop_mask[is_nucl] = False
            pop_mask[is_prot] = False            
            if torch.sum(is_prot) < torch.sum(is_nucl):
                nonzero_indices = self._get_simple_crop_indices(is_nucl, self.max_size) 
            else:
                nonzero_indices = self._get_simple_crop_indices(is_prot, self.max_size) 
            pop_mask[nonzero_indices] = True
        return pop_mask

    def _get_simple_crop_indices(self, mask: torch.Tensor, max_size: int) -> torch.Tensor:
        """
        Get a simple crop of the mask to a maximum size.

        Args:
            mask (torch.Tensor): The mask to crop.
            max_size (int): The maximum size of the crop.
            min_size (int): The minimum size of the crop (attempted)

        Returns:
            torch.Tensor: The cropped mask as nonzero indices
        """
        n = torch.sum(mask).item()
        # Select random index
        rand_idx = random.randint(0, n-1)
        end_idx = min([n, rand_idx+max_size])
        if n < max_size:
            start_idx = 0
            end_idx = n
        else:
            start_idx = np.random.randint(0, n - max_size + 1)
            end_idx = start_idx + max_size
        nonzero_indices = mask.nonzero()[start_idx:end_idx]
        return nonzero_indices

    def _crop_max_radial(self, 
                         indep: Indep, 
                         crop_indices: torch.Tensor, 
                         prot_contact: torch.Tensor):
        """
        Crop the `crop_indices` tensor to have a maximum size of `self.max_size` based on radial distance
        from the crop center. 

        Args:
            indep (Indep): The holy Indep
            crop_indices (torch.Tensor): The tensor containing the indices to be cropped. [C,]
            prot_contact (torch.Tensor): The tensor containing the protein contact information, shape [3]

        Returns:
            torch.Tensor: The cropped tensor with a maximum size of `self.max_size`.
        """
        if crop_indices.shape[0] < self.max_size:
            return crop_indices
        residue_coms = torch.nanmean(indep.xyz, dim=1)
        # Get pairwise distances, shape [1, L]
        distances = torch.cdist(torch.unsqueeze(prot_contact, 0), residue_coms).flatten()
        sorted_idx = torch.sort(distances, stable=True).indices.flatten()
        sorted_idx = sorted_idx[torch.isin(sorted_idx, crop_indices)]
        new_crop_indices = sorted_idx[:self.max_size]
        return new_crop_indices

    def _drop_islands(self, 
                      indep: Indep, 
                      crop_indices: torch.Tensor, 
                      all_chains_inorder: List[List[List[int]]]):
        """
        Drops islands from the given crop indices based on the minimum island size.

        Args:
            indep (Indep): The holy Indep.
            crop_indices (torch.Tensor): The indices to crop.
            all_chains_inorder (List[List[List[int]]]): The list of chains with segments of contiguous residues (int)

        Returns:
            torch.Tensor: The cropped indices after dropping islands.
        """
        indices_to_drop = []
        curr_contiguous_segments = []
        for chain in all_chains_inorder:            
            curr_contiguous_segment = []
            # Check all segments for contiguousness
            for segment in chain:
                # Determine min island size for the current segment based on polymer type
                if torch.any(get_resi_type_mask(indep.seq[segment], 'na')):
                    min_island_size = self.min_island_size_na
                else:
                    curr_contiguous_segment.append(index)

                for index in segment:
                    # Decide if this index should be dropped
                    if index not in crop_indices:
                        if len(curr_contiguous_segment) < min_island_size:
                            indices_to_drop.extend(curr_contiguous_segment)
                        curr_contiguous_segments.append(curr_contiguous_segment)
                        curr_contiguous_segment = []
                    else:
                        curr_contiguous_segment.append(index)

                if len(curr_contiguous_segment) < min_island_size:
                    indices_to_drop.extend(curr_contiguous_segment)
                
                curr_contiguous_segments.append(curr_contiguous_segment)
                curr_contiguous_segment = []
        
        # additional logic to remove segments where the min distance from any DNA residue or protein residue on a different chain is greater than X.
        
        # if len(curr_contiguous_segments) >= 3:
        #     for segment in curr_contiguous_segments:
        #         segment_mask = torch.zeros(len(indep.seq))
        #         segment_mask[torch.tensor(segment)] = 1

        #         this_segment_coords = indep.xyz[segment_mask, 1]
        #         rest_coords = indep.xyz[~segment_mask, 1]
        #         distances = torch.cdist(this_segment_coords, rest_coords)
        #         if distances.min() > SOME_THRESHOLD:
        #             indices_to_drop.extend(segment)

        return crop_indices[~torch.isin(crop_indices, torch.tensor(indices_to_drop))]

    

    def _connect_disconnected(self, indep, crop_indices, prot_chain_indices, dna_chain_indices, rna_chain_indices):
        indices_to_add = []
        all_chains_inorder = []
        for chain_all_indices in prot_chain_indices + dna_chain_indices + rna_chain_indices:
            chain_starts = self._find_chain_starts(indep, chain_all_indices)
            chains_inorder = self._get_chain_order_from_chain_start(indep, chain_all_indices, chain_starts)
            all_chains_inorder.append(chains_inorder)
            
            for chain in chains_inorder:
                potential_indices_to_add = []
                curr_gap_length = 0
                
                for index in chain:
                    if index in crop_indices:
                        indices_to_add.extend(potential_indices_to_add)
                        potential_indices_to_add = []
                        curr_gap_length = 0
                        continue

                    curr_gap_length += 1
                    potential_indices_to_add.append(index)
                    if curr_gap_length > self.max_gap_to_add:
                        potential_indices_to_add = []

        print(indices_to_add)
        
        return torch.cat((crop_indices, torch.tensor(indices_to_add))).long(), all_chains_inorder
                    
    
    def _find_chain_starts(self, indep, chain):
        chain_starts = [
            chain[i].item()
            for i, row in enumerate(indep.bond_feats[chain])
            if torch.count_nonzero(row) <= 1
        ]
        return chain_starts
            
    def _get_chain_order_from_chain_start(self, indep, chain, chain_starts):
        visited = set()
        chain_order = []
        last = None

        for start in chain_starts:
            this_chain = []
            if start in visited:
                continue

            this_chain.append(start)
            visited.add(start)
            last = start

            while len(visited) != len(chain):
                bonds = [bond.item() for bond in torch.nonzero(indep.bond_feats[last]).flatten() if bond.item() not in visited]
                if len(bonds) > 1:
                    raise RuntimeError
                elif not bonds:
                    break

                this_chain.append(bonds[0])
                visited.add(bonds[0])
                last = bonds[0]

            chain_order.append(this_chain)

        return chain_order
        
    
    def _crop_indices_within_distance(self, indep, prot_contact, fallback_distance):
        distance = max(fallback_distance, random.randint(self.distance_ball_around_contact_angstroms_min, self.distance_ball_around_contact_angstroms_max))

        prot_contact_unsqueezed = torch.unsqueeze(prot_contact, 0)

        residue_coms = torch.nanmean(indep.xyz, dim=1)
        distances = torch.cdist(prot_contact_unsqueezed, residue_coms).flatten()

        is_prot = get_resi_type_mask(indep.seq, 'prot_and_mask')
        residue_indices_mask_prot = (distances < distance) * is_prot

        is_na = ~is_prot # Can include NA and other things like ligands
        residue_indices_mask_na = (distances < distance_na) * is_na

        # Combine all residues to be maintained
        residue_indices = torch.nonzero(torch.logical_or(residue_indices_mask_prot, residue_indices_mask_na)).flatten()

        return residue_indices

        
    def _traverse_chains_get_crop_indices(self, indep, nearby_chain_indices, prot_contact):
        # does not handle atomization well atm
        # get nearest residue to prot_contact in each chain
        # will need some randomization to expand the crop
        prot_contact_unsqueezed = torch.unsqueeze(prot_contact, 0)

        to_keep = []
        visited = set()

        for chain_indices in nearby_chain_indices:
            chain_atom_coms = torch.nanmean(indep.xyz[chain_indices], dim=1)
            distances = torch.cdist(prot_contact_unsqueezed, chain_atom_coms)

            # populate to_visit with all residues within 10A of the contact

            closest_residue_index = chain_indices[torch.argmin(distances)]

            # traverse
            to_visit = [closest_residue_index.item()]  # Fix: Convert tensor to integer

            while to_visit:
                current_index = to_visit.pop()
                visited.add(current_index)

                if ((torch.nanmean(indep.xyz[current_index], dim=0) - prot_contact) ** 2).sum().sqrt() > self.distance_ball_around_contact_angstroms:
                    continue

                to_keep.append(current_index)

                bonds = [bond.item() for bond in torch.nonzero(indep.bond_feats[current_index]).flatten() if bond.item() not in visited]
                to_visit.extend(bonds)

        return torch.tensor(to_keep, dtype=torch.long)  # Fix: Specify dtype for the tensor


    def _get_contacts_and_chains(self, indep):
        indep = copy.deepcopy(indep)

        prot_chains, dna_chains, rna_chains = self._chop_na_prot_chains(indep)

        if self.contact_type == 'protein_dna':
            na_indices = torch.cat(dna_chains)
        elif self.contact_type == 'protein_rna':
            na_indices = torch.cat(rna_chains)
        elif self.contact_type == 'protein_na':
            na_indices = torch.cat(dna_chains + rna_chains)
        
        indep.xyz[indep.xyz == 0] = torch.nan
        na_com = torch.nanmean(indep.xyz[na_indices], dim=1)
        # replacement_na_coords = indep.xyz[na_indices, 0] # in case of atomization etc
        # Adjust condition to check across the correct dimension for all elements
        # cond = (na_coords != 0).any(dim=-1) & (~torch.isnan(na_coords)).any(dim=-1)
        # na_coords = torch.where(cond.unsqueeze(-1), na_coords, replacement_na_coords)

        prot_indices = torch.cat(prot_chains)
        protein_com = torch.nanmean(indep.xyz[prot_indices], dim=1)
        # replacement_prot_coords = indep.xyz[prot_indices, 0]
        # cond = (prot_coords != 0).any(dim=-1) & (~torch.isnan(prot_coords)).any(dim=-1)
        # prot_coords = torch.where(cond.unsqueeze(-1), prot_coords, replacement_prot_coords)

        
        distances = torch.cdist(na_com, protein_com)
        distances[distances == 0] = 9999999999 # spoofing

        smallest_indices = torch.topk(distances.view(-1), k=self.closest_k, largest=False).indices
        smallest_indices = np.unravel_index(smallest_indices.numpy(), distances.shape)
        distances = distances[smallest_indices]

        na_contacts = na_indices[smallest_indices[0]]
        prot_contacts = prot_indices[smallest_indices[1]]

        return na_contacts, prot_contacts, prot_chains, dna_chains, rna_chains, distances

    def _chop_na_prot_chains(self, indep):
        seq = copy.deepcopy(indep.seq)
        same_chain = copy.deepcopy(indep.same_chain)

        protein_chains = []
        dna_chains = []
        rna_chains = []
        visited_indices = set()
        for i in range(len(seq)):
            if i in visited_indices:
                continue
            if seq[i] < 22: # or something like this
                this_chain = torch.nonzero(same_chain[i]).flatten()
                visited_indices = visited_indices.union(set(this_chain.tolist()))
                protein_chains.append(this_chain)
            elif seq[i] <= 26:
                this_chain = torch.nonzero(same_chain[i]).flatten()
                visited_indices = visited_indices.union(set(this_chain.tolist()))
                dna_chains.append(this_chain)
            elif seq[i] >= 27 and seq[i] <= 31:
                this_chain = torch.nonzero(same_chain[i]).flatten()
                visited_indices = visited_indices.union(set(this_chain.tolist()))
                rna_chains.append(this_chain)
        return protein_chains, dna_chains, rna_chains
    
    def _get_all_chains_within_chain_search_angstroms(self, indep, prot_contact, prot_chain_indices, dna_chain_indices, rna_chain_indices, distance):
        fallback_distance = distance * 2
        search_distance = max(self.chain_search_angstroms, fallback_distance)

        nearby_chain_indices = []
        prot_contact = torch.unsqueeze(prot_contact, 0)

        for chain_indices in prot_chain_indices + dna_chain_indices + rna_chain_indices:
            chain_atom_coords = indep.xyz[chain_indices].reshape(-1, 3)
            
            distances = torch.cdist(prot_contact, chain_atom_coords)
            
            if (distances < search_distance).any():
                nearby_chain_indices.append(chain_indices)

        return nearby_chain_indices
        

        

        # find NA chain indices
        # find protein chains indices
        # get closest contacts between NA chains and proteins

        if self.contact_type == 'protein_dna':
            is_nucleic_acid = get_dna_residues(indep.seq)
            na_coords = indep.xyz[is_nucleic_acid, 18] # C5 on C, N7 on A, N7 on G, C7 on T - should be fine
        elif self.contact_type == 'protein_rna':
            is_nucleic_acid = get_rna_residues(indep.seq)
            raise NotImplementedError
            # Need to ensure the logic below for selecting atoms to use for contacts is done correctly
        elif self.contact_type == 'protein_any':
            is_nucleic_acid = torch.logical_or(get_dna_residues(indep.seq), get_rna_residues(indep.seq))
            raise NotImplementedError
            # Need to ensure the logic below for selecting atoms to use for contacts is done correctly
        
        prot_coords = indep.xyz[~is_nucleic_acid, 1] # Ca on protein

        self.prot_indices = convert_boolean_mask_to_indices(~is_nucleic_acid)
        self.na_indices = convert_boolean_mask_to_indices(is_nucleic_acid)

        # Calculate distances
        distances = torch.cdist(na_coords, prot_coords)
        distances[distances == 0] = 9999999999 # spoofing

        smallest_indices = torch.topk(distances.view(-1), k=self.closest_k, largest=False).indices
        smallest_indices = np.unravel_index(smallest_indices.numpy(), distances.shape)
        na_contacts = self.na_indices[smallest_indices[0]]
        prot_contacts = self.prot_indices[smallest_indices[1]]

        return na_contacts, prot_contacts
    
    def _get_valid_expand(self, real_Ls, na_contact_ind, complementary_ind, complementarity, direction):
        ranges_inclusive = [(0, real_Ls[0])-1]
        for L in real_Ls[1:]:
            ranges_inclusive.append((ranges_inclusive[-1][1] + 1, ranges_inclusive[-1][1] + 1 + L))

        if complementarity == 'none':
            contact_ind_range = self._find_range(ranges_inclusive, na_contact_ind)
            if direction == 'left':
                max_left_expand = na_contact_ind - contact_ind_range[0]
                return max_left_expand
            elif direction == 'right':
                max_right_expand = contact_ind_range[1] - na_contact_ind
                return max_right_expand

        if direction == 'left':
            if complementarity == 'same':
                contact_ind_range = self._find_range(ranges_inclusive, na_contact_ind)
                complementary_ind_range = self._find_range(ranges_inclusive, complementary_ind)
                max_left_expand = min(na_contact_ind - contact_ind_range[0], complementary_ind - complementary_ind_range[0])
                return max_left_expand
            elif complementarity == 'reverse':
                contact_ind_range = self._find_range(ranges_inclusive, na_contact_ind)
                complementary_ind_range = self._find_range(ranges_inclusive, complementary_ind)
                max_left_expand = min(na_contact_ind - contact_ind_range[0], complementary_ind_range[1] - complementary_ind)
                return max_left_expand
        elif direction == 'right':
            if complementarity == 'same':
                contact_ind_range = self._find_range(ranges_inclusive, na_contact_ind)
                complementary_ind_range = self._find_range(ranges_inclusive, complementary_ind)
                max_right_expand = min(contact_ind_range[1] - na_contact_ind, complementary_ind_range[1] - complementary_ind)
                return max_right_expand
            elif complementarity == 'reverse':
                contact_ind_range = self._find_range(ranges_inclusive, na_contact_ind)
                complementary_ind_range = self._find_range(ranges_inclusive, complementary_ind)
                max_right_expand = min(contact_ind_range[1] - na_contact_ind, complementary_ind - complementary_ind_range[0])
                return max_right_expand
            
    
    def _get_valid_protein_expand(self, real_Ls, prot_contact_ind):
        ranges_inclusive = [(0, real_Ls[0])-1]
        for L in real_Ls[1:]:
            ranges_inclusive.append((ranges_inclusive[-1][1] + 1, ranges_inclusive[-1][1] + 1 + L))
        
        contact_ind_range = self._find_range(ranges_inclusive, prot_contact_ind)
        max_left_expand = prot_contact_ind - contact_ind_range[0]
        max_right_expand = contact_ind_range[1] - prot_contact_ind
        return max_left_expand, max_right_expand
    
    
    def _find_range(self, ranges_inclusive, index):
        for range in ranges_inclusive:
            if index >= range[0] and index <= range[1]:
                return range

    
    def __call__(self, indep, real_Ls, base_complementarity_indices_map, **kwargs):
        na_contacts, prot_contacts = self._get_contacts(indep)

        selection = np.random.randint(len(na_contacts))
        na_contact_ind, prot_contact_ind = na_contacts[selection], prot_contacts[selection]

        Ls = indep.get_Ls()

        if na_contact_ind in base_complementarity_indices_map.keys():
            complementary_dna_base = base_complementarity_indices_map[na_contact_ind]

            # Direction of complementarity
            if base_complementarity_indices_map[na_contact_ind + 1] == complementary_dna_base + 1:
                complementarity = 'same'
            else:
                complementarity = 'reverse'

            max_left_expand = self._get_valid_expand(real_Ls, na_contact_ind, complementary_dna_base, complementarity, 'left')
            max_right_expand = self._get_valid_expand(real_Ls, na_contact_ind, complementary_dna_base, complementarity, 'right')

            left_na_expand = np.random.randint(min(self.min_nucleic_crop, max_left_expand), min(max_left_expand + 1, self.max_nucleic_crop))
            right_na_expand = np.random.randint(min(self.min_nucleic_crop, max_right_expand), min(max_right_expand + 1, self.max_nucleic_crop))

            na_start_one = na_contact_ind - left_na_expand
            na_end_one = na_contact_ind + right_na_expand

            if complementarity == 'same':
                na_start_two = complementary_dna_base - left_na_expand
                na_end_two = complementary_dna_base + right_na_expand
            else:
                na_start_two = complementary_dna_base - right_na_expand
                na_end_two = complementary_dna_base + left_na_expand

            prot_crop_left_max, prot_crop_right_max = self._get_valid_protein_expand(real_Ls, prot_contact_ind)

            prot_crop_left = np.random.randint(min(self.min_prot_crop, prot_crop_left_max), min(self.max_prot_crop, prot_crop_left_max + 1))
            prot_crop_right = np.random.randint(min(self.min_prot_crop, prot_crop_right_max), min(self.max_prot_crop, prot_crop_right_max + 1))
            prot_crop_start = prot_contact_ind - prot_crop_left
            prot_crop_end = prot_contact_ind + prot_crop_right

            crop = torch.zeros(sum(Ls))
            crop[na_start_one:na_end_one] = 1
            crop[na_start_two:na_end_two] = 1
        else:
            max_left_expand = self._get_valid_expand(real_Ls, na_contact_ind, None, 'none', 'left')
            max_right_expand = self._get_valid_expand(real_Ls, na_contact_ind, None, 'none', 'right')
            left_na_expand = np.random.randint(min(self.min_nucleic_crop, max_left_expand), min(max_left_expand + 1, self.max_nucleic_crop))
            right_na_expand = np.random.randint(min(self.min_nucleic_crop, max_right_expand), min(max_right_expand + 1, self.max_nucleic_crop))
            na_start_one = na_contact_ind - left_na_expand
            na_end_one = na_contact_ind + right_na_expand

            prot_crop_left_max, prot_crop_right_max = self._get_valid_protein_expand(real_Ls, prot_contact_ind)

            prot_crop_left = np.random.randint(min(self.min_prot_crop, prot_crop_left_max), min(self.max_prot_crop, prot_crop_left_max + 1))
            prot_crop_right = np.random.randint(min(self.min_prot_crop, prot_crop_right_max), min(self.max_prot_crop, prot_crop_right_max + 1))
            prot_crop_start = prot_contact_ind - prot_crop_left
            prot_crop_end = prot_contact_ind + prot_crop_right

            crop = torch.zeros(sum(Ls))
            crop[na_start_one:na_end_one] = 1
        crop[prot_crop_start:prot_crop_end] = 1

        return crop.bool()

class Clean_NA_Chains_Find_Complements:
    def __call__(self, indep, **kwargs):
        Ls = indep.get_Ls()
        is_dna = get_dna_residues(indep.seq)
        dna_chains = self._get_dna_chains(indep, is_dna, Ls)

        unaccounted_break_indices = self._search_for_unaccounted_break(indep.xyz, is_dna, Ls)
        if len(unaccounted_break_indices) == 0 and sum(dna_chains) == 1: # single stranded
            return dict(
                indep=indep,
                base_complementarity_indices_map={},
                real_Ls = Ls,
                **kwargs
            )
        
        elif len(unaccounted_break_indices) == 0: # no breaks needed to fix
            complementarity = self._get_complementarity(indep, Ls, is_dna, dna_chains)
            return dict(
                indep=indep,
                base_complementarity_indices_map=complementarity,
                real_Ls = Ls,
                **kwargs
            )
        else:
            real_Ls = self._get_correct_Ls(Ls, unaccounted_break_indices)
            dna_chains = self._get_dna_chains(indep, is_dna, real_Ls)
            complementarity = self._get_complementarity(indep, real_Ls, is_dna, dna_chains)

            return dict(
                indep=indep,
                base_complementarity_indices_map=complementarity,
                real_Ls = real_Ls,
                **kwargs
            )

    def _get_dna_chains(self, indep, is_dna, Ls):
        cum_Ls = np.cumsum(Ls)
        is_dna_chain = []
        for index in cum_Ls:
            if is_dna[index - 1]:
                is_dna_chain.append(True)
            else:
                is_dna_chain.append(False)

        return np.array(is_dna_chain)
    
    def _search_for_unaccounted_break(self, xyz, is_dna, Ls):
        accounted_breaks = [(Ls[0] - 1, Ls[0])]
        for i in range(1, len(Ls)):
            accounted_breaks.append((accounted_breaks[-1][1] + Ls[i], accounted_breaks[-1][1] + Ls[i] + 1))

        break_indices = []
        for i in range(1, len(is_dna)):
            if (i-1, i) in accounted_breaks:
                continue
            if not (is_dna[i] and is_dna[i - 1]):
                continue
            diff = ((xyz[i, 1] - xyz[i-1, 1])**2).sum()**0.5
            if diff > 9:
                break_indices.append((i-1, i))

        return break_indices
    
    def _get_correct_Ls(self, Ls, break_indices):
        new_Ls = []
        ranges_inclusive = [(0, Ls[0])-1]
        for L in Ls[1:]:
            ranges_inclusive.append((ranges_inclusive[-1][1] + 1, ranges_inclusive[-1][1] + 1 + L))

        while True:
            for L, range in zip(Ls, ranges_inclusive):
                for break_index in break_indices:
                    if break_index[0] > range[0] and break_index[1] < range[1]:
                        new_Ls.append(break_index[0] - range[0] + 1)
                        new_Ls.append(range[1] - break_index[1] + 1)

                    else:
                        new_Ls.append(L)
            if len(new_Ls) == len(Ls):
                break
            Ls = new_Ls
            new_Ls = []

        return new_Ls

def get_atom_coordinates(NA_seq, NA_coords, residue_token_index, atom_token_indices):
    indices = NA_seq == residue_token_index
    coordinates = NA_coords[indices]
    coordinates = coordinates[:, torch.tensor(atom_token_indices), :].reshape(-1, 3)
    return coordinates

def get_dna_contacts(indep, distance_cutoff):
    """
    want to find all contacts to 
    C - C6, C5, N4 -> seqtoken 23, atom indices 18, 17, 16
    A - N7, N6 -> seqtoken 22, atom indices 18, 20
    G - N7, O6 -> seqtoken 24, atom_indices 18, 21
    T - C6, C7 -> seqtoken 25, atom_indices 19, 18
    
    so we first isolate these atoms from xyz
    then we compute pairwise distances
    then we find indices where these pairwise distances is smaller then X
    and we return those indices
    """

    is_dna = get_resi_type_mask(indep.seq, 'dna')
    is_prot = get_resi_type_mask(indep.seq, 'prot')

    if len(is_dna) == 0 or len(is_prot) == 0:
        return None

    na_seq = indep.seq[is_dna]
    na_coords = indep.xyz[is_dna]
    DNAC_coords = get_atom_coordinates(na_seq, na_coords, 23, [16, 17, 18])
    DNAA_coords = get_atom_coordinates(na_seq, na_coords, 22, [18, 20])
    DNAG_coords = get_atom_coordinates(na_seq, na_coords, 24, [18, 21])
    DNAT_coords = get_atom_coordinates(na_seq, na_coords, 25, [18, 19])

    DNA_coords = torch.cat((DNAC_coords, DNAA_coords, DNAG_coords, DNAT_coords), dim=0)

    # prot_seq = indep.seq[~indep.is_nucleic_acid]
    prot_indices = torch.nonzero(is_prot).flatten()
    prot_coords = indep.xyz[is_prot]

    distances = torch.cdist(prot_coords, DNA_coords)
    contact_residues = torch.logical_and((distances < distance_cutoff), (distances != 0)).any(dim=1).any(dim=1).nonzero().flatten()

    if len(contact_residues) == 0:
        return None
    
    prot_contact_indices = prot_indices[contact_residues].unique()
    return prot_contact_indices

def get_atomize_prot_na_bond_feats(i_start, msa, ra, n_res_atomize=5):
    """ 
    generate atom bond features for atomized residues 
    currently ignores long-range bonds like disulfides
    """
    ra2ind = {tuple(two_d.numpy()): i for i, two_d in enumerate(ra)}
    N = len(ra2ind.keys())
    bond_feats = torch.zeros((N, N))
    for i, res in enumerate(msa[0, i_start:i_start+n_res_atomize]):
        for j, bond in enumerate(ChemData().aabonds[res]):
            start_idx = ChemData().aa2long[res].index(bond[0])
            end_idx = ChemData().aa2long[res].index(bond[1])
            if (i, start_idx) not in ra2ind or (i, end_idx) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i, start_idx)]
            end_idx = ra2ind[(i, end_idx)]

            # maps the 2d index of the start and end indices to btype
            bond_feats[start_idx, end_idx] = ChemData().aabtypes[res][j]
            bond_feats[end_idx, start_idx] = ChemData().aabtypes[res][j]
        #accounting for peptide bonds
        if res <= 20 and i > 0:
            if (i-1, 2) not in ra2ind or (i, 0) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i-1, 2)]
            end_idx = ra2ind[(i, 0)]
            bond_feats[start_idx, end_idx] = ChemData().SINGLE_BOND
            bond_feats[end_idx, start_idx] = ChemData().SINGLE_BOND
        if res >= 22 and res <= 31 and i > 0:
            if (i-1, 2) not in ra2ind or (i, 0) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            if res <= 26:
                DA = ChemData().aa2num[' DA'] # Assume all bases are the same for base positions
                start_idx = ra2ind[(i-1, ChemData().aa2long[DA].index('  P '))]
                end_idx = ra2ind[(i,  ChemData().aa2long[DA].index(" O3'"))]
            if res >= 27:
                RA = ChemData().aa2num[' RA'] # Assume all bases are the same for base positions
                start_idx = ra2ind[(i-1, ChemData().aa2long[RA].index('  P '))]
                end_idx = ra2ind[(i,  ChemData().aa2long[RA].index(" O3'"))]
            bond_feats[start_idx, end_idx] = ChemData().SINGLE_BOND
            bond_feats[end_idx, start_idx] = ChemData().SINGLE_BOND
    return bond_feats


def atomize_prot_na(i_start, msa, xyz, mask, n_res_atomize=5):
    """ given an index i_start, make the following flank residues into "atom" nodes """
    residues_atomize = msa[0, i_start:i_start+n_res_atomize]
    residues_atom_types = [ChemData().aa2elt[num][:ChemData().NHEAVY] for num in residues_atomize]
    residue_atomize_mask = mask[i_start:i_start+n_res_atomize].float() # mask of resolved atoms in the sidechain
    residue_atomize_allatom_mask = ChemData().allatom_mask[residues_atomize][:, :ChemData().NHEAVY] # the indices that have heavy atoms in that sidechain
    xyz_atomize = xyz[i_start:i_start+n_res_atomize]

    # handle symmetries
    xyz_alt = torch.zeros_like(xyz.unsqueeze(0))
    xyz_alt.scatter_(2, ChemData().long2alt[msa[0],:,None].repeat(1,1,1,3), xyz.unsqueeze(0))
    xyz_alt_atomize = xyz_alt[0, i_start:i_start+n_res_atomize]

    coords_stack = torch.stack((xyz_atomize, xyz_alt_atomize), dim=0)
    swaps = (coords_stack[0] == coords_stack[1]).all(dim=1).all(dim=1).squeeze() #checks whether theres a swap at each position
    swaps = torch.nonzero(~swaps).squeeze() # indices with a swap eg. [2,3]
    if swaps.numel() != 0:
        # if there are residues with alternate numbering scheme, create a stack of coordinate with each combo of swaps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=UserWarning)
            combs = torch.combinations(torch.tensor([0,1]), r=swaps.numel(), with_replacement=True) #[[0,0], [0,1], [1,1]]
        stack = torch.stack((combs, swaps.repeat(swaps.numel()+1,1)), dim=-1).squeeze()
        coords_stack = coords_stack.repeat(swaps.numel()+1,1,1,1)
        nat_symm = coords_stack[0].repeat(swaps.numel()+1,1,1,1) # (N_symm, num_atomize_residues, natoms, 3)
        swapped_coords = coords_stack[stack[...,0], stack[...,1]].squeeze(1) #
        nat_symm[:,swaps] = swapped_coords
    else:
        nat_symm = xyz_atomize.unsqueeze(0)
    # every heavy atom that is in the sidechain is modelled but losses only applied to resolved atoms
    ra = residue_atomize_allatom_mask.nonzero()
    lig_seq = torch.tensor([ChemData().aa2num[residues_atom_types[r][a]] if residues_atom_types[r][a] in ChemData().aa2num else ChemData().aa2num["ATM"] for r,a in ra])
    ins = torch.zeros_like(lig_seq)

    r,a = ra.T
    lig_xyz = torch.zeros((len(ra), 3))
    lig_xyz = nat_symm[:, r, a]
    lig_mask = residue_atomize_mask[r, a].repeat(nat_symm.shape[0], 1)
    bond_feats = get_atomize_prot_na_bond_feats(i_start, msa, ra, n_res_atomize=n_res_atomize)
    #HACK: use networkx graph to make the atom frames, correct implementation will include frames with "residue atoms"
    G = nx.from_numpy_array(bond_feats.numpy())
        
    frames = get_atom_frames(lig_seq, G)
    chirals = get_atomize_prot_na_chirals(residues_atomize, lig_xyz[0], residue_atomize_allatom_mask, bond_feats)
    return lig_seq, ins, lig_xyz, lig_mask, frames, bond_feats, ra, chirals


def get_atomize_prot_na_chirals(residues_atomize, lig_xyz, residue_atomize_mask, bond_feats):
    """
    Enumerate chiral centers in residues and provide features for chiral centers
    """
    angle = np.arcsin(1/3**0.5) # perfect tetrahedral geometry
    chiral_atoms = ChemData().aachirals[residues_atomize]
    ra = residue_atomize_mask.nonzero()
    r,a = ra.T

    chiral_atoms = chiral_atoms[r,a].nonzero().squeeze(1) #num_chiral_centers
    num_chiral_centers = chiral_atoms.shape[0]
    chiral_bonds = bond_feats[chiral_atoms] # find bonds to each chiral atom
    chiral_bonds_idx = chiral_bonds.nonzero() # find indices of each bonded neighbor to chiral atom
    # in practice all chiral atoms in proteins have 3 heavy atom neighbors, so reshape to 3 
    chiral_bonds_idx = chiral_bonds_idx.reshape(num_chiral_centers, 3, 2)
    
    chirals = torch.zeros((num_chiral_centers, 5))
    chirals[:,0] = chiral_atoms.long()
    chirals[:, 1:-1] = chiral_bonds_idx[...,-1].long()
    chirals[:, -1] = angle
    n = chirals.shape[0]
    if n>0:
        chirals = chirals.repeat(3,1).float()
        chirals[n:2*n,1:-1] = torch.roll(chirals[n:2*n,1:-1],1,1)
        chirals[2*n: ,1:-1] = torch.roll(chirals[2*n: ,1:-1],2,1)
        dih = get_dih(*lig_xyz[chirals[:,:4].long()].split(split_size=1,dim=1))[:,0]
        chirals[dih<0.0,-1] = -angle
    else:
        chirals = torch.zeros((0,5))
    return chirals

def get_nucl_prot_contacts(indep: Indep, 
                           dist_thresh: float = 4.5,
                           is_gp: Union[bool, torch.Tensor] = False,
                           ignore_prot_bb: bool = False) \
    -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate nucleic acid-protein contacts.

    Args:
        indep (Indep): The input object containing sequence and coordinates.
        dist_thresh (float, optional): The distance threshold for defining contacts. Defaults to 4.5.
        is_gp (bool, Tensor[bool]): tensor describing if each residue is guide post or not
        ignore_prot_sc (bool): whether to ignore sidechain atoms in the protein


    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two tensors:
            - normal_contacts_indices: Indices of nucleic acid-protein contacts. [n_contacts,]
            - base_contacts_indices: Indices of base-specific nucleic acid-protein contacts. [n_base_contacts,]
    """
    if isinstance(is_gp, bool):
        is_gp = torch.ones(indep.length(), dtype=bool) * is_gp
    # Get masks
    is_nucl = get_resi_type_mask(indep.seq, 'na') * ~is_gp * ~indep.is_sm
    nucl_is_rna = get_resi_type_mask(indep.seq[is_nucl], 'rna')
    nucl_is_dna = get_resi_type_mask(indep.seq[is_nucl], 'dna')
    is_prot = get_resi_type_mask(indep.seq, 'prot_and_mask') * ~is_gp * ~indep.is_sm

    # Get coordinates and distances between proteins and nucleic acids
    xyz_prot = indep.xyz[is_prot,:]            
    xyz_nucl = indep.xyz[is_nucl,:]
    is_valid_prot = ~torch.any(torch.logical_or(xyz_prot == 0.0, torch.isnan(xyz_prot)), dim=2)
    is_valid_nucl = ~torch.any(torch.logical_or(xyz_nucl == 0.0, torch.isnan(xyz_nucl)), dim=2)
    if ignore_prot_bb:
        # Discard N, CA, C, O, CB in this case
        is_valid_prot[:,:5] = False
    A, B = xyz_prot.shape[0], xyz_prot.shape[1]
    C, D = xyz_nucl.shape[0], xyz_nucl.shape[1]
    cdist = torch.cdist(xyz_prot.view(-1, 3), xyz_nucl.view(-1, 3)).view(A, B, C, D)
    cdist = torch.nan_to_num(cdist, 99999)
    
    # Get the mask of invalid positions and make invalid pair have a large distance
    pair_mask_valid = is_valid_prot[:,:,None,None] * is_valid_nucl[None,None,:,:]
    normal_cdist = cdist*pair_mask_valid + (~pair_mask_valid) * 99999
    # Calculate the minimum values for each nucleic acid base position    
    normal_cdist_min = torch.min(torch.min(normal_cdist, dim=0)[0], dim=0)[0]
    normal_contacts = torch.any(normal_cdist_min < dist_thresh, dim=1)

    # Restore original indices
    normal_contacts_full = torch.zeros(indep.length(), dtype=bool)
    normal_contacts_full[is_nucl] = normal_contacts

    # Now get base speciic contacts by masking off positions that are not bases
    is_valid_nucl_bases = is_valid_nucl.clone()
    is_valid_nucl_bases[nucl_is_dna, :11] = False # Hardcoded, but depends on chemdata aa2long ordering
    is_valid_nucl_bases[nucl_is_rna, :12] = False # Hardcoded, but depends on chemdata aa2long ordering

    # Get the mask of invalid positions and make invalid pair have a large distance    
    pair_mask_valid_bases = is_valid_prot[:,:,None,None] * is_valid_nucl_bases[None,None,:,:]    
    # Because it's subset of normal, do not need to recompute normal_cdist
    base_cdist = cdist*pair_mask_valid_bases + (~pair_mask_valid_bases) * 99999
    # Calculate the minimum values for each nucleic acid base position
    base_cdist_min = torch.min(torch.min(base_cdist, dim=0)[0], dim=0)[0]
    base_contacts = torch.any(base_cdist_min < dist_thresh, dim=1)   

    # Restore original indices
    base_contacts_full = torch.zeros(indep.length(), dtype=bool)
    base_contacts_full[is_nucl] = base_contacts

    # Get the true indices for return
    normal_contacts_indices = torch.nonzero(normal_contacts_full, as_tuple=False).flatten()
    base_contacts_indices = torch.nonzero(base_contacts_full, as_tuple=False).flatten()  
    return normal_contacts_indices, base_contacts_indices


def protein_dna_sidechain_base_contacts(indep, contact_distance, expand_prot=True):
    is_dna = get_resi_type_mask(indep.seq, 'dna')
    protein_index_seq_residue = {}
    na_index_seq_residue = {}
    if not is_dna.any():
        return None, None
    for i, (residue_id, coords, isna) in enumerate(zip(indep.seq, indep.xyz, is_dna)):
        if isna:
            na_index_seq_residue[i] = (residue_id.item(), coords.clone().detach())
        else:
            protein_index_seq_residue[i] = (residue_id.item(), coords.clone().detach())

    # we will be using atoms according to the following dictionary to detect base contacts
    sidechain_atom_dict = {
        15:[5],
        16:[5],
        2:[6, 7],
        5:[7, 8],
        18:[11],
        1:[7, 9, 10],
        11:[8],
        6:[7, 8],
        3:[6, 7],
        8:[6, 9],
        17:[8],
        22:[18, 20],
        25:[16],
        23:[16],
        24:[18, 21]
    }

    sidechain_atom_coords = []
    sidechain_atom_indices = []

    na_base_atom_coords = []
    na_base_atom_indices = []

    for i, (residue_id, coords) in protein_index_seq_residue.items():
        if residue_id not in sidechain_atom_dict.keys():
            continue
        for atom in sidechain_atom_dict[residue_id]:
            sidechain_atom_coords.append(coords[atom, :])
            sidechain_atom_indices.append(i)

    for i, (residue_id, coords) in na_index_seq_residue.items():
        if residue_id not in sidechain_atom_dict.keys():
            continue
        for atom in sidechain_atom_dict[residue_id]:
            na_base_atom_coords.append(coords[atom, :])
            na_base_atom_indices.append(i)

    if not sidechain_atom_coords or not na_base_atom_coords:
        return None, None

    sidechain_atom_coords = torch.stack(sidechain_atom_coords, dim=0)
    na_base_atom_coords = torch.stack(na_base_atom_coords, dim=0)

    # calculate pairwise distances between sidechain_atom_coords and na_base_atom_coords
    # (n_sidechain_atoms, n_na_base_atoms)
    pdist_matrix = torch.cdist(sidechain_atom_coords, na_base_atom_coords)

    # get indices of sidechain atoms that are within 3.5 angstroms of a base atom
    contacts = torch.where(pdist_matrix < contact_distance, 1, 0).bool()
    if not contacts.any():
        return None, None

    prot_contacts = contacts.any(dim=1).detach().cpu()
    dna_contacts = contacts.any(dim=0).detach().cpu()

    protein_contact_indices = torch.tensor(sidechain_atom_indices)[prot_contacts]
    na_contact_indices = torch.tensor(na_base_atom_indices)[dna_contacts]

    # expanding contacts to fill in the gaps
    if expand_prot:
        if protein_contact_indices.max() - protein_contact_indices.min() < 8:
            protein_contact_indices = torch.arange(protein_contact_indices.min(), protein_contact_indices.max()+1).detach().cpu().numpy().tolist()
        else:
            protein_contact_indices = list(set(protein_contact_indices.detach().cpu().numpy().tolist()))

        # just adding adjacent residues either side of the contact
        if len(protein_contact_indices) <= 3:
            for index in protein_contact_indices.copy():
                if index >= 1: protein_contact_indices.append(index - 1)
                if index + 1 < len(indep.seq): protein_contact_indices.append(index + 1)

    na_contact_indices = list(set(na_contact_indices.detach().cpu().numpy().tolist()))

    return torch.tensor(protein_contact_indices), torch.tensor(na_contact_indices)


### Transmutate code ###

def create_orthonormal_basis(vectors: torch.Tensor) -> torch.Tensor:
    """
    Create an orthonormal basis from a set of input vectors.

    Args:
        vectors (torch.Tensor): Input vectors of shape [B, 4, 3], where B is the batch size.

    Returns:
        torch.Tensor: Orthonormal basis matrix of shape [B, 3, 3], where each column represents a basis vector.

    """                                                                                                                                                                                                                
    # B = vectors.shape[0]
    # Extract vectors v0 (origins), v1, and v2                                                                                                                                                                                                               
    v0 = vectors[:, 0, :]  # Shape [B, 3]                                                                                                                                                                                                                    
    v1 = vectors[:, 1, :]  # Shape [B, 3]                                                                                                                                                                                                                    
    v2 = vectors[:, 2, :]  # Shape [B, 3]                                                                                                                                                                                                                    

    # Compute x_hat (unit vector in direction of v1 - v0)                                                                                                                                                                                                    
    x_hat = v1 - v0
    x_hat = x_hat / torch.norm(x_hat, dim=1, keepdim=True)  # Normalize                                                                                                                                                                                      
    # Compute projection of (v2-v0) onto plane perpendicular to x_hat                                                                                                                                                                                        
    v2_proj = v2 - v0
    v2_proj -= (v2_proj * x_hat).sum(dim=1, keepdim=True) * x_hat  # Remove component along x_hat                                                                                                                                                            

    # Normalize v2_proj to get y_hat                                                                                                                                                                                                                         
    y_hat = v2_proj / torch.norm(v2_proj, dim=1, keepdim=True)
    # Compute z_hat as cross product of x_hat and y_hat                                                                                                                                                                                                      
    z_hat = torch.cross(x_hat, y_hat, dim=1)
    # Stack x_hat, y_hat, z_hat to form the transformation matrix M (B, 3, 3)                                                                                                                                                                                
    M = torch.stack((x_hat, y_hat, z_hat), dim=-1)  # Columns are basis vectors                                                                                                                                                                              

    return M

class TransmuteNA:
    """
    A class that provides methods for transmuting nucleic acids (DNA to RNA and RNA to DNA) by 
    converting the sequence and updating the coordinates.
    """

    # Empirically derived data for estimating imputed atom positions @Altaeth for questions
    T_mean = np.array([-7.4354905e-01, -1.2985681e+00, -1.1353756e-03], dtype=np.float32)
    T_cov = np.array([[ 5.4675451e-04, -5.2435248e-04,  5.6376348e-06],
                      [-5.2435248e-04,  1.1092696e-03, -5.0541353e-06],
                      [ 5.6376348e-06, -5.0541353e-06,  2.4706524e-04]], dtype=np.float32)
    O2_mean = np.array([-0.46448016, -0.63049215, -1.1776375 ], dtype=np.float32)
    O2_cov = np.array([[ 0.00131963,  0.00122451, -0.00129338],
                       [ 0.00122451,  0.00231871, -0.00190492],
                       [-0.00129338, -0.00190492,  0.00175247]], dtype=np.float32)

    @staticmethod
    def init_data(xyz: torch.Tensor) -> Tuple[dict, dict]:
        """
        Initializes the data for transm_seq_mapper and transm_xyz_atom_mapper for performing
        mapping between nucleic acid sequences and corresponding atomic coordinates.
        xyz input is used to deterine the shape information of the number of atomic coordinates

        Args:
            xyz (torch.Tensor): The input tensor containing atomic coordinates. [B, A, 3] where num atoms, B, can be any number >= ChemData().NHEAVY

        Returns:
            tuple: A tuple containing the transm_seq_mapper and transm_xyz_atom_mapper dictionaries.
        """
        assert xyz.shape[1] >= ChemData().NHEAVY, "Input tensor must have at least ChemData().NHEAVY atoms to determine shape information"

        # Determine sequence mapping for transmutating 
        transm_seq_mapper = {}
        for r in ['A', 'C', 'G', 'T', 'X']:
            i = ' D'+r
            j = ' R'+r.replace('T', 'U')
            idx_i = ChemData().aa2num[i]
            idx_j = ChemData().aa2num[j]
            transm_seq_mapper[idx_i] = idx_j
            transm_seq_mapper[idx_j] = idx_i

        transm_xyz_atom_mapper = {}
        na_base_idx = [ChemData().aa2num[b] for b in [' DA', ' DC', ' DG', ' DT', ' DX', 
                                                        ' RA', ' RC', ' RG', ' RU', ' RX']]
        for i in na_base_idx:
            j = transm_seq_mapper[i]
            # Generate mapping between atomic coordinates in original vs transmuted base 
            orig_map = {k : v for k,v in enumerate(ChemData().aa2long[i][:ChemData().NHEAVY]) 
                        if v is not None}
            new_map_rev = {v : k for k,v in enumerate(ChemData().aa2long[j][:ChemData().NHEAVY]) 
                           if v is not None}
            transm_map = {u : new_map_rev[orig_map[u]]  # Calculate chained mapping
                        for u in range(ChemData().NHEAVY) 
                        if u in orig_map 
                        and orig_map[u] in new_map_rev}
            # sort via V then U to ensure resultant mask i in sorted order 
            U,V = zip(*sorted(transm_map.items(), key=lambda x: (x[1], x[0])))
            # Conver the V list into a mask for easy indexing
            V_mask = torch.zeros_like(xyz[0,:,0], device=xyz.device, dtype=bool)
            V_mask[list(V)] = True    
            transm_xyz_atom_mapper[i] = (U,V_mask)

        return transm_seq_mapper, transm_xyz_atom_mapper

    @staticmethod
    def transmute_dna_to_rna(seq: torch.Tensor, 
                             xyz: torch.Tensor, 
                             seq_new: torch.Tensor,
                             xyz_new: torch.Tensor, 
                             mask: torch.Tensor,
                             safe: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transmutes DNA to RNA by converting the sequence and updating the coordinates.
        Uses statitical imputation for missing atoms with empirically determined multivaraite normal distributions.
        Logically, set mask to is_dna from the original sequence before calling this function.

        Args:
            seq (torch.Tensor): The original sequence tensor.
            xyz (torch.Tensor): The original coordinate tensor.
            seq_new (torch.Tensor): The new sequence tensor.
            xyz_new (torch.Tensor): The new coordinate tensor.
            mask (torch.Tensor): The mask indicating which elements are DNA and will have the transmute applied to.
            safe (bool): If True, will do extra checking to ensure that the input is valid. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated sequence tensor and coordinate tensor.
        """
        # Generate initial data 
        transm_seq_mapper, transm_xyz_atom_mapper = TransmuteNA.init_data(xyz)

        dna_base_idx = [ChemData().aa2num[b] for b in [' DA', ' DC', ' DG', ' DT', ' DX']]
        for r in dna_base_idx: 
            is_r = torch.logical_and(seq == r, mask)
            U, V_mask = transm_xyz_atom_mapper[r]
            xyz_new[is_r.unsqueeze(1)*V_mask.unsqueeze(0)] = xyz[is_r][:,U].reshape(-1, 3) 
        # Add back in the O2  to all RNAs
        is_r = mask
        # Use adenine rna as a base index (assumes sugar atom positions are the same for all nucleotides in RNA)
        id_ex = ChemData().aa2num[' RA']
        # Generate basis vectors
        vectors = xyz_new[is_r][:,[ChemData().aa2long[id_ex].index(" C2'"),
                                ChemData().aa2long[id_ex].index(" C1'"),
                                ChemData().aa2long[id_ex].index(" C3'"),
                                ChemData().aa2long[id_ex].index(" O2'")]]
        # Assert inputed position is missing
        if safe:
            assert torch.all(torch.isnan(vectors[:,3])) or torch.all(vectors[:,3] == 0.0), "O2' atom must be missing or zeroed"

        vectors = vectors[:,:3]
        # Do not worry about nans / missing atoms, as this stage should not have any            
        T_M = create_orthonormal_basis(vectors)
        # Sample from normal distribution and transform back to original basis
        dist = MultivariateNormal(torch.tensor(TransmuteNA.O2_mean, device=seq.device), 
                                torch.tensor(TransmuteNA.O2_cov, device=seq.device))
        samples = dist.sample((torch.sum(is_r),))
        samples_transformed = torch.bmm(T_M, samples.unsqueeze(2))[:,:,0] + vectors[:,0]
        xyz_new[is_r,ChemData().aa2long[id_ex].index(" O2'")] = samples_transformed        

        # Transform sequence
        seq_new[mask] = torch.tensor([transm_seq_mapper[key.item()] 
                                        for key in seq_new[mask]], device=seq.device)
        
        return seq_new, xyz_new


    @staticmethod
    def transmute_rna_to_dna(seq: torch.Tensor, 
                             xyz: torch.Tensor, 
                             seq_new: torch.Tensor,
                             xyz_new: torch.Tensor, 
                             mask: torch.Tensor,
                             safe: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transmutes RNA to DNA by converting the sequence and updating the coordinates.
        Uses statitical imputation for missing atoms with empirically determined multivaraite normal distributions.
        Logically, set mask to is_rna from the original sequence before calling this function.

        Args:
            seq (torch.Tensor): The original sequence tensor.
            xyz (torch.Tensor): The original coordinate tensor.
            seq_new (torch.Tensor): The new sequence tensor.
            xyz_new (torch.Tensor): The new coordinate tensor.
            mask (torch.Tensor): The mask indicating which elements are RNA and will have the transmute applied to.
            safe (bool): If True, will do extra checking to ensure that the input is valid. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated sequence tensor and coordinate tensor.
        """
        # Generate initial data 
        transm_seq_mapper, transm_xyz_atom_mapper = TransmuteNA.init_data(xyz)

        rna_base_idx = [ChemData().aa2num[b] for b in [' RA', ' RC', ' RG', ' RU', ' DX']]    
        for r in rna_base_idx:
            is_r = torch.logical_and(seq == r, mask)
            U, V_mask = transm_xyz_atom_mapper[r]
            xyz_new[is_r.unsqueeze(1)*V_mask.unsqueeze(0)] = xyz[is_r][:,U].reshape(-1, 3) 
            if r == ChemData().aa2num[' RU']:
                # Add back in the C7 methyl group to Thyamine
                r = ChemData().aa2num[' DT'] # Reset as if it were T and correct                
                # Generate basis vectors                
                vectors = xyz_new[is_r][:,[ChemData().aa2long[r].index(' C5 '),
                                        ChemData().aa2long[r].index(' C4 '),
                                        ChemData().aa2long[r].index(' C6 '),
                                        ChemData().aa2long[r].index(' C7 ')]]
                # Assert inputed position is missing or zeroed
                if safe:
                    assert torch.all(torch.isnan(vectors[:,3])) or torch.all(vectors[:,3] == 0.0), "C7 atom must be missing or zeroed"

                vectors = vectors[:,:3]
                # Do not worry about nans / missing atoms, as this stage should not have any            
                T_M = create_orthonormal_basis(vectors)
                # Sample from normal distribution and transform back to original basis
                dist = MultivariateNormal(torch.tensor(TransmuteNA.T_mean, device=seq.device), torch.tensor(TransmuteNA.T_cov, device=seq.device))
                samples = dist.sample((torch.sum(is_r),))
                samples_transformed = torch.bmm(T_M, samples.unsqueeze(2))[:,:,0] + vectors[:,0]
                xyz_new[is_r,ChemData().aa2long[r].index(' C7 ')] = samples_transformed

        # Transform sequence
        seq_new[mask] = torch.tensor([transm_seq_mapper[key.item()] 
                                        for key in seq_new[mask]], device=seq.device)  
        
        return seq_new, xyz_new

class NA_Motif_Preserving_Tight_Crop:

    def __init__(self, min_na_expand, max_na_expand, min_prot_expand, max_prot_expand, closest_k):
        self.min_na_expand = min_na_expand
        self.max_na_expand = max_na_expand
        self.min_prot_expand = min_prot_expand
        self.max_prot_expand = max_prot_expand
        self.closest_k = closest_k


    def __call__(self, indep: Indep, atom_mask: torch.Tensor, **kwargs) -> dict:
        # first, generate indices for chain 1, chain 2, and protein, in order
        dna_chain_indices, protein_chain_indices, dna_basepairs = self.get_indices_and_basepairs(indep, atom_mask)

        # then call the get contacts
        na_contacts, prot_contacts = self.get_contacts(indep, dna_chain_indices, protein_chain_indices, self.closest_k)

        selection = np.random.randint(len(na_contacts))
        na_contact_ind = na_contacts[selection]
        prot_contact_ind = prot_contacts[selection]

        contact_chain_selection = self._get_selection_safe(dna_chain_indices, na_contact_ind, self.min_na_expand, self.max_na_expand)
        
        basepaired_chain_selection = [dna_basepairs[contact_chain_index] for contact_chain_index in contact_chain_selection if contact_chain_index in dna_basepairs.keys()]

        prot_chain_selection = self._get_selection_safe(protein_chain_indices, prot_contact_ind, self.min_prot_expand, self.max_prot_expand)

        crop = torch.zeros(len(indep.seq)).bool()
        crop[contact_chain_selection + basepaired_chain_selection + prot_chain_selection] = True

        aa_model.pop_mask(indep, crop)
        atom_mask = atom_mask[crop]

        return dict(
            indep=indep,
            atom_mask=atom_mask,
            **kwargs
        )


    def get_contacts(self, indep, dna_chain_indices, protein_chain_indices, closest_k):
        all_dna_indices = np.array([index for chain in dna_chain_indices for index in chain])
        all_prot_indices = np.array([index for chain in protein_chain_indices for index in chain])

        na_coords = indep.xyz[all_dna_indices, 18] # C5 on C, N7 on A, N7 on G, C7 on T - should be fine
        prot_coords = indep.xyz[all_prot_indices, 1] # Ca

        distances = torch.cdist(na_coords, prot_coords)
        distances[distances == 0] = 9999999999 # spoofing

        smallest_indices = torch.topk(distances.view(-1), k=closest_k, largest=False).indices
        smallest_indices = np.unravel_index(smallest_indices.numpy(), distances.shape)

        na_contacts = smallest_indices[0]
        prot_contacts = smallest_indices[1]

        na_contact_indices = all_dna_indices[na_contacts]
        prot_contact_indices = all_prot_indices[prot_contacts]

        return na_contact_indices, prot_contact_indices

    
    def _get_selection_safe(self, chains, contact_index, min_expand, max_expand):
        contact_chain = [chain for chain in chains if contact_index in chain][0]
        contact_position_in_chain = contact_chain.index(contact_index)
        right_expand = min(np.random.randint(min_expand, max_expand), len(contact_chain) - contact_position_in_chain)
        left_expand = min(np.random.randint(min_expand, max_expand), contact_position_in_chain)

        return contact_chain[contact_position_in_chain-left_expand : contact_position_in_chain+right_expand]
        
    
    def get_indices_and_basepairs(self, indep: Indep, atom_mask):
        """
        Returns
            protein indices: list of lists of protein chains, ordering refers to bondedness
            dna_indices: list of lists of dna chains, ordering refers to bondedness
            dna_basepairs: dictionary
        """
        protein_indices = get_resi_type_mask(indep.seq, nuc_type='prot').nonzero().flatten()
        dna_indices = get_resi_type_mask(indep.seq, nuc_type='dna').nonzero().flatten()

        visited_indices = []
        # get all dna basepairs
        basepairs = self.get_basepairs(indep, atom_mask)
        dna_chains = []
        protein_chains = []

        chain_starts = self._find_chain_starts(indep)
        dna_chain_starts = set(chain_starts).intersection({i.item() for i in dna_indices})
        protein_chain_starts = set(chain_starts).intersection({i.item() for i in protein_indices})

        for start_index in dna_chain_starts:
            if start_index in visited_indices:
                continue

            visited_indices.append(start_index)

            current_chain = [start_index]
            bonded = [bond.item() for bond in torch.nonzero(indep.bond_feats[start_index]).flatten() if bond.item() not in visited_indices]

            while len(bonded) == 1:
                current_chain.append(bonded[0])
                visited_indices.append(bonded[0])
                bonded = [bond.item() for bond in torch.nonzero(indep.bond_feats[bonded[0]]).flatten() if bond.item() not in visited_indices]
                assert len(bonded) <= 1, "bad bond definition"

            dna_chains.append(current_chain)

        for start_index in protein_chain_starts:
            if start_index in visited_indices:
                continue

            visited_indices.append(start_index)
            current_chain = [start_index]
            bonded = [bond.item() for bond in torch.nonzero(indep.bond_feats[start_index]).flatten() if bond.item() not in visited_indices]

            while len(bonded) == 1:
                current_chain.append(bonded[0])
                visited_indices.append(bonded[0])
                bonded = [bond.item() for bond in torch.nonzero(indep.bond_feats[bonded[0]]).flatten() if bond.item() not in visited_indices]
                assert len(bonded) <= 1, "bad bond definition"

            protein_chains.append(current_chain)

        return dna_chains, protein_chains, basepairs
    
    def get_basepairs(self, indep, mask, 
                            canonical_partner_filter=True,
                            vert_diff_cutoff=6.7,  # Max distance between bases based on vertical projection
                            centroid_cutoff=6.2, # Max distance between base atom centers of mass
                            bp_cutoff=3.2, # Max distance between repatoms
                            base_angle_cutoff=0.06, # Maximum twist angle between base normal-vectors
                            eps=1e-6, # Numerical stability constant
                            # we are just gonna leave these guys hardcoded for now
                            ):
        
        seq = indep.seq
        xyz = indep.xyz
        seq_neighbors = indep.bond_feats
        len_s = len(indep.seq)
        mask = mask[:, :ChemData().NHEAVY]

        is_protein = torch.logical_and((0 <= seq),(seq <= 21))
        is_dna = torch.logical_and((22 <= seq),(seq <= 26))
        # is_rna = torch.logical_and((27 <= seq),(seq <= 31))

        len_s_na = (~is_protein).sum()

        # Using Frank's method with distance between representative atoms:
        repatom = torch.zeros(len_s, dtype=torch.long, device=xyz.device)

        repatom[seq==22] = ChemData().aa2long[22].index(' N1 ') # DA - N1
        repatom[seq==23] = ChemData().aa2long[23].index(' N3 ') # DC - N3
        repatom[seq==24] = ChemData().aa2long[24].index(' N1 ') # DG - N1
        repatom[seq==25] = ChemData().aa2long[25].index(' N3 ') # DT - N3
        # repatom[seq==27] = ChemData().aa2long[27].index(' N1 ') # A - N1
        # repatom[seq==28] = ChemData().aa2long[28].index(' N3 ') # C - N3
        # repatom[seq==29] = ChemData().aa2long[29].index(' N1 ') # G - N1
        # repatom[seq==30] = ChemData().aa2long[30].index(' N3 ') # U - N3


        xyz_na_rep = torch.gather(xyz, 1, repatom[:,None,None].repeat(1,1,3)).squeeze(1)
        contact_dist = torch.cdist(xyz_na_rep, xyz_na_rep) < bp_cutoff
        cond = torch.logical_and(contact_dist, ~seq_neighbors)

        protein_protein = torch.outer(is_protein.int(), is_protein.int()).bool()
        protein_dna = torch.outer(is_protein.int(), is_dna.int()).bool()
        cond = torch.logical_and(cond, ~protein_protein)
        cond = torch.logical_and(cond, ~protein_dna)
        

        base_atom_xyz = torch.zeros((len_s_na,11,3), dtype=torch.float, device=xyz.device)
        mask_na = mask[~is_protein].unsqueeze(-1).repeat(1,1,3)

        base_xyz_masked = torch.where(mask_na, xyz[~is_protein], torch.nan)

        # Select full range for guanines, since those have the most atoms
        dna_base_atoms_start = ChemData().aa2long[24].index(' N9 ')
        dna_base_atoms_stop = ChemData().aa2long[24].index(' O6 ')


        base_atom_xyz[is_dna[~is_protein],:,:] = base_xyz_masked[is_dna[~is_protein],dna_base_atoms_start:dna_base_atoms_stop+1,:]
        # base_atom_xyz[is_rna[~is_protein],:,:] = base_xyz_masked[is_rna[~is_protein],rna_base_atoms_start:rna_base_atoms_stop+1,:]


        # Compute the centroid of the points
        centroid = torch.nanmean(base_atom_xyz, dim=1, keepdim=True)
        # centroid_contact_dist = torch.cdist(centroid[:,0,:],centroid[:,0,:])
        centroid_in_contact = (torch.cdist(centroid[:,0,:],centroid[:,0,:]) < centroid_cutoff)
        cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], centroid_in_contact)


        # Center the points
        centered_points = base_atom_xyz - centroid
        centered_nan_mask = ~torch.isnan(centered_points)
        centered_zero_nan = torch.where(centered_nan_mask, centered_points, 0.0)

        ###    COMPUTING THE BASE ANGLES    ###
        # Compute the covariance matrix
        covariance_matrix_unscaled = torch.matmul(centered_zero_nan.transpose(-1, -2), centered_zero_nan)
        denom = ( centered_nan_mask.sum(-2) - 1 ).unsqueeze(-1).repeat((1,1,3))
        covariance_matrix = covariance_matrix_unscaled / (denom + eps)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

        # The normal to the plane is the eigenvector associated with the smallest eigenvalue
        base_normals = torch.real(eigenvectors)[torch.arange(eigenvectors.shape[0]), torch.argmin(torch.real(eigenvalues),dim=-1)]
        cosines = torch.clamp(torch.einsum('ni,mi->nm', base_normals, base_normals), -1, 1)
        angle_differences = torch.acos(torch.abs(cosines))
        bases_in_plane = (angle_differences <= base_angle_cutoff)

        # Don't use base-angle logic for protein contacts:
        cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], bases_in_plane)

        ###    COMPUTING THE PLANE DISTANCE   ###
        r_ij_mat = centroid - centroid.transpose(0, 1)
        d_ij_on_norm_i = torch.norm(torch.sum(r_ij_mat * base_normals.unsqueeze(1), dim=-1).unsqueeze(-1) * base_normals.unsqueeze(1) , dim=-1)

        base_close_vert_dist = (d_ij_on_norm_i <= vert_diff_cutoff)
        cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], base_close_vert_dist)
        
        cond = torch.logical_or(cond, cond.t())
        
        # Final filter: check for canonical base pairing.
        # probably bad for RNA, but might help DNA folks during training or someshit.
        if canonical_partner_filter:

            # initialize as all false, then add in true where possible.
            bp_partners_canon = torch.zeros((len_s, len_s), dtype=torch.bool, device=xyz.device)

            # Define the conditions as boolean masks
            cond_AA = (seq[:, None] == 22) | (seq[:, None] == 27)
            cond_TU = (seq[:, None] == 25) | (seq[:, None] == 30)
            cond_CC = (seq[:, None] == 23) | (seq[:, None] == 28)
            cond_GG = (seq[:, None] == 24) | (seq[:, None] == 29)

            # Update the matrix based on the conditions
            bp_partners_canon[cond_AA & cond_TU.T] = True
            bp_partners_canon[cond_TU & cond_AA.T] = True
            bp_partners_canon[cond_CC & cond_GG.T] = True
            bp_partners_canon[cond_GG & cond_CC.T] = True
            # This should be symmetric by definion. 
            
            # update conditions of base pairing to satisfy all conditions
            cond = torch.logical_and(cond, bp_partners_canon)
        
        basepair_inds = torch.nonzero(cond)
        basepair_dict = {pair[0].item(): pair[1].item() for pair in basepair_inds}

        return basepair_dict


    def _find_chain_starts(self, indep: Indep):
        chain_starts = [
            i
            for i, row in enumerate(indep.bond_feats)
            if torch.count_nonzero(row) <= 1
        ]
        return chain_starts
