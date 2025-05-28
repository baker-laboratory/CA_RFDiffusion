from __future__ import annotations  # Fake import for type hinting, must be at beginning of file

import types
import pickle
# Imports for typing only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rf_diffusion.aa_model import Indep

import numpy as np
import torch
import logging
from icecream import ic
import copy
import rf_diffusion.util
from rf_diffusion import aa_model
from rf_diffusion.aa_model import Indep
from rf_diffusion.contigs import ContigMap

from rf2aa.chemical import ChemicalData as ChemData

from typing import Union
from rf_diffusion import nucleic_compatibility_utils as nucl_utils

from rf_diffusion import run_inference
from rf_diffusion import mask_generator
from rf_diffusion import ppi
from rf_diffusion.conditions.util import expand_1d_atomized_ok_gp_not, pop_conditions_dict
######## Group all imported transforms together ###########

from rf_diffusion.conditions.ss_adj.sec_struct_adjacency import (LoadTargetSSADJTransform, AutogenerateTargetSSADJTransform, GenerateSSADJTrainingTransform,  # noqa: F401
                                    SSSprinkleTransform, ADJSprinkleTransform)
from rf_diffusion.ppi import (PPITrimTailsChain0ComplexTransform, PPIRejectUnfoldedInterfacesTransform, PPIJoeNateDatasetRadialCropTransform,  # noqa: F401
                                    FindHotspotsTrainingTransform, HotspotAntihotspotResInferenceTransform, ExposedTerminusTransform, RenumberCroppedInput)
from rf_diffusion.conditions.ideal_ss import AddIdealSSTrainingTransform, AddIdealSSInferenceTransform  # noqa: F401

###########################################################


logger = logging.getLogger(__name__)

def get_center_of_mass(xyz14, mask):
    assert mask.any(), f'{mask=}'
    points = xyz14[mask]
    return points.mean(dim=0)

LEGACY_TRANSFORMS_TO_IGNORE = ['PopMask']


def spy(*args):
    return None

class PopMask:
    def __call__(self, indep: Indep, metadata: dict, masks_1d: torch.Tensor, conditions_dict: dict, **kwargs):
        
        aa_model.pop_mask(indep, masks_1d['pop'])
        masks_1d['input_str_mask'] = masks_1d['input_str_mask'][masks_1d['pop']]
        masks_1d['input_seq_mask'] = masks_1d['input_seq_mask'][masks_1d['pop']]
        masks_1d['is_atom_motif'] = aa_model.reindex_dict(masks_1d['is_atom_motif'], masks_1d['pop'])
        masks_1d['can_be_gp'] = masks_1d['can_be_gp'][masks_1d['pop']]
        metadata['covale_bonds'] = None
        metadata['ligand_names'] = np.array(['LIG']*indep.length(),  dtype='<U3')
        pop_conditions_dict(conditions_dict, masks_1d['pop'])
        # is_atom_str_shown = masks_1d['is_atom_motif']
        # Pop atom_mask if exists
        if 'atom_mask' in kwargs:
            kwargs['atom_mask'] = kwargs['atom_mask'][masks_1d['pop']]

        return dict(
            indep=indep,
            metadata=metadata,
            masks_1d=masks_1d,
            conditions_dict=conditions_dict,
            **kwargs
        )


class NullTransform:
    """
    A null transform for testing purposes
    """
    def __call__(self, **kwargs):
        return dict(**kwargs)
    

class OmitFramePermutations: 
    """
    Set indep instance property to use omit_frame_permutations=True.
    """
    def __call__(self, indep: Indep, **kwargs):
        indep.frames_omit_permutation = True
        return dict(
            indep=indep,
            **kwargs
        )


def label_true_islands(tensor) -> torch.Tensor:
    """
    Written by ChatGPT. 

    Incrementally labels True islands in a boolean tensor.

    Parameters:
        tensor (torch.Tensor): A 1D boolean tensor.
    """
    labels = torch.full(tensor.shape, -1)  # Initialize all positions to -1
    current_label = 0  # Initialize the label for islands of True values

    # Iterate through the tensor and label islands
    for i in range(len(tensor)):
        if tensor[i]:  # If current position is True
            if i == 0 or not tensor[i - 1]:  # Start a new island if the previous was False
                current_label += 1  # Increment the label for each new island
            labels[i] = current_label - 1  # Assign the label

    return labels


def compute_token_pair_mask(is_motif_w_chunk_id, coarse_ij_matrix):
    """
    Compute 2D mask for motif chunk pairs. 

    Parameters: 
        is_motif_w_chunk_id (torch.Tensor): 1D tensor desginating motif chunks, with 
            each chunk labeled by an integer. 
        coarse_ij_matrix (torch.Tensor): 2D tensor designating which motif chunks can see each other.

    Returns: 
        torch.Tensor: 2D tensor designating which pairs of tokens will be constrained w.r.t each other.
    """
    # cartesian product of pairs of motif chunks constrained w.r.t each other. 
    visible_pairs = torch.nonzero(coarse_ij_matrix)
    visible_pairs = [tuple(p.numpy()) for p in visible_pairs]

    L = is_motif_w_chunk_id.shape[0]
    A = is_motif_w_chunk_id[None].repeat(L,1)   # L x L, each row is a copy of is_motif_w_chunk
    B = A.T                                     # L x L, ^^ but transposed

    stacks = []
    for (i,j) in visible_pairs: 
        is_i = A == i
        is_j = B == j

        is_i_and_j = is_i & is_j  # find tokens that are part of group i and j intersection
        stacks.append(is_i_and_j)

    stacks = torch.stack(stacks+[t.T for t in stacks]) 
    return torch.any(stacks, dim=0) # combine them all

def get_coarse_ij_matrix(ij_visible: str): 
    """
    Computes the coarse matrix of motif chunk pairs that can see each other.
    1 indicating chunks are constrained w.r.t each other, 0 otherwise.
    """

    abet = 'abcdefghijklmnopqrstuvwxyz'
    abet = abet + abet.upper()
    abet = [a for a in abet]
    abet2num = {a:i for i,a in enumerate(abet)}

    # compute 1D array designating which tokens correspond to each motif chunk in contig map
    motif_visibility_groupings = ij_visible.split('-')
    motif_visibility_groupings_int = [tuple([abet2num[a] for a in s]) for s in motif_visibility_groupings]
    
    # total motif chunks, supplied by the user, including ligand
    nchunk_total = len( set( [i for S in motif_visibility_groupings_int for i in S] ) )

    # Make a 2D array designating which motif chunks can see each other 
    coarse_ij_visibility = torch.eye(nchunk_total)
    for S in motif_visibility_groupings_int:
        for i in S:
            for j in S:
                coarse_ij_visibility[i,j] = 1

    return coarse_ij_visibility



class GetTemplatedMotifMasks:
    """
    Given contig map and indep, compute the 2D and 1D masks used for motif template addition. 
    """

    def __init__(self, ij_visible: str=None, template_ligand: bool=True): 
        self.ij_visible = ij_visible
        self.template_ligand = template_ligand 

    def __call__(self, 
                 indep: Indep, 
                 contig_map: ContigMap, 
                 masks_1d: dict, 
                 conditions_dict: dict, 
                 **kwargs):

        if self.ij_visible is None:
            # we must be doing refinement and need to compute ij_visible from loaded .trb
            assert conditions_dict.get('ref_dict', False), 'Did not recieve refinement dictionary in conditions_dict. Did you set the inference.ca_rfd_refine flag?'
            ref_dict = conditions_dict['ref_dict']
            self.ij_visible = ref_dict['ij_visible']
            assert (isinstance(self.ij_visible, str)) and (len(self.ij_visible) > 0), f'Got invalid ij_visible from .trb: {self.ij_visible}'
        else:
            # it was passed in from config 
            pass
        

        coarse_ij_visibility = get_coarse_ij_matrix(self.ij_visible)
        nchunk_total = coarse_ij_visibility.shape[0]

        # 1D feature designating motif tokens 
        is_motif = torch.zeros(indep.length(), dtype=torch.bool)
        for i,entry in enumerate(contig_map.inpaint): 
            if entry != ('_','_'): 
                is_motif[i] = True
        is_motif[indep.is_sm] = True # assumes the ligand is motif as well

        is_motif_w_chunk_id = label_true_islands(is_motif)
        # make some assertion about how the computed number of motif chunks from 
        # contig map + indep needs to be the same number computed from user supplied ij_visible
        assert nchunk_total == len(set(is_motif_w_chunk_id[is_motif_w_chunk_id != -1].tolist())), f'{nchunk_total=} {is_motif_w_chunk_id=}'

        # 2D feature designating which pairs of tokens will be constrained w.r.t each other  
        token_pair_is_revealed = compute_token_pair_mask(is_motif_w_chunk_id, coarse_ij_visibility)

        masks_1d['input_str_mask'] = is_motif
        masks_1d['is_motif_2d'] = token_pair_is_revealed

        return {'indep'             : indep, 
                'masks_1d'          : masks_1d,
                'conditions_dict'   : conditions_dict, 
                'contig_map'        : contig_map, 
                **kwargs}

class ComputeMotifTemplate: 
    """
    Takens in an unnoised indep, 1D is_motif mask and 2D is_motif_2d mask and 
    computes a new template containing motif structure.
    """
    def __init__(self, use_cb: bool, omit_frame_permutation: bool):
        self.use_cb = use_cb
        self.omit_frame_permutation = omit_frame_permutation

    def __call__(self, 
                 indep           : Indep,
                 masks_1d        : dict,
                 conditions_dict : dict,
                 **kwargs): 
        
        """
        Args: 
            indep: The unnoised indep object
            masks_1d: The masks containing 1d and 2d motif info
            use_cb: Whether to use cb atoms for t2d calculation
        """
        spy({'indep_into_ComputeMotifTemplate':copy.deepcopy(indep),
             'is_motif':masks_1d['input_str_mask'],
             'is_motif_2d':masks_1d['is_motif_2d']
             })

        is_motif    = masks_1d['input_str_mask'] 
        is_motif_2d = masks_1d['is_motif_2d']

        motif_xyz_t = indep.xyz.clone() # perfect native structure 
        motif_xyz_t[~is_motif] = 0.0    # remove non-motif information 

        ### t2d ### 
        t2d_motif, _ = rf_diffusion.util.get_t2d(motif_xyz_t[None], 
                                                 indep.is_sm, 
                                                 indep.get_atom_frames(self.omit_frame_permutation),  
                                                 use_cb    = self.use_cb,
                                                 mask_t_2d = is_motif_2d[None])

        ### xyz_t ###
        xyz_t_out = motif_xyz_t[:,1] # CA's of motif residues 

        conditions_dict['template_t2d'] = t2d_motif
        conditions_dict['template_xyz'] = xyz_t_out

        masks_1d['input_str_mask'] = torch.zeros_like(is_motif) # ensure it gets diffused in 3D 
        masks_1d['is_templated_motif'] = is_motif # use this later inside of preprocess.add_motif_template

        return {'indep'             : indep,
                'masks_1d'          : masks_1d,
                'conditions_dict'   : conditions_dict,
                **kwargs}


class ComputeMotifTemplateRefine: 
    """
    Computes the motif template during refinement. The reason we must have a separate 
    transform from ComputeMotifTemplate is because we don't have access to an indep 
    which contains the perfect motif information at refinement time. So, we use the 
    perfect motif information collected in metadata, along with the unnoised indep 
    to compute the motif template. 
    """
    def __init__(self, use_cb: bool, omit_frame_permutation: bool):
        self.use_cb = use_cb
        self.omit_frame_permutation = omit_frame_permutation

    def __call__(self, 
                 indep: Indep, 
                 conditions_dict: dict,
                 masks_1d: dict,
                 **kwargs): 
        """
        Parameters: 
            indep (aa_model.Indep): Should be unnoised version of backbone,
                potentially with ligand, to be noised and refined.
            
            conditions_dict (dict): For storing/accessing the 
                original contig info and original motif structure.
        """
        ref_dict = conditions_dict['ref_dict']
        motif_indep  = ref_dict['motif_indep']
        ij_visible   = ref_dict['ij_visible']
        con_hal_idx0 = torch.from_numpy(ref_dict['con_hal_idx0'])
        con_ref_idx0 = torch.from_numpy(ref_dict['con_ref_idx0'])

        ### Compute 1D/2D mask information ### 
        ######################################

        # hal structure (design from diffusion)
        L_sm = indep.is_sm.sum()
        L_hal_w_ligand = indep.length()
        add_con_hal0 = torch.arange(L_hal_w_ligand-L_sm, L_hal_w_ligand)
        con_hal_idx0 = torch.cat((con_hal_idx0, add_con_hal0))
        # reference structure (original motif pdb)
        L_ref_w_ligand = motif_indep.length()
        add_con_ref0 = torch.arange(L_ref_w_ligand-L_sm, L_ref_w_ligand)
        con_ref_idx0 = torch.cat((con_ref_idx0, add_con_ref0))


        # Get constraints between motif chunks - coarse
        coarse_ij_visibility = get_coarse_ij_matrix(ij_visible)

        is_motif = torch.zeros(L_hal_w_ligand, dtype=torch.bool)
        is_motif[con_hal_idx0] = True # contains protein motif + ligand motif
        
        is_motif_w_chunk_id = label_true_islands(is_motif)
        token_pair_is_revealed = compute_token_pair_mask(is_motif_w_chunk_id, coarse_ij_visibility)

        masks_1d['input_str_mask'] = is_motif
        masks_1d['is_motif_2d'] = token_pair_is_revealed

        ### Compute motif template ###
        ##############################

        # Need a new set of crds with same size as indep.xyz, 
        # but with original motif crds sliced in.
        hal_motif_template_xyz = torch.zeros_like(indep.xyz)
        hal_motif_template_xyz[con_hal_idx0] = motif_indep.xyz[con_ref_idx0] # <fire emoji>

        ### t2d ###
        t2d_motif, _ = rf_diffusion.util.get_t2d(hal_motif_template_xyz[None],
                                                 indep.is_sm, 
                                                 indep.get_atom_frames(self.omit_frame_permutation),
                                                 use_cb     = self.use_cb,
                                                 mask_t_2d  = token_pair_is_revealed[None])

        xyz_t_out = hal_motif_template_xyz[:,1] # CA's of motif residues

        conditions_dict['template_t2d'] = t2d_motif
        conditions_dict['template_xyz'] = xyz_t_out

        masks_1d['input_str_mask'] = torch.zeros_like(is_motif) # ensure it gets diffused in 3D
        masks_1d['is_templated_motif'] = is_motif

        return {'indep'           : indep,
                'masks_1d'        : masks_1d,
                'conditions_dict' : conditions_dict,
                **kwargs}

class GenerateMasks:
    """
    Non-public class for generating masks. Is not a traditional transform
    because it uses configurations from many places in the configuration file
    and also relies on hard coded functions and their names within mask_generator.py

    When using, note that the params come from dataloader params in the config.

    The reason this function is a transform and not hard-coded is because some transforms might want to be applied before masks are generated - for example more specific cropping. 
    """
    def __init__(self, datahub_config=None):
        self.datahub_config = datahub_config

    def __call__(self, 
                 indep: Indep, 
                 task: str,
                 params,
                 chosen_dataset: str, 
                 atom_mask: torch.Tensor, 
                 metadata: dict, 
                 mask_gen_seed: int, 
                 **kwargs):
        # Asserts for deprecated arguments
        assert task == "diff", "All tasks except 'diff' are deprecated and not supported anymore."
        assert chosen_dataset not in ["complex", "negative"], "The chosen dataset is not directly supported anymore. (they may work)"
        assert "full_chain" not in kwargs, "The full_chain argument is not supported anymore."

        # Mask the independent inputs.
        run_inference.seed_all(mask_gen_seed) # Reseed the RNGs for test stability.
        masks_1d = mask_generator.generate_masks(
            indep=indep, 
            task=task, 
            loader_params=params, 
            chosen_dataset=chosen_dataset, 
            full_chain=None, 
            atom_mask=atom_mask[:, :ChemData().NHEAVY], 
            metadata=metadata,  # holds e.g. `covale_bonds`
            datahub_config=self.datahub_config
        )
        # Pop masks_1d if exists since it is being overwritten
        if 'masks_1d' in kwargs: 
            logger.warning('`masks_1d` is being overwritten by GenerateMasks. Was: %s', kwargs['masks_1d'])
            kwargs.pop('masks_1d')

        return dict(
            indep=indep,
            task=task,
            params=params,
            chosen_dataset=chosen_dataset,
            atom_mask=atom_mask,
            metadata=metadata,
            masks_1d=masks_1d,
            mask_gen_seed=mask_gen_seed,
            **kwargs
        )


class NAInterfaceTightCrop:
    """
    A class for performing tight cropping of nucleic acid interfaces. This class handles data passes 
    from other dataets that may not contain relevent nucleic acid interfaces.

    Args:
        contact_type (str): The type of contact to consider. Default is 'protein_na'.
        closest_k (int): The number of closest contacts to consider. Default is 2.
        distance_ball_around_contact_angstroms (int): The distance around each contact to consider for cropping. Default is 80.
        chain_search_angstroms (int): The distance to search for chains around each contact. Default is 10.
        max_gap_to_add (int): The maximum gap size to add when cropping. Default is 5.
        contact_offcenter_var (int): The variance of the offcenter of the contact. Default is 3.
        min_island_size (int): The minimum size of the island to consider. Default is 20.
        min_island_size_na (int): The minimum size of the island to consider for nucleic acids. Default is 4.
        min_indep_len (int): The minimum length of the independent input. Default is 10.
        max_size (int): The maximum size of the independent input. Default is 256.        
    """

    def __init__(self,
                 contact_type: str = 'protein_na',
                 closest_k: int = 2,
                 distance_ball_around_contact_angstroms_min_prot: float = 30,
                 distance_ball_around_contact_angstroms_max_prot: float = 85,
                 distance_ball_around_contact_angstroms_min_na: float=20,
                 distance_ball_around_contact_angstroms_max_na: float=50,
                 chain_search_angstroms: float = 10.,
                 max_gap_to_add: int = 5,
                 contact_offcenter_var: float = 3.,
                 min_island_size: int = 20,
                 min_island_size_na: int = 4,
                 max_size: int = 256,
                 ) -> None:        
        # Initialize cropper
        # distance_ball_around_contact_angstroms = dist_min
        # distance_ball_around_contact_angstroms_na = dist_min_na
        self.cropper = nucl_utils.NucleicAcid_Interface_Preserving_Crop(contact_type=contact_type,
                                                                        closest_k=closest_k,
                                                                        distance_ball_around_contact_angstroms_min_prot=distance_ball_around_contact_angstroms_min_prot,
                                                                        distance_ball_around_contact_angstroms_max_prot=distance_ball_around_contact_angstroms_max_prot,
                                                                        distance_ball_around_contact_angstroms_min_na=distance_ball_around_contact_angstroms_min_na,
                                                                        distance_ball_around_contact_angstroms_max_na=distance_ball_around_contact_angstroms_max_na,
                                                                        chain_search_angstroms=chain_search_angstroms,
                                                                        max_gap_to_add=max_gap_to_add,
                                                                        contact_offcenter_var=contact_offcenter_var,
                                                                        min_island_size=min_island_size,
                                                                        min_island_size_na=min_island_size_na,
                                                                        max_size=max_size,
                                                                        )

    def __call__(self, indep: Indep, atom_mask, **kwargs):
        out = self.cropper(indep, atom_mask)
        indep = out['indep']

        return dict(
            indep=indep,
            atom_mask=atom_mask,
            **kwargs
        )


class UnmaskAndFreezeNA:
    """
    A class that conditionally converts nucleic acid residues to masked residues in the input sequence
    for diffusing chains. Note when randomly applied, this will override masks when used

    Args:
        p_unmask_na (float): The probability of unmasking nucleic acid residue sequence when diffusing
        p_freeze_na (float): The probability of freezing the nucleic acid residue
        p_unmask_prot (float): The probability of unmasking protein residue sequence when diffusing
        p_freeze_prot (float): The probability of freezing the protein residue

    Returns:
        dict: A dictionary containing the updated inputs, metadata, masks, and other keyword arguments.
    """
    def __init__(self, p_unmask_na: float = 0.0, 
                 p_freeze_na: float = 0.0,
                 p_unmask_prot: float = 0.0, 
                 p_freeze_prot: float = 0.0):
        self.p_unmask_na = p_unmask_na
        self.freeze_na = p_freeze_na
        self.p_unmask_prot = p_unmask_prot
        self.p_freeze_prot = p_freeze_prot

    def __call__(self, indep: Indep, masks_1d: torch.Tensor, **kwargs):    
        is_res_str_shown = masks_1d['input_str_mask']  # True when the sequence is shown
        is_atom_str_shown = masks_1d['is_atom_motif']
        is_atom_mask = torch.zeros_like(indep.seq, dtype=torch.bool, device=indep.seq.device)
        for k in is_atom_str_shown:
            is_atom_mask[k] = True
        is_diffused = ~is_res_str_shown * ~is_atom_mask

        chain_masks = indep.chain_masks()
        chain_masks_rand = [chain_masks[i] for i in np.random.permutation(len(indep.chain_masks()))]

        diffusing_chains = [False] * len(chain_masks_rand)
        for ch_id,chain_mask in enumerate(chain_masks_rand):
            if torch.any(is_diffused[chain_mask]):
                diffusing_chains[ch_id] = True
        
        assert len(diffusing_chains) > 0, "No diffusing chains found."

        for ch_id,chain_mask in enumerate(chain_masks_rand):
            is_na_chain = torch.any(nucl_utils.get_resi_type_mask(indep.seq[chain_mask], 'na'))
            is_prot_chain = torch.any(nucl_utils.get_resi_type_mask(indep.seq[chain_mask], 'prot'))

            # Iterate but do not allow there to be fewer than 1 diffusing chain
            if sum(diffusing_chains) <= 1:
                continue

            # Previous mask may indicate motif is present, in which case it should be frozen 
            # When is_res_str_shown is True, this should stay the case and is_diffused should be False 
            
            if is_na_chain and (np.random.rand() < self.freeze_na):
                # In this case, freeze (do not diffuse and show as motif, does not affect other motifs)
                is_diffused[chain_mask] = False
                # Atomized residues cannot be shown 
                is_res_str_shown[chain_mask] = torch.ones(sum(chain_mask), dtype=bool) * ~is_atom_mask[chain_mask]
                diffusing_chains[ch_id] = False
            elif is_na_chain and (np.random.rand() < self.p_unmask_na):
                # In this case, diffuse, but show the sequence. Do not difuse motif residues or atomized residues                           
                is_diffused[chain_mask] = torch.ones(sum(chain_mask), dtype=bool) * ~is_res_str_shown[chain_mask] * ~is_atom_mask[chain_mask]
                # Atomized residues cannot be shown 
                is_res_str_shown[chain_mask] = torch.ones(sum(chain_mask), dtype=bool) * ~is_atom_mask[chain_mask]
            elif is_prot_chain and (np.random.rand() < self.p_freeze_prot):
                # In this case, freeze (do not diffuse and show as motif, does not affect other motifs)
                is_diffused[chain_mask] = False
                # Atomized residues cannot be shown 
                is_res_str_shown[chain_mask] = torch.ones(sum(chain_mask), dtype=bool) * ~is_atom_mask[chain_mask]
                diffusing_chains[ch_id] = False
            elif is_prot_chain and (np.random.rand() < self.p_unmask_prot):
                # In this case, diffuse, but show the sequence. Do not diffuse motif residues or atomized residues           
                is_diffused[chain_mask] = torch.ones(sum(chain_mask), dtype=bool) * ~is_res_str_shown[chain_mask] * ~is_atom_mask[chain_mask]
                # Atomized residues cannot be shown 
                is_res_str_shown[chain_mask] = torch.ones(sum(chain_mask), dtype=bool) * ~is_atom_mask[chain_mask]
            else:
                # Do nothing, use default of token being diffused and not shown (or being motif)
                pass

        assert len(diffusing_chains) > 0, "No diffusing chains found."

        return dict(
            indep=indep,
            masks_1d=masks_1d,
            is_diffused_seq_shown=is_diffused,
            **kwargs
        )


class TransmuteNA:
    """
    A class that probabilistically converts DNA to RNA and vice versa

    Args:
        p_mask (float): The probability of freezing nucleic acid residues.

    Returns:
        dict: A dictionary containing the updated inputs, metadata, masks, and other keyword arguments.
    """
    def __init__(self, p_dna_to_rna: float = 0.0, p_rna_to_dna: float = 0.0):
        self.p_dna_to_rna = p_dna_to_rna
        self.p_rna_to_dna = p_rna_to_dna

    def __call__(self, indep: Indep, **kwargs):       
        indep = indep.clone()

        # Calculate guidance information about indep
        is_dna = nucl_utils.get_resi_type_mask(indep.seq, 'dna')
        is_rna = nucl_utils.get_resi_type_mask(indep.seq, 'rna')
        is_other = ~torch.logical_or(is_dna, is_rna)

        # Create new sequences
        xyz_new = torch.zeros_like(indep.xyz)
        seq_new = copy.deepcopy(indep.seq)

        chain_masks = indep.chain_masks()

        for chain_mask in chain_masks:
            chain_mask = torch.tensor(chain_mask, device=indep.xyz.device)
            has_dna_chain = torch.any(is_dna[chain_mask])
            has_rna_chain = torch.any(is_rna[chain_mask])
            chain_mask_dna = torch.logical_and(chain_mask, is_dna)
            chain_mask_rna = torch.logical_and(chain_mask, is_rna)
            chain_mask_other = torch.logical_and(chain_mask, is_other)

            # Apply transmute operations probabilistically
            if has_dna_chain and (np.random.rand() < self.p_dna_to_rna):
                seq_new, xyz_new = nucl_utils.TransmuteNA.transmute_dna_to_rna(indep.seq, indep.xyz, seq_new, xyz_new, chain_mask_dna)
            else:
                xyz_new[chain_mask_dna] = indep.xyz[chain_mask_dna]

            if has_rna_chain and (np.random.rand() < self.p_rna_to_dna):
                seq_new, xyz_new = nucl_utils.TransmuteNA.transmute_rna_to_dna(indep.seq, indep.xyz, seq_new, xyz_new, chain_mask_rna)
            else:
                xyz_new[chain_mask_rna] = indep.xyz[chain_mask_rna]
            xyz_new[chain_mask_other] = indep.xyz[chain_mask_other]

        indep.xyz = xyz_new
        indep.seq = seq_new

        return dict(
            indep=indep,
            **kwargs
        )


class RejectBadComplexes:
    """
    Use before AddConditionalInputs to reject complexes with inter chain distances that are too far apart
    
    TODO: Implement sc vs all

    Args:
        max_inter_chain_dist (float): The maximum allowed closest inter chain distance. Defaults to 6.0 angstroms.
        n_interacting_residues (int): The number of interfacing residues to require being below distance threshold. Defaults to 5.
        mode (str): Currently only one mode, all_vs_all, which looks for interactions between all atom types. Other modes could be side chain only vs all etc
        debug (bool): Whether to print debug information. Defaults to True.
    """
    def __init__(self, max_inter_chain_dist: float = 5.0,
                 n_interacting_residues: int = 10,
                 mode: str = 'all_vs_all',
                 debug: bool = True):
        self.max_inter_chain_dist = max_inter_chain_dist
        self.n_interacting_residues = n_interacting_residues
        assert mode == 'all_vs_all'
        self.debug = debug
        self.count = 0

    def __call__(self, indep: Indep, **kwargs):
        # Empty tensor of interchain distances
        n_consider = min([self.n_interacting_residues, indep.xyz.shape[0]])
        D = 9999*torch.ones(indep.xyz.shape[0], dtype=indep.xyz.dtype, device=indep.xyz.device)            

        # By default use the first atom in the sequence as the atom to check for interface contacts
        default_atom_list = [None, 'A', None] + [None] * (ChemData().NTOTAL - 3)

        # Get tensor mask of valid atoms
        # NOTE: the use of this code and similar code in calc_loss could be condensed into a single function,
        # though are calculated differently for numerical stability purposes since calc_loss is used for gradients
        is_valid = torch.zeros(indep.xyz.shape[:2], dtype=torch.bool, device=indep.xyz.device)
        for i in range(indep.seq.shape[0]):   
            r = indep.seq[i]             
            is_valid[i] = torch.tensor([atom is not None and atom.find('H') == -1 
                                        for atom in (ChemData().aa2long[r] 
                                                    if r < len(ChemData().aa2long) 
                                                    else default_atom_list)
                                        ][:indep.xyz.shape[1]], 
                                        dtype=torch.bool)

        # Iterate through all chain pairs and calculate inter chain distances for kernel weights
        n_chains = len(np.unique(indep.chains()))
        if n_chains > 1:
            for i,i_chain_mask in enumerate(indep.chain_masks()):
                i_chain_mask = torch.tensor(i_chain_mask, device=indep.xyz.device)
                for j,j_chain_mask in enumerate(indep.chain_masks()):
                    j_chain_mask = torch.tensor(j_chain_mask, device=indep.xyz.device)
                    if i == j:
                        continue
                    xyz_i = indep.xyz[i_chain_mask]
                    xyz_j = indep.xyz[j_chain_mask]

                    is_valid_i = is_valid[i_chain_mask]
                    is_valid_j = is_valid[j_chain_mask]
                    Li, N_atoms = xyz_i.shape[0], xyz_i.shape[1]
                    Lj, N_atoms = xyz_j.shape[0], xyz_j.shape[1]
                    cdist = torch.cdist(xyz_i.view(-1, 3), xyz_j.view(-1, 3)).view(Li, N_atoms, Lj, N_atoms)
                    cdist = torch.nan_to_num(cdist, 9999)

                    # Get the mask of invalid positions and make invalid pair have a large distance
                    pair_mask_valid = is_valid_i[:,:,None,None] * is_valid_j[None,None,:,:]
                    cdist = cdist*pair_mask_valid + (~pair_mask_valid) * 9999
                    # Calculate the minimum values for each nucleic acid base position
                    cdist_min_i = cdist.min(3)[0].min(2)[0].min(1)[0]  # [Li,]
                    cdist_min_j = cdist.min(3)[0].min(1)[0].min(0)[0]  # [Lj,]
                    D[i_chain_mask] = torch.minimum(D[i_chain_mask], cdist_min_i)
                    D[j_chain_mask] = torch.minimum(D[j_chain_mask], cdist_min_j)

            if torch.sum(D <= self.max_inter_chain_dist).item() < n_consider:
                ic('failed!')
                if self.debug:
                    indep.write_pdb(f'bad_complex_{self.count}.pdb')
                self.count += 1
                assert False, f'Rejection using RejectBadComplexes: {torch.topk(D, n_consider, largest=False, sorted=True)[0][-1].item()} > {self.max_inter_chain_dist}'

        return dict(
            indep=indep,
            **kwargs
        )
    

class RejectOutOfMemoryHazards:
    """
    A class that rejects instances with a length exceeding a maximum size to avoid out-of-memory hazards.

    Args:
        max_size (int): The maximum size allowed for an instance. Defaults to 396.
        debug (bool): Whether to enable debug mode. Defaults to True.
    """

    def __init__(self, max_size: int = 364, debug: bool = True):
        self.max_size = max_size
        self.debug = debug
        self.count = 0

    def __call__(self, indep: Indep, **kwargs):
        if indep.length() > self.max_size:
            if self.debug:
                indep.write_pdb(f'oom_candidate_{self.count}.pdb')
                with open(f'oom_candidate_{self.count}.pkl', 'wb') as f:
                    pickle.dump((indep, kwargs), f)
            self.count += 1
            assert False, f'Rejection using RejectOutOfMemoryHazards: {indep.length()} > {self.max_size}'

        return dict(
            indep=indep,
            **kwargs
        )
    
    
class CenterOnCA():
    """
    Centers on CAs + ligand atoms, irrespective of the motif.
    """
    def __call__(self, indep: Indep, **kwargs):
        indep.xyz -= indep.xyz[:,1].mean(dim=0)
        return {'indep': indep, **kwargs}


class Center:
    def __call__(self, indep: Indep, masks_1d: torch.Tensor, **kwargs):

        is_res_str_shown = masks_1d['input_str_mask']
        is_atom_str_shown = masks_1d['is_atom_motif']

        # For debugging
        is_sm_shown = indep.is_sm[is_res_str_shown.nonzero()[:, 0]]
        n_atomic_motif = is_sm_shown.sum()
        n_residue_motif = (~is_sm_shown).sum()
        logging.debug(
            f'{n_atomic_motif=} {n_residue_motif=} {is_atom_str_shown=} {is_res_str_shown.nonzero()[:, 0]=}', 
        )

        motif_atom_name_by_res_idx = {
            i: aa_model.CA_ONLY for i in is_res_str_shown.nonzero()[:, 0]
        }
        motif_atom_name_by_res_idx |= is_atom_str_shown
        is_motif14 = aa_model.make_is_motif14(indep.seq, motif_atom_name_by_res_idx)
        center_of_mass_mask = is_motif14
        if not center_of_mass_mask.any():
            # Unconditional case
            center_of_mass_mask[:, 1] = True

        indep.xyz -= get_center_of_mass(indep.xyz[:,:ChemData().NHEAVY], center_of_mass_mask)
        return dict(
            indep=indep,
            masks_1d=masks_1d,
            **kwargs
        )


class CenterPostTransform:
    """
    A class recentering around the diffused frames. Allows jittering of the center to prevent overfitting
    and memorization of exact placement of ligands / fixed motifs relative to center of mass.
    Must be used after AddConditionalInputs
    Attributes:
        jitter (float): The expected average distance between the center point and the origin
        jitter_clip (float): The maximum amount of distance between the center point and origin. Set this depending on
            the size of jitter, possibly 3 * jitter
        center_type (str): The type of centering to apply. Options are 'is_diffused' and 'is_not_diffused'    
    """    
    def __init__(self, 
                 jitter: float = 0.0,
                 jitter_clip: float = 50.0,
                 center_type: str = 'is_diffused'):
        """
            Centering around diffused atoms for traning stability and design control during inference.
        could solve the problem of the proteins drifting off and ligands being centered. Need to pair with
        extra flags and new parser at inference to specify diffusion origin. This code reduces the requirement
        for the model to learn large center of mass translations. However, it is more prone to memorization 
        of the training data if there are not many examples since data leak occurs under this training 
        regime. This is because the model can memorize the exact placement of the ligands and fixed motifs

        Center_types:
            is_diffused: Center on CoM of is_diffused
            is_not_diffused: Center on CoM of ~is_diffused
            all: Center on CoM of indep
            target_hotspot: PPI inference side of 'all'. Binder CoM is taken to be CoM of the hotspot CBs

        Args:
            jitter (float): The expected average distance between the center point and the origin
            jitter_clip (float): The maximum amount of distance between the center point and origin. Set this depending on
                the size of jitter, possibly 3 * jitter
            center_type (str): The mode of centering
        """

        self.jitter = jitter
        self.jitter_clip = jitter_clip
        self.center_type = center_type
        assert center_type in ['is_diffused', 'is_not_diffused', 'all', 'target_hotspot', 'hotspot', 'normal_to_target_hotspot'], "must use 'is_diffused' or 'is_not_diffused' for center_type"

    def __call__(self,
                     indep: Indep, 
                     is_diffused: torch.Tensor,
                     conditions_dict: dict,
                     origin: torch.Tensor = None,
                     **kwargs) -> dict:
        """
        Computes centering for the indep. Must happen post transform_indep

        Args:
            indep (Indep): the holy Indep
            is_diffused (torch.Tensor): the diffused residues as a boolean mask
            origin (torch.Tensor): the origin to center around. If None, the center of mass is calculated
            coditions_dict (dict): The conditions dict
        """
        # if not ((origin is not None) and (self.center_type == 'is_diffused')):
        if origin is None:
            # Default behavior: calculate center of mass for default case where ground truth of the protein and other targets are available
            center_of_mass_mask = torch.zeros(indep.xyz.shape[:2], dtype=torch.bool)
            if self.center_type == 'is_diffused':
                # CA atoms (position 1) of each frame forms center of rigid translations
                center_of_mass_mask[is_diffused,1] = True 
            elif self.center_type == 'is_not_diffused':
                # In this case center on not diffused if there are any not diffused atoms, otherwise center on diffused
                if torch.sum(~is_diffused) != 0:
                    center_of_mass_mask[~is_diffused,1] = True
                elif torch.sum(~is_diffused) == 0:
                    center_of_mass_mask[is_diffused,1] = True  
            elif self.center_type == 'all':
                center_of_mass_mask[:, 1] = True
            elif self.center_type == 'target_hotspot':
                # We are emulating center_type: all except the entire mass of the binder is at the CoM of the hotspot CBs
                origin = ppi.get_origin_target_hotspot(indep, conditions_dict, is_diffused)
            elif self.center_type == 'normal_to_target_hotspot':
                origin = ppi.get_origin_normal_to_target_hotspot(indep, conditions_dict, is_diffused)
            elif self.center_type == 'hotspot':
                origin = ppi.get_origin_target_hotspot(indep, conditions_dict, is_diffused, only_hotspots=True)

            # Calculate center of mass
            if origin is None:
                origin = get_center_of_mass(indep.xyz, center_of_mass_mask)

        # Calculate jitter amount and add to the origin
        if self.jitter > 0:
            gauss_norm_3d_mean = 1.5956947  # Expected L2 norm of a 3d unit gaussian
            jitter_amount = torch.randn_like(origin) / gauss_norm_3d_mean * self.jitter
            if torch.norm(jitter_amount).item() > self.jitter_clip:
                jitter_amount = jitter_amount / torch.norm(jitter_amount) * self.jitter_clip 
            origin += jitter_amount
                
        # Perform centering
        indep.xyz = indep.xyz - origin[None, None, :]

        return kwargs | dict(
            indep=indep,
            is_diffused=is_diffused,
            conditions_dict=conditions_dict,
            origin=origin
        )


class AddConditionalInputs:
    def __init__(self, p_is_guidepost_example: Union[float,bool], guidepost_bonds):
        """
        Args:
            p_is_guidepost_example (Union[float,bool]): The probability of using guideposts. If a boolean is given, guideposts are always used or never used
        """
        self.p_is_guidepost_example = p_is_guidepost_example
        self.guidepost_bonds = guidepost_bonds

    def __call__(self, indep: Indep, metadata: dict, masks_1d: dict, contig_map=types.SimpleNamespace(), **kwargs):
        """
        Duplicates/masks parts of a protein to create a conditional input.

        This method creates guideposts, performs atomization, and creates some extra masks for the updated
        indep.

        Args:
            indep: The independent variables of the structure.
            metadata: Metadata associated with the structure.
            masks_1d: Dictionary of 1D masks for various features (e.g. for motif, ...).
            contig_map: A contig map object for storing contig information. (LEGACY)
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the modified inputs and additional information:
                - indep: The modified independent variables of the structure.
                - is_diffused: 1D boolean mask indicating what will be diffused.
                - is_masked_seq: 1D boolean mask indicating what part of the sequence is masked.
                - atomizer: The atomizer used on the indep to atomize residues.
                - metadata: The metadata (copied from input).
                - masks_1d: The 1D masks (copied from input).
                - contig_as_guidepost: Whether the contig is used as a guidepost.
                - contig_map: The updated contig map (LEGACY).

        WARNING:
            The 'is_gp' (is guidepost) mask is not necessarily contiguous at the end of the structure.
            Atomized residues can appear after guidepost residues and may not be guideposts themselves.
            This can occur for example with covalent modifications that are not part of the motif, as 
            these are always atomized but not necessarily part of the motif.
        """
        is_res_str_shown = masks_1d['input_str_mask']
        is_res_seq_shown = masks_1d['input_seq_mask']
        is_atom_str_shown = masks_1d['is_atom_motif']
        can_be_gp = masks_1d['can_be_gp']

        # Sample guide posts with probability p_is_guidepost_example, or simply set true or false if p_is_guidepost_example is a boolean
        use_guideposts = self.p_is_guidepost_example if isinstance(self.p_is_guidepost_example, bool) else (torch.rand(1) < self.p_is_guidepost_example).item()
        masks_1d['use_guideposts'] = use_guideposts
        indep, is_diffused, is_masked_seq, atomizer, contig_map.gp_to_ptn_idx0 = aa_model.transform_indep(
            indep=indep,
            is_res_str_shown=is_res_str_shown,
            is_res_seq_shown=is_res_seq_shown,
            is_atom_str_shown=is_atom_str_shown,
            can_be_gp=can_be_gp,
            use_guideposts=use_guideposts,
            guidepost_bonds=self.guidepost_bonds,
            metadata=metadata
        )

        masks_1d['is_masked_seq']=is_masked_seq

        # WARNING:
        #  All 'is_gp' are not necessarily contiguously at the end.
        #  Atomized residues can come after gp residues (and aren't necessarily gp themselves), for 
        #  example when you have atomized residues that are not part of the motif: 
        #  This may for instance happen when there are covalent modifications on residues that are not part 
        #  of the guideposts, because a covalently modified residue is always atomized, but not necessarily motif.
        aa_model.assert_valid_seq_mask(indep, is_masked_seq)
        
        return kwargs | dict(
            indep=indep,
            is_diffused=is_diffused,
            is_masked_seq=is_masked_seq,
            atomizer=atomizer,
            metadata=metadata,
            masks_1d=masks_1d,
            contig_as_guidepost=use_guideposts,
            contig_map=contig_map,
        )


class ExpandConditionsDict:
    '''
    This class exists because conditions_dict gets generated on the vanilla indep before aa_model.transform_indep is called.
    Inside that function, some residues are atomized and others are duplicated as guidepost residues.
    Many of the variables inside conditions_dict are tensors that were the same length as indep before that transformation.
    This class allows those conditions to react to that change and adjust accordingly

    Also asserts the contents of conditions_dict
    '''
    def __init__(self, extra_expand_1d_atomized_ok_gp_not_conditions=[]):
        '''
        Args:
            extra_expand_1d_atomized_ok_gp_not_conditions (list[str]): List of items in conditions dict that are specified in yaml
        '''
        self.extra_expand_1d_atomized_ok_gp_not_conditions = extra_expand_1d_atomized_ok_gp_not_conditions
        pass

    def __call__(self, indep, atomizer, conditions_dict, contig_map, **kwargs):
        '''
        Translates conditions_dict from pre-transform_indep to post-transform_indep

        Args:
            indep (indep): indep
            atomizer (AtomizeResidues or None): The atomizer used on the indep
            conditions_dict (dict): The dictionary of conditions
            contig_map (ContigMap): The contig map
        '''

        conditions_dict_contents = set([
            'is_hotspot', # torch.Tensor[bool]: Which residues are hotspots? (7A cross-chain contacts)
            'hotspot_values', # torch.Tensor[float]: An alternative to is_hotspot where you can stick a value in it (like num 10a neighbors)
            'is_antihotspot', # torch.Tensor[bool]: Which residues are antihotspots? (10A cross-chain no-contacts)
            'antihotspot_values', # torch.Tensor[float]: An alternative to is_hotspot where you can stick a value in it (like dist from diffused)
            'ss_adj', # sec_struct_adj.SecStructAdjacency: The secondary structure and block adjacency conditioning
            'ideal_ss', # torch.Tensor[float]: A number 0 to 1 describing how ideal this piece of secondary structure is
            'avg_scn', # torch.Tensor[float]: A number 0 to 1 describing the sidechain neighbors of this piece of protein
            'loop_frac', # torch.Tensor[float]: A number 0 to 1 describing fraction of this chain that is loop by dssp
            'topo_spec', # torch.Tensor[float]: A class label of which of the topo_spec_choices this chain fits into (int but allows nan)


            # Conditions dict elements that kinda break the rules and do more calculations than they should (because they need atomized indeps)
            'target_hbond_satisfaction', # Enters: HBondSatisfactionApplierBase
                                         # Exists: dict[string,torch.Tensor[float]]: Various metrics related to h-bond satisfaction
        ] + self.extra_expand_1d_atomized_ok_gp_not_conditions)

        post_idx_from_pre_idx, is_atomized = aa_model.generate_pre_to_post_transform_indep_mapping(indep, atomizer, contig_map.gp_to_ptn_idx0)

        new_conditions_dict = {}
        for key in conditions_dict:
            assert key in conditions_dict_contents, (f'Please add {key} to ExpandConditionsDict! (or ExpandConditionsDict.extra_expand_1d_atomized_ok_gp_not_conditions '
                            'if you are pulling your values in a purely yaml way (like via datahub))' )

            if key == 'ss_adj':
                new_conditions_dict['ss_adj'] = conditions_dict['ss_adj'].expand_for_atomization_and_gp(indep, post_idx_from_pre_idx)

            elif key == 'is_hotspot':
                new_conditions_dict['is_hotspot'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['is_hotspot'], post_idx_from_pre_idx, False, 'is_hotspot')

            elif key == 'hotspot_values':
                new_conditions_dict['hotspot_values'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['hotspot_values'], post_idx_from_pre_idx, 0, 'hotspot_values')

            elif key == 'is_antihotspot':
                new_conditions_dict['is_antihotspot'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['is_antihotspot'], post_idx_from_pre_idx, False, 'is_antihotspot')

            elif key == 'antihotspot_values':
                new_conditions_dict['antihotspot_values'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['antihotspot_values'], post_idx_from_pre_idx, 0, 'antihotspot_values')

            elif key == 'ideal_ss':
                new_conditions_dict['ideal_ss'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['ideal_ss'], post_idx_from_pre_idx, np.nan, 'ideal_ss')

            elif key == 'avg_scn':
                new_conditions_dict['avg_scn'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['avg_scn'], post_idx_from_pre_idx, np.nan, 'avg_scn')

            elif key == 'loop_frac':
                new_conditions_dict['loop_frac'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['loop_frac'], post_idx_from_pre_idx, np.nan, 'loop_frac')

            elif key == 'topo_spec':
                new_conditions_dict['topo_spec'] = expand_1d_atomized_ok_gp_not(indep, conditions_dict['topo_spec'], post_idx_from_pre_idx, np.nan, 'topo_spec')


            elif key in self.extra_expand_1d_atomized_ok_gp_not_conditions:
                # For "temporary" values in the conditions_dict that could perhaps come from datahub
                fill_value = torch.nan if torch.is_floating_point(conditions_dict[key]) else 0
                new_conditions_dict[key] = expand_1d_atomized_ok_gp_not(indep, conditions_dict[key], post_idx_from_pre_idx, fill_value, key)

            else:
                assert False, f'Key {key}: not processed in ExpandConditionsDict'

        return kwargs | dict(
            indep=indep,
            atomizer=atomizer,
            conditions_dict=new_conditions_dict,
            contig_map=contig_map
        )


def get_contig_map(indep: Indep, input_str_mask: torch.Tensor, is_atom_motif: dict[int, list[str]]) -> ContigMap:

    motif_resis = sorted(list(set(
        indep.idx[input_str_mask].tolist() +
        indep.idx[list(is_atom_motif.keys())].tolist()
    )))

    contigs = []
    for ch, i in zip(indep.chains(), indep.idx):
        if i in motif_resis:
            contigs.append(f'{ch}{i}-{i}')
        else:
            contigs.append('1-1')
    contig_atoms = {
        f'{indep.chains()[i]}{indep.idx[i]}': atom_names
        for i, atom_names in is_atom_motif.items()
    }
    contig_map_args = {
        'parsed_pdb': {
            'seq': indep.seq.numpy(),
            'pdb_idx': [(ch,int(i)) for ch, i in zip(indep.chains(), indep.idx)],
        },
        'contigs': [','.join(contigs)],
        'contig_atoms': str({idx: ','.join(atom_names) for idx, atom_names in contig_atoms.items()}),
    }
    return ContigMap(**contig_map_args)


class ReconstructContigMap:

    def __call__(self, indep, masks_1d, **kwargs):
        contig_map = get_contig_map(indep, masks_1d['input_str_mask'], masks_1d['is_atom_motif'])
        return dict(
            contig_map=contig_map,
            indep=indep,
            masks_1d=masks_1d,
        ) | kwargs
    

class NAMotifPreservingTightCrop:
    """
    Cropping based on Paul's older motif cropping code

    Args:
        min_na_expand: minimum amount to expand around the crop index in each direction
        max_na_expand:
        min_prot_expand:
        max_prot_expand:
        closest_k: top k closest contacts to select from
    """

    def __init__(self,
                 min_na_expand = 5,
                 max_na_expand = 20,
                 min_prot_expand = 15,
                 max_prot_expand = 50,
                 closest_k = 4
                 ) -> None:        
        # Initialize cropper
        self.cropper = nucl_utils.NA_Motif_Preserving_Tight_Crop(
            min_na_expand=min_na_expand,
            max_na_expand=max_na_expand,
            min_prot_expand=min_prot_expand,
            max_prot_expand=max_prot_expand,
            closest_k=closest_k
        )

    def __call__(self, indep: Indep, atom_mask: torch.Tensor, **kwargs): 
        # Simply call the cropper and return the output indep     
        has_prot = torch.any(nucl_utils.get_resi_type_mask(indep.seq, 'prot_and_mask'))
        has_na = torch.any(nucl_utils.get_resi_type_mask(indep.seq, 'na'))

        if has_prot and has_na:
            # Only apply when there is protein and nucleic acid
            out = self.cropper(indep, atom_mask)
            indep = out['indep']
            atom_mask = out['atom_mask']
        
        return dict(
            indep=indep,
            atom_mask=atom_mask,
            **kwargs
        )
