import logging

from rf_diffusion import aa_model
from rf_diffusion import data_loader
from rf_diffusion import conditioning
from rf_diffusion.contigs import ContigMap
import rf_diffusion.inference.centering as centering
import rf_diffusion.inference.scaffold as scaffold
import rf_diffusion.conditions.ss_adj.sec_struct_adjacency as sec_struct_adj

import copy
from omegaconf import OmegaConf
import torch
import numpy as np
import torch.utils.data
from rf_diffusion.inference import utils as iu

from typing import Tuple

logger = logging.getLogger(__name__)


def spy(*args):
    pass 

class PDBLoaderDataset(torch.utils.data.Dataset):
    """
    Makes indeps from PDBs for inference. Subsequent transforms are applied to complete the indep
    featurization process.
    """
    def __init__(self, conf: OmegaConf):
        """
        Args:
            confs (OmegaConf): Omegaconf for inference time
        """                
        self.conf = conf
        self.pdb_fp = conf.inference.input_pdb
        # Create contig map from the conf
        self.target_feats = iu.process_target(self.pdb_fp, parse_hetatom=False, center=False)

    def __getitem__(self, idx):
        '''
        We wrap getitem_inner here to prevent an internal IndexError from causing
        a StopIteration which would hide the internal error when iterating over
        iter(PDBLoaderDataset).
        '''
        try:
            return self.getitem_inner(idx)
        except IndexError as e:
            raise Exception(f"Failed to access PDBLoaderDataset[{idx}]") from e

    def getitem_inner(self, idx, contig_conf=None, origin_override=None):
        conf = self.conf

        # Create contig map from the conf
        L = len(self.target_feats['pdb_idx'])              

        # Create the indep from the pdb file along with ligand metadata   
        indep_orig, metadata = aa_model.make_indep(self.pdb_fp, 
                                                   conf.inference.ligand, 
                                                   return_metadata=True)
                                                   
        spy({'indep_getitem_inner_point_A':copy.deepcopy(indep_orig)})
        
        # If doing refinement, add OG motif coordinates to metadata for later templating
        conditions_dict = {}
        if conf.inference.get('ca_rfd_refine'):
            aa_model.get_refinement_metadata(conditions_dict, conf)

        contig_map = ContigMap(self.target_feats, **(contig_conf or conf.contigmap))

        
        if conf.inference.get('ca_rfd_refine'):
            # refinement sanity check - is built contigs identical in length to input pdb? 
            assert indep_orig.is_sm.sum() + len(contig_map.ref) == len(indep_orig.xyz)
        
        # Calculate centering context and validate
        if origin_override is None:
            origin = centering.extract_centering_origin(indep_orig, self.pdb_fp, for_partial_diffusion(conf))
        else:
            print("Overriding origin with:", origin_override)
            origin = origin_override

        # Prepare non atomization contig insertion (pre tranform-indep)
        model_adaptor = aa_model.Model(conf)
        indep, masks_1d = model_adaptor.insert_contig_pre_atomization(indep_orig, 
                                                                      contig_map, 
                                                                      metadata, 
                                                                      for_partial_diffusion(conf) or conf.inference.get('ca_rfd_refine'))

        # Validate strategies
        centering.validate_centering_strategy(origin, for_partial_diffusion(conf), conf)    
        sec_struct_adj.validate_ss_adj_strategy(conf)

        feats = {'contig_map': contig_map, 'indep': indep, 'metadata': metadata, 
                 'masks_1d': masks_1d, 'L': L, 'conf': conf,
                 'origin': origin, 'conditions_dict': conditions_dict}

        spy({'indep_getitem_inner_point_B':copy.deepcopy(indep)})
        return feats

    def __len__(self):
        return 1


class ScaffoldPDBLoaderDataset(PDBLoaderDataset):
    """
    Makes indeps from PDBs for inference but where the free portion of the contig is loaded as a scaffold
    """
    def __init__(self, conf, scaffold_loader):
        """
        Args:
            confs (OmegaConf): Omegaconf for inference time
            scaffold_loader (ScaffoldLoader): The scaffold loader to use
        """                
        super().__init__(conf)
        self.scaffold_loader = scaffold_loader
        self.scaffold_loader.set_target_feats(self.target_feats)

    def getitem_inner(self, idx, **kwargs):
        contig_conf, ss_adj = self.scaffold_loader[idx]
        feats = super().getitem_inner(idx, contig_conf=contig_conf, **kwargs)
        feats['conditions_dict']['ss_adj'] = ss_adj

        return feats

    def __len__(self):
        return len(self.scaffold_loader)

def for_partial_diffusion(conf: OmegaConf):
    return conf.diffuser.partial_T is not None


def validate_partial_diffusion(conf: OmegaConf, 
                    contig_map: ContigMap, 
                    indep: aa_model.Indep, 
                    is_diffused: torch.tensor, 
                    L: int):
    """
    Validate the partial diffusion settings for the current inference

    Raises:
        AssertionError: If the partial diffusion settings are invalid
    """
    if for_partial_diffusion(conf)  :
        mappings = contig_map.get_mappings()
        # This is due to the fact that when inserting a contig, the non-motif coordinates are reset.
        if conf.inference.safety.sidechain_partial_diffusion:
            print("You better know what you're doing when doing partial diffusion with sidechains")
        else:
            # Validate coordinates and strategy
            assert indep.xyz.shape[0] ==  L + torch.sum(indep.is_sm), f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {indep.xyz.shape[0]} != {L+torch.sum(indep.is_sm)}"
            assert torch.all(is_diffused[indep.is_sm] == 0), "all ligand atoms must be in the motif"
        assert (mappings['con_hal_idx0'] == mappings['con_ref_idx0']).all(), f"all positions in the input PDB must correspond to the same index in the output pdb: {list(zip(mappings['con_hal_idx0'], mappings['con_ref_idx0']))=}"


def get_t_inference(conf: OmegaConf) -> Tuple[int, float]:
    """
    Determine the diffusion time step to use during inference

    Returns:
        t_step_input (int, float): the time step to use for diffusion
        t_cont (float): the time step to use for diffusion as a fraction of the total diffusion time
    """
    t_step_input = conf.diffuser.T
    # Diffuse the contig-mapped coordinates 
    if for_partial_diffusion(conf):
        t_step_input = conf.diffuser.partial_T
        assert conf.diffuser.partial_T <= conf.diffuser.T   

    t_cont = t_step_input/conf.diffuser.T

    return t_step_input, t_cont


def get_pdb_loader(conf):
    '''
    Factory method to figure out which base PDB Loader to generate

    Args:
        conf (OmegaConf): The config

    Returns:
        loader (torch.utils.data.Dataset): A loader that looks like PDBLoaderDataset

    '''

    # Support for the old Joe + Nate ss/adj files using conf.scaffoldguided
    if sec_struct_adj.user_wants_ss_adj_scaffold(conf):
        scaffold_loader = scaffold.FileSSADJScaffoldLoader(conf)
        return ScaffoldPDBLoaderDataset(conf, scaffold_loader)
    
    # Not using scaffolds
    return PDBLoaderDataset(conf)

class InferenceDataset:
    """
    Dataset for inference time. Applies the transformations used to modify and featurize indep
    """
    def __init__(self, conf: OmegaConf, diffuser=None):
        """
        Args:
            conf (OmegaConf): The configuration for the inference
            diffuser (Diffuser): The diffuser for the inference, can be set to None explicitly to avoid diffusion on the data            
        """

        def update_inference_state(indep, metadata, contig_map, **kwargs):
            """
            Update stateful attributes for inference
            """            
            # Update contig_map with ligand information
            contig_map.ligand_names = np.full(indep.length(), '', dtype='<U3')
            contig_map.ligand_names[contig_map.hal_idx0.astype(int)] = metadata['ligand_names'][contig_map.ref_idx0] 

            return kwargs | dict(contig_map=contig_map, indep=indep, metadata=metadata)

        def diffuse(conf: OmegaConf,
                    indep: aa_model.Indep, 
                    metadata: dict,
                    is_diffused: torch.tensor, 
                    is_masked_seq: torch.tensor,
                    contig_map: ContigMap, 
                    L: int,
                    **kwargs):
            """
            Perform inference time diffusion on the incoming indep using the is_diffused mask
            Mirrors the logic performed during training, but deviates because diffusion is performed
            to completion (or partially during partial diffusion) during inference, whereas time t values 
            are randomly selected during training.

            Args:
                conf (OmegaConf): the config for the inference
                indep (Indep): the holy Indep
                metadata (dict): the metadata for the ligand
                is_diffused (torch.tensor): the diffused residues as a boolean mask
                is_masked_seq (torch.tensor): the masked sequence residues as a boolean mask
                contig_map (ContigMap): the contig map post contig insertion
                L (int): the length of the target sequence
            """
            # As in training, determine the diffusion time step to use   
            validate_partial_diffusion(conf, contig_map, indep, is_diffused, L)       
            t_step_input, t_cont = get_t_inference(conf)

            # At some point, it may be possible to place extra t1d features here, but currently it produces bugged outputs 

            indep_uncond, indep_cond = aa_model.diffuse_then_add_conditional(conf, diffuser, copy.deepcopy(indep.clone()), is_diffused, t_step_input)
            
            aa_model.mask_indep(indep_uncond, is_masked_seq)  # mask as in training (after diffuse)
            aa_model.mask_indep(indep_cond, is_masked_seq)                

            return kwargs | dict(
                indep_uncond=indep_uncond,
                indep_cond=indep_cond,
                is_diffused=is_diffused,
                indep_orig=indep,  # overwrite indep_orig with the pre diffused indep
                contig_map=contig_map,
                metadata=metadata,
                t_step_input=t_step_input,
            )

        def feature_tuple_from_feature_dict(**kwargs):
            indep_uncond = kwargs.pop('indep_uncond')
            indep_orig = kwargs.pop('indep_orig')
            indep_cond = kwargs.pop('indep_cond')
            metadata = kwargs.pop('metadata')
            is_diffused = kwargs.pop('is_diffused')
            atomizer = kwargs.pop('atomizer')
            contig_map = kwargs.pop('contig_map')
            t_step_input = kwargs.pop('t_step_input')
            conditions_dict = kwargs.pop('conditions_dict')
            masks_1d = kwargs.pop('masks_1d')
            return (
                    indep_uncond,
                    indep_orig,
                    indep_cond,
                    metadata,
                    is_diffused,
                    atomizer,
                    contig_map,
                    t_step_input,
                    conditions_dict,
                    masks_1d,
                    kwargs
            )

        # Create dataset from the pdb input and config
        dataset = get_pdb_loader(conf)
        
        # Curate transforms as in training 
        transforms = []
        # Add training only transforms
        upstream_names = conf.upstream_inference_transforms.names if hasattr(conf, 'upstream_inference_transforms') else []
        for transform_name in upstream_names:
            transforms.append(
                getattr(conditioning, transform_name)(**conf.upstream_inference_transforms.configs[transform_name]),
            )
        for transform_name in conf.transforms.names:
            # Ignore legacy blacklist transforms during inference (this is for backward compatibility)
            if transform_name in conditioning.LEGACY_TRANSFORMS_TO_IGNORE: continue
            transforms.append(
                getattr(conditioning, transform_name)(**conf.transforms.configs[transform_name]),
            )
        # Add related transforms used to those in training, but for inference            
        transforms.extend([
            update_inference_state,
            diffuse,
            feature_tuple_from_feature_dict
        ])

        # Create the transformed dataloader
        self.dataset = data_loader.TransformedDataset(dataset, transforms)

    def __getitem__(self, i):
        return self.getitem_inner(i)

    def getitem_inner(self, i, **kwargs):
        if hasattr(self.dataset, 'getitem_inner'):
            return self.dataset.getitem_inner(i, **kwargs)
        else:
            return self.dataset[i]

    def __len__(self):
        return len(self.dataset)
