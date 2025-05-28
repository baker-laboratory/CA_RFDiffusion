from __future__ import annotations  # Fake Import for type hinting

import math
import torch
from collections.abc import Mapping
import rf_diffusion.conditions.v2 as v2
from rf_diffusion import aa_model
from rf_diffusion import structure
import rf_diffusion.conditions.ss_adj.sec_struct_adjacency as sec_struct_adj
import rf_diffusion.conditions.ideal_ss as ideal_ss

from rf_diffusion import ppi

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rf_diffusion.aa_model import Indep        

from rf_diffusion import sasa

import rf_diffusion.nucleic_compatibility_utils as nucl_utils
import numpy as np

from omegaconf import OmegaConf

def get_extra_t1d(indep, featurizer_names, **kwargs):
    if not featurizer_names:
        return torch.zeros((indep.length(),0))
    t1d = []
    for name in featurizer_names:
        feats_1d = featurizers[name](indep, kwargs[name], **kwargs)
        t1d.append(feats_1d)
    return torch.cat(t1d, dim=-1)

def get_extra_t1d_inference(indep, featurizer_names, params_train, params_inference, features_cache, **kwargs):
    if not featurizer_names:
        return torch.zeros((indep.length(),0))
    t1d = []
    for name in featurizer_names:
        assert name in params_train
        assert name in params_inference
        feats_1d = inference_featurizers[name](indep, params_train[name], params_inference[name], cache=features_cache[name], **kwargs)
        t1d.append(feats_1d)
    return torch.cat(t1d, dim=-1)

def init_tXd_inference(indep, featurizer_names, params_train, params_inference, **kwargs):
    cache = {}
    for name in featurizer_names:
        assert name in params_train
        assert name in params_inference
        cache[name] = inference_featurizer_initializers.get(name, init_default)(indep, params_train[name], params_inference[name], **kwargs)
    return cache


def init_default(indep: Indep, feature_conf: OmegaConf, feature_inference_conf: OmegaConf, **kwargs):
    """
    Initializes the OmegaConf variables if needed between samples

    Args:
        indep (Indep): The holy indep.
        feature_conf (OmegaConf): The training configuration.
        feature_inference_conf (OmegaConf): The training configuration for inference.
        **kwargs: Additional keyword arguments.

    Returns:
        empty dictionary that can be a sub cache for this feature
    """
    return {}

def one_hot_bucket(x: torch.Tensor, boundaries: torch.Tensor):
    '''
    Return a one-hot encoding of the bucket x falls into.
    x must be in the interval (boundaries_low, boundaries_high).
    '''
    n_cat = len(boundaries) - 1
    cat_int = torch.bucketize(x, boundaries) - 1
    return torch.nn.functional.one_hot(cat_int, n_cat)

def get_boundary_values(style: str, T:int):
    '''
    Inputs
        style: Different ways of constructing the boundary values.
        T: Controls how finely the [0, 1] interval is binned.
    Returns
        Boundaries for little t embeddings. Spans [0, 1]
    '''
    if style == 'linear':
        return torch.linspace(0, 1, T + 1),
    elif style == 'low_t_heavy':
        return torch.cat([
            torch.arange(0,    0.05, 1 / (T * 16)),
            torch.arange(0.05, 0.10, 1 / (T * 8)),
            torch.arange(0.10, 0.20, 1 / (T * 4)),
            torch.arange(0.20, 0.40, 1 / (T * 2)),
            torch.arange(0.40, 1.00, 1 / (T * 1)),
            torch.tensor([1.]),
        ])

def get_little_t_embedding_inference(indep, feature_conf, feature_inference_conf, cache, t_cont, **kwargs):
    return get_little_t_embedding(indep, feature_conf, t_cont, **kwargs)

def get_little_t_embedding(indep, feature_conf, t_cont: float=None, **kwargs):
    '''
    Args
        t_cont [0, 1]: "continuous" time little_t

        feature_conf:
            style: Different ways of constructing the time boundary values.
            T: Controls how finely the [0, 1] interval is binned. Higher is finer.

    Returns
        One-hot encoding of the selected time bin.
    '''
    boundary_values = get_boundary_values(feature_conf.boundary_style, feature_conf.T)
    oh = one_hot_bucket(t_cont, boundary_values)[None]
    return {'t1d':oh.tile(indep.length(), 1)}

def get_radius_of_gyration(indep, radius_of_gyration=None, **kwargs):
    rog = torch.zeros((indep.length(),))
    rog_std = torch.zeros((indep.length(),))
    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), -1.0)
    rog_std_prot = torch.full((indep_prot.length(),), -1.0)
    for is_chain in indep_prot.chain_masks():
        std = torch.abs(torch.normal(0.0, radius_of_gyration.std_std, (1,)))
        rog_chain = radius_of_gyration_xyz(indep_prot.xyz[is_chain, 1])
        rog_chain = torch.normal(rog_chain, std)
        rog_prot[is_chain] = rog_chain
        rog_std_prot[is_chain] = std
    rog[is_prot] = rog_prot
    rog_std[is_prot] = rog_std_prot
    return {'t1d':(rog, rog_std)}

def radius_of_gyration_xyz(xyz):
    L, _ = xyz.shape

    com = torch.mean(xyz, dim=0)
    dist = torch.cdist(xyz[None,...], com[None,...])[0]
    return torch.sqrt( torch.sum(torch.square(dist)) / L)

def get_relative_sasa(indep, relative_sasa=None, **kwargs):
    return {'t1d':sasa.noised_relative_sasa(indep, relative_sasa.std_std)}

def get_sinusoidal_timestep_embedding_inference(indep, feature_conf, feature_inference_conf, cache, t_cont, **kwargs):
    return get_sinusoidal_timestep_embedding_training(indep, feature_conf, t_cont)

def get_sinusoidal_timestep_embedding_training(indep, feature_conf, t_cont: float=None, **kwargs):
    emb = get_sinusoidal_timestep_embedding(torch.tensor([t_cont]), feature_conf.embedding_dim, feature_conf.max_positions)
    return {'t1d':emb.tile((indep.length(),1))}

def get_sinusoidal_timestep_embedding(timesteps, embedding_dim, max_positions):
    # Adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert (embedding_dim % 2 == 0)
    assert ((0 <= timesteps) * (1 >= timesteps)).all()
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def get_radius_of_gyration_inference(indep, feature_conf):

    rog = torch.zeros((indep.length(),))
    rog_std = torch.zeros((indep.length(),))

    is_prot = ~indep.is_sm * ~indep.is_gp
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), -1.0)
    rog_std_prot = torch.full((indep_prot.length(),), -1.0)
    for is_chain in indep_prot.chain_masks():
        rog_prot[is_chain] = feature_conf.mean
        rog_std_prot[is_chain] = feature_conf.std
    
    rog[is_prot] = rog_prot
    rog_std[is_prot] = rog_std_prot
    return {'t1d':(rog, rog_std)}

def get_relative_sasa_inference(indep, feature_conf, **kwargs):
    sasa = torch.full((indep.length(),), -10.0)
    sasa[indep.is_sm] = feature_conf.mean
    std = torch.full((indep.length(),), feature_conf.std)
    return {'t1d':(sasa, std)}


def get_extra_tXd(indep, featurizer_names, **kwargs):
    '''
    Get the extra_t1d and extra_t2d features for training

    Args:
        indep (indep): the indep
        featurizer_names (list[str]): The list of featurizers that you wish to use
        kwargs (dict): Additional keyword arguments

    Returns:
        extra_t1d (torch.Tensor[float]): The extra_t1d values [L,x] (where x is variable based on features)
        extra_t2d (torch.Tensor[float]): The extra_t2d values [L,L,x] (where x is variable based on features)
    '''
    if not featurizer_names:
        return (torch.zeros((indep.length(),0)), torch.zeros((indep.length(),indep.length(),0)))
    t1d = [torch.zeros((indep.length(),0))]
    t2d = [torch.zeros((indep.length(),indep.length(),0))]
    for name in featurizer_names:
        feats = featurizers[name](indep, kwargs[name], **kwargs)
        assert isinstance(feats, Mapping), 'The get extra_tXd functions now return a dictionary.'
        if 't1d' in feats:
            t1d.append(feats['t1d'])
        if 't2d' in feats:
            t2d.append(feats['t2d'])
    return (torch.cat(t1d, dim=-1), torch.cat(t2d, dim=-1))



def get_extra_tXd_inference(indep, featurizer_names, params_train, params_inference, features_cache, **kwargs):
    '''
    Get the extra_t1d and extra_t2d features for inference

    Args:
        indep (indep): the indep
        featurizer_names (list[str]): The list of featurizers that you wish to use
        params_train (omegaconf.dictconfig.DictConfig): The value of conf.extra_tXd_params
        params_train (omegaconf.dictconfig.DictConfig): The value of conf.inference.conditions
        features_cache (dict): The cache of feature state information 
        kwargs (dict): Additional keyword arguments

    Returns:
        extra_t1d (torch.Tensor[float]): The extra_t1d values [L,x] (where x is variable based on features)
        extra_t2d (torch.Tensor[float]): The extra_t2d values [L,L,x] (where x is variable based on features)
    '''
    if not featurizer_names:
        return (torch.zeros((indep.length(),0)), torch.zeros((indep.length(),indep.length(),0)))
    t1d = [torch.zeros((indep.length(),0))]
    t2d = [torch.zeros((indep.length(),indep.length(),0))]
    for name in featurizer_names:
        assert name in params_train, f'Featurizer: {name} not found in training params'
        assert name in params_inference, f'Featurizer: {name} not found in inferece params'
        feats = inference_featurizers[name](indep, params_train[name], params_inference[name], cache=features_cache[name], **kwargs)
        assert isinstance(feats, Mapping), 'The get extra_tXd functions now return a dictionary.'
        if 't1d' in feats:
            t1d.append(feats['t1d'])
        if 't2d' in feats:
            t2d.append(feats['t2d'])
    return (torch.cat(t1d, dim=-1), torch.cat(t2d, dim=-1))

def get_nucleic_ss(indep, feature_conf, **kwargs):
    """
    This placeholder function is an example of how we add additional t2d features.
    Additional t2d dimensions added: 3

    Args:
        indep (indep): the indep
        feature_conf (omegaconf.dictconfig.DictConfig): Configuration for this feature
        kwargs (dict): Additional keyword arguments

    Returns:
        dict:
            t2d (torch.Tensor[float]): The t2d params [L,L,3]
    """
    L = indep.seq.shape[0] # get that seq length
    ss_matrix = (2*torch.ones((L,L))).long() # Make a matrix 
    ss_templ_onehot = F.one_hot(ss_matrix, num_classes=3)
    return {'t2d':ss_templ_onehot}


def get_nucleic_ss_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    """
    This placeholder function is an example of how we add additional t2d features for inference.
    Additional t2d dimensions added: 3

    Args:
        indep (indep): the indep
        feature_conf (omegaconf.dictconfig.DictConfig): Configuration for this feature from training
        feature_conf (omegaconf.dictconfig.DictConfig): Configuration for this feature from inference
        kwargs (dict): Additional keyword arguments

    Returns:
        dict:
            t2d (torch.Tensor[float]): The t2d params [L,L,3]
    """
    L = indep.seq.shape[0] # get that seq length
    ss_matrix = (2*torch.ones((L,L))).long() # Make a matrix 
    ss_templ_onehot = F.one_hot(ss_matrix, num_classes=3)
    return {'t2d':ss_templ_onehot}

def get_ss_comp(indep: Indep, 
                train_conf: dict, **kwargs):
    """
    Calculate the secondary structure composition of a protein sequence.

    Args:
        indep (Indep): The holy indep
        train_conf (dict): A dictionary containing training configurations.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: A tensor representing the secondary structure composition of the protein sequence. [L, 4]

    WARNING: Not compatible with multiple diffused proteins
    """    
    ss = torch.zeros(indep.length(), 4)

    # Skip if no protein or no nucleic acid with default zero values
    if sum(nucl_utils.get_resi_type_mask(indep.seq, 'prot_and_mask')) == 0 or \
       sum(nucl_utils.get_resi_type_mask(indep.seq, 'na')) == 0:
        return ss
    
    # Iterate over all chains and calculate ss composition per chain, skip non proteins
    for chain_mask in indep.chain_masks():    
        # Currently, only index protein chains 
        if sum(nucl_utils.get_resi_type_mask(indep.seq[chain_mask], 'prot_and_mask')) == 0:
            continue        
        chain_mask = torch.tensor(indep.chain_masks()[0], dtype=bool)  # Convert to torch tensor else bugs
        indep_slice, _ = aa_model.slice_indep(indep, chain_mask)
        min_prot_size = 8
        if float(torch.rand(1)) < train_conf['p_unconditional'] or sum(chain_mask) <= min_prot_size:
            ss[chain_mask,0] = 1
        else:              
            try:
                ss_assign, _ = structure.get_dssp(indep_slice)
                ss_assign = ss_assign[ss_assign != structure.ELSE]
                # Calculate ss composition and assign values to the ss matrix  
                n_loop = sum(ss_assign == structure.LOOP)
                n_sheet = sum(ss_assign == structure.STRAND)
                n_helix = sum(ss_assign == structure.HELIX)
                n_total = n_loop + n_sheet + n_helix        
                ss[chain_mask,1] = n_helix / n_total 
                ss[chain_mask,2] = n_sheet / n_total 
                ss[chain_mask,3] = n_loop / n_total                
            except RuntimeError:
                # If it doesn't work, set unconditional
                ss[chain_mask,0] = 1
        #ic(ss)
    return ss

def get_ss_comp_inference(indep: Indep, 
                          feature_conf: OmegaConf, 
                          feature_inference_conf: OmegaConf, 
                          cache:dict, 
                          **kwargs):
    """
    Calculate the secondary structure composition inference for a given sequence.

    Args:
        indep (Indep): The holy indep.
        feature_conf (dict): feature config
        feature_inference_conf (dict): feature inference config
        cache (dict): data cache
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: A tensor representing the secondary structure composition. [L, 4]]

    """
    ss = torch.zeros(indep.length(), 4)
    is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot_and_mask')
    if float(torch.sum(is_prot)) == 0:
        return ss    
    if 'unconditional' in feature_inference_conf and feature_inference_conf['unconditional']:
        ss[is_prot,0] = 1
    else:
        # Check if cached values are available
        if 'ss_comp_cache' in cache:
            helix, sheet, loop = cache['ss_comp_cache']
        else:
            if 'helix_only' in feature_inference_conf and feature_inference_conf['helix_only']:
                loop_lb,loop_ub = [float(f) for f in feature_inference_conf['loop_range'].split('-')]
                loop = min([1.0, loop_lb + (loop_ub - loop_lb) * float(torch.rand(1))])
                helix = 1.0 - loop
                sheet = 0.0
            else:
                helix_lb,helix_ub = [float(f) for f in feature_inference_conf['helix_range'].split('-')]
                loop_lb,loop_ub = [float(f) for f in feature_inference_conf['loop_range'].split('-')]
                loop = min([1.0, loop_lb + (loop_ub - loop_lb) * float(torch.rand(1))])
                helix = min([1.0 - loop, helix_lb + (helix_ub - helix_lb) * float(torch.rand(1))])        
                sheet = 1.0 - loop - helix
            cache['ss_comp_cache'] = (helix, sheet, loop)
        ss[is_prot,1] = helix
        ss[is_prot,2] = sheet
        ss[is_prot,3] = loop
    # ic(ss[0:2])
    return ss


"""
Nucleic acid hotspots
"""


def get_nucleic_base_hotspots(indep: Indep, 
                              train_conf: OmegaConf,
                              **kwargs) -> torch.Tensor:
    """
    Calculates the hotspots for nucleic bases in the given independent variable.
    Used only for training.

    Args:
        indep (Indep): The independent variable.
        train_conf: The training configuration.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: A tensor representing the hotspots for nucleic bases. [L, 2]
    """    
    hotspots = torch.zeros(indep.length(), 2)
    if sum(nucl_utils.get_resi_type_mask(indep.seq, 'prot_and_mask')) == 0 or \
       sum(nucl_utils.get_resi_type_mask(indep.seq, 'na')) == 0:
        return hotspots

    contacts_idx, base_contacts_idx = nucl_utils.get_nucl_prot_contacts(indep, dist_thresh=4.5, is_gp=indep.is_gp, ignore_prot_bb=True)
    if len(base_contacts_idx) == 0 or float(torch.rand(1)) < train_conf['p_unconditional']:        
        # Do not sample, but set flag for unconditional generation
        is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na') * ~indep.is_gp * ~indep.is_sm
        hotspots[is_nucl, 0] = 1
    else:
        # Sample a random number of base contacts
        assert len(base_contacts_idx) > 0  # Ensure, there are some that can be sampled
        lb,ub = 0.2, 0.8
        perc_contact_sample = np.random.rand() * (ub - lb) + lb
        n_sample = min([len(base_contacts_idx), max([1, int(perc_contact_sample * len(base_contacts_idx))])])
        idx_sample = np.random.choice(base_contacts_idx, n_sample, replace=False)
        for i in idx_sample:
            hotspots[i, 1] = 1
    #ic(torch.sum(hotspots[:,1]), len(base_contacts_idx), base_contacts_idx)
    return hotspots

def get_nucleic_base_hotspots_inference(indep: Indep, 
                              feature_conf: OmegaConf,
                              feature_inference_conf: OmegaConf,       
                              cache: dict,                     
                              **kwargs) -> torch.Tensor:
    """
    Calculates the hotspots for nucleic bases in the given independent variable.
    Used for inference.

    NOTE: indexing for hotspots is 1-based starting from the first nucleic acid base.
    NOTE: Will not work with discontinuous nucleic acid chains from masking and breaks in the chain.

    Args:
        indep (Indep): The independent variable.
        feature_conf (OmegaConf): The config.
        feature_inference_conf (OmegaConf): The inference config
        cache (dict): data cache
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: A tensor representing the hotspots for nucleic bases. [L, 2]
    """      
    # TODO: depends on contig reader  
    hotspots = torch.zeros(indep.length(), 2)
    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    if feature_inference_conf['hotspots'] == '':
        # Set unconditional mode
        hotspots[is_nucl,0] = 1
        return hotspots
    hotspot_keys = [r.split('-') for r in feature_inference_conf['hotspots'].split(',')]    
    chain_order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chain_masks = indep.chain_masks()
    for chain_id, pdb_idx in hotspot_keys:
        pdb_idx = int(pdb_idx)
        chain_mask = chain_masks[chain_order.index(chain_id)]
        start_index = min(indep.idx[chain_mask]) - 1
        pdb_idx = start_index + pdb_idx
        mask = torch.logical_and(torch.tensor(indep.chains() == chain_id, dtype=bool), indep.idx == pdb_idx)
        hotspots[mask, 0] = 1
        assert sum(mask) == 1   
    
    return hotspots





def get_per_res_plddt_pae_int_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    '''
    See get_per_res_plddt_pae_int_conditioning()
    '''
    return get_per_res_plddt_pae_int_conditioning(indep, feature_conf, **kwargs)

def get_per_res_plddt_pae_int_conditioning(indep, feature_conf, per_res_plddt=None, per_res_pae_int=None, per_res_plddt_std=None, per_res_pae_int_std=None, **kwargs):
    '''
    Generates the per res plddt and pae interaction conditioning extra t1d

    Args:
        indep (Indep): indep
        feature_conf (OmegaConf): The configuration for this feature

    Returns:
        dict:
            t1d (torch.Tensor[bool]): The extra t1d [L, 6 + (N_topos)]
    '''

    plddt_scaling = feature_conf.get('plddt_scaling', 0.01)
    pae_int_scaling = feature_conf.get('pae_int_scaling', 0.05)

    # Either initialize a blank version of each feature or clone them
    if per_res_plddt is None:
        per_res_plddt = torch.full((indep.length(),), torch.nan)
    else:
        per_res_plddt = per_res_plddt.clone()
    if per_res_pae_int is None:
        per_res_pae_int = torch.full((indep.length(),), torch.nan)
    else:
        per_res_pae_int = per_res_pae_int.clone()
    if per_res_plddt_std is None:
        per_res_plddt_std = torch.full((indep.length(),), torch.nan)
    else:
        per_res_plddt_std = per_res_plddt_std.clone()
    if per_res_pae_int_std is None:
        per_res_pae_int_std = torch.full((indep.length(),), torch.nan)
    else:
        per_res_pae_int_std = per_res_pae_int_std.clone()


    assert len(per_res_plddt) == indep.length(), 'per_res_plddt vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(per_res_pae_int) == indep.length(), 'per_res_pae_int vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(per_res_plddt_std) == indep.length(), 'per_res_plddt_std vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(per_res_pae_int_std) == indep.length(), 'per_res_pae_int_std vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'

    # convert nan to 0
    plddt_mask = ~torch.isnan(per_res_plddt)
    per_res_plddt[~plddt_mask] = 0
    per_res_plddt_std[~plddt_mask] = 0

    pae_int_mask = ~torch.isnan(per_res_pae_int)
    per_res_pae_int[~pae_int_mask] = 0
    per_res_pae_int_std[~pae_int_mask] = 0

    # Scale the values to roughly 0-1
    per_res_plddt *= plddt_scaling
    per_res_plddt_std *= plddt_scaling

    per_res_pae_int *= pae_int_scaling
    per_res_pae_int_std *= pae_int_scaling

    # print('pae', per_res_pae_int)
    # print('plddt', per_res_plddt)

    # Generate the feature
    extra_t1d = torch.stack((plddt_mask, per_res_plddt, per_res_plddt_std, pae_int_mask, per_res_pae_int, per_res_pae_int_std), axis=-1)

    return {'t1d':extra_t1d}



# Add user specific featurizers to this dictionary for training
featurizers = {
    'radius_of_gyration': get_radius_of_gyration,
    'relative_sasa': get_relative_sasa,
    'radius_of_gyration_v2': v2.get_radius_of_gyration,
    'relative_sasa_v2': v2.get_relative_sasa,
    'little_t_embedding': get_little_t_embedding,
    'sinusoidal_timestep_embedding': get_sinusoidal_timestep_embedding_training,
    'nucleic_ss' : get_nucleic_ss,
    'secondary_structure_composition' : get_ss_comp,
    'nucleic_base_hotspots': get_nucleic_base_hotspots,
    'ss_adj_cond': sec_struct_adj.get_ss_adj_conditioning,
    'ppi_hotspots_antihotspots': ppi.get_hotspots_antihotspots_conditioning,
    'ideal_ss_cond': ideal_ss.get_ideal_ss_conditioning,
    'per_res_plddt_pae_int_cond': get_per_res_plddt_pae_int_conditioning,
}

# Add user specific featurizers to this dictionary for inference
inference_featurizers = {
    'radius_of_gyration': get_radius_of_gyration_inference,
    'relative_sasa': get_relative_sasa_inference,
    'radius_of_gyration_v2': v2.get_radius_of_gyration_inference,
    'relative_sasa_v2': v2.get_relative_sasa_inference,
    'little_t_embedding': get_little_t_embedding_inference,
    'sinusoidal_timestep_embedding': get_sinusoidal_timestep_embedding_inference,
    'nucleic_ss' : get_nucleic_ss_inference,
    'secondary_structure_composition' : get_ss_comp_inference,    
    'nucleic_base_hotspots' : get_nucleic_base_hotspots_inference,
    'ss_adj_cond': sec_struct_adj.get_ss_adj_conditioning_inference,
    'ppi_hotspots_antihotspots': ppi.get_hotspots_antihotspots_conditioning_inference,
    'ideal_ss_cond': ideal_ss.get_ideal_ss_conditioning_inference,
    'per_res_plddt_pae_int_cond': get_per_res_plddt_pae_int_conditioning_inference,
}

# Add user specific featurizer initializer functions to this dictionary (optional) for inference
inference_featurizer_initializers = {
    'radius_of_gyration_v2' : v2.init_radius_of_gyration,
}
