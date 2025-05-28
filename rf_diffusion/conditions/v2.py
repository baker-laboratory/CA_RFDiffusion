import torch

from rf_diffusion import aa_model

from rf_diffusion import sasa
from rf_diffusion import nucleic_compatibility_utils as nucl_utils

import torch.nn.functional as F
import numpy as np

def get_relative_sasa(indep, conf=None, **kwargs):
    if 1 - torch.rand(1) > conf.get('prob', 0.5): # 1 - for test consistency
        return {'t1d':torch.zeros((indep.length(), conf.n_bins + 1))}
    rasa = sasa.get_relative_sasa(indep)
    is_feature_applicable = indep.is_sm
    one_hot = one_hot_buckets(rasa, conf.low, conf.high, conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}

def radius_of_gyration_xyz(xyz):
    L, _ = xyz.shape
    com = torch.mean(xyz, dim=0)
    dist = torch.cdist(xyz[None,...], com[None,...])[0]
    return torch.sqrt( torch.sum(torch.square(dist)) / L)

def get_radius_of_gyration(indep, conf=None, **kwargs):
    if 1 - torch.rand(1) > conf.get('prob', 0.5): # 1 - for test consistency
        return {'t1d':torch.zeros((indep.length(), conf.n_bins + 1))}
    rog = torch.zeros((indep.length(),))
    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), 0.0)
    
    # Iterate over all protein chains and calculate radii of gyration
    for is_chain in indep_prot.chain_masks():
        rog_chain = radius_of_gyration_xyz(indep_prot.xyz[is_chain, 1])
        rog_prot[is_chain] = rog_chain
    rog[is_prot] = rog_prot
    is_feature_applicable = is_prot
    one_hot = one_hot_buckets(rog, conf.low, conf.high, conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}


def one_hot_buckets(a, low, high, n, eps=1e-6):
    '''
    First category absorbs anything below low
    Last category absorbs anything above high
    '''
    step = (high-low) / n
    bins = torch.linspace(low+step, high-step, n-1)
    cat = torch.bucketize(a, bins).long()
    return F.one_hot(cat, num_classes=n)

def init_radius_of_gyration(indep, feature_conf, feature_inference_conf, **kwargs):
    """
    Initialize the radius of gyration fature


    During interface use the following additional parameters
    "spread" to give a normal distribution std in addition to rog, which now becomes the mean

    Args:
        indep (Indep): The independent variable.
        feature_conf (omegaconf): The feature config.
        feature_inference_conf (omegaconf): The feature inference config.

    Returns:
        None
    """
    cache = {}

    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)     
    # Create random values if required
    rog_vals = [(max([0, np.random.normal(feature_inference_conf.rog, 
                                            feature_inference_conf.spread)]) 
                if 'spread' in feature_inference_conf 
                else feature_inference_conf.rog) # Use default value if not randomized
                for _ in indep_prot.chain_masks()]
    cache['rog_vals_cache'] = rog_vals

    return cache

def get_radius_of_gyration_inference(indep, feature_conf, feature_inference_conf, cache, **kwargs):
    """
    Calculates the radius of gyration fature

    Args:
        indep (Indep): The holy indep.
        feature_conf (omegaconf): The feature config.
        feature_inference_conf (omegaconf): The feature inference config.
        cache (dict): data cache

    Returns:
        rog feature
    """    
    # TODO: Currently assumes single diffusing chain
    if not feature_inference_conf.active:
        return {'t1d':torch.zeros((indep.length(), feature_conf.n_bins + 1))}
    rog = torch.zeros((indep.length(),))
    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)    
    rog_prot = torch.full((indep_prot.length(),), 0.0)
    rog_vals = cache['rog_vals_cache']

    for is_chain, rog_chain in zip(indep_prot.chain_masks(), rog_vals):
        rog_prot[is_chain] = rog_chain

    rog[is_prot] = rog_prot
    is_feature_applicable = is_prot
    one_hot = one_hot_buckets(rog, feature_conf.low, feature_conf.high, feature_conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    out = torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)
    ic(out[0:2, :], out[-3:-1, :])
    return out


def get_relative_sasa_inference(indep, feature_conf, feature_inference_conf, cache, **kwargs):
    """
    Calculates the radius of gyration fature

    Args:
        indep (Indep): The holy indep.
        feature_conf (omegaconf): The feature config.
        feature_inference_conf (omegaconf): The feature inference config.
        cache (dict): data cache

    Returns:
        sasa feature
    """  
    if not feature_inference_conf.active:
        return {'t1d':torch.zeros((indep.length(), feature_conf.n_bins + 1))}
    rasa = torch.full((indep.length(),), feature_inference_conf.rasa)
    one_hot = one_hot_buckets(rasa, feature_conf.low, feature_conf.high, feature_conf.n_bins)
    is_feature_applicable = indep.is_sm
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}
