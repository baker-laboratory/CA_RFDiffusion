import itertools
from collections import defaultdict

import tree
import networkx as nx
import torch
import numpy as np

from rf_diffusion import aa_model

import logging
logger = logging.getLogger(__name__)

def flatten_dictionary(dictionary, parent_key='', separator='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dictionary(value, new_key, separator))
        else:
            flattened_dict[new_key] = value
    return flattened_dict


def calc_atom_bond_loss(indep, pred_xyz, true_xyz, is_diffused, point_types, masks=None):
    """
    Loss on distances between bonded atoms
    """
    # Uncomment in future to distinguish between ligand / atomized_residue
    # is_residue = ~indep.is_sm
    # is_atomized = indep.is_sm & (indep.seq < rf2aa.chemical.NPROTAAS)
    # is_ligand = indep.is_sm & ~(indep.seq < rf2aa.chemical.NPROTAAS)
    mask_by_name = {}
    for k, v in {
        'residue': point_types == aa_model.POINT_RESIDUE,
        'atomized_sidechain': point_types == aa_model.POINT_ATOMIZED_SIDECHAIN,
        'atomized_backbone': point_types == aa_model.POINT_ATOMIZED_BACKBONE,
        'atomized': np.isin(point_types, [aa_model.POINT_ATOMIZED_BACKBONE, aa_model.POINT_ATOMIZED_SIDECHAIN]),
        'ligand': point_types == aa_model.POINT_LIGAND,
        'any': np.full(indep.length(), True),
    }.items():
        for prefix, mask in {
            'diffused': is_diffused,
            'motif': ~is_diffused,
            'any': np.full(indep.length(), True),
        }.items():
            mask_by_name[f'{prefix}_{k}'] = torch.tensor(v)*mask
    bond_losses = {}
    is_bonded = torch.triu(indep.bond_feats > 0)
    for (a, a_mask), (b, b_mask) in itertools.combinations_with_replacement(mask_by_name.items(), 2):
        if masks and f'{a}:{b}' not in masks:
            continue
        is_pair = a_mask[..., None] * b_mask[None, ...]
        is_pair = torch.triu(is_pair)
        is_bonded_pair = is_bonded * is_pair
        i, j = torch.where(is_bonded_pair)
        
        true_dist = torch.norm(true_xyz[i,1]-true_xyz[j,1],dim=-1)
        pred_dist = torch.norm(pred_xyz[i,1]-pred_xyz[j,1],dim=-1)
        bond_losses[f'{a}:{b}'] = torch.mean(torch.abs(true_dist - pred_dist))
    return bond_losses

def expand_nodes_within_one_edge(G, nodes):
    expanded_nodes = set(nodes)
    
    for node in nodes:
        neighbors = list(G.neighbors(node))
        expanded_nodes.update(neighbors)
    
    return expanded_nodes

def find_all_rigid_groups(bond_feats):
    """
    Params:
        bond_feats: torch.tensor([N, N])
    
    Returns:
        list of tensors, where each tensor contains the indices of atoms within the same rigid group.
    """
    bond_feats = bond_feats.cpu().numpy()
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    any_atom_bonds = (bond_feats != 0) * (bond_feats<5)
    G_rigid = nx.from_numpy_array(rigid_atom_bonds)
    G_bonds = nx.from_numpy_array(any_atom_bonds)
    connected_components = list(nx.connected_components(G_rigid))
    for i, cc in enumerate(connected_components):
        cc = expand_nodes_within_one_edge(G_bonds, cc)
        connected_components[i] = cc

    # Filter out rigids of 2 or fewer atoms as those with two atoms are captured by bond length metrics
    connected_components = [cc for cc in connected_components if len(cc)>2]
    connected_components = [torch.tensor(list(cc)) for cc in connected_components]
    return connected_components

def find_all_rigid_groups_human_readable(bond_feats, point_ids):
    rigid_groups = find_all_rigid_groups(bond_feats)
    return tree.map_structure(lambda i: point_ids[i], rigid_groups)

def align(xyz1, xyz2, eps=1e-6):

    # center to CA centroid
    xyz1_mean = xyz1.mean(0)
    xyz1 = xyz1 - xyz1_mean
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    return xyz2_ + xyz1_mean

def calc_rigid_loss(indep, pred_xyz, true_xyz, is_diffused, point_types):
    '''
    Params:
        indep: atomized aa_mode.Indep corresponding to the true structure
        pred_xyz: atomized xyz coordinates [L, A, 3]
    Returns:
        Dictionary mapping the composition of a rigid group to the maximum aligned RMSD of the group to the true 
        coordinates in indep.
        i.e. {'diffused_atom_motif_atom': 5.5} implies that the worst predicted rigid group which has
        at least 1 diffused atom and at least 1 motif atom has an RMSD to the corresponding set of atoms
        in the true structure of 5.5.

        The suffix '_determined' is added to groups which contain >=3 motif atoms, as all DoFs of these
        groups are determined by the motif atoms.
    '''
    rigid_groups = find_all_rigid_groups(indep.bond_feats)
    group_dists = []
    for rigid_idx in rigid_groups:
        true_ca = true_xyz[rigid_idx, 1]
        pred_ca = pred_xyz[rigid_idx, 1]
        pred_ca_aligned = align(true_ca, pred_ca)
        dist = torch.norm(pred_ca_aligned - true_ca, dim=-1)
        group_dists.append(dist)

    mask_by_name_a = {}
    for k, v in {
        'ligand': point_types == aa_model.POINT_LIGAND,
        'atomized': np.isin(point_types, [aa_model.POINT_ATOMIZED_BACKBONE, aa_model.POINT_ATOMIZED_SIDECHAIN]),
    }.items():
        for prefix, mask in {
            'diffused': is_diffused,
            'motif': ~is_diffused,
        }.items():
            mask_by_name_a[f'{prefix}_{k}'] = torch.tensor(v)*mask

    L = indep.length()
    mask_by_name_coarse = {
        'all': torch.ones(L).bool()
    }

    dist_by_composition = {}
    for prefix, mask_by_name in [
        ('fine', mask_by_name_a),
        ('coarse', mask_by_name_coarse),
    ]:
        dist_by_composition[prefix] = get_dists_by_composition(rigid_groups, group_dists, mask_by_name)

    motif_idx = torch.nonzero(~is_diffused)[...,-1]
    dist_by_composition['determined'] = []
    for rigid_idx, dists in zip(rigid_groups, group_dists):
        n_motif = np.in1d(rigid_idx, motif_idx).sum()
        is_determined = n_motif >= 3
        has_diffused = n_motif != len(rigid_idx)
        if is_determined and has_diffused:
            dist_by_composition['determined'].append(dists)

    dist_by_composition = flatten_dictionary(dist_by_composition)
    dist_by_composition = {k:v for k,v in dist_by_composition.items() if v}
    dist_by_composition = {k:torch.concat(v) for k,v in dist_by_composition.items()}
    
    out = {}
    out['max'] = {k:max(v) for k,v in dist_by_composition.items()}
    out['mean'] = {k:torch.mean(v) for k,v in dist_by_composition.items()}
    return out

def get_dists_by_composition(rigid_groups, group_dists, mask_by_name):
    mask_0 = list(mask_by_name.values())[0]
    L = len(mask_0)
    atom_types = torch.full((L,), -1)
    for i, (mask_name, mask) in enumerate(mask_by_name.items()):
        atom_types[mask] = i
    mask_names = list(mask_by_name)

    dist_by_composition =  defaultdict(list)
    for rigid_idx, dists in zip(rigid_groups, group_dists):
        composition = atom_types[rigid_idx].tolist()
        composition = tuple(sorted(list(set(composition))))
        composition_string = ':'.join(mask_names[e] for e in composition)
        dist_by_composition[composition_string].append(dists)

    
    return dist_by_composition
    
