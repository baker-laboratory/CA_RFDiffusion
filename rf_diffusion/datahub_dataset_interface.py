from biotite.structure import AtomArray
import numpy as np
import torch
import pandas as pd
from typing import Any
from pathlib import Path

from datahub.transforms.base import Transform
from datahub.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from datahub.transforms.bonds import AddTokenBondAdjacency, AddRF2AABondFeaturesMatrix
from datahub.utils.token import get_token_starts
from datahub.transforms.encoding import atom_array_to_encoding
from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import AddProteinTerminiAnnotation
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.utils.token import spread_token_wise
from datahub.datasets.parsers.base import MetadataRowParser
from rf_diffusion.conditions.util import pop_conditions_dict

from rf_diffusion.aa_model import Indep, N_TERMINUS, C_TERMINUS, is_occupied, pop_mask, deatomize_covales
class BackwardCompatibleDataLoaderProcessOut(Transform):
    requires_previous_transforms = [
        FlagNonPolymersForAtomization,
        AddProteinTerminiAnnotation,
        AtomizeByCCDName,
        AddTokenBondAdjacency,
        AddRF2AABondFeaturesMatrix
    ]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "rf2aa_bond_features_matrix", "extra_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            ["is_polymer", "pn_unit_iid", "molecule_iid", "molecule_entity", "chain_entity", "chain_iid", "atomize", "is_N_terminus", "is_C_terminus"],
        )
        check_atom_array_annotation(
            data,
            self.conditions_dict_atom_array_keys,
        )

    def __init__(
        self,
        sel_item_keys=['example_id', 'pdb_id', 'assembly_id', 'cluster'],
        task='diff',
        conditions_dict_atom_array_keys=[],
    ):
        self.sel_item_keys = sel_item_keys
        self.task = task
        self.conditions_dict_atom_array_keys = conditions_dict_atom_array_keys

    def _get_indep_and_atom_mask(self, data):
        encoded = atom_array_to_encoding(data['atom_array'], RF2AA_ATOM36_ENCODING)

        token_starts = get_token_starts(data['atom_array'])
        token_wise_atom_array = data['atom_array'][token_starts]

        _make_is_same_matrix = lambda x: x.unsqueeze(0) == x.unsqueeze(0).T  # noqa
        is_same_molecule_iid = _make_is_same_matrix(
            torch.from_numpy(token_wise_atom_array.molecule_iid.astype(np.int64))
        )

        terminus_type = torch.zeros(len(encoded['seq']), dtype=int)
        terminus_type[token_wise_atom_array.is_N_terminus] = N_TERMINUS
        terminus_type[token_wise_atom_array.is_C_terminus] = C_TERMINUS

        indep = Indep(
            torch.tensor(encoded['seq']),
            torch.tensor(encoded['xyz']),
            torch.arange(len(encoded['seq'])),
            data['rf2aa_bond_features_matrix'].to(torch.int64),
            data['chiral_feats'],
            is_same_molecule_iid,
            terminus_type
        )

        # Build the idx of the indep using the standard diffusion notation of +33 for chainbreaks
        # This code will respect gaps in the input numbering

        # idx is easier to do once we have access to chain_masks
        CHAIN_GAP = 33
        original_resid = token_wise_atom_array.res_id
        idx = torch.full((indep.length(),), -CHAIN_GAP, dtype=int)
        for sm_pass in [False, True]:
            for mask in indep.chain_masks():
                mask = torch.tensor(mask)
                mask &= sm_pass == indep.is_sm
                if mask.sum() == 0:
                    continue
                if sm_pass:
                    original_idx = torch.arange(mask.sum(), dtype=int) # small molecules get arange numbering no matter what
                else:
                    original_idx = torch.tensor(original_resid[mask])
                # Assign the numbering as 33 past the end of the previously largest idx
                idx[mask] = original_idx - original_idx.min() + idx.max() + CHAIN_GAP
        assert (idx >= 0).all()
        indep.idx = idx

        return indep, torch.tensor(encoded['mask'])

    def _get_conditions_dict(self, data):
        '''
        Use conditions_dict_atom_array_keys to fill conditions_dict with tensor variables

        The door is wide-open here to add scalar and 2d masks but I'll let whoever needs that implement it
        '''
        token_starts = get_token_starts(data['atom_array'])
        token_wise_atom_array = data['atom_array'][token_starts]

        conditions_dict = {}
        for key in self.conditions_dict_atom_array_keys:
            conditions_dict[key] = torch.tensor(getattr(token_wise_atom_array, key))

        return conditions_dict


    def forward(self, data: dict) -> dict:
        mask_gen_seed = np.random.randint(0, 99999999)

        indep, atom_mask = self._get_indep_and_atom_mask(data)
        conditions_dict = self._get_conditions_dict(data)

        pop = is_occupied(indep, atom_mask)
        # For now, do not pop unoccupied small molecule atoms, exit instead, as popping them can lose covale information.
        unoccupied_sm = (~pop) * indep.is_sm
        if unoccupied_sm.any():
            raise Exception(f'there are small molecule atoms that are unoccupied at indices:  {unoccupied_sm.nonzero()[:,0]}')

        pop_mask(indep, pop)
        atom_mask = atom_mask[pop]
        pop_conditions_dict(conditions_dict, pop)

        # if self.conf.dataloader.max_residues > -1:
        #     # Kind of hacky as it may break covales.  Only for debugging.
        #     pop = indep.is_sm.clone()
        #     residue_indices = torch.where(~indep.is_sm)[0]
        #     residue_indices = residue_indices[:self.conf.dataloader.max_residues]
        #     pop[residue_indices] = True
        #     pop_mask(indep, pop)
        #     atom_mask = atom_mask[pop]

        indep, atom_mask, metadata = deatomize_covales(indep, atom_mask)

        sel_item = {key: data['extra_info'][key] for key in self.sel_item_keys if key in data['extra_info']}
        
        return {
            'indep': indep,
            'atom_mask': atom_mask,
            'metadata': metadata,
            'sel_item': sel_item,
            'task': self.task,
            'mask_gen_seed': mask_gen_seed,
            'conditions_dict': conditions_dict
        }

class ReorderChains(Transform):
    '''
    A datahub transform to transpose the order of chains based on the 'reorder_chains' field

    The string 'None' causes this to do nothing
    Otherwise the string should contain comma separated values starting at 0 to describe a new order
    Examples:
        0,1 -- Don't actually change the order
        1,0 -- Flip/flop a 2-chain examples
        2,1,0 -- Reverse a 3-chain example

    See config/training/debug_dhub.yaml for example usage
    '''

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_contains_keys(data['extra_info'], ["reorder_chains"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            ["chain_iid"],
        )

    def __init__(self):
        pass

    def forward(self, data: dict) -> dict:

        # Check for magic string None and return if found
        reorder_string = data['extra_info']['reorder_chains']
        if reorder_string == 'None':
            return data

        # Get the new chain order
        reorder_map = torch.tensor([int(x) for x in reorder_string.split(',')])

        # Figure out how many chains we have
        known_chain_iids = dict() # Sets aren't ordered which is the dumbest thing ever
        for iid in data['atom_array'].chain_iid:
            known_chain_iids[iid] = True

        known_chain_iids = list(known_chain_iids)

        assert len(reorder_map) == len(known_chain_iids), f"ReorderChains reorder_chains length ({reorder_string}) doesn't match chain_iids ({known_chain_iids})"

        # Make our own chain masks
        chain_masks = []
        for iid in known_chain_iids:
            chain_masks.append( data['atom_array'].chain_iid == iid )

        # AtomArrays behave like lists so just += them
        new_atom_array = data['atom_array'][chain_masks[reorder_map[0]]]
        for i_mask in reorder_map[1:]:
            new_atom_array += data['atom_array'][chain_masks[i_mask]]

        # Store modified AtomArray
        data['atom_array'] = new_atom_array

        return data


class Load1DAtomArrayFeatures(Transform):
    '''
    A datahub transform to load 1D features stored in your parquet into the AtomArray at a per-residue level

    Features are specified via Load1DAtomArrayFeatures.features

    Often used with build_rf_diffusion_transform_pipeline.conditions_dict_atom_array_keys to get these values into conditions_dict

    See config/training/debug_dhub.yaml for example usage
    '''
    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "extra_info"])
        check_contains_keys(data['extra_info'], self.features)
        check_is_instance(data, "atom_array", AtomArray)


    def __init__(self, features=[]):
        '''
        Args:
            features (list[str]): The list of columns from your parquet to turn into AtomArray fields
        '''
        self.features = features

    def forward(self, data: dict) -> dict:

        # Figure out the "residues"
        token_starts = get_token_starts(data['atom_array'])
        token_wise_atom_array = data['atom_array'][token_starts]

        # Load each feature into the atom array
        for key in self.features:
            to_store = data['extra_info'][key]
            assert len(to_store) == len(token_wise_atom_array), (f'Load1DAtomArrayFeatures length mismatch: atom_array has {len(token_starts)} tokens'
                                f' but len({key}) is {len(to_store)}.')

            data['atom_array'].set_annotation(key, spread_token_wise(data['atom_array'], to_store))

        return data

class SimplePathParser(MetadataRowParser):
    """
    A straightforward way to load PDBs with datahub

    In addition to standard fields (example_id, path), this parser also includes:
        - Any extra information from the DataFrame, which is stored in the `extra_info` field.
    """

    def __init__(self):
        pass

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        assert 'example_id' in row
        assert 'path' in row

        # Put the full row in the extra info dictionary
        extra_info = row.to_dict()

        if 'assembly_id' in row:
            assembly_id = row['assembly_id']
        else:
            assembly_id = "1"

        return {
            "example_id": row["example_id"],
            "path": Path(row['path']),
            "extra_info": extra_info,
            "assembly_id": assembly_id
        }


