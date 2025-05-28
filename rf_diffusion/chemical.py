from dataclasses import dataclass
import rf2aa.chemical
import torch

@dataclass
class ChemConf:
    use_phospate_frames_for_NA: bool
    use_lj_params_for_atoms: bool

@dataclass
class ChemConfConf:
    chem_params: ChemConf

def initialize_chemical_data(use_phospate_frames_for_NA: bool = True, use_lj_params_for_atoms: bool = False):
    chem_conf = ChemConf(
        use_phospate_frames_for_NA,
        use_lj_params_for_atoms
    )
    chem_conf_conf = ChemConfConf(chem_conf)
    rf2aa.chemical.initialize_chemdata(chem_conf_conf)
    return rf2aa.chemical.ChemicalData

# Default initialization
ChemicalData = initialize_chemical_data()

# Function to reinitialize with custom parameters
def reinitialize_chemical_data(use_phospate_frames_for_NA: bool = True, use_lj_params_for_atoms: bool = False):
    global ChemicalData
    ChemicalData = initialize_chemical_data(use_phospate_frames_for_NA, use_lj_params_for_atoms)

_residue_bond_feats = None
def get_residue_bond_feats():
    '''
    Generates per-residue bond_feats matrices like for small molecules

    This could really just be another field of ChemData

    Returns:
        residue_bond_feats (list[torch.tensor[int,int]]): A matrix for each residue [30,36,36]
    '''
    global _residue_bond_feats
    if _residue_bond_feats is not None:
        return _residue_bond_feats

    _residue_bond_feats = torch.zeros((len(ChemicalData().aabonds), ChemicalData().NTOTAL, ChemicalData().NTOTAL), dtype=int)

    for i_aa in range(len(_residue_bond_feats)):
        # Don't worry about the mask tokens
        if i_aa in [ChemicalData().UNKINDEX, ChemicalData().MASKINDEX, ChemicalData().MASKINDEXDNA, ChemicalData().MASKINDEXRNA]:
            continue
        atom_names = ChemicalData().aa2long[i_aa]
        bonds_left = torch.tensor([atom_names.index(a) for a, b in ChemicalData().aabonds[i_aa]])
        bonds_right = torch.tensor([atom_names.index(b) for a, b in ChemicalData().aabonds[i_aa]])

        bond_orders = torch.zeros((ChemicalData().NTOTAL, ChemicalData().NTOTAL), dtype=int)

        bond_orders[(bonds_left, bonds_right)] = torch.tensor(ChemicalData().aabtypes[i_aa])
        bond_orders[(bonds_right, bonds_left)] = torch.tensor(ChemicalData().aabtypes[i_aa])
                
        _residue_bond_feats[i_aa] = bond_orders

    return _residue_bond_feats


def missing_heavy_atoms(xyz, seq, nan_is_missing=True, origin_is_missing=True):
    '''
    Returns an atom mask of the locations where this xyz is missing heavy atoms

    Args:
        xyz (torch.Tensor[float]): The xyz coordinates [...,>=NHEAVY,3]
        seq (torch.Tensor[int]): The sequence [...]
        nan_is_missing (bool): Does nan count as being missing?
        origin_is_missing (bool): Does being at the origin count as being missing?

    Returns:
        missing_heavy_atoms (torch.Tensor[bool]): Which atoms are missing [...,NHEAVY]
    '''

    xyz = xyz[:,:ChemicalData().NHEAVY]

    xyz_is_origin = (xyz == 0).all(axis=-1)
    xyz_is_nan = torch.isnan(xyz).any(axis=-1)

    xyz_is_missing = torch.zeros(xyz.shape[:-1], dtype=bool)

    if nan_is_missing:
        xyz_is_missing |= xyz_is_nan
    if origin_is_missing:
        xyz_is_missing |= xyz_is_origin

    expecting_atom = ChemicalData().allatom_mask[seq][:,:ChemicalData().NHEAVY]

    return xyz_is_missing & expecting_atom
