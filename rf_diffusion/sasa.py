from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import io
import numpy as np
from Bio import PDB
import networkx as nx
import assertpy
import torch
from collections import defaultdict
import Bio.PDB.SASA
# Monkey patch for bug in Biopython
monkey_patch_atomic_radii = defaultdict(lambda: 2.0)
monkey_patch_atomic_radii.update(Bio.PDB.SASA.ATOMIC_RADII)
Bio.PDB.SASA.ATOMIC_RADII = monkey_patch_atomic_radii

import tree
np.int = np.int64

from rf_diffusion import aa_model

p = PDBParser(PERMISSIVE=0, QUIET=1)

def get_sasa_indep(indep, probe_radius=1.4):

    # Biopython can't handle duplicate ligand names
    ligand_name_arr = np.full((indep.length(),), 'LG0')
    number_alpha = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for ichain, chain_mask in enumerate(indep.chain_masks()):
        chain_lig_name = f'LG{number_alpha[ichain]}'
        ligand_name_arr[chain_mask] = chain_lig_name

    buffer = io.StringIO()
    names = indep.write_pdb_file(buffer, ligand_name_arr=ligand_name_arr)
    buffer.seek(0)
    return get_sasa_pdb_file(buffer, probe_radius=probe_radius), names

def get_sasa_pdb_file(pdb_file, probe_radius=1.4):
    struct = p.get_structure('none', pdb_file)
    sr = ShrakeRupley(probe_radius=probe_radius)
    sr.compute(struct, level="A")
    return struct

def translate_biopython_res_list(indep, names, res_list):
    '''
    Translate the results of aa_model.write_pdb and biopython's parser into a coherent mapping from biopython to indep

    Args:
        indep (indep): Indep
        names (list[str]): The names return argument from indep.write_pdb
        res_list (list[Bio.PDB.Residue.Residue]): The return value from PDB.Selection.unfold_entities(..., target_level='R')

    Returns:
        res_list_to_indep_i (list[list]): A mapping from res_list to either a single indep residue or a list of indep is_sm atoms
    '''

    assert len(names) == indep.length()
    assert len(torch.unique(indep.idx)) == indep.length(), 'Indep has duplicate idx'

    # We are going to assume that the names coming out of write_pdb match the order in the file
    # But not necessarily that they match the order of the indep because Magnus wants to change that
    res_list_to_indep_i = []
    idx_list = list(indep.idx)
    iname = 0
    ireslist = 0
    i_sm = 0
    wh_sm = torch.where(indep.is_sm)[0]
    while iname < len(names) and ireslist < len(res_list):

        # Get the res and the name and make sure they jive
        res = res_list[ireslist]
        res_is_protein = res.id[0].strip() == '' # kinda sketchy, see https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
        name = names[iname]
        name_is_protein = name.split()[0] == 'resi'
        assert res_is_protein == name_is_protein, 'Bad assumption. write_pdb.names != biopython read order'

        if res_is_protein:
            # If it's a protein residue. Figure out the indep_i
            resi = int(name.split()[1])
            assert resi in idx_list, f"write_pdb wrote a residue index that doesn't exist: {name}"
            i_indep = idx_list.index(resi)
            assert res.id[1] == indep.idx[i_indep], f"biopython residue number didn't match indep. {res.id[1]} != {indep.idx[i_indep]}"

            # Store the result and increment the counters
            res_list_to_indep_i.append([i_indep])
            iname += 1
            ireslist += 1
        else:
            # Now we're dealing with a small molecule
            res_natoms = len(res.child_list)
            this_list = []
            for i in range(res_natoms):
                name = names[iname]
                assert name.startswith('id'), 'Bad assumption. write_pdb.names != biopythonread order'
                this_list.append(wh_sm[i_sm])
                i_sm += 1
                iname += 1
            res_list_to_indep_i.append(this_list)
            ireslist += 1

    assert len(res_list_to_indep_i) == len(res_list)
    assert iname == len(names)
    assert ireslist == len(res_list)

    return res_list_to_indep_i

def get_indep_sasa_per_res(indep, probe_radius=1.4):
    '''
    Uses biopython to calculate the SASA of each residue in indep

    Singular atoms are treated as individual residues

    Args:
        indep (indep): Indep. No guideposts please!
        probe_radius (float): Radius of water molecule

    Returns:
        per_res_sasa (torch.Tensor[bool]): SASA of each residue
    '''

    struct, names = get_sasa_indep(indep, probe_radius=probe_radius)
    res_list = PDB.Selection.unfold_entities(struct, target_level='R')

    res_list_to_indep_i = translate_biopython_res_list(indep, names, res_list)


    # Sum up the SASA of each residue
    per_res_sasa = torch.full((indep.length(),), np.nan)

    for ireslist, indep_i_list in enumerate(res_list_to_indep_i):
        res = res_list[ireslist]
        if len(indep_i_list) == 1:
            per_res_sasa[indep_i_list[0]] = sum([child.sasa for child in res.child_list])
        else:
            for iatom, indep_i in enumerate(indep_i_list):
                per_res_sasa[indep_i] = res.child_list[iatom].sasa


    assert not torch.isnan(per_res_sasa)[0].any(), f'Some residue sasas not calculated! {torch.where(torch.isnan(per_res_sasa)[0])}'

    return per_res_sasa

def small_molecules(indep):
    e = indep.bond_feats.detach().cpu().clone().numpy()
    e[~indep.is_sm] = 0
    e[:,~indep.is_sm] = 0
    e *= (e != aa_model.GP_BOND)
    G = nx.from_numpy_matrix(e)
    cc = list(nx.connected_components(G))
    cc.sort(key=min)
    o = [np.array([], dtype=np.int32)]
    for c in cc:
        c = np.array(list(c))
        is_sm = indep.is_sm[c]
        assertpy.assert_that(is_sm.unique()).is_length(1)
        if is_sm[0]:
            o.append(c)
    return o

def get_max_sasa(atom, probe_radius=1.4):
    return 4 * np.pi * (Bio.PDB.SASA.ATOMIC_RADII[atom.element] + probe_radius)**2

def get_relative_sasa_sm(indep):
    sm_sasa, sm_atoms, sm = get_sm_sasa(indep, probe_radius=1.40)
    sm_max_sasa = tree.map_structure(get_max_sasa, sm_atoms)
    assertpy.assert_that(len(sm_max_sasa)).is_equal_to(len(sm_sasa))
    sm_relative_sasa = []
    for sasa, max_sasa in zip(sm_sasa, sm_max_sasa):
        sm_relative_sasa.append(sasa / max_sasa)
    return sm, sm_atoms, sm_relative_sasa, sm_sasa, sm_max_sasa

    
def get_sm_sasa(indep, probe_radius=1.40):
    struct, names = get_sasa_indep(indep, probe_radius=probe_radius)
    atomList = PDB.Selection.unfold_entities(struct, target_level='A')
    atom_by_id = {atom.serial_number:atom for atom in atomList}
    sm = small_molecules(indep)
    sm_atom_ids = []
    for sm_i in sm:
        sm_names = names[sm_i]
        for n in sm_names:
            assertpy.assert_that(n).starts_with('id ')
        sm_atom_ids.append([int(n.split()[1]) for n in sm_names])
    sm_atoms = []
    for atom_ids in sm_atom_ids:
        sm_atoms.append([atom_by_id[i] for i in atom_ids])
    
    sm_atom_sasa = []
    for atoms in sm_atoms:
        sm_atom_sasa.append(np.array([atom.sasa for atom in atoms]))
    return sm_atom_sasa, sm_atoms, sm

def get_relative_sasa(indep):
    sm, _, sm_relative_sasa, _, _ = get_relative_sasa_sm(indep)
    flat_sm_indices = np.concatenate(sm)
    flat_relative_sasa = np.concatenate(sm_relative_sasa)
    relative_sasa = torch.full((indep.length(),), 0.0)
    relative_sasa[flat_sm_indices] = torch.tensor(flat_relative_sasa).float()
    return relative_sasa

def noised_relative_sasa(indep, std_std):
    sasa = get_relative_sasa(indep)
    std = torch.zeros((indep.length(),))
    if np.random.random() < 0.5:
        std[:] = torch.abs(torch.normal(0.0, std_std, (1,)))
    else:
        std = torch.abs(torch.normal(0.0, std_std, std.shape))
    sasa = torch.normal(sasa, std)
    sasa[~indep.is_sm] = -10
    return sasa, std
