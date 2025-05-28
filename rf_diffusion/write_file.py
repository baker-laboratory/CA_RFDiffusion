from collections import defaultdict

import numpy as np
import torch

from rf_diffusion.chemical import ChemicalData as ChemData
from rf2aa.scoring import *
from rf_diffusion import build_coords

def writepdb(filename, *args, file_mode='w', **kwargs, ):
    f = open(filename, file_mode)
    names = writepdb_file(f, *args, **kwargs)
    f.close()
    return names

def writepdb_file(f, atoms, seq, modelnum=None, chain="A", idx_pdb=None, bfacts=None, 
             bond_feats=None, file_mode="w",atom_mask=None, atom_idx_offset=0, chain_Ls=None,
             remap_atomtype=True, lig_name=None, atom_names=None, chain_letters=None,
             ligand_name_arr=None, fix_corrupt_sidechains=True):
    
    # PDBs coordinate range is (-999.99, 9999.99)
    atoms = torch.where(atoms >= 10000, 9999.99, atoms)
    atoms = torch.where(atoms <= -1000, -999.99, atoms)

    atom_count_by_res = defaultdict(lambda: defaultdict(int))
    ligand_count = 0

    def _get_atom_type(atom_name):
        atype = ''
        if atom_name[0].isalpha():
            atype += atom_name[0]
        atype += atom_name[1]
        return atype

    # if needed, correct mistake in atomic number assignment in RF2-allatom (fold&dock 3 & earlier)
    atom_names_ = [
        "F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P",  "As", "Sb",
        "C",  "Si", "Ge", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", 
        "Pd", "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", 
        "Mo", "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]
    atom_num = [
        9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51,
        6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28,
        46,   78,   27,   45,   77,   59,   26,   44,   76,   25,   75,   24,   
        42,   74,   23,   92,   65,   39,   4,    12,   20,   3,    19,   0] 
    atomnum2atomtype_ = dict(zip(atom_num,atom_names_))
    if remap_atomtype:
        atomtype_map = {v:atomnum2atomtype_[k] for k,v in ChemData().atomnum2atomtype.items()}
    else:
        atomtype_map = {v:v for k,v in ChemData().atomnum2atomtype.items()} # no change
        
    ctr = 1+atom_idx_offset
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    assert not (chain_Ls and chain_letters)
    if chain_letters is None:
        if chain_Ls is not None:
            chain_letters = np.concatenate([np.full(L, alphabet[i]) for i,L in enumerate(chain_Ls)])
        else:
            chain_letters = np.array([chain]*len(scpu)) # torch can't handle strings apparently

    assert not ((ligand_name_arr is not None) and lig_name)
    if ligand_name_arr is None:
        lig_name = lig_name or 'LG1'
        ligand_name_arr = np.full(len(scpu), lig_name)
        
    if modelnum is not None:
        f.write(f"MODEL        {modelnum}\n")

    if fix_corrupt_sidechains and atoms.shape[1] > 4:
        res_fixed = fix_null_sidechains(atoms, seq, atom_mask=atom_mask).numpy()
        if res_fixed.any():
            # The strange slicing on the next line is because this writepdb_file doesnt enforce that idx_pdb is the same length as xyz
            print('Building fake sidechains for positions:', ','.join(f'{chain}{seqpos}' for (chain, seqpos) in 
                                                            zip(chain_letters[:len(res_fixed)][res_fixed], idx_pdb[:len(res_fixed)][res_fixed])))

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    atom_idxs = {}
    i_res_lig = 0
    names = []
    writing_ligand = False
    for i_res,s,ch,lig in zip(range(len(scpu)), scpu, chain_letters, ligand_name_arr):
        
        natoms = atomscpu.shape[-2]
        if s >= len(ChemData().aa2long):
            writing_ligand = True
            atom_idxs[i_res] = ctr

            # hack to make sure H's are output properly (they are not in RFAA alphabet)
            res_idx = int((idx_pdb.max()+10 * (ligand_count + 1)))
            if atom_names is not None:
                if s <= len(ChemData().num2aa):
                    atom_type = atomtype_map.get(ChemData().num2aa[s])
                else:
                    atom_type = _get_atom_type(atom_names[i_res_lig])
                atom_name = atom_names[i_res_lig]
            else:
                atom_type = atomtype_map[ChemData().num2aa[s]]
                atom_count_by_res[res_idx][atom_type] += 1
                atom_name_suffix = atom_count_by_res[res_idx][atom_type]
                atom_name = f'{atom_type}{atom_name_suffix}'

            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %+2s\n"%(
                    "HETATM", ctr, atom_name, lig,
                    ch, res_idx, atomscpu[i_res,1,0], atomscpu[i_res,1,1], atomscpu[i_res,1,2],
                    1.0, Bfacts[i_res],  atom_type) )
            i_res_lig += 1
            name = f'id {ctr}'
            ctr += 1
        else:
            if writing_ligand:
                ligand_count += 1
            writing_ligand = False
            atms = ChemData().aa2long[s]

            max_atm_index = max(idx_pdb)
            if max_atm_index > 9999:
                raise Exception('PDB residue index overflow')
            for i_atm,atm in enumerate(atms):
                if atom_mask is not None and not atom_mask[i_res,i_atm]: continue # skip missing atoms
                if (i_atm<natoms and atm is not None and not torch.isnan(atomscpu[i_res,i_atm,:]).any()):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm, ChemData().num2aa[s],
                        ch, idx_pdb[i_res], atomscpu[i_res,i_atm,0], atomscpu[i_res,i_atm,1], atomscpu[i_res,i_atm,2],
                        1.0, Bfacts[i_res] ) )
                    ctr += 1
            name = f'resi {idx_pdb[i_res]}'
        names.append(name)
    if bond_feats is not None:
        atom_bonds = (bond_feats > 0) * (bond_feats <5)
        atom_bonds = atom_bonds.cpu()
        b, i, j = atom_bonds.nonzero(as_tuple=True)
        for start, end in zip(i,j):
            f.write(f"CONECT{atom_idxs[int(start.cpu().numpy())]:5d}{atom_idxs[int(end.cpu().numpy())]:5d}\n")
    if modelnum is not None:
        f.write("ENDMDL\n")
    return np.array(names)




def fix_null_sidechains(atoms, seq, atom_mask=None, too_close_thresh=0.01, flag_atoms_exactly_0000=True):
    '''
    If sidechains atoms of a protein residue are xyz identical. Replace with an ideal sidechain

    Future iterations of this function could also fix dna and perhaps try to match correctly-built atoms if any exist

    Args:
        atoms (torch.Tensor[float]): xyz from an indep [L,4]
        seq (torch.Tensor[int]): seq from an indep [L]
        atom_mask (torch.Tensor[bool] or None): Which atoms exist for each residue? [L,36]
        too_close_thresh (float): Any any two sidechain atoms are closer than this rebuild the sidechain
        flag_atoms_exactly_0000 (bool): If a sidechain atom has coordinates exactly 0.000 then call the sidechain bad (catches alanine)

    Returns:
        res_fixed (torch.tensor[bool]): The mask of residues that were rebuilt [L]
    '''

    assert atoms.shape[1] >= 4, "You're trying to build sidechains on a backbone that doesn't even have oxygens. Something is probably wrong."

    too_close_squared = too_close_thresh**2

    needs_fix = torch.zeros(len(atoms), dtype=bool)
    atom_mask_of_fixes = []

    for pos in range(len(atoms)):

        res_atoms = atoms[pos]
        res_seq = seq[pos]

        # Only deal with protein sidechains for now
        if res_seq >= 20:
            continue

        # Sanity check so we don't crash when people output truly cursed proteins
        # Make sure that N, CA, C are valid
        if torch.isnan(res_atoms[:2]).any():
            continue
        if torch.sum(torch.square(res_atoms[0]-res_atoms[1])) < too_close_squared:
            continue
        if torch.sum(torch.square(res_atoms[1]-res_atoms[2])) < too_close_squared:
            continue

        # Figure out what the atom mask should be. We choose the input with the fewest number of atoms
        atom_names = ChemData().aa2long[res_seq]
        true_atom_mask = torch.tensor([name is not None for name in atom_names])

        in_atom_mask = true_atom_mask if atom_mask is None else atom_mask[pos]

        natoms = min([len(true_atom_mask), len(in_atom_mask), len(res_atoms)])
        final_atom_mask = true_atom_mask[:natoms] & in_atom_mask[:natoms] & ~torch.isnan(res_atoms[:natoms]).any(axis=-1)

        # We don't care about backbone corruption since only O can be salvaged and you'd fix that differently
        final_atom_mask[:4] = False
        final_atom_mask[1] = True # But put CA back in there to catch atoms overlapping with CA

        # GLY
        if final_atom_mask.sum() == 0:
            continue

        # Check for overlapping atoms
        all_by_d2 = torch.sum( torch.square( res_atoms[final_atom_mask][:,None] - res_atoms[final_atom_mask][None,:] ), axis=-1 )
        all_by_d2[torch.eye(final_atom_mask.sum(), dtype=bool)] = too_close_squared + 1

        # Strip CA from this calculation
        exactly_origin = (torch.abs( res_atoms[final_atom_mask][1:] ) < 0.0001).all(axis=-1)

        # if they're all separated and no one is at the origin we're good
        if (all_by_d2 > too_close_squared).all() and not exactly_origin.any():
            continue

        # Mark the residue as needs fixing and store its atom mask
        needs_fix[pos] = True
        atom_mask_of_fixes.append(final_atom_mask)


    if needs_fix.sum() == 0:
        return needs_fix

    atom_mask_of_fixes = torch.stack(atom_mask_of_fixes, axis=0)
    atom_mask_of_fixes[:,:4] = False # Don't mess with the backbone

    # Generate ideal sidechain coordinates (ideal in the sense that they look good and don't clash with themselves)
    xyz_ideal = build_coords.build_ideal_sidechains(atoms[needs_fix], seq[needs_fix])
    xyz_ideal = xyz_ideal[:,:atom_mask_of_fixes.shape[-1]]

    # Store the ideal coordinates back into the atoms
    expanded_fix_mask = torch.zeros((len(atoms), atom_mask_of_fixes.shape[1]), dtype=bool)
    expanded_fix_mask[needs_fix] = atom_mask_of_fixes
    atoms[expanded_fix_mask] = xyz_ideal[atom_mask_of_fixes]

    return needs_fix
