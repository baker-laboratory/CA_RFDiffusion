import sys
import torch
import numpy as np
import string
import gzip
import rf2aa.util
from rf_diffusion.chemical import ChemicalData as ChemData
import rf2aa.data.parsers
import rf_diffusion.error

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)
    
    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file

def parse_pdb(filename, xyz27=False,seq=False):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines_target(lines, xyz27, seq)

def parse_pdb_lines_target(lines, parse_hetatom=False, ignore_het_h=True):

    def first_atom_iter():
        pdb_idx_set = set()
        for l in lines:
            if l[:4]!="ATOM":
                continue
            chain, residue_index = ( l[21:22].strip(), int(l[22:26].strip()) )
            if (chain, residue_index) not in pdb_idx_set:
                pdb_idx_set.add((chain, residue_index))
                yield l

    res = [(l[22:26],l[17:20]) for l in first_atom_iter()]
    seq = [ChemData().aa2num[r[1]] if r[1] in ChemData().aa2num.keys() else 20 for r in res]
    pdb_idx = [(l[21:22].strip(), int(l[22:26].strip())) for l in first_atom_iter()]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), ChemData().NHEAVY, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
        for i_atm, tgtatm in enumerate(ChemData().aa2long[ChemData().aa2num[aa]][:ChemData().NHEAVY]): # Nate's proposed change            
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    out = {
        'xyz':xyz, # cartesian coordinates, [Lx14]
        'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
        'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
        'seq':np.array(seq), # amino acid sequence, [L]
        'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            with rf_diffusion.error.context(l):
                if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                    info_het.append(dict(
                        idx=int(l[7:11]),
                        atom_id=l[12:16],
                        atom_type=l[77],
                        name=l[17:20],
                        res_idx=int(l[22:26]),
                    ))
                    xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out

def load_ligand_from_pdb(fn, lig_name=None, remove_H=True):
    """Loads a small molecule ligand from pdb file `fn` into feature tensors.
    If no ligand is found, returns empty tensors with the same dimensions as
    usual.

    PDB format: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html

    Parameters
    ----------
    fn : str
        Name of PDB file
    lig_name : str
        3-letter residue name of ligand to load. If None, assumes
        there is only 1 ligand and loads it from all HETATM lines.
    remove_H : bool
        If True, does not load H atoms

    Returns
    -------
    xyz_sm : torch.Tensor (N_symmetry, L_sm, 3)
        Atom coordinates of ligand
    mask_sm : torch.Tensor (N_symmetry, L_sm)
        Boolean mask for whether atoms exist
    msa_sm : torch.Tensor (L_sm,)
        Integer-encoded (rf2aa.chemical) sequence (atom types) of ligand.
    bond_feats_sm : torch.Tensor (L_sm, L_sm)
        Bond features for ligand
    idx_sm : torch.Tensor (L_sm,)
        Residue number for ligand (all the same)
    atom_names : list of str
        Atom names of ligand (including whitespace) from columns 13-16 of
        PDB HETATM lines.
    """
    with open(fn, 'r') as fh:
        stream = [l for l in fh
                  if (("HETATM" in l) and (lig_name is None or l[17:20].strip()==lig_name))\
                     or "CONECT" in l]

    if len(stream)==0:
        sys.exit(f'ERROR (load_ligand_from_pdb): no HETATM records found in file {fn}.')

    mol, msa_sm, ins_sm, xyz_sm, mask_sm = \
        rf2aa.data.parsers.parse_mol("".join(stream), filetype="pdb", string=True, remove_H=remove_H,
                                find_automorphs=False)
    bond_feats_sm = rf2aa.util.get_bond_feats(mol)

    atom_names = []
    for line in stream:
        if line.startswith('HETATM'):
            atom_type = line[76:78].strip()
            if atom_type == 'H' and remove_H:
                continue
            atom_names.append(line[12:16])

    return mol, xyz_sm, mask_sm, msa_sm, bond_feats_sm, atom_names


def load_ligands_from_pdb(fn, lig_names=None, remove_H=True):
    xyz_stack = []
    mask_stack = []
    msa_stack = []
    bond_feats_stack = []
    atom_names_stack = []
    ligand_names_arr = []
    for ligand in lig_names:
        mol, xyz, mask, msa, bond_feats, atom_names = load_ligand_from_pdb(fn, ligand, remove_H=remove_H)
        xyz = xyz[0]
        xyz_stack.append(xyz)
        mask_stack.append(mask)
        msa_stack.append(msa)
        bond_feats_stack.append(bond_feats)
        atom_names_stack.append(atom_names)
        L = xyz.shape[0]
        ligand_names_arr.extend([ligand]*L)
    
    xyz = torch.cat(xyz_stack)
    mask = torch.cat(mask_stack, dim=1)
    msa = torch.cat(msa_stack)
    bond_feats = torch.block_diag(*bond_feats_stack)
    atom_names = []
    for a in atom_names_stack:
        atom_names.extend(a)
    return xyz[None,...], mask, msa, bond_feats, atom_names, np.array(ligand_names_arr)
