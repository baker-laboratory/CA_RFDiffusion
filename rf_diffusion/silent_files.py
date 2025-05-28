import torch
import os
import io
import numpy as np
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion import aa_model
from rf_diffusion import write_file
from rf_diffusion.import_pyrosetta import pyrosetta as pyro
from rf_diffusion.import_pyrosetta import rosetta as ros

import rf2aa.util

def add_indep_to_silent(silent_path, tag, indep, scores={}, **kwargs):
    '''
    Add this indep to a silent file

    Args:
        silent_path (str): Path to silent file
        tag (str): Name of this design
        indep (indep): indep
        scores (dict[float or str]): scores to be added to the silent file's internal scorefile
        **kwargs: Additional args to be passed to write_file.writepdb_file
    '''

    chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain)
    add_xyz_to_silent(silent_path, tag, indep.xyz, indep.seq, indep.bond_feats, idx_pdb=indep.idx, chain_Ls=chain_Ls, scores=scores, **kwargs)


def add_xyz_to_silent(silent_path, tag, xyz, seq, bond_feats, scores={}, **kwargs):
    '''
    Add this structure to a silent file

    Args:
        silent_path (str): Path to silent file
        tag (str): Name of this design
        xyz (torch.Tensor): xyz of this structure [L,3]
        bond_feats (torch.Tensor): the bond feats from indep [L,L]
        scores (dict[float or str]): scores to be added to the silent file's internal scorefile
        **kwargs: Additional args to be passed to write_file.writepdb_file
    '''

    # First turn the coordinates into a pdb
    xyz_stack = xyz[None]
    xyz23 = aa_model.pad_dim(xyz_stack, 2, ChemData().NHEAVY)[:,:,:ChemData().NHEAVY]
    if bond_feats is not None:
        bond_feats = bond_feats[None]
    fh = io.StringIO()
    for i, xyz in enumerate(xyz23):
        write_file.writepdb_file(fh, xyz, seq, bond_feats=bond_feats, modelnum=i, **kwargs)

    fh.seek(0)
    add_pdb_stream_to_silent(silent_path, tag, fh.readlines(), scores=scores)

def add_pdb_stream_to_silent(silent_path, tag, pdb_stream, scores={}):
    '''
    Add this structure to a silent file

    Args:
        silent_path (str): Path to silent file
        tag (str): Name of this design
        pdb_stream (list[str]): The contents of the pdbfile (from readlines())
        scores (dict[float or str]): scores to be added to the silent file's internal scorefile
        **kwargs: Additional args to be passed to write_file.writepdb_file
    '''

    # Then turn that pdb into a pose
    pose = pyro().Pose()
    ros().core.import_pose.pose_from_pdbstring(pose, ''.join(pdb_stream))

    # Then add that pose to a silent file (using append)
    sfd_out = ros().core.io.silent.SilentFileData( silent_path, False, False, "binary", ros().core.io.silent.SilentFileOptions())
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct(pose, tag)
    add_dict_to_silent(struct, scores)
    sfd_out.add_structure(struct)
    sfd_out.write_all(silent_path, False)


def add_dict_to_silent(struct, d):
    '''
    Loads a silent struct's internal scorefile with values from a dictionary

    Args:
        struct (pyrosetta.rosetta.core.io.silent.SilentStruct): The silent struct
        d (dict[float or str]): The dictionary of values to load into the silent file
    '''
    d = format_scores(d)

    for key in d:
        struct.add_string_value(key, value)


def load_silent_checkpoint(run_prefix):
    '''
    Loads the list of finished structures from the file pointed to by run_prefix

    Don't try to parse the silent file for this info since that could potentially involve reading a lot of data

    Args:
        run_prefix (str): The checkpoint file

    Returns:
        chckpoint_done (dict[str,str]): out_prefix:message. Which designs are finished and why
    '''
    checkpoint_name = run_prefix + '_ckpt'
    checkpoint_done = dict()
    if os.path.exists(checkpoint_name):
        with open(checkpoint_name) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                message = f'{line} already exists in {checkpoint_name}.'
                checkpoint_done[line] = message
    return checkpoint_done


def silent_checkpoint_design(run_prefix, individual_prefix):
    '''
    Add another design to the silent checkpoint file

    Args:
        run_prefix (str): The checkpoint file
        individual_prefix (str): The name of the finished design
    '''
    checkpoint_name = run_prefix + '_ckpt'
    with open(checkpoint_name, 'a') as f:
        f.write(individual_prefix + '\n')


def read_pose_from_silent(silent_path, tag, sfd=None):
    '''
    Get a pose from a silent file by tag

    Note! This function reads the entire silent file if sfd=None. If you're going to read more than one structure
       consider caching the sfd that is returned

    Args:
        silent_path (str): Path to silent file
        tag (str): Tag from silent file
        sfd (pyrosetta.rosetta.core.io.silent.SilentFileData or None): A previously opened sfd

    Returns:
        pose (pyrosetta.rosetta.core.pose.Pose): The structure
        sfd (pyrosetta.rosetta.core.io.silent.SilentFileData): The SilentFileData that you could reuse
    '''
    if sfd is None:
        sfd = ros().core.io.silent.SilentFileData(ros().core.io.silent.SilentFileOptions())
        sfd.read_file(silent_path)

    assert tag in list(sfd.tags()), f'Tag: {tag} not found in silent file: {silent_path}'

    pose = pyro().Pose()
    sfd.get_structure(tag).fill_pose(pose)

    return pose, sfd




def format_scores(scores, float_transition_value=0.01):
    '''
    Turn a dict of scores into strings that are more human readable

    Args:
        scores (dict[str,?]): The scores to convert to strings
        float_transition_value (str): The value to switch from %.3f to %.3g at
    '''
    new_scores = {}

    for key, value in scores.items():

        # Torch tensors get converted to their values
        if torch.is_tensor(value):
            if torch.numel(value) == 1:
                value = value.item()
            else:
                # If you decide to pass a multi-valued vector to this function I'll let you decide what to do with it
                new_scores[key] = str(value)
                continue

        # Numpy arrays get converted to their values
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.item()
            else:
                # If you decide to pass a multi-valued vector to this function I'll let you decide what to do with it
                new_scores[key] = str(value)
                continue

        # Strings get passed through directly
        if isinstance(value, str):
            new_scores[key] = value
            continue

        try:
            # ints, bools
            if int(value) == value:
                new_scores[key] = '%i'%int(value)
                continue
        except (ValueError, TypeError):
            pass

        try:
            # floats
            if float(value) == value or np.isnan(value):
                value = float(value)
                if abs(value) > float_transition_value:
                    new_scores[key] = '%.3f'%value
                else:
                    new_scores[key] = '%.3g'%value
                continue
        except (ValueError, TypeError):
            pass

        # Anything else we just cast to string because in theory you could pass anything to this function
        new_scores[key] = str(value)

    return new_scores
