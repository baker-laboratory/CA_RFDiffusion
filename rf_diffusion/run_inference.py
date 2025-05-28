#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""
import os


import re
from collections import defaultdict
import time
import pickle
import dataclasses
import torch 
from omegaconf import OmegaConf
import hydra
import logging
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
from rf_diffusion.inference import model_runners
import rf2aa.tensor_util
import rf2aa.util
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.chemical import reinitialize_chemical_data
from rf_diffusion import aa_model
import copy
#from rf_diffusion import atomize
from rf_diffusion.dev import idealize_backbone
from rf_diffusion.idealize import idealize_pose
import rf_diffusion.features as features
from rf_diffusion.import_pyrosetta import prepare_pyrosetta
from rf_diffusion import silent_files
import rf_diffusion.inference.utils as iu
import rf_diffusion.conditions.util
import tqdm
#import rf_diffusion.atomization_primitives
from rf_diffusion.inference.filters import init_filters, FilterFailedException, do_filtering
from rf_diffusion.inference.t_setup import setup_t_arrays
from rf_diffusion.inference.mid_run_modifiers import apply_mid_run_modifiers
ic.configureOutput(includeContext=True)

import rf_diffusion.nucleic_compatibility_utils as nucl_utils
from rf_diffusion.kinematics import th_kabsch

logger = logging.getLogger(__name__)

def make_deterministic(seed=0, ignore_if_cuda=False):
    # if not (ignore_if_cuda and torch.cuda.device_count() > 0):
    #     torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True)
    seed_all(seed)

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seeds():
    return {
        'torch': torch.get_rng_state(),
        'np': np.random.get_state(),
        'python': random.getstate(),
    }

@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    if 'custom_chemical_config' in conf:
        reinitialize_chemical_data(**conf.custom_chemical_config)
    prepare_pyrosetta(conf)
        
    sampler = get_sampler(conf)
    sample(sampler)

def get_sampler(conf):
    if conf.inference.deterministic:
        seed_all()

    # Loop over number of designs to sample.
    design_startnum = conf.inference.design_startnum
    if conf.inference.design_startnum == -1:
        existing = glob.glob(conf.inference.output_prefix + '*.pdb')
        indices = [-1]
        for e in existing:
            m = re.match(fr'{conf.inference.output_prefix}_(\d+).*\.pdb$', e)
            if m:
                m = m.groups()[0]
                indices.append(int(m))
        design_startnum = max(indices) + 1   

    conf.inference.design_startnum = design_startnum
    # Generate an empty sampler in case every output has already been generated
    sampler = model_runners.sampler_selector(conf, skip_initialization=True)
    return sampler

def finish_sampler_initialization(sampler):
    '''
    For the run_inference.py executable, we don't fully initialize the sampler on load such that the model
        does not need to be loaded for full-completed runs

    This function finishes the initialization process

    Args:
        sampler (Sampler): The sampler to finish initializing
    '''
    if sampler.initialized:
        return

    if sampler._conf.inference.deterministic:
        seed_all()
    sampler.initialize(sampler._conf)


def expand_config(conf):
    confs = {}
    if conf.inference.guidepost_xyz_as_design:
        sub_conf = copy.deepcopy(conf)
        for val in conf.inference.guidepost_xyz_as_design_bb:
            sub_conf.inference.guidepost_xyz_as_design_bb = val
            suffix = f'atomized-bb-{val}'
            confs[suffix] = copy.deepcopy(sub_conf)
    else:
        confs = {'': conf}
    return confs

def add_carfd_sidechains(px0: torch.Tensor, ref_dict: dict) -> torch.Tensor:
    """Computes ideal sidechains aligned onto px0 from original motif
    
    Args:
        px0: final output from model fwd/ refinement prediction
        ref_dict: contains information on the motif/contigs, and perfect original
                  motif structure
    """
    motif_indep = ref_dict['motif_indep']
    motif_idx_ref = ref_dict['con_ref_idx0']
    motif_idx_hal = ref_dict['con_hal_idx0']

    # grab the sequence & structure of non-SM motif residues  
    motif_seq_ref = motif_indep.seq[motif_idx_ref]
    motif_crds_ref = motif_indep.xyz[motif_idx_ref]

    # compute sidechain torsions in the original (ref) motif
    converter = rf2aa.util_module.XYZConverter()
    tors, tors_alt, tors_mask, tors_planar = converter.get_torsions(motif_crds_ref[None], 
                                                                    motif_seq_ref[None])
    

    # grab predicted ("hal") motif backbone coordinates, and extend sidechains onto them 
    # according to the torsions from perfect motif
    motif_bb_hal = px0[motif_idx_hal,:3,:]
    motif_allatom_hal = converter.compute_all_atom(motif_seq_ref[None], motif_bb_hal[None], tors)
    (RTframes, xyz_full) = motif_allatom_hal 
    # slice it in
    px0[motif_idx_hal,:14] = xyz_full[0,:,:14]

    # now align the output to the original motif on backbone
    motif_bb_ref = motif_crds_ref[:,:3]
    com_ref = motif_bb_ref.reshape(-1,3).mean(dim=0)
    com_hal = motif_bb_hal.reshape(-1,3).mean(dim=0)
    bb_rms, _, R = th_kabsch(motif_bb_ref.reshape(-1,3), motif_bb_hal.reshape(-1,3))
    # bring to origin, rotate, then translate to reference/native position
    px0 = torch.einsum('lai,ij->laj', px0 - com_hal, R) + com_ref

    return px0


def sampler_i_des_bounds(sampler):
    '''
    Get i_des_start and i_des_end for a given sampler

    Args:
        sampler (Sampler): sampler

    Returns:
        i_des_start (int): The first i_des to sample
        i_des_end (int): One past the last i_des to sample
    '''
    i_des_start = sampler._conf.inference.design_startnum
    i_des_end = i_des_start + sampler._conf.inference.num_designs
    return i_des_start, i_des_end

def sampler_out_prefix(sampler, i_des=0):
    '''
    Get the output prefix for a sampler run

    Args:
        sampler (Sampler): sampler
        i_des (int): Which design

    Returns:
        run_prefix (str): A prefix that is general for all outputs from this run
        individual_prefix (str): A prefix for this individual i_des
    '''
    run_prefix = sampler._conf.inference.output_prefix
    individual_prefix = f'{run_prefix}_{i_des}'

    return run_prefix, individual_prefix

def load_checkpoint_done(sampler):
    '''
    Load a dict of which designs have already been finished

    Args:
        sampler (Sampler): sampler

    Returns:
        checkpoint_set (set[str,str]): A list of all of the individual prefixes that have already run and a message as to why they are done
    '''
    run_prefix, _ = sampler_out_prefix(sampler)

    if sampler._conf.inference.silent_out:
        # Someday this might switch to using a .silent.idx file
        checkpoint_done = silent_files.load_silent_checkpoint(run_prefix)
    else:
        # Run glob exactly once since it's not good for the file system
        files = glob.glob(run_prefix + '*')

        # Loop through what the names will be and look for the outputs
        checkpoint_done = dict()
        i_des_start, i_des_end = sampler_i_des_bounds(sampler)
        for i_des in range(i_des_start, i_des_end):
            _, individual_prefix = sampler_out_prefix(sampler, i_des=i_des)

            # Check for 4 output patterns that might exist
            patterns = ['[.]trb', '-.*[.]trb']
            if not bool(sampler._conf.inference.write_trb): # trbs can still be written to denote filter-failures
                patterns += ['[.]pdb', '-.*[.]pdb']

            for pattern in patterns:
                re_comp = re.compile(individual_prefix + pattern)

                for file in files:
                    match = re_comp.match(file)
                    if match:
                        message = f'{match.group(0)} already exists.'
                        checkpoint_done[individual_prefix] = message

    return checkpoint_done

def checkpoint_i_des(sampler, i_des, nothing_written=False):
    '''
    Note that a design has finished

    Args:
        sampler (Sampler): sampler
        i_des (int): Number of design
        nothing_written (bool): Indicates that nothing was written to disk for this one but it's still considered done
    '''
    run_prefix, individual_prefix = sampler_out_prefix(sampler, i_des)
    if sampler._conf.inference.silent_out:
        silent_files.silent_checkpoint_design(run_prefix, individual_prefix)
    else:
        if nothing_written:
            # We drop a blank trb to indicate that this design failed
            with open(f'{individual_prefix}.trb','wb') as f_out:
                pickle.dump({}, f_out)

def sample(sampler):
    log = logging.getLogger(__name__)

    # Load a dictionary of finished designs with their finished messages
    checkpoint_done = load_checkpoint_done(sampler)

    # Sample each of the designs
    i_des_start, i_des_end = sampler_i_des_bounds(sampler)
    for i_des in range(i_des_start, i_des_end):

        start_time = time.time()
        _, out_prefix = sampler_out_prefix(sampler, i_des=i_des)
        log.info(f'Making design {out_prefix}')
        if sampler._conf.inference.cautious and out_prefix in checkpoint_done:
             log.info(f'(cautious mode) Skipping this design because {checkpoint_done[out_prefix]}')
             continue
        finish_sampler_initialization(sampler)
        # Set the actual per-design seed if deterministic (critically after sampler_initialization)
        if sampler._conf.inference.deterministic:
            seed_all(i_des + sampler._conf.inference.seed_offset)
        print(f'making design {i_des} of {i_des_start}:{i_des_end}', flush=True)
        sampler_out = sample_one_w_retry(sampler, i_des)
        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')
        if sampler_out is not None:
            original_conf = copy.deepcopy(sampler._conf)
            confs = expand_config(sampler._conf)
            for suffix, conf in confs.items():
                sampler._conf = conf
                out_prefix_suffixed = out_prefix
                if suffix:
                    out_prefix_suffixed += f'-{suffix}'
                if sampler._conf.inference.get('ca_rfd_refine', False):
                    out_prefix_suffixed = f'{sampler._conf.inference.output_prefix}_{i_des}'
                print(f'{out_prefix_suffixed=}, {conf.inference.guidepost_xyz_as_design_bb=}')
                # TODO: See what is being altered here, so we don't have to copy sampler_out
                save_outputs(sampler, out_prefix_suffixed, *(copy.deepcopy(o) for o in sampler_out))
                sampler._conf = original_conf
        checkpoint_i_des(sampler, i_des, nothing_written=sampler_out is None)

def sample_one_w_retry(sampler, i_des, **kwargs):
    '''
    Wrapper function for sample_one() that handles the case where a failed filter can request a redo

    Args:
        sampler (Sampler): The sampler
        i_des (int): The sequential number of this design

    Returns:
        sampler_out (tuple or None): The ever-growing list of items the sampler returns or None if filters killed this example
    '''
    _, prefix = sampler_out_prefix(sampler, i_des=i_des)

    max_attempts = sampler._conf.filters.max_attempts_per_design
    max_steps = sampler._conf.filters.max_steps_per_design

    cumulative_steps = 0
    for attempt in range(max_attempts):
        try:
            return sample_one(sampler, i_des=i_des, **kwargs)

        except FilterFailedException as e:
            # Check that we have not taken more steps than we are allowed
            cumulative_steps += e.n_steps_taken
            if cumulative_steps >= max_steps:
                print(f'Aborting {prefix}: Too many steps ({cumulative_steps})')
                break

            # Print the message to tell the user what is happening
            if attempt < max_attempts - 1:
                print(f'Retrying {prefix}: Attempt {attempt + 2} / {max_attempts} ({cumulative_steps} / {max_steps} steps taken)')
            else:
                print(f'Aborting {prefix}: Too many attempts ({max_attempts})')
    return None


def sample_one(sampler, i_des=0, simple_logging=False):
    '''
    Args:
        sampler (Sampler): The sampler
        i_des (int): The sequential number of this design
        simple_logging (bool): Print out some dots during inference

    Returns:
        sampler_out (tuple): The ever-growing list of items the sampler returns

    Throws:
        FilterFailedException: If filters are enabled and one fails
    '''

    indep, contig_map, _, t_step_input = sampler.sample_init(i_des)
    log = logging.getLogger(__name__)
    filters = init_filters(sampler._conf)

    traj_stack = defaultdict(list)
    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []
    scores = {}

    rfo = None
    extra = {
        'rfo_uncond': None,
        'rfo_cond': None,
        'n_steps': None
    }

    # Initialize featurizers
    extra_tXd_names = getattr(sampler._conf, 'extra_tXd', [])
    features_cache = features.init_tXd_inference(indep, extra_tXd_names, sampler._conf.extra_tXd_params, sampler._conf.inference.conditions)

    # On a vanilla diffusion run. ts=range(t_step_input,final_step-1,-1). n_steps=ones(), self_cond=[str_self_cond], final_it=len(ts)-1
    (
        ts,                 # The "diffusion t" we will use for each step
        n_steps,            # Number of diffuser.reverse() steps to take
        self_cond,          # Should we self condition?
        final_it,           # Which timestep is the normal last output
        addtl_write_its,    # Are there other timepoints we should write to disc?
        mid_run_modifiers,  # Trajectory altering movers for specific protocols
    ) = setup_t_arrays(sampler._conf, t_step_input)

    # Loop over number of reverse diffusion time steps.
    for it, t in tqdm.tqdm(list(enumerate(ts))):
        sampler._log.info(f'Denoising {t=}')
        if simple_logging:
            e = '.'
            if t%10 == 0:
                e = t
            print(f'{e}', end='')

        if sampler._conf.preprocess.randomize_frames:
            print('randomizing frames')
            indep.xyz = aa_model.randomly_rotate_frames(indep.xyz)
        elif sampler._conf.preprocess.get('eye_frames',False): 
            print('Eye frames')
            indep.xyz = aa_model.eye_frames(indep.xyz, recenter=True)
        else:
            pass
        extra['n_steps'] = n_steps[it]
        extra['self_cond'] = self_cond[it]
        px0, x_t, seq_t, rfo, extra = sampler.sample_step(
            t, indep, rfo, extra, features_cache)
        # assert_that(indep.xyz.shape).is_equal_to(x_t.shape)
        rf2aa.tensor_util.assert_same_shape(indep.xyz, x_t)
        # ic(t)
        # # @Altaeth EXPERIMENT: final step, replace x_t with px0
        # if sampler._conf.experiment.get('final_sc_pred', False) and t == sampler.inf_conf.final_step:
        #     x_t = px0[:,:ChemData().NHEAVY,:]
        # ic(x_t.shape)
        indep.xyz = x_t
        x_t = copy.deepcopy(x_t)
        # @Altaeth, this code below might not be necessary, but it blocks important coordinates
        #x_t[:,3:] = np.nan
            
        aa_model.assert_has_coords(indep.xyz, indep)
        # missing_backbone = torch.isnan(indep.xyz).any(dim=-1)[...,:3].any(dim=-1)
        # prot_missing_bb = missing_backbone[~indep.is_sm]
        # sm_missing_ca = torch.isnan(indep.xyz).any(dim=-1)[...,1]
        # try:
        #     assert not prot_missing_bb.any(), f'{t}:prot_missing_bb {prot_missing_bb}'
        #     assert not sm_missing_ca.any(), f'{t}:sm_missing_ca {sm_missing_ca}'
        # except Exception as e:
        #     print(e)
        #     import ipdb
        #     ipdb.set_trace()
        if sampler._conf.inference.get('ca_rfd_refine', False):
            px0 = add_carfd_sidechains(px0, sampler.conditions_dict['ref_dict'])
        px0_xyz_stack.append(px0)
        denoised_xyz_stack.append(x_t)
        seq_stack.append(seq_t)
        for k, v in extra['traj'].items():
            traj_stack[k].append(v)

        # If len(filters) == 0 this immediately returns
        do_filtering(filters, indep, t, it, px0, scores, contig_map=contig_map, is_diffused=sampler.is_diffused)

        # These mid_run_modifiers are not normally present
        #   Even though this block of code is scary. It would even scarier if this for-loop was full of 20 if statements
        (
            sampler,
            indep,
            contig_map,
            rfo,
            px0_xyz_stack,
            denoised_xyz_stack,
            seq_stack,
            final_it,
            stop,
        ) = apply_mid_run_modifiers(mid_run_modifiers[it],
                                    i_des,
                                    sampler,
                                    indep,
                                    contig_map,
                                    rfo,
                                    px0_xyz_stack,
                                    denoised_xyz_stack,
                                    seq_stack,
                                    final_it,
                                    )
        if stop:
            break

    if t_step_input == 0:
        # Null-case: no diffusion performed.
        px0_xyz_stack.append(sampler.indep_orig.xyz)
        denoised_xyz_stack.append(indep.xyz)
        alanine_one_hot = torch.nn.functional.one_hot(torch.tensor(torch.zeros((indep.length(),), dtype=int)), ChemData().NAATOKENS)
        seq_stack.append(alanine_one_hot)
    
    # Flip order for better visualization in pymol
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack)
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
    ts = torch.flip(ts, [0,])
    seq_stack = list(reversed(seq_stack))
    final_it = len(ts) - 1 - final_it
    addtl_write_its = [(len(ts) - 1 - it, suff) for it,suff in addtl_write_its]

    for k, v in traj_stack.items():
        traj_stack[k] = torch.flip(torch.stack(v), [0,])

    raw = (px0_xyz_stack, denoised_xyz_stack)

    # Add back any (implicit) side chain atoms from the motif
    denoised_xyz_stack = add_implicit_side_chain_atoms(
        seq=indep.seq,
        act_on_residue=~sampler.is_diffused,
        xyz=denoised_xyz_stack,
        xyz_with_sc=sampler.indep_orig.xyz,
    )
    px0_xyz_stack_filler = add_implicit_side_chain_atoms(
        seq=indep.seq,
        act_on_residue=~sampler.is_diffused,
        xyz=px0_xyz_stack[..., :ChemData().NHEAVY, :],
        # xyz_with_sc=sampler.indep_orig.xyz,
        xyz_with_sc=sampler.indep_orig.xyz[..., :ChemData().NHEAVY, :],
    )
    px0_xyz_stack[..., :ChemData().NHEAVY, :] = px0_xyz_stack_filler

    for k, v in traj_stack.items():
        traj_stack[k] = add_implicit_side_chain_atoms(
            seq=indep.seq,
            act_on_residue=~sampler.is_diffused,
            xyz=v[..., :ChemData().NHEAVY, :],
            xyz_with_sc=sampler.indep_orig.xyz[..., :ChemData().NHEAVY, :],
        )

    # Idealize protein backbone
    is_protein = rf2aa.util.is_protein(indep.seq)
    denoised_xyz_stack[:, is_protein] = idealize_backbone.idealize_bb_atoms(
        xyz=denoised_xyz_stack[:, is_protein],
        idx=indep.idx[is_protein]
    )
    px0_xyz_stack_idealized = torch.clone(px0_xyz_stack)
    px0_xyz_stack_idealized[:, is_protein] = idealize_backbone.idealize_bb_atoms(
        xyz=px0_xyz_stack[:, is_protein],
        idx=indep.idx[is_protein]
    )
    log = logging.getLogger(__name__)
    backbone_ideality_gap = idealize_backbone.backbone_ideality_gap(px0_xyz_stack[0], px0_xyz_stack_idealized[0])
    log.debug(backbone_ideality_gap)
    px0_xyz_stack = px0_xyz_stack_idealized


    for k, v in traj_stack.items():
        traj_stack[k][:, is_protein] = idealize_backbone.idealize_bb_atoms(
            xyz=v[:, is_protein],
            idx=indep.idx[is_protein]
        )

    is_diffused = sampler.is_diffused.clone()

    return indep, contig_map, sampler.atomizer, t_step_input, denoised_xyz_stack, px0_xyz_stack, seq_stack, is_diffused, raw, traj_stack, ts, scores, final_it, addtl_write_its

def add_implicit_side_chain_atoms(seq, act_on_residue, xyz, xyz_with_sc):
    '''
    Copies the coordinates of side chain atoms (in residues marked "True" 
    in `act_on_residue`) in `xyz_with_sc` to `xyz`.

    Inputs
    ------------
        seq (L,)
        act_on_residue (L,): Only residues marked True will have side chain atoms added.
        xyz (..., L, n_atoms, 3)
        xyz_with_sc (L, n_atoms, 3)

    '''
    # ic(xyz.shape, xyz_with_sc.shape)
    # Shape checks
    L, n_atoms = xyz_with_sc.shape[:2]
    assert xyz.shape[-3:] == xyz_with_sc.shape, f'{xyz.shape[-3:]=} != {xyz_with_sc.shape=}'
    assert len(seq) == L
    assert len(act_on_residue) == L

    replace_sc_atom = ChemData().allatom_mask[seq][:, :n_atoms]
    is_prot = nucl_utils.get_resi_type_mask(seq, 'prot_and_mask')
    replace_sc_atom[is_prot, :5] = False  # Does not add cb, since that can be calculated from N, CA and C for proteins
    replace_sc_atom[~act_on_residue] = False
    xyz[..., replace_sc_atom, :] = xyz_with_sc[replace_sc_atom]

    return xyz


def save_outputs(sampler, out_prefix, indep, contig_map, atomizer, t_step_input, denoised_xyz_stack, px0_xyz_stack, seq_stack, is_diffused_in, raw, traj_stack, ts, scores, final_it, addtl_write_its):
    log = logging.getLogger(__name__)

    # Make the output folder
    out_head, out_tail = os.path.split(out_prefix)
    os.makedirs(out_head, exist_ok=True)

    final_seq = seq_stack[final_it]


    if sampler._conf.seq_diffuser.seqdiff is not None:
        # When doing sequence diffusion the model does not make predictions beyond category 19
        #final_seq = final_seq[:,:20] # [L,20]
        # Cannot do above code for NA, but instead get rid of mask tokens
        final_seq = final_seq[20:22] = 0 


    # All samplers now use a one-hot seq so they all need this step, get rid of non polymer residues
    final_seq[~indep.is_sm, ChemData().NNAPROTAAS:] = 0 
    final_seq = torch.argmax(final_seq, dim=-1)

    # replace mask and unknown tokens in the final seq with alanine
    final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)

    # determine lengths of protein and ligand for correct chain labeling in output pdb
    chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain)

    # Figure out which timesteps we are going to output
    write_ts = list(addtl_write_its)
    write_ts.append((final_it, '')) # Make sure the default is last so that the variables all end up correct after the loop

    for stack_idx, t_suffix in write_ts:

        # ic(seq_design)

        # Make copies of sampler outputs for final modifications
        xyz_design = px0_xyz_stack[stack_idx].clone()
        seq_design = final_seq.clone()
        is_atomized = torch.zeros(indep.length()).bool()
        is_diffused = is_diffused_in.clone()
        if atomizer is not None:
            is_atomized = copy.deepcopy(atomizer.residue_to_atomize)


        # Save idealized pX0 last step
        xyz_design_idealized = xyz_design.clone()[None]

        idealization_rmsd = float('nan')
        if sampler._conf.inference.idealize_sidechain_outputs:
            log.info('Idealizing atomized sidechains for pX0 of the last step...')
            # Only idealize residues that are atomized.
            xyz_design_idealized[0, is_atomized], idealization_rmsd, _, _ = idealize_pose(
                xyz_design[None, is_atomized],
                seq_design[None, is_atomized]
            )
        # Create pdb, idealize the backbone, and rename ligand atoms
        idealized_pdb_stream = aa_model.write_traj(None, xyz_design_idealized, seq_design, indep.bond_feats, ligand_name_arr=contig_map.ligand_names, chain_Ls=chain_Ls, idx_pdb=indep.idx)
        idealized_pdb_stream = idealize_backbone.rewrite(None, None, pdb_stream=idealized_pdb_stream)
        idealized_pdb_stream = aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, None, pdb_stream=idealized_pdb_stream)


        run_prefix, _ = sampler_out_prefix(sampler)

        if sampler._conf.inference.silent_out:
            # Tag starts out the same as the pdb name as usual
            tag = out_tail + t_suffix

            # But then we add in the folder path so that people can cat everything together and not have duplicate names
            if bool(sampler._conf.inference.silent_folder_sep):
                tag = os.path.join(out_head, tag).replace('/', sampler._conf.inference.silent_folder_sep)

            silent_name = run_prefix + '_out.silent'
            silent_files.add_pdb_stream_to_silent(silent_name, tag, idealized_pdb_stream, scores=scores)
            des_path = f'{silent_name}:{tag}'
        else:
            # Write pdb to disk
            out_idealized = f'{out_prefix}{t_suffix}.pdb'
            des_path = os.path.abspath(out_idealized)
            with open(des_path, 'w') as fh:
                fh.write(''.join(idealized_pdb_stream))

            tag = out_idealized[:-len('.pdb')]

        if len(scores) > 0 and (
                (sampler._conf.inference.silent_out and str(sampler._conf.inference.write_scorefile) == 'FORCE')
                or
                (not sampler._conf.inference.silent_out and bool(sampler._conf.inference.write_scorefile))
            ):
            iu.append_run_to_scorefile(run_prefix, tag, scores, sampler._conf.inference.scorefile_delimiter)

    # pX0 last step
    write_unidealized = not sampler._conf.inference.silent_out
    if write_unidealized:
        unidealized_dir = os.path.join(out_head, 'unidealized')
        os.makedirs(unidealized_dir, exist_ok=True)
        out_unidealized = os.path.join(unidealized_dir, f'{out_tail}.pdb')
        aa_model.write_traj(out_unidealized, xyz_design[None,...], seq_design, indep.bond_feats, ligand_name_arr=contig_map.ligand_names, chain_Ls=chain_Ls, idx_pdb=indep.idx)

    # Setup stack_mask for writing smaller trajectories
    t_int = ts.clone() #np.arange(int(t_step_input), sampler.inf_conf.final_step-1, -1)[::-1]
    stack_mask = torch.ones(len(denoised_xyz_stack), dtype=bool)
    if len(sampler._conf.inference.write_trajectory_only_t) > 0:
        assert sampler._conf.inference.write_trajectory or sampler._conf.inference.write_trb_trajectory, ('If inference.write_trajectory_only_t is enabled'
                ' at least one of inference.write_trajectory or inference.write_trb_trajectory must be enabled')
        stack_mask[:] = False
        for write_t in sampler._conf.inference.write_trajectory_only_t:
            stack_mask[write_t == t_int] = True
        assert stack_mask.sum() > 0, ('Your inference.write_trajectory_only_t has led to no frames being selected for output. Something is specified wrong.'
            f' inference.write_trajectory_only_t={inference.write_trajectory_only_t}')

    # trajectory pdbs
    if sampler._conf.inference.write_trajectory:
        traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
        os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

        out = f'{traj_prefix}_Xt-1_traj.pdb'
        aa_model.write_traj(out, denoised_xyz_stack[stack_mask], final_seq, indep.bond_feats, ligand_name_arr=contig_map.ligand_names, chain_Ls=chain_Ls, idx_pdb=indep.idx)
        xt_traj_path = os.path.abspath(out)
        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, xt_traj_path)

        out=f'{traj_prefix}_pX0_traj.pdb'
        aa_model.write_traj(out, px0_xyz_stack[stack_mask], final_seq, indep.bond_feats, chain_Ls=chain_Ls, ligand_name_arr=contig_map.ligand_names, idx_pdb=indep.idx)
        x0_traj_path = os.path.abspath(out)
        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, x0_traj_path)

        for k, v in traj_stack.items():
            out=f'{traj_prefix}_{k}_traj.pdb'
            aa_model.write_traj(out, v[stack_mask], final_seq, indep.bond_feats, chain_Ls=chain_Ls, ligand_name_arr=contig_map.ligand_names, idx_pdb=indep.idx)
            traj_path = os.path.abspath(out)
            aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, traj_path)

    # run metadata
    sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
    write_trb = (sampler._conf.inference.write_trb and not sampler._conf.inference.silent_out) or str(sampler._conf.inference.write_trb) == 'FORCE'
    if write_trb:
        trb = dict(
            config = OmegaConf.to_container(sampler._conf, resolve=True),
            device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
            t_int=t_int,
            t=np.arange(int(t_step_input), sampler.inf_conf.final_step-1, -1)[::-1] / sampler._conf.diffuser.T,
            is_diffused=sampler.is_diffused,
            point_types=aa_model.get_point_types(sampler.indep_orig, atomizer),
            atomizer_spec=None,
            con_hal_idx0=contig_map.get_mappings()['con_hal_idx0'],
            con_ref_idx0=contig_map.get_mappings()['con_ref_idx0'],
            contigmap_ref=contig_map.ref, 
            contigmap_hal=contig_map.hal,
        )
        if len(scores) > 0 and sampler._conf.inference.write_scores_to_trb:
            trb['scores'] = scores
        # The trajectory and the indep are big and contributed to the /net/scratch crisis of 2024
        if sampler._conf.inference.write_trb_trajectory:
            trb['px0_xyz_stack'] = raw[0].detach().cpu()[stack_mask].numpy()
            trb['denoised_xyz_stack'] = raw[1].detach().cpu()[stack_mask].numpy()
        if sampler._conf.inference.write_trb_indep:
            trb['indep'] = {k:v.detach().cpu().numpy() if hasattr(v, 'detach') else v for k,v in dataclasses.asdict(indep).items()}
            trb['indep_true'] = {k:v.detach().cpu().numpy() if hasattr(v, 'detach') else v for k,v in dataclasses.asdict(sampler.indep_orig).items()}
        if contig_map:
            for key, value in contig_map.get_mappings().items():
                trb[key] = value

        if atomizer:
            motif_deatomized = atomize.convert_atomized_mask(atomizer, ~sampler.is_diffused)
            trb['motif'] = motif_deatomized
        trb['idealization_rmsd'] = idealization_rmsd

        with open(f'{out_prefix}.trb','wb') as f_out:
            pickle.dump(trb, f_out)
    else:
        assert not sampler._conf.inference.write_trb_trajectory, 'If you want to write the trajectory to the trb you must enable inference.write_trb'
        assert not sampler._conf.inference.write_trb_indep, 'If you want to write the indep to the trb you must enable inference.write_trb'

    log.info(f'design : {des_path}')
    if sampler._conf.inference.write_trajectory:
        log.info(f'Xt traj: {xt_traj_path}')
        log.info(f'X0 traj: {x0_traj_path}')


if __name__ == '__main__':
    main()
