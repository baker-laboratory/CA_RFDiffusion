import pandas as pd
import numpy as np
import functools
from omegaconf import OmegaConf
import pickle
from dataclasses import dataclass
from hydra.core.hydra_config import HydraConfig
import glob
import os
from itertools import *
from collections import defaultdict
import torch
import re
from rf_diffusion.inference import utils as iu
import hydra
from rf_diffusion import util
from icecream import ic
import tqdm


def get_logged_examples(path):
    d = defaultdict(dict)
    for p in sorted(glob.glob(path + '*')):
        b = os.path.basename(p)
        epoch = b.split('_')[2]
        if epoch != '0':
            continue
        m = re.match('(.*\d+)(.*)', b)
        ex, f = m.groups()
        d[ex][f] = p
    return d

def write_backbone(filename, bb_stack):
    T, L, _, _ = bb_stack.shape
    fa_stack = torch.zeros((T,L,14,3), dtype=bb_stack.dtype)
    fa_stack[:,:,:3,:] = bb_stack[:,:,:3]
    bfacts = torch.ones((L,))
    seq = torch.full((L,), 21)
    return util.writepdb_multi(filename, fa_stack, bfacts, seq, use_hydrogens=False, backbone_only=True)

def c_alpha_disp(a, b):
    assert a.ndim == 3
    assert b.ndim == 3
    return torch.sqrt(torch.sum(torch.square(a[:,1]-b[:,1]), dim=-1))

def c_alpha_rmsd(a,b):
    disp = c_alpha_disp(a,b)
    return torch.sqrt(torch.mean(torch.square(disp)))

def c_alpha_rmsd_traj_loop(traj):
    o = []
    for i in range(traj.shape[0]-1):
        o.append(c_alpha_rmsd(traj[i+1], traj[i]))
    o = torch.stack(o)
    return o

def c_alpha_rmsd_trajs(a,b):
    assert a.ndim == 4
    assert b.ndim == 4
    assert a.shape == b.shape
    return torch.sqrt(torch.mean(torch.sum(torch.square(a[:,:,1] - b[:,:,1]), dim=-1), dim=-1))

def get_examples():
    training_pdbs_path = '/home/ahern/projects/BFF/rf_diffusion/training_pdbs/training_pdbs/'
    logged_examples = get_logged_examples(training_pdbs_path)


    for k,d in logged_examples.items():
        true_path = logged_examples[k]['true.pdb']

        parsed = iu.parse_pdb(true_path)

        input_xyz = parsed['xyz']

        true_seq = parsed['seq']

        true_seq = torch.tensor(true_seq)

        yield input_xyz, true_seq




def reverse(sampler, xyz_true, seq_true, mask, final_steps=None, use_true=False, inject_true_x0=False):
    #true_seq = torch.tensor(parsed['seq'])
    forward_traj = None

    seq_true_one_hot = torch.nn.functional.one_hot(seq_true, 22)
    # diffuser = diffusion.Diffuser(**sampler._conf.diffuser)
    fa_stack, aa_masks, xyz_true = sampler.diffuser.diffuse_pose(
    torch.tensor(xyz_true),
    seq_true,
    torch.tensor(mask))
    forward_traj = torch.cat([xyz_true[None], fa_stack[:,:,:14]])
    
    # x_init, seq_init, forward_traj, aa_masks = sampler.sample_init(return_forward_trajectory=True)
    x_init, seq_init = sampler.sample_init()
    
    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []
    chi1_stack = []
    plddt_stack = []
    # pseq_stack = []

    x_t = torch.clone(x_init)
    seq_t = torch.clone(seq_init)
    final_steps = final_steps or sampler.t_step_input


    # Loop over number of reverse diffusion time steps.
    for t in tqdm.tqdm(range(int(final_steps), 0, -1)):

        # if use_true or t==final_steps:
        #     x_t = fa_stack[t-1,:,:14]
        #     aa_mask = aa_masks[t-1]

        #     seq_t = torch.full_like(seq_true, 21)
        #     seq_t[aa_mask] = seq_true[aa_mask]


        #if len(denoised_xyz_stack) == 0:
        #    denoised_xyz_stack.append(x_t)
        #    seq_stack.append(seq_t)

        #ic(seq_t.shape, x_t.shape, seq_init.shape)
        if inject_true_x0:
            if t > 1:
                x_t, seq_t, tors_t, px0 = sampler.denoiser.get_next_pose(
                    xt=x_t,
                    px0=xyz_true,
                    t=t,
                    diffusion_mask=sampler.mask_str.squeeze(),
                    seq_t=seq_t,
                    pseq0=seq_true_one_hot,
                    diffuse_sidechains=self.preprocess.sidechain_input,
                    align_motif=sampler.inf_conf.align_motif,
                )
            else:
                x_t = xyz_true
                px0 = xyz_true
                seq_t = seq_true
            plddt=[-1]
        else:
            px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init)
        
        
        
        px0_xyz_stack.append(px0)
        denoised_xyz_stack.append(x_t)
        seq_stack.append(seq_t.cpu())
        chi1_stack.append(tors_t[:,:])
        plddt_stack.append(plddt[0]) # remove singleton leading dimension

    denoised_xyz_stack = torch.stack(denoised_xyz_stack).cpu()
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack).cpu()
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
    seq_stack = torch.stack(seq_stack).cpu()
    seq_stack = torch.flip(seq_stack, [0,])
    #return denoised_xyz_stack, px0_xyz_stack, forward_traj, seq_stack
    return Trajectory(denoised_xyz_stack, px0_xyz_stack, forward_traj.cpu(), seq_stack)


@dataclass
class Trajectory:
    denoised_xt: torch.FloatTensor
    px0: torch.FloatTensor
    forward: torch.FloatTensor
    seq: torch.FloatTensor
    logits: torch.FloatTensor
    diffusion_mask: torch.BoolTensor

def run_partial_trajectory_sweep(sampler, xyz, seq, mask, T=None):
    stack_by_step = defaultdict(list)
    T = T or torch.arange(1, sampler._conf.diffuser.T)
    for final_steps in T:
        print('Running partial trajectory from {final_steps}')
        stacks = reverse(sampler, xyz, seq, mask, use_true=False, final_steps=final_steps)
        stack_by_step[final_steps] = stacks

    return stack_by_step

def get_sampler(conf):
    parsed = iu.parse_pdb(conf.inference.input_pdb)
    seq_true = parsed['seq']
    seq_true = torch.tensor(seq_true)

    #conf.contigmap.contigs=[f'{L}']

    sampler = iu.sampler_selector(conf)
    return sampler


@hydra.main(version_base=None, config_path='config/inference', config_name='no_fape_t1d_22')
def main(conf: HydraConfig) -> None:
    print(f'conf.likelihood: {conf.likelihood}')
    config = OmegaConf.to_container(conf, resolve=True),
    with open(os.path.join(conf.likelihood.output_dir, 'config.trb'),'wb') as fh:
        pickle.dump(config, fh)
    parsed = iu.parse_pdb(conf.inference.input_pdb)                                                        
    xyz_true = parsed['xyz']                                                                                                            
    seq_true = parsed['seq']
    seq_true = torch.tensor(seq_true)
    mask = parsed['mask']
    sampler = get_sampler(conf)
    o = run_partial_trajectory_sweep(sampler, xyz_true, seq_true, mask, conf.likelihood.T)
    output_path = os.path.join(conf.likelihood.output_dir, 'trajectories.pkl')
    with open(output_path, 'wb') as fh:
        pickle.dump(o, fh)
    print('success')


if __name__ == '__main__':
    main()


def c_alpha_rmsd_traj(traj):
    assert traj.shape[-1] == 3
    return torch.sqrt(torch.mean(torch.sum(torch.square(traj[...,1:,:,1,:] - traj[...,:-1,:,1,:]), dim=-1), dim=-1))


def df_from_tensor(arr, columns, val='val', values_by_column=None):
    assert arr.ndim == len(columns)
    
    df = pd.DataFrame(cartesian_product_transpose(*[np.arange(i) for i in arr.shape]),
                      columns=columns)
    df[val] = arr.ravel()
    if values_by_column:
        df.replace({col: dict(enumerate(vals)) for col, vals in values_by_column.items()}, inplace=True)
        #for column, values in values_by_column.items():
            #df.replace(colum
    return df

def cartesian_product_transpose(*arrays):
    """
    http://stackoverflow.com/a/11146645/190597 (senderle)
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    dtype = np.find_common_type([arr.dtype for arr in broadcasted], [])
    rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T

import itertools
def reverse_simple(sampler, feed_true_xt=False, stop_after=99999):
    x_init, seq_init, forward_traj, aa_masks, seq_orig = sampler.sample_init(return_forward_trajectory=True)
    #x_init, seq_init = sampler.sample_init()
    
    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []
    chi1_stack = []
    plddt_stack = []
    logits_stack = []
    # pseq_stack = []

    x_t = torch.clone(x_init)
    seq_t = torch.clone(seq_init)


    # Loop over number of reverse diffusion time steps.
    for t in itertools.islice(tqdm.tqdm(range(sampler.t_step_input, 0, -1)), stop_after):
        if feed_true_xt:
            #ic(forward_traj.shape, aa_masks.shape)
            x_t = forward_traj[t,:,:14]
            aa_mask = aa_masks[t-1]
            seq_t = torch.full_like(seq_orig, 21)
            seq_t[aa_mask] = seq_orig[aa_mask]
            if t == sampler.t_step_input:
                ic('asserting seq_t matches seq_init')
                torch.testing.assert_close(seq_t, seq_init)

        logits = torch.zeros(5)
        px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
            t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init,final_step=sampler.inf_conf.final_step)
        # return extra is not enabled by NRB's sampler
        #px0, x_t, seq_t, tors_t, plddt, logits = sampler.sample_step(
        #    t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init,final_step=sampler.inf_conf.final_step, return_extra=True)

        px0_xyz_stack.append(px0)
        denoised_xyz_stack.append(x_t)
        seq_stack.append(seq_t.clone().cpu())
        chi1_stack.append(tors_t[:,:])
        plddt_stack.append(plddt[0]) # remove singleton leading 
        logits_stack.append(logits.squeeze().cpu())

    denoised_xyz_stack = torch.stack(denoised_xyz_stack).cpu()
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack).cpu()
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
    seq_stack = torch.stack(seq_stack).cpu()
    seq_stack = torch.flip(seq_stack, [0,])
    logits_stack = torch.stack(logits_stack)
    logits_stack = torch.flip(logits_stack, [0,])
    #return denoised_xyz_stack, px0_xyz_stack, forward_traj, seq_stack
    return Trajectory(denoised_xyz_stack, px0_xyz_stack, forward_traj.cpu(), seq_stack, logits_stack, sampler.mask_str[0]), {'x_init': x_init, 'mask_str':sampler.mask_str[0]}

