#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import sys
import os
import traceback
import logging
import datetime
from rf_diffusion import master_addr
import wandb
import hydra
from hydra.core.global_hydra import GlobalHydra
import shutil
import copy
import time
import pickle 
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from contextlib import ExitStack
import assertpy
import re
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.chemical import reinitialize_chemical_data
import rf2aa.data.data_loader
import rf2aa.util
from rf2aa.util_module import XYZConverter
import rf2aa.loss.loss
import rf2aa.tensor_util
from rf_diffusion import run_inference
from rf_diffusion import aa_model
from rf_diffusion.data_loader import get_fallback_dataset_and_dataloader
from rf_diffusion.import_pyrosetta import prepare_pyrosetta
import rf_diffusion
PKG_DIR = rf_diffusion.__path__[0]
REPO_DIR = os.path.dirname(PKG_DIR)
from rf_diffusion import error
import pytimer.timer
pytimer.timer.default_logger.propagate = False

from rf_diffusion import loss as loss_module
from rf_diffusion.loss import *
from scheduler import get_stepwise_decay_schedule_with_warmup

from rf_diffusion import test_utils
from openfold.utils import rigid_utils as ru
from rf_diffusion.frame_diffusion.data import all_atom
import types 
import rf_diffusion.fake_torch_scatter as fake_torch_scatter
sys.modules["torch_scatter"] = fake_torch_scatter
from se3_flow_matching.data import all_atom as all_atom_fm

#added for inpainting training
from icecream import ic
import random

# added for logging git diff
import subprocess

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import noisers


logger = logging.getLogger(__name__)

global N_EXAMPLE_PER_EPOCH
global DEBUG 
global WANDB
USE_AMP = False

# For nucleic
import nucleic_compatibility_utils as nucl_utils

#BATCH_SIZE = 1 * torch.cuda.device_count()

class ReturnTrueOnFirstInvocation:
    def __init__(self):
        self.called = False
    
    def __call__(self):
        if self.called:
            return False
        self.called = True
        return True

firstLog = ReturnTrueOnFirstInvocation()

def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            if param.requires_grad:
                shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)

def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")

class Trainer():
    def __init__(self, conf=None):
        self.conf=conf
        self._exp_conf = conf.experiment

        if 'diffuser_hydra' in conf:
            self.diffuser = hydra.utils.instantiate(conf.diffuser_hydra)
        else:
            self.diffuser = noisers.get(self.conf.diffuser)

        self.diffuser.T = conf.diffuser.T

        # for all-atom str loss
        self.ti_dev = ChemData().torsion_indices
        self.ti_flip = ChemData().torsion_can_flip
        self.ang_ref = ChemData().reference_angles
        self.l2a = ChemData().long2alt
        self.allatom_converter = XYZConverter()
        self.fi_dev = ChemData().frame_indices

        self.hbtypes = ChemData().hbtypes
        self.hbbaseatoms = ChemData().hbbaseatoms
        self.hbpolys = ChemData().hbpolys

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        self.wandb_run_id = 'no_wandb'
        self.wandb_group = 'no_wandb_group'
    
    def make_rundir(self, wandb_name):
        rundir = self.conf.rundir or '.'
        rundir = os.path.join(rundir, wandb_name)
        return rundir

    def calc_loss(self, indep, loss_weights, model_out, diffuser_out, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_exp,
                  pred_in, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, dataset, chosen_task, t, xyz_in, is_diffused,
                  seq_diffusion_mask, seq_t, is_sm, unclamp=False, negative=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0,
                  lj_lin=0.75, use_H=False, w_disp=0.0, w_motif_disp=0.0, 
                  w_ax_ang=0.0, w_frame_dist=0.0, eps=1e-6, 
                  backprop_non_displacement_on_given=False, atomizer=None,
                  masks_1d={}, atom_frames=None):

        aux_data = {}

        device = model_out['rigids'].device
        batch_size, _, num_res, _ = model_out['rigids'].shape
        is_diffused = is_diffused[None].to(device)
        loss_mask = is_diffused.clone()
        loss_mask[...] = True
        is_sm = is_sm[None]
        assertpy.assert_that(is_diffused.ndim) == 2
        assertpy.assert_that(is_sm.ndim) == 2
        t = torch.tensor([t/self.conf.diffuser.T]).to(device)

        diffuser_out['rigids_0'] = diffuser_out['rigids_0'].to(device)
        gt_rot_score = diffuser_out['rot_score'][None].to(device)
        gt_trans_score = diffuser_out['trans_score'][None].to(device)
        rot_score_scaling = torch.tensor(diffuser_out['rot_score_scaling'])[None].to(device)
        trans_score_scaling = torch.tensor(diffuser_out['trans_score_scaling'])[None].to(device)

        # Dictionary for keeping track of losses 
        loss_dict = {}

        pred_rot_score = model_out['rot_score']
        pred_trans_score = model_out['trans_score']

        # Translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)


        # Translation x0 loss
        gt_trans_x0 = diffuser_out['rigids_0'][..., 4:] * self.conf.diffuser.r3.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] * self.conf.diffuser.r3.coordinate_scaling
        
        # Take translation loss only over diffused atoms
        gt_trans_x0 = gt_trans_x0[:, is_diffused.squeeze()]
        pred_trans_x0 = pred_trans_x0[:, :, is_diffused.squeeze()]
        trans_x0_loss = loss_module.mse(pred_trans_x0, gt_trans_x0)

        if self.conf.experiment.normalize_trans_x0:
            noise_var = float(1 - torch.exp(-self.diffuser._r3_diffuser.marginal_b_t(t)) + 1/self.conf.experiment.normalize_trans_x0_max_scaling)
            trans_x0_loss = trans_x0_loss / noise_var

        loss_dict['trans_score'] = trans_score_loss * (t > self._exp_conf.trans_x0_threshold) * int(self.conf.diffuser.diffuse_trans)
        loss_dict['trans_x0'] = trans_x0_loss * (t <= self._exp_conf.trans_x0_threshold) * int(self.conf.diffuser.diffuse_trans)

        # Rotation loss
        has_rot_loss = is_diffused * ~is_sm
        rot_mse = (gt_rot_score - pred_rot_score)**2 * has_rot_loss[:,:,None]
        rot_loss = torch.sum(
            rot_mse,
            dim=(-1, -2)
        ) / (has_rot_loss.sum(dim=-1) + 1e-10) # [B, I]

        # logging rot scores
        unscaled_mean_rot_loss = loss_module.normalize_loss(rot_loss.sum(dim=0), gamma=self.conf.experiment.gamma)
        aux_data['unscaled_rot_score_mean'] = torch.clone(unscaled_mean_rot_loss.detach())
        unscaled_rot_loss = rot_loss.sum(dim=0)
        aux_data['unscaled_rot_score_over_i'] = torch.clone(unscaled_rot_loss.detach())

        rot_loss /= rot_score_scaling[:, None]**2
        rot_loss *= int(self.conf.diffuser.diffuse_rot)
        loss_dict['rot_score'] = rot_loss

        # Backbone atom loss
        pred_atom37 = model_out['atom37'][...,:self.conf.experiment.n_bb,:]
        # try:
            # test_utils.assert_no_nan(pred_atom37)
        test_utils.assert_no_nan(pred_atom37)
        # except Exception:
            # indep.write_pdb('nan_pred.pdb')
            # if atomizer:
                # indep_write = atomizer.deatomize(indep.clone())
                # indep_write.write_pdb('nan_pred_deatomized.pdb')            
            # test_utils.assert_no_nan(pred_atom37)
        gt_rigids = ru.Rigid.from_tensor_7(diffuser_out['rigids_0'].type(torch.float32))
        # gt_psi = diffuser_out['torsion_angles_sin_cos'][..., 2, :][None] <-- ignored since n_bb is 3
        gt_psi = torch.rand(gt_rigids.shape+(2,))
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(
            gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :self.conf.experiment.n_bb]
        atom37_mask = atom37_mask[:, :, :self.conf.experiment.n_bb]
        only_c_alpha = torch.zeros(self.conf.experiment.n_bb).bool().to(device)
        only_c_alpha[1] = True
        has_atom_mask = (~is_sm[...,None] + only_c_alpha)
        atom37_mask = has_atom_mask

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]

        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)

        bb_atom_loss *= t < self._exp_conf.bb_atom_loss_t_filter
        loss_dict['bb_atom'] = bb_atom_loss


        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*self.conf.experiment.n_bb, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        # Insert iteration dimension if not present
        pred_flat_atoms = torch.flatten(pred_atom37, start_dim=-3, end_dim=-2)
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[..., None, :] - pred_flat_atoms[..., None, :, :], dim=-1)
        flat_loss_mask = torch.tile(has_atom_mask, (1, 1, 1, 1)) # unsqueeze
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*self.conf.experiment.n_bb])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_loss_mask[:, None, :]

        # No loss on anything >6A
        # FOR DEBUGGING:
        # proximity_mask = gt_pair_dists < 60000000
        proximity_mask = gt_pair_dists < 60000000
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(-2, -1))
        dist_mat_loss_normalization = (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        if dist_mat_loss_normalization == 0:
            dist_mat_loss = 0
        else:
            dist_mat_loss /= dist_mat_loss_normalization
        dist_mat_loss *= t < self._exp_conf.dist_mat_loss_t_filter
        loss_dict['dist_mat'] = dist_mat_loss

        # Auxillary losses to help with full-atom (nucleic) coordinate generation:
        if 'fa_disp_loss_weight' in self.conf.experiment:
            # Full-atom displacement loss (necessary for learning/remembering how to predict good alphas):
            # (this matters for nucleic acids, since the sugar backbone needs torsions for generation)
            # Define mask: compute for defined atoms (all if known seq, just backbone if diffused)
            mask_seq = nucl_utils.get_full_mask_seq(label_aa_s[0,0,:])[None,None,:]
            seq_for_scoring = torch.where(~mask_aa_s, label_aa_s, mask_seq)
            unk_crds_mask = ChemData().allatom_mask.to(seq_for_scoring.device)[seq_for_scoring[0,0,:]]
            unk_crds_mask[:, ChemData().NHEAVY:].fill_(False)
            unk_crds_mask = torch.tile(unk_crds_mask[None,:,:,None], (1,1,1,3))

            I, B, L, _, _ = pred_in.shape
            _, pred_xyz_fa = self.allatom_converter.compute_all_atom(
                                torch.tile(seq_for_scoring[:,0,:], (I,1)), 
                                pred_in[:,0,:,:,:], 
                                pred_tors[:,0,:,:,:]
                                )
            pred_fa_crds = pred_xyz_fa * self.conf.diffuser.r3.coordinate_scaling
            gt_fa_crds = torch.where(
                                ~torch.isnan(true), true, torch.zeros_like(true)
                                ) * self.conf.diffuser.r3.coordinate_scaling
            fa_crds_loss_mask = unk_crds_mask * loss_mask[:,:,None,None]
            # calc displacement, and mask accordingly:
            fa_disp_loss = torch.sum(
                ((gt_fa_crds - pred_fa_crds) ** 2) * fa_crds_loss_mask,
                dim=(-1, -2, -3)
            ) / (torch.sum(fa_crds_loss_mask) + eps)
            # Add full-atom loss to dict, weight only if at low t:
            fa_disp_loss *= t < self._exp_conf.bb_atom_loss_t_filter
            loss_dict['fa_disp'] = fa_disp_loss



        # C6D loss
        for i, label in enumerate(['dist', 'omega', 'theta', 'phi']):
            loss = self.loss_fn(logit_s[i], label_s[...,i]) # (B, L, L)
            loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
            loss_dict[f'c6d_{label}'] = loss.clone()


        # FAPE losses
        if self.conf.loss.use_fapes:
            # find out the small molecule lengths 
            sm_Ls = rf_diffusion.util.get_sm_lengths(indep.is_sm.clone().cpu().detach().numpy(), 
                                                     indep.same_chain.clone().cpu().detach().numpy())

            L_tot = indep.length()
            L_prot = L_tot - sum(sm_Ls)
            fape_mask = torch.zeros_like(mask_crds).to(dtype=torch.bool)
            fape_mask[:,:L_prot,:3] = True
            fape_mask[:,L_prot:,1] = True

            fape_kwargs = { 'masks_1d'       : masks_1d,     # dict - masks from mask_generator
                            'pred_in'        : pred_in,
                            'indep'          : indep,        # contains true structure and sequence for scoring 
                            'mask_crds'      : fape_mask,
                            'fi_dev'         : self.fi_dev,
                            'atom_frames'    : atom_frames, 
                            'sm_Ls'          : sm_Ls,  
                            'conf'           : self.conf,       
                            'diffusion_mask' : ~masks_1d['was_noised_in_3d']
                      }   

            # some FAPE losses will often be NaN coming out of this function, b.c.
            # some fapes are undefined if e.g., no motif (unconditional)
            fape_loss_dict, rmsd_dict = compute_fape_losses(**fape_kwargs)
            good_flosses = {k:v for k,v in fape_loss_dict.items() if not torch.isnan(v).any()}
            nan_flosses = {k:v for k,v in fape_loss_dict.items() if torch.isnan(v)}
            
            loss_dict.update(good_flosses)
            aux_data.update(rmsd_dict)

            # loss_dict will need the NaN losses too for logging, since
            # the weights are nonzero. 
            for k,v in nan_flosses.items():
                aux_data[f'weighted.{k}'] = v # it's nan, weight doesn't matter!

        # Average over batches
        for k, loss in loss_dict.items():
            loss_dict[k] = loss.sum(dim=0)
        
        tot_loss = 0
        for k, loss in loss_dict.items():
            mean_block_loss = loss_module.normalize_loss(loss, gamma=self.conf.experiment.gamma)
            aux_data[f'mean_block.{k}'] = mean_block_loss
            last_block_loss = loss
            if last_block_loss.ndim == 1:
                assert last_block_loss.shape == (40,)
                last_block_loss = last_block_loss[-1]
            aux_data[f'last_block.{k}'] = last_block_loss
            weight = loss_weights[k]
            aux_data[f'weight.{k}'] = weight
            weighted_loss = mean_block_loss*weight
            aux_data[f'weighted.{k}'] = weighted_loss

            tot_loss += weighted_loss
        
        return tot_loss, aux_data

    def calc_acc(self, prob, dist, idx_pdb, mask_2d, return_cnt=False):
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1] # (B, L)
        seqsep = torch.abs(idx_pdb[:,:,None] - idx_pdb[:,None,:]) + 1
        mask = seqsep > 24
        mask = torch.triu(mask.float())
        mask *= mask_2d
        #
        cnt_ref = dist < 20
        cnt_ref = cnt_ref.float() * mask
        #
        cnt_pred = prob[:,:20,:,:].sum(dim=1) * mask
        #
        top_pred = torch.topk(cnt_pred.view(B,-1), L)
        kth = top_pred.values.min(dim=-1).values
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        cnt_pred = torch.stack(tmp_pred, dim=0)
        cnt_pred = cnt_pred.float()*mask
        #
        condition = torch.logical_and(cnt_pred==cnt_ref, cnt_ref==torch.ones_like(cnt_ref))
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        n_total_pred = (cnt_pred == torch.ones_like(cnt_pred)).float().sum() + 1e-9
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0*prec*recall / (prec+recall+1e-9)
        if return_cnt:
            return torch.stack([prec, recall, F1]), cnt_pred, cnt_ref

        return torch.stack([prec, recall, F1])

    def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=False):

        chk_fn = self.conf.ckpt_load_path
        # Prevents mistakes in training: this is the config field used for inference.
        assertpy.assert_that(self.conf.rf.ckpt_path).is_false()

        loaded_epoch = 0
        best_valid_loss = 999999.9
        if self.conf.zero_weights:
            return 0, best_valid_loss
        if not os.path.exists(chk_fn):
            raise Exception(f'no model found at path: {chk_fn}, pass -zero_weights if you intend to train the model with no initialization and no starting weights')
        print('*** FOUND MODEL CHECKPOINT ***')
        print('Located at ',chk_fn)

        if isinstance(rank, str):
            # For CPU debugging
            map_location = {"cuda:%d"%0: "cpu"}
        else:
            map_location = {"cuda:%d"%0: "cuda:%d"%rank}
            ic(f'loading model onto {"cuda:%d"%rank}')
        ic(chk_fn)
        checkpoint = torch.load(chk_fn, map_location=map_location, weights_only=False)
        # Set to false for faster loading when debugging
        cautious = True
        for m, weight_state in [
            (model.module.model, checkpoint['final_state_dict']),
            (model.module.shadow, checkpoint['model_state_dict']),
        ]:
            if self.conf.reinitialize_missing_params:
                raise Exception('stop')
                model_state = m.state_dict()
                if cautious:
                    new_chk = {}
                    for param in model_state:
                        if param not in weight_state:
                            print ('missing',param)
                        elif (weight_state[param].shape == model_state[param].shape):
                            new_chk[param] = weight_state[param]
                        else:
                            print (
                                'wrong size',param,
                                weight_state[param].shape,
                                model_state[param].shape )

                else:
                    new_chk = weight_state
                
                m.load_state_dict(new_chk, strict=False)
            else:
                # Handle loading from structure prediction model.
                model_name = ''
                model_name = checkpoint.get('model_name', '')
                if not model_name:
                    model_name = getattr(checkpoint.get('model'), '__name__', '')
                if model_name != 'RFScore':
                    weight_state = {f'model.{k}':v for k,v in weight_state.items()}

                m.load_state_dict(weight_state, strict=True)

        if resume_train:
            loaded_epoch = checkpoint['epoch']
        
        if self.conf.resume_scheduler:
            print (' ... loading optimization params')
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                #print (' ... loading scheduler params')
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                if self.conf.get('override_scheduler_safe_loading', False):
                    scheduler.last_epoch = (loaded_epoch)*self.conf.epoch_size / self.conf.pseudobatch
                else:
                    assert False, 'scheduler_state_dict not found in checkpoint, to override and recreate scheduler at the current epoch, set override_scheduler_safe_loading: True in config'
            #if 'best_loss' in checkpoint:
            #    best_valid_loss = checkpoint['best_loss']
        
        
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists(f"{self.outdir}/models"):
            os.mkdir(f"{self.outdir}/models")
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join(f"{self.outdir}/models", name)
    
    # main entry function of training
    # 1) make sure ddp env vars set
    # 2) figure out if we launched using slurm or interactively
    #   - if slurm, assume 1 job launched per GPU
    #   - if interactive, launch one job for each GPU on node
    def run_model_training(self, world_size):
        if self.conf.resume:
            assert self.conf.resume_scheduler
        is_ddp = (not self.conf.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ)
        if is_ddp:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
        if ('MASTER_ADDR' not in os.environ or os.environ['MASTER_ADDR'] == ''):
            if world_size <= 1:
                os.environ['MASTER_ADDR'] = 'localhost'
            else:
                os.environ['MASTER_ADDR'] = master_addr.get()
        if self.conf.master_port:
            os.environ['MASTER_PORT'] = f'{self.conf.master_port}'
        if ('MASTER_PORT' not in os.environ):
            assertpy.assert_that(world_size).is_less_than(2)
            os.environ['MASTER_PORT'] = f'{random.randint(12000, 12200)}'

        if is_ddp:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()
            if world_size <= 1:
                self.train_model(0, 1)
            else:
                mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)
    
    def load_ckpt(self, rank):
        chk_fn = self.conf.ckpt_load_path
        if isinstance(rank, str):
            # For CPU debugging
            map_location = {"cuda:%d"%0: "cpu"}
        else:
            map_location = {"cuda:%d"%0: "cuda:%d"%rank}
            ic(f'loading model onto {"cuda:%d"%rank}')
        ic(chk_fn)
        checkpoint = torch.load(chk_fn, map_location=map_location, weights_only=False)
        return checkpoint

    def train_model(self, rank, world_size, return_setup=False):

        self.accum_step = self.conf.pseudobatch / world_size
        ic(self.accum_step)
        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        
        ic(os.environ['MASTER_ADDR'], rank, world_size, torch.cuda.device_count())
        print(f'{rank=} {world_size=} {self.conf.ddp_backend=} initializing process group')
        dist.init_process_group(backend=self.conf.ddp_backend, world_size=world_size, rank=rank)
        print(f'{rank=} {world_size=} initialized process group')
        print(f'DEBUG2 print: {rank=} {world_size=} initialized process group')
        ic(f'DEBUG2 ic: {rank=} {world_size=} initialized process group')
        if torch.cuda.device_count():
            gpu = rank % torch.cuda.device_count()
            torch.cuda.set_device("cuda:%d"%gpu)
            ic(rank, torch.cuda.get_device_name(f'cuda:{gpu}'))
        else:
            gpu = 'cpu'
        
        resume = 'never'
        id = None
        entity="bakerlab"
        if not self.conf.resume:
            date_string = get_datetime()
            obj_list = [date_string]
            dist.broadcast_object_list(obj_list, 0)
            date_string = obj_list[0]
            wandb_name = self.conf.wandb_prefix + date_string
            self.rundir = self.make_rundir(wandb_name)
        else:
            checkpoint = self.load_ckpt(gpu)
            self.rundir = checkpoint['rundir']
            wandb_name = checkpoint['wandb_group']
            if WANDB:
                resume = 'must'
                api = wandb.Api()
                name=f'{wandb_name}_rank_{rank}'
                runs = list(api.runs(
                        path=f'{entity}/{self.conf.wandb_project}',
                        filters={"display_name": {"$eq": name}}
                    ))
                if len(runs) == 1:
                    id = runs[0].id
                elif len(runs) == 0:
                    assert self.conf.allow_more_gpus_on_resume
                    # Check that at least rank 0 exists
                    rank_0_name = f'{wandb_name}_rank_0'
                    runs = list(api.runs(
                            path=f'{entity}/{self.conf.wandb_project}',
                            filters={"display_name": {"$eq": rank_0_name}}
                        ))
                    assert len(runs) == 1, f'resume=True and allow_more_gpus_on_resume=True, but {len(runs)} != 1 matching {rank_0_name} found: {[r.display_name for r in runs]}'
                    resume = 'allow'
                else:
                    raise Exception(f'resume=True, but {len(runs)} matching {name} found: {[r.display_name for r in runs]}')
        
        if rank == 0 and self.conf.benchmark and self.conf.benchmark.run_early_validation:
            print(f'Running early validation with config: {self.conf.benchmark}')
            self.benchmark_model(self.conf.ckpt_load_path, self.conf.benchmark, outdir=self.rundir + '/early_validation')

        if WANDB:
            print(f'initializing wandb on rank {rank}')
            wandb.init(
                dir=self.conf.wandb_dir,
                project=self.conf.wandb_project,
                entity=entity,
                group=wandb_name,
                name=f'{wandb_name}_rank_{rank}',
                id=id,
                resume=resume
            )
            print(f'{wandb.run.id=}')
            self.wandb_run_id = wandb.run.id
            self.wandb_group = wandb_name
            wandb.config = self.conf

        self.outdir = os.path.join(self.rundir, f'rank_{rank}')
        print(f'{rank=} {self.outdir=}')
        os.makedirs(self.outdir, exist_ok=True)
        
        if rank == 0:
            print(f'Saving git diff between current state and last commit and git log to {self.outdir}')
            for cmd in ['pwd', 'git log', 'git diff']:
                path = f'{self.outdir}/{"_".join(cmd.split())}.txt'
                with open(path, 'w') as fh:
                    subprocess.Popen([cmd], cwd = os.getcwd(), shell=True, stdout=fh, stderr=subprocess.PIPE)
                if WANDB:
                    wandb.save(os.path.join(os.getcwd(), path))
        
        self.n_train = N_EXAMPLE_PER_EPOCH

        # Get the fallback dataset and dataloader
        train_set, train_loader = get_fallback_dataset_and_dataloader(
            conf=self.conf,
            diffuser=self.diffuser,
            num_example_per_epoch=N_EXAMPLE_PER_EPOCH,
            world_size=world_size,
            rank=rank,
            LOAD_PARAM=LOAD_PARAM,
        )

        # move some global data to cuda device
        self.ti_dev = self.ti_dev.to(gpu)
        self.fi_dev = self.fi_dev.to(gpu)
        self.ti_flip = self.ti_flip.to(gpu)
        self.ang_ref = self.ang_ref.to(gpu)
        self.allatom_converter = self.allatom_converter.to(gpu)
       
        # define model
        print('Making model...')
        ddp_model, optimizer, scheduler, scaler, loaded_epoch = self.init_model(gpu)

        if return_setup:
            return ddp_model, train_loader, optimizer, scheduler, scaler
        
        for epoch in range(loaded_epoch, self.conf.n_epoch):
            train_loader.sampler.set_epoch(epoch)
            print('Just before calling train cycle...')
            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch)
            if rank == 0: # save model
                model_path = self.save_model(epoch+1, ddp_model, optimizer, scheduler, scaler)
                if self.conf.benchmark:
                    self.benchmark_model(model_path, self.conf.benchmark)

        dist.destroy_process_group()

    @staticmethod
    def benchmark_model(model_path, benchmark_config, outdir=None):
        '''
        If outdir is None, then the benchmark results will be saved in the same directory as the model checkpoint.
        '''
        # Make dir for benchmark results
        model_dir, model_tail = os.path.split(model_path)
        model_name, _ = os.path.splitext(model_tail)
        outdir = outdir or model_dir
        benchmark_dir = os.path.join(outdir, 'auto_benchmark', model_name, 'out')
        os.makedirs(benchmark_dir, exist_ok=True)

        # Make overrides
        overrides=[
            f'sweep.command_args="--config-name={benchmark_config["inference_yaml"]} inference.ckpt_path={model_path}"',
            f'outdir={benchmark_dir}',
        ]

        # Dump a yaml file to be used by the pipeline slurm job
        conf = construct_conf(
            overrides=overrides,
            yaml_path=f'{PKG_DIR}/benchmark/configs/{benchmark_config["pipeline_yaml"]}',
        )
        benchmark_yaml = f'{benchmark_dir}/pipeline.yaml'
        OmegaConf.save(conf, benchmark_yaml)

        # Submit slurm job to run the pipeline
        config_path, config_name = os.path.split(benchmark_yaml)
        cmd = f'"{PKG_DIR}/benchmark/pipeline.py --config-path={config_path} --config-name={config_name}"'
        print(f'Running mid-training benchmark of model {model_path} with command: {cmd}')
        cmd_sbatch = (
            f'sbatch -t 96:00:00 -J autobench_{model_name} -o {benchmark_dir}/slurm-%j.out --export PYTHONPATH={REPO_DIR} '
            f'--wrap {cmd}'
        )
        proc = subprocess.run(cmd_sbatch, shell=True, stdout=subprocess.PIPE)
        slurm_job = re.findall(r'\d+', str(proc.stdout))[0]
        slurm_job = int(slurm_job)
        print(f'Submitted slurm job {slurm_job} to benchmark model checkpoint {model_path}.')

    def save_model(self, suffix, ddp_model, optimizer, scheduler, scaler):
        #save every epoch     
        model_path = self.checkpoint_fn(self.conf.model_name, str(suffix))
        print(f'saving model to {model_path}')
        torch.save({
                    'model': ddp_model.module.model.__class__,
                    'epoch': suffix,
                    #'model_state_dict': ddp_model.state_dict(),
                    'model_state_dict': ddp_model.module.shadow.state_dict(),
                    'final_state_dict': ddp_model.module.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'conf':self.conf,
                    'wandb_run_id':self.wandb_run_id,
                    'wandb_group':self.wandb_group,
                    'rundir': self.rundir,
                    },
                    model_path)
        return model_path

    def init_model(self, device):
        from rf_diffusion.frame_diffusion.rf_score.model import RFScore
        model = RFScore(self.conf.rf.model, self.diffuser, device).to(device)
        # if self.log_inputs:
        #     pickle_dir, self.pickle_counter = pickle_function_call(model, 'forward', 'training', minifier=aa_model.minifier)
        #     print(f'pickle_dir: {pickle_dir}')
        if self.conf.verbose_checks:
            model.verbose_checks = True
        model = EMA(model, 0.999)
        print('Instantiating DDP')
        ddp_model = model
        ic(device)
        if torch.cuda.device_count():
            ddp_model = DDP(model, device_ids=[f'cuda:{device}'], find_unused_parameters=False, broadcast_buffers=False)
        else:
            ddp_model = DDP(model, find_unused_parameters=False)
        print('Initializing optimizer')
        # if rank == 0:
        #     print ("# of parameters:", count_parameters(ddp_model))
        
        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.conf.l2_coeff)
        optimizer = torch.optim.AdamW(opt_params, lr=self.conf.lr)
        scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, self.conf.scheduler.n_warmup_steps, 10000, 0.95) # for fine-tuning
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
       
        # load model
        print('About to load model...')
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, optimizer, scheduler, scaler, 
                                                       self.conf.model_name, device, resume_train=self.conf.resume or self.conf.resume_scheduler)

        print('Done loading model')

        return ddp_model, optimizer, scheduler, scaler, loaded_epoch

        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch):

        loss_weights = {
            'trans_score': self._exp_conf.get('trans_score_weight', self._exp_conf.trans_loss_weight),
            'trans_x0':  self._exp_conf.trans_loss_weight,
            'rot_score': self._exp_conf.rot_loss_weight,
            'bb_atom': self._exp_conf.bb_atom_loss_weight * self._exp_conf.aux_loss_weight,
            'dist_mat': self._exp_conf.dist_mat_loss_weight * self._exp_conf.aux_loss_weight,
            'c6d_dist': self._exp_conf.c6d_loss_weight,
            'c6d_phi': self._exp_conf.c6d_loss_weight,
            'c6d_theta': self._exp_conf.c6d_loss_weight,
            'c6d_omega': self._exp_conf.c6d_loss_weight,
            # FM
            'fm_translation': self._exp_conf.fm_translation_loss_weight,
            'fm_rotation': self._exp_conf.fm_rotation_loss_weight,
            'fm_bb_atom': self._exp_conf.fm_bb_atom,
            'fm_dist_mat': self._exp_conf.fm_dist_mat,
            'i_fm_translation': self._exp_conf.i_fm_translation_loss_weight if 'i_fm_translation_loss_weight' in self._exp_conf else 0.0,   
            'i_fm_rotation': self._exp_conf.i_fm_rotation_loss_weight if 'i_fm_rotation_loss_weight' in self._exp_conf else 0.0,            
            # Extras:
            'fa_disp': self._exp_conf.fa_disp_loss_weight if 'fa_disp_loss_weight' in self._exp_conf else 0.0,     
            # FAPEs
            'motif_fape': self._exp_conf.get('motif_fape_weight'),
            'nonmotif_fape': self._exp_conf.get('nonmotif_fape_weight'),
            'ligand_intra_fape': self._exp_conf.get('ligand_intra_fape_weight'),
            'prot_lig_inter_fape': self._exp_conf.get('prot_lig_inter_fape_weight')
        }

        print('Entering self.train_cycle')
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        from pytimer import Timer
        timer = Timer()
        
        counter = 0
        
        print('About to enter train loader loop')

        for loader_out in train_loader:
            timer.checkpoint('data loading')
            indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

            context_msg = f'rank: {rank}: {item_context} Size: {rfi.xyz.shape} Mask: {masks_1d["mask_name"]}'
            with error.context(context_msg):
                N_cycle = np.random.randint(1, self.conf.maxcycle+1) # number of recycling

                # Defensive assertions
                assert little_t > 0
                
                
                if self.conf.inference['str_self_cond']:
                    assert self.conf['prob_self_cond'] > 0, 'prob_self_cond must be > 0 for str_self_cond to be active'

                # Checking whether this example was of poor quality and the dataloader just returned None - NRB
                if indep.seq.shape[0] == 0:
                    ic('Train cycle received bad example, skipping')
                    continue

                # Save trues for writing pdbs later.
                xyz_prev_orig = rfi.xyz[0]

                # transfer inputs to device
                B, _, L, _ = rfi.msa_latent.shape
                rf2aa.tensor_util.to_device(rfi, gpu)

                counter += 1 


                # get diffusion_mask for the displacement loss


                use_cb = self.conf.preprocess.use_cb_to_get_pair_dist

                # Some percentage of the time, provide the model with the model's prediction of x_0 | x_t+1
                # When little_t == T should not unroll as we cannot go back further in time.
                self_cond = not (little_t == self.conf.diffuser.T) and (torch.tensor(self.conf['prob_self_cond']) > torch.rand(1))
                timer.checkpoint('device loading')
                if self_cond:
                    rf2aa.tensor_util.to_device(rfi, gpu)

                    # Take 1 step back in time to get the training example to feed to the model
                    # For this model evaluation msa_prev, pair_prev, and state_prev are all None and i_cycle is
                    # constant at 0
                    with torch.no_grad():
                        with ddp_model.no_sync():
                            with torch.cuda.amp.autocast(enabled=USE_AMP):
                                model_out = ddp_model.forward(
                                        rfi,
                                        torch.tensor([little_t/self.conf.diffuser.T]),
                                        use_checkpoint=False, # Checkpointing unnecessary since there are no gradients.
                                        # return_raw=False
                                        )
                                rfo = model_out['rfo']
                                rfi = aa_model.self_cond(indep, rfi, rfo, use_cb=use_cb)
                                xyz_prev_orig = rfi.xyz[0,:,:ChemData().NHEAVY].clone()

                timer.checkpoint('self-conditioning')

                with ExitStack() as stack:
                    if counter%self.accum_step != 0:
                        stack.enter_context(ddp_model.no_sync())
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        model_out = ddp_model.forward(
                                        rfi,
                                        torch.tensor([little_t/self.conf.diffuser.T]),
                                        use_checkpoint=True,
                                        N_cycle=N_cycle,
                                        training=True)
                        model_out['rigids'].retain_grad()

                        timer.checkpoint('model forward')
                        logit_s, logit_aa_s, logits_pae, logits_pde, p_bind, pred_crds, alphas, px0_allatom, pred_lddts, _, _, _, _ = model_out['rfo'].unsafe_astuple()

                        is_diffused = is_diffused.to(gpu)
                        indep.seq = indep.seq.to(gpu)
                        indep.xyz = indep.xyz.to(gpu)
                        indep.same_chain = indep.same_chain.to(gpu)
                        indep.idx = indep.idx.to(gpu)
                        true_crds = torch.zeros((1,L,36,3)).to(gpu)
                        true_crds[0,:,:ChemData().NHEAVY,:] = indep.xyz[:,:ChemData().NHEAVY,:]
                        mask_crds = ~torch.isnan(true_crds).any(dim=-1)
                        true_crds, mask_crds = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
                        mask_crds[:,~is_diffused,:] = False
                        mask_BB = ~indep.is_sm[None]
                        mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
                        # assert torch.sum(mask_2d) > 0, "mask_2d is blank"
                        true_crds_frame = rf2aa.util.xyz_to_frame_xyz(true_crds, indep.seq[None], indep.atom_frames[None])
                        kinematics_params = copy.deepcopy(rf2aa.kinematics.PARAMS)
                        kinematics_params['USE_CB'] = use_cb
                        c6d = rf2aa.kinematics.xyz_to_c6d(true_crds_frame, params=kinematics_params)
                        #c6d = rf2aa.kinematics.xyz_to_c6d(true_crds_frame)
                        negative = torch.tensor([False])
                        c6d = rf2aa.kinematics.c6d_to_bins(c6d, indep.same_chain[None], negative=negative)

                        label_aa_s = indep.seq[None, None]
                        mask_aa_s = is_diffused[None, None]
                        same_chain = indep.same_chain[None]
                        seq_diffusion_mask = torch.ones(L).bool()
                        seq_t = torch.nn.functional.one_hot(indep.seq, 80)[None].float()
                        xyz_t = rfi.xyz[None]
                        unclamp = torch.tensor([False])

                        seq_diffusion_mask[:] = True
                        mask_crds[:] = False
                        # remove hydrogens                      
                        mask_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot')
                        mask_na = nucl_utils.get_resi_type_mask(indep.seq, 'na')
                        true_crds[:,mask_prot,ChemData().NHEAVYPROT:] = 0
                        true_crds[:,mask_na,ChemData().NHEAVY:] = 0
                        xyz_t[:] = 0
                        seq_t[:] = 0
                        loss, loss_dict = self.calc_loss(indep, loss_weights, model_out, diffuser_out, logit_s, c6d,
                                logit_aa_s, label_aa_s, mask_aa_s, None,
                                pred_crds, alphas, true_crds, mask_crds,
                                mask_BB, mask_2d, same_chain,
                                pred_lddts, indep.idx[None], chosen_dataset, chosen_task, is_diffused=is_diffused,
                                seq_diffusion_mask=seq_diffusion_mask, seq_t=seq_t, xyz_in=xyz_t, is_sm=indep.is_sm, unclamp=unclamp,
                                negative=negative, t=little_t, atomizer=atomizer, masks_1d=masks_1d, atom_frames=rfi.atom_frames)
                        # Force all model parameters to participate in loss. Truly a cursed workaround.
                        loss += 0.0 * (
                            logits_pae.mean() +
                            logits_pde.mean() +
                            alphas.mean() +
                            pred_lddts.mean() +
                            p_bind.mean() +
                            logit_aa_s.mean() +
                            sum(l.mean() for l in logit_s)
                        )
                    loss = loss / self.accum_step
                    timer.checkpoint('loss calculation')

                    # if gpu != 'cpu':
                    #     print(f'DEBUG: {rank=} {gpu=} {counter=} size: {indep.xyz.shape[0]} {torch.cuda.max_memory_reserved(gpu) / 1024**3:.2f} GB reserved {torch.cuda.max_memory_allocated(gpu) / 1024**3:.2f} GB allocated {torch.cuda.get_device_properties(gpu).total_memory / 1024**3:.2f} GB total')
                    if not torch.isnan(loss) and not self.conf.skip_backward:
                        scaler.scale(loss).backward()
                    else:
                        msg = f'NaN loss encountered, skipping: {context_msg}'
                        weighted_losses = {k:v for k,v in loss_dict.items() if k.startswith('weighted') and v != 0}
                        msg += f" weighted_losses: {weighted_losses}"
                        if not DEBUG:
                            print(msg)
                        else:
                            raise Exception(msg)
                    timer.checkpoint('loss backwards')
                    grad_norm = np.nan
                    if counter%self.accum_step == 0:
                        if rank == 0:
                            print(f'ACCUMULATING {counter=}')
                        # gradient clipping
                        scaler.unscale_(optimizer)
                        grad_norm=torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
                        grad_norm = grad_norm.cpu().detach().item()
                        scaler.step(optimizer)
                        scale = scaler.get_scale()
                        scaler.update()
                        new_scale = scaler.get_scale()
                        skip_lr_sched = (scale != new_scale)
                        optimizer.zero_grad()
                        if not skip_lr_sched:
                            scheduler.step()
                        timer.checkpoint('scaling')
                        ddp_model.module.update() # apply EMA
                        timer.checkpoint('ddp update')
                
                ## check parameters with no grad
                #if rank == 0:
                #    for n, p in ddp_model.named_parameters():
                #        if p.grad is None and p.requires_grad is True:
                #            print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model

                metrics_inputs = dict(
                    indep=indep,
                    pred_crds=pred_crds[-1, 0].cpu(),
                    true_crds=true_crds[0, :, :3].cpu(),
                    input_crds=xyz_prev_orig[:, :3].cpu(),
                    t=little_t/self.diffuser.T,
                    is_diffused=is_diffused.cpu(),
                    point_types=aa_model.get_point_types(indep, atomizer),
                    pred_crds_stack=pred_crds[:, 0].cpu(),
                    atomizer_spec=None,
                )
                pdb_dir = os.path.join(self.outdir, 'training_pdbs')
                n_processed = self.conf.batch_size*world_size * counter
                can_log=True
                output_pdb_prefix = f'{pdb_dir}/epoch_{epoch}_{n_processed}_{chosen_task}_{chosen_dataset}_t_{int( little_t )}'
                log_metrics = self.conf.log_every_n_examples and (counter % self.conf.log_every_n_examples == 0) and can_log
                save_pdb = self.conf.n_write_pdb and (counter % self.conf.n_write_pdb == 0) and can_log
                if log_metrics:
                    train_time  = time.time() - start_time
                    
                    if 'diff' in chosen_task:
                        task_str = f'diff_t{int(little_t)}'
                    else:
                        task_str = chosen_task
                    
                    outstr = f"Task: {task_str} | Dataset: {chosen_dataset : >12} | Epoch:[{epoch:02}/{self.conf.n_epoch}] | Batch: [{counter*self.conf.batch_size*world_size:05}/{self.n_train}] | Time: {train_time:.2f}"
                    outstr += f' | t={little_t/self.diffuser.T}'

                    outstr += (f' | Loss: {round( float(loss * self.accum_step), 4)} = \u03A3 ')
                    str_stack = []
                    for k, weight in list(loss_weights.items()):
                        if weight == 0:
                            continue
                        weighted = loss_dict[f'weighted.{k}']
                        str_stack.append(f'{k}: {float(weighted):4.2f}')
                        # Hack to not log these to wandb, except for scores, we love scores.
                        if weighted == 0 and 'score' not in k:
                            loss_dict[f'weighted.{k}'] = float('nan')
                            loss_dict[f'mean_block.{k}'] = float('nan')
                            loss_dict[f'last_block.{k}'] = float('nan')
                    outstr += '  '.join(str_stack)
                    if rank == 0:
                        sys.stdout.write(outstr+'\n')

                    try:
                        r3_grad = model_out['rigids'].grad[...,4:].detach().cpu()
                    except Exception as e:
                        r3_grad = torch.tensor(0.0)
                        print(f'WARNING: getting r3_grad raised exception: {format_exception(e)}')

                    loss_dict.update({
                        'training_start': firstLog(),
                        't':little_t,
                        'total_examples':epoch*self.n_train+counter*world_size,
                        'epoch': epoch,
                        'rank': rank,
                        'item_context': item_context,
                        'dataset':chosen_dataset,
                        'task':chosen_task,
                        'self_cond': self_cond,
                        'extra_t1d': indep.extra_t1d.cpu().detach() if hasattr(indep.extra_t1d, 'cpu') else indep.extra_t1d,
                        'extra_t2d': indep.extra_t2d.cpu().detach() if hasattr(indep.extra_t2d, 'cpu') else indep.extra_t2d,
                        'loss':loss.detach() * self.accum_step,
                        'output_pdb_prefix':output_pdb_prefix,
                        'use_guideposts': masks_1d['use_guideposts'],
                        'mask_name': masks_1d['mask_name'],
                        'grad_norm': grad_norm,
                        'lr': scheduler.get_last_lr()[0],
                        'length': indep.length(),
                        'save_pdb': save_pdb,
                        'r3_grad_norm': torch.linalg.vector_norm(r3_grad).item(),
                        'r3_grad_max': torch.linalg.vector_norm(r3_grad, ord=torch.inf).item(),
                    })

                    rf2aa.tensor_util.to_device(indep, 'cpu')
                                
                    loss_dict['meta'] = {
                        'n_atomized_residues': len(masks_1d['is_atom_motif'])
                    }
                    # loss_dict['loss_weights'] = loss_weights
                    if WANDB:
                        wandb.log(loss_dict)
                    if rank == 0:
                        sys.stdout.flush()
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                timer.checkpoint('logging')

                if save_pdb:
                    (L,) = indep.seq.shape
                    os.makedirs(pdb_dir, exist_ok=True)

                    rf2aa.tensor_util.to_device(indep, 'cpu')
                    pred_xyz = xyz_prev_orig.clone()
                    pred_xyz[:,:3] = pred_crds[-1, 0]

                    for suffix, xyz in [
                        ('input', xyz_prev_orig),
                        ('pred', pred_xyz),
                        ('true', indep.xyz),
                    ]:
                        indep_write = copy.deepcopy(indep)
                        indep_write.xyz[:,:ChemData().NHEAVY] = xyz[:,:ChemData().NHEAVY]
                        pymol_names = indep_write.write_pdb(f'{output_pdb_prefix}_{suffix}.pdb')
                timer.checkpoint('pdb writing')

                log_metrics_inputs = self.conf.log_metrics_inputs_every_n_examples and (counter % self.conf.log_metrics_inputs_every_n_examples == 0) and can_log
                if log_metrics_inputs:
                    indep_true = indep
                    motif_deatomized = None
                    if not save_pdb:
                        pymol_names = "not available, did not write pdb"
                        os.makedirs(pdb_dir, exist_ok=True)

                    with open(f'{output_pdb_prefix}_info.pkl', 'wb') as fh:
                        pickle.dump({
                            'metrics_inputs': metrics_inputs,
                            'motif': motif_deatomized,
                            'masks_1d': masks_1d,
                            'idx': indep_true.idx,
                            'extra_t1d': indep_true.extra_t1d,
                            'extra_t2d': indep_true.extra_t2d,
                            'is_sm': indep_true.is_sm,
                            'pymol_names': pymol_names,
                            'dataset': chosen_dataset,
                            'little_t': float(little_t),
                            # Redundant.
                            # 'loss_dict': tree.map_structure(
                            #     lambda x: x.cpu() if hasattr(x, 'cpu') else x, loss_dict)
                        }, fh)

                    if self.conf.log_inputs:
                        shutil.copy(self.pickle_counter.last_pickle, f'{output_pdb_prefix}_input_pickle.pkl')

                # Expected epoch time logging
                if rank == 0:
                    now = time.time()
                    elapsed_time = now - start_time
                    mean_rate = n_processed / elapsed_time
                    expected_epoch_time = int(self.n_train / mean_rate)
                    m, s = divmod(expected_epoch_time, 60)
                    h, m = divmod(m, 60)
                    print(f'Expected time per epoch of size ({self.n_train}) (h:m:s) based off {counter} measured pseudo batch times: {h:d}:{m:02d}:{s:.0f}   ' \
                          f'Examples / (GPU second): {mean_rate / world_size:.4f}')

                if self.conf.saves_per_epoch and rank==0:
                    n_processed_next = self.conf.batch_size*world_size * (counter+1)
                    n_fractionals = np.arange(1, self.conf.saves_per_epoch) / self.conf.saves_per_epoch
                    for fraction in n_fractionals:
                        save_before_n = fraction * self.n_train
                        if n_processed <= save_before_n and n_processed_next > save_before_n:
                            self.save_model(f'{epoch + fraction:.2f}', ddp_model, optimizer, scheduler, scaler)
                            break
                timer.checkpoint('metrics inputs logging')
                timer.restart()
                if rank == 0 and (counter % self.conf.timing_summary_every) == 0:
                    timer.summary()
                
                if self.conf.restart_timer:
                    timer = Timer()

        # TODO(fix or delete)
        # write total train loss
        # train_tot /= float(counter * world_size)
        # train_loss /= float(counter * world_size)
        # train_acc  /= float(counter * world_size)

        # dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        # dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        # dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        # train_tot = train_tot.cpu().detach()
        # train_loss = train_loss.cpu().detach().numpy()
        # train_acc = train_acc.cpu().detach().numpy()
        train_tot = train_loss = train_acc = -1

        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f \n"%(\
                    epoch, self.conf.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    ))
            sys.stdout.flush()

            
        return train_tot, train_loss, train_acc


def make_trainer(conf):
    
    global N_EXAMPLE_PER_EPOCH
    global DEBUG 
    global WANDB
    global LOAD_PARAM
    global LOAD_PARAM2
    # set epoch size 
    N_EXAMPLE_PER_EPOCH = conf.epoch_size 

    # set global debug and wandb params 
    WANDB = conf.wandb
    if conf.debug:
        DEBUG = True 
        # loader_param['DATAPKL'] = 'subsampled_dataset.pkl'
        # loader_param['DATAPKL_AA'] = 'subsampled_all-atom-dataset.pkl'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        ic.configureOutput(includeContext=True)
    else:
        DEBUG = False 

    # set load params based on debug
    global LOAD_PARAM
    global LOAD_PARAM2

    ic(test_utils.available_cpu_count())
    max_workers = conf.num_workers

    LOAD_PARAM = {'shuffle': False,
              'num_workers': max_workers,
              'pin_memory': conf.pin_memory}
    LOAD_PARAM2 = {'shuffle': False,
              'num_workers': max_workers,
              'pin_memory': conf.pin_memory}


    # set random seed
    run_inference.seed_all(conf.seed)

    mp.freeze_support()
    train = Trainer(
                    conf=conf)
    return train

def format_exception(e: Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

@hydra.main(version_base=None, config_path="config/training", config_name="base")
def run(conf: DictConfig) -> None:

    # Necessary to compose another contextual config (i.e. benchmarking config).
    GlobalHydra.instance().clear()

    if 'custom_chemical_config' in conf:
        reinitialize_chemical_data(**conf.custom_chemical_config) 
    prepare_pyrosetta(conf)
        
    train = make_trainer(conf=conf)
    train.run_model_training(torch.cuda.device_count())

if __name__ == "__main__":
    # master_addr_set('MASTER_ADDR' not in os.environ or os.environ['MASTER_ADDR'] == '')
    run()
