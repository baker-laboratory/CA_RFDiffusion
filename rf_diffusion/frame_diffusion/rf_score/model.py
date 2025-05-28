from icecream import ic
import assertpy
import numpy as np
import torch
import torch.nn as nn
from typing import List
from openfold.utils.rigid_utils import Rigid
from rf_diffusion.frame_diffusion.data import all_atom
import rf2aa.util
from dataclasses import dataclass
import dataclasses
from contextlib import nullcontext 

import rf2aa.model.RoseTTAFoldModel
from rf2aa.chemical import ChemicalData as ChemData
import rf_diffusion.aa_model
import addict
from scipy.interpolate import Akima1DInterpolator
import os
from datetime import datetime
from openfold.utils.rigid_utils import Rotation
from rf_diffusion.frame_diffusion.data import utils as du
import logging
logger = logging.getLogger(__name__)

# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self


def make_traj_from_rfo(rfi, rfo, traj_pdb):
    num_interp = 1
    xyz_prev_orig = rfi.xyz.to(rfo.xyz.device)
    all_pred = rfo.xyz
    seq = rfi.seq[0]
    all_pred = torch.cat([xyz_prev_orig[0:1,None,:,:3]]+[all_pred], dim=0)
    T = all_pred.shape[0]
    t = np.arange(T)
    n_frames = 1*(T-1)+1
    L = seq.shape[0]
    Y = np.zeros((n_frames,L,3,3))
    for i_res in range(L):
        for i_atom in range(3):
            for i_coord in range(3):
                interp = Akima1DInterpolator(t,all_pred[:,0,i_res,i_atom,i_coord].detach().cpu().numpy())
                Y[:,i_res,i_atom,i_coord] = interp(np.arange(n_frames)/num_interp)
    Y = torch.from_numpy(Y).float()

    # # 1st frame is final pred so pymol renders bonds correctly
    # util.writepdb(out_prefix+"_traj.pdb", Y[-1], seq[0,-1], 
    #     modelnum=0, bond_feats=bond_feats, file_mode="w")
    if os.path.exists(traj_pdb):
        os.remove(traj_pdb)
    for i in range(Y.shape[0]-1,0,-1):
        rf2aa.util.writepdb(traj_pdb, Y[i], seq, 
            modelnum=i+1, bond_feats=rfi.bond_feats, file_mode="a")

rf_conf = addict.Dict()
# rf_conf.NTOKENS = chemical.NAATOKENS
# rf_conf.conf.diffuser.T = 200
# rf_conf.conf.preprocess.sidechain_input = False
# rf_conf.converter = XYZConverter()
# rf_conf.conf.preprocess.d_t1d = 99999
# rf_conf.conf.inference.contig_as_guidepost = True

rf_conf.diffuser.T = 200
rf_conf.preprocess.sidechain_input = False
rf_conf.preprocess.d_t1d = 99999
rf_conf.inference.contig_as_guidepost = True


from dataclasses import fields

def stack_dataclass_fields(dataclass_list: List[dataclass]) -> dataclass:
    # Get the field names and types of the dataclass
    field_names = [field.name for field in fields(dataclass_list[0])]

    # Initialize lists to store stacked fields
    stacked_fields = [[] for _ in field_names]

    # Extract and stack each field from the dataclass objects
    for dc in dataclass_list:
        for i, field_name in enumerate(field_names):
            field_value = getattr(dc, field_name)
            stacked_fields[i].append(field_value)

    # Stack the fields
    ic([(field_name, torch.is_tensor(field_list[0])) for field_name, field_list in zip(field_names, stacked_fields)])
    stacked_fields = [torch.cat(field_list) if torch.is_tensor(field_list[0]) else field_list for field_list in stacked_fields]


    # Create a new dataclass with the stacked fields
    stacked_dataclass = dataclass_list[0]
    for i, field_name in enumerate(field_names):
        setattr(stacked_dataclass, field_name, stacked_fields[i])

    return stacked_dataclass


def rfi_from_input_feats_batched(input_feats):
    init_frames = input_feats['rigids_t'].type(torch.float32)    
    B, L, _ = init_frames.shape

    rfis = []
    for b in range(B):
        input_feats_b = {}
        for k in ['rigids_t', 'aatype', 'chain_idx', 'seq_idx', 't']:
            input_feats_b[k] = input_feats[k][b:b+1]
        rfis.append(rfi_from_input_feats(input_feats_b))

    # rfi = rf_diffusion.aa_model.RFI
    # d = tensor_util.to_ordered_dict(rfi)

    # rfi.msa_latent = torch.stack([e.msa_latent for e in rfis])
    # return rfi
    rfi = stack_dataclass_fields(rfis)
    rfi.msa_prev = None
    rfi.pair_prev = None
    rfi.state_prev = None
    return rfi

def rfi_from_input_feats(input_feats):
    
    init_frames = input_feats['rigids_t'].type(torch.float32)    
    B, L, _ = init_frames.shape
    assertpy.assert_that(B).is_equal_to(1)

    if 'aatype' in input_feats:
        # Training
        seq = input_feats['aatype'][0]
        chain_idx = input_feats['chain_idx'][0]
    else:
        # Inference
        seq = torch.tensor([0] * L)
        chain_idx = torch.zeros((L,))

    curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
    psi_t = torch.zeros((1,L,2))
    bb_representations = all_atom.compute_backbone(curr_rigids, psi_t)
    atom37 = bb_representations[0][0]
    xyz = torch.zeros((L, ChemData().NHEAVY, 3))
    xyz[:,:3,:] = atom37[:,:3,:]
    idx = input_feats['seq_idx'][0]
    Ls = [len(idx)]
    # for k in input_feats:
    #     print(k)
    same_chain = chain_idx[None] == chain_idx[...,None]

    is_diffused = torch.ones(len(idx)).bool()

    # Unused small molecule stuff
    chirals = torch.Tensor()
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = rf2aa.util.get_protein_bond_feats(Ls[0])
    terminus_type = torch.zeros(sum(Ls))
    
    indep = rf_diffusion.aa_model.Indep(
        seq,
        xyz,
        idx,
        bond_feats,
        chirals,
        same_chain,
        terminus_type
    )
    rf2aa.tensor_util.to_device(indep, 'cpu')
    T = 200
    # TODO: change this when atomization comes into play
    is_seq_masked = is_diffused
    indep.seq[is_seq_masked] = ChemData().MASKINDEX
    aa_model_converter = rf_diffusion.aa_model.Model(rf_conf)
    return aa_model_converter.prepro(indep, input_feats['t'].to('cpu') * T, is_diffused.to('cpu'))


def unpack_out_raw_yesquats(o:tuple):
    """Unpack 'raw' outputs the RF forward pass outputs
    
    Args:
        o: output from LegacyRoseTTAFoldModule
    """
    assert len(o) == 13 # it's 13 if model had get_quaternion=True
    _, _, _, _, _, _, alpha_s, xyz_allatom, _, msa_prev, pair_prev, state_prev, _ = o
    xyz_last = xyz_allatom[-1].unsqueeze(0)
    return msa_prev, pair_prev, xyz_last, state_prev, alpha_s[-1]


def multi_recycle_prediction(rfi, N_cycle, model, use_checkpoint, return_raw, training=False):
    """Perform a forward pass with multiple recycles
    
    Args:
        rfi (aa_model.RFI): model inputs 
        N_cycle (int): number of recylces
        model (rf2aa.RoseTTAFoldModule.LegacyRoseTTAFoldModule): RFdiffusion model
        use_checkpoint (bool): whether to use checkpointing in forward pass 
        return_raw (bool): whether to have final output be 'raw' return from LegacyRoseTTAFoldModule
        training: (bool): whether or not this call is during training.
                          If False, will torch.no_grad() for everything 
                          If True, will torch.no_grad() for N-1 recycles, then use grad.
    """
    with torch.no_grad():

        # Do the first N-1 recycles
        for i in range(N_cycle - 1):
            print('On recycle: ',i)
            if i == 0:
                rfi_dict = dataclasses.asdict(rfi)
                model_input = {**rfi_dict}
                model_input['msa_prev'] = None
                model_input['pair_prev'] = None
                model_input['state_prev'] = None
                model_input['return_raw'] = False
                out = model(**model_input)

            else:
                msa_prev, pair_prev, xyz_prev, state, alpha_prev = unpack_out_raw_yesquats(out)

                rfi_dict = dataclasses.asdict(rfi)
                rfi_dict['msa_prev']    = msa_prev
                rfi_dict['pair_prev']   = pair_prev
                rfi_dict['xyz']         = xyz_prev
                rfi_dict['state_prev']  = state

                model_input = {**rfi_dict}
                model_input['return_raw'] = False
                out = model(**model_input)
    
    if training:
        context = nullcontext()
    else:
        context = torch.no_grad()

    with context:
        # Do the last recycle, and return RFO
        msa_prev, pair_prev, xyz_prev, state, alpha_prev = unpack_out_raw_yesquats(out)

        rfi_dict = dataclasses.asdict(rfi)
        rfi_dict['msa_prev']    = msa_prev
        rfi_dict['pair_prev']   = pair_prev
        rfi_dict['xyz']         = xyz_prev
        rfi_dict['state_prev']  = state

        model_input = {**rfi_dict}
        model_input['use_checkpoint'] = use_checkpoint
        model_input['return_raw'] = return_raw

        return rf_diffusion.aa_model.RFO(*model(**model_input))

class RFScore(nn.Module):
    def __init__(self, model_conf, diffuser, device, stopgrad_rotations=True):
        self.diffuser = diffuser
        self.stopgrad_rotations=stopgrad_rotations
        super(RFScore, self).__init__()

        self.register_buffer('aamask',
            ChemData().allatom_mask.int().to(device), persistent=False
        )
        self.register_buffer('num_bonds',
            ChemData().num_bonds.to(device), persistent=False
        )
        self.register_buffer('atom_type_index',
            ChemData().atom_type_index.to(device), persistent=False,
        )
        self.register_buffer('ljlk_parameters',
            ChemData().ljlk_parameters.to(device), persistent=False,
        )
        self.register_buffer('lj_correction_parameters',
            ChemData().lj_correction_parameters.int().to(device), persistent=False,
        )
        self.register_buffer('cb_len',
            ChemData().cb_length_t.to(device), persistent=False
        )
        self.register_buffer('cb_ang',
            ChemData().cb_angle_t.to(device), persistent=False
        )
        self.register_buffer('cb_tor',
            ChemData().cb_torsion_t.to(device), persistent=False
        )


        # self.register_buffer('buffer_aa_mask', self.aamask)
        # self.register_buffer('buffer_num_bonds', self.num_bonds)
        # self.register_buffer('buffer_atom_type_index', self.atom_type_index)
        # self.register_buffer('buffer_ljlk_parameters', self.ljlk_parameters)
        # self.register_buffer('buffer_lj_correction_parameters', self.lj_correction_parameters)
        # self.register_buffer('buffer_cb_len', self.cb_len)
        # self.register_buffer('buffer_cb_ang', self.cb_ang)
        # self.register_buffer('buffer_cb_tor', self.cb_tor)

        model = rf2aa.model.RoseTTAFoldModel.LegacyRoseTTAFoldModule(
            **model_conf,
            aamask=self.aamask.float(),
            atom_type_index=self.atom_type_index.float(),
            ljlk_parameters=self.ljlk_parameters.float(),
            lj_correction_parameters=self.lj_correction_parameters.float(),
            num_bonds=self.num_bonds.float(),
            cb_len = self.cb_len.float(),
            cb_ang = self.cb_ang.float(),
            cb_tor = self.cb_tor.float(),
            assert_single_sequence_input=False,
            )
 
        self.model = model
        self.log_dir = os.path.join('training_pdbs', datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss"))

    def device(self):
        return self.named_parameters().__next__()[1].device

    def get_rfo(self, rfi):
        device = self.device()
        rf2aa.tensor_util.to_device(rfi, device)
        rfi_dict = dataclasses.asdict(rfi)
        rfo = rf_diffusion.aa_model.RFO(*self.model(**{**rfi_dict, 'use_checkpoint':True}))
        return rfo
    
    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return self.forward_from_input_feats(*args, **kwargs)
        return self.forward_from_rfi(*args, **kwargs)
    
    def forward_from_input_feats(self, input_feats):
        # indep, rfi_tp1_t, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, item_context = input_feats
        # _, rfi = rfi_tp1_t
        t = input_feats['t']
        rfi = rfi_from_input_feats(input_feats)
        model_out = self.forward_from_rfi(rfi, t)
        node_mask = input_feats['res_mask'].type(torch.float32)
        model_out['rot_score'] = model_out['rot_score'] * node_mask[:,None,:,None]
        model_out['trans_score'] = model_out['trans_score'] * node_mask[:,None,:,None]
        return model_out

    def forward_from_rfi(self, rfi, t, use_checkpoint=True, return_raw=False, N_cycle=1, training=False):
        device = self.device()
        rf2aa.tensor_util.to_device(rfi, device)
        rigids_t = du.rigid_frames_from_atom_14(rfi.xyz)
        rfi_dict = dataclasses.asdict(rfi)

        if N_cycle == 1:
            rfo = rf_diffusion.aa_model.RFO(*self.model(**{**rfi_dict, 'use_checkpoint':use_checkpoint, 'return_raw':return_raw}))
        else:
            # refinement uses recycling 
            rfo = multi_recycle_prediction(
                                           rfi=rfi, 
                                           N_cycle=N_cycle, 
                                           model=self.model, 
                                           use_checkpoint=use_checkpoint, 
                                           return_raw=return_raw,
                                           training=training
                                        )
        # # Traj writing
        # rfo_cpy = tensor_util.apply_to_tensors(rfo, lambda x:x.detach().cpu())
        # os.makedirs(self.log_dir, exist_ok=True)
        # traj_path = os.path.join(self.log_dir, f't_{input_feats["t"][0].item():.5f}_.pdb')
        # make_traj_from_rfo(rfi, rfo_cpy, traj_path)
        # ic(traj_path)
        B, I, L, _  = rfo.quat.shape

        # rigids from predictions of X0
        curr_rigids = rigids_from_rfo(rfo, rigids_t.get_rots(), stopgrad_rotations=self.stopgrad_rotations)

        trans_score = None
        rot_score = None
        if self.diffuser is not None:
            trans_score, rot_score = calc_score(curr_rigids, rigids_t, self.diffuser, t)

        psi_pred = torch.rand((B,I,L,2)).to(curr_rigids.device)
        bb_representations = all_atom.compute_backbone(curr_rigids, psi_pred)
        
        model_out = {
            'psi': psi_pred,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'rigids': curr_rigids.to_tensor_7(),
            'rigids_raw': curr_rigids,
            'atom37': bb_representations[0].to(curr_rigids.device),
            'atom14': bb_representations[-1].to(curr_rigids.device),
            'rfo': rfo,
        }
        return model_out

def rigids_from_rfo(rfo, rots_t, stopgrad_rotations):
    xyz_stack = rfo.xyz.transpose(0,1)
    B, I, L, _, _ = xyz_stack.shape
    c_alpha = xyz_stack[:,:,:,1,:]
    
    # init_rigids = Rigid.from_tensor_7(init_frames)
    curr_rots = Rotation(quats=rots_t.get_quats())
    rot_blocks = torch.zeros((B, I, L, 4), dtype=rfo.quat.dtype, device=rfo.quat.device)
    for i in range(I):
        # curr_rots = curr_rots.compose_r(Rotation(quats=rfo.quat[:, i]))
        curr_rots = Rotation(quats=rfo.quat[:, i]).compose_q(curr_rots)
        rot_blocks[:, i] = curr_rots.get_quats()
        if stopgrad_rotations:
            curr_rots = curr_rots.detach()
    rots_compose = Rotation(quats=rot_blocks)

    curr_rigids = Rigid(
        trans=c_alpha,
        rots=rots_compose,
    )
    return curr_rigids


def calc_score(curr_rigids, init_rigids, diffuser, t):
    """
    Parameters: 
        curr_rigids (ru.Rigid): Actual or predicted time=0 rigids 
        init_rigids (ru.Rigid): Rigids at the current time=t
        diffuser    (frame_diffusion.data diffuser class): Diffuser 
        t           (float): Current time
    """
    device = curr_rigids.device
    B, I, L = curr_rigids.shape
    rot_score = torch.zeros((B,I,L,3)).to(device)
    trans_score = torch.zeros((B,I,L,3)).to(device)
    
    for i in range(I):
        rigid_block = curr_rigids[:,i]
        rot_score[:,i] = diffuser.calc_rot_score(
            init_rigids.get_rots(),  # rots_t
            rigid_block.get_rots(),  # rots_0
            t
        )

        trans_score[:,i] = diffuser.calc_trans_score(
            init_rigids.get_trans(),  # trans_t
            rigid_block.get_trans(),  # trans_0
            t,
            use_torch=True,
        )
    return trans_score, rot_score
