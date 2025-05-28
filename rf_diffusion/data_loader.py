import contextlib
import logging
import torch
import itertools
import traceback
import pprint
import torch.nn.functional as F
import copy
import time
from dataclasses import dataclass
from torch.utils import data
from collections import OrderedDict
import os
import csv
from dateutil import parser
import numpy as np
from rf_diffusion.parsers import parse_a3m, parse_pdb
from rf_diffusion.kinematics import xyz_to_t2d
from rf_diffusion.preprocess import wrap_featurize 
import rf2aa.data.compose_dataset
import rf2aa.util
import rf2aa.tensor_util
import rf2aa.kinematics
from rf_diffusion.chemical import ChemicalData as ChemData

# for diffusion training
from icecream import ic
import pickle
import random
import math
from functools import partial
import pandas as pd
import torch.distributed as dist
import hydra

from rf_diffusion import run_inference
from rf_diffusion import aa_model
from rf_diffusion import error
from rf_diffusion import features
from rf_diffusion import distributions
from rf_diffusion import conditioning
from rf_diffusion.train_data.exceptions import NextExampleException
from rf_diffusion.train_data import fast_filters

from typing import Tuple
from omegaconf import OmegaConf
import pdb 

logger = logging.getLogger(__name__)

USE_DEFAULT = '__USE_DEFAULT__'

base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"
compl_dir = "/projects/ml/RoseTTAComplex"
fb_dir = "/projects/ml/TrRosetta/fb_af"
cn_dir = "/home/jwatson3/torch/cn_ideal"
na_dir = "/home/dimaio/TrRosetta/nucleic"
sm_compl_dir = "/projects/ml/RF2_allatom"

def set_data_loader_params(args):
    ic(args)
    PARAMS = {
        "COMPL_LIST": f"{compl_dir}/list.hetero.csv",
        "HOMO_LIST": f"{compl_dir}/list.homo.csv",
        "NEGATIVE_LIST": f"{compl_dir}/list.negative.csv",
        "PDB_LIST": f"{base_dir}/list_v02.csv",
        "FB_LIST": f"{fb_dir}/list_b1-3.csv",
        "VAL_PDB": f"{base_dir}/val/xaa",
        "VAL_COMPL": f"{compl_dir}/val_lists/xaa",
        "VAL_NEG": f"{compl_dir}/val_lists/xaa.neg",
        "DATAPKL": args.data_pkl,
        "DATAPKL_AA": args.data_pkl_aa,
        "PDB_DIR": base_dir,
        "FB_DIR": fb_dir,
        "CN_DIR": cn_dir,
        "CN_DICT": os.path.join(cn_dir, 'cn_ideal_train_test.pt'),
        "COMPL_DIR": compl_dir,
        "MINTPLT": 0,
        "MAXTPLT": 5,
        "MINSEQ": 1,
        "MAXSEQ": 1024,
        "MAXLAT": 128,
        "CROP": 256,
        "DATCUT": "2020-Apr-30",
        "RESCUT": 5.0,
        "BLOCKCUT": 5,
        "PLDDTCUT": 70.0,
        "SCCUT": 90.0,
        "ROWS": 1,
        "SEQID": 95.0,
        "MAXCYCLE": 4,
        "HAL_MASK_HIGH": 35,
        "HAL_MASK_LOW": 10,
        "HAL_MASK_HIGH_AR": 50,
        "HAL_MASK_LOW_AR": 20,
        "COMPLEX_HAL_MASK_HIGH": 35,
        "COMPLEX_HAL_MASK_LOW": 10,
        "COMPLEX_HAL_MASK_HIGH_AR": 50,
        "COMPLEX_HAL_MASK_LOW_AR": 20,
        "FLANK_HIGH": 6,
        "FLANK_LOW": 3,
        "STR2SEQ_FULL_LOW": 0.9,
        "STR2SEQ_FULL_HIGH": 1.0,
        "MAX_LENGTH": 260,
        "MAX_COMPLEX_CHAIN": 200,
        "TASK_NAMES": ['seq2str'],
        "TASK_P": [1.0],
        "DIFF_MASK_PROBS": args.diff_mask_probs,
        "DIFF_MASK_LOW": args.diff_mask_low,
        "DIFF_MASK_HIGH": args.diff_mask_high,
        "DATASETS": args.dataset,
        "DATASET_PROB": args.dataset_prob,
        "MASK_MIN_PROPORTION": args.mask_min_proportion,
        "MASK_MAX_PROPORTION": args.mask_max_proportion,
        "MASK_BROKEN_PROPORTION": args.mask_broken_proportion,
        "SPOOF_ITEM": args.spoof_item,
        "MOL_DIR": rf2aa.data.compose_dataset.default_dataloader_params[
            'MOL_DIR'
        ],
        "DISCONTIGUOUS_CROP": True,
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            v = getattr(args, param.lower())
            if v == USE_DEFAULT:
                continue
            PARAMS[param] = getattr(args, param.lower())

    print('This is params from get train valid')
    for key,val in PARAMS.items():
        print(key, val)
    return PARAMS

def MSABlockDeletion(msa, ins, nb=5):
    '''
    Input: MSA having shape (N, L)
    output: new MSA with block deletion
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    csum = torch.zeros(N_seq, N_res, data.shape[-1]).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params, eps=1e-6, nmer=1, L_s=None):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences

    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
        - N-term or C-term? (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
        - N-term or C-term? (2)
    '''
    if L_s is None: L_s = []
    N, L = msa.shape

    term_info = torch.zeros((L,2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain

    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0)

    # Select Nclust sequence randomly (seed MSA or latent MSA)
    Nclust = (min(N, params['MAXLAT'])-1) // nmer
    Nclust = Nclust*nmer + 1

    if N > Nclust*2:
        Nextra = N - Nclust
    else:
        Nextra = N
    Nextra = min(Nextra, params['MAXSEQ']) // nmer
    Nextra = max(1, Nextra * nmer)
    #
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample_mono = torch.randperm((N-1)//nmer, device=msa.device)
        sample = [sample_mono + imer*((N-1)//nmer) for imer in range(nmer)]
        sample = torch.stack(sample, dim=-1)
        sample = sample.reshape(-1)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)

        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < 0.15
        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        b_seq.append(msa_masked[0].clone())

        ## get extra sequenes
        if N > Nclust*2:  # there are enough extra sequences
            msa_extra = msa[1:,:][sample[Nclust-1:]]
            ins_extra = ins[1:,:][sample[Nclust-1:]]
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
        elif N - Nclust < 1:
            msa_extra = msa_masked.clone()
            ins_extra = ins_clust.clone()
            extra_mask = mask_pos.clone()
        else:
            msa_add = msa[1:,:][sample[Nclust-1:]]
            ins_add = ins[1:,:][sample[Nclust-1:]]
            mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
            msa_extra = torch.cat((msa_masked, msa_add), dim=0)
            ins_extra = torch.cat((ins_clust, ins_add), dim=0)
            extra_mask = torch.cat((mask_pos, mask_add), dim=0)
        N_extra = msa_extra.shape[0]

        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=22)
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20) # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20)
        agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T)
        assignment = torch.argmax(agreement, dim=-1)

        # seed MSA features
        # 1. one_hot encoded aatype: msa_clust_onehot
        # 2. cluster profile
        count_extra = ~extra_mask
        count_clust = ~mask_pos
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_profile
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L)
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]
        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        #
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust, term_info[None].expand(Nclust,-1,-1)), dim=-1)

        # extra MSA features
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)

        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)

    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def MSAFeaturize_fixbb(msa, params, L_s=[]):
    '''
    Input: full msa information
    Output: Single sequence, with some percentage of amino acids mutated (but no residues 'masked')
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask). In inpainting task, masked sequence is set to 22nd plane (i.e. mask token)
        - profile of clustered sequences (22) (just single sequence copied again)
        - insertion statistics (2) (set to 0 here)
        - N-term or C-term? (2). This is used as normal for inpainting, if training on complexes.
    extra sequence features:
        - aatype of extra sequence (22) (just single sequence again)
        - insertion info (1). Set to zero
        - N-term or C-term? (2)
    '''
    N, L = msa.shape
    term_info = torch.zeros((L,2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0)
    b_seq = []
    b_msa_clust = []
    b_msa_seed = []
    b_msa_extra = []
    b_mask_pos = []
    for _ in range(params['MAXCYCLE']):
        assert torch.max(msa) < 22
        msa_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=22)
        msa_fakeprofile_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=24) #add the extra two indel planes, which will be set to zero
        msa_seed = torch.cat((msa_onehot, msa_fakeprofile_onehot, term_info[None]), dim=-1)
        #make fake msa_extra
        msa_extra_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=23) #add one extra plane for blank indel
        msa_extra = torch.cat((msa_extra_onehot, term_info[None]), dim=-1)
        #make fake msa_clust and mask_pos
        msa_clust = msa[:1]
        mask_pos = torch.full_like(msa_clust, 1).bool()

        b_seq.append(msa[0].clone())
        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)

    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, pick_top=True):
    seqID_cut = params['SEQID']

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ
        xyz = torch.full((1, qlen, 27, 3), np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        conf = torch.zeros((1, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        return xyz, t1d
    
    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[tplt_valid_idx]
    else:
        tplt_valid_idx = torch.arange(len(tplt['ids']))
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates
        xyz = torch.full((1,qlen,27,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        conf = torch.zeros((1, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        return xyz, t1d

    if not pick_top: # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else: # only consider top 50 templates
        sample = torch.randperm(min(50,ntplt))[:npick]

    xyz = torch.full((npick,qlen,27,3),np.nan).float()
    mask = torch.full((npick,qlen,27),False)
    t1d = torch.full((npick, qlen), 20).long()
    t1d_val = torch.zeros((npick, qlen)).float()

    for i,nt in enumerate(sample):
        tplt_idx = tplt_valid_idx[nt]
        sel = torch.where(tplt['qmap'][0,:,1]==tplt_idx)[0]
        pos = tplt['qmap'][0,sel,0] + offset
        xyz[i,pos,:14] = tplt['xyz'][0,sel]
        mask[i,pos,:14] = tplt['mask'][0,sel]
        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2] # alignment confidence

    t1d = torch.nn.functional.one_hot(t1d, num_classes=21).float()
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    xyz = torch.where(mask[...,None], xyz.float(),torch.full((npick,qlen,27,3),np.nan).float()) # (T, L, 27, 3)
    
    center_CA = ((mask[:,:,1,None]) * torch.nan_to_num(xyz[:,:,1,:])).sum(dim=1) / ((mask[:,:,1,None]).sum(dim=1)+1e-4) # (T, 3)
    xyz = xyz - center_CA.view(npick,1,1,3)

    return xyz, t1d

def TemplFeaturizeFixbb(seq, conf_1d=None):
    """
    Template 1D featurizer for fixed BB examples

    Parameters:

        seq (torch.tensor, required): Integer sequence

        conf_1d (torch.tensor, optional): Precalcualted confidence tensor
    """
    seq=seq[:1]
    t1d  = torch.nn.functional.one_hot(seq, num_classes=21) # one hot sequence

    if conf_1d is None:
        conf = torch.ones_like(seq)[...,None]
    else:
        conf = conf_1d[:,None]


    t1d = torch.cat((t1d, conf), dim=-1)

    return t1d

def get_train_valid_set(params, OFFSET=1000000):

    if (not os.path.exists(params['DATAPKL'])):
        # read validation IDs for PDB set
        val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
        val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])


        # read validation IDs for PDB set
        val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
        val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])

        # read homo-oligomer list
        homo = {}
        # with open(params['HOMO_LIST'], 'r') as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     # read pdbA, pdbB, bioA, opA, bioB, opB
        #     rows = [[r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in reader]
        # for r in rows:
        #     if r[0] in homo.keys():
        #         homo[r[0]].append(r[1:])
        #     else:
        #         homo[r[0]] = [r[1:]]

        # read & clean list.csv

        with open(params['PDB_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[3],int(r[4]), int(r[-1].strip())] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT']) and len(r[-2]) <= params['MAX_LENGTH'] and len(r[-2]) >= 60] #added length max so only have full chains, and minimum length of 60aa

        # compile training and validation sets
        val_hash = []
        train_pdb = {}
        valid_pdb = {}
        valid_homo = {}

        for r in rows:
            if r[2] in val_pdb_ids:
                val_hash.append(r[1])
                if r[2] in valid_pdb:
                    valid_pdb[r[2]].append((r[:2], r[-1]))
                else:
                    valid_pdb[r[2]] = [(r[:2], r[-1])]
                #
                if r[0] in homo:
                    if r[2] in valid_homo:
                        valid_homo[r[2]].append((r[:2], r[-1]))
                    else:
                        valid_homo[r[2]] = [(r[:2], r[-1])]
            else:
                if r[2] in train_pdb:
                    train_pdb[r[2]].append((r[:2], r[-1]))
                else:
                    train_pdb[r[2]] = [(r[:2], r[-1])]

        val_hash = set(val_hash)

        # compile facebook model sets
        with open(params['FB_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                     if float(r[1]) > 80.0 and
                     len(r[-1].strip()) > 100 and len(r[-1].strip()) <= params['MAX_LENGTH']] #added max length to allow only full chains. Also reduced minimum length to 100aa

        fb = {}

        for r in rows:
            if r[2] in fb:
                fb[r[2]].append((r[:2], r[-1]))
            else:
                fb[r[2]] = [(r[:2], r[-1])]

        #compile complex sets

        with open(params['COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxID, assembly (bioA,opA,bioB,opB)
            rows = [[r[0], r[3], int(r[4]), [int(plen) for plen in r[5].split(':')], r[6] , [int(r[7]), int(r[8]), int(r[9]), int(r[10])]] for r in reader
                     if float(r[2]) <= params['RESCUT'] and
                     parser.parse(r[1]) <= parser.parse(params['DATCUT']) and min([int(i) for i in r[5].split(":")]) < params['MAX_COMPLEX_CHAIN'] and min([int(i) for i in r[5].split(":")]) > 50] #require one chain of the hetero complexes to be smaller than a certain value so it can be kept complete. This chain must also be > 50aa long.

        train_compl = {}
        valid_compl = {}

        for r in rows:
            if r[2] in val_compl_ids:
                if r[2] in valid_compl.keys():
                    valid_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1])) # ((pdb, hash), length, taxID, assembly, negative?)
                else:
                    valid_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]
            else:
                # if subunits are included in PDB validation set, exclude them from training
                hashA, hashB = r[1].split('_')
                if hashA in val_hash:
                    continue
                if hashB in val_hash:
                    continue
                if r[2] in train_compl.keys():
                    train_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1]))
                else:
                    train_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]

        # compile negative examples
        # remove pairs if any of the subunits are included in validation set
        # with open(params['NEGATIVE_LIST'], 'r') as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy
        #     rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')],r[6]] for r in reader
        #             if float(r[2])<=params['RESCUT'] and
        #             parser.parse(r[1])<=parser.parse(params['DATCUT'])]


        train_neg = {}
        valid_neg = {}

        # for r in rows:
        #     if r[2] in val_neg_ids:
        #         if r[2] in valid_neg.keys():
        #             valid_neg[r[2]].append((r[:2], r[-2], r[-1], []))
        #         else:
        #             valid_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]
        #     else:
        #         hashA, hashB = r[1].split('_')
        #         if hashA in val_hash:
        #             continue
        #         if hashB in val_hash:
        #             continue
        #         if r[2] in train_neg.keys():
        #             train_neg[r[2]].append((r[:2], r[-2], r[-1], []))
        #         else:
        #             train_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]


        train_cn = {}
        valid_cn = {}
        cn_dict = torch.load(params['CN_DICT'], weights_only=False)
        for r in cn_dict['train_seqs']:
            if r[2] < params['MAX_LENGTH']:
                if r[0] in train_cn.keys():
                    train_cn[r[0]].append((r[1], r[2]))
                else:
                    train_cn[r[0]] = [(r[1], r[2])]

        for r in cn_dict['test_seqs']:
            if r[2] < params['MAX_LENGTH']:
                if r[0] in valid_cn.keys():
                    valid_cn[r[0]].append((r[1], r[2]))
                else:
                    valid_cn[r[0]] = [(r[1], r[2])]

        # Get average chain length in each cluster and calculate weights
        pdb_IDs = list(train_pdb.keys())
        fb_IDs = list(fb.keys())
        compl_IDs = list(train_compl.keys())
        neg_IDs = list(train_neg.keys())
        cn_IDs = list(train_cn.keys())

        pdb_weights = []
        fb_weights = []
        compl_weights = []
        neg_weights = []
        cn_weights = []

        for key in pdb_IDs:
            plen = sum([plen for _, plen in train_pdb[key]]) // len(train_pdb[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            pdb_weights.append(w)

        for key in fb_IDs:
            plen = sum([plen for _, plen in fb[key]]) // len(fb[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            fb_weights.append(w)

        for key in compl_IDs:
            plen = sum([sum(plen) for _, plen, _, _ in train_compl[key]]) // len(train_compl[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            compl_weights.append(w)

        for key in neg_IDs:
            plen = sum([sum(plen) for _, plen, _, _ in train_neg[key]]) // len(train_neg[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            neg_weights.append(w)

        for key in cn_IDs:
            plen = sum([plen for _, plen in train_cn[key]]) // len(train_cn[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            cn_weights.append(w)

        # save

        obj = (
           pdb_IDs, pdb_weights, train_pdb,
           fb_IDs, fb_weights, fb,
           compl_IDs, compl_weights, train_compl,
           neg_IDs, neg_weights, train_neg, cn_IDs, cn_weights, train_cn,
           valid_pdb, valid_homo, valid_compl, valid_neg, valid_cn, homo
        )
        with open(params["DATAPKL"], "wb") as f:
            print ('Writing',params["DATAPKL"])
            pickle.dump(obj, f)
            print ('Done')

    else:
        start = time.time()
        with open(params["DATAPKL"], "rb") as f:
            print ('Loading',params["DATAPKL"])
            (
               pdb_IDs, pdb_weights, train_pdb,
               fb_IDs, fb_weights, fb,
               compl_IDs, compl_weights, train_compl,
               neg_IDs, neg_weights, train_neg, cn_IDs, cn_weights, train_cn,
               valid_pdb, valid_homo, valid_compl, valid_neg, valid_cn, homo
            ) = pickle.load(f)
            elapsed = time.time() - start
            print (f'Loaded {params["DATAPKL"]} in {elapsed:.1f}s')

    return (pdb_IDs, torch.tensor(pdb_weights).float(), train_pdb), \
           (fb_IDs, torch.tensor(fb_weights).float(), fb), \
           (compl_IDs, torch.tensor(compl_weights).float(), train_compl), \
           (neg_IDs, torch.tensor(neg_weights).float(), train_neg),\
           (cn_IDs, torch.tensor(cn_weights).float(), train_cn),\
           valid_pdb, valid_homo, valid_compl, valid_neg, valid_cn, homo

# slice long chains
def get_crop(l, mask, device, params, unclamp=False):

    sel = torch.arange(l,device=device)
    if l <= params['CROP']:
        return sel
    raise Warning("Example is being cropped. Is this intended?") 
    size = params['CROP']

    mask = ~(mask[:,:3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[0]
    res_idx = exists[torch.randperm(len(exists))[0]].item()

    lower_bound = max(0, res_idx-size+1)
    upper_bound = min(l-size, res_idx+1)
    start = np.random.randint(lower_bound, upper_bound)
    return sel[start:start+size]

def get_complex_crop(len_s, mask, device, params):
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)

    n_added = 0
    n_remaining = sum(len_s)
    preset = 0
    sel_s = []
    for k in range(len(len_s)):
        n_remaining -= len_s[k]
        crop_max = min(params['CROP']-n_added, len_s[k])
        crop_min = min(len_s[k], max(1, params['CROP'] - n_added - n_remaining))

        if k == 0:
            crop_max = min(crop_max, params['CROP']-5)
        crop_size = np.random.randint(crop_min, crop_max+1)
        n_added += crop_size

        mask_chain = ~(mask[preset:preset+len_s[k],:3].sum(dim=-1) < 3.0)
        exists = mask_chain.nonzero()[0]
        res_idx = exists[torch.randperm(len(exists))[0]].item()
        lower_bound = max(0, res_idx - crop_size + 1)
        upper_bound = min(len_s[k]-crop_size, res_idx) + 1
        start = np.random.randint(lower_bound, upper_bound) + preset
        sel_s.append(sel[start:start+crop_size])
        preset += len_s[k]
    return torch.cat(sel_s)

def get_spatial_crop(xyz, mask, sel, len_s, params, label, cutoff=10.0, eps=1e-6):

    device = xyz.device
    
    # get interface residue
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
    cond = torch.logical_and(cond, mask[:len_s[0],None,1]*mask[None,len_s[0]:,1]) 
    i,j = torch.where(cond)
    ifaces = torch.cat([i,j+len_s[0]])
    if len(ifaces) < 1:
        print ("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + torch.arange(len(xyz), device=xyz.device)*eps
    cond = mask[:,1]*mask[cnt_idx,1]
    dist[~cond] = 999999.9
    _, idx = torch.topk(dist, params['CROP'], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel

def get_spatial_crop_fixbb(xyz, mask, sel, len_s, params, cutoff=10.0, eps=1e-6):

    chainA_idx_max = sel[len_s[0]-1]

    #choose which chain to keep whole
    if sum(len_s) < params['MAX_LENGTH']: # don't need to crop
        if all(i < params['MAX_COMPLEX_CHAIN'] for i in len_s): #both chains are small enough
            chain=torch.randint(0,2,(1,))[0] #choose either first or second chain
        else:
            chain=np.argmin(len_s)

        return sel, [chain, len_s[0]] #give length of first chain, as either end or start point

    else:
        chain=np.argmin(len_s) #choose smaller chain
    # get interface residue
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff #find interface residue first, to make crop later
    cond = torch.logical_and(cond, mask[:len_s[0],None,1]*mask[None,len_s[0]:,1])
    i,j = torch.where(cond)
    if chain == 0:
        ifaces = i
    else:
        ifaces = j+len_s[0]
    assert len(ifaces) > 0

    cnt_idx = ifaces[np.random.randint(len(ifaces))] #this gets an interface residue, on the full chain

    dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + torch.arange(len(xyz), device=xyz.device)*eps #this gives distance of all residues to the randomly chosen interface residue
    cond = mask[:,1]*mask[cnt_idx,1]
    dist[~cond] = 999999.9
    if chain==0:
        dist[:len_s[0]] = 0 #therefore include whole chain in the crop
    else:
        dist[len_s[0]:] = 0
    _, idx = torch.topk(dist, params['CROP'], largest=False)
    sel, _ = torch.sort(sel[idx])
    n_chainA = torch.sum(torch.where(sel <= chainA_idx_max, True, False)).item()
    return sel, [chain,n_chainA] #give length of first chain, as either end or start point

def get_contacts(complete_chain, xyz_t, cutoff_distance=10):
    """
    Function to take complete_chain (in the form [chain_id, len_first_chain]) and output a tensor (length L) indicating whether a residue should be diffused (False) or not (True).
    Also outputs 1D tensor (length L) indicating whether residues in the target (the non-diffused chain) are in contact with the binder.
    Some fraction of contacting residues are masked (set to 0). This is to encourage the network to be able to make extra contacts at inference time (i.e. not all contacting residues are given).
    """
    L = xyz_t.shape[1]
    chain_tensor = torch.ones(L).bool()
    if complete_chain[0] == 0:
        chain_tensor[:complete_chain[1]] = False
    else:
        chain_tensor[complete_chain[1]:] = False 

    cutoff_bin = np.floor((cutoff_distance-2)*2) #converts distance in angstroms to respective bin in t2d
    pair_distances = xyz_to_t2d(xyz_t[None,:,:,:3])[:,:,:,:,:37]

    #find pairwise distances that are within the cutoff
    contact_map = torch.where(torch.argmax(pair_distances, dim=-1) < cutoff_bin, True, False).squeeze().squeeze()
    
    #mask out intra-chain contacts
    contact_map[~chain_tensor] = False
    contact_map[:,chain_tensor] = False
    contact_tensor = torch.where(torch.sum(contact_map, dim=1) > 0, True, False)
    
    #only display contacting residues in the fixed chain (i.e. the target)
    contact_tensor = contact_tensor * chain_tensor
    to_mask = random.uniform(0,1) #randomly mask some proportion of contacting residues 
    mask_tensor = torch.where(torch.rand(L) < to_mask, False, True)
    contact_tensor *= mask_tensor
    return chain_tensor, contact_tensor.long()


# merge msa & insertion statistics of two proteins having different taxID
def merge_a3m_hetero(a3mA, a3mB, L_s):
    # merge msa
    query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]).unsqueeze(0) # (1, L)
    msa = [query]
    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['msa'][1:], (0,L_s[1]), "constant", 20) # pad gaps
        msa.append(extra_A)
    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['msa'][1:], (L_s[0],0), "constant", 20)
        msa.append(extra_B)
    msa = torch.cat(msa, dim=0)
    
    # merge ins
    query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]).unsqueeze(0) # (1, L)
    ins = [query]
    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['ins'][1:], (0,L_s[1]), "constant", 0) # pad gaps
        ins.append(extra_A)
    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['ins'][1:], (L_s[0],0), "constant", 0)
        ins.append(extra_B)
    ins = torch.cat(ins, dim=0)
    return {'msa': msa, 'ins': ins}

# merge msa & insertion statistics of units in homo-oligomers
def merge_a3m_homo(msa_orig, ins_orig, nmer):
    N, L = msa_orig.shape[:2]
    msa = torch.full((1+(N-1)*nmer, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
    ins = torch.full((1+(N-1)*nmer, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)
    start=0
    start2 = 1
    for _ in range(nmer):
        msa[0, start:start+L] = msa_orig[0]
        msa[start2:start2+(N-1), start:start+L] = msa_orig[1:]
        ins[0, start:start+L] = ins_orig[0]
        ins[start2:start2+(N-1), start:start+L] = ins_orig[1:]
        start += L
        start2 += (N-1)
    return msa, ins

# Generate input features for single-chain
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True):
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t,f1d_t = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top)
    
    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),27,3),np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), 27), False)
    mask[:,:14] = pdb['mask']

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed  = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa  = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz   = xyz[crop_idx]
    mask  = mask[crop_idx]
    idx   = idx[crop_idx]

    # get initial coordinates
    xyz_prev  = xyz_t[0]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()

    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = ChemData().INIT_CRDS.reshape(1, 27, 3).repeat(len(xyz), 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)
    
    #print ("loader_single", xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, unclamp, False, [0, None], mask #0 is the complete chain

def featurize_single_chain_fixbb(msa, pdb, params, unclamp=False, pick_top=False, fb=False):
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize_fixbb(msa, params)
    f1d_t = TemplFeaturizeFixbb(seq)
    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz']))
    xyz = torch.full((len(idx),27,3),np.nan).float()
    mask = torch.full((len(idx), 27), False)
    xyz[:,:14,:] = pdb['xyz']
    mask[:,:14] = pdb['mask']
    xyz_t = torch.clone(xyz)[None]

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # get initial coordinates
    xyz_prev = xyz_t[0]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()

    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = ChemData().INIT_CRDS.reshape(1, 27, 3).repeat(len(xyz), 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, unclamp, False, [0,None], mask

# Generate input features for homo-oligomers
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True):
    L = msa_orig.shape[1]

    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, nmer=2, L_s=[L,L])

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']//2+1)
    xyz_t_single, f1d_t_single = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top)
    ntempl = max(1, ntempl)
    # duplicate
    xyz_t = torch.full((2*ntempl, L*2, 27, 3), np.nan).float()
    f1d_t = torch.full((2*ntempl, L*2), 20).long()
    f1d_t = torch.cat((torch.nn.functional.one_hot(f1d_t, num_classes=21).float(), torch.zeros((2*ntempl, L*2, 1)).float()), dim=-1)
    xyz_t[:ntempl,:L] = xyz_t_single
    xyz_t[ntempl:,L:] = xyz_t_single
    f1d_t[:ntempl,:L] = f1d_t_single
    f1d_t[ntempl:,L:] = f1d_t_single

    # get initial coordinates
    xyz_prev = torch.cat((xyz_t_single[0], xyz_t_single[0]), dim=0)

    # get ground-truth structures
    # load metadata
    PREFIX = f"{params['PDB_DIR']}/torch/pdb/{pdbid[1:3]}/{pdbid}"
    meta = torch.load(PREFIX+".pt", weights_only=False)

    # get all possible pairs
    npairs = len(interfaces)
    xyz = torch.full((npairs, 2*L, 27, 3), np.nan).float()
    mask = torch.full((npairs, 2*L, 27), False)
    for i_int,interface in enumerate(interfaces):
        pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+interface[0][1:3]+'/'+interface[0]+'.pt', weights_only=False)
        xformA = meta['asmb_xform%d'%interface[1]][interface[2]]
        xformB = meta['asmb_xform%d'%interface[3]][interface[4]]
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz[i_int,:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask[i_int,:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)

    idx = torch.arange(L*2)
    idx[L:] += 200 # to let network know about chain breaks

    # indicator for which residues are in same chain
    chain_idx = torch.zeros((2*L, 2*L)).long()
    chain_idx[:L, :L] = 1
    chain_idx[L:, L:] = 1

    # Residue cropping
    if 2*L > params['CROP']:
        spatial_crop_tgt = np.random.randint(0, npairs)
        crop_idx = get_spatial_crop(xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params, interfaces[spatial_crop_tgt][0])
        seq = seq[:,crop_idx]
        msa_seed_orig = msa_seed_orig[:,:,crop_idx]
        msa_seed = msa_seed[:,:,crop_idx]
        msa_extra = msa_extra[:,:,crop_idx]
        mask_msa = mask_msa[:,:,crop_idx]
        xyz_t = xyz_t[:,crop_idx]
        f1d_t = f1d_t[:,crop_idx]
        xyz = xyz[:,crop_idx]
        mask = mask[:,crop_idx]
        idx = idx[crop_idx]
        chain_idx = chain_idx[crop_idx][:,crop_idx]
        xyz_prev = xyz_prev[crop_idx]

    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = ChemData().INIT_CRDS.reshape(1, 1, 27, 3).repeat(npairs, xyz.shape[1], 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, False, [0, None], mask

def get_pdb(pdbfilename, plddtfilename, item, lddtcut, sccut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    
    # update mask info with plddt (ignore sidechains if plddt < 90.0)
    mask_lddt = np.full_like(mask, False)
    mask_lddt[plddt > sccut] = True
    mask_lddt[:,:5] = True
    mask = np.logical_and(mask, mask_lddt)
    mask = np.logical_and(mask, (plddt > lddtcut)[:,None])
    
    return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx': torch.tensor(res_idx), 'label':item}

def get_msa(a3mfilename, item):
    msa,ins = parse_a3m(a3mfilename)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'label':item}

# Load PDB examples
def loader_pdb(item, params, homo, unclamp=False, pick_top=True, p_homo_cut=0.5):
    # load MSA, PDB, template info
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt', weights_only=False)
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+item[1][:3]+'/'+item[1]+'.pt', weights_only=False)
   
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)

    if item[0] in homo: # Target is homo-oligomer
        p_homo = np.random.rand()
        if p_homo < p_homo_cut: # model as homo-oligomer with p_homo_cut prob
            pdbid = item[0].split('_')[0]
            interfaces = homo[item[0]]
            return featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top)
        else:
            return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
    else:
        return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)

# Load PDB examples for fixbb tasks
def loader_pdb_fixbb(item, params, homo = None, unclamp=False, pick_top=False,p_homo_cut=None, aa=False):
    with open('loader_inputs.pkl', 'wb') as f:
        pickle.dump([item, params, homo, unclamp, pick_top, p_homo_cut], f)
    """
    Loader for fixbb tasks, from pdb dataset
    """
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt', weights_only=False)
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])

    # get msa features
    msa = a3m['msa'].long()[:1]
    
    out = featurize_single_chain_fixbb(msa, pdb, params, unclamp=unclamp, pick_top=pick_top, fb=False)
    if not aa:
        return out
    (seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d,         xyz_prev,            same_chain, unclamp, negative, complete_chain, atom_mask) = out

    C, L = seq.shape

    n_pad = ChemData().NTOTAL - 27
    #xyz_prev = F.pad(input=xyz_prev, pad=(0, n_pad, 0), mode='constant', value=0)
    xyz_prev = F.pad(input=xyz_prev, pad=(0, 0, 0, n_pad), mode='constant', value=0)
    true_crds = F.pad(input=true_crds, pad=(0, 0, 0, n_pad), mode='constant', value=0)
    mask_prev = F.pad(input=atom_mask, pad=(0, n_pad), mode='constant', value=0)
    xyz_t = xyz_prev.clone()[None]
    mask_t = mask_prev[None].clone()
    atom_frames=torch.zeros(seq.shape)
    bond_feats = rf2aa.data.data_loader.get_protein_bond_feats(L).long()
    chirals=torch.Tensor()
    is_sm = torch.zeros(L).bool()
    return seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, is_sm
    
    # return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
    #        xyz.float(), mask, idx.long(),\
    #        xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
    #        chain_idx, unclamp, False, [0,None], mask


def loader_fb(item, params, unclamp=False):
    
    # loads sequence/structure/plddt information 
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'], params['SCCUT'])
    
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    l_orig = msa.shape[1]
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features -- None
    xyz_t = torch.full((1,l_orig,27,3),np.nan).float()
    f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=21).float() # all gaps
    conf = torch.zeros((1,l_orig,1)).float() # zero confidence
    f1d_t = torch.cat((f1d_t, conf), -1)
    
    idx = pdb['idx']
    xyz = torch.full((len(idx),27,3),np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), 27), False)
    mask[:,:14] = pdb['mask']

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # initial structure
    xyz_prev = xyz_t[0]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    
    #print ("loader_fb", xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, unclamp, False, [0, None], mask #0 is for complete_chain

def loader_fb_fixbb(item, params, unclamp=False, pick_top=False):
    # loads sequence/structure/plddt information
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'], params['SCCUT'])

    # get msa features
    msa = a3m['msa'].long()[:1] #only load first sequence
    
    return featurize_single_chain_fixbb(msa, pdb, params, unclamp=unclamp, pick_top=pick_top, fb=True)

def loader_cn_fixbb(item, params, unclamp=False, pick_top=False):
    seq = torch.load(os.path.join(params["CN_DIR"],"seq",item+".pt"), weights_only=False)['seq'][0].long()
    pdb = torch.load(os.path.join(params['CN_DIR'], 'pdb', item+'.pt'), weights_only=False)
    pdb['xyz'] = pdb['xyz'][0]
    return featurize_single_chain_fixbb(seq, pdb, params, unclamp=unclamp, pick_top=pick_top, fb=False)

def loader_complex(item, L_s, taxID, assem, params, negative=False, pick_top=True):
    pdb_pair = item[0]
    pMSA_hash = item[1]

    msaA_id, msaB_id = pMSA_hash.split('_')
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m'
        a3m = get_msa(pMSA_fn, pMSA_hash)
    else:
        # read MSA for each subunit & merge them
        a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
        a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
        a3mA = get_msa(a3mA_fn, msaA_id)
        a3mB = get_msa(a3mB_fn, msaB_id)
        a3m = merge_a3m_hetero(a3mA, a3mB, L_s)

    # get MSA features
    msa = a3m['msa'].long()
    if negative: # Qian's paired MSA for true-pairs have no insertions... (ignore insertion to avoid any weird bias..)
        ins = torch.zeros_like(msa)
    else:
        ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=L_s)

    # read template info
    tpltA_fn = params['PDB_DIR'] + '/torch/hhr/' + msaA_id[:3] + '/' + msaA_id + '.pt'
    tpltB_fn = params['PDB_DIR'] + '/torch/hhr/' + msaB_id[:3] + '/' + msaB_id + '.pt'
    tpltA = torch.load(tpltA_fn, weights_only=False)
    tpltB = torch.load(tpltB_fn, weights_only=False)
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']//2+1)
    xyz_t_A, f1d_t_A = TemplFeaturize(tpltA, sum(L_s), params, offset=0, npick=ntempl, pick_top=pick_top)
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']//2+1)
    xyz_t_B, f1d_t_B = TemplFeaturize(tpltB, sum(L_s), params, offset=L_s[0], npick=ntempl, pick_top=pick_top)
    xyz_t = torch.cat((xyz_t_A, xyz_t_B), dim=0)
    f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=0)

    # get initial coordinates
    xyz_prev = torch.cat((xyz_t_A[0][:L_s[0]], xyz_t_B[0][L_s[0]:]), dim=0)

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt', weights_only=False)
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt', weights_only=False)

    if len(assem) > 0:
        # read metadata
        pdbid = pdbA_id.split('_')[0]
        meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt', weights_only=False)

        # get transform
        xformA = meta['asmb_xform%d'%assem[0]][assem[1]]
        xformB = meta['asmb_xform%d'%assem[2]][assem[3]]

        # apply transform
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz = torch.full((sum(L_s), 27, 3), np.nan).float()
        xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
    else:
        xyz = torch.full((sum(L_s), 27, 3), np.nan).float()
        xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
    mask = torch.full((sum(L_s), 27), False)
    mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += 200

    chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
    chain_idx[:L_s[0], :L_s[0]] = 1
    chain_idx[L_s[0]:, L_s[0]:] = 1

    # Do cropping
    if sum(L_s) > params['CROP']:
        if negative:
            sel = get_complex_crop(L_s, mask, seq.device, params)
        else:
            sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params, pdb_pair)
        #
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[sel]
        mask = mask[sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        xyz_prev = xyz_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]

    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = ChemData().INIT_CRDS.reshape(1, 27, 3).repeat(len(xyz), 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    #print ("loader_compl", xyz_t.shape, f1d_t.shape, xyz_prev.shape, negative)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, negative, [0, None], mask #0 is for complete_chain

def loader_complex_fixbb(item, L_s, taxID, assem, params, negative=False, pick_top=True):
    pdb_pair = item[0]
    pMSA_hash = item[1]

    msaA_id, msaB_id = pMSA_hash.split('_')
    assert pMSA_hash.split("_")[0] != pMSA_hash.split("_")[1], "homo oligomer set ended up in fixedBB task"
    # read MSA for each subunit & merge them
    a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
    a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
    a3mA = get_msa(a3mA_fn, msaA_id)
    a3mB = get_msa(a3mB_fn, msaB_id)
    a3m = merge_a3m_hetero(a3mA, a3mB, L_s)

    # get MSA features
    msa = a3m['msa'].long()[:1] #get first sequence
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize_fixbb(msa, params, L_s=L_s)

    f1d_t = TemplFeaturizeFixbb(seq)
    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt', weights_only=False)
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt', weights_only=False)

    if len(assem) > 0:
        # read metadata
        pdbid = pdbA_id.split('_')[0]
        meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt', weights_only=False)

        # get transform
        xformA = meta['asmb_xform%d'%assem[0]][assem[1]]
        xformB = meta['asmb_xform%d'%assem[2]][assem[3]]

        # apply transform
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz = torch.full((sum(L_s), 27, 3), np.nan).float()
        xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
    else:
        xyz = torch.full((sum(L_s), 27, 3), np.nan).float()
        xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
    mask = torch.full((sum(L_s), 27), False)
    mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += 200
    chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
    chain_idx[:L_s[0], :L_s[0]] = 1
    chain_idx[L_s[0]:, L_s[0]:] = 1

    xyz_t = torch.clone(xyz)[None]
    # get initial coordinates
    xyz_prev = xyz_t[0]
    # Do cropping
    sel, complete_chain = get_spatial_crop_fixbb(xyz, mask, torch.arange(sum(L_s)), L_s, params)
    seq = seq[:,sel]
    msa_seed_orig = msa_seed_orig[:,:,sel]
    msa_seed = msa_seed[:,:,sel]
    msa_extra = msa_extra[:,:,sel]
    mask_msa = mask_msa[:,:,sel]
    xyz = xyz[sel]
    mask = mask[sel]
    xyz_t = xyz_t[:,sel]
    f1d_t = f1d_t[:,sel]
    xyz_prev = xyz_prev[sel]

    idx = idx[sel]
    chain_idx = chain_idx[sel][:,sel]
    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = ChemData().INIT_CRDS.reshape(1, 27, 3).repeat(len(xyz), 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    #print ("loader_complex_fixbb", xyz_t.shape, f1d_t.shape, xyz_prev.shape, negative)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, False, complete_chain, mask #complete chain is [chainA or B, length of first chain]


@dataclass
class WeightedDataset:
    ids: dict
    dic: pd.DataFrame
    task_loaders: dict
    weights: np.array

@dataclass
class DatahubBackwardCompatibilityWeightedDataset:
    """
    DistilledDataset expects a dataset_configs dict with WeightedDatasets for some functions. With Datahub dataloaders don't actually need ids to be a dict (only len(ids) is relevant), and task_loaders and dic are never used. For ease of use here we use this spoof object.
    """
    ids: np.array
    weights: np.array

def default_dataset_configs(loader_param, debug=False):
    ic(loader_param['MOL_DIR'])
    print('Getting train/valid set...')
    #add in all-atom datasets
    # (
    #     pdb_items, fb_items, compl_items, neg_items, na_compl_items, na_neg_items, rna_items,
    #     sm_compl_items, sm_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_na_compl, 
    #     valid_na_neg, valid_rna, valid_sm_compl, valid_sm_compl_ligclus, valid_sm_compl_strict, 
    #     valid_sm, valid_pep, homo
    # ) = rf2aa.data.data_loader.get_train_valid_set({**rf2aa.data.compose_dataset.default_dataloader_params, **loader_param, **{'DATAPKL': loader_param['DATAPKL_AA']}}, no_match_okay=debug)
    ic(loader_param)
    dataloader_params = copy.deepcopy(rf2aa.data.compose_dataset.default_dataloader_params)
    overrides = [
        ['DATAPKL_AA', 'DATAPKL'],
        ['MOL_DIR', 'MOL_DIR'],
        ['MAX_LENGTH', 'MAXMONOMERLENGTH'],
        ['CROP', 'CROP']
    ]
    for k_diff, k_rf2aa in overrides:
        v = loader_param.get(k_diff, None)
        ic(k_diff, k_rf2aa, v)
        if v is not None:
            dataloader_params[k_rf2aa] = v

    train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, homo, chid2hash, chid2taxid, *extra = \
            rf2aa.data.data_loader.get_train_valid_set({**rf2aa.data.compose_dataset.default_dataloader_params, \
            **dataloader_params},
            no_match_okay=debug, diffusion_training=True)

    if loader_param.use_validation_config:
        train_ID_dict = valid_ID_dict
        train_dict = valid_dict
        weights_dict = {}
        ic(list(train_ID_dict.keys()))
        for k in ['pdb', 'compl', 'sm_compl', 'sm_compl_covale', 'sm_compl_asmb', 'sm_compl_multi', 'metal_compl', 'na_compl']:
            weights_dict[k] = torch.full(train_ID_dict[k].shape, 0.5)

    #all the pdb sets use the default rf2aa loader_pdb, but the fixbb adaptor will not be applied to the seq2str task
    pdb_config = WeightedDataset(train_ID_dict["pdb"], train_dict["pdb"], {
        'seq2str':      rf2aa.data.data_loader.loader_pdb,
        'str2seq':      rf2aa.data.data_loader.loader_pdb, 
        'str2seq_full': rf2aa.data.data_loader.loader_pdb, 
        'hal':          rf2aa.data.data_loader.loader_pdb, 
        'hal_ar':       rf2aa.data.data_loader.loader_pdb,
        'diff':         rf2aa.data.data_loader.loader_pdb},
        weights_dict["pdb"])

    # def pdb_aa_loader_fixbb(item, *args, **kwargs):
    #     return sm_compl_loader_fixbb(item + [()], *args, **kwargs)

    pdb_aa_config = WeightedDataset(train_ID_dict["pdb"], train_dict["pdb"], {
        'seq2str':      rf2aa.data.data_loader.loader_pdb,
        'str2seq':      rf2aa.data.data_loader.loader_pdb, 
        'str2seq_full': rf2aa.data.data_loader.loader_pdb, 
        'hal':          rf2aa.data.data_loader.loader_pdb, 
        'hal_ar':       rf2aa.data.data_loader.loader_pdb,
        'diff':         rf2aa.data.data_loader.loader_pdb},
        weights_dict["pdb"])


    compl_config = WeightedDataset(train_ID_dict["compl"], train_dict["compl"], {
        'seq2str':     rf2aa.data.data_loader.loader_complex,
        'str2seq':     rf2aa.data.data_loader.loader_complex, 
        'str2seq_full':rf2aa.data.data_loader.loader_complex, 
        'hal':         rf2aa.data.data_loader.loader_complex, 
        'hal_ar':      rf2aa.data.data_loader.loader_complex,
        'diff':        rf2aa.data.data_loader.loader_complex},
        weights_dict["compl"])

    na_compl_config = WeightedDataset(train_ID_dict["na_compl"], train_dict["na_compl"], {
        'seq2str':     rf2aa.data.data_loader.loader_na_complex,
        'str2seq':     rf2aa.data.data_loader.loader_na_complex, 
        'str2seq_full':rf2aa.data.data_loader.loader_na_complex, 
        'hal':         rf2aa.data.data_loader.loader_na_complex, 
        'hal_ar':      rf2aa.data.data_loader.loader_na_complex,
        'diff':        rf2aa.data.data_loader.loader_na_complex},
        weights_dict["na_compl"])        

    # Just need to regenerate the dataset pickle file with reference to distill tf csv file path
    # tf_distill_config = WeightedDataset(train_ID_dict["distil_tf"], train_dict["distil_tf"], {
    #     'seq2str':     rf2aa.data.data_loader.loader_distil_tf,
    #     'str2seq':     rf2aa.data.data_loader.loader_distil_tf, 
    #     'str2seq_full':rf2aa.data.data_loader.loader_distil_tf, 
    #     'hal':         rf2aa.data.data_loader.loader_distil_tf, 
    #     'hal_ar':      rf2aa.data.data_loader.loader_distil_tf,
    #     'diff':        rf2aa.data.data_loader.loader_distil_tf},
    #     weights_dict["distil_tf"])   

    rna_config = WeightedDataset(train_ID_dict["rna"], train_dict["rna"], {
        'seq2str':     rf2aa.data.data_loader.loader_dna_rna,
        'str2seq':     rf2aa.data.data_loader.loader_dna_rna, 
        'str2seq_full':rf2aa.data.data_loader.loader_dna_rna, 
        'hal':         rf2aa.data.data_loader.loader_dna_rna, 
        'hal_ar':      rf2aa.data.data_loader.loader_dna_rna,
        'diff':        rf2aa.data.data_loader.loader_dna_rna},
        weights_dict["rna"])    

    # Just need to regenerate the dataset pickle file with reference to dna csv file path
    # dna_config = WeightedDataset(train_ID_dict["dna"], train_dict["dna"], {
    #     'seq2str':     rf2aa.data.data_loader.loader_dna_rna,
    #     'str2seq':     rf2aa.data.data_loader.loader_dna_rna, 
    #     'str2seq_full':rf2aa.data.data_loader.loader_dna_rna, 
    #     'hal':         rf2aa.data.data_loader.loader_dna_rna, 
    #     'hal_ar':      rf2aa.data.data_loader.loader_dna_rna,
    #     'diff':        rf2aa.data.data_loader.loader_dna_rna},
    #     weights_dict["dna"])                                
    
    # neg_config = WeightedDataset(neg_IDs, neg_dict, loader_complex, neg_weights)
    # fb_config = WeightedDataset(train_ID_dict["fb"], train_dict["fb"], {
    #                     'seq2str':      rf2aa.data.data_loader.loader_fb,
    #                     'str2seq':      rf2aa.data.data_loader.loader_fb, 
    #                     'str2seq_full': rf2aa.data.data_loader.loader_fb, 
    #                     'hal':          rf2aa.data.data_loader.loader_fb, 
    #                     'hal_ar':       rf2aa.data.data_loader.loader_fb,
    #                     'diff':         rf2aa.data.data_loader.loader_fb},
    #                     weights_dict["fb"])
    # cn_config = WeightedDataset(cn_IDs, cn_dict, {
    #                     'seq2str':      None,
    #                     'str2seq':      loader_cn_fixbb,
    #                     'str2seq_full': loader_cn_fixbb,
    #                     'hal':          loader_cn_fixbb,
    #                     'hal_ar':       loader_cn_fixbb,
    #                     'diff':         loader_cn_fixbb},
    #                     cn_weights)

    # RF2aa change compat. Can be deleted once aa_dataset_256_subsampled_10.pkl is regenerated
    if chid2hash is None:
        chid2hash = {}
    if chid2taxid is None:
        chid2taxid = {}

    sm_compl_loader_fixbb = partial(rf2aa.data.loaders.rcsb_loader.loader_sm_compl_assembly, \
                                    chid2hash=chid2hash,chid2taxid=chid2taxid,remove_residue=False)
    sm_compl_config = WeightedDataset(
                train_ID_dict["sm_compl"], train_dict["sm_compl"], sm_compl_loader_fixbb, weights_dict["sm_compl"])
    
    sm_compl_covale_config = WeightedDataset(
                train_ID_dict["sm_compl_covale"], train_dict["sm_compl_covale"], sm_compl_loader_fixbb, weights_dict["sm_compl_covale"])
    
    o = OrderedDict({
        'pdb': pdb_config,
        'compl': compl_config,
        'na_compl': na_compl_config,
        # 'tf_distill': tf_distill_config,
        'rna': rna_config,
        # 'dna': dna_config,
        # 'negative': neg_config,
        # 'fb': fb_config,
        # 'cn': cn_config,
        # AA configs
        'pdb_aa': pdb_aa_config,
        'sm_complex': sm_compl_config,
        'sm_compl_covale': sm_compl_covale_config,
        })
    
    for k in ['sm_compl_asmb', 'sm_compl_multi', 'metal_compl']:
        o[k] = WeightedDataset(
            train_ID_dict[k], train_dict[k], sm_compl_loader_fixbb, weights_dict[k])
        
    return o, homo


fallback_spoof = {
    'chosen_dataset': 'sm_complex',
    'mask_gen_seed': 38968613,
    'sel_item': {   'ASSEMBLY': 1,
                    'CHAINID': '3gnc_C',
                    'CLUSTER': 19246,
                    'COVALENT': [],
                    'DEPOSITION': '2009-03-16',
                    'HASH': '022964',
                    'LEN_EXIST': 383,
                    'LIGAND': [('I', '396', 'QQQ')],
                    'LIGATOMS': 16,
                    'LIGATOMS_RESOLVED': 16,
                    'LIGXF': [('I', 8)],
                    'PARTNERS': [   (   'C',
                                        2,
                                        190,
                                        2.7337806224823,
                                        'polypeptide(L)'),
                                    (   [('J', '397', 'EPE')],
                                        [('J', 9)],
                                        0,
                                        7.141089916229248,
                                        'nonpoly'),
                                    (   'A',
                                        0,
                                        0,
                                        7.238761901855469,
                                        'polypeptide(L)'),
                                    (   'D',
                                        3,
                                        0,
                                        16.770910263061523,
                                        'polypeptide(L)')],
                    'RESOLUTION': 2.15,
                    'SEQUENCE': 'GPGSMAAATFHWDDPLLLDQQLADDERMVRDAAHAYAQGKLAPRVTEAFRHETTDAAIFREMGEIGLLGPTIPEQYGGPGLDYVSYGLIAREVERVDSGYRSMMSVQSSLVMVPIFEFGSDAQKEKYLPKLATGEWIGCFGLTEPNHGSDPGSMVTRARKVPGGYSLSGSKMWITNSPIADVFVVWAKLDEDGRDEIRGFILEKGCKGLSAPAIHGKVGLRASITGEIVLDEAFVPEENILPHVKGLRGPFTCLNSARYGIAWGALGAAESCWHIARQYVLDRKQFGRPLAANQLIQKKLADMQTEITLGLQGVLRLGRMKDEGTAAVEITSIMKRNSCGKALDIARLARDMLGGNGISDEFGVARHLVNLEVVNTYEGTHDIHALILGRAQTGIQAFF'},
    'task': 'diff'
}

class DistilledDatasetUnnoised(data.Dataset):
    def __init__(self,
                 dataset_configs,
                 params,
                 preprocess_param,
                 conf,
                 homo=None,
                 p_homo_cut=0.5):
        
        self.homo = homo if homo is not None else pd.DataFrame()
        self.params = params
        self.p_task = [1.0]
        self.task_names = ['diff']
        self.unclamp_cut = 0.9
        self.p_homo_cut = p_homo_cut
        self.dataset_configs = dataset_configs

        # get torsion variables
        self.preprocess_param = preprocess_param

        self.conf = conf
        self.model_adaptor = aa_model.Model(conf)
        self.last_idx = None

        self.dataset_param_overrides = self.conf.dataloader.get('dataset_param_overrides', {})
        self.fast_filters = self.conf.dataloader.get('fast_filters', {})

        def fallback_out():
            spoof = fallback_spoof
            sel_item = spoof['sel_item']
            chosen_dataset = spoof['chosen_dataset']
            dataset_config = self.dataset_configs[chosen_dataset]
            out = dataset_config.task_loaders(
                        sel_item,
                        {**rf2aa.data.compose_dataset.default_dataloader_params, **self.params, "P_ATOMIZE_MODRES": -1}, num_protein_chains=1, num_ligand_chains=1, 
                        fixbb=True,
                    )
            return out
        self.fallback_out = fallback_out

    def __len__(self):
        return sum(len(d.ids) for d in self.dataset_configs.values())

    def dataset_index_from_index(self, index):
        cur_index = index
        for dataset_name, config in self.dataset_configs.items():
            n_ids = len(config.ids)
            if cur_index < n_ids:
                return dataset_name, cur_index
            cur_index -= n_ids
        raise Exception(f'index {index} greater than combined sum of datasets: {len(self)}')
    
    def range_by_dataset_name(self):
        d = {}
        counter = 0
        for dataset_name, config in self.dataset_configs.items():
            n_ids = len(config.ids)
            d[dataset_name] = (counter, counter + n_ids)
            counter += n_ids
        return d

    def get_dataset_bounds_from_index(self, index):
        '''
        Return the bounds of the dataset specified by this index.
        i.e. if index 100 corresponds to pdb_aa and pdb_aa spans from 0-3000, return 0-3001
        '''
        dataset, _ = self.dataset_index_from_index(index)
        return self.range_by_dataset_name()[dataset]

    def getitem_unsafe(self, index):
        # This function is run when items are loaded in pytorch dataloaders
        mask_gen_seed = np.random.randint(0, 99999999)
        p_unclamp = np.random.rand()
        task_idx = np.random.choice(np.arange(len(self.task_names)), 1, p=self.p_task)[0]
        task = self.task_names[task_idx]

        chosen_dataset, index = self.dataset_index_from_index(index)
        # Uncomment to debug fallback dataset
        # if chosen_dataset == 'sm_complex':
        #     raise Exception('sm_complex debug fail')
        dataset_config = self.dataset_configs[chosen_dataset]

        ID = dataset_config.ids[index]
        if chosen_dataset == "sm_complex":
            sel_item = rf2aa.data.data_loader.sample_item_sm_compl(dataset_config.dic, ID)
        else:
            sel_item = rf2aa.data.data_loader.sample_item(dataset_config.dic, ID)

        # Run fast filters to allow for rapid rejection of example via NextExampleException
        if chosen_dataset in self.fast_filters:
            names = self.fast_filters[chosen_dataset].names
            configs = self.fast_filters[chosen_dataset].configs
            for name in names:
                getattr(fast_filters, name)(sel_item=sel_item, **configs[name])

        # use fixbb settings for MSA generation if not sequence to structure task  
        fixbb = task != "seq2str"
        
        # For reproducibility.
        if self.conf['spoof_item']:
            if hasattr(self, 'spoofed'):
                raise Exception('stopping after succesful spoofing of one item')
            spoof = eval(self.conf['spoof_item'])
            mask_gen_seed = spoof['mask_gen_seed']
            sel_item = spoof['sel_item']
            task = spoof['task']
            chosen_dataset = spoof['chosen_dataset']
            dataset_config = self.dataset_configs[chosen_dataset]
            self.spoofed=True
        run_inference.seed_all(mask_gen_seed) # Reseed the RNGs for test stability.

        item_context = pprint.pformat({
            'chosen_dataset': chosen_dataset,
            'index': index,
            'sel_item': sel_item,
            'task': task,
            'mask_gen_seed': mask_gen_seed}, indent=4)

        # Allow datasets to specifically override params
        params = copy.deepcopy(self.params)
        if chosen_dataset in self.dataset_param_overrides:
            for key, value in self.dataset_param_overrides[chosen_dataset].items():
                params[key] = value
        
        with error.context(item_context):
            if chosen_dataset == 'cn':
                raise NotImplementedError("new aa dataset don't have backwards compatibility with CN set")
                # chosen_loader = dataset_config.task_loaders[task]
                # out = chosen_loader(sel_item[0], self.params)

            elif chosen_dataset == 'negative':
                raise NotImplementedError("new aa dataset don't have backwards compatibility with negative set")
                # out = dataset_config.task_loaders(sel_item[0], sel_item[1], sel_item[2], sel_item[3], self.params, negative=True)

            elif chosen_dataset == 'compl':
                chosen_loader = dataset_config.task_loaders[task]
                out = chosen_loader(sel_item, 
                    {**rf2aa.data.compose_dataset.default_dataloader_params, **params}, fixbb=fixbb)
            elif chosen_dataset in ['na_compl']:
                # Note, fixbb not available for rf2aa data loaders
                chosen_loader = dataset_config.task_loaders[task]
                out = chosen_loader(sel_item, 
                    {**rf2aa.data.compose_dataset.default_dataloader_params, **params}, native_NA_frac=0.0, fixbb=fixbb)
            elif chosen_dataset in ['rna', 'dna']:
                # Note, fixbb not available for rf2aa data loaders
                chosen_loader = dataset_config.task_loaders[task]
                out = chosen_loader(sel_item, 
                    {**rf2aa.data.compose_dataset.default_dataloader_params, **params}, fixbb=fixbb)
            elif chosen_dataset == 'pdb' or chosen_dataset == 'pdb_aa':
                chosen_loader = dataset_config.task_loaders[task]
                # if p_unclamp > self.unclamp_cut:
                #     out = chosen_loader(sel_item[0], self.params, self.homo, unclamp=True, p_homo_cut=self.p_homo_cut)
                # else:
                #     out = chosen_loader(sel_item[0], self.params, self.homo, unclamp=False, p_homo_cut=self.p_homo_cut)
                out = chosen_loader(sel_item, 
                    {**rf2aa.data.compose_dataset.default_dataloader_params, **params}, self.homo, p_homo_cut=-1.0,fixbb=fixbb)
            elif chosen_dataset == 'fb':
                # print('Chose fb')
                chosen_loader = self.fb_loaders[task]
                if p_unclamp > self.unclamp_cut:
                    out = chosen_loader(sel_item, params, unclamp=True, fixbb=fixbb)
                else:
                    out = chosen_loader(sel_item, params, unclamp=False,fixbb=fixbb)
            elif chosen_dataset in {'sm_complex', 'sm_compl_covale', 'sm_compl_asmb', 'sm_compl_multi', 'metal_compl'}:
                num_protein_chains = None
                num_ligand_chains = None
                if chosen_dataset in {'sm_complex', 'sm_compl_covale'}:
                    num_protein_chains = 1
                    num_ligand_chains = 1
                out = dataset_config.task_loaders(
                    sel_item,
                    {**rf2aa.data.compose_dataset.default_dataloader_params, **params, "P_ATOMIZE_MODRES": -1},
                    num_protein_chains=num_protein_chains, num_ligand_chains=num_ligand_chains,
                    fixbb=fixbb,
                )
            else:
                raise Exception(f'chosen_dataset {chosen_dataset} not implemented')
            
            assert fixbb
            assert chosen_dataset != 'complex', 'complex requires passing same_chain to mask_generators, and this is not implemented'

            def process_out(out):
                # Convert template-based modeling inputs to a description of a single structure (the query structure).
                indep, atom_mask = aa_model.adaptor_fix_bb_indep(out)
                if indep.is_sm.all():
                    raise Exception('is_sm is true for all indices')
                
                # Uncomment for debugging weird heteroatoms from the rf2aa dataloaders
                # from dev import analyze
                # analyze.make_network_cmd(show.cmd)
                # analyze.clear()
                # show.one(indep, None, 'both')
                # # analyze.make_network_cmd(show.cmd)
                # prot, _ = show.one(aa_model.slice_indep(indep, ~indep.is_sm)[0], None, 'prot')
                # het, _ = show.one(aa_model.slice_indep(indep, indep.is_sm)[0], None, 'het')

                pop = aa_model.is_occupied(indep, atom_mask)  # pop = True iff residue with N CA C present or atom with coords
                # For now, do not pop unoccupied small molecule atoms, exit instead, as popping them can lose covale information.
                unoccupied_sm = (~pop) * indep.is_sm
                if unoccupied_sm.any():
                    raise Exception(f'there are small molecule atoms that are unoccupied at indices:  {unoccupied_sm.nonzero()[:,0]}')
                aa_model.pop_mask(indep, pop)
                atom_mask = atom_mask[pop]
                # name, names = show.one(indep, None, 'before_deatomize_covales')
                # show.cmd.do(f'util.cbc {name}')

                if self.conf.dataloader.max_residues > -1:
                    # Kind of hacky as it may break covales.  Only for debugging.
                    pop = indep.is_sm.clone()
                    residue_indices = torch.where(~indep.is_sm)[0]
                    residue_indices = residue_indices[:self.conf.dataloader.max_residues]
                    pop[residue_indices] = True
                    aa_model.pop_mask(indep, pop)
                    atom_mask = atom_mask[pop]

                #indep, atom_mask, metadata = aa_model.deatomize_covales(indep, atom_mask)
                metadata={}

                return {
                    'indep': indep,
                    'atom_mask': atom_mask,
                    'metadata': metadata,
                    'chosen_dataset': chosen_dataset,
                    'sel_item': sel_item,
                    'task': task,
                    'item_context': item_context,
                    'mask_gen_seed': mask_gen_seed,
                    'params': self.params,
                    'conditions_dict': {}
                }
            processed = process_out(out)
            return processed

    def __getitem__(self, index):
        return self.getitem_unsafe(index)
    


def get_class_name(f):
    clas = getattr(f, '__class__', None)
    if clas is None:
        return None
    return getattr(clas, '__qualname__')

def spy(*args):
    return None

class TransformedDataset(data.Dataset):
    '''
    Applies transformations to a dataset following the pytorch Transformation paradigm:
    https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

    Called after the dataset is loaded, but before it is returned to the user.
    '''

    def __init__(self,
                 dataset,
                 transforms):
            self.transforms = transforms
            self.dataset = dataset
            
    def __getitem__(self, index: int):
        return self.getitem_inner(index)
        
    def getitem_inner(self, index: int, **kwargs):
        # Entry point for dataset iteration
        if hasattr(self.dataset, 'getitem_inner'):
            feats = self.dataset.getitem_inner(index, **kwargs)
        else:
            feats = self.dataset[index]  # feats has {'indep', 'atom_mask', 'metadata', 'chosen_dataset', 'sel_item', 'task', 'item_context', 'mask_gen_seed', 'params', 'conditions_dict'}

        spy({'feats_into_transform_stack':copy.deepcopy(feats)})

        logger.debug(f'Transform root inputs: {set(feats.keys()) if isinstance(feats, dict) else type(feats)}')

        # Iterate through all transforms in order and update the features
        with error.context(feats.get('item_context', 'no item context')):
            for T in self.transforms:
                feats_before = set(feats.keys()) if isinstance(feats, dict) else set()
                feats = T(**feats)

                # Logger information to track changes
                feats_after = set(feats.keys()) if isinstance(feats, dict) else set()
                new_feats = feats_after - feats_before
                removed_feats = feats_before - feats_after
                logger.debug(f'Transform[{get_class_name(T)}] added: {new_feats}    removed: {removed_feats}')
            logger.debug(f'Transform root outputs: {set(feats.keys()) if isinstance(feats, dict) else type(feats)}')
            return feats


    def get_dataset_bounds_from_index(self, index):
        return self.dataset.get_dataset_bounds_from_index(index)

    def __len__(self):
        return len(self.dataset)

def feature_tuple_from_feature_dict(**kwargs):
    return (
            kwargs['indep'],
            kwargs['rfi'],
            kwargs['chosen_dataset'],
            kwargs['sel_item'],
            kwargs['t'],
            kwargs['is_diffused'],
            kwargs['task'],
            kwargs['atomizer'],
            kwargs['masks_1d'],
            kwargs['diffuser_out'],
            kwargs['item_context'],
            kwargs['conditions_dict']
    )


def get_t_training(conf: OmegaConf)-> Tuple[int, float]:
    """
    Get the diffusion time t for the diffuser for training

    Args:
        conf (OmegaConf): training config with diffuser info

    Returns:
        t (int, float): discrete time step
        t_cont (float): continuous time step
    """
    if conf.diffuser.time_type == 'discrete':
        t = random.randint(1, conf.diffuser.T)
        t_cont = t / conf.diffuser.T
    elif conf.diffuser.time_type == 'continuous':
        distribution = getattr(distributions, conf.diffuser.t_distribution)
        t_cont = distribution.rvs(1)[0]
        t = t_cont * conf.diffuser.T
    else:
        raise ValueError(f"Invalid option: {conf.diffuser.time_type}. Please choose from <'discrete', 'continuous'>.")
    return t, t_cont


class DistilledDataset(data.Dataset):
    def __init__(self, dataset, params, diffuser, preprocess_param, conf, homo=None, p_homo_cut=0.5, **kwargs):
        self.diffuser = diffuser
        self.params = params
        self.conf = conf
        self.preprocess_param = preprocess_param
        self.model_adaptor = aa_model.Model(conf)


        def diffuse(indep, metadata, chosen_dataset, sel_item, task, masks_1d, item_context, mask_gen_seed, 
                    is_masked_seq, is_diffused, atomizer, conditions_dict, **kwargs):
            t, t_cont = get_t_training(self.conf)

            assert not hasattr(self.conf, 'extra_t1d'), 'extra_t1d has been replaced by extra_tXd'
            assert not hasattr(self.conf, 'extra_t1d_params'), 'extra_t1d has been replaced by extra_tXd'

            # ... create extra 1d and 2d track features
            indep.extra_t1d, indep.extra_t2d = features.get_extra_tXd(indep, self.conf.extra_tXd, t_cont=t_cont, **self.conf.extra_tXd_params, **conditions_dict)

            # ... re-seed the RNGs for test stability
            run_inference.seed_all(mask_gen_seed)

            if conf.get('motif_only_2d', False):
                is_diffused[:] = True

            masks_1d['was_noised_in_3d'] = is_diffused

            # diffuse the indep
            indep_t, diffuser_out = aa_model.diffuse(self.conf, diffuser, indep, is_diffused, t)

            # Frame corruption
            if self.preprocess_param['randomize_frames']:
                print('randomizing frames')
                indep_t.xyz = aa_model.randomly_rotate_frames(indep_t.xyz)  # TODO: Q(Woody) Make sure only the diffused frames are randomized?
            elif self.preprocess_param.get('eye_frames', False): 
                indep_t.xyz = aa_model.eye_frames(indep_t.xyz)
            else: 
                pass
            # ... mask the sequence
            indep_t = aa_model.mask_seq(indep_t, is_masked_seq) # Changed to new function that allows for multiple polymers

            assert not kwargs.get('t', False), 't should not be passed in kwargs'
            featurize_kwargs = {'indep_t'           : indep_t, 
                                't'                 : t, 
                                'is_diffused'       : is_diffused, 
                                'model_adaptor'     : self.model_adaptor,
                                'masks_1d'          : masks_1d,
                                'template_t2d'      : conditions_dict.get('template_t2d'),
                                'template_xyz'      : conditions_dict.get('template_xyz')
                            }
        
            rfi = wrap_featurize(**featurize_kwargs)


            # Sanity checks
            if torch.sum(~is_diffused) > 0:
                mydiff = torch.mean(rfi.xyz[:,~is_diffused,1] - indep.xyz[None,~is_diffused,1]) 
                assert mydiff < 0.001, f'xyz diff: {mydiff}'
            run_inference.seed_all(mask_gen_seed) # Reseed the RNGs for test stability.
            indep.metadata = metadata


            return dict(
                indep=indep,
                rfi=rfi,
                chosen_dataset=chosen_dataset,
                sel_item=sel_item,
                t=t,
                is_diffused=is_diffused,
                task=task,
                atomizer=atomizer,
                masks_1d=masks_1d,
                diffuser_out=diffuser_out,
                item_context=item_context,
                conditions_dict=conditions_dict
            )

        transforms = []
        # Add training only transforms
        upstream_names = self.conf.upstream_training_transforms.names if hasattr(self.conf, 'upstream_training_transforms') else []
        for transform_name in upstream_names:
            transforms.append(
                getattr(conditioning, transform_name)(**self.conf.upstream_training_transforms.configs[transform_name]),
            )
        # Add shared training/inference transforms
        for transform_name in self.conf.transforms.names:
            transforms.append(
                getattr(conditioning, transform_name)(**self.conf.transforms.configs[transform_name]),
            )
        # Add training only downstream transforms
        transforms.extend([
            diffuse,
            feature_tuple_from_feature_dict
        ])

        self.dataset = TransformedDataset(dataset, transforms)

    def get_dataset_bounds_from_index(self, index):
        return self.dataset.get_dataset_bounds_from_index(index)
    
    def __getitem__(self, i):
        return self.dataset[i]
    
    def __len__(self):
        return len(self.dataset)

class DatasetWithNextExampleRetry(data.Dataset):
    '''
    A dataset that allows one to throw NextExampleException in order to retry the train loader with a different example

    Only currently supports retrying the same dataset, but ideally it would retry the same mask again too if thrown from a mask
    '''

    def __init__(self,
                 dataset,
                 max_retries=200):
        self.dataset = dataset
        self.max_retries = max_retries

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except NextExampleException as e:
            if e.get_message():
                print(e.get_message())

        rng = random.Random(index) # Seed once with the index to prevent unlucky short random seed loops
        lb, ub = self.dataset.get_dataset_bounds_from_index(index)

        for attempt in range(self.max_retries):
            next_index = rng.randint(lb, ub-1)
            try:
                return self.dataset[next_index]
            except NextExampleException as e:
                if e.get_message():
                    print(e.get_message())

        print(f"WARNING: dataset.__getitem__[{index}] raised NextExampleException {self.max_retries} times! Aborting DatasetWithNextExampleRetry")
        assert False

    def __len__(self):
        return len(self.dataset)

    def get_dataset_bounds_from_index(self, index):
        return self.dataset.get_dataset_bounds_from_index(index)

class DatasetFromMultipleDatasets(data.Dataset):
    '''
    A dataset that allows one to combine multiple datasets
    '''

    def __init__(self,
                 datasets,
                 dataset_configss,
                 dataset_optionss,
                 dataset_probs
        ):

        assert len(dataset_configss) == len(datasets)
        assert len(dataset_optionss) == len(datasets)
        assert len(dataset_probs) == len(datasets)

        self.datasets = datasets

        dataset_lengths = torch.tensor([len(x) for x in datasets], dtype=int)
        self.dataset_starts = torch.cat((torch.tensor([0], dtype=int), torch.cumsum(dataset_lengths, 0)[:-1]))

        # We're not going to allow people to have the same dataset names otherwise we'd have to internally rename them which is bad
        self.dataset_configs = {}
        for dataset_configs in dataset_configss:
            for key in dataset_configs:
                assert key not in self.dataset_configs, f'A dataset name was duplicated between datahub and old: {key}'
                self.dataset_configs[key] = dataset_configs[key]


        # Split on commas, turn into catted list, rejoin on commas. Protects against empty lists
        parsed_dataset_options = list(itertools.chain(*[x.split(',') for x in dataset_optionss]))
        self.dataset_options = ','.join(parsed_dataset_options)

        # Turn everything into a list. Cat lists. Turn it back into a tensor
        self.dataset_prob = torch.tensor(list(itertools.chain(*[list(x) for x in dataset_probs])), dtype=float)


    def __getitem__(self, index):
        i_dataset, local_index = self._index_to_local_index(index)
        return self.datasets[i_dataset][local_index]

    def __len__(self):
        return sum(len(x) for x in self.datasets)


    def _index_to_local_index(self, index):
        i_dataset = torch.searchsorted(self.dataset_starts, index, right=True)-1
        local_index = index - self.dataset_starts[i_dataset]
        return int(i_dataset), int(local_index)

    def get_dataset_bounds_from_index(self, index):
        i_dataset, local_index = self._index_to_local_index(index)
        return self.datasets[i_dataset].get_dataset_bounds_from_index(local_index)



class DatasetWithFallback(data.Dataset):

    def __init__(self,
                 dataset,
                 fallback_dataset,
                 fallback_sampler,
                 use_fallback=True,
                 max_attempts=10):
            self.dataset = dataset
            self.fallback_dataset = fallback_dataset
            self.fallback_sampler = fallback_sampler
            self.fallback_iter = itertools.cycle(fallback_sampler.__iter__())
            self.use_fallback = use_fallback
            self.max_attempts = max_attempts
        
    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except Exception as e:
            # Attempt up retry the new dataset up to max_attempts times
            orig_traceback = traceback.format_exc()
            if not self.use_fallback:
                print(f'WARNING: dataset.__getitem__[{index}] raised exception with no fallback: {orig_traceback}')
                raise e
            for _ in range(self.max_attempts):
                fallback_index = next(self.fallback_iter)
                with contextlib.suppress(Exception):
                    out = self.fallback_dataset[fallback_index]
                    print(f'WARNING: dataset.__getitem__[{index}] raised exception, falling back to fallback_dataset[{fallback_index}]: {orig_traceback}')
                    return out
            # Unable to find fallback
            print(f'ERROR: dataset.__getitem__[{index}] raised exception that could not be resolved with fallback dataset: {traceback.format_exc()}')
            raise e


class DistributedWeightedSampler(data.Sampler):
    def __init__(self, dataset_configs, dataset_options, dataset_prob, num_example_per_epoch, \
                 num_replicas=None, rank=None, replacement=False, seed_offset=0):
        self.weights_by_dataset = OrderedDict({k:c.weights for k, c in dataset_configs.items()})
        dataset_options = dataset_options.split(",")
        num_datasets = len(dataset_options)
        for dset in dataset_options:
            assert dset in self.weights_by_dataset, f'{dset} not in {list(self.weights_by_dataset.keys())}'
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        # make dataset divisible among devices 
        num_example_per_epoch -= num_example_per_epoch % num_replicas 
        assert num_example_per_epoch % num_replicas == 0

        self.num_replicas = num_replicas
        
        # Parse dataset_options and dataset_prop
        self.dataset_dict = {k:0 for k in self.weights_by_dataset}

        if dataset_prob is None:
            dataset_prob = [1.0/num_datasets]*num_datasets
        else:
            assert math.isclose(sum(dataset_prob), 1.0), dataset_prob
            assert len(dataset_prob) == len(dataset_options)
        for idx,dset in enumerate(dataset_options):
            if not idx == num_datasets-1:
                self.dataset_dict[dset] = int(dataset_prob[idx]*num_example_per_epoch)
            else:
                self.dataset_dict[dset] = num_example_per_epoch - sum([val for k, val in self.dataset_dict.items()])

        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.seed_offset = seed_offset

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed_offset)
        run_inference.seed_all(self.epoch * self.num_replicas + self.rank + self.seed_offset) # Reseed the RNGs for test stability.
        self.epoch += 1

        # get indices (fb + pdb models)
        # indices = torch.arange(len(self.dataset))

        total_dataset_size = sum(len(weights) for weights in self.weights_by_dataset.values())
        indices = torch.arange(total_dataset_size)

        # weighted subsampling
        # 1. subsample fb and pdb based on length
        sel_indices = torch.tensor((),dtype=int)
        offset = 0
        for dataset_name, weights in self.weights_by_dataset.items():
            n_for_dataset = self.dataset_dict[dataset_name]
            if n_for_dataset != 0:
                sampled = torch.multinomial(weights, n_for_dataset, self.replacement, generator=g)
                sel_indices = torch.cat((sel_indices, indices[sampled + offset]))
            offset += len(weights)

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def no_batch_collate_fn(data):
    assert len(data) == 1
    return data[0]

def get_dataset_and_sampler(dataset_configs, dataset_options, dataset_prob, conf, diffuser, num_example_per_epoch, world_size, rank, homo):

    datasets = []
    dataset_configss = []
    dataset_optionss = []
    dataset_probs = []

    datasets.append( DistilledDatasetUnnoised(
                dataset_configs=dataset_configs,
                params=conf.dataloader,
                preprocess_param=conf.preprocess,
                conf=conf,
                homo=homo,
                )
        )
    dataset_configss.append(dataset_configs)
    dataset_optionss.append(dataset_options)
    dataset_probs.append(dataset_prob)


    unnoised_dataset = DatasetFromMultipleDatasets(datasets, dataset_configss, dataset_optionss, dataset_probs)
    dataset_configs = unnoised_dataset.dataset_configs
    dataset_options = unnoised_dataset.dataset_options
    dataset_prob = unnoised_dataset.dataset_prob

    dataset = DistilledDataset(
        dataset=unnoised_dataset,
        params=conf.dataloader,
        diffuser=diffuser, 
        preprocess_param=conf.preprocess,
        conf=conf,
        homo=homo
    )

    dataset = DatasetWithNextExampleRetry(dataset)

    sampler = DistributedWeightedSampler(
        dataset_configs=dataset_configs,
        dataset_options=dataset_options,
        dataset_prob=dataset_prob,
        num_example_per_epoch=num_example_per_epoch,
        num_replicas=world_size, 
        rank=rank, 
        replacement=True
    )

    return dataset, sampler
    

def get_fallback_dataset_and_dataloader(conf, diffuser, num_example_per_epoch, world_size, rank, LOAD_PARAM):
    # Make primary dataset
    primary_dataset_configs, homo = default_dataset_configs(conf.dataloader, debug=conf.debug)
    primary_dataset, primary_sampler = get_dataset_and_sampler(
        dataset_configs=primary_dataset_configs, 
        dataset_options=conf.dataloader['DATASETS'], 
        dataset_prob=conf.dataloader['DATASET_PROB'], 
        conf=conf, 
        diffuser=diffuser, 
        num_example_per_epoch=num_example_per_epoch,
        world_size=world_size, 
        rank=rank,
        homo=homo,
    )

    # Make secondary dataset
    secondary_dataset_configs = {'pdb_aa': primary_dataset_configs['pdb_aa']}
    secondary_dataset, secondary_sampler = get_dataset_and_sampler(
        dataset_configs = secondary_dataset_configs, 
        dataset_options='pdb_aa',
        dataset_prob=[1.0],
        conf=conf, 
        diffuser=diffuser, 
        num_example_per_epoch=num_example_per_epoch,
        world_size=world_size, 
        rank=rank,
        homo=homo,
    )

    # Combine primary and secondary datasets to make the fallbacks
    fallback_dataset = DatasetWithFallback(primary_dataset, secondary_dataset, secondary_sampler, use_fallback=conf.dataloader.use_fallback)
    fallback_train_loader = data.DataLoader(
        dataset=fallback_dataset, 
        sampler=primary_sampler, 
        batch_size=conf.batch_size, 
        collate_fn=no_batch_collate_fn, 
        **LOAD_PARAM
    )

    return fallback_dataset, fallback_train_loader
