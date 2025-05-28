from __future__ import annotations  # Fake Import for type hinting, must be at beginning of file

import torch
import numpy as np
from opt_einsum import contract as einsum

from rf_diffusion.util import rigid_from_3_points, get_mu_xt_x0
from rf2aa.kinematics import get_dih
import rf2aa.util
from rf2aa.scoring import HbHybType
from rf_diffusion.diff_util import th_min_angle 
import rf_diffusion.nucleic_compatibility_utils as nucl_util
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.kinematics import th_kabsch
from rf2aa.loss.loss import compute_general_FAPE
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rf_diffusion.aa_model import Indep
    from omegaconf import OmegaConf


def weighted_decay_sum(losses, gamma=0.99):
    assert len(losses.shape) == 1
    I = losses.shape[0]

    w_loss = torch.pow(torch.full((I,), gamma, device=losses.device), torch.arange(I, device=losses.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * losses).sum()
    return tot_loss


def calc_discontiguous_motif_rmsd(pred, true, is_computed):
    """
    Computes Kabsh RMSD over each contiguous chunk of residues in <is_computed>. 

    Assumes the islands of True are independent.
    """
    assert len(pred.shape) == 5     # (I,B,L,atoms,3)
    pred = pred[-1,0]               # (L,atoms,3)
    assert len(true.shape) == 4     # (B,L,atoms,3)
    true = true[0]                  # (L,atoms,3)
    chunk_start_stops = torch.where(torch.diff(torch.cat([torch.tensor([0]), is_computed, torch.tensor([0])])))[0].view(-1,2)

    rmsds = []

    for start, stop in chunk_start_stops:
        cur_pred = pred[start:stop,:3,:]
        cur_true = true[start:stop,:3,:]

        rms,_,_ = th_kabsch(cur_pred.reshape(-1,3), cur_true.reshape(-1,3))
        rmsds.append(rms)

    return torch.tensor(rmsds).mean()


def calc_discontiguous_motif_rmsd_atoms(pred, true, is_computed, Ls):
    """
    Computes Kabsh RMSD over each contiguous chunk of residues in <is_computed>. 

    Assumes the islands of True are independent.
    """
    assert all(is_computed), 'Assuming all atomized regions are computed in rmsd calc'
    assert isinstance(Ls, list) and all([isinstance(a,int) for a in Ls])
    assert len(pred.shape) == 5     # (I,B,L,atoms,3)
    pred = pred[-1,0]               # (L,atoms,3)
    assert len(true.shape) == 4     # (B,L,atoms,3)
    true = true[0]                  # (L,atoms,3)


    cum_Ls = np.cumsum(Ls)

    rmsds = []
    for i,end in enumerate(cum_Ls):
        start = cum_Ls[i-1] if i > 0 else 0

        cur_pred = pred[start:end,1]
        cur_true = true[start:end,1]

        rms,_,_ = th_kabsch(cur_pred, cur_true)
        rmsds.append(rms)

    return torch.tensor(rmsds).mean()



def compute_fape_losses(indep           : Indep, 
                        masks_1d        : dict, 
                        pred_in         : torch.Tensor,
                        conf            : OmegaConf,
                        mask_crds       : torch.Tensor,
                        fi_dev          : torch.Tensor,
                        atom_frames     : torch.Tensor,
                        diffusion_mask  : torch.Tensor,
                        negative        : bool = False,
                        sm_Ls           : list = []):
    """
    Computes FAPE losses for the example. There are multiple possible contributors: 

    (1) FAPE on protein motif elements  
    (2) FAPE on protein non-motif elements 
    (3) FAPE on intra-ligand token pairs
    (4) FAPE on inter protein-ligand token pairs

    Args: 
        indep (aa_model.Indep): The true structure and sequence. 
        
        masks_1d (dict): Masks from mask generator

        pred_in (torch.Tensor): The predicted structure

        loss_weights (dict): Loss weights for the different losses

        conf (OmegaConf): The training config 

        mask_crds (torch.Tensor): Mask denoting which crds exist in the true structure

        fi_dev (torch.Tensor): The frame indices for the atoms

        atom_frames (torch.Tensor): indices of atoms for each atom frame

        diffusion_mask (torch.Tensor): False if token was diffused, True if held fixed at native crds

        negative (bool): whether or not the example was a "negative" ppi example

        sm_Ls (list): List of the lengths of each ligand (# atoms) in the example.
                      Used for splitting up the RMSD calculations if there are multiple ligands.

    Written by DJ, sourcing compute_general_FAPE and helper functions from Krishna et al.
    """
    
    ## Organize some variables  
    floss_dict = {} # fape loss dictionary
    true = indep.xyz[None]
    
    if conf.loss.backprop_non_displacement_on_given:
        pred = pred_in 
    else: 
        pred = torch.clone(pred_in)
        pred[:,:,diffusion_mask] = pred_in[:,:,diffusion_mask].detach()

    I,B,L = pred.shape[:3]


    is_sm         = indep.is_sm
    # is_motif      = masks_1d['input_str_mask'].to(device=pred.device)
    is_motif      = masks_1d['is_templated_motif'].to(device=pred.device)
    is_prot_motif = is_motif & ~is_sm
    is_sm_motif   = is_motif & is_sm
    is_prot_only_non_motif = ~is_motif & ~is_sm

    seq         = indep.seq[None]
    same_chain  = indep.same_chain[None]

    norm_fape   = conf.experiment.norm_fape
    clamp_fape  = conf.experiment.d_clamp_fape
    unclamp     = torch.tensor([False])
    

    frames, frame_mask = rf2aa.util.get_frames(pred_in[-1,:,...], mask_crds, seq, fi_dev, atom_frames)
    # update frames and frames_mask to only include BB frames (have to update both for compatibility with compute_general_FAPE)
    frames_BB = frames.clone()
    frames_BB[..., 1:, :, :] = 0        # all frames except BB frames are set to zero
    frame_mask_BB = frame_mask.clone()
    frame_mask_BB[...,1:] =False
    

    ###################################################
    ### (1) FAPE on protein motif (non SM) residues ###
    ###################################################
    t2d_is_revealed = masks_1d['is_motif_2d']
    t2d_is_revealed_protein_only = t2d_is_revealed.clone()
    # mask out all (i,j) pairs where i or j is a small molecule
    t2d_is_revealed_protein_only[is_sm,:] = False
    t2d_is_revealed_protein_only[:,is_sm] = False


    if is_prot_motif.sum() > 0:
        had_prot_motif = True
        tot_motif_fape, _ = calc_str_loss(pred, true, t2d_is_revealed_protein_only.to(device=pred.device), same_chain, negative=negative,
                                            A=norm_fape/2, d_clamp=None if unclamp else clamp_fape/2, gamma=1.0)
    else:
        had_prot_motif = False
        tot_motif_fape = torch.tensor(float('nan')) 


    ######################################
    ### (2) FAPE on non-motif residues ###
    ######################################
    tot_nonmotif_fape, _ = calc_str_loss(pred, true, ~t2d_is_revealed.to(device=pred.device), same_chain, negative=negative,
                                             A=norm_fape, d_clamp=None if unclamp else clamp_fape, gamma=1.0)


    #############################
    ### (3) Ligand Intra FAPE ###
    #############################
    dclamp_sm = 4.0
    Z_sm = 4.0
    dclamp = clamp_fape
    Z = norm_fape
    res_mask = ~((mask_crds[:,:,:3].sum(dim=-1) < 3.0) * ~(rf2aa.util.is_atom(seq)))

    # create 2d masks for intrachain and interchain fape calculations
    nframes = frame_mask.shape[-1]
    frame_atom_mask_2d_allatom = torch.einsum('bfn,bra->bfnra', frame_mask_BB, mask_crds).bool() # B, L, nframes, L, natoms
    # frame_atom_mask_2d = frame_atom_mask_2d_allatom[:, :, :, :, :3]
    frame_atom_mask_2d_intra_allatom = frame_atom_mask_2d_allatom * same_chain[:, :,None, :, None].bool().expand(-1,-1,nframes,-1, ChemData().NTOTAL)
    frame_atom_mask_2d_intra = frame_atom_mask_2d_intra_allatom[:, :, :, :, :3]

    # FAPE within ligand
    sm_res_mask = (rf2aa.util.is_atom(seq)*res_mask).squeeze()

    # DJ - fix masks so invalid sm atoms are not scored 
    frame_atom_mask_2d_intra[...,sm_res_mask, 0] = False
    frame_atom_mask_2d_intra[...,sm_res_mask, 2] = False
    mask_crds[...,sm_res_mask, 0] = False
    mask_crds[...,sm_res_mask, 2] = False

    if rf2aa.util.is_atom(seq).sum() > 0:
        had_sm = True
        l_fape_sm_intra, _, _ = compute_general_FAPE(
                pred[:,sm_res_mask[None],:,:3],
                true[:,sm_res_mask,:3,:3],
                atom_mask           = mask_crds[:,sm_res_mask, :3],
                frames              = frames_BB[:,sm_res_mask],
                frame_mask          = frame_mask_BB[:,sm_res_mask],
                frame_atom_mask_2d  = frame_atom_mask_2d_intra[:, sm_res_mask][:, :, :, sm_res_mask],
                dclamp=dclamp_sm,
                Z=Z_sm
            )
        # l_fape_sm_intra is (N,) shape, need to apply weighted sum
        l_fape_sm_intra = weighted_decay_sum(l_fape_sm_intra, gamma=0.99) # in ./loss.py 

    else:
        had_sm = False
        l_fape_sm_intra = torch.tensor(float('nan')) # will skip adding later
    
    ##########################################################
    ### (4) FAPE protein --> ligand and ligand --> protein ###
    ##########################################################
    if (had_sm and had_prot_motif):

        sm_prot_inter_frame_atom_is_scored = torch.zeros((L, nframes, L, ChemData().NTOTAL), device=pred.device).bool()

        # Cursed scatter of True's into 2D mask
        score_2d = sm_prot_inter_frame_atom_is_scored
        a,b,c,d = score_2d.shape

        # SM frames against protin BB atoms 
        for i in range(a):
            if is_sm_motif[i]:
                for k in range(c):
                    if is_prot_motif[k]:
                        # set i,0,k,:3 -->True 
                        score_2d[i,0,k,:3] = True

        # Prot frames against SM atoms (only available atom is first index)
        for i in range(a):
            if is_prot_motif[i]:
                for k in range(c):
                    if is_sm_motif[k]:
                        # set i,0,k,1:2 --> True 
                        score_2d[i,0,k,1:2] = True


        l_fape_prot_sm_inter, _, _ = compute_general_FAPE(
            pred[...,:3,:].squeeze(),
            true[...,:3,:],
            atom_mask           = mask_crds[...,:3],
            frames              = frames_BB,
            frame_mask          = frame_mask_BB,
            frame_atom_mask_2d  = sm_prot_inter_frame_atom_is_scored[None,...,:3],
            dclamp              = dclamp,
            Z                   = Z)

        l_fape_prot_sm_inter= weighted_decay_sum(l_fape_prot_sm_inter, gamma=0.99) # in ./loss.py

    else:
        l_fape_prot_sm_inter = torch.tensor(float('nan'))


    # Add to loss dictionary
    floss_dict['motif_fape']          = tot_motif_fape
    floss_dict['nonmotif_fape']       = tot_nonmotif_fape
    floss_dict['ligand_intra_fape']   = l_fape_sm_intra
    floss_dict['prot_lig_inter_fape'] = l_fape_prot_sm_inter



    # Report RMSD on the motif chunks that were templated # 
    motif_rmsd      = calc_discontiguous_motif_rmsd(pred.detach(), true.detach(), is_prot_motif.detach().cpu())
    non_motif_rmsd  = calc_discontiguous_motif_rmsd(pred.detach(), true.detach(), is_prot_only_non_motif.detach().cpu())

    if had_sm:
        ligand_rmsd = calc_discontiguous_motif_rmsd_atoms(pred.detach()[:,:,is_sm],
                                                            true.detach()[:,is_sm],
                                                            is_sm[is_sm],
                                                            sm_Ls)
    else:                   
        ligand_rmsd = float('nan')
                                

    # check if there was just a single ligand atom
    if rf2aa.util.is_atom(seq).sum() <= 1:
        ligand_rmsd = float('nan')


    # add rmsds
    rmsd_dict = {}
    rmsd_dict['motif_rmsd']    = motif_rmsd
    rmsd_dict['nonmotif_rmsd'] = non_motif_rmsd
    rmsd_dict['ligand_rmsd']   = ligand_rmsd

    return floss_dict, rmsd_dict

    

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss

def normalize_loss(x: torch.Tensor, gamma: float):
    '''
    Exponentially downweight losses from the structure blocks.

    Args
        x (I,): Loss from the structure predicted by each of the I structure blocks.
        gamma: Exponenet of how heavily to downwieght the losses. 
    '''
    # Exponential decay
    if x.ndim != 1:
        return x
    I, = x.shape
    device = x.device
    w_loss = torch.pow(torch.full((I,), gamma, device=device), torch.arange(I, device=device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    return torch.sum(x * w_loss, dim=-1)


def calc_generalized_interface_weights(indep: Indep, dtype: type, device:str, conf: OmegaConf):
    """
    Calculate the generalized interface weights for inter-chain interactions.

    Args:
        indep (Indep): The holy indep.
        dtype (torch.dtype): The torch data type.
        device (str): The device to place the weights on
        conf (OmegaConf): The training config.

    Returns:
        torch.tensor: The generalized interface weights with shape [1, L,]

    """
    K = torch.zeros(indep.xyz.shape[0], dtype=dtype, device=device)            

    # By default use the first atom in the sequence as the atom to check for interface contacts
    default_atom_list = [None, 'A', None] + [None] * (ChemData().NTOTAL - 3)

    # Get tensor mask of valid atoms
    is_valid = torch.zeros(indep.xyz.shape[:2], dtype=torch.bool, device=device )
    for i in range(indep.seq.shape[0]):   
        r = indep.seq[i]             
        is_valid[i] = torch.tensor([atom is not None and atom.find('H') == -1 
                                    for atom in (ChemData().aa2long[r] 
                                                if r < len(ChemData().aa2long) 
                                                else default_atom_list)
                                    ][:indep.xyz.shape[1]], 
                                    dtype=torch.bool)

    interface_kernel_width = conf.experiment.i_fm_kernel_width if 'i_fm_kernel_width' in conf.experiment else 7.5
    interface_kernel_activation_thresh = conf.experiment.i_fm_kernel_activation_thresh if 'i_fm_kernel_activation_thresh' in conf.experiment else 0.0
    interface_kernel_exponent = conf.experiment.i_fm_kernel_exponent if 'i_fm_kernel_exponent' in conf.experiment else 1

    # Iterate through all chain pairs and calculate inter chain distances for kernel weights
    for i,i_chain_mask in enumerate(indep.chain_masks()):
        for j,j_chain_mask in enumerate(indep.chain_masks()):
            i_chain_mask = torch.tensor(i_chain_mask, device=device)
            j_chain_mask = torch.tensor(j_chain_mask, device=device)
            if i == j:
                continue
            xyz_i = indep.xyz[i_chain_mask]
            xyz_j = indep.xyz[j_chain_mask]

            is_valid_i = is_valid[i_chain_mask]
            is_valid_j = is_valid[j_chain_mask]                    
            Li, N_atoms = xyz_i.shape[0], xyz_i.shape[1]
            Lj, N_atoms = xyz_j.shape[0], xyz_j.shape[1]
            cdist = torch.cdist(xyz_i.view(-1, 3), xyz_j.view(-1, 3)).view(Li, N_atoms, Lj, N_atoms)
            cdist = torch.nan_to_num(cdist, 999)
            
            # Get the mask of invalid positions and make invalid pair have a large distance
            pair_mask_valid = is_valid_i[:,:,None,None] * is_valid_j[None,None,:,:]
            cdist = cdist*pair_mask_valid + (~pair_mask_valid) * 999
            # Calculate the minimum values for each nucleic acid base position    
            cdist_min_i = cdist.min(3)[0].min(2)[0].min(1)[0]  # [Li,]
            cdist_min_j = cdist.min(3)[0].min(1)[0].min(0)[0]  # [Lj,]
            if interface_kernel_exponent == 1:
                K_i = torch.exp(- torch.relu(cdist_min_i - interface_kernel_activation_thresh) / interface_kernel_width)
                K_j = torch.exp(- torch.relu(cdist_min_j - interface_kernel_activation_thresh) / interface_kernel_width)
            elif interface_kernel_exponent == 2:
                K_i = torch.exp(- torch.square(torch.relu(cdist_min_i - interface_kernel_activation_thresh)) / interface_kernel_width**2)
                K_j = torch.exp(- torch.square(torch.relu(cdist_min_j - interface_kernel_activation_thresh)) / interface_kernel_width**2)
            else:
                K_i = torch.exp(- torch.pow(torch.relu(cdist_min_i - interface_kernel_activation_thresh), interface_kernel_exponent) / interface_kernel_width**interface_kernel_exponent)   
                K_j = torch.exp(- torch.pow(torch.relu(cdist_min_j - interface_kernel_activation_thresh), interface_kernel_exponent) / interface_kernel_width**interface_kernel_exponent)   
            # Replace with max values (corresponding to closest interchain chains interactions)
            K[i_chain_mask] = torch.maximum(K_i, K[i_chain_mask])
            K[j_chain_mask] = torch.maximum(K_j, K[j_chain_mask])

    K = K.unsqueeze(0)  # [1, L]
    return K


def mse(xyz_other: torch.Tensor, xyz_true: torch.Tensor):
    '''
    Mean squared error.

    Args
        xyz_other (..., n_atoms, n_dims): Coordinates to compare to.
        xyz_true (..., n_atoms, n_dims): True coordinates.

    Returns
        mse (...,): Mean squared error
    '''
    return (xyz_other - xyz_true).pow(2).sum(-1).mean(-1)

def frame_distance_err(R_pred, R_true):
    '''
    Args
        R_pred (I, B, L, 3, 3)
        R_true (L, 3, 3)
    '''

    I,B,L = R_pred.shape[:3]
    assert len(R_true.shape) == 3
    assert B == 1

    eps=1e-8

    true_repeated = R_true.repeat((I,B,1,1,1))
    eye_repeated  = torch.eye(3,3).repeat((I,B,L,1,1)).to(device=true_repeated.device)

    # apply transpose of prediction to true
    mm = torch.einsum('ablij,ablkj->ablik',true_repeated,R_pred)

    # Squared L2 (squared Frobenius) norm of deviation from mm and eye (I,)
    err = torch.square((mm - eye_repeated)+eps).sum(dim=(-1,-2)).mean(dim=-1).squeeze()
    return err

def frame_distance_loss(R_pred, R_true, eps=1e-8, gamma=0.99):
    """
    Calculates squared L2 loss on frames
    """
    I,B,L = R_pred.shape[:3]
    err = frame_distance_err(R_pred, R_true, eps)

    # decay on loss over iterations
    w_loss = torch.pow(torch.full((I,), gamma, device=R_pred.device), torch.arange(I, device=R_pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    err = err*w_loss

    return err.sum()


def normalize_ax_ang(V):
    """
    Gets axis angle representation normalized between [0,pi]
    
    If original AA vector was beyond pi in norm, switches the direction of the vector and scales 
    it appropriate magnitude to represent the angle going around other side of circle 
    """
    V_norm = torch.norm(V, p=2, dim=-1).detach()

    # normalize AA vectors to be norm 1 
    V_magnitude_1 = V / torch.norm(V, p=2, dim=-1)[...,None]
    
    # calculate the "good norms" - minimum angle between angle and 0
    good_norms = th_min_angle(torch.zeros_like(V_norm), V_norm*180/np.pi, radians=False)*np.pi/180
    
    # normalize the ax-ang vectors with the good norms 
    # if norm(V) > pi, switches the directon and magnitude of V to go other way around circle 
    V_w_good_norms = V_magnitude_1 * good_norms[...,None]
    
    return V_w_good_norms 


def axis_angle_loss(aa_pred, aa_true, gamma=0.99, d_clamp=None):
    """
    Calculates the squared L2 loss between the predict axis angle and true axis angle 

    'aa' is supposed to stand for 'axis_angle'/'ax_ang' here 
    """
    I = aa_pred.shape[0]
    B = aa_pred.shape[1]
    assert B == 1 # calculation assumes batch size 1 by doing squeeze  
    
    # normalize vectors to be in [0,pi] in magnitude 
    aa_pred = normalize_ax_ang(aa_pred)
    aa_true = normalize_ax_ang(aa_true)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=aa_pred.device), torch.arange(I, device=aa_pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    # squared L2 - sum down xyz dim 
    err = torch.sum(torch.square(aa_pred - aa_true[None]), dim=-1) 
    
    # NOTE: Need to set up clamp between [0,pi)
    if d_clamp is not None:
        # set squared distance clamp to d_clamp**2
        d_clamp=torch.tensor(d_clamp**2)[None].to(err.device)
        err = torch.where(err>d_clamp, d_clamp, err)

    err = torch.mean(err, dim=-1).squeeze() # (I,)

    err = err*w_loss

    return err.sum()


def get_loss_schedules(T, loss_names, schedule_types, schedule_params, constant=False):

    """
    Given a list of loss functions and schedule types, produce multiplicative weights
    as a function of timestep to apply on the loss.

    loss_list (list, required): List of loss functions (i.e., the callables)

    schedule_types (list, optional): type of schedules to use for each
    """

    assert len(schedule_types) == len(loss_names)
    assert len(schedule_params) == len(loss_names)
    if constant:
        return {}

    loss_schedules = {}

    for i,name in enumerate(loss_names):
        t = torch.arange(T)

        if schedule_types[i] == 'sigmoid':
            a = schedule_params[i]['sig_stretch']
            b = schedule_params[i]['sig_shift']*T
            # stretched and shifted sigmoid between (0,1)
            loss_schedules[name] = (1/(1+torch.exp(a*(-t+b)))).flip(0) # flip to have high point in sigmoid near 0

        elif schedule_types[i] == 'linear':
            a = schedule_params[i]['linear_start']
            b = schedule_params[i]['linear_end']
            loss_schedules[name] = torch.linspace(a, b, T) # low at zero by default
        else:
            raise NotImplementedError

    return loss_schedules

def track_xt1_displacement(true, pred, xyz_in, t, diffusion_mask, schedule, alphabar_schedule):
    """
    Function to track the displacement, and squared displacement, of the backcalculated xt-1
    Inputs:
        true (B,L,3,3)
        pred (I, B, L, 3, 3)
        xyz_in (B, I, L, 3, 3)
        t (float)
        diffusion_mask (L)
        schedule + alphabarschedule
    outputs:
        squared displacement & displacement in diffused region
    """
    mu_true,_ = get_mu_xt_x0(xyz_in[0,0], true[0], t, schedule, alphabar_schedule)
    mu_backcalc,_ = get_mu_xt_x0(xyz_in[0,0], pred[0,0], t, schedule, alphabar_schedule)

    xt1_squared_disp = calc_ca_displacement_loss(mu_true[~diffusion_mask], mu_backcalc[~diffusion_mask], squared=True).mean()
    xt1_disp = calc_ca_displacement_loss(mu_true[~diffusion_mask], mu_backcalc[~diffusion_mask], squared=False).mean()
    return xt1_squared_disp, xt1_disp

def calc_ca_displacement_loss(xyz_1, xyz_2, squared=False):
    """
    Displacement (squared or not of Ca coordinates)
    """
    if squared:
        return torch.sum(torch.square(xyz_1 - xyz_2), dim=-1)
    else:
        return torch.sqrt(torch.sum(torch.square(xyz_1 - xyz_2), dim=-1))

def calc_displacement_loss(pred, true, gamma=0.99, d_clamp=None):
    """
    Calculates squared L2 norm of error between predicted and true CA 

    pred - (I,B,L,3, 3)
    true - (  B,L,27,3)
    diffusion_mask - (L, dtype=bool)
    """
    I = pred.shape[0]
    B = pred.shape[1]


    assert B == 1
    pred = pred.squeeze(1)
    true = true.squeeze(0)

    pred_ca = pred[:,:,1,...] # (I,L,3)
    true_ca = true[:,1,...]   # (L,3)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    err = torch.sum(torch.square(pred_ca - true_ca[None,...]), dim=-1) # (I, L)

    if d_clamp is not None:
        # set squared distance clamp to d_clamp**2
        d_clamp=torch.tensor(d_clamp**2)[None].to(err.device)
        err = torch.where(err>d_clamp, d_clamp, err)

    err = torch.mean(err, dim=-1) # (I,)

    err = err*w_loss
    return err.sum()

# use improved coordinate frame generation
def get_t(N, Ca, C, non_ideal=False, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), non_ideal=non_ideal, eps=eps)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    t = Ts[:,:,None] - Ts[:,:,:,None] # t[0,1] = residue 0 -> residue 1 vector
    return einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,L,3)

def calc_str_loss(pred, true, mask_2d, same_chain, negative=False, d_clamp=10.0, d_clamp_inter=10.0, A=10.0, gamma=0.99, eps=1e-6):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2], non_ideal=True)
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])
    
    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    if d_clamp is not None:
        clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
        clamp = clamp[None]
        difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if negative:
        mask = mask_2d * same_chain
    else:
        mask = mask_2d
    # calculate masked loss (ignore missing regions when calculate loss)
    loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    return tot_loss, loss.detach() 

#resolve rotationally equivalent sidechains
def resolve_symmetry(xs, Rsnat_all, xsnat, Rsnat_all_alt, xsnat_alt, atm_mask):
    dists = torch.linalg.norm( xs[:,:,None,:] - xs[atm_mask,:][None,None,:,:], dim=-1)
    dists_nat = torch.linalg.norm( xsnat[:,:,None,:] - xsnat[atm_mask,:][None,None,:,:], dim=-1)
    dists_natalt = torch.linalg.norm( xsnat_alt[:,:,None,:] - xsnat_alt[atm_mask,:][None,None,:,:], dim=-1)

    drms_nat = torch.sum(torch.abs(dists_nat-dists),dim=(-1,-2))
    drms_natalt = torch.sum(torch.abs(dists_nat-dists_natalt), dim=(-1,-2))

    Rsnat_symm = Rsnat_all
    xs_symm = xsnat

    toflip = drms_natalt<drms_nat

    Rsnat_symm[toflip,...] = Rsnat_all_alt[toflip,...]
    xs_symm[toflip,...] = xsnat_alt[toflip,...]

    return Rsnat_symm, xs_symm

# resolve "equivalent" natives
def resolve_equiv_natives(xs, natstack, maskstack):
    if (len(natstack.shape)==4):
        return natstack, maskstack
    if (natstack.shape[1]==1):
        return natstack[:,0,...], maskstack[:,0,...]
    dx = torch.norm( xs[:,None,:,None,1,:]-xs[:,None,None,:,1,:], dim=-1)
    dnat = torch.norm( natstack[:,:,:,None,1,:]-natstack[:,:,None,:,1,:], dim=-1)
    delta = torch.sum( torch.abs(dnat-dx), dim=(-2,-1))
    return natstack[:,torch.argmin(delta),...], maskstack[:,torch.argmin(delta),...]


#torsion angle predictor loss
def torsionAngleLoss( alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8 ):
    I = alpha.shape[0]
    lnat = torch.sqrt( torch.sum( torch.square(alpha), dim=-1 ) + eps )
    anorm = alpha / (lnat[...,None])

    l_tors_ij = torch.min(
            torch.sum(torch.square( anorm - alphanat[None] ),dim=-1),
            torch.sum(torch.square( anorm - alphanat_alt[None] ),dim=-1)
        )

    l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sum( tors_planar )*I + eps)

    return l_tors+0.02*l_norm+0.02*l_planar

def compute_FAPE(Rs, Ts, xs, Rsnat, Tsnat, xsnat, Z=10.0, dclamp=10.0, eps=1e-4):
    xij = torch.einsum('rji,rsj->rsi', Rs, xs[None,...] - Ts[:,None,...])
    xij_t = torch.einsum('rji,rsj->rsi', Rsnat, xsnat[None,...] - Tsnat[:,None,...])

    diff = torch.sqrt( torch.sum( torch.square(xij-xij_t), dim=-1 ) + eps )
    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean()

    return loss

def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B,L = a.shape[:2]

    u1 = a-b
    u2 = c-b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B*L, 3)
    u2 = u2.reshape(B*L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1) # (B,L,1)
    cos_theta = torch.matmul(u1[:,None,:], u2[:,:,None]).reshape(B, L, 1)
    
    return torch.cat([cos_theta, sin_theta], axis=-1) # (B, L, 2)

def length(a, b):
    return torch.norm(a-b, dim=-1)

def torsion(a,b,c,d, eps=1e-6):
    #A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b-a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c-b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d-c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1) #[B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)
    
    cos_angle = torch.matmul(t1[:,:,None,:], t2[:,:,:,None])[:,:,0]
    sin_angle = torch.norm(u2, dim=-1,keepdim=True)*(torch.matmul(u1[:,:,None,:], t2[:,:,:,None])[:,:,0])
    
    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1)/(t1_norm*t2_norm+eps) #[B,L,2]
    return cos_sin

## ideal N-C distance, ideal cos(CA-C-N angle), ideal cos(C-N-CA angle)
def calc_BB_bond_geom_absolute(pred, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, sig_len=0.02, sig_ang=0.05):
    # WIP
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    B, L = pred.shape[:2]

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    CN_loss = torch.clamp( torch.abs(blen_CN_pred - ideal_NC) - sig_len, min=0.0 )
    CN_loss = (CN_loss).sum() / (L)
    blen_loss = CN_loss   #fd squared loss

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    bang_CNCA_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    CACN_loss = torch.clamp( torch.abs(bang_CACN_pred - ideal_CACN) - sig_ang,  min=0.0 )
    CACN_loss = (CACN_loss).sum() / (L)
    CNCA_loss = torch.clamp( torch.abs(bang_CNCA_pred - ideal_CNCA) - sig_ang,  min=0.0 )
    CNCA_loss = (CNCA_loss).sum() / (L)
    bang_loss = CACN_loss + CNCA_loss

    return blen_loss, bang_loss
def calc_BB_bond_geom(pred, true, mask_crds, eps=1e-6):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    B, L = pred.shape[:2]

    # bond length: N-CA, CA-C, C-N
    #blen_NCA_pred = length(pred[:,:,0], pred[:,:,1]).reshape(B, L) # (B, L)
    #blen_CAC_pred = length(pred[:,:,1], pred[:,:,2]).reshape(B, L)
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    
    #blen_NCA_true = length(true[:,:,0], true[:,:,1]) # (B, L)
    #blen_CAC_true = length(true[:,:,1], true[:,:,2])
    blen_CN_true  = length(true[:,:-1,2], true[:,1:,0]) # (B, L-1)
    mask_CN = blen_CN_true < 3.0
    mask_CN = torch.logical_and(mask_CN, mask_crds[:,:-1])
    mask_CN = torch.logical_and(mask_CN, mask_crds[:,1:])

    blen_loss = 0.0
    CN_loss = torch.square(blen_CN_pred - blen_CN_true)
    CN_loss = (CN_loss*mask_CN).sum() / (mask_CN.sum()+eps)
    blen_loss += torch.sqrt(CN_loss + eps)

    # bond angle: N-CA-C, CA-C-N, C-N-CA
    #bang_NCAC_pred = angle(pred[:,:,0], pred[:,:,1], pred[:,:,2]).reshape(B,L,2)
    bang_CACN_pred = angle(pred[:,:-1,1], pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1,2)
    bang_CNCA_pred = angle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1,2)

    #bang_NCAC_true = angle(true[:,:,0], true[:,:,1], true[:,:,2])
    bang_CACN_true = angle(true[:,:-1,1], true[:,:-1,2], true[:,1:,0])
    bang_CNCA_true = angle(true[:,:-1,2], true[:,1:,0], true[:,1:,1])

    bang_loss = 0.0
    CACN_loss = torch.square(bang_CACN_pred - bang_CACN_true).sum(-1)
    CACN_loss = (CACN_loss*mask_CN).sum() / (mask_CN.sum()+eps)
    CNCA_loss = torch.square(bang_CNCA_pred - bang_CNCA_true).sum(-1)
    CNCA_loss = (CNCA_loss*mask_CN).sum() / (mask_CN.sum()+eps)
    bang_loss += torch.sqrt(CACN_loss + eps)
    bang_loss += torch.sqrt(CNCA_loss + eps)

    return blen_loss, bang_loss

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, same_chain, ljparams, ljcorr, num_bonds, use_H=False, negative=False,
    lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8, normalize=True, 
):
    def ljV(dist, sigma, epsilon, lj_lin, lj_maxrad):
        linpart = dist<lj_lin*sigma
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])
        if (lj_maxrad>0):
            sd2 = sd*sd
            sd6 = sd2 * sd2 * sd2
            sd12 = sd6 * sd6
            ljE = ljE - epsilon * (sd12 - 2 * sd6)
        return ljE

    L = xs.shape[0]

    # mask keeps running total of what to compute
    aamask = aamask[seq]
    if not use_H:
        mask_prot = nucl_util.get_resi_type_mask(seq, 'prot')
        mask_na = nucl_util.get_resi_type_mask(seq, 'na')
        aamask[...,mask_prot,ChemData().NHEAVYPROT:] = False
        aamask[...,mask_na, ChemData().NHEAVY:] = False
    mask = aamask[seq][...,None,None]*aamask[seq][None,None,...]
    if negative:
        # ignore inter-chains
        mask *= same_chain.bool()[:,None,:,None]
    idxes1r = torch.tril_indices(L,L,-1)
    mask[idxes1r[0],:,idxes1r[1],:] = False
    idxes2r = torch.arange(L)
    idxes2a = torch.tril_indices(27,27,0)
    mask[idxes2r[:,None], idxes2a[:1], idxes2r[:,None], idxes2a[1:2]] = False

    # "countpair" can be enforced by making this a weight
    mask[idxes2r,:,idxes2r,:] *= num_bonds[seq,:,:]>=3 #intra-res
    mask[idxes2r[:-1],:,idxes2r[1:],:] *= (
        num_bonds[seq[:-1],:,2:3] + num_bonds[seq[1:],0:1,:] + 1 >=3 #inter-res
    )
    si,ai,sj,aj = mask.nonzero(as_tuple=True)
    ds = torch.sqrt( torch.sum ( torch.square( xs[si,ai]-xs[sj,aj] ), dim=-1 ) + eps )

    # hbond correction
    use_hb_dis = (
        ljcorr[seq[si],ai,0]*ljcorr[seq[sj],aj,1]
        + ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,0] )
    use_ohdon_dis = ( # OH are both donors & acceptors
        ljcorr[seq[si],ai,0]*ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,0]
        +ljcorr[seq[si],ai,0]*ljcorr[seq[sj],aj,0]*ljcorr[seq[sj],aj,1]
    )

    ljrs = ljparams[seq[si],ai,0] + ljparams[seq[si],aj,0]
    ljrs[use_hb_dis] = lj_hb_dis
    ljrs[use_ohdon_dis] = lj_OHdon_dis

    if use_H:
        use_hb_hdis = (
            ljcorr[seq[si],ai,2]*ljcorr[seq[sj],aj,1]
            +ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,2]
        )
        ljrs[use_hb_hdis] = lj_hbond_hdis

    # disulfide correction
    potential_disulf = ljcorr[seq[si],ai,3]*ljcorr[seq[sj],aj,3] 

    ljss = torch.sqrt( ljparams[seq[si],ai,1] * ljparams[seq[sj],aj,1] + eps )
    ljss [potential_disulf] = 0.0

    ljval = ljV(ds,ljrs,ljss,lj_lin,lj_maxrad)

    if (normalize):
        return (torch.sum( ljval )/torch.sum(aamask[seq]))
    else:
        return torch.sum( ljval )

def calc_hb(
    seq, xs, aamask, hbtypes, hbbaseatoms, hbpolys,
    hb_sp2_range_span=1.6, hb_sp2_BAH180_rise=0.75, hb_sp2_outer_width=0.357, 
    hb_sp3_softmax_fade=2.5, threshold_distance=6.0, eps=1e-8, normalize=True
):
    def evalpoly( ds, xrange, yrange, coeffs ):
        v = coeffs[...,0]
        for i in range(1,10):
            v = v * ds + coeffs[...,i]
        minmask = ds<xrange[...,0]
        v[minmask] = yrange[minmask][...,0]
        maxmask = ds>xrange[...,1]
        v[maxmask] = yrange[maxmask][...,1]
        return v
    
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    hbts = hbtypes[seq]
    hbba = hbbaseatoms[seq]

    rh,ah = (hbts[...,0]>=0).nonzero(as_tuple=True)
    ra,aa = (hbts[...,1]>=0).nonzero(as_tuple=True)
    H_xs = xs[rh,ah][:,None,:]
    A_xs = xs[ra,aa][None,:,:]
    B_xs = xs[ra,hbba[ra,aa,0]][None,:,:]
    B0_xs = xs[ra,hbba[ra,aa,1]][None,:,:]
    hyb = hbts[ra,aa,2]
    polys = hbpolys[hbts[rh,ah,0][:,None],hbts[ra,aa,1][None,:]]

    AH = torch.sqrt( torch.sum( torch.square( H_xs-A_xs), axis=-1) + eps )
    AHD = torch.acos( cosangle( B_xs, A_xs, H_xs) )
    
    Es = polys[...,0,0]*evalpoly(
        AH,polys[...,0,1:3],polys[...,0,3:5],polys[...,0,5:])
    Es += polys[...,1,0] * evalpoly(
        AHD,polys[...,1,1:3],polys[...,1,3:5],polys[...,1,5:])

    Bm = 0.5*(B0_xs[:,hyb==HbHybType.RING]+B_xs[:,hyb==HbHybType.RING])
    cosBAH = cosangle( Bm, A_xs[:,hyb==HbHybType.RING], H_xs )
    Es[:,hyb==HbHybType.RING] += polys[:,hyb==HbHybType.RING,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.RING,2,1:3], 
        polys[:,hyb==HbHybType.RING,2,3:5], 
        polys[:,hyb==HbHybType.RING,2,5:])

    cosBAH1 = cosangle( B_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    cosBAH2 = cosangle( B0_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    Esp3_1 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH1, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Esp3_2 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH2, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Es[:,hyb==HbHybType.SP3] += torch.log(
        torch.exp(Esp3_1 * hb_sp3_softmax_fade)
        + torch.exp(Esp3_2 * hb_sp3_softmax_fade)
    ) / hb_sp3_softmax_fade

    cosBAH = cosangle( B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs )
    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.SP2,2,1:3], 
        polys[:,hyb==HbHybType.SP2,2,3:5], 
        polys[:,hyb==HbHybType.SP2,2,5:])

    BAH = torch.acos( cosBAH )
    B0BAH = get_dih(B0_xs[:,hyb==HbHybType.SP2], B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs)

    d,m,l = hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width
    Echi = torch.full_like( B0BAH, m-0.5 )

    mask1 = BAH>np.pi * 2.0 / 3.0
    H = 0.5 * (torch.cos(2 * B0BAH) + 1)
    F = d / 2 * torch.cos(3 * (np.pi - BAH[mask1])) + d / 2 - 0.5
    Echi[mask1] = H[mask1] * F + (1 - H[mask1]) * d - 0.5

    mask2 = BAH>np.pi * (2.0 / 3.0 - l)
    mask2 *= ~mask1
    outer_rise = torch.cos(np.pi - (np.pi * 2 / 3 - BAH[mask2]) / l)
    F = m / 2 * outer_rise + m / 2 - 0.5
    Echi[mask2] = H[mask2] * F + (1 - H[mask2]) * d - 0.5

    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * Echi

    tosquish = torch.logical_and(Es > -0.1,Es < 0.1)
    Es[tosquish] = -0.025 + 0.5 * Es[tosquish] - 2.5 * torch.square(Es[tosquish])
    Es[Es > 0.1] = 0.
    if (normalize):
        return (torch.sum( Es ) / torch.sum(aamask[seq]))
    else:
        return torch.sum( Es )

def calc_lddt_loss(pred_ca, true_ca, pred_lddt, idx, mask_crds, mask_2d, same_chain, negative=False, eps=1e-6):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (B, nbin, L)

    I, B, L = pred_ca.shape[:3]
    
    pred_dist = torch.cdist(pred_ca, pred_ca) # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0) # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0) # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    
    delta = torch.abs(pred_dist-true_dist+eps) # (I, B, L, L)

    true_lddt = torch.zeros((I,B,L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += 0.25*torch.sum((delta<=distbin)*mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    true_lddt = torch.clamp(true_lddt, min=0.0, max=1.0)
    
    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    true_lddt_label = torch.bucketize(true_lddt, lddt_bins).long()
    lddt_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred_lddt,
                                                      true_lddt_label[-1])
    lddt_loss = (lddt_loss * mask_crds).sum() / (mask_crds.sum() + eps)
   
    # Get true CA-lddt for logging
    true_lddt = (true_lddt * mask_crds[None]).sum(dim=(1,2)) / (mask_crds.sum() + eps)
    return lddt_loss, true_lddt

def calc_lddt(pred_ca, true_ca, mask_crds, mask_2d, same_chain, negative=False, eps=1e-6):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]
    
    pred_dist = torch.cdist(pred_ca, pred_ca) # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0) # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0) # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    delta = torch.abs(pred_dist-true_dist) # (I, B, L, L)

    true_lddt = torch.zeros((I,B,L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += 0.25*torch.sum((delta<=distbin)*mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    
    true_lddt = mask_crds[None]*true_lddt
    true_lddt = true_lddt.sum(dim=(1,2)) / (mask_crds.sum() + eps)
    return true_lddt

#fd allatom lddt
def calc_allatom_lddt(P, Q, atm_mask, idx, same_chain, negative=False, eps=1e-8):
    # Inputs
    #  - P: predicted coordinates (L, 14, 3)
    #  - Q: ground truth coordinates (L, 14, 3)
    #  - atm_mask: valid atoms (L, 14)
    #  - idx: residue index (L)

    # distance matrix
    Pij = torch.square(P[:,None,:,None,:]-P[None,:,None,:,:]) # (L, L, 14, 14)
    Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
    Qij = torch.square(Q[:,None,:,None,:]-Q[None,:,None,:,:]) # (L, L, 14, 14)
    Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:,None,:,None] * atm_mask[None,:,None,:]).float()
    # ignore atoms within same residue
    pair_mask *= (idx[:,None,None,None] != idx[None,:,None,None]).float() # (L, L, 14, 14)
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:,:,None,None]

    delta_PQ = torch.abs(Pij-Qij) # (L, L, 14, 14)

    lddt = torch.zeros( P.shape[:2], device=P.device ) # (L, 14)
    for distbin in (0.5,1.0,2.0,4.0):
        lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*pair_mask, dim=(1,3)
            ) / ( torch.sum( pair_mask, dim=(1,3) ) + 1e-8)
    lddt = lddt.sum(dim=-1) / (atm_mask.sum(dim=-1)+1e-8) # L
    
    res_mask = atm_mask.any(dim=-1)
    lddt = (res_mask*lddt).sum() / (res_mask.sum() + 1e-8)

    return lddt
