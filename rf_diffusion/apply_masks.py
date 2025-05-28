import torch
import math
import random
from blosum62 import p_j_given_i as P_JGI
import numpy as np
from diffusion import get_beta_schedule
from inference.utils import get_next_ca, get_next_frames
import rf2aa.tensor_util
from rf_diffusion.chemical import ChemicalData as ChemData

def sample_blosum_mutations(seq, *args, **kwargs):
    assert len(seq.shape) == 1
    is_sm = rf2aa.util.is_atom(seq) # (L)
    sampled_seq = sample_blosum_mutations_protein(seq[~is_sm], *args, **kwargs)
    out_seq = torch.clone(seq)
    out_seq[~is_sm] = sampled_seq
    return out_seq


def sample_blosum_mutations_protein(seq, p_blosum, p_uni, p_mask):
    """
    Given a sequence,
    """
    assert len(seq.shape) == 1
    assert math.isclose(sum([p_blosum, p_uni, p_mask]), 1)
    L = len(seq)

    # uniform prob
    U = torch.full((L,20), .05)
    U = torch.cat((U,torch.zeros(L,2)), dim=-1)
    U = U*p_uni

    # mask prob
    M = torch.full((L,22),0)
    M[:,-1] = 1
    M = M*p_mask

    # blosum probs
    blosum_padded = np.full((22,22),0.)
    # handle missing token
    blosum_padded[20,20] = 1.
    blosum_padded[:20,:20] = P_JGI
    B = torch.from_numpy( blosum_padded[seq] ) # slice out the transition probabilities from blossom
    B = B*p_blosum
    # build transition probabilities for each residue
    P = U+M+B

    C = torch.distributions.categorical.Categorical(probs=P)

    sampled_seq = C.sample()

    return sampled_seq


def mask_inputs(seq, 
                msa_masked, 
                msa_full, 
                xyz_t, 
                t1d, 
                mask_msa, 
                atom_mask,
                is_sm,
                preprocess_param,
                diffusion_param,
                model_param,
                input_seq_mask=None, 
                input_str_mask=None, 
                input_floating_mask=None, 
                input_t1d_str_conf_mask=None, 
                input_t1d_seq_conf_mask=None, 
                loss_seq_mask=None, 
                loss_str_mask=None, 
                loss_str_mask_2d=None,
                diffuser=None,
                seq_diffuser=None,
                predict_previous=False,
                true_crds_in=None):
    """
    Parameters:
        seq (torch.tensor, required): (I,L) integer sequence 

        msa_masked (torch.tensor, required): (I,N_short,L,48)

        msa_full  (torch,.tensor, required): (I,N_long,L,25)
        
        xyz_t (torch,tensor): (T,L,27,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (T,L,22) this is the t1d before tacking on the chi angles 
        
        input_seq_mask (torch.tensor, required): Shape (L) rank 1 tensor where sequence is masked at False positions 

        input_str_mask (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 

        input_t1d_str_conf_mask (torch.tensor, required): Shape (L) rank 1 tensor with entries for str confidence

        input_t1d_seq_conf_mask (torch.tensor, required): Shape (L) rank 1 tensor with entries for seq confidence

        loss_seq_mask (torch.tensor, required): Shape (L)

        loss_str_mask (torch.tensor, required): Shape (L)

        loss_str_mask_2d (torch.tensor, required): Shape (L,L) 

        ...

        Other parameters are then contained in preprocess_param and diffusion_param:
        
        -preprocess_param:

            sidechain_input (boolean). Whether or not diffused sidechains should be input to the model
        
        -diffusion_param:

            decode_mask_frac (float, optional): Fraction of decoded residues which are to be corrupted 

            corrupt_blosum (float, optional): Probability that a decoded residue selected for corruption will transition according to BLOSUM62 probs 

            corrupt_unifom (float, optional): Probability that ... according to uniform probs 
        

    NOTE: in the MSA, the order is 20aa, 1x unknown, 1x mask token. We set the masked region to 22 (masked).
        For the t1d, this has 20aa, 1x unkown, and 1x template conf. Here, we set the masked region to 21 (unknown).
        This, we think, makes sense, as the template in normal RF training does not perfectly correspond to the MSA.
    """
    # print('Made it into mask inputs')
    ### Perform diffusion, pick a random t and then let that be the input template and xyz_prev
    if (diffuser is not None) :


        # NOTE: assert that xyz_t is the TRUE coordinates! Should come from fixbb loader 
        #       also assumes all 4 of seq are identical 

        # pick t uniformly
        t = random.randint(1,diffuser.T)
        #t_list, t_dict, t_dict2list = get_t_list(n_cond_steps=diffusion_param['n_self_cond_steps'],
        #                                         predict_previous=diffusion_param['predict_previous'],
        #                                         T=diffuser.T)

        # Now get 

        if t == diffuser.T: 
            t_list = [t,t]
        else: 
            t_list = [t+1,t]


        if predict_previous:
            # grab previous t. if t is 0 force a prediction of x_t=0
            assert t > 0
            tprev = t-1 
            t_list.append(tprev)
        
        # Ensures that all I dimensions are size 1 - NRB
        seq        = seq[:1] 
        msa_masked = msa_masked[:1]
        msa_full   = msa_full[:1]

        assert(seq.shape[0] == 1), "Number of repeats of seq must be 1"
        L = seq.shape[-1]

        assert xyz_t.shape[0] == 1, 'multiple xyz_t templates not supported'
        kwargs = {'xyz'                     :xyz_t.squeeze(),
                  'seq'                     :seq.squeeze(0),
                  'atom_mask'               :atom_mask.squeeze(),
                  'diffusion_mask'          :input_str_mask.squeeze(),
                  't_list'                  :t_list,
                  'diffuse_sidechains'      :preprocess_param['sidechain_input'],
                  'include_motif_sidechains':preprocess_param['motif_sidechain_input'],
                  'is_sm': is_sm}
        diffused_fullatoms, aa_masks, true_crds = diffuser.diffuse_pose(**kwargs)

        ############################################
        ########### New Self Conditioning ##########
        ############################################

        # JW noticed that the frames returned from the diffuser are not from a single noising trajectory
        # So we are going to take a denoising step from x_t+1 to get x_t and have their trajectories agree

        # Only want to do this process when we are actually using self conditioning training
        if preprocess_param['new_self_cond'] and t < 200: # Only can get t+1 if we are at t < 200

            tmp_x_t_plus1 = diffused_fullatoms[0]

            beta_schedule, _, alphabar_schedule = get_beta_schedule(
                                       T=diffusion_param['diff_T'],
                                       b0=diffusion_param['diff_b0'],
                                       bT=diffusion_param['diff_bT'],
                                       schedule_type=diffusion_param['diff_schedule_type'],
                                       inference=False)

            _, ca_deltas = get_next_ca(
                                       xt=tmp_x_t_plus1,
                                       px0=true_crds,
                                       t=t+1,
                                       diffusion_mask=input_str_mask.squeeze(),
                                       crd_scale=diffusion_param['diff_crd_scale'],
                                       beta_schedule=beta_schedule,
                                       alphabar_schedule=alphabar_schedule,
                                       noise_scale=1)

            # Noise scale ca hard coded for now. Maybe can eventually be piped down from inference configs? - NRB

            frames_next = get_next_frames(
                                       xt=tmp_x_t_plus1,
                                       px0=true_crds,
                                       t=t+1,
                                       diffuser=diffuser,
                                       so3_type=diffusion_param['diff_so3_type'],
                                       diffusion_mask=input_str_mask.squeeze(),
                                       noise_scale=1) # Noise scale frame hard coded for now - NRB

            frames_next = torch.from_numpy(frames_next) + ca_deltas[:,None,:]  # translate
            
            tmp_x_t = torch.zeros_like(tmp_x_t_plus1)
            tmp_x_t[:,:3] = frames_next
            
            if preprocess_param['motif_sidechain_input']:
                tmp_x_t[input_str_mask.squeeze(),:] = tmp_x_t_plus1[input_str_mask.squeeze()]
            
            diffused_fullatoms[1] = tmp_x_t
            
        ############################################
        ######### End New Self Conditioning ########
        ############################################

        if seq_diffuser is not None:
            seq_args = {
                        'seq'            : seq.squeeze(0),
                        'diffusion_mask' : input_seq_mask.squeeze(),
                        't_list'         : t_list
                       }
            diffused_seq, true_seq = seq_diffuser.diffuse_sequence(**seq_args)

            if seq_diffuser.continuous_seq():
                diffused_seq_bits = diffused_seq

                assert(diffused_seq.shape[-1] == 20) # Must be probabilities
                if predict_previous: raise NotImplementedError() # This involves changing the shape of true crds - NRB
            else:
                diffused_seq_bits = torch.nn.functional.one_hot(diffused_seq, num_classes=20).float()
         
        if predict_previous: 
            assert(diffused_fullatoms.shape[0] == 3)
        else: 
            assert(diffused_fullatoms.shape[0] == 2)

        # seq_mask - True-->revealed, False-->masked 
        seq_mask = torch.ones(2,L).to(dtype=bool) # all revealed [t,L]

        # JW - moved this here
        mask_msa = torch.stack([mask_msa,mask_msa], dim=0) # [n,I,N_long,L,25]

        if seq_diffuser is not None:
            seq_mask[:,~input_seq_mask.squeeze()] = False # All non-fixed positions are diffused in sequence diffusion
            
            # JW - moved mask_msa masking here
            mask_msa[:,:,:,~input_seq_mask.squeeze()] = False # don't score non-diffused positions
        else:

            if diffusion_param['aa_decode_steps'] == 0:
                print("Amino acids are not being decoded in this run")
                aa_masks[:,:] = False

            # mark False where aa_mask_raw is False -- assumes everybody is potentially diffused
            seq_mask[0,~aa_masks[0]] = False
            seq_mask[1,~aa_masks[1]] = False

            # reset to True any positions which aren't being diffused 
            seq_mask[:,input_seq_mask.squeeze()] = True

            ###  DJ new - make mutations in the decoded sequence 
            ### JW - changed this so the sampled mutations are the same for both timesteps
            #sampled_blosum = torch.stack([sample_blosum_mutations(seq.squeeze(0), p_blosum=diffusion_param['decode_corrupt_blosum'], p_uni=diffusion_param['decode_corrupt_uniform'], p_mask=0),\
            #                              sample_blosum_mutations(seq.squeeze(0), p_blosum=diffusion_param['decode_corrupt_blosum'], p_uni=diffusion_param['decode_corrupt_uniform'], p_mask=0)], dim=0) # [n,L]
            sampled_blosum = sample_blosum_mutations(seq.squeeze(0), p_blosum=diffusion_param['decode_corrupt_blosum'], p_uni=diffusion_param['decode_corrupt_uniform'], p_mask=0)[None].repeat(2,1) # [n, L]
            # find decoded residues and select them with 21% probability 
            decoded_non_motif = torch.ones_like(sampled_blosum).to(dtype=bool) # [n,L]

            # mark False where residues are masked via diffusion 
            decoded_non_motif[0,~aa_masks[0]] = False
            decoded_non_motif[1,~aa_masks[1]] = False

            decoded_non_motif[:,input_seq_mask.squeeze()] = False      # mark False where motif exists 
            
            # set (1-decode_mask_frac) proportion to False, keeping <decode_mask_frac> proportion still available 
            # JW - changed this slightly, for scoring purposes.
            # In a manner akin to RF/AF, we apply loss on sequence that has been mutated/corrupted, as well as to the same proportion
            # of sequence that has not been corrupted. 
            # Therefore, network must learn which residues to change, and which to keep fixed.
            # This should also make aa_cce more comparable between runs, if we vary decode_mask_frac
            
            # First, make the 'is_scored' mask, which scores twice the proportion of residues as 'decode_mask_frac'. True=scored
            p_score = 2*diffusion_param['decode_mask_frac'] 
            is_scored = torch.rand(decoded_non_motif.shape) < p_score # [n,L]

            # Now, make tmp mask, which is 50:50 corrupt vs not (so total proportion to be corrupted == 'decode_mask_frac'
            # This yields a mask where True == unchanged, False == Corrupted residue
            tmp_mask = torch.logical_and(is_scored, torch.rand(decoded_non_motif.shape) < 0.5)
            
            decoded_non_motif[0,tmp_mask[0]] = False 
            decoded_non_motif[1,tmp_mask[0]] = False #use same mask for both, which seems sensible for now.

            # Anything left as True, replace with blosum sample
            # These may be different lengths so cannot convert to a Tensor - NRB
            blosum_replacement = []

            blosum_replacement.append(sampled_blosum[0,decoded_non_motif[0]])
            blosum_replacement.append(sampled_blosum[1,decoded_non_motif[1]])

            onehot_blosum_rep = [torch.nn.functional.one_hot(i, num_classes=22).float() for i in blosum_replacement] # [n,dim_replace,22]

            # JW Move mask_msa masking here, and apply new masks (see above)
            # 1.) make all decoded residues False (so not scored scored)
            mask_msa[0,:,:,seq_mask[0]] = False
            mask_msa[1,:,:,seq_mask[1]] = False
            # 2.) Now, bring back any residues to score (both those corrupted by blosum, and the equivalent proportion of non-corrupted)
            is_scored[:,input_seq_mask.squeeze()] = False
            mask_msa[0,:,:,is_scored[0]] = True
            mask_msa[1,:,:,is_scored[1]] = True
            ### End DJ new
       
        xyz_t       = diffused_fullatoms[:2].unsqueeze(1) # [n,T,L,27,3]

        if predict_previous:
            true_crds = diffused_fullatoms[-1][None]

        # Scale str confidence wrt t 
        # multiplicitavely applied to a default conf mask of 1.0 everywhwere
        # TODO NRB make this work for sinusoidal embedding + sequence diffusion
        input_t1d_str_conf_mask = torch.stack([input_t1d_str_conf_mask,input_t1d_str_conf_mask], dim=0) # [n,L]

        #if model_param['d_time_emb'] > 0:
        # d_time_emb == 0 in paper args
        if False:
            if seq_diffuser is not None:
                raise NotImplementedError("Sinuisoidal timestep embedding isn't implemented for sequence diffusion yet, because sequence diffusion has both sequence & structure timestep")
            # sinusoidal embedding
            input_t1d_str_conf_mask[:,~input_str_mask.squeeze()] = 0
        else:
            # linear timestep
            input_t1d_str_conf_mask[0,~input_str_mask.squeeze()] = 1 - t_list[0]/diffuser.T
            input_t1d_str_conf_mask[1,~input_str_mask.squeeze()] = 1 - t_list[1]/diffuser.T
        

        # Scale seq confidence wrt t
        input_t1d_seq_conf_mask = torch.stack([input_t1d_seq_conf_mask,input_t1d_seq_conf_mask], dim=0) # [n,L]
        input_t1d_seq_conf_mask[0,~input_seq_mask.squeeze()] = 1 - t_list[0]/diffuser.T
        input_t1d_seq_conf_mask[1,~input_seq_mask.squeeze()] = 1 - t_list[1]/diffuser.T

    else:
        print('WARNING: Diffuser not being used in apply masks')

    ###########

    #seq_mask = input_seq_mask[0] # DJ - old, commenting out bc using seq mask from diffuser 
    seq = torch.stack([seq,seq], dim=0) # [n,I,L]
    if seq_diffuser is not None:
        raise Exception('not implemented')
        # alldim_diffused_seq = diffused_seq_bits[:,None,:,:] # [n,I,L,20]
        # zeros = torch.zeros(2,1,L,2)
        # seq   = torch.cat((alldim_diffused_seq, zeros), dim=-1) # [n,I,L,22]

    else:
        # This may not need to be changed for non-small-molecule
        # N_SEQ_PROT = 22
        # N_SEQ_AA = 80
        # N_CHAR = 80
        # MASK_TOKEN = rf2aa.chemical.MASK_INDEX
        '''
        msa_full:   NSEQ,NINDEL,NTERMINUS,
        msa_masked: NSEQ,NSEQ,NINDEL,NINDEL,NTERMINUS
        '''
        NINDEL = 1

        # raise Exception('needs to be adapted to aa')
        assert len(blosum_replacement[0]) == decoded_non_motif[0].sum() and len(blosum_replacement[1]) == decoded_non_motif[1].sum()
        seq = torch.nn.functional.one_hot(seq, num_classes=ChemData().NAATOKENS).float() # [n,I,L,22]
        seq[0,:,~seq_mask[0],:] = 0
        seq[1,:,~seq_mask[1],:] = 0 

        seq[0,:,~seq_mask[0],ChemData().MASKINDEX] = 1 # mask token categorical value
        seq[1,:,~seq_mask[1],ChemData().MASKINDEX] = 1 # mask token categorical value

        seq[0,:,decoded_non_motif[0],:22] = onehot_blosum_rep[0]
        seq[1,:,decoded_non_motif[1],:22] = onehot_blosum_rep[1] 
    
    ### msa_masked ###
    ################## 
    msa_masked = torch.stack([msa_masked,msa_masked], dim=0) # [n,I,N_short,L,48]
    if seq_diffuser is not None:
        raise Exception('not implemented')
        msa_masked[...,:20]   = diffused_seq_bits[:,None,None,:,:]
        msa_masked[...,22:42] = diffused_seq_bits[:,None,None,:,:]

        # These dimensions are gap and mask - NRB
        msa_masked[...,20:22] = 0
        msa_masked[...,42:44] = 0

        # insertion/deletion stuff 
        msa_masked[:,:,:,~input_seq_mask.squeeze(),44:46] = 0
        
    else:
        # Standard autoregressive masking
        msa_masked[0,:,:,~seq_mask[0],:ChemData().NAATOKENS-1] = 0
        msa_masked[1,:,:,~seq_mask[0],:ChemData().NAATOKENS-1] = 0

        msa_masked[0,:,:,~seq_mask[0],ChemData().MASKINDEX]  = 1 # set to mask token
        msa_masked[1,:,:,~seq_mask[1],ChemData().MASKINDEX]  = 1 # set to mask token

        # index 44/45 is insertion/deletion
        # index 43 is the masked token NOTE check this
        # index 42 is the unknown token 
        msa_masked[0,:,:,~seq_mask[0],ChemData().NAATOKENS:2*ChemData().NAATOKENS-1] = 0
        msa_masked[1,:,:,~seq_mask[1],ChemData().NAATOKENS:2*ChemData().NAATOKENS-1] = 0

        msa_masked[0,:,:,~seq_mask[0],ChemData().NAATOKENS+ChemData().MASKINDEX]    = 1
        msa_masked[1,:,:,~seq_mask[1],ChemData().NAATOKENS+ChemData().MASKINDEX]    = 1 

        # insertion/deletion stuff 
        msa_masked[0,:,:,~seq_mask[0],2*ChemData().NAATOKENS:2*ChemData().NAATOKENS+2*NINDEL] = 0
        msa_masked[1,:,:,~seq_mask[1],2*ChemData().NAATOKENS:2*ChemData().NAATOKENS+2*NINDEL] = 0

        # blosum mutations 
        msa_masked[0,:,:,decoded_non_motif[0],:] = 0
        msa_masked[1,:,:,decoded_non_motif[1],:] = 0

        msa_masked[0,:,:,decoded_non_motif[0],blosum_replacement[0]]  = 1
        msa_masked[1,:,:,decoded_non_motif[1],blosum_replacement[1]]  = 1

        msa_masked[0,:,:,decoded_non_motif[0],ChemData().NAATOKENS:2*ChemData().NAATOKENS] = 0                  
        msa_masked[1,:,:,decoded_non_motif[1],ChemData().NAATOKENS:2*ChemData().NAATOKENS] = 0                  

        msa_masked[0,:,:,decoded_non_motif[0],ChemData().NAATOKENS+blosum_replacement[0]] = 1
        msa_masked[1,:,:,decoded_non_motif[1],ChemData().NAATOKENS+blosum_replacement[1]] = 1

    ### msa_full ### 
    ################
    msa_full = torch.stack([msa_full,msa_full], dim=0) # [n,I,N_long,L,25]
    
    if seq_diffuser is not None:
        # These sequences will only go up to 20
        msa_full[...,:20]   = diffused_seq_bits[:,None,None,:,:]

        # These dimensions are gap and mask - NRB
        msa_full[...,20:22] = 0

        msa_full[:,:,:,~input_seq_mask.squeeze(),-3:]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 
        
    else:
        # Standard autoregressive masking
        msa_full[0,:,:,~seq_mask[0],:ChemData().NAATOKENS-1] = 0
        msa_full[1,:,:,~seq_mask[1],:ChemData().NAATOKENS-1] = 0

        msa_full[0,:,:,~seq_mask[0],ChemData().MASKINDEX]  = 1
        msa_full[1,:,:,~seq_mask[1],ChemData().MASKINDEX]  = 1

        msa_full[0,:,:,~seq_mask[0],ChemData().NAATOKENS:ChemData().NAATOKENS+NINDEL] = 0   
        msa_full[1,:,:,~seq_mask[1],ChemData().NAATOKENS:ChemData().NAATOKENS+NINDEL] = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

        # blosum mutations 
        msa_full[0,:,:,decoded_non_motif[0],:] = 0
        msa_full[1,:,:,decoded_non_motif[1],:] = 0

        msa_full[0,:,:,decoded_non_motif[0],blosum_replacement[0]]  = 1
        msa_full[1,:,:,decoded_non_motif[1],blosum_replacement[1]]  = 1


    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.stack([t1d,t1d], dim=0) # [n,I,L,22]
    # NOTE DJ, add the sinusoidal timestep embedding logic here
    if seq_diffuser is not None:
        t1d[...,:20] = diffused_seq_bits[:,None,:,:]
        
        t1d[...,20]  = 0 # No unknown characters in seq diffusion

        #JW moved this here
        t1d[:,:,:,21] = input_t1d_str_conf_mask[:,None,:]
        t1d[:,:,:,22] = input_t1d_seq_conf_mask[:,None,:]
    else:
        # TODO: adapt this for small molecules using t1d shift
        # Mask the diffused sequence

        # # ic(blosum_replacement.shape)
        # ic(blosum_replacement)
        # blosum_replacement = torch.cat(blosum_replacement)
        # ic(blosum_replacement.shape)
        # seq_cat_shifted = blosum_replacement.argmax(dim=-1)
        # ic(seq_cat_shifted)
        # seq_cat_shifted[seq_cat_shifted>=MASKINDEX] -= 1
        # # t1d_motif = torch.nn.functional.one_hot(seq_cat_shifted, num_classes=NAATOKENS-1)
        # # ic(t1d)
        # # t1d = t1d[None, None] # [L, NAATOKENS-1] --> [1,1,L, NAATOKENS-1]

        # ic(t1d.shape, t1d.argmax(dim=-1))

        t1d[0,:,~seq_mask[0],:20] = 0 
        t1d[1,:,~seq_mask[1],:20] = 0 

        t1d[0,:,~seq_mask[0],20]  = 1
        t1d[1,:,~seq_mask[1],20]  = 1 # unknown

        # ONLY FOR aa_decoding
        # Add the motif sequence
        t1d_before = torch.clone(t1d)
        t1d[0,:,decoded_non_motif[0],:]  = 0
        t1d[1,:,decoded_non_motif[1],:]  = 0

        t1d[0,:,decoded_non_motif[0],blosum_replacement[0]] = 1
        t1d[1,:,decoded_non_motif[1],blosum_replacement[1]] = 1
        # Sanity check that t1d is not altered when not doing aa decoding.
        if diffusion_param['aa_decode_steps'] == 0:
            rf2aa.tensor_util.assert_equal(t1d, t1d_before)

        t1d[:,:,:,-1] = input_t1d_str_conf_mask[:,None,:]
    
    # mask sidechains in the diffused region if preprocess_param['sidechain_input'] is False
    if preprocess_param['sidechain_input'] is False:
        xyz_t[:,:,~input_str_mask.squeeze(),3:,:] = float('nan')
    else:
        #only mask the sequence-masked sidechains
        xyz_t[:,:,~seq_mask,3:,:] = float('nan') # don't know sidechain information for masked seq
    if preprocess_param['motif_sidechain_input'] is False:
        xyz_t[:,:,input_str_mask.squeeze(),3:,:] = float('nan')
    # Structure masking
    # str_mask = input_str_mask[0]
    # xyz_t[:,:,~str_mask,:,:] = float('nan') # NOTE: not using this because diffusion is effectively the mask 
    return seq, msa_masked, msa_full, xyz_t, t1d, mask_msa, t_list[:2], true_crds
