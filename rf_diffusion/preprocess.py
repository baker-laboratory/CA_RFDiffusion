import torch 
from rf_diffusion.aa_model import RFI
from rf_diffusion.chemical import ChemicalData as ChemData

def add_motif_template(rfi: RFI, t2d_motif: torch.Tensor, xyz_t_motif: torch.Tensor, masks_1d: dict) -> RFI:
    """
    Adds on a third template to the RoseTTAFold input features containing
    the motif structure.

    Parameters:
        rfi (aa_model.RFI): RoseTTAFold input features

        motif_template (dict): dictionary containing precomputed components of the motif structure.

    Returns:
        rfi (aa_model.RFI): RoseTTAFold input features with the motif template added.
    """
    # unpack
    t1d     = rfi.t1d
    t2d     = rfi.t2d
    mask_t  = rfi.mask_t
    alpha_t = rfi.alpha_t
    xyz_t   = rfi.xyz_t

    is_motif = masks_1d['is_templated_motif']
    is_motif_2d = masks_1d['is_motif_2d']

    """
    NOTE: Need to slice in noised crds into motif template xyz for two reasons:
    1. Although it is seemingly arbitrary, this enables us to match the golden
       xyz_t_motif from OG CA RFdiffusion branch.
    2. Need to do it HERE specifically because ComputeTemplateMotif Transform is
       called upstream of DistilledDataset.diffuse transform, and thus doesn't
       have access to the noised crds when motif template is computed.
    """
    xyz_t_motif[~is_motif] = xyz_t[0,0][~is_motif] # xyz_t_motif is (L,3)

    assert not (torch.isnan(t2d_motif).any()), 'no NaNs in motif t2d'
    assert not (torch.isnan(xyz_t_motif).any()), 'no NaNs in motif xyz_t'

    # control shapes
    L = t1d.shape[2]
    N_prev = t1d.shape[1]
    N_new  = N_prev + 1

    assert all(t.shape[2] == L      for t in [t1d, t2d, mask_t, alpha_t, xyz_t])
    assert all(t.shape[1] == N_prev for t in [t1d, t2d, mask_t, alpha_t, xyz_t])

    ### alpha_t ###
    ###############
    # grab the first on and duplicate it N_new times
    alphas = alpha_t[0,0] # (L, n_alpha)
    alpha_t_out = torch.cat([alphas[None]]*N_new, dim=0)[None] # (1, N_new, L, n_alpha)

    ### t2d ###
    ###########
    t2d_out = torch.zeros((1, N_new, L, L, t2d.shape[-1])).to(t2d.device, t2d.dtype)
    t2d_out[0,:-1] = t2d[0] # copy the first N_prev templates into the new t2d

    # add the motif template
    t2d_out[0,-1] = t2d_motif[0]

    # add in the indicator features--denotes pairs of residues constrained w.r.t each other
    blank = torch.ones(L,L)*-1 # (L, L)
    # first N_prev templates will have -1 in this channel
    cattable_is_motif_2d = torch.stack([blank]*N_prev + [is_motif_2d.int()], dim=-1).permute(2,0,1) # (L,L,N_new) --> (N_new, L, L)
    cattable_is_motif_2d = cattable_is_motif_2d[None,...,None] # (1, N_new, L, L, 1)

    # tack this last feature onto the end of t2d
    t2d_out = torch.cat([t2d_out, cattable_is_motif_2d], dim=-1)

    ### t1d ###
    ###########
    t1d_out = torch.zeros((1, N_new, L, t1d.shape[-1])).to(t1d.device, t1d.dtype)
    t1d_out[0,:] = t1d[0,0] # copy the first template across new t1d

    # reset the -1 feature for second (Xt) template, as was done in prepro
    t1d_out[0,1,:,ChemData().NAATOKENS-1] = -1
    # reset also for the last (motif) template
    t1d_out[0,-1,:,ChemData().NAATOKENS-1] = -1

    # feature for 3rd template - is it motif or not?
    cattable_is_motif = torch.tile(is_motif[None,None,:,None], (1, N_new, 1, 1)) # (1, N_new, L, 1)
    cattable_is_motif = cattable_is_motif.to(device=t1d.device, dtype=t1d.dtype)
    cattable_is_motif[:,:-1,...] = -1 #first N_prev templates get -1 for the motif feature
    t1d_out = torch.cat([t1d_out, cattable_is_motif], dim=-1)

    ### mask_t ###
    ##############
    mask_t_out = torch.ones(1, N_new, L, L).bool().to(mask_t.device)
    mask_t_out[0,-1] = is_motif_2d

    ### xyz_t ###
    #############
    xyz_t_out = torch.zeros((1, N_new, L, 3)).to(xyz_t.device, xyz_t.dtype)
    xyz_t_out[0,:-1] = xyz_t[0] # copy the first N_prev templates into the new xyz_t
    xyz_t_out[0,-1]  = xyz_t_motif


    rfi.t1d = t1d_out
    rfi.t2d = t2d_out
    rfi.mask_t = mask_t_out
    rfi.alpha_t = alpha_t_out
    rfi.xyz_t = xyz_t_out

    return rfi

def wrap_featurize( indep_t,
                    t,
                    is_diffused,
                    model_adaptor,
                    masks_1d,
                    template_t2d:torch.tensor=None,
                    template_xyz:torch.tensor=None
                   ):
    """
    Wrapper to handle extra tXd, prepro, and adding extra templates.

    Args:
        template_t2d: templated motif, as 6D disto/anglo input 
        template_xyz: templated motif, ca coordinates 
    """

    # create RosettaFold input features
    rfi = model_adaptor.prepro(indep_t, t, is_diffused)

    # add the templated motif information/features
    if template_t2d is not None:
        assert template_xyz is not None, "If using 2D template, need xyz for template."
        # Have to change the frames before computing the template
        rfi = add_motif_template(rfi, template_t2d, template_xyz, masks_1d)
        
        # Make sure motif isn't frozen
        rfi.is_motif = torch.zeros_like(rfi.is_motif)

    return rfi
