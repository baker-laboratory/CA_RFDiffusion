import torch

from rf_diffusion import aa_model
from rf_diffusion.conditions.util import pop_conditions_dict



def apply_mid_run_modifiers(mrms,
    i_des,
    # ^^ Not in the output
    sampler,
    indep,
    contig_map,
    rfo,
    px0_xyz_stack,
    denoised_xyz_stack,
    seq_stack,
    final_it
    ):
    '''
    Potentially modify just about anything except conf. (The rule is that changes can't persist between designs)
        You probably shouldn't modify ts either. That's better done in TSetup

    This is the place where you can do stuff that other people are afraid to let you merge and you can just claim they'll never have to deal
        with your change as long as your MRM isn't present in their run

    Args:
        mrms (list[MidRunModifier]): The MidRunModifiers for this time-step
        ... Exactly arguments from run_inference

    Returns:
        ... Exactly arguments from run_inference
        stop (bool): If you want to stop the run, return True here. You should probably modify final_it
    '''

    args = dict(
        i_des=i_des,
        sampler=sampler,
        indep=indep,
        contig_map=contig_map,
        rfo=rfo,
        px0_xyz_stack=px0_xyz_stack,
        denoised_xyz_stack=denoised_xyz_stack,
        seq_stack=seq_stack,
        final_it=final_it,
        # VV Additional args not in input
        stop=False,
        )

    for mrm in mrms:
        args = mrm(**args)

    return (
        args['sampler'],
        args['indep'],
        args['contig_map'],
        args['rfo'],
        args['px0_xyz_stack'],
        args['denoised_xyz_stack'],
        args['seq_stack'],
        args['final_it'],
        args['stop'],
        )



class MidRunModifier:
    '''
    A class to modify virtually anything during a diffusion trajectory

    The only rule is that you're only allowed to modify stuff that will not persist between runs
        (So don't modify conf or anything that's only loaded once (like the model))
    '''

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        return kwargs




class PartiallyDiffusePx0Toxt(MidRunModifier):
    '''
    Partially diffuse the current px0 to xt at the given t
    '''

    def __init__(self, t):
        '''
        Args:
            t (int): The t to partially diffuse to
        '''
        super().__init__()
        self.t = t


    def __call__(self, indep, sampler, rfo, **kwargs):

        indep.xyz[:,:3] = rfo.xyz[-1, 0, :]
        indep.xyz = indep.xyz.to('cpu')
        indep, _ = aa_model.diffuse(sampler._conf, sampler.diffuser, indep, sampler.is_diffused, int(self.t))

        return dict(indep=indep, sampler=sampler, rfo=rfo, **kwargs)


class ReplaceXtWithPx0(MidRunModifier):
    '''
    Replace the current diffused coordinates (xt) with the Px0 from a different i_t
    '''

    def __init__(self, it):
        '''
        Args:
            it (int): The it to grab the px0 from
        '''
        super().__init__()
        self.it = it

    def __call__(self, indep, px0_xyz_stack, **kwargs):
        indep.xyz = px0_xyz_stack[self.it]

        return dict(indep=indep, px0_xyz_stack=px0_xyz_stack, **kwargs)


class RemoveGuideposts(MidRunModifier):
    '''
    Drop all guidepost residues. Make it look like there were never guideposts
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, indep, sampler, px0_xyz_stack, denoised_xyz_stack, seq_stack, rfo, contig_map, **kwargs):

        # Drop guideposts from literally everywhere
        was_gp = indep.is_gp.clone()
        indep, _ = aa_model.slice_indep(indep, ~was_gp)
        sampler.indep_orig, _ = aa_model.slice_indep(sampler.indep_orig, ~was_gp)
        px0_xyz_stack = [x[~was_gp] for x in px0_xyz_stack]
        denoised_xyz_stack = [x[~was_gp] for x in denoised_xyz_stack]
        seq_stack = [x[~was_gp] for x in seq_stack]
        sampler.is_diffused = sampler.is_diffused[~was_gp]
        rfo.xyz = rfo.xyz[:,:,~was_gp]
        pop_conditions_dict(sampler.conditions_dict, ~was_gp)
        sampler.atomizer.residue_to_atomize = sampler.atomizer.residue_to_atomize[~was_gp]
        sampler.atomizer.deatomized_state = [sampler.atomizer.deatomized_state[i] for i in torch.where(~was_gp)[0]]
        contig_map.gp_to_ptn_idx0 = {}

        return dict(indep=indep, sampler=sampler, px0_xyz_stack=px0_xyz_stack, denoised_xyz_stack=denoised_xyz_stack,
                seq_stack=seq_stack, rfo=rfo, contig_map=contig_map, **kwargs)



class DiffuseChains(MidRunModifier):
    '''
    Used with fast_partial_trajectories. State that the following chains are now fully diffused
    '''

    def __init__(self, diffused_chains):
        '''
        diffused_chains (list[int] | 'all' | None): A list of chains that will be diffused after this. 'all' for all
        '''
        super().__init__()
        self.diffused_chains = diffused_chains

    def __call__(self, indep, sampler, **kwargs):

        operate_mask = torch.zeros(indep.length(), dtype=bool)

        if self.diffused_chains is None:
            pass
        elif isinstance(self.diffused_chains, str) and self.diffused_chains == 'all':
            operate_mask[:] = True
        else:
            if len(self.diffused_chains) > 0:
                assert torch.tensor([x == int(x) for x in self.diffused_chains]).all()

                for ichain, chain_mask in enumerate(indep.chain_masks()):
                    if ichain in self.diffused_chains:
                        chain_mask = torch.tensor(chain_mask)
                        operate_mask[chain_mask] = True

        operate_mask[indep.is_gp] = False # don't mess with guideposts. Add a new flag if you want this to work

        sampler.is_diffused[operate_mask] = True

        return dict(indep=indep, sampler=sampler, **kwargs)



class ReinitializeWithCOMOri(MidRunModifier):
    '''
    Used with ORI_guess. Reinitialize the sampler using the current com of is_diffused as the new origin
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, indep, sampler, px0_xyz_stack, contig_map, i_des, **kwargs):

        px0 = px0_xyz_stack[-1]
        px0_com = px0[sampler.is_diffused,1,:].mean(axis=0)

        if 'origin' in sampler.extra_transform_kwargs and sampler.extra_transform_kwargs['origin'] is not None:
            new_com = px0_com + sampler.extra_transform_kwargs['origin']
        else:
            print("Warning: ReinitializeWithCOMOri: Origin wasn't stored in the transforms. This may not work correctly.")
            new_com = px0_com
        print("ReinitializeWithCOMOri: Found", px0_com, "in centered indep. New ORI will be:", new_com)

        indep, contig_map, _, _ = sampler.sample_init(i_des, extra=dict(origin_override=new_com))

        return dict(indep=indep, sampler=sampler, px0_xyz_stack=px0_xyz_stack, contig_map=contig_map, i_des=i_des, **kwargs)




