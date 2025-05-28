import torch
import sys

from rf_diffusion.inference.mid_run_modifiers import PartiallyDiffusePx0Toxt, ReplaceXtWithPx0, RemoveGuideposts, DiffuseChains, ReinitializeWithCOMOri


def assert_t_setups(conf, t_setups):
    '''
    Give option flags a chance to assert that their TSetup is present

    Args:
        conf (omegaconf): The config
        t_setups (list[str]): The names of the TSetups that are present
    '''

    if conf.inference.custom_t_range:
        assert 'CustomTRangeTSetup' in t_setups, 'inference.custom_t_range requires CustomTRangeTSetup'

    if conf.inference.fast_partial_trajectories:
        assert 'FastPartialTrajectoriesTSetup' in t_setups, 'inference.fast_partial_trajectories requires FastPartialTrajectoriesTSetup'

    if conf.inference.start_str_self_cond_at_t:
        assert 'StartSelfCondTSetup' in t_setups, 'inference.start_str_self_cond_at_t requires StartSelfCondTSetup'

    if conf.inference.write_extra_ts:
        assert 'WriteExtraTsTSetup' in t_setups, 'inference.write_extra_ts requires WriteExtraTsTSetup'

    if conf.inference.ORI_guess:
        assert 'ORIGuessTSetup' in t_setups, 'inference.ORI_guess requires ORIGuessTSetup'


def init_default_self_cond_from_ts(conf, ts):
    '''
    Use inference.str_self_cond to initialize the self_cond vector

    The first step never uses self conditioning
    '''

    self_cond = torch.full((len(ts),), conf.inference.str_self_cond, dtype=bool)
    self_cond[0] = False # can't self condition the first step
    return self_cond


def setup_t_arrays(conf, t_step_input):
    '''
    Prepare the steps that the trajectory will take

    Args:
        conf (omegaconf): The config
        t_step_input (int): The starting t

    Returns:
        ts (tensor[int]): The t-steps
        n_steps (tensor[int]): The number of diffuser backward steps to take (used if skipping ts)
        self_cond (tensor[bool]): Whether or not to self condition from the previous px0 at this t
        final_it (int): The it that represents the final step
        addtl_write_its (list[int]): Which outputs to also write
    '''

    ts = torch.arange(int(t_step_input), conf.inference.final_step-1, -1, dtype=int)
    n_steps = torch.ones(len(ts), dtype=int)
    self_cond = init_default_self_cond_from_ts(conf, ts)
    final_it = len(ts)-1
    addtl_write_its = []
    mid_run_modifiers = [[] for t in ts]

    TSetups = conf.inference.t_setups.names
    assert_t_setups(conf, TSetups)

    args = dict(
        ts=ts,
        n_steps=n_steps,
        self_cond=self_cond,
        final_it=final_it,
        addtl_write_its=addtl_write_its,
        mid_run_modifiers=mid_run_modifiers
        )

    thismodule = sys.modules[__name__]
    for TSetup_name in TSetups:

        TSetup = getattr(thismodule, TSetup_name)()
        args = TSetup(conf=conf, **args)

        N = len(args['ts'])
        assert len(args['n_steps']) == N, TSetup_name
        assert len(args['self_cond']) == N, TSetup_name
        assert len(args['mid_run_modifiers']) == N, TSetup_name

    if conf.inference.t_setups.debug:
        print("inference.t_setups.debug: Printing out the trajectory information")
        for it in range(len(args['ts'])):
            print(f"it:{it:3d} t:{args['ts'][it]:3d} n_steps:{args['n_steps'][it]:3d} self_cond:{args['self_cond'][it]:2}")
            for mrm in args['mid_run_modifiers'][it]:
                print('   ', type(mrm).__name__)
            if it == args['final_it']:
                print('    ==== Final t ====')
            for write_it, suffix in args['addtl_write_its']:
                if write_it == it:
                    print(f'   Write as: {suffix}')

    return (
        args['ts'],
        args['n_steps'],
        args['self_cond'],
        args['final_it'],
        args['addtl_write_its'],
        args['mid_run_modifiers']
        )


class TSetup:
    '''
    Base class for TSetup
    '''

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        return kwargs


class CustomTRangeTSetup(TSetup):
    '''
    Specify custom t-ranges. This one really should come first

    Only calculate pX0 at t steps given in this list. Negative numbers can be used to partially diffuse to new ts [50,25,-40,25,1]

    Parser for inference.custom_t_range
    Example: [50,49,48,-40,30,-40,30,20,10,1]

    Positive values mean diffuse like normal
    Positive values with gaps will reusing the same pX0
    Negative values mean partially diffuse pX0 to the t step

    Flags:
        inference.custom_t_range
    '''

    def __call__(self, conf, ts, n_steps, self_cond, final_it, addtl_write_its, mid_run_modifiers, **kwargs):

        if conf.inference.custom_t_range is not None:
            assert conf.inference.model_runner == 'NRBStyleSelfCond', 'Only the NRBStyleSelfCond model_runner supports inference.custom_t_range!'

            assert len(addtl_write_its) == 0, 'CustomTRangeTSetup really needs to come first'
            assert (torch.tensor([len(x) for x in mid_run_modifiers]) == 0).all(), 'CustomTRangeTSetup really needs to come first'

            ts = []
            n_steps = []
            mid_run_modifiers = []
            self_cond = []
            last_t = None
            for t in conf.inference.custom_t_range:
                assert abs(t) >= conf.inference.final_step, ("inference.custom_t_range can't have values smaller than "
                                                                        f"inference.final_step: {abs(t)} < {conf.inference.final_step}")
                assert abs(t) <= conf.diffuser.T, ("inference.custom_t_range can't have values larger than "
                                                                        f"diffuser.T: {abs(t)} < {conf.diffuser.T}")

                if last_t is None:
                    # First step
                    assert t > 0, f'inference.custom_t_range: If you want to start off with partial diffusion you should instead specify {diffuser.partial_T}'
                    ts.append(t)
                    n_steps.append(1)
                    mid_run_modifiers.append([])
                else:
                    if t < 0:
                        # Diffuse to a new t
                        ts.append(-t)
                        n_steps.append(1)
                        mid_run_modifiers[-1].append(PartiallyDiffusePx0Toxt(-t)) # we partially diffuse after the previous step to this step
                        mid_run_modifiers.append([])
                    else:
                        abs_last_t = abs(last_t)
                        if abs_last_t - t == 1:
                            # Normal single step
                            ts.append(t)
                            n_steps.append(1)
                            mid_run_modifiers.append([])
                        else:
                            # Forward many steps without calling rf2 again
                            assert t < abs_last_t
                            ts.append(t)
                            n_steps.append(abs_last_t - t)
                            mid_run_modifiers.append([])

                last_t = t


            ts = torch.tensor(ts)
            n_steps = torch.tensor(n_steps)
            self_cond = init_default_self_cond_from_ts(conf, ts)
            final_it = len(ts)-1
            addtl_write_its = []

            assert ts[-1] == conf.inference.final_step, (f'The final element of inference.custom_t_range must be inference.final_step. {ts[-1]} != {conf.inference.final_step}')


        return dict(ts=ts, n_steps=n_steps, self_cond=self_cond, final_it=final_it, addtl_write_its=addtl_write_its, mid_run_modifiers=mid_run_modifiers, **kwargs)




def lookup_fpt_source(ts_in, fpt_string):
    '''
    Figure out the i_t for this specific t in the fpt_string

    Note that we pick the latest time this t occurs

    Args:
        ts_in (tensor[int]): The t trajectory without the fpt ts
        fpt_string (str): The fpt string that we're looking up

    Returns:
        i_t (int): The i_t that is the source
    '''

    source_t = int(fpt_string.split('-')[1])
    wh = torch.where(ts_in == source_t)[0]
    if len(wh) > 1:
        print('Warning: fast_partial_trajectory {fpt_string} specified a source t ({t}) that is used multiple times in the run. Using the last one.')
    assert len(wh) > 0, f'Source t not found ({t}) from fast_partial_trajectory: {fpt_string}'
    return wh[-1]



class FastPartialTrajectoriesTSetup(TSetup):
    '''
    Modify the run_inference loop variables to produce extra single-step partial diffusion trajectories

    Takes List of lists like [[1,20,3],[5,25,3]]. 
        Specifies that original t-steps (here 1 and 5) should be partially diffused to 20 then 3 (or 25 then 3) and output

    Flags:
        inference.fast_partial_trajectories
        inference.fpt_drop_guideposts
        inference.fpt_diffuse_chains
    '''

    def __call__(self, conf, ts, n_steps, self_cond, final_it, addtl_write_its, mid_run_modifiers, **kwargs):

        if conf.inference.fast_partial_trajectories is not None:

            # Figure out all of the names as well as intermediate steps we need to calculate
            traj_strings = []
            is_output = []
            for traj_list in conf.inference.fast_partial_trajectories:
                assert len(traj_list) > 1, f'Invalid fast_partial_trajectory specification. [source_t,t(,t...)]: {traj_list}'

                # The first number is the source
                source = traj_list[0]
                string_builder = f'fpt-{source}'

                # For each additional part, make the name of the output and check if that name is already in stuff we were going to make anyways
                for ipart, part in enumerate(traj_list[1:]):
                    string_builder += f'-{part}'
                    if string_builder not in traj_strings:
                        traj_strings.append(string_builder)
                        is_output.append( ipart == len(traj_list[1:])-1 )


            N_old = len(ts)
            N_new = len(traj_strings)

            # Modify the diffusion setup to do a better job with partial diffusion
            if conf.inference.fpt_drop_guideposts:
                mid_run_modifiers[-1].append(RemoveGuideposts())
            if conf.inference.fpt_diffuse_chains is not None:
                value = conf.inference.fpt_diffuse_chains
                try:
                    to_give = [int(value)]
                except ValueError:
                    assert isinstance(value, str), f'inference.fpt_diffuse_chains should be a string or an int {value}'
                    if value == 'all':
                        to_give = 'all'
                    else:
                        to_give = [int(x) for x in value.split(',')]
                mid_run_modifiers[-1].append(DiffuseChains(to_give))

            # Generate additional steps at the end to do partial diffusions on
            ts_in = ts.clone()
            # Cat zeros onto all of the vectors so we can explicitly define them below
            ts = torch.cat((ts, torch.zeros(N_new, dtype=int)))
            n_steps = torch.cat((n_steps, torch.zeros(N_new, dtype=n_steps.dtype)))
            self_cond = torch.cat((self_cond, torch.zeros(N_new, dtype=self_cond.dtype)))

            for itraj, (traj_string, output) in enumerate(zip(traj_strings, is_output)):

                sp = traj_string.split('-')

                assert len(sp) >= 3
                if len(sp) == 3:
                    prev_it = lookup_fpt_source(ts_in, traj_string)
                else:
                    prev_string = '-'.join(sp[:-1])
                    prev_it = N_old + traj_strings.index(prev_string)

                t = int(sp[-1]) 

                ts[N_old+itraj] = t # the t is specified directly by the user
                n_steps[N_old+itraj] = 1 # we're only taking 1 step (this doesn't matter anyways, xt isn't even used at this point)

                # We need to load the correct px0 then partially diffuse it
                mid_run_modifiers[-1].append(ReplaceXtWithPx0(prev_it))
                mid_run_modifiers[-1].append(PartiallyDiffusePx0Toxt(t))
                mid_run_modifiers.append([])
                self_cond[N_old+itraj] = False # This doesn't matter a whole lot

                if output:
                    addtl_write_its.append( (itraj + N_old, '_' + traj_string))


        return dict(ts=ts, n_steps=n_steps, self_cond=self_cond, final_it=final_it, addtl_write_its=addtl_write_its,
                        mid_run_modifiers=mid_run_modifiers, **kwargs)


class StartSelfCondTSetup(TSetup):
    '''
    Sets up self conditioning to be False at first and then True later

    flags:
        inference.start_str_self_cond_at_t
    '''

    def __call__(self, conf, ts, self_cond, **kwargs):

        if conf.inference.start_str_self_cond_at_t is not None:

            wh = torch.where(ts == conf.inference.start_str_self_cond_at_t)[0]
            assert len(wh) > 0, f'inference.start_str_self_cond_at_t value {conf.inference.start_str_self_cond_at_t} not found in t-list'

            if len(wh) > 1:
                print(f"Warning: inference.start_str_self_cond_at_t value {conf.inference.start_str_self_cond_at_t} occurs multiple times in t-list."
                            " Using first instance")

            first_self_cond = wh[0]
            assert first_self_cond > 0, "You can't start self conditioning at the very first step!"

            self_cond[:first_self_cond] = False
            self_cond[first_self_cond:] = True

        return dict(ts=ts, self_cond=self_cond, **kwargs)


class WriteExtraTsTSetup(TSetup):
    '''
    Write out additional timesteps from the initial diffusion run

    Flags:
        inference.write_extra_ts
    '''

    def __call__(self, conf, ts, addtl_write_its, **kwargs):

        for extra_t in conf.inference.write_extra_ts:

            wh = torch.where(ts == extra_t)[0]
            assert len(wh) > 0, f'inference.write_extra_ts value {extra_t} not found in t-list'

            if len(wh) > 1:
                print(f"Warning: inference.write_extra_ts value {extra_t} occurs multiple times in t-list."
                            " Using first instance")

            it = wh[0]
            addtl_write_its.append((it, f'_t{extra_t}'))

        return dict(ts=ts, addtl_write_its=addtl_write_its, **kwargs)


class ORIGuessTSetup(TSetup):
    '''
    Peforms the first step of diffusion and notes the location of the CoM of the diffused xT
    Then reinitializes the diffuser with that CoM as the ORI and begins again

    flags:
        inference.ORI_guess
    '''

    def __call__(self, conf, ts, n_steps, self_cond, final_it, addtl_write_its, mid_run_modifiers, **kwargs):

        if conf.inference.ORI_guess:

            for i in range(int(conf.inference.ORI_guess)):

                # Add an additional timestep to the front
                ts = torch.cat((torch.full((1,), ts[0], dtype=ts.dtype), ts))
                n_steps = torch.cat((torch.full((1,), 1, dtype=n_steps.dtype), n_steps)) # Add a 1 but this won't get used
                self_cond = torch.cat((torch.full((1,), False, dtype=self_cond.dtype), self_cond)) # We can't self condition this
                final_it += 1 # final_it gets bumped up by 1
                addtl_write_its = [(x+1,y) for (x,y) in addtl_write_its] # Increase the indexes of the additional writes

                # Add our magic MidRunModifier to the first frame
                mid_run_modifiers = [[ReinitializeWithCOMOri()]] + mid_run_modifiers


        return dict(ts=ts, n_steps=n_steps, self_cond=self_cond, final_it=final_it, addtl_write_its=addtl_write_its,
                mid_run_modifiers=mid_run_modifiers, **kwargs)



