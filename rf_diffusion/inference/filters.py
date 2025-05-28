import torch
import sys


class FilterFailedException(Exception):
    '''
    An exception to throw to indicate that a filter has failed

    Set n_steps_taken for accurate job accounting
    '''
    def __init__(self, n_steps_taken, *args, **kwargs):
        '''
        Args:
            n_steps_taken (int): How many steps were taken in this trajectory before the fail?
        '''
        super().__init__(*args, **kwargs)
        self.n_steps_taken = n_steps_taken


def init_filters(conf):
    '''
    Initialize the filters for this inference run

    The filter.names have 2 formats:
        - FilterClass                  -- Use this if you only have 1 instance of each FilterClass
        - UserDefinedName:FilterClass  -- Use this to have multiple instances of each FilterClass

    The filter.configs then use either the FilterClass or the UserDefinedName if it exists

    Args:
        conf (OmegaConf): The main config

    Returns:
        filters (list[FilterBase]): The list of filters for this run
    '''
    thismodule = sys.modules[__name__]

    filters = []
    for name in conf.filters.names:
        if ':' in name:
            # They have renamed this filter
            filter_name, class_name = name.split(':')
        else:
            # They are just using the filter class as the name
            filter_name = class_name = name

        assert filter_name in conf.filters.configs, f"You didn't set filters.config.{filter_name} for your filter: {name}"
        filters.append(getattr(thismodule, class_name)(name=filter_name, **getattr(conf.filters.configs, filter_name)))

    return filters

def do_filtering(filters, indep, t, i_t, px0, scores, **kwargs):
    '''
    To be called after each denoising step

    Goes through each filter invoking the ones that want to be called and potentially raising a FilterFailed exception

    Args:
        filters (list[FilterBase]): The list of filters to use
        indep (Indep): The xT indep
        t (int): The current t-step
        i_t (int): The index of this t for step-counting purposes
        px0 (torch.Tensor[float]): The xyz of the px0 prediction
        scores (dict[str,?]): The current scores for this run
        kwargs (dict[str,?]): Additional kwargs that the filters want

    Throws:
        FilterFailed Exception
    '''

    if len(filters) == 0:
        return

    # Add indep and t to the kwargs
    kwargs['indep'] = indep
    kwargs['t'] = t

    # Prepare an indep using px0 for filters that want it
    px0_indep = indep.clone()
    px0_indep.xyz = px0
    kwargs['px0_indep'] = px0_indep

    # Any single filter can kill the run. Make sure they are all passing
    passing = True
    for filt in filters:
        local_passing, local_scores = filt.do_filter(**kwargs)
        passing = passing and local_passing
        scores.update(local_scores)

    if not passing:
        raise FilterFailedException(i_t + 1)



class FilterBase:
    '''
    Base class for all filters

    Child classes are expected to implement inner_do_filter()
    '''

    def __init__(self, t=None, suffix=None, prefix=None, verbose=False, name=None):
        '''
        Initialize the FilterBase

        Arguments to this function are specified via yaml

        Args:
            t (str): A comma separated list of which t-steps to perform this filter at
            suffix (str): A string suffix to add to the returned score terms
            prefix (str): A string prefix to add to the returned score terms
            verbose (str): Should this filter output anything to the terminal when it passes?
            name (str): The name of this filter. Defaults to class name
        '''
        self.suffix = suffix or ''
        self.prefix = prefix or ''
        self.verbose = verbose
        if name is None:
            name = self.__class__.__name__
        self.name = name

        # Determine which t we will activate at
        self.active_t = set()
        try:
            t = int(t)              # If you only specify a single number it can get parsed as an int
            self.active_t.add(t)
        except ValueError as _:
            if t is not None:
                for inner_t in t.split(','): # But if that fails we switch to string splitting
                    inner_t = int(inner_t)
                    self.active_t.add(inner_t)

    def do_filter(self, t, **kwargs):
        '''
        The function that is called by do_filtering() to invoke this filter

        Args:
            t (int): The current t-step
            kwargs (dict): See do_filtering()

        Returns:
            passing (bool): Whether or not this filter passed. False indicates the run should terminate
            final_scores (dict[str,?]): The values that this filter wishes to report
        '''

        assert hasattr(self, 'active_t'), 'You forgot to call super().__init__(**kwargs)'

        # int() to drop tensor for set lookup
        t = int(t)

        # This t is not in our set so do nothing
        if t not in self.active_t:
            return True, {}

        # Perform the actual filtering
        passing, scores = self.inner_do_filter(t=t, **kwargs)

        # Print the filtering result
        if self.verbose or not passing:
            print(f'Filter: {self.name} at t:{t} {"passes" if passing else "fails"} with: {scores}')

        # Transform the score terms to their final keys
        final_scores = {}
        for key, value in scores.items():
            final_scores[f'{self.prefix}t-{t}_{key}{self.suffix}'] = value

        return passing, final_scores


    def inner_do_filter(self, **kwargs):
        assert False, 'FilterBase.inner_do_filter() was called! You should overwrite this!'



class ChainBreak(FilterBase):
    '''
    A filter that looks for the presence of a chain-break for guidepost purposes

    Specifically, it reports the largest C->N distance
    '''

    def __init__(self, C_N_dist_limit=None, monitor_chains=None, use_px0=True, **kwargs):
        '''
        Args:
            C_N_dist_limit (float or None): The distance (in angstrom) of any pair of N->C beyond which this filter will fail. 1.7 could be reasonable
            monitor_chains (str or None): Either None (monitor all chains) or a comma separated string (starting at 0) of which chains to look at ('0,1')
            use_px0 (bool): Look at px0 instead of xT to inspect for chainbreaks (default True)
        '''
        super().__init__(**kwargs)

        # If they didn't specify a C_N_dist_limit just set it to a very high value
        self.C_N_dist_limit = C_N_dist_limit if C_N_dist_limit is not None else 9e9
        self.use_px0 = use_px0

        if monitor_chains is None:
            self.monitor_chains = None
        else:
            monitor = set()
            try:
                monitor.add(int(monitor_chains))          # Catch the case where a single value got interpretted as an int
            except ValueError as _:
                for inner_chain in monitor_chains.split(','):  # Else split on commas
                    monitor.add(int(inner_chain))

            self.monitor_chains = monitor

    def inner_do_filter(self, indep, px0_indep, **kwargs):
        '''
        Look for the largets C->N gap
        '''

        if self.use_px0:
            indep = px0_indep

        scores = {}
        scores['max_chain_gap'] = 0

        # Loop through all chains
        for ichain, chain_mask in enumerate(indep.chain_masks()):
            chain_mask = torch.tensor(chain_mask)
            chain_mask[indep.is_gp] = False

            # Make sure we're supposed to be looking at this chain
            if self.monitor_chains is not None:
                if ichain not in self.monitor_chains:
                    continue

            # If there are any small molecules in this chain, just clear out everything past the first sm because
            #  idk what's going on with the connectivity at that point
            mask_is_sm = indep.is_sm & chain_mask
            wh_sm = torch.where(mask_is_sm)[0]
            if len(wh_sm) > 0:
                first_sm = wh_sm[0]
                chain_mask[first_sm:] = False

            # Can't have a chainbreak if there is only 1 residue
            if chain_mask.sum() <= 1:
                continue

            # Extract the atoms of interest
            local_Ns = indep.xyz[chain_mask,0]
            local_Cs = indep.xyz[chain_mask,2]

            # Find the largest N->C gap
            C_from_N = local_Cs[:-1] - local_Ns[1:]
            C_N_dists = torch.linalg.norm( C_from_N, axis=-1)
            max_gap = C_N_dists.max()

            scores['max_chain_gap'] = max(scores['max_chain_gap'], max_gap)

        # The filter passes if the max_gap is less than the limit
        return scores['max_chain_gap'] <= self.C_N_dist_limit, scores



class InterchainClashFilter(FilterBase):
    '''
    A filter to find interchain clashes and stop runs if they appear
    '''

    def __init__(self, chainA=None, chainB=None, max_bb_clashes=5, clash_dist=3, use_px0=True, **kwargs):
        '''
        Args:
            chainA (int): Which is the first chain we'll look at. None for all
            chainB (int): Which is the second chain we'll look at. None for all
            max_bb_clashes (int): How many backbone clashes are acceptable between two chains
            clash_dist (float): At what distance do we consider CAs to be clashing
            use_px0 (bool): Use px0 as the structure to look for clashes in
        '''
        super().__init__(**kwargs)

        if chainA is not None:
            self.chainA = int(chainA)
        else:
            self.chainA = None

        if chainB is not None:
            self.chainB = int(chainB)
        else:
            self.chainB = None

        self.max_bb_clashes = max_bb_clashes
        self.clash_dist = clash_dist
        self.use_px0 = use_px0
        self.verbose = kwargs.get('verbose', False)

    def inner_do_filter(self, indep, px0_indep, **kwargs):
        '''
        Figure out how many clashes there are
        '''

        if self.use_px0:
            indep = px0_indep

        scores = {}
        scores['max_clashes'] = 0

        N_CA_C_O = indep.xyz[:,:3]
        clash_map = torch.sum( torch.square( N_CA_C_O[:,None,:,None] - N_CA_C_O[None,:,None,:] ), axis=-1) < self.clash_dist**2

        chain_masks = [torch.tensor(x) for x in indep.chain_masks()]

        for achain in range(len(chain_masks)):
            if self.chainA is not None and achain != self.chainA:
                continue

            for bchain in range(len(chain_masks)):
                if self.chainB is not None and bchain != self.chainB:
                    continue
                if achain == bchain:
                    continue

                mask_a = chain_masks[achain]
                mask_b = chain_masks[bchain]
                clashes = clash_map[mask_a][:,mask_b].sum()

                if self.verbose:
                    print(f'InterchainClashFilter: chain{achain}-chain{bchain}: {clashes} clashes')

                scores['max_clashes'] = max(clashes, scores['max_clashes'])

        passing = scores['max_clashes'] <= self.max_bb_clashes

        return passing, scores


class TestFilter(FilterBase):
    '''
    Debugging filter for unit tests
    '''

    def __init__(self, test_value_threshold=5, **kwargs):
        '''
        Args:
            test_value_threshold (int): If test_value is less than this, we fail
        '''
        super().__init__(**kwargs)

        self.test_value_threshold = test_value_threshold

    def get_test_value(self):
        '''
        This function is to be overwritten in the unit tests
        '''
        return 0

    def inner_do_filter(self, indep, px0_indep, contig_map, is_diffused, **kwargs):
        '''
        Maybe fail depending on value of get_test_value()
        '''
        test_value = self.get_test_value()

        scores = {}
        scores['test_value'] = test_value

        return scores['test_value'] >= self.test_value_threshold, scores

