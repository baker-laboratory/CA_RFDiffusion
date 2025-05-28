import torch
import itertools
import rf_diffusion.structure as structure
import rf_diffusion.ppi as ppi
import random
import os
import numpy as np

from rf_diffusion.conditions.util import expand_1d_atomized_ok_gp_not, expand_2d_atomized_ok_gp_not


# must match structure.HELIX, STRAND, LOOP
SS_HELIX = 0
SS_STRAND = 1
SS_LOOP = 2
SS_MASK = 3
SS_SM = 4 # SS_SM != structure.ELSE so that we maintain compatibility with old ss files
N_SS = 5

SS_oneletter = 'HEL-?'
SS_oneletter_np = np.array(list(SS_oneletter))

ADJ_FAR = 0
ADJ_CLOSE = 1
ADJ_MASK = 2
ADJ_STRAND_PAIR = 3
N_ADJ = 4

class SecStructAdjacency:
    '''
    Class that manage secondary structure conditioning and block adjacency conditioning
    '''

    ss: torch.Tensor # [L]
    adj: torch.Tensor # [L,L]

    def __init__(self, indep=None, full_mask_length=None, ss=None, adj=None):
        '''
        Either generate an exact SS/ADJ if indep is specified
        Or, generate a fully-masked object if full_mask_length is specified (that does nothing)
        Or, manually specify ss/adj yourself

        Args:
            indep (Indep): indep. If specified, generate ss_adj exactly
            full_mask_length (int): If specified, generate fully-masked ss and adj of length this
            ss (torch.Tensor[bool]): Use this ss
            adj (torch.Tensor[bool]): Use this adj
        '''

        if indep is not None:
            assert ss is None
            assert adj is None
            assert full_mask_length is None
            self.ss, self.adj = generate_ss_adj(indep)
            return

        if full_mask_length is not None:
            assert ss is None
            assert adj is None
            self.reset_ss(full_mask_length)
            self.reset_adj(full_mask_length)
            return

        self.ss = ss
        self.adj = adj


    def clone(self):
        return SecStructAdjacency(ss=self.ss, adj=self.adj)

    def is_fully_masked(self):
        '''
        Check if this ss_adj object is fully masked (i.e. it does nothing)

        Returns:
            is_fully_masked (bool): True if this ss_adj is fully masked
        '''

        return (self.ss == SS_MASK).all() and (self.adj == ADJ_MASK).all()

    def reset_ss(self, length):
        '''
        Convert self.ss to entirely SS_MASK at a new length

        Args:
            length (int): New length
        '''
        self.ss = torch.full((length,), SS_MASK, dtype=int)

    def reset_adj(self, length):
        '''
        Convert self.adj to entirely ADJ_MASK at a new length

        Args:
            length (int): New length
        '''
        self.adj = torch.full((length,length), ADJ_MASK, dtype=int)

    def mask_for_training(self, ss_min_mask=0, ss_max_mask=1.0, adj_min_mask=0, adj_max_mask=1.0,
                            adj_strand_pair_min_mask=0, adj_strand_pair_max_mask=1.0):
        '''
        Mask out regions of the ss and adj for training
        Chunks of ss near the edges of secondary structural elements are masked
        Full blocks of adj are masked
        Then any sections of ss == SS_MASK are masked in adj

        Args:
            ss_min_mask (float): Minimum fraction of ss that will be converted to SS_MASK
            ss_max_mask (float): Maximum fraction of ss that will be converted to SS_MASK
            adj_min_mask (float): Minimum fraction of adj that will be directly converted to ADJ_MASK
            adj_max_mask (float): Maximum fraction of adj that will be directly converted to ADJ_MASK
        '''

        new_adj = mask_strand_pairs(self.ss, self.adj, min_mask=adj_strand_pair_min_mask, max_mask=adj_strand_pair_max_mask)

        new_ss = mask_ss(self.ss, min_mask=ss_min_mask, max_mask=ss_max_mask)
        new_adj = mask_adj(self.ss, new_adj, min_mask=adj_min_mask, max_mask=adj_max_mask)
        new_adj = mask_adj_from_masked_ss(new_ss, new_adj)

        self.ss = new_ss
        self.adj = new_adj


    def generate_extra_t1d(self):
        '''
        Generate the extra_t1d values for ss

        Returns:
            extra_t1d_ss (torch.Tensor[float]): A one-hot vector with N_SS classes defining the secondary structure conditioning [L,5]
        '''
        assert self.ss is not None

        return torch.nn.functional.one_hot(self.ss, num_classes=N_SS).float()

    def generate_extra_t2d(self):
        '''
        Generate the extra_t1d values for ss

        Returns:
            extra_t2d_adj (torch.Tensor[float]): A one-hot vector with N_ADJ classes defining the block adjacency conditioning [L,L,3]
        '''
        assert self.adj is not None

        # ADJ_STRAND_PAIR implies ADJ_CLOSE. This keeps them independent
        one_hot = torch.nn.functional.one_hot(self.adj, num_classes=N_ADJ)
        one_hot[:,:,ADJ_CLOSE] |= one_hot[:,:,ADJ_STRAND_PAIR]
        return one_hot.float()

    def merge_other_into_this(self, other_ss_adj, other_overwrite_1d=None, other_overwrite_2d=None):
        '''
        Merge another SecStructAdj object into this one
        Values are completely overwritten in the regions specified and ignored otherwise

        One of other_overwrite_1d or other_overwrite_2d must be specified otherwise this function would do nothing

        Args:
            other_ss_adj (SecStructAdj): The ss_adj object where we are obtaining new values from
            other_overwrite_1d (torch.Tensor[bool]): Which values of ss should be overwritten? [L]
            other_overwrite_2d (torch.Tensor[bool]): Which values of adj should be overwritten? [L,L]
        '''
        assert other_overwrite_1d is not None or other_overwrite_2d is not None
        assert other_ss_adj.ss.shape == self.ss.shape
        assert other_ss_adj.adj.shape == self.adj.shape

        if other_overwrite_1d is not None:
            if (self.ss[other_overwrite_1d] != SS_MASK).any():
                print("Warning! Overwriting previously defined SS")
            self.ss[other_overwrite_1d] = other_ss_adj.ss[other_overwrite_1d]

        if other_overwrite_2d is not None:
            if (self.adj[other_overwrite_2d] != ADJ_MASK).any():
                print("Warning! Overwriting previously defined ADJ")
            self.adj[other_overwrite_2d] = other_ss_adj.adj[other_overwrite_2d]


    def expand_for_atomization_and_gp(self, indep, post_idx_from_pre_idx):
        '''
        Expand this SecStructAdjacency object after indep has been transformed by transform_indep

        For atomized residues, everything remains the same
        For gp residues, ss=SS_MASK and any interactions in adj are set to SS_MASK

        Args:
            indep (Indep): indep
            post_idx_from_pre_idx (list[list[int]]): Mapping from pre-transform to post-transform [L pre-transform]

        Returns:
            self (SecStructAdjacency): But the ss and adj have been expanded
        '''

        self.ss = expand_1d_atomized_ok_gp_not(indep, self.ss, post_idx_from_pre_idx, SS_MASK, key='ss_adj-ss')
        self.adj = expand_2d_atomized_ok_gp_not(indep, self.adj, post_idx_from_pre_idx, ADJ_MASK, key='ss_adj-adj')
        return self

    def pop_mask(self, pop):
        '''
        Remove positions from indep and subsequently this ss/adj object

        Args:
            pop (torch.Tensor[bool]): Positions to keep [L]

        Returns:
            None
        '''
        if self.ss is not None:
            assert len(self.ss) == len(pop), f'SecStructAdjacency.pop_mask() called but ss {len(self.ss)} does not match pop {len(pop)}'
            self.ss = self.ss[pop]

        if self.adj is not None:
            assert len(self.adj) == len(pop), f'SecStructAdjacency.pop_mask() called but adj {len(self.adj)} does not match pop {len(pop)}'
            self.adj = self.adj[pop]


def mask_adj_from_masked_ss(ss, adj):
    '''
    Apply ADJ_MASK to any position where ss == SS_MASK

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment
        adj (torch.Tensor[long]): Secondary structure adjacency
    '''

    adj = adj.clone()
    adj[ss==SS_MASK,:] = ADJ_MASK
    adj[:,ss==SS_MASK] = ADJ_MASK

    return adj

def generate_ss_adj(indep, dist_cutoff=6):
    '''
    we are going to deviate from the original Joe + Nate ss_adj script in the following ways
     1. Loops are able to take on ADJ_FAR and ADJ_CLOSE
     2. Small molecules are present and take on SS_SM
     3. Every heavy-atom of a small molecule will be considered a separate element for adj calculations
     4. The diagonal block adjacency is set to ADJ_CLOSE
    This function never returns SS_MASK or ADJ_MASK

    Args:
        indep (Indep): indep
        dist_cutoff (float): The cb_distance cutoff used to determine if structures are ADJ_CLOSE or ADJ_FAR

    Returns:
        ss (torch.Tensor[long]): Secondary structure assignment [L]
        adj (torch.Tensor[long]): Secondary structure adjacency [L]
    '''

    this_dssp, strand_pairs = structure.get_dssp(indep, compute_pairs=True)
    ss = this_dssp.clone()
    ss[this_dssp == structure.ELSE] = SS_SM

    # New strand pairing feature
    pair_map = torch.zeros((indep.length(), indep.length()), dtype=bool)
    for ((a,b),(c,d)) in strand_pairs:
        a,b = (a,b) if a < b else (b,a)
        c,d = (c,d) if c < d else (d,c)

        pair_map[a:b+1,c:d+1] = True
        pair_map[c:d+1,a:b+1] = True

    Cb = ppi.Cb_or_atom(indep)

    # Which Cbs are close enough to call the elements adjacent?
    Cb_dist2 = torch.sum( torch.square( Cb[:,None] - Cb[None,:] ), axis=-1 )
    Cb_close = Cb_dist2 < dist_cutoff**2

    # Iterate through the segments and fill the block adjacency matrix with any that are close
    block_adj = torch.full((indep.length(), indep.length()), ADJ_FAR)

    segments = ss_to_segments(ss)

    # For all segments
    for i_i_seg in range(len(segments)):
        i_seg = segments[i_i_seg]

        begin_i = i_seg[1]
        end_i = i_seg[2] + 1 # plus 1 so that the rest of the function looks like the original

        # For all other segments
        for j_j_seg in range(i_i_seg, len(segments)):
            j_seg = segments[j_j_seg]

            begin_j = j_seg[1]
            end_j = j_seg[2] + 1

            # If any two residues are within dist_cutoff, the entire elements are considered close
            if Cb_close[begin_i:end_i, begin_j:end_j].any():

                store_value = ADJ_CLOSE
                # ADJ_STRAND_PAIR takes precedence. But ADJ_CLOSE is also stored to extra_t2d
                if pair_map[begin_i:end_i, begin_j:end_j].any():
                    store_value = ADJ_STRAND_PAIR

                # Matrix is symmetric
                block_adj[begin_i:end_i, begin_j:end_j] = store_value
                block_adj[begin_j:end_j, begin_i:end_i] = store_value

    return ss, block_adj

def mask_strand_pairs(ss, adj, min_mask=0, max_mask=1.0):
    '''
    Randomly mask some proportion of the ADJ_STRAND_PAIR regions back to ADJ_CLOSE

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment
        adj (torch.Tensor[long]): Secondary structure adjacency
        min_mask (float): Minimum fraction of strand pairs that will be converted to ADJ_CLOSE
        max_mask (float): Maximum fraction of strand pairs that will be converted to ADJ_CLOSE

    Returns:
        adj (torch.Tensor[long]): Secondary structure adjacency [L]
    '''

    adj = adj.clone()

    # First we need to find all the strand-pair blocks
    assert not (ss == SS_MASK).any()
    segments = ss_to_segments(ss)

    # Enumerate all pairs of segments
    pairing_blocks = []
    for i_i_seg in range(len(segments)):
        i_type, i_start, i_end = segments[i_i_seg]
        for j_j_seg in range(i_i_seg, len(segments)):
            j_type, j_start, j_end = segments[j_j_seg]

            is_pairing = (adj[i_start:i_end+1,j_start:j_end+1] == ADJ_STRAND_PAIR).any()

            if is_pairing:
                assert i_type == SS_STRAND, i_type
                assert j_type == SS_STRAND, j_type
                assert i_i_seg != j_j_seg, "It's pairing to itself!"
                pairing_blocks.append((i_i_seg, j_j_seg))


    remove_frac = random.uniform(min_mask, max_mask)
    n_pairs = len(pairing_blocks)
    n_remove = int(np.round(remove_frac * n_pairs))

    if n_remove == 0:
        return adj

    idx_remove = np.random.choice(np.arange(n_pairs), n_remove, replace=False)
    for idx in idx_remove:
        i_i_seg, j_j_seg = pairing_blocks[idx]
        _, i_start, i_end = segments[i_i_seg]
        _, j_start, j_end = segments[j_j_seg]

        adj[i_start:i_end+1,j_start:j_end+1] = ADJ_CLOSE
        adj[j_start:j_end+1,i_start:i_end+1] = ADJ_CLOSE


    return adj



def mask_ss(ss, min_mask = 0, max_mask = 1.0):
    '''
    Mask out chunks of the ss vector near transitions from one to another
    Masks out up to 9 AA at a time

    nearly identical to the original code
    only changes are 3 -> SS_MASK
    and this function simply returns a vector like ss

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment [L]
        min_mask (float): Minimum fraction of ss that will be converted to SS_MASK
        max_mask (float): Maximum fraction of ss that will be converted to SS_MASK

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment but masked [L]
    '''
    ss = ss.clone()
    mask_prop = random.uniform(min_mask, max_mask)
    transitions = torch.where(ss[:-1] - ss[1:] != 0)[0] #gets last index of each block of ss
    stuck_counter = 0

    if len(transitions) == 0:
        return ss

    # Randomly try to mask around the transitons of ss until we've either masked enough or tried 100 times
    while len(ss[ss == SS_MASK])/len(ss) < mask_prop and stuck_counter < 100:
        width = random.randint(1,9)
        start = random.choice(list(transitions))
        offset = random.randint(-8,1)
        try:

            ss[start+offset:start+offset+width] = SS_MASK
        except IndexError:
            pass

        stuck_counter += 1
    return ss


def mask_adj(ss, adj, min_mask = 0, max_mask = 1.0):
    '''
    Mask out interactions in the adj matrix randomly
    This function picks entire blocks and converts them to SS_MASK

    the original code didn't do this but bcov thinks it might help

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment [L]
        adj (torch.Tensor[long]): Secondary structure adjacency [L]
        min_mask (float): Minimum fraction of adj that will be converted to ADJ_MASK
        max_mask (float): Maximum fraction of adj that will be converted to ADJ_MASK

    Returns:
        adj (torch.Tensor[long]): Secondary structure adjacency but masked[L]

    '''
    adj = adj.clone()
    mask_prop = random.uniform(min_mask, max_mask)

    assert not (ss == SS_MASK).any()

    segments = ss_to_segments(ss)

    # Enumerate all pairs of segments
    blocks = []
    for i_i_seg in range(len(segments)):
        for j_j_seg in range(i_i_seg, len(segments)):
            blocks.append((i_i_seg, j_j_seg))

    # Iterate through the blocks setting each one to ADJ_MASK until we've achieved mask_prob
    while (adj == ADJ_MASK).float().mean() < mask_prop:

        assert len(blocks) > 0, 'bcov messed up somehow. Send him this error message.'

        i_block = random.randint(0, len(blocks)-1)

        i_i_seg, j_j_seg = blocks[i_block]

        _, i_start, i_end = segments[i_i_seg]
        _, j_start, j_end = segments[j_j_seg]

        # Since matrix is symmetric. Mask out both ways these two segments interact
        adj[i_start:i_end+1,j_start:j_end+1] = ADJ_MASK
        adj[j_start:j_end+1,i_start:i_end+1] = ADJ_MASK

        blocks.pop(i_block)

    return adj


def repeating_regions(vector, only_keep=None):
    '''
    Returns regions of repeating elements
        (value, start, stop)
        stop is the last element of the region. For slicing using stop+1

    Args:
        vector (iterable): The vector to find repeating regions within
        only_keep (any): If set, only keep regions that match this value

    Returns:
        A list of regions
    '''
    offset = 0
    regions = []
    for value, group in itertools.groupby(vector):
        this_len = len(list(group))
        next_offset = offset + this_len
        if ( only_keep is None or only_keep == value ):
            regions.append( [value, offset, next_offset-1])
        offset = next_offset

    return regions


def ss_to_segments(ss, is_dssp=False):
    '''
    Split up a secondary string into secondary structural elements
        Every atom of a small molecule is a different element

    Args:
        ss (torch.Tensor): The secondary structure, either from structure.get_dssp() or SS files
        is_dssp (bool): True if this ss is directly from structure.get_dssp()

    Returns:
        segments (list[tuple[int, int, int]]): (ss_type, start, end) for each segment in the secondary structure
                                                to slice this element, use ss[start:end+1]

    '''
    SM = structure.ELSE if is_dssp else SS_SM

    # In the original from Nate + Joe, the "end" value of each segment is 1 past (like python slicing)
    # In this version, the "end" value is the final position with that ss type
    pre_segments = repeating_regions(ss)

    # We are calling every small molecule residue a different SS segment
    segments = []
    for seg in pre_segments:
        if seg[0] != SM:
            segments.append(seg)
        else:
            for i in range(seg[1], seg[2]+1):
                segments.append((SM, i, i))

    return segments



def is_legacy_ss_adj(ss, adj):
    '''
    There's not a great way to detect legacy ss adj files.
    Their biggest sin was calling all interactions with loops ADJ_FAR

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment
        adj (torch.Tensor[long]): Secondary structure adjacency

    Returns:
        is_legacy_ss_adj (bool): Is this from the previous Joe + Nate ss/adj generator?
    '''

    if (ss == SS_SM).any():
        return False

    is_loop = ss == SS_LOOP
    if (adj[is_loop] != ADJ_FAR).any():
        return False

    if (adj[:,is_loop] != ADJ_FAR).any():
        return False

    return True

def convert_legacy_ss_adj(ss, adj):
    '''
    If it's a legacy, we need to convert all loop interactions to SS_MASK

    Args:
        ss (torch.Tensor[long]): Secondary structure assignment
        adj (torch.Tensor[long]): Secondary structure adjacency

    Returns:
        ss (torch.Tensor[long]): Secondary structure assignment, but converted
        adj (torch.Tensor[long]): Secondary structure adjacency, but converted
    '''

    if not is_legacy_ss_adj(ss, adj):
        return ss, adj

    print("This looks like a legacy SS and ADJ file. Converting...")

    ss = ss.clone()
    adj = adj.clone()

    is_loop = ss == SS_LOOP
    adj[is_loop,:] = SS_MASK
    adj[:,is_loop] = SS_MASK

    return ss, adj


class LoadTargetSSADJTransform:
    def __call__(self, indep, conf, conditions_dict, **kwargs):
        '''
        A dataloader transform that applies scaffoldguided.target_ss and target_adj to the conditions_dict

        The ss and adj files are loaded to the back of the indep and must fully encompass chains (can't partially do a chain)
        You can theoretically specify the entire ss/adj with this overwriting your scaffold. But a warning is shown

        Args:
            indep (Indep): Indep
            conf (OmegaConf): The config
            conditions_dict (dict): The inference conditions

        Returns:
            Return signature is the same as call signature but conditions_dict['ss_adj'] has been updated
        '''

        ss = None
        size = None
        if conf.scaffoldguided.target_ss is not None:
            assert os.path.exists(conf.scaffoldguided.target_ss), f'scaffoldguided.target_ss: {conf.scaffoldguided.target_ss} does not exist!'
            ss = torch.load(conf.scaffoldguided.target_ss, weights_only=False)
            assert len(ss.shape) == 1
            size = ss.shape[0]

        adj = None
        if conf.scaffoldguided.target_adj is not None:
            assert os.path.exists(conf.scaffoldguided.target_adj), f'scaffoldguided.target_adj: {conf.scaffoldguided.target_adj} does not exist!'
            adj = torch.load(conf.scaffoldguided.target_adj, weights_only=False)
            assert len(adj.shape) == 2
            assert adj.shape[0] == adj.shape[1]
            size = adj.shape[0]

        if ss is not None and adj is not None:
            assert ss.shape[0] == adj.shape[0], f'Your target_ss {ss.shape} and target_adj {adj.shape} have different sizes!'

            if not conf.scaffoldguided.not_legacy_adj:
                convert_legacy_ss_adj(ss, adj)

        # If either is specified
        if size is not None:
            assert size > 0, 'Your target_ss or target_adj have size 0!'
            assert size <= indep.length(), 'Your target_ss/target_adj is bigger than your entire diffusion scenario!'

            # Make sure that they are not crossing chain boundaries with their ss or adj
            if size == indep.length():
                print('Warning! Your target_ss/target_adj are the same size as your entire diffusion scenario. Scaffold ss/adj will be overwritten!')
            else:
                chains = indep.chains()
                not_included = set(list(chains[:-size]))
                included = set(list(chains[-size:]))
                split_chains = included & not_included

                assert len(split_chains) == 0, f'Your target_ss/target_adj files are partially including chains {split_chains} which is not allowed.'
            
            # Actually do the assignment
            ss_adj = SecStructAdjacency(full_mask_length=indep.length())
            merge_1d = torch.zeros((indep.length(),), dtype=bool)
            merge_2d = torch.zeros((indep.length(), indep.length()), dtype=bool)

            if ss is not None:
                ss_adj.ss[-size:] = ss
                merge_1d[-size:] = True
            if adj is not None:
                ss_adj.adj[-size:,-size:] = adj
                merge_2d[-size:,-size:] = True

            if 'ss_adj' not in conditions_dict:
                conditions_dict['ss_adj'] = ss_adj
            else:
                conditions_dict['ss_adj'].merge_other_into_this(ss_adj, merge_1d, merge_2d)

        return kwargs | dict(
            indep=indep,
            conf=conf,
            conditions_dict=conditions_dict,
            )


class AutogenerateTargetSSADJTransform:
    def __call__(self, indep, conf, conditions_dict, **kwargs):
        '''
        A dataloader transform that autogenerates ss_adj for all chains besides the first and adds to the conditions_dict
            based on the presence of conf.scaffoldguided.autogenerate_target_ss_adj

        These are autogenerated in the same manner as make_secstruc_adj.py which is also the same manner as training
           The only difference is that loop regions do not have their adj set to ADJ_CLOSE but rather ADJ_FAR

        Args:
            indep (Indep): Indep
            conf (OmegaConf): The config
            conditions_dict (dict): The inference conditions

        Returns:
            Return signature is the same as call signature but conditions_dict['ss_adj'] has been updated
        '''

        if conf.scaffoldguided.autogenerate_target_ss_adj:
            chain_masks = indep.chain_masks()
            assert len(chain_masks) > 1, 'scaffoldguided.autogenerate_target_ss_adj called in a situation where there is only 1 chain!'
            binder_mask = chain_masks[0]
            target_mask = torch.tensor(~binder_mask)
            target_mask_2d = target_mask[:,None] & target_mask[None,:]

            # Build the precise ss/adj
            ss_adj = SecStructAdjacency(indep)

            # Clear out areas that aren't the target
            ss_adj.ss[binder_mask] = SS_MASK
            ss_adj.adj[~target_mask_2d] = ADJ_MASK


            if 'ss_adj' not in conditions_dict:
                conditions_dict['ss_adj'] = ss_adj
            else:
                conditions_dict['ss_adj'].merge_other_into_this(ss_adj, target_mask, target_mask_2d)

        return kwargs | dict(
            indep=indep,
            conf=conf,
            conditions_dict=conditions_dict,
            )


def get_ss_adj_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    '''
    See get_ss_adj_conditioning()
    '''
    return get_ss_adj_conditioning(indep, feature_conf, **kwargs)


def get_ss_adj_conditioning(indep, feature_conf, ss_adj=None, **kwargs):
    '''
    Generates the secondary structure conditioning and block adjacency features for training and inference

    Args:
        indep (Indep): indep
        feature_conf (OmegaConf): The configuration for this feature
        ss_adj (SecStructAdjacency or None): The secondary structure and adjacency matrices to apply from conditions_dict [L]

    Returns:
        dict:
            t1d (torch.Tensor[bool]): secondary structure conditioning [L,5]
            t1d (torch.Tensor[bool]): block adjacency conditioning [L,L,3]
    '''

    if ss_adj is None:
        ss_adj = SecStructAdjacency(full_mask_length=indep.length())

    assert len(ss_adj.ss) == indep.length(), 'ss_adj is not the same length as indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(ss_adj.adj) == indep.length(), 'ss_adj is not the same length as indep.length(). Is ExpandConditionsDict in conf.transforms?'

    ss_t1d = ss_adj.generate_extra_t1d()
    adj_t2d = ss_adj.generate_extra_t2d()

    return {'t1d':ss_t1d, 't2d':adj_t2d}


class GenerateSSADJTrainingTransform:
    """
    Transform that generates the masked ss/adj matrices for training

    Since this overwrites the entire ss_adj
    """    
    def __init__(self, p_is_ss_example=0, p_is_adj_example=0, ss_min_mask=0, ss_max_mask=1, adj_min_mask=0, adj_max_mask=1,
                    adj_strand_pair_min_mask=0, adj_strand_pair_max_mask=1, p_any_strand_pairs=0.2):
        """
        Args:
            p_is_ss_example (float): Probability we show any secondary structure at all
            p_is_adj_example (float): Probability we show any adjacency matrix at all
            ss_min_mask (float): Minimum fraction of ss that will be converted to SS_MASK
            ss_max_mask (float): Maximum fraction of ss that will be converted to SS_MASK
            adj_min_mask (float): Minimum fraction of adj that will be directly converted to ADJ_MASK
            adj_max_mask (float): Maximum fraction of adj that will be directly converted to ADJ_MASK
            adj_strand_pair_min_mask (float): Minimum fraction of strand pairs that will be converted to ADJ_CLOSE
            adj_strand_pair_max_mask (float): Maximum fraction of strand pairs that will be converted to ADJ_CLOSE
            p_any_strand_pairs (float): Probability that any strand pairs are shown at all
        """

        self.p_is_ss_example = p_is_ss_example
        self.p_is_adj_example = p_is_adj_example
        self.ss_min_mask = ss_min_mask
        self.ss_max_mask = ss_max_mask
        self.adj_min_mask = adj_min_mask
        self.adj_max_mask = adj_max_mask
        self.p_any_strand_pairs = p_any_strand_pairs
        self.adj_strand_pair_min_mask = adj_strand_pair_min_mask
        self.adj_strand_pair_max_mask = adj_strand_pair_max_mask


    def __call__(self, indep, conditions_dict, **kwargs):
        '''
        Args:
            indep (indep): indep
            conditions_dict (dict): The conditions_dict for training

        Returns:
            conditions_dict['ss_adj'] (SecStructAdjacency): Set to a masked version of the true ss_adj
        '''


        use_ss = (torch.rand(1) < self.p_is_ss_example).item()
        use_adj = (torch.rand(1) < self.p_is_adj_example).item()

        if use_ss or use_adj:
            ss_adj = SecStructAdjacency(indep=indep)

            pair_min = self.adj_strand_pair_min_mask
            pair_max = self.adj_strand_pair_max_mask
            if torch.rand(1) > self.p_any_strand_pairs:
                pair_min = 1
                pair_max = 1

            ss_adj.mask_for_training(ss_min_mask=self.ss_min_mask, ss_max_mask=self.ss_max_mask, 
                                    adj_min_mask=self.adj_min_mask, adj_max_mask=self.adj_max_mask,
                                    adj_strand_pair_min_mask=pair_min, adj_strand_pair_max_mask=pair_max)

            if not use_ss:
                ss_adj.reset_ss(indep.length())
            if not use_adj:
                ss_adj.reset_adj(indep.length())

            assert ('ss_adj' not in conditions_dict) or conditions_dict['ss_adj'].is_fully_masked(), ('GenerateSSADJTrainingTransform cannot'
                                                                " handle the situation where the conditions_dict['ss_adj'] has already been modified")
            conditions_dict['ss_adj'] = ss_adj

        return kwargs | dict(
            indep=indep,
            conditions_dict=conditions_dict
        )


def user_wants_ss_adj_scaffold(conf):
    '''
    The ss_adj stuff is turned on via the presence of any of 3 flags:
        - scaffoldguided.scaffold_list
        - scaffoldguided.scaffold_dir
        - scaffoldguided.scaffold_arc

    Args:
        conf (OmegaConf): The config

    Returns:
        user_wants_ss_adj (bool): Are they trying to use it?
    '''

    return bool(conf.scaffoldguided.scaffold_list) or bool(conf.scaffoldguided.scaffold_dir) or bool(conf.scaffoldguided.scaffold_arc)


def user_wants_ss_adj(conf):
    '''
    The ss_adj stuff is turned on via the presence of any of 3 flags:
        - scaffoldguided.scaffold_list
        - scaffoldguided.scaffold_dir
        - scaffoldguided.scaffold_arc

    Args:
        conf (OmegaConf): The config

    Returns:
        user_wants_ss_adj (bool): Are they trying to use it?
    '''

    scaffold_guided = user_wants_ss_adj_scaffold(conf)

    ss_sprinkle = ( 'SSSprinkleTransform' in conf.upstream_inference_transforms.names
                    and 'SSSprinkleTransform' in conf.upstream_inference_transforms.configs
                    and conf.upstream_inference_transforms.configs.SSSprinkleTransform.get('active', False)
        )

    return scaffold_guided or ss_sprinkle



def validate_ss_adj_strategy(conf):
    '''
    Makes sure that the checkpoint is capable of doing ss/adj if requested
        and that the proper transforms are in place
    '''

    wants_ss_adj = user_wants_ss_adj(conf)

    if bool(conf.scaffoldguided.target_ss) or bool(conf.scaffoldguided.target_ss):
        assert 'LoadTargetSSADJTransform' in conf.upstream_inference_transforms.names, ('To use scaffoldguided.target_ss and scaffoldguided.target_adj,'
                                            ' you must have LoadTargetSSADJTransform in your upstream_inference_transforms')
        wants_ss_adj = True

    if bool(conf.scaffoldguided.autogenerate_target_ss_adj):
        assert 'AutogenerateTargetSSADJTransform' in conf.upstream_inference_transforms.names, ('To use scaffoldguided.autogenerate_target_ss_adj,'
                                            ' you must have AutogenerateTargetSSADJTransform in your upstream_inference_transforms')
        wants_ss_adj = True


    if wants_ss_adj:
        assert 'GenerateSSADJTrainingTransform' in conf.upstream_training_transforms.names, "It seems like this model wasn't trained to do ss/adj"
        p_sum = conf.upstream_training_transforms.configs.GenerateSSADJTrainingTransform.get('p_is_ss_example', 0)
        p_sum += conf.upstream_training_transforms.configs.GenerateSSADJTrainingTransform.get('p_is_adj_example', 0)
        assert p_sum > 0, "It seems like this model wasn't trained to do ss/adj"
        assert 'ss_adj_cond' in conf.extra_tXd, "It seems like this model wasn't trained to do ss/adj"

    return



def randomly_insert_A_into_B_spaced(A, B):
    '''
    Insert the shorter vector A into the longer vector B making sure than no two A are next to each other
    Additionally, unless len(A) == len(B), the first and last elements will always come from B

    The order of A and B are preserved

    Args:
        A (torch.Tensor[type]): the shorter vector that no two shall be next to each other
        B (torch.Tensor[type]): the longer vector that we are going to insert into
    '''

    assert len(A) <= len(B)

    output = torch.zeros(len(A) + len(B), dtype=A.dtype)

    # special case of simply every-other (can't use below algorithm because there's a 0 at the start or end)
    if len(A) == len(B):
        A_first = torch.rand(1) < 0.5
        torch.as_strided(output, (len(A),), (2,), 0 if A_first else 1)[:] = A
        torch.as_strided(output, (len(A),), (2,), 1 if A_first else 0)[:] = B
        return output


    # Start by ensuring there is at least 1 B between every A and a B at the start and end
    spans_of_B = torch.ones(len(A)+1, dtype=int)

    # Now randomly divy the extra B into the gaps
    extra_B = len(B) - spans_of_B.sum()
    if extra_B.sum() > 0:
        spans_of_B += divy_among_weighted_bins( extra_B, torch.rand(len(spans_of_B)) + 0.0001 )

    # Store into output. I can't think of a way to do this without a for-loop
    out_offset = 0
    B_offset = 0
    for i in range(len(spans_of_B)):
        span_size = spans_of_B[i]
        output[out_offset:out_offset+span_size] = B[B_offset:B_offset+span_size]
        out_offset += span_size
        B_offset += span_size
        if i < len(A):
            output[out_offset] = A[i]
            out_offset += 1

    assert out_offset == len(output) and B_offset == len(B)

    return output



def try_to_shuffle_chunks_no_consecutive_loops_masks(input_chunks):
    '''
    Shuffle chunks but make sure that SS_LOOP and SS_MASK tokens are not next to themselves or each other
    Or do the best we can

    Args:
        input_chunks (torch.Tensor[int]): The chunks we are going to shuffle

    Returns:
        output_chunks (torch.Tensor[int]): The shuffled chunks
    '''

    # First separate out the two classes and shuffle them separately
    not_loop_mask = input_chunks[(input_chunks != SS_LOOP) & (input_chunks != SS_MASK)]
    loop_mask = input_chunks[(input_chunks == SS_LOOP) | (input_chunks == SS_MASK)]

    not_loop_mask = ppi.torch_rand_choice_noreplace(not_loop_mask, len(not_loop_mask))
    loop_mask = ppi.torch_rand_choice_noreplace(loop_mask, len(loop_mask))

    # Either it's possible to do this or it's not
    #  In either case, we insert the smaller vector into the larger vector
    if len(not_loop_mask) >= len(loop_mask):
        return randomly_insert_A_into_B_spaced(loop_mask, not_loop_mask)
    else:
        return randomly_insert_A_into_B_spaced(not_loop_mask,loop_mask)


def divy_among_weighted_bins(N, weights):
    '''
    Allocate N elements among weighted bins such that the number of elements in each bin
      correlates with weights but such that the sum == N

    Like out = N * weights but where it handles the quantized natured of integers

    Args:
        N (int): Number of elements to divy
        weights (torch.tensor[float]): The weight of each bin
    '''

    weights = weights / weights.sum()

    # First assign the whole-number parts
    output = torch.floor( N * weights ).long()

    remaining = N - output.sum()
    assert remaining >= 0

    # Now figure out who most deserves the few extra slots
    weight_remaining = weights - output / N
    arg_most_remaining = torch.argsort(-weight_remaining)

    # Store the remaining N to the most deserving bins
    output[arg_most_remaining[:remaining]] += 1

    return output


class SSSprinkleTransform:
    '''
    A transform that constructs on-the-fly SS tensors to bias the output towards different folds

    The SS mask is fully initialized to "background" and then chunks of helix, strand, loop, and mask are injected into the
      space. The chunks are spaced out somewhat evenly depending on spread_efficiency
    '''

    def __init__(self, active=False, background='MASK', helix_chunk_size=3, strand_chunk_size=3, loop_chunk_size=3, mask_chunk_size=3, min_helix=0, max_helix=0,
            min_strand=0, max_strand=0, min_loop=0, max_loop=0, min_mask=0, max_mask=0, spread_efficiency=0.7, chain0_only=True,
            use_this_ss_ordering=None, no_consecutive_loops_masks=True):
        '''
        Args:
            active (bool): Whether or not this transform does anything at all
            background (str): The default value for the SS that we insert chunks into ['HELIX', 'STRANd', 'LOOP', 'MASK']
            helix_chunk_size (int): Size of the chunk of helix to insert into the SS
            strand_chunk_size (int): Size of the chunk of strand to insert into the SS
            loop_chunk_size (int): Size of the chunk of loop to insert into the SS
            mask_chunk_size (int): Size of the chunk of mask to insert into the SS
            min_helix (int): Minimum number of helix sections to insert
            max_helix (int): Maximum number of helix sections to insert
            min_strand (int): Minimum number of strand sections to insert
            max_strand (int): Maximum number of strand sections to insert
            min_loop (int): Minimum number of loop sections to insert
            max_loop (int): Maximum number of loop sections to insert
            min_mask (int): Minimum number of mask sections to insert
            max_mask (int): Maximum number of mask sections to insert
            spread_efficiency (float): 0-1 How balanced the gaps between chunks should be. 1 = perfectly balanced. 0 = could be side-by-side
            chain0_only (bool): Only apply this transform to the first chain
            use_this_ss_ordering (str): Instead of randomly inserting chunks from the max_ and min_ variables. Directly use this string (HEEHE for instance)
            no_consecutive_loops_masks (bool): If possible, ensure that no LOOP or MASK chunks are inserted next to each other
        '''

        background_keys = {'HELIX':SS_HELIX, 'STRAND':SS_STRAND, 'LOOP':SS_LOOP, 'MASK':SS_MASK}

        assert background in background_keys
        self.background = background_keys[background]
        self.active = active
        self.min_helix = min_helix
        self.max_helix = max_helix
        self.min_strand = min_strand
        self.max_strand = max_strand
        self.min_loop = min_loop
        self.max_loop = max_loop
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.spread_efficiency = spread_efficiency
        self.chain0_only = chain0_only
        self.use_this_ss_ordering = use_this_ss_ordering
        self.no_consecutive_loops_masks = no_consecutive_loops_masks

        self.chunk_sizes = torch.zeros(N_SS, dtype=int)
        self.chunk_sizes[SS_HELIX] = helix_chunk_size
        self.chunk_sizes[SS_STRAND] = strand_chunk_size
        self.chunk_sizes[SS_LOOP] = loop_chunk_size
        self.chunk_sizes[SS_MASK] = mask_chunk_size

        if self.use_this_ss_ordering:
            for letter in self.use_this_ss_ordering:
                assert letter in SS_oneletter, f'SSSprinkleTransform use_this_ss_ordering: Error! Valid ss choices are {SS_oneletter} and you picked {letter}'
            assert self.min_helix + self.max_helix + self.min_strand + self.max_strand + self.min_loop + self.max_loop + self.min_mask + self.max_mask == 0, (
                "SSSprinkleTransform use_this_ss_ordering: Error! If you specify use_this_ss_ordering you can't specify any of the min_ or max_ settings")

        assert self.spread_efficiency >= 0 and self.spread_efficiency <= 1, 'SSSprinkleTransform spread_efficiency must be between 0 and 1 (inclusive)'

    def __call__(self, indep, conditions_dict, **kwargs):

        if self.active:

            # Figure out which positions we want to assign
            store_mask = ~indep.is_sm
            if self.chain0_only:
                store_mask[~torch.tensor(indep.chain_masks()[0])] = False

            # Build our lists of chunks to insert
            if self.use_this_ss_ordering:
                # User directly specified the order
                chunks = torch.tensor([SS_oneletter.index(s) for s in self.use_this_ss_ordering], dtype=int)
            else:
                # Build a list of chunks and then shuffle them
                chunks = []
                chunks.extend( [SS_HELIX]*random.randint(self.min_helix, self.max_helix))
                chunks.extend( [SS_STRAND]*random.randint(self.min_strand, self.max_strand))
                chunks.extend( [SS_LOOP]*random.randint(self.min_loop, self.max_loop))
                chunks.extend( [SS_MASK]*random.randint(self.min_mask, self.max_mask))
                if self.no_consecutive_loops_masks:
                    chunks = try_to_shuffle_chunks_no_consecutive_loops_masks(torch.tensor(chunks, dtype=int))
                else:
                    random.shuffle(chunks)
                    chunks = torch.tensor(chunks, dtype=int)

            # Assign the sizes of each gap. Start with perfectly even assignment and add randomness based on spread_efficiency
            gap_weights = torch.ones(len(chunks)+1)
            gap_weights -= (torch.rand(len(chunks)+1) + 0.01) * (1.0-self.spread_efficiency*0.99999) # make sure there is some slight randomness
            gap_weights /= gap_weights.sum()

            # Figure out the exact sizes of the gaps
            chunk_sizes = self.chunk_sizes[chunks]

            if chunk_sizes.sum() > store_mask.sum():
                print('SSSprinkleTransform: Warning! Chunk sizes sum to more than available amino acids in protein')
                gap_sizes = torch.zeros(len(gap_weights), dtype=int)
            else:
                remaining = store_mask.sum() - chunk_sizes.sum()
                gap_sizes = divy_among_weighted_bins(remaining, gap_weights)
                assert gap_sizes.sum() + chunk_sizes.sum() == store_mask.sum()

            # Generate the final ss tensor spacing the chunks out by the gaps
            inner_ss = torch.full((store_mask.sum(),), self.background)
            for i_chunk in range(len(chunks)):
                start_idx = chunk_sizes[:i_chunk].sum() + gap_sizes[:i_chunk+1].sum()
                end_idx = start_idx + chunk_sizes[i_chunk]
                inner_ss[start_idx:end_idx] = chunks[i_chunk]

            # Expand the final ss tensor to the full indep size
            new_ss = torch.full((indep.length(),), SS_MASK)
            new_ss[store_mask] = inner_ss

            print("SSSprinkleTransform:", ''.join(SS_oneletter_np[new_ss.numpy()]))

            # Generate the ss_adj object to merge into conditions dict
            ss_adj = SecStructAdjacency(full_mask_length=indep.length())
            ss_adj.ss[:] = new_ss

            if 'ss_adj' not in conditions_dict:
                conditions_dict['ss_adj'] = ss_adj
            else:
                conditions_dict['ss_adj'].merge_other_into_this(ss_adj, store_mask, torch.zeros((indep.length(), indep.length()), dtype=bool))


        return kwargs | dict(
            indep=indep,
            conditions_dict=conditions_dict,
            )

class ADJSprinkleTransform:
    '''
    A transform that constructs on-the-fly ADJ matrices to bias the output towards different things

    '''

    def __init__(self, active=False, use_existing_ss=False, use_existing_mode='full', use_existing_chunk_size=4, use_existing_loop_is_mask=True,
                      from_scratch_chunk_size=4, from_scratch_off_diag_min=8, from_scratch_off_diag_max=12,
                      from_scratch_coverage=0.5, from_scratch_spread_efficiency=0.7, chain0_only=True):

        self.active = active
        self.use_existing_ss = use_existing_ss
        self.use_existing_mode = use_existing_mode
        self.use_existing_chunk_size = use_existing_chunk_size
        self.use_existing_loop_is_mask = use_existing_loop_is_mask
        self.from_scratch_chunk_size = from_scratch_chunk_size
        self.from_scratch_off_diag_min = from_scratch_off_diag_min
        self.from_scratch_off_diag_max = from_scratch_off_diag_max
        self.from_scratch_coverage = from_scratch_coverage
        self.from_scratch_spread_efficiency = from_scratch_spread_efficiency
        self.chain0_only = chain0_only

        assert use_existing_mode in ['junction', 'distal', 'middle', 'full', 'random']


    def __call__(self, indep, conditions_dict, **kwargs):

        if self.active:

            if self.chain0_only:
                use_mask = torch.tensor(indep.chain_masks()[0])
            else:
                use_mask = torch.ones(indep.length(), dtype=bool)

            adj_to_store = torch.full((use_mask.sum(), use_mask.sum()), ADJ_MASK)

            if self.use_existing_ss:

                assert 'ss_adj' in conditions_dict, 'ADJSprinkleTransform: If you wish to use use_existing_ss, there must actually be an SS'
                prev_ss = conditions_dict['ss_adj'].ss[use_mask].clone()

                assert not (prev_ss == SS_MASK).all(), ('ADJSprinkleTransform: If you wish to use use_existing_ss, there must actually be an SS.'
                                " (Technically there is one. But it's all SS_MASK which probably means it's not initialized)" )

                if self.use_existing_loop_is_mask:
                    prev_ss[prev_ss == SS_LOOP] = SS_MASK


                segments = [x for x in ss_to_segments(prev_ss) if x[0] != SS_MASK]

                for iseg in range(len(segments)-1):
                    tp1, start1, end1 = segments[iseg]
                    tp2, start2, end2 = segments[iseg+1]

                    size1 = end1-start1+1
                    size2 = end2-start2+1

                    N = self.use_existing_chunk_size

                    mode = self.use_existing_mode
                    if mode == 'random':
                        modes = ['junction', 'distal', 'middle', 'full']
                        mode = modes[random.randint(0, len(modes)-1)]

                    if mode == 'junction':
                        # at the junction
                        left_edge = [end1-N+1, start2]
                        right_edge = [end1+1, start2+N]
                    elif mode == 'distal':
                        # at the distal tips
                        left_edge = [start1, end2-N+1]
                        right_edge = [start1+N, end2+1]
                    elif mode == 'middle':
                        # in the middle
                        c1 = int(size1 //2) + start1
                        c2 = int(size2 //2) + start2
                        left_N = int(N//2)
                        right_N = N - left_N
                        left_edge = [c1-left_N, c2-left_N]
                        right_edge = [c1+right_N, c2+right_N]
                    elif mode == 'full':
                        left_edge = [start1, start2]
                        right_edge = [end1+1, end2+1]
                    else:
                        assert False, f'unknown mode {mode}'

                    left_edge = torch.tensor(left_edge)
                    right_edge = torch.tensor(right_edge)

                    stacked_edges = torch.stack([left_edge, right_edge])
                    stacked_edges[:,0] = torch.clip(stacked_edges[:,0], start1, end1+1)
                    stacked_edges[:,1] = torch.clip(stacked_edges[:,1], start2, end2+1)

                    ((l1, l2), (r1, r2)) = stacked_edges

                    adj_to_store[l1:r1,l2:r2] = ADJ_CLOSE
                    adj_to_store[l2:r2,l1:r1] = ADJ_CLOSE

            else:

                N_total = len(adj_to_store)

                N_chunks = int(N_total * self.from_scratch_coverage / self.from_scratch_chunk_size)

                N_close = N_chunks * self.from_scratch_chunk_size

                N_excess = N_total - N_close

                N_gaps = N_chunks + 1

                gap_weights = torch.ones(N_gaps)
                gap_weights -= (torch.rand(N_gaps) + 0.01) * (1.0-self.from_scratch_spread_efficiency*0.99999)
                gap_weights /= gap_weights.sum()

                gap_sizes = divy_among_weighted_bins(N_excess, gap_weights)


                for ichunk in range(N_chunks):
                    offset_from_chunks = self.from_scratch_chunk_size * ichunk
                    offset_from_gaps = gap_sizes[:ichunk+1].sum()

                    l1 = offset_from_gaps + offset_from_chunks
                    r1 = l1 + self.from_scratch_chunk_size

                    l2 = l1 + random.randint(self.from_scratch_off_diag_min, self.from_scratch_off_diag_max)
                    r2 = l2 + self.from_scratch_chunk_size

                    adj_to_store[l1:r1,l2:r2] = ADJ_CLOSE
                    adj_to_store[l2:r2,l1:r1] = ADJ_CLOSE


            full_adj = torch.full((indep.length(), indep.length()), ADJ_MASK)
            full_store_mask = torch.ones((indep.length(), indep.length()), dtype=bool)

            full_store_mask[~use_mask] = False
            full_store_mask[:,~use_mask] = False

            full_adj[full_store_mask] = adj_to_store.reshape(-1)

            # Generate the ss_adj object to merge into conditions dict
            ss_adj = SecStructAdjacency(full_mask_length=indep.length())
            ss_adj.adj[:] = full_adj

            if 'ss_adj' not in conditions_dict:
                conditions_dict['ss_adj'] = ss_adj
            else:
                conditions_dict['ss_adj'].merge_other_into_this(ss_adj, torch.zeros(indep.length(), dtype=bool), full_adj != ADJ_MASK)


            print('adj sprinkle active')
            import matplotlib.pyplot as plt
            import seaborn as sns
            to_plot = torch.nn.functional.pad(adj_to_store, [0, 1, 0, 1])
            prev_ss = conditions_dict['ss_adj'].ss[use_mask].clone()
            to_plot[-1,:-1] = prev_ss
            to_plot[:-1,-1] = prev_ss
            sns.heatmap(to_plot)
            plt.savefig('my_adj.png')



        return kwargs | dict(
            indep=indep,
            conditions_dict=conditions_dict,
            )