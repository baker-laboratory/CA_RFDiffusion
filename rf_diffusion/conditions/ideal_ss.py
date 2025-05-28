import torch
import numpy as np


from rf_diffusion import structure
from rf_diffusion import ppi
from rf_diffusion import aa_model
from rf_diffusion.conditions.ss_adj import sec_struct_adjacency as sec_struct_adj

from collections.abc import Mapping



def my_pca(x):
    '''
    Manual PCA decomposition to avoid sklearn dependency
    https://jamesmccaffrey.wordpress.com/2021/07/16/computing-pca-using-numpy-without-scikit/
    
    Matches sklearn.decomposition.PCA() exactly
    
    Args:
        x (torch.Tensor): A 2d array [L,n]
        
    Returns:
        prin_comp (torch.Tensor): The principle components sorted by importance [n,n]
        variance_explained (torch.Tensor): Fraction of data explained by each component [n]
    '''
    z = x - torch.mean(x, axis=0) # center
    square_m = torch.matmul(z.T, z)
    (evals, evecs) = torch.linalg.eig(square_m)  # 'right-hand'
    trans_x = torch.matmul(torch.complex(z, torch.tensor(0, dtype=z.dtype)), evecs)
    assert torch.allclose(trans_x.imag, torch.tensor(0, dtype=z.dtype), atol=1e-4), f'Imaginary result. What does this mean? {trans_x}'
    trans_x = trans_x.real
    prin_comp = evecs.real.T  # principal components are eigenvecs T
    v = torch.var(trans_x, axis=0)  # col sample var
    sv = torch.sum(v)
    variance_explained = v / sv
    # order everything based on variance explained
    ordering = torch.argsort(-variance_explained)  # sort order high to low
    prin_comp = prin_comp[ordering]
    variance_explained = variance_explained[ordering]
    return (prin_comp, variance_explained)

def orient_a_vector_to_b(a, b):
    '''
    Given two vectors a and b. Maybe flip a such that it points in the same direction as b

    Args:
        a (torch.Tensor): The vector to maybe flip
        b (torch.Tensor): The guide direction

    Returns:
        a (torch.Tensor): a but pointing in the same direction as b
    '''
    
    if ( a * b ).sum() < 0:
        return a * -1
    else:
        return a


def get_pca_and_orient_forwards(cas):
    '''
    Find the principle component of a chunk of protein and ensure that the vector points N to C

    This function will basically give you the helical axis of a helix or the strand axis of a beta strand

    Args:
        cas (torch.Tensor): A collection of points. Ideally from a protein N to C [L,3]

    Return:
        axis (torch.Tensor): The principle component of those points [3]
    '''

    # Get PCA and make sure it's normalized
    axis = my_pca(cas)[0][0]
    axis /= torch.linalg.norm(axis)
    
    # now we orient the PCAs so that they all point forwards
    #  this handles sharp kinks better than dotting with the previous pca
    first_to_last = cas[-1] - cas[0]

    axis = orient_a_vector_to_b( axis, first_to_last )

    return axis


# A horrible dictionary that gives Brian's ideality score to a secondary structure of this length
ss_len_to_ideality = {
    'strand': {
        1: 0,
        2: 0.2,
        3: 0.5,
        4: 0.7,
        5: 0.8,
        6: 0.9,
        7: 1.0
    },
    'helix': {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0.1,
        6: 0.2,
        7: 0.3,
        8: 0.4,
        9: 0.5,
        10: 0.6,
        11: 0.67,
        12: 0.74,
        13: 0.81,
        14: 0.88,
        15: 0.95,
        16: 1.0,
    },
    'loop_strand_strand': {
        1: 0.85,
        2: 1.0,
        3: 1.0,
        4: 0.7,
        5: 0.5,
        6: 0.3,
        7: 0.1,
        8: 0.0,
    },
    'loop_terminal': {
        1: 1,
        2: 0.8,
        3: 0.6,
        4: 0.3,
        5: 0,
    },

    'loop_else': {
        1: 0.85,
        2: 1.0,
        3: 1.0,
        4: 1.0,
        5: 0.85,
        6: 0.7,
        7: 0.55,
        8: 0.4,
        9: 0.2,
        10: 0.0,
    },
}

def check_curviness(cas, start, end, look_length, gap=0):
    '''
    Check how much a chunk of secondary structure curves in degrees

    This calculationis performed by finding the pca of consecutive chunks and 
        calculating the angle between them

    Args:
        cas (torch.Tensor): A chunk of protein CA atoms [L,3]
        start (int): The starting offset to look at
        end (int): The ending offset to look at (inclusive)
        look_length (int): How big the chunks are to find the PCA of
        gap (int): Instead of being consecutive, the chunks will have a gap between them
    '''

    worst_dot = torch.tensor(1)
    for i_low in range(start, end-look_length*2-gap):
        i_upper = i_low + look_length + gap
        lower_pca = get_pca_and_orient_forwards(cas[i_low:i_low+look_length])
        upper_pca = get_pca_and_orient_forwards(cas[i_upper:i_upper+look_length])

        dot = (lower_pca * upper_pca).sum()
        worst_dot = min(dot, worst_dot)

    return torch.rad2deg(torch.arccos(worst_dot))


def merge_barely_broken_ss(segments, cas, segment_join_angle=45):
    '''
    Merge secondary structural segments that have 1 aa breaks in them

    Args:
        segments (list[tuple[int]]): The output from sec_struct_adj.ss_to_segments
        cas (torch.Tensor[float]): The CAs of your indep
        segment_join_angle (float): If the angle between two same-type segments separated by 1aa is less than this, merge them

    Returns:
        new_segments (list[tuple[int]]): The original segments but with 1aa gaps removed
    '''

    # Find the direction every ss segment is pointing
    pcas = []
    for seg in segments:
        typ, start, end = seg

        if end > start:
            pca = get_pca_and_orient_forwards(cas[start:end+1])
        else:
            pca = None
        pcas.append(pca)

    # Try to absorbe 1aa gaps in helices and strands
    seg_should_be_absorbed = torch.zeros(len(segments), dtype=bool)
    for iseg in range(1, len(segments)-1):
        tp, start, end = segments[iseg]
        # Can only absorb 1-sized elements
        if start != end:
            continue
        # don't mess around with small molecules
        if tp == structure.ELSE:
            continue

        prev_seg = segments[iseg-1]
        next_seg = segments[iseg+1]

        # Can't join dissimilar elements
        if prev_seg[0] != next_seg[0]:
            continue

        # Make sure we're joining a helix or a strand
        if prev_seg[0] not in [structure.STRAND, structure.HELIX]:
            continue

        last_pca = pcas[iseg-1]
        next_pca = pcas[iseg+1]

        # If there are two 1-unit segs in a row give up
        if last_pca is None or next_pca is None:
            continue

        angle = torch.rad2deg(torch.arccos( torch.sum(last_pca * next_pca) ))
        # Now we only absorbe them if the elements are fairly aligned
        if angle < segment_join_angle:
            seg_should_be_absorbed[iseg] = True
            seg_should_be_absorbed[iseg+1] = True

    # Combine segments into longer ones
    new_segments = []
    iseg = 0
    while iseg < len(segments):

        tp, start, end = segments[iseg]
        while iseg < len(segments)-1 and seg_should_be_absorbed[iseg+1]:
            iseg += 1
            _, _, end = segments[iseg]

        new_segments.append((tp, start, end))
        iseg += 1

    return new_segments




def get_ideal_ss_score(indep, segment_join_angle=45, short_curve_low=15, short_curve_high=120, long_curve_low=15, long_curve_high=120, helix_look_length=8, debug=False):
    '''
    Get Brian's ideality score for each secondary structure. This score is totally arbitrary and Brian made it up
    Classic rosetta designs usually score about 0.80 and above

    The ideality score is: length_score * short_curve_score * long_curve_score
    Where 1 is perfectly ideal and 0 is totally not ideal

    Where:
        length_score: A score decided by ss_len_to_ideality. Longer is better for strands and helices but shorter is better for loops
        short_curve_score: Only for helices, how curved is it at short scales? Angle between consecutive 8aa chunks from <= 15 deg (1) to >= 120 deg (0)
        long_curve_score: Only for helices, how curved is it at long scales? Angle between 8aa chunks but with a 4aa gap from <= 15 deg (1) to >= 120 deg (0)

    Args:
        indep (indep): Indep
        segment_join_angle (int): Join adjacent SS elements if there is a 1aa loop between them and their angles differ by less than this
        short_curve_low (int): The angle below which helices are considered perfectly straight
        short_curve_high (int): The angle above which helices are totally not straight
        long_curve_low (int): The angle below which helices are considered perfectly straight
        long_curve_high (int): The angle above which helices are totally not straight
        helix_look_length (int): The length of the chunk of helix to examine
        debug (bool): Return the actual scores that went into the calculation

    '''

    # Prepare the vectors
    ideal_ss = torch.full((indep.length(),), np.nan)
    short_curve = torch.full((indep.length(),), np.nan)
    long_curve = torch.full((indep.length(),), np.nan)

    # Look at each chain independently
    for chain_mask in indep.chain_masks():

        # Get the chain
        chain_mask = torch.tensor(chain_mask)
        indep_chain, _ = aa_model.slice_indep(indep, chain_mask, break_chirals=True)

        # Find the secondary structural elements
        dssp, _ = structure.get_dssp(indep_chain)
        segments = sec_struct_adj.ss_to_segments(dssp, is_dssp=True)
        cas = indep_chain.xyz[:,1]

        new_segments = merge_barely_broken_ss(segments, cas, segment_join_angle=45)

        # Calculate the score of each secondary structure based on its length
        length_score = torch.full((indep_chain.length(),), np.nan)

        for iseg in range(len(new_segments)):
            tp, start, end = new_segments[iseg]
            length = end - start + 1

            score = np.nan
            if tp == structure.HELIX:
                max_length = max(list(ss_len_to_ideality['helix'].keys()))
                if length > max_length:
                    score = 1
                else:
                    score = ss_len_to_ideality['helix'][length]

            if tp == structure.STRAND:
                max_length = max(list(ss_len_to_ideality['strand'].keys()))
                if length > max_length:
                    score = 1
                else:
                    score = ss_len_to_ideality['strand'][length]

            if tp == structure.LOOP:
                d = ss_len_to_ideality['loop_else']
                if iseg > 0 and iseg < len(new_segments)-1 and new_segments[iseg-1][0] == structure.STRAND and new_segments[iseg+1][0] == structure.STRAND:
                    d = ss_len_to_ideality['loop_strand_strand']

                if iseg == 0 or iseg == len(new_segments)-1:
                    d = ss_len_to_ideality['loop_terminal']

                max_length = max(list(d.keys()))
                if length > max_length:
                    score = 0
                else:
                    score = d[length]

            length_score[start:end+1] = score

        # Calculate scores for helices based on their curviness
        short_curve_score = torch.ones(indep_chain.length())
        long_curve_score = torch.ones(indep_chain.length())
        for tp, start, end in new_segments:
            if tp != structure.HELIX:
                continue

            angle_short = check_curviness(cas, start, end, helix_look_length, gap=0)
            angle_long = check_curviness(cas, start, end, helix_look_length, gap=4)

            if angle_short < short_curve_low:
                short_score = 1
            elif angle_short < short_curve_high:
                short_score = 1 - (angle_short - short_curve_low) / (short_curve_high - short_curve_low)
            else:
                short_score = 0

            if angle_long < long_curve_low:
                long_score = 1
            elif angle_long < long_curve_high:
                long_score = 1 - (angle_long - long_curve_low) / (long_curve_high - long_curve_low)
            else:
                long_score = 0


            short_curve_score[start:end+1] = short_score
            long_curve_score[start:end+1] = long_score

        # Store the scores
        ideal_score = length_score * short_curve_score * long_curve_score
        ideal_ss[chain_mask] = ideal_score
        short_curve[chain_mask] = short_curve_score
        long_curve[chain_mask] = long_curve_score

    if debug:
        return ideal_ss, short_curve, long_curve

    return ideal_ss



def get_loop_frac_by_chain(indep):
    '''
    For each chain, what fraction of the DSSP is loop?

    Args:
        indep (indep): indep

    Returns:
        loop_frac (torch.Tensor[float]): Fraction of this chain that is loop [L]
    '''

    loop_frac = torch.full((indep.length(),), np.nan)

    for chain_mask in indep.chain_masks():

        # Grab the chain
        chain_mask = torch.tensor(chain_mask)
        indep_chain, _ = aa_model.slice_indep(indep, chain_mask, break_chirals=True)

        # Tally the loop and non-loop portions
        dssp, _ = structure.get_dssp(indep_chain)
        n_loop = (dssp == structure.LOOP).sum()
        n_helix_strand = ((dssp == structure.HELIX) | (dssp == structure.STRAND)).sum()

        # NaN if there is no secondary structure or calculate the fraction
        if n_loop + n_helix_strand == 0:
            frac_loop = np.nan
        else:
            frac_loop = n_loop / ( n_loop + n_helix_strand )

        loop_frac[chain_mask] = frac_loop

    return loop_frac



def get_avg_scn_by_chain(indep):
    '''
    Calculate the average sidechain neighbors for each chain

    Args:
        indep (indep): indep

    Returns:
        avg_scn (torch.Tensor[float]): The average sidechain neighbors for each chain [L]
    '''
    Cbs = ppi.Cb_or_atom(indep)
    Cas = indep.xyz[:,1]

    avg_scn = torch.full((indep.length(),), np.nan)

    for mask in indep.chain_masks():

        # small molecules can't be used
        mask = torch.tensor(mask)
        mask &= ~indep.is_sm

        if mask.sum() == 0:
            continue

        scn = ppi.sidechain_neighbors(Cas[mask], Cbs[mask], Cas[mask])
        avg_scn[mask] = torch.mean(scn)

    return avg_scn

def get_scn_by_chain(indep):
    '''
    Get the exact sidechain neighbors of every position on every chain when their are separated

    Args:
        indep (indep): indep

    Returns:
        actual_scn (torch.Tensor[float]): The actual sidechain neighbors for each position [L]
    '''
    Cbs = ppi.Cb_or_atom(indep)
    Cas = indep.xyz[:,1]

    actual_scn = torch.full((indep.length(),), np.nan)

    for mask in indep.chain_masks():

        # small molecules can't be used
        mask = torch.tensor(mask)
        mask &= ~indep.is_sm

        if mask.sum() == 0:
            continue

        scn = ppi.sidechain_neighbors(Cas[mask], Cbs[mask], Cas[mask])
        actual_scn[mask] = scn

    return actual_scn


def assign_topo_spec(indep, topo_spec_choices, min_helix_length=8):
    '''
    Assign the class label of which topology each chain is

    Calculates the dssp, turns it into chunks, removes loops, then compares against choices

    Args:
        indep (indep): indep
        topo_spec_choices (list[str]): The topology choices for the topologis (think HHH and HEEH)
        min_helix_length (int): If a helix exists with length shorter than this assign topology ELSE

    Returns:
        topo_spec (torch.Tensor[float]): The class label of each chain's topology
    '''

    topo_ELSE = len(topo_spec_choices)

    topo_spec = torch.full((indep.length(),), torch.nan)

    # Look at each chain independently
    for chain_mask in indep.chain_masks():

        # Get the chain
        chain_mask = torch.tensor(chain_mask)
        indep_chain, _ = aa_model.slice_indep(indep, chain_mask, break_chirals=True)

        # I'm worried about small molecules somehow causing helix breaks or something
        if indep_chain.is_sm.any():
            continue

        # Find the secondary structural elements
        dssp, _ = structure.get_dssp(indep_chain)
        segments = sec_struct_adj.ss_to_segments(dssp, is_dssp=True)
        cas = indep_chain.xyz[:,1]

        new_segments = merge_barely_broken_ss(segments, cas, segment_join_angle=45)

        # Gather the condensed topology string
        topo_string = ''
        assign_else = False
        for (typ, start, end) in new_segments:
            if str(typ) not in 'HE':
                continue
            length = end - start + 1
            if typ == 'H' and length < min_helix_length:
                assign_else = True
                break
            topo_string += typ

        # Figure out if it's one of the ones we care about
        assign_else = assign_else or topo_string not in topo_spec_choices
        topo_class = topo_ELSE if assign_else else topo_spec_choices.index(topo_string)

        topo_spec[chain_mask] = topo_class

    return topo_spec





def maybe_chain_mask(arr, indep, p_chain_mask, chain_shown_prob=0.5, store=np.nan):
    '''
    A masking function. Maybe do nothing, maybe mask some chains

    If masking chains, maybe mask each chain

    Args:
        arr (torch.Tensor): The array to maybe mask
        indep (indep): indep
        p_chain_mask (float): Probability this function does anything
        chain_shown_prob (float): If we are masking, probability each chain is not masked
        store (arr.dtype): Value to store if masked

    Returns:
        arr (torch.Tensor): The array but masked

    '''
    if torch.rand(1) < p_chain_mask:

        chain_masks = indep.chain_masks()
        chain_shown = torch.rand(len(chain_masks)) < chain_shown_prob

        for mask, shown in zip(chain_masks, chain_shown):
            if shown:
                continue
            arr[torch.tensor(mask)] = store

    return arr


def window_smooth(arr, size):
    '''
    Perform a rolling mean of a vector

    Args:
        arr (torch.Tensor): Array to smooth
        size (size): The full size of the rolling window

    Returns:
        out_arr (torch.Tensor): The array but with rolling mean smoothing
    '''

    # Ignore nan positions
    valid_mask = ~torch.isnan(arr)
    valid_arr = arr[valid_mask]
    valid_out_arr = torch.zeros(len(valid_arr))

    look = int(size / 2)

    # Perform the rolling mean over the valid positions
    for i in range(len(valid_arr)):
        lb = max(i - look, 0)
        ub = min(i + look, len(valid_arr)-1)

        valid_out_arr[i] = torch.mean(valid_arr[lb:ub+1])

    # Only store the valid positions
    out_arr = torch.full((len(arr),), np.nan)
    out_arr[valid_mask] = valid_out_arr

    return out_arr


def add_gaussian_noise(arr, std, clip_low=None, clip_high=None):
    '''
    Add gaussian noise to a vector

    Args:
        arr (torch.Tensor[float]): arr [L]
        std (float): Standard deviation of the gaussian noise
        clip_low (float or None): Lowest value that makes sense to output (or None)
        clip_high (float or None): Highest value that makes sense to output (or None)

    Returns:
        out_arr (torch.Tensor[float]): Arr but with gaussian noise added
    '''

    valid_mask = ~torch.isnan(arr)
    valid_arr = arr[valid_mask]

    values = torch.normal(torch.zeros(len(valid_arr)), std)

    new_arr = torch.clip( valid_arr + values, clip_low, clip_high )

    out_arr = torch.full((len(arr),), np.nan)
    out_arr[valid_mask] = new_arr

    return out_arr

class AddIdealSSTrainingTransform:
    '''
    A condition family that allows one to try to make their outputs more ideal
    '''

    def __init__(self, p_ideal_ss=0, p_loop_frac=0, p_avg_scn=0.0, p_topo_spec=0.0, p_chain_mask=0.5, p_ideal_speckle=0.2,
            ideal_smooth_window=9, ideal_gaussian_std=0.2, scn_min_value=None, scn_max_value=None, scn_per_res=True, scn_smooth_window=9, scn_gaussian_std=0.3,
            topo_spec_choices=None, topo_spec_min_helix_length=8):
        '''
        
        Args:
            p_ideal_ss (float): Probability that we show the ideal_ss condition
            p_loop_frac (float): Probability that we show the loop_frac condition
            p_avg_scn (float): Probability that we show the sidechain neighbors condition
            p_topo_spec (float): Probability we tell the model what the topology is
            p_chain_mask (float): Probability that we do any chain masking
            p_ideal_speckle (float): Probability that we randomly hide some fraction of the ideal_ss condition
            ideal_smooth_window (float): Rolling mean window size to smooth ideal_ss
            ideal_gaussian_std (float): Gaussian std for adding noise to ideal_ss
            scn_min_value (float): The value below which we call sidechain neighbors 0
            scn_max_value (float): The value beyond which we call sidechain neighbors 1
            scn_per_res (bool): If showing scn, 50% of the time store the exact values
            scn_smooth_window (int): If storing scn per res, smooth with this rolling window size
            scn_gaussian_std (float): If storing scn per res, add gaussian noise with this std
            topo_spec_choices (list[str]): The choices for topology in topo spec
            topo_spec_min_helix_length (int): If a helix exists in the structure with length less than this, then immeidately assign ELSE
        '''

        if p_avg_scn > 0:
            assert scn_min_value is not None, 'upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_min_value must be specified in yaml. A good choice is: 1.5'
            assert scn_max_value is not None, 'upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_max_value must be specified in yaml. A good choice is: 2.5'

        if p_topo_spec > 0:
            assert topo_spec_choices is not None, "upstream_training_transforms.configs.AddIdealSSTrainingTransform.topo_spec_choices must be specified in yaml. A good choice is: [HH, 'HHH', 'HHHH', 'HHHHH']"

        self.p_ideal_ss = p_ideal_ss
        self.p_loop_frac = p_loop_frac
        self.p_avg_scn = p_avg_scn
        self.p_topo_spec = p_topo_spec
        self.p_chain_mask = p_chain_mask
        self.p_ideal_speckle = p_ideal_speckle
        self.ideal_smooth_window = ideal_smooth_window
        self.ideal_gaussian_std = ideal_gaussian_std
        self.scn_min_value = scn_min_value
        self.scn_max_value = scn_max_value
        self.scn_per_res = scn_per_res
        self.scn_smooth_window = scn_smooth_window
        self.scn_gaussian_std = scn_gaussian_std
        self.topo_spec_choices = topo_spec_choices
        self.topo_spec_min_helix_length = topo_spec_min_helix_length


    def __call__(self, indep, conditions_dict, **kwargs):

        # Are we going to do loop_frac?
        if torch.rand(1) < self.p_loop_frac:
            loop_frac = get_loop_frac_by_chain(indep)

            loop_frac = maybe_chain_mask(loop_frac, indep, self.p_chain_mask)

            loop_frac[indep.is_sm] = np.nan
            conditions_dict['loop_frac'] = loop_frac

        # Are we going to do sidechain neighbors
        if torch.rand(1) < self.p_avg_scn:

            scn_range = self.scn_max_value - self.scn_min_value

            # If we're doing scn_per_res there's a 50% chance we either do it per res or by chain
            if self.scn_per_res and torch.rand(1) < 0.5:
                avg_scn = get_scn_by_chain(indep)
                avg_scn = window_smooth(avg_scn, self.scn_smooth_window)

                # The gaussian noise is relative to the 0-1 final range, so multiply but the current scn range
                avg_scn = add_gaussian_noise(avg_scn, self.scn_gaussian_std*scn_range, self.scn_min_value, self.scn_max_value)
            else:
                avg_scn = get_avg_scn_by_chain(indep)

            valid_mask = ~torch.isnan(avg_scn)

            # Convert scn into our valid range
            scn_valid_0_to_1 = (avg_scn[valid_mask] -  self.scn_min_value) / scn_range
            avg_scn[valid_mask] = torch.clip(scn_valid_0_to_1, 0, 1)

            avg_scn = maybe_chain_mask(avg_scn, indep, self.p_chain_mask)

            avg_scn[indep.is_sm] = np.nan
            conditions_dict['avg_scn'] = avg_scn

        # Are we doing to do ideal_ss?
        if torch.rand(1) < self.p_ideal_ss:

            ideal_ss = get_ideal_ss_score(indep)

            ideal_ss = window_smooth(ideal_ss, self.ideal_smooth_window)
            ideal_ss = add_gaussian_noise(ideal_ss, self.ideal_gaussian_std, 0, 1)

            ideal_ss = maybe_chain_mask(ideal_ss, indep, self.p_chain_mask)

            # Are we going to randomly remove the values for amino acids?
            if torch.rand(1) < self.p_ideal_speckle:
                shown_frac = torch.rand(1)
                is_shown = torch.rand(len(ideal_ss)) < shown_frac
                ideal_ss[~is_shown] = np.nan

            ideal_ss[indep.is_sm] = np.nan
            conditions_dict['ideal_ss'] = ideal_ss


        # Are we going to do topo spec?
        if torch.rand(1) < self.p_topo_spec:

            assert self.topo_spec_choices is not None and len(self.topo_spec_choices) > 0

            topo_spec = assign_topo_spec(indep, self.topo_spec_choices, min_helix_length=self.topo_spec_min_helix_length)
            topo_spec = maybe_chain_mask(topo_spec, indep, self.p_chain_mask)

            topo_spec[indep.is_sm] = np.nan
            conditions_dict['topo_spec'] = topo_spec


        return kwargs | dict(
            indep=indep,
            conditions_dict=conditions_dict
        )


class AddIdealSSInferenceTransform:
    '''
    The inference half of ideal_ss_cond
    '''

    def __init__(self, only_first_chain=True):
        '''
        Args:
            only_first_chain (bool): Only mark ideal_ss_cond values for the first chain
        '''
        self.only_first_chain = only_first_chain


    def __call__(self, indep, conditions_dict, conf, **kwargs):

        assert 'AddIdealSSTrainingTransform' in conf.upstream_training_transforms.names, 'Model not set up for ideal_ss'

        # Has the user specified the ideal_ss flag?
        if conf.ideal_ss.ideal_value is not None:
            assert conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.get('p_ideal_ss', 0) > 0, 'Model not trained for ideal_ss.ideal_value'

            # Fill in the ideal_ss tensor
            value = conf.ideal_ss.ideal_value
            std = conf.ideal_ss.ideal_std

            ideal_ss = torch.full((indep.length(),), value)
            if std > 0:
                ideal_ss = add_gaussian_noise(ideal_ss, std, 0, 1)

            if self.only_first_chain:
                ideal_ss[ ~torch.tensor(indep.chain_masks()[0]) ] = np.nan

            ideal_ss[indep.is_sm] = np.nan
            conditions_dict['ideal_ss'] = ideal_ss

        # Has the user specified the avg_scn flag?
        if conf.ideal_ss.avg_scn is not None:
            assert conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.get('p_avg_scn', 0) > 0, 'Model not trained for ideal_ss.avg_scn'

            # Fill in the avg scn tensor
            value = conf.ideal_ss.avg_scn
            std = conf.ideal_ss.scn_std

            min_value = conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_min_value
            max_value = conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_max_value

            assert value <= max_value, (f'ideal_ss.avg_scn error: You specified {value} but the model was trained such'
                f' that anything above {max_value} was clipped to {max_value}. Please instead specify {max_value}')
            assert value >= min_value, (f'ideal_ss.avg_scn error: You specified {value} but the model was trained such'
                f' that anything below {min_value} was clipped to {min_value}. Please instead specify {min_value}')

            avg_scn = torch.full((indep.length(),), value)
            if std > 0:
                avg_scn = add_gaussian_noise(avg_scn, std, min_value, max_value)

            avg_scn_0_1 = (avg_scn - min_value) / (max_value - min_value)

            if self.only_first_chain:
                avg_scn_0_1[ ~torch.tensor(indep.chain_masks()[0]) ] = np.nan

            avg_scn_0_1[indep.is_sm] = np.nan
            conditions_dict['avg_scn'] = avg_scn_0_1

        # Has the user specified the loop_frac flag?
        if conf.ideal_ss.loop_frac is not None:
            assert conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.get('p_loop_frac', 0) > 0, 'Model not trained for ideal_ss.loop_frac'

            # Fill in the loop_frac tensor
            value = conf.ideal_ss.loop_frac

            loop_frac = torch.full((indep.length(),), value)

            if self.only_first_chain:
                loop_frac[ ~torch.tensor(indep.chain_masks()[0]) ] = np.nan

            loop_frac[indep.is_sm] = np.nan
            conditions_dict['loop_frac'] = loop_frac


        if conf.ideal_ss.topo_spec is not None:
            assert conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.get('p_topo_spec', 0) > 0, 'Model not trained for ideal_ss.topo_spec'

            # First we have to convert whatever the user passed into a usble format
            assert isinstance(conf.ideal_ss.topo_spec, Mapping), ('ideal_ss.topo_spec should be a dictionary. Instead it was '
                                    f'{type(conf.ideal_ss.topo_spec)}. {conf.ideal_ss.topo_spec}')

            assert 'AddIdealSSTrainingTransform' in conf.upstream_training_transforms.configs, "This model can't do ideal_ss" 
            assert 'topo_spec_choices' in conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform, "This model can't do ideal_ss.topo_spec"

            # ELSE is handled gracefully by the model. None we have to convert to nan manually
            topo_choices = list(conf.upstream_training_transforms.configs.AddIdealSSTrainingTransform.topo_spec_choices)
            # topo_ELSE = len(topo_choices)
            topo_choices.append('ELSE')
            topo_None = len(topo_choices)
            topo_choices.append('None')

            # Convert to numeric topologies
            topo_spec_keys = []
            topo_spec_probs = []
            for key, prob in conf.ideal_ss.topo_spec.items():
                assert key in topo_choices, (f'ideal_ss.topo_spec error: This model was not trained to recognize {key}. Your choices are: {topo_choices}')
                topo_spec_keys.append(topo_choices.index(key))
                topo_spec_probs.append(prob)

            topo_spec_probs = torch.tensor(topo_spec_probs)
            assert torch.sum(topo_spec_probs) == 1, f'Your ideal_ss.topo_spec probabilities did not sum to 1: {conf.ideal_ss.topo_spec}'

            # Assign each chain independently
            topo_spec = torch.full((indep.length(),), torch.nan)
            for i_chain, chain_mask in enumerate(indep.chain_masks()):
                if self.only_first_chain and i_chain != 0:
                    continue

                # small molecules were set to NaN during training so they are set the same here
                chain_mask[indep.is_sm] = False

                # Randomly select a topology based on the user probabilities
                i_choice = np.random.choice(np.arange(len(topo_spec_probs)), p=topo_spec_probs)

                # Translate the index to the numerical topology
                topo_key = topo_spec_keys[i_choice]

                # topo_None == torch.nan correction
                if topo_key == topo_None:
                    topo_key = torch.nan
                topo_spec[chain_mask] = topo_key

            conditions_dict['topo_spec'] = topo_spec


        return kwargs | dict(
            indep=indep,
            conf=conf,
            conditions_dict=conditions_dict
        )



def get_ideal_ss_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    '''
    See get_ideal_ss_conditioning()
    '''
    return get_ideal_ss_conditioning(indep, feature_conf, **kwargs)

def get_ideal_ss_conditioning(indep, feature_conf, ideal_ss=None, avg_scn=None, loop_frac=None, topo_spec=None, **kwargs):
    '''
    Generates the ideal ss conditioning extra t1d

    Args:
        indep (Indep): indep
        feature_conf (OmegaConf): The configuration for this feature
        ideal_ss (torch.Tensor[float] or None): A number 0 to 1 describing how ideal this piece of secondary structure is
        avg_scn (torch.Tensor[float] or None): A number 0 to 1 describing the sidechain neighbors of this piece of protein
        loop_frac (torch.Tensor[float] or None): A number 0 to 1 describing fraction of this chain that is loop by dssp
        topo_spec (torch.Tensor[int] or None): A class label of which of the topo_spec_choices this chain fits into

    Returns:
        dict:
            t1d (torch.Tensor[bool]): The extra t1d [L, 6 + (N_topos)]
    '''

    assert 'topo_spec_choices' in feature_conf, ('You have to also tell ideal_ss_cond.topo_spec_choices about '
                                                    '${upstream_training_transforms.configs.AddIdealSSTrainingTransform.topo_spec_choices}')
    if feature_conf['topo_spec_choices'] is None:
        N_topos = 0
    else:
        N_topos = len(feature_conf['topo_spec_choices']) + 1

    if ideal_ss is None:
        ideal_ss = torch.full((indep.length(),), np.nan)
    if avg_scn is None:
        avg_scn = torch.full((indep.length(),), np.nan)
    if loop_frac is None:
        loop_frac = torch.full((indep.length(),), np.nan)
    if topo_spec is None:
        topo_spec = torch.full((indep.length(),), np.nan)


    assert len(ideal_ss) == indep.length(), 'ideal_ss vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(avg_scn) == indep.length(), 'avg_scn vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(loop_frac) == indep.length(), 'loop_frac vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(topo_spec) == indep.length(), 'topo_spec vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'

    # convert nan to 0
    ideal_ss_mask = ~torch.isnan(ideal_ss)
    ideal_ss[~ideal_ss_mask] = 0

    avg_scn_mask = ~torch.isnan(avg_scn)
    avg_scn[~avg_scn_mask] = 0

    loop_frac_mask = ~torch.isnan(loop_frac)
    loop_frac[~loop_frac_mask] = 0

    extra_t1d = torch.stack((ideal_ss_mask, ideal_ss, avg_scn_mask, avg_scn, loop_frac_mask, loop_frac), axis=-1)

    if N_topos > 0:
        # nans will have 0 in every column. So assign them ELSE + 1 then slice that away
        topo_spec_nan = torch.isnan(topo_spec)
        topo_spec[topo_spec_nan] = N_topos
        one_hot_topo_spec = torch.nn.functional.one_hot(topo_spec.long(), num_classes=N_topos+1).float()[:,:N_topos]

        extra_t1d = torch.cat((extra_t1d, one_hot_topo_spec), axis=-1)

    return {'t1d':extra_t1d}



