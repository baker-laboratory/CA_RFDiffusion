import torch


def expand_1d_atomized_ok_gp_not(indep, input_mask, post_idx_from_pre_idx, null_value=0, key='unknown'):
    '''
    For use with ExpandConditionsDict

    Expands a 1d mask (of any type) to the full length of the transformed indep
    Atomized copies of the original residues get the same value as the original
    is_gp copies of the original residues get null_value

    Args:
        indep (Indep): indep
        input_mask (torch.Tensor): The mask from the pre-transformed indep that we are expanding [L pre-transform]
        post_idx_from_pre_idx (list[list[int]]): Mapping from pre-transform to post-transform [L pre-transform]
        null_value (any): The value to store if input_mask is invalid for this position
        key (str): The name of this value in conditions_dict (for error messages)

    Returns:
        new_mask (torch.Tensor): The input_mask but expanded to the post-transformed indep [L post-transform]
    '''
    assert len(input_mask) == len(post_idx_from_pre_idx), (f'{key} is not the same length as pre-transform_indep indep. This probably happened'
        " because one of the transforms cropped the indep but didn't call conditions.util.pop_conditions_dict()")
    new_mask = torch.full((indep.length(),), null_value, dtype=input_mask.dtype)
    for pre_idx, post_idxs in enumerate(post_idx_from_pre_idx):
        new_mask[post_idxs] = input_mask[pre_idx]
    new_mask[indep.is_gp] = null_value
    return new_mask


def expand_2d_atomized_ok_gp_not(indep, input_mask, post_idx_from_pre_idx, null_value=0, key='unknown'):
    '''
    For use with ExpandConditionsDict

    Expands a 2d mask (of any type) to the full length of the transformed indep
    Atomized copies of the original residues maintain their interactions as in the original
    All is_gp residues have all interactions set to null_value

    Args:
        indep (Indep): indep
        input_mask (torch.Tensor): The mask from the pre-transformed indep that we are expanding [L pre-transform, L pre-transform]
        post_idx_from_pre_idx (list[list[int]]): Mapping from pre-transform to post-transform [L pre-transform]
        null_value (any): The value to store if input_mask is invalid for this position
        key (str): The name of this value in conditions_dict (for error messages)

    Returns:
        new_mask (torch.Tensor): The input_mask but expanded to the post-transformed indep [L post-transform, L post-transform]
    '''
    assert input_mask.shape[0] == len(post_idx_from_pre_idx), (f'{key} is not the same length as pre-transform_indep indep. This probably happened'
        " because one of the transforms cropped the indep but didn't call conditions.util.pop_conditions_dict()")
    assert input_mask.shape[1] == len(post_idx_from_pre_idx), (f'{key} is not the same length as pre-transform_indep indep. This probably happened'
        " because one of the transforms cropped the indep but didn't call conditions.util.pop_conditions_dict()")
    new_mask = torch.full((indep.length(),indep.length(),), null_value, dtype=input_mask.dtype)

    post_idx_from_pre_idx_torch = [torch.tensor(x) for x in post_idx_from_pre_idx]

    # No good way to do this besides all-by-all
    for pre_idx_a, post_idxs_a in enumerate(post_idx_from_pre_idx_torch):
        for pre_idx_b, post_idxs_b in enumerate(post_idx_from_pre_idx_torch):
            # The singular value from pre gets expanded to the rectangular-matrix from post
            new_mask[post_idxs_a[:,None],post_idxs_b[None,:]] = input_mask[pre_idx_a,pre_idx_b]

    # Mask out any interaction with a gp residue
    new_mask[indep.is_gp,:] = null_value
    new_mask[:,indep.is_gp] = null_value

    return new_mask


def pop_conditions_dict(conditions_dict, pop):
    '''
    Like pop_indep(), this function makes conditions dict smaller by removing token positions

    Use whenever you crop indep in transforms

    Args:
        conditions_dict (dict[string, Any]): Conditions dict
        pop (torch.Tensor[bool]): The positions to keep

    Returns:
        None
    '''

    for key in conditions_dict:

        if key == 'ss_adj':
            conditions_dict[key] = conditions_dict[key].pop_mask(pop)

        elif key == 'hbond_satisfaction':
            conditions_dict[key] = conditions_dict[key].pop_mask(pop)

        else:
            # We'll try to make a catch-all for people so they don't have to update this function
            # Really, the only important things to update are tensors and numpy arrays since anything else
            #  would require a special function

            # Not sure what to do about 2d arrays. These might have to be special cases (because we can't differentiate them on really small indeps)
            val = conditions_dict[key]

            if hasattr(val, 'shape') and len(val.shape) > 0:

                assert len(val) == len(pop), (f'len(conditions_dict[{key}]) == {len(val)} but len(pop) == {len(pop)}.'
                                        " There's probably an indep crop somewhere that doesn't call conditions.util.pop_conditions_dict()" )

                conditions_dict[key] = val[pop]
            else:
                continue

