import torch
import random
import numpy as np
import rf_diffusion.structure as structure
from rf2aa.kinematics import generate_Cbeta
import rf_diffusion.conditions.ss_adj.sec_struct_adjacency as sec_struct_adjacency
from rf_diffusion.train_data.exceptions import NextExampleException
from rf_diffusion import aa_model
from rf_diffusion import sasa
from rf_diffusion.conditions.util import pop_conditions_dict

def Cb_or_atom(indep, xyz=None):
    '''
    Generates the virtual CB atom for protein residues and returns atom for small molecule atoms

    Args:
        indep (indep): indep or None
        xyz (torch.Tensor[float]): xyz if indep is None. Assumes no small molecules

    Returns:
        Cb (torch.Tensor): The xyz of virtual Cb atom for protein atoms and atom for small molecule atoms [L,3] 
    '''
    if xyz is None:
        xyz = indep.xyz
        is_sm = indep.is_sm
    else:
        assert indep is None, 'If you are going to specify xyz, indep must be None'
        is_sm = torch.zeros(xyz.shape[0], dtype=bool)
    N = xyz[~is_sm,0]
    Ca = xyz[~is_sm,1]
    C = xyz[~is_sm,2]

    # small molecules will use atom 1 for neighbor calculations
    Cb = xyz[:,1].clone()
    Cb[~is_sm] = generate_Cbeta(N,Ca,C)
    return Cb

def torch_rand_choice_noreplace(arr, n):
    '''
    Exactly the same as np.random.choice(arr, n, replace=False)

    Args:
        arr (torch.Tensor): The array
        n (int): How many of the array you want
    '''
    return arr[torch.randperm(len(arr))[:n]]

def downsample_bool_mask(mask, n_to_keep=None, max_frac_to_keep=None):
    '''
    Given a bool mask, return another bool mask with only a subsample of the True positions

    Only specify one of n_to_keep or max_frac_to_keep

    Args:
        mask (torch.Tensor[bool]): The mask you wish to downsample
        n_to_keep (int): How many True should remain in the end
        max_frac_to_keep (float): Between 0 and max_frac_to_keep * mask.sum() True wil remain in the end
    '''
    assert not (n_to_keep is None and max_frac_to_keep is None), 'You have to specify one of them'
    assert not (n_to_keep is not None and max_frac_to_keep is not None), "You can't specify both of them"

    total = mask.sum()
    if n_to_keep is None:
        n_to_keep = random.randint(0, int(torch.ceil( total * max_frac_to_keep )))

    # Take a subset of the True positions
    wh_full = torch.where(mask)[0]
    wh = torch_rand_choice_noreplace(wh_full, n_to_keep)

    # Fill the new output mask
    mask_out = torch.zeros(len(mask), dtype=bool)
    mask_out[wh] = True

    return mask_out

def find_hotspots_antihotspots(indep, dist_cutoff=10, only_of_first_chain=False):
    '''
    Finds all possible hotspots and antihotspots for training
        A hotspot is any motif residue within 10A (CB-dist) of a residue on a different chain.
        An antihotspot is same the criteria but greater than 10A from all residues.

    Args:
        indep (Indep): An indep
        dist_cutoff (float): What is the distance cutoff for hotspots (you can't change this without retraining)
        only_of_first_chain (bool): If true, hotspots will only be considered if the first chain is near them

    Returns:
        is_hotspot (torch.Tensor[bool]): Is this position a hotspot [L]
        is_antihotspot (torch.Tensor[bool]): Is this position a hotspot [L]

    '''
    Cb = Cb_or_atom(indep)

    d2_map = torch.sum( torch.square( Cb[:,None] - Cb[None,:] ), axis=-1 )
    are_close = d2_map <= dist_cutoff**2


    # Can these pairs be hotspot/antihotspot?
    hotspot_mask = are_close & ~indep.same_chain
    antihotspot_mask = ~are_close & ~indep.same_chain

    is_hotspot_full = hotspot_mask.any(axis=-1)
    # antihotspots additionally need to be not hotspots
    #  (just because you're 10A away from 1 residue doesn't mean your 10A away from all of them)
    is_antihotspot_full = antihotspot_mask.any(axis=-1) & ~is_hotspot_full

    # Do this separate so we don't mess up antihotspots
    if only_of_first_chain:
        first_chain_mask = torch.tensor(indep.chain_masks()[0])

        # Get the distances to the first chain
        # [binderpos, full indep]
        d2_first_chain = d2_map[first_chain_mask]
        are_close_first = d2_first_chain <= dist_cutoff**2

        # Set all binder positions such that they are not close (can't be hotspot of self)
        are_close_first[:,first_chain_mask] = False

        # Run any on axis 0 to say "Is full indep position near any binder positions"
        new_is_hotspot_full = are_close_first.any(axis=0)

        # The new set of hotspots should be a subset of the old one
        if new_is_hotspot_full.sum() > 0:
            assert is_hotspot_full[new_is_hotspot_full].all()
            assert not new_is_hotspot_full[first_chain_mask].any()

        is_hotspot_full = new_is_hotspot_full

    return is_hotspot_full, is_antihotspot_full


def radial_crop(indep, is_diffused, is_hotspot, is_target, distance=25):
    '''
    Crop anything on is_target that isn't within distance(=25) of the hotspots

    Args:
        indep (Indep): indep
        is_diffused (torch.Tensor[bool]): Whether or not this residue is diffused (these won't be cropped) [L]
        is_hotspot (torch.Tensor[bool]): Is this residue a hotspot [L]
        is_target (torch.Tensor[bool]): Is this residue part of the target [L]
        distance (float): Distance from hotspot residues beyond which target residues are removed

    Returns:
        torch.Tensor[bool]: Which residues should remain after cropping [L]
    '''

    if is_hotspot.sum() == 0:
        print("Warning! radial_crop didn't receive any hotspot residues. Not cropping")
        return torch.ones(indep.length(), dtype=bool)

    Ca = indep.xyz[:,1]

    hotspot_Ca = Ca[is_hotspot]

    d2_to_hotspot = torch.sum( torch.square( hotspot_Ca[None,:] - Ca[:,None] ), axis=-1).min( axis=-1 ).values
    close_enough = d2_to_hotspot < distance**2

    crop_residues = is_target & ~close_enough
    assert not (crop_residues & is_hotspot).any()
    assert not (crop_residues & is_diffused).any()

    return ~crop_residues


def random_unit_vector():
    '''
    Returns a unit vector uniformly oriented in any direction

    Returns:
        torch.Tensor: Unit vector [3]
    '''
    unit = torch.normal( torch.zeros(3), torch.ones(3) )
    unit /= torch.linalg.norm(unit)
    return unit


def planar_crop(indep, is_diffused, is_hotspot, is_target, distance=10):
    '''
    Crops everything extending beyone a plane from the hotspots. Similar to shift-highlighting in pymol
        The plane orientation is randomly chosen and must remove at least 1 target residue
        This method gives up after 100 tries if it can't find a valid plane

    Args:
        indep (Indep): indep
        is_diffused (torch.Tensor[bool]): Whether or not this residue is diffused (these won't be cropped) [L]
        is_hotspot (torch.Tensor[bool]): Is this residue a hotspot [L]
        is_target (torch.Tensor[bool]): Is this residue part of the target [L]
        distance (float): How far away from the plane the closest hotspot can be

    Returns:
        torch.Tensor[bool]: Which residues should remain after cropping [L]
    '''

    if is_hotspot.sum() == 0:
        print("Warning! planar_crop didn't receive any hotspot residues. Not cropping")
        return torch.ones(indep.length(), dtype=bool)

    Ca = indep.xyz[:,1]
    hotspot_Ca = Ca[is_hotspot]
    target_Ca = Ca[is_target]

    use_upper_crop = None

    # Since we're picking random unit vectors. It's possible some won't work
    #  Try 100 times and if that doesn't work give up
    for attempt in range(100):

        unit = random_unit_vector()

        # t is how far along the unit the points are
        #  if you multiplied t * unit you'd get the projection
        hotspot_projection_t = (hotspot_Ca * unit[None]).sum(axis=-1)
        target_projection_t = (target_Ca * unit[None]).sum(axis=-1)

        upper_t = hotspot_projection_t.max() + distance
        lower_t = hotspot_projection_t.min() - distance

        cropped_upper = (target_projection_t > upper_t).sum()
        cropped_lower = (target_projection_t < lower_t).sum()

        if cropped_upper > 0 or cropped_lower > 0:
            use_upper_crop = cropped_upper > cropped_lower
            break

    if use_upper_crop is None:
        # couldn't find a crop, just dont crop (because the hotspots might envelop the target or something)
        print("Warning! Couldn't find a planar crop!")
        return torch.ones(indep.length(), dtype=bool)


    projection_t = (Ca * unit[None]).sum(axis=-1)

    if use_upper_crop:
        potentially_cropped = projection_t > upper_t
    else:
        potentially_cropped = projection_t < lower_t

    crop_residues = potentially_cropped & is_target

    assert not (crop_residues & is_hotspot).any()
    assert not (crop_residues & is_diffused).any()

    return ~crop_residues



def decide_target(indep, use_first_chain=False):
    '''
    Decides which chain/s shall be the target for PPI training examples.

    The binder must be a single chain but the target can be arbitrarily many chains

    Args:
        indep (Indep): indep

    Returns:
        is_target (torch.Tensor[bool]): Which residues are the target [L]
    '''

    if indep.same_chain.all():
        return None

    chain_masks = torch.tensor(np.array(indep.chain_masks()))

    chain_is_elgible_for_binder = torch.zeros(len(chain_masks), dtype=bool)

    for imask, chain_mask in enumerate(chain_masks):
        
        # They say we can diffuse small molecules
        # if (chain_mask & indep.is_sm).any():
        #     continue
        chain_is_elgible_for_binder[imask] = True

    if chain_is_elgible_for_binder.sum() == 0:
        return None

    wh_elgible = torch.where(chain_is_elgible_for_binder)[0]
    i_wh = random.randint(0, len(wh_elgible)-1)
    if use_first_chain:
        i_wh = 0

    i_binder_chain = wh_elgible[i_wh]

    # the opposite of the binder is the target
    return ~chain_masks[i_binder_chain]



def training_extract_ppi_motifs(indep, is_target, max_frac_ppi_motifs=0.8, max_ppi_motif_trim_frac=0.4, dist=8):
    '''
    Simulates a motif-graft case
        First we extract all ss elements that have a CB within 8Å of target
        Then we randomly trim them and delete some

    Args:
        indep (Indep): indep
        is_target (torch.Tensor[bool]): Which residues are the target [L]
        max_frac_ppi_motifs (float): What's the max fraction of motif chunks should be kept?
        max_ppi_motif_trim_frac (float): Whats the max fraction of a motif that should be trimmed away?
        dist (float): How close must a single CB atom on the motif be to count as a motif?

    Returns:
        is_ppi_motif (torch.Tensor[bool]): Is this residue part of a PPI motif?
    '''

    is_binder = ~is_target

    Cb = Cb_or_atom(indep)
    full_dssp, _ = structure.get_dssp(indep)

    binder_Cb = Cb[is_binder]
    target_Cb = Cb[is_target]

    binder_dssp = full_dssp[is_binder]

    # Segments are all the secondary structral elements
    segments = sec_struct_adjacency.ss_to_segments(binder_dssp, is_dssp=True)

    res_close_enough = torch.sum( torch.square( binder_Cb[:,None] - target_Cb[None,:] ), axis=-1) < dist**2

    # Go through segments and store them as motifs if any of the Cbs in that segment are within 8A of a Cb on the target
    motifs = []
    for typ, start, end in segments:
        if typ == structure.ELSE:    # not small molecule motifs for now
            continue
        if res_close_enough[start:end+1].any():
            motifs.append((typ, start, end))

    # Downsample the motifs
    n_keep = random.randint(min(1, len(motifs)), int(np.ceil(len(motifs)*max_frac_ppi_motifs)) )
    keep_idx = torch.randperm(len(motifs))[:n_keep]

    # Work through the motifs that we're keeping and trim from both N and C termini
    final_motifs = []
    for idx in keep_idx:
        typ, start, end = motifs[idx]

        length = end - start + 1
        n_trim_start = int(random.uniform(0, max_ppi_motif_trim_frac) * length)
        n_trim_end = int(random.uniform(0, max_ppi_motif_trim_frac) * length)

        start = start + n_trim_start
        end = end - n_trim_end
        # if we've trimmed to nothing, then convert to 1aa motif
        if start > end:
            start = end
        
        final_motifs.append((typ, start, end))

    # Store the motifs into a is_ppi_motif mask
    is_ppi_motif = torch.zeros(indep.length(), dtype=bool)
    wh_binder = torch.where(is_binder)[0]

    for tp, start, end in final_motifs:
        motif_indices = wh_binder[start:end+1]
        is_ppi_motif[motif_indices] = True

    return is_ppi_motif


# rosetta/main/source/src/core/select/util/SelectResiduesByLayer.cc
def sidechain_neighbors(binder_Ca, binder_Cb, else_Ca):

    conevect = binder_Cb - binder_Ca
    conevect /= torch.sqrt(torch.sum(torch.square(conevect), axis=-1))[:,None]

    vect = else_Ca[:,None] - binder_Cb[None,:]
    vect_lengths = torch.sqrt(torch.sum(torch.square(vect), axis=-1))
    vect_normalized = vect / vect_lengths[:,:,None]

    dist_term = 1 / ( 1 + torch.exp( vect_lengths - 9  ) )

    angle_term = (((conevect[None,:] * vect_normalized).sum(axis=-1) + 0.5) / 1.5).clip(0, None)

    sc_neigh = (dist_term * np.square( angle_term )).sum(axis=0)

    return sc_neigh




class PPITrimTailsChain0ComplexTransform:
    '''
    A transform that trims long disordered tails from chain0 in training examples
    Should be called before GenerateMasks

    Args:
        operate_on_datasets (list[str]): Which datasets should this be applied to?
        scn_unfolded_thresh (float): SideChainNeighbor threshold for something to be considered folded
        can_remove_hotspots (bool): Can disordered parts be removed even if they are touching the target?
        all_but_1_cutoff (int): An nmer must contain at least this many residues to be eliminated if all but 1 is unfolded
        all_but_2_cutoff (int): An nmer must contain at least this many residues to be eliminated if all but 2 are unfolded
        nmer (int): Look through the tails in chunks of this size for stretches of nearly-entirely unfolded residues
        verbose (bool): Print what this Transform does
    '''

    def __init__(self,
        operate_on_datasets=['all'],
        scn_unfolded_thresh=1.0,
        can_remove_hotspots=True,
        all_but_1_cutoff=4,
        all_but_2_cutoff=8,
        n_mer=9,
        verbose=True
    ):
        self.operate_on_datasets = operate_on_datasets
        self.scn_unfolded_thresh = scn_unfolded_thresh
        self.can_remove_hotspots = can_remove_hotspots
        self.all_but_1_cutoff = all_but_1_cutoff
        self.all_but_2_cutoff = all_but_2_cutoff
        self.n_mer = n_mer
        self.verbose = verbose


    def __call__(self, indep, atom_mask, chosen_dataset, metadata, conditions_dict, **kwargs):

        # The input arguments unchanged if we decide not to do anthing
        do_nothing_return = dict(indep=indep, atom_mask=atom_mask, chosen_dataset=chosen_dataset, metadata=metadata, conditions_dict=conditions_dict, **kwargs)

        # Only operate on the datasets we're told to
        if not ('all' in self.operate_on_datasets or chosen_dataset in self.operate_on_datasets):
            return do_nothing_return


        chain_masks = indep.chain_masks()
        if len(chain_masks) == 1:
            print('PPITrimTailsChain0ComplexTransform got passed a single chain')
            return do_nothing_return

        # Get the basic backbone data
        Ca = indep.xyz[:,1]
        Cb = Cb_or_atom(indep)
        is_binder = torch.tensor(chain_masks[0])
        binderlen = is_binder.sum()
        if not is_binder[:binderlen].any():
            raise NextExampleException("PPITrimTailsChain0ComplexTransform: Binder isn't contiguous") # need to see if this ever happens

        # Find sidechain neighbors and hotspots
        sc_neigh = sidechain_neighbors(Ca[is_binder], Cb[is_binder], Ca[is_binder])

        is_hotspots_both, _ = find_hotspots_antihotspots(indep, only_of_first_chain=False)
        hotspot_on_binder = is_hotspots_both & is_binder
        if hotspot_on_binder.sum() == 0:
            raise NextExampleException("PPITrimTailsChain0ComplexTransform: Chains aren't touching?")
        wh_hotspot_on_binder = torch.where(hotspot_on_binder)[0]

        # Mark trimming bounds
        first_hotspot = wh_hotspot_on_binder[0]
        last_hotspot = wh_hotspot_on_binder[-1]

        # Find unfolded residues
        is_unfolded = sc_neigh < self.scn_unfolded_thresh

        # If we can remove hotspots, we first start at the hotspot bounds and work into the binder
        if self.can_remove_hotspots:
            # Move the first hotspot into the binder
            for i_slice in range(first_hotspot-1, binderlen):
                all_unfolded = self.my_unfolded(is_unfolded[max(i_slice-self.n_mer+1, 0):i_slice+1])
                if not all_unfolded:
                    break
            if i_slice >= first_hotspot:
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Trimming {i_slice - first_hotspot + 1} residues into binder start')
            first_hotspot = i_slice + 1

            # Move the last hotspot into the binder
            for i_slice in range(last_hotspot+1, -1, -1):
                all_unfolded = self.my_unfolded(is_unfolded[i_slice:min(i_slice+self.n_mer, binderlen)])
                if not all_unfolded:
                    break
            if i_slice < last_hotspot:
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Trimming {last_hotspot - i_slice + 1} residues into binder end')
            last_hotspot = i_slice - 1


        if first_hotspot >= last_hotspot:
            raise NextExampleException("PPITrimTailsChain0ComplexTransform: The whole binder got trimmed!")


        # Starting from the first and last residues on the binder that are near the target
        #  Move outwards looking for the first and last n_mers that are totally unfolded
        keep_mask = torch.ones(indep.length(), dtype=bool)

        # i_slice is that last residue that will be removed
        for i_slice in range(first_hotspot-1, -1, -1):
            all_unfolded = self.my_unfolded( is_unfolded[max(i_slice-self.n_mer+1, 0):i_slice+1] )
            if all_unfolded:
                keep_mask[:i_slice+1] = False
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Removing first {i_slice+1} residues')
                break

        # i_slice is that first residue that will be removed
        for i_slice in range(last_hotspot+1, binderlen):
            all_unfolded = self.my_unfolded( is_unfolded[i_slice:min(i_slice+self.n_mer, binderlen)] )
            if all_unfolded:
                keep_mask[i_slice:binderlen] = False
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Removing last {binderlen-i_slice} residues')
                break

        # This implies a bug in this code
        assert (keep_mask[~is_binder]).all(), 'A target residue was going to get removed'

        if keep_mask.all():
            return do_nothing_return

        # Remove the residues we said we wanted to remove
        aa_model.pop_mask(indep, keep_mask)
        atom_mask = atom_mask[keep_mask]
        metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], keep_mask)
        pop_conditions_dict(conditions_dict, keep_mask)

        return dict(
            indep=indep,
            atom_mask=atom_mask,
            chosen_dataset=chosen_dataset,
            metadata=metadata,
            conditions_dict=conditions_dict,
            **kwargs
            )


    def my_unfolded(self, is_unfolded):
        '''
        Determine whether a stretch of residues counts as "totally unfolded"

        Args:
            is_unfolded (torch.Tensor[bool]): A stretch of residues with their folded states
        '''

        # Check how many are folded
        n_folded = (~is_unfolded).sum()
        total = len(is_unfolded)

        # If its a short segment, they all have to be folded
        if total <= self.all_but_1_cutoff:
            return n_folded <= 0

        # A little longer and 1 can be folded
        if total <= self.all_but_2_cutoff:
            return n_folded <= 1

        # Otherwise it's max length and up to 2 can be folded
        return n_folded <= 2



class PPIRejectUnfoldedInterfacesTransform:
    '''
    A transform that rejects training examples where the binder is too unfolded

    Uses Fraction Boundary by SideChainNeighbors as the metric

    '''

    def __init__(self, operate_on_datasets=['all'], binder_fbscn_cut=0.12, binder_fbscn_at_interface_cut=0.13, verbose=True):
        '''
        Args:
            operate_on_datasets (list[str]): Which datasets should this be applied to?
            binder_fbscn_cut (float): Fraction Boundary by SideChainNeighbor cut for the binder as a whole
            binder_fbscn_at_interface_cut (float): Fraction Boundary by SideChainNeighbor cut for the binder parts at the interface
            verbose (bool): Print something when this filter fails
        '''

        self.operate_on_datasets = operate_on_datasets
        self.binder_fbscn_cut = binder_fbscn_cut
        self.binder_fbscn_at_interface_cut = binder_fbscn_at_interface_cut
        self.verbose = verbose

    def __call__(self, indep, chosen_dataset, **kwargs):

        # The input arguments unchanged if we decide not to do anthing
        do_nothing_return = dict(indep=indep, chosen_dataset=chosen_dataset, **kwargs)

        # Only operate on the datasets we're told to
        if not ('all' in self.operate_on_datasets or chosen_dataset in self.operate_on_datasets):
            return do_nothing_return


        chain_masks = indep.chain_masks()
        if len(chain_masks) == 1:
            print('PPIRejectUnfoldedInterfacesTransform got passed a single chain')
            return do_nothing_return

        # Get the basic backbone data
        Ca = indep.xyz[:,1]
        Cb = Cb_or_atom(indep)
        is_binder = torch.tensor(chain_masks[0])

        # Get sidechain neighbors and hotspots
        sc_neigh = sidechain_neighbors(Ca[is_binder], Cb[is_binder], Ca[is_binder])

        is_hotspots_both, _ = find_hotspots_antihotspots(indep, only_of_first_chain=False)
        hotspot_on_binder = is_hotspots_both & is_binder
        if hotspot_on_binder.sum() == 0:
            raise NextExampleException("PPIRejectUnfoldedInterfacesTransform: Chains aren't touching?")

        # The standard scn cutoff for boundary is 4.0
        binder_fbscn = (sc_neigh > 4.0).float().mean()
        interface_fbscn = (sc_neigh[hotspot_on_binder[is_binder]] > 4.0).float().mean()

        if binder_fbscn < self.binder_fbscn_cut:
            raise NextExampleException(f'PPIRejectUnfoldedInterfacesTransform: Failed binder_fbscn_cut: {binder_fbscn} < {self.binder_fbscn_cut}',
                                                                                                                            quiet=not self.verbose)
        if interface_fbscn < self.binder_fbscn_at_interface_cut:
            raise NextExampleException('PPIRejectUnfoldedInterfacesTransform: Failed binder_fbscn_at_interface_cut:' +
                                        f' {interface_fbscn} < {self.binder_fbscn_at_interface_cut}', quiet=not self.verbose)

        return do_nothing_return



class PPIJoeNateDatasetRadialCropTransform:
    '''
    An interesting trick used by Joe and Nate.

    Ensure that the binder is less than a certain size uncropped, then radially crop the target around a random hotspot
    Binder can be ensured to be smaller than that by using data_loader.fast_filters.*.reject_chain0_longer_than 
    '''

    def __init__(self, operate_on_datasets=['all'], CROP=300):
        '''
        Args:
            operate_on_datasets (list[str]): Which datasets should this be applied to?
            CROP (int): How many residues should this be cropped to?
        '''
        self.operate_on_datasets = operate_on_datasets
        self.CROP = CROP

    def __call__(self, indep, atom_mask, chosen_dataset, metadata, conditions_dict, **kwargs):

        # The input arguments unchanged if we decide not to do anthing
        do_nothing_return = dict(indep=indep, atom_mask=atom_mask, chosen_dataset=chosen_dataset, metadata=metadata, conditions_dict=conditions_dict, **kwargs)

        # Only operate on the datasets we're told to
        if not ('all' in self.operate_on_datasets or chosen_dataset in self.operate_on_datasets):
            return do_nothing_return

        chain_masks = indep.chain_masks()
        if len(chain_masks) == 1:
            print('PPIJoeNateDatasetRadialCropTransform got passed a single chain')
            return do_nothing_return

        # No need to crop if it's already small enough
        if indep.length() <= self.CROP:
            return do_nothing_return

        # Get the basic backbone data
        Cb = Cb_or_atom(indep)
        is_binder = torch.tensor(chain_masks[0])
        binderlen = is_binder.sum()

        if binderlen > self.CROP:
            raise NextExampleException('PPIJoeNateDatasetRadialCropTransform: Do you have data_loader.fast_filters.*.reject_chain0_longer_than ' + 
                                                                                                            f'set up? Binderlen: {binderlen}')


        # Find hotspots
        is_hotspots_both, _ = find_hotspots_antihotspots(indep, only_of_first_chain=False)
        hotspot_on_binder = is_hotspots_both & is_binder
        if hotspot_on_binder.sum() == 0:
            raise NextExampleException("PPIJoeNateDatasetRadialCropTransform: Chains aren't touching?")
        wh_hotspot_on_binder = torch.where(hotspot_on_binder)[0]

        # Randomly pick a single hotspot
        chosen_hotspot = torch_rand_choice_noreplace(wh_hotspot_on_binder, 1)[0]
        hotspot_Cb = Cb[chosen_hotspot]

        # Calculate distance squared to hotspot and set binders to all have distance 0
        dist2_hotspot = torch.sum( torch.square( Cb - hotspot_Cb ), axis=-1 )
        dist2_hotspot[is_binder] = 0

        # Find the closest residues to the hotspot
        _, keep_idx = torch.topk(dist2_hotspot, self.CROP, largest=False)

        keep_mask = torch.zeros(indep.length(), dtype=bool)
        keep_mask[keep_idx] = True

        # This implies a bug in this code
        assert (keep_mask[is_binder]).all(), 'A binder residue was going to get removed'

        aa_model.pop_mask(indep, keep_mask)
        atom_mask = atom_mask[keep_mask]
        metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], keep_mask)
        pop_conditions_dict(conditions_dict, keep_mask)

        return dict(
            indep=indep,
            atom_mask=atom_mask,
            chosen_dataset=chosen_dataset,
            metadata=metadata,
            conditions_dict=conditions_dict,
            **kwargs
            )


class ExposedTerminusTransform:
    def __call__(self, conf, indep, conditions_dict, **kwargs):
        '''
        Use ppi.exposed_N_terminus and ppi.exposed_N_terminus to mark the termini of the binder as antihotspots (so they'll repel the target)

        Args:
            indep (Indep): Indep
            conf (OmegaConf): The config
            conditions_dict (dict): The inference conditions

        Returns:
            Return signature is the same as call signature but conditions_dict['is_antihotspot'] has been updated
        '''

        if conf.ppi.exposed_N_terminus > 0 or conf.ppi.exposed_C_terminus > 0:
            chain_masks = indep.chain_masks()
            binder_mask = chain_masks[0]
            wh_binder_mask = torch.tensor(np.where(binder_mask)[0])

            if 'is_antihotspot' not in conditions_dict:
                conditions_dict['is_antihotspot'] = torch.zeros(indep.length(), dtype=bool)

            if conf.ppi.exposed_N_terminus > 0:
                to_expose = wh_binder_mask[:conf.ppi.exposed_N_terminus]
                conditions_dict['is_antihotspot'][to_expose] = True

            if conf.ppi.exposed_C_terminus > 0:
                to_expose = wh_binder_mask[-conf.ppi.exposed_C_terminus:]
                conditions_dict['is_antihotspot'][to_expose] = True


        return kwargs | dict(
            indep=indep,
            conf=conf,
            conditions_dict=conditions_dict,
            )



class HotspotAntihotspotResInferenceTransform:

    def __call__(self, conf, indep, conditions_dict, contig_map, **kwargs):
        '''
        Apply ppi.hotspot_res and ppi.antihotspot_res to conditions_dict[{'is_hotspot', 'hotspot_value', 'is_antihotspot', 'antihotspot_value}']

        Uses:
            conf.ppi.hotspot_res
            conf.ppi.hotspot_res_values
            conf.ppi.antihotspot_res
            conf.ppi.antihotspot_values

        Args:
            indep (Indep): Indep
            conf (OmegaConf): The config
            contig_map (ContigMap): The contig map
            conditions_dict (dict): The inference conditions

        Returns:
            Return signature is the same as call signature
        '''


        if bool(conf.ppi.hotspot_res):
            assert isinstance(conf.ppi.hotspot_res, str), f'ppi.hotspot_res must be a string! You passed "{conf.ppi.hotspot_res}"'
            hotspot_mask = contig_map.res_list_to_mask(conf.ppi.hotspot_res)

            assert 'FindHotspotsTrainingTransform' in conf.upstream_training_transforms.names, 'Model not set up for hotspots'
            assert conf.upstream_training_transforms.configs.FindHotspotsTrainingTransform.get('p_is_hotspot_example', 0) > 0, 'Model not trained for hotspots'

            if 'is_hotspot' in conditions_dict:
                conditions_dict['is_hotspot'] |= hotspot_mask
            else:
                conditions_dict['is_hotspot'] = hotspot_mask


        if bool(conf.ppi.super_hotspot_res):
            assert isinstance(conf.ppi.super_hotspot_res, str), f'ppi.super_hotspot_res must be a string! You passed "{conf.ppi.super_hotspot_res}"'
            super_hotspot_mask = contig_map.res_list_to_mask(conf.ppi.super_hotspot_res)

            assert 'FindHotspotsTrainingTransform' in conf.upstream_training_transforms.names, 'Model not set up for super_hotspots'
            assert str(conf.upstream_training_transforms.configs.FindHotspotsTrainingTransform.get('hotspot_values_mean', False)) in ['tenA_neighbors'], (
                                                                                                                                'Model not trained for super_hotspots')

            if 'hotspot_values' in conditions_dict:
                conditions_dict['hotspot_values'][super_hotspot_mask] = 1.0 # Until we know who else wants to write to this vector just overwrite them
            else:
                conditions_dict['hotspot_values'] = super_hotspot_mask.float()


        if bool(conf.ppi.antihotspot_res):
            assert isinstance(conf.ppi.antihotspot_res, str), f'ppi.antihotspot_res must be a string! You passed "{conf.ppi.antihotspot_res}"'
            antihotspot_mask = contig_map.res_list_to_mask(conf.ppi.antihotspot_res)

            assert 'FindHotspotsTrainingTransform' in conf.upstream_training_transforms.names, 'Model not set up for antihotspots'
            assert conf.upstream_training_transforms.configs.FindHotspotsTrainingTransform.get('p_is_antihotspot_example', 0) > 0, 'Model not trained for antihotspots'

            if 'is_antihotspot' not in conditions_dict:
                conditions_dict['is_antihotspot'] = antihotspot_mask
            else:
                conditions_dict['is_antihotspot'] |= antihotspot_mask

        # conditions_dict['antihotspot_value'] coming soon!


        return kwargs | dict(
            indep=indep,
            conf=conf,
            contig_map=contig_map,
            conditions_dict=conditions_dict,
            )



def filter_hotspots_by_sasa(indep, is_hotspot, sasa_cut=30, probe_radius=2.8, only_hotspots_of_first_chain=True):
    '''
    Filter hotspots such that they have some exposed area to solvent

    If only_hotspots_of_first_chain is true. The target as a whole is used for SASA calcs
    If only_hotspots_of_first_chain is false. Each chain individually is used for SASA calcs

    Args:
        is_hotspot (torch.Tensor[bool]): The current bool mask of hotspots [L]
        indep (indep): The indep
        sasa_cut (float): The amount of solvent accessible surface area a residue needs in A^2
        probe_radius (float): The radius of the sasa probe. Default at 2.8 to avoid accidental pockets
        only_hotspots_of_first_chain (bool): Implies everything but chain 0 is the target

    Returns:
        is_hotspot (torch.Tensor[bool]): The original hotspot mask but where they all have enough sasa
    '''
    # Copy inputs
    is_hotspot = is_hotspot.clone().bool()

    if only_hotspots_of_first_chain:
        # Assume everything but chain 0 is the target
        binder_mask = torch.tensor(indep.chain_masks()[0])
        target_mask = ~binder_mask
        target_indep, _ = aa_model.slice_indep(indep, target_mask, break_chirals=True)

        # Get the SASA of the target alone
        per_res_sasa = torch.zeros(indep.length())
        per_res_sasa[target_mask] = sasa.get_indep_sasa_per_res(target_indep, probe_radius=probe_radius)

    else:
        # Do each chain individually
        per_res_sasa = torch.zeros(indep.length())
        for chain_mask in indep.chain_masks():
            chain_mask = torch.tensor(chain_mask)

            # Get just this chain
            chain_indep, _ =  aa_model.slice_indep(indep, chain_mask, break_chirals=True)

            # Get the SASA of just this chain
            per_res_sasa[chain_mask] = sasa.get_indep_sasa_per_res(chain_indep, probe_radius=probe_radius)

    # Mask it out
    enough_sasa_for_hotspot = per_res_sasa > sasa_cut
    is_hotspot &= enough_sasa_for_hotspot

    return is_hotspot

def hotspot_downsample_n_closest_to_one(indep, is_hotspot, n):
    '''
    Find the n closest hotspot residues to a randomly chosen on and return that mask

    Args:
        indep (Indep): Indep
        is_hotspot (torch.Tensor[boo]): The incoming hotspot mask
        n (int): How many hotspots to keep

    Returns:
        is_hotspot (torch.Tensor[bool]): The new hotspots
    '''
    wh_hotspot = torch.where(is_hotspot)[0]

    # Pick the center of the cluster
    central = torch_rand_choice_noreplace(wh_hotspot, 1)[0]

    # Find distance to central residue
    Cb = Cb_or_atom(indep)
    hotspot_cb = Cb[wh_hotspot]
    center_cb = Cb[central]

    d2_center = torch.sum( torch.square( hotspot_cb - center_cb), axis=-1 )

    # Only keep n closest
    n_keep = min(n, len(wh_hotspot))

    _, keep_idx = torch.topk(d2_center, n_keep, largest=False)

    keep_hotspot = wh_hotspot[keep_idx]

    # Mark new hotspots
    is_hotspot[:] = False
    is_hotspot[keep_hotspot] = True

    return is_hotspot


def hotspot_downsample(indep, is_hotspot, method, max_hotspot_frac):
    '''
    Perform hotspot downsampling such that all hotspots are not always shown

    In all cases, the number of hotspots that remain is somewhere betweeen 0 and is_hotspot.sum() * max_hotspot_frac

    Methods:
        random - Pay no attention to the spatial locations of the hotspots
        speckle_or_region - Of the hotspots to remove, first remove half randomly. Then the other half either randomly again or
                                        keep only those closest to a randomly chosen hotspot

    Args:
        indep (indep): Indep
        is_hotspot (torch.Tensor[bool]): The incoming hotspot mask [L]
        method (str): Which method do you want to use?
        max_hotspot_frac (float): The maximum fraction of hotspots that could remain

    Returns:
        is_hotspot (torch.Tensor[bool]): The new hotspots
    '''

    if is_hotspot.sum() <= 1:
        return is_hotspot

    if method == 'random':
        is_hotspot = downsample_bool_mask(is_hotspot, max_frac_to_keep=max_hotspot_frac)

    elif method == 'speckle_or_region':
        # Perform half the reduction by speckling. The other half by either another speckle or closest to res

        n_total = is_hotspot.sum()
        n_keep = random.randint(1, int(torch.ceil( n_total * max_hotspot_frac )))

        n_reduce = n_total - n_keep

        speckle_reduce = int(n_reduce / 2)

        if speckle_reduce > 0:
            is_hotspot = downsample_bool_mask(is_hotspot, n_to_keep=is_hotspot.sum() - speckle_reduce)

        if torch.rand(1) < 0.5:
            # speckle
            is_hotspot = downsample_bool_mask(is_hotspot, n_to_keep=n_keep)
        else:
            # region
            is_hotspot = hotspot_downsample_n_closest_to_one(indep, is_hotspot, n_keep)

    else:
        assert False, f'Unknown downsample method: {method}'

    return is_hotspot



def get_hotspot_values(indep, is_hotspot, hotspot_values_mean, max_10a_neighbors=12):
    '''
    Instead of just being a boolean. Hotspot values allows you to specify a float for each hotspot

    Methods:
        boolean -- Just 0 or 1
        tenA_neighbors -- The number of 10A neighbors that the hotspots has / max_10a_neighbors

    Args:
        indep (indep): Indep
        is_hotspot (torch.Tensor[bool]): The incoming hotspot mask [L]
        hotspot_values_mean (str): What method is used to select the hotspot values
        max_10a_neighbors (int): The value above which the tenA_neighbors method returns 1

    Returns:
        hotspot_values (torch.Tensor[float]): The value of each hotspot
    '''

    if hotspot_values_mean == 'boolean':
        return is_hotspot.float()
    elif hotspot_values_mean == 'tenA_neighbors':

        # Get all by all CA distance map
        Ca = indep.xyz[:,1]
        all_by_dist2 = torch.sum( torch.square( Ca[:,None] - Ca[None,:] ), axis=-1)

        # All by all bool map of if they're close
        all_by_close = all_by_dist2 < 10**2

        # Find neighbors of all residues. A little wasteful but this probably isn't the slow step
        num_neighbors = torch.zeros(indep.length())
        for chain_mask in indep.chain_masks():
            chain_mask = torch.tensor(chain_mask)

            # Count the number of residues on other chains that are close to this chain
            not_chain_mask = ~chain_mask
            num_neighbors[chain_mask] = all_by_close[chain_mask][:,not_chain_mask].sum(axis=-1).float()

        # Random +- 1 to avoid exactness
        num_neighbors += torch.rand(len(num_neighbors)) * 2 - 1
        num_neighbors = torch.clip(num_neighbors, 0, max_10a_neighbors)

        # Store them 0-1
        hotspot_values = torch.zeros(indep.length())
        hotspot_values[is_hotspot] = num_neighbors[is_hotspot] / max_10a_neighbors

        return hotspot_values
    else:
        assert False, f'Unknown hotspot_values_mean: {hotspot_values_mean}'


class FindHotspotsTrainingTransform:
    """
    Transform that loads conditions_dict with is_hotspot and is_antihotspot for training
    """
    def __init__(self, p_is_hotspot_example=0, p_is_antihotspot_example=0, max_hotspot_frac=0.2, max_antihotspot_frac=0.05, only_hotspots_of_first_chain=True,
            hotspot_distance=7, antihotspot_distance=10, hotspot_downsample_method='speckle_or_region', hotspot_sasa_cut=30, hotspot_values_mean=False):
        """
        Args:
            p_is_hotspot_example (float): Probability we show any hotspots at all
            p_is_antihotspot_example (float): Probability we show any antihotspots at all
            max_hotspot_frac (float): Maximum fraction of all possible hotspot residues that are shown
            max_antihotspot_frac (float): Maximum fraction of all possible antihotspot residues that are shown
            only_hotspots_of_first_chain (bool): Use the standard definition of hotspots where they are only chain 0's contacts elswehere
            hotspot_distance (float): How close must the CB be to each other to count as a hotspot?
            antihotspot_distance (float): How far away must a CB be from the binder to be called an antihotspot?
            hotspot_sasa_cut (float): How much solvent accessible surface area must a residue have to be called a hotspot?
            hotspot_downsample_method (str): How should the hotspots be downsample? ['random', 'speckle_or_region']
            hotspot_values_mean (str or False): Do hotspot values exist and if so what do they mean? [False, 'boolean', 'tenA_neighbors']
        """

        self.p_is_hotspot_example = p_is_hotspot_example
        self.p_is_antihotspot_example = p_is_antihotspot_example
        self.max_hotspot_frac = max_hotspot_frac
        self.max_antihotspot_frac = max_antihotspot_frac
        self.only_hotspots_of_first_chain = only_hotspots_of_first_chain
        self.hotspot_distance = hotspot_distance
        self.antihotspot_distance = antihotspot_distance
        self.hotspot_sasa_cut = hotspot_sasa_cut
        self.hotspot_downsample_method = hotspot_downsample_method
        self.hotspot_values_mean = hotspot_values_mean

        assert self.hotspot_downsample_method in ['random', 'speckle_or_region']
        if self.hotspot_values_mean:
            assert self.hotspot_values_mean in ['boolean', 'tenA_neighbors']


    def __call__(self, indep, conditions_dict, **kwargs):
        '''
        Args:
            indep (indep): indep
            conditions_dict (dict): The conditions_dict for training

        Returns:
            conditions_dict['is_hotspot'] (torch.Tensor[bool]): A subset of the true hotspot residues
            conditions_dict['hotspot_values'] (torch.Tensor[float]): An alternative to hotspots using float values to denote something
            conditions_dict['is_antihotspot'] (torch.Tensor[bool]): A subset of the true antihotspot residues
            conditions_dict['antihotspot_values'] (torch.Tensor[float]): An alternative to antihotspots using float values to denote something
        '''

        use_hotspot = (torch.rand(1) < self.p_is_hotspot_example).item()
        use_antihotspot = (torch.rand(1) < self.p_is_antihotspot_example).item()
        use_hotspot_values = use_hotspot and bool(self.hotspot_values_mean)

        if use_hotspot or use_antihotspot:
            is_hotspot_full, _ = find_hotspots_antihotspots(indep, only_of_first_chain=self.only_hotspots_of_first_chain,
                                                                                                            dist_cutoff=self.hotspot_distance )
            _, is_antihotspot_full = find_hotspots_antihotspots(indep, dist_cutoff=self.antihotspot_distance )

            is_antihotspot = downsample_bool_mask(is_antihotspot_full, max_frac_to_keep=self.max_antihotspot_frac)

            is_hotspot = is_hotspot_full.clone()

            # Downsample hotspots by SASA
            if is_hotspot.sum() > 0 and self.hotspot_sasa_cut > 0:

                is_hotspot = filter_hotspots_by_sasa(indep, is_hotspot, sasa_cut=self.hotspot_sasa_cut,
                                                                                    only_hotspots_of_first_chain=self.only_hotspots_of_first_chain)

            # Downsample hotspots by random method
            if is_hotspot.sum() > 0:

                if self.hotspot_downsample_method != 'random':
                    assert self.only_hotspots_of_first_chain, "You'll have to make your own downsample method if you don't like random."

                is_hotspot = hotspot_downsample(indep, is_hotspot, self.hotspot_downsample_method, self.max_hotspot_frac)

            if use_hotspot_values and is_hotspot.sum() > 0:
                is_hotspot_values = get_hotspot_values(indep, is_hotspot, self.hotspot_values_mean)
            else:
                is_hotspot_values = is_hotspot.float()


            if use_hotspot:

                # Hotspots are stored to either is_hotspot or hotspot value and we're deciding where each should go
                stores_to_values = torch.zeros(len(is_hotspot), dtype=bool)
                if use_hotspot_values:
                    # 50% of the time it's pegged at all-shown or none-shown. Else its between the two
                    values_prob = torch.clip((torch.rand(1) * 2) - 0.5, 0, 1) # clip the range [-0.5, 1.5] to [0, 1]
                    stores_to_values = torch.rand(len(is_hotspot)) <= values_prob

                assert 'is_hotspot' not in conditions_dict, 'FindHotspotsTrainingTransform must create hotspot info'
                assert 'hotspot_values' not in conditions_dict, 'FindHotspotsTrainingTransform must create hotspot info'

                # Initialize the vectors then store the values to them
                conditions_dict['is_hotspot'] = torch.zeros(len(is_hotspot), dtype=bool)
                conditions_dict['is_hotspot'][~stores_to_values] = is_hotspot[~stores_to_values]

                conditions_dict['hotspot_values'] = torch.zeros(len(is_hotspot))
                conditions_dict['hotspot_values'][stores_to_values] = is_hotspot_values[stores_to_values]

                if not use_hotspot_values:
                    assert (conditions_dict['hotspot_values'] == 0).all()

            if use_antihotspot:
                if 'is_antihotspot' in conditions_dict:
                    conditions_dict['is_antihotspot'] |= is_antihotspot
                else:
                    conditions_dict['is_antihotspot'] = is_antihotspot

        return kwargs | dict(
            indep=indep,
            conditions_dict=conditions_dict
        )


def get_hotspots_antihotspots_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    '''
    See get_hotspots_antihotspots_conditioning()
    '''
    return get_hotspots_antihotspots_conditioning(indep, feature_conf, **kwargs)


def get_hotspots_antihotspots_conditioning(indep, feature_conf, is_hotspot=None, hotspot_values=None, is_antihotspot=None, antihotspot_values=None, **kwargs):
    '''
    Generates the hotspot and antihotspot features for training and inference

    Args:
        indep (Indep): indep
        feature_conf (OmegaConf): The configuration for this feature
        is_hotspot (torch.Tensor[bool] or None): Boolean mask denoting which residues are hotspots from conditions_dict [L]
        hotspot_values (torch.Tensor[float] or None): Float mask alternative to is_hotspot from conditions_dict [L]
        is_antihotspot (torch.Tensor[bool] or None): Boolean mask denoting which residues are antihotspots from conditions_dict [L]
        antihotspot_values (torch.Tensor[float] or None): Float mask alternative to is_antihotspot from conditions_dict [L]

    Returns:
        dict:
            t1d (torch.Tensor[bool]): is_hotspot, hotspot_values, is_antihotspot, antihotspot_values [L,4]
    '''

    if is_hotspot is None:
        is_hotspot = torch.zeros(indep.length(), dtype=bool)
    if hotspot_values is None:
        hotspot_values = torch.zeros(indep.length())
    if is_antihotspot is None:
        is_antihotspot = torch.zeros(indep.length(), dtype=bool)
    if antihotspot_values is None:
        antihotspot_values = torch.zeros(indep.length())


    assert len(is_hotspot) == indep.length(), 'is_hotspot vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(hotspot_values) == indep.length(), 'hotspot_values vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(is_antihotspot) == indep.length(), 'is_antihotspot vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'
    assert len(antihotspot_values) == indep.length(), 'antihotspot_values vector does not match indep.length(). Is ExpandConditionsDict in conf.transforms?'


    ret_stack = [is_hotspot.float(), hotspot_values.float(), is_antihotspot.float(), antihotspot_values.float()]

    return {'t1d':torch.stack(ret_stack, axis=-1)}



class RenumberCroppedInput:
    """
    Adjust indep.idx for input pdbs that have been cropped and numbered 1-N

    Guesses how many residues should go in the gap using angstroms_per_aa
    """
    def __init__(self, enabled=True, angstroms_per_aa=4, verbose=True):
        '''
        Args:
            enable (bool): Whether or not this should run
            angstroms_per_aa (float): How many A do we assume each AA takes up
            verbose (bool): Print something when insertions happen
        '''

        self.enabled = enabled
        self.angstroms_per_aa = angstroms_per_aa
        self.verbose = verbose


    def __call__(self, indep, contig_map, conf, masks_1d, **kwargs):

        if self.enabled:

            # Accumulator variable for where we are inserting
            insert_n_after_res = torch.zeros(indep.length(), dtype=int)

            # Get the backbone
            Ns = indep.xyz[:,0]
            Cas = indep.xyz[:,1]
            Cs = indep.xyz[:,2]

            assert len(contig_map.ref) == indep.length()
            had_input_coords = torch.tensor([x != ('_', '_') for x in contig_map.ref])

            for chain_mask in indep.chain_masks():
                chain_mask = torch.tensor(chain_mask)

                chain_sm = chain_mask & indep.is_sm

                # Entirely a small molecule. We definitely don't want to mess with it
                if chain_sm[chain_mask].all():
                    continue
                if chain_sm.any():
                    # Make sure the small molecules are entirely after the protein
                    wh_sm = torch.where(chain_sm)[0]
                    wh_prot = torch.where(chain_mask & ~indep.is_sm)[0]
                    assert wh_sm.min() > wh_prot.max(), ('You have small molecules mixed into your protein. '
                                                'Maybe set upstream_inference_transforms.configs.RenumberCroppedInput.enabled=False and tell bcov?')

                # Small molecules all at the end so drop them
                chain_mask[indep.is_sm] = False
                chain_mask[~had_input_coords] = False

                # Guidepost residues shouldn't be renumbered. The indices that we would be changing are actually the diffused indices
                if conf.inference.contig_as_guidepost:
                    chain_mask[masks_1d['can_be_gp']] = False


                if chain_mask.sum() <= 1:
                    continue

                wh_chain_mask = torch.where(chain_mask)[0]

                chain_Ns = Ns[chain_mask]
                chain_Cas = Cas[chain_mask]
                chain_Cs = Cs[chain_mask]

                chain_idxs = indep.idx[chain_mask]

                # d_idx is how many idx until the residue after this one
                d_idx = chain_idxs[1:] - chain_idxs[:-1]

                # C_N_dist is how far the next N is from this C
                C_N_dist = torch.linalg.norm( chain_Ns[1:] - chain_Cs[:-1], axis=-1)

                C_N_too_far = C_N_dist > 3

                Ca_Ca_dist = torch.linalg.norm( chain_Cas[:-1] - chain_Cas[1:], axis=-1)

                # if the C_N distance is bad
                # 3.1A insert 1
                # 3.9A insert 1
                # 4.1A insert 2

                missing_n_res = torch.ceil(Ca_Ca_dist / self.angstroms_per_aa).long()

                wh_insert = wh_chain_mask[:-1][C_N_too_far & (d_idx == 1)]

                insert_n_after_res[wh_insert] = missing_n_res[C_N_too_far & (d_idx == 1)]

                if self.verbose:
                    for idx, n in zip(wh_insert, missing_n_res[C_N_too_far]):
                        print(f'RenumberCroppedInput: Inserting {n} res after {contig_map.ref[idx]}')

            indep.idx[1:] += torch.cumsum( insert_n_after_res[:-1], 0 )


        return kwargs | dict(
            indep=indep,
            contig_map=contig_map,
            conf=conf,
            masks_1d=masks_1d
        )

def get_origin_normal_to_target_hotspot(indep, conditions_dict, is_diffused, normal_extension=10):
    assert 'is_hotspot' in conditions_dict or 'hotspot_values' in conditions_dict, 'You must use ppi.hotspot_res or ppi.hotspot_res_values to use center_type: target-hotspot'
    is_hotspot = torch.zeros(indep.length(), dtype=bool)

    if 'is_hotspot' in conditions_dict:
        is_hotspot |= conditions_dict['is_hotspot']
    if 'hotspot_values' in conditions_dict:
        is_hotspot |= conditions_dict['hotspot_values'] != 0

    assert is_hotspot.sum() > 0, 'You must use ppi.hotspot_res or ppi.hotspot_res_values to use center_type: target-hotspot'

    # use CB for hotspots to get it a little closer to the CoM of the theoretical binder
    Cb = Cb_or_atom(indep)
    hotspot_cb_com = Cb[is_hotspot].mean(axis=0)
    target_ca = indep.xyz[~is_diffused,1]
    
    target_ca_near_hotspot = target_ca[(target_ca - hotspot_cb_com).norm(dim=1) < 10]
    # target_ca_near_hotspot_com = np.mean(target_ca_near_hotspot, axis=0)
    target_ca_near_hotspot_com = target_ca_near_hotspot.mean(axis=0)

    from_core_to_hotspot = hotspot_cb_com - target_ca_near_hotspot_com
    from_core_to_hotspot = from_core_to_hotspot / from_core_to_hotspot.norm()
    
    return hotspot_cb_com + normal_extension * from_core_to_hotspot


def get_origin_target_hotspot(indep, conditions_dict, is_diffused, only_hotspots=False):
    '''
    A function to emulate center_type: all for PPI.

    Assigns the CoM of the binder to the CoM of the hotspots and calculates the CoM of the system from that

    Args:
        indep (indep): indep
        conditions_dict (dict): The conditions dict

    Returns:
        origin (torch.Tensor[float]): The specified origin [3]
    '''


    assert 'is_hotspot' in conditions_dict or 'hotspot_values' in conditions_dict, 'You must use ppi.hotspot_res or ppi.hotspot_res_values to use center_type: target-hotspot'
    is_hotspot = torch.zeros(indep.length(), dtype=bool)

    if 'is_hotspot' in conditions_dict:
        is_hotspot |= conditions_dict['is_hotspot']
    if 'hotspot_values' in conditions_dict:
        is_hotspot |= conditions_dict['hotspot_values'] != 0

    assert is_hotspot.sum() > 0, 'You must use ppi.hotspot_res or ppi.hotspot_res_values to use center_type: target-hotspot'

    # use CB for hotspots to get it a little closer to the CoM of the theoretical binder
    Cb = Cb_or_atom(indep)
    hotspot_com = Cb[is_hotspot].mean(axis=0)
    target_com = indep.xyz[~is_diffused,1].mean(axis=0)

    if only_hotspots:
        return hotspot_com

    # Calculate CoM by using the relative weights of the two proteins
    target_size = (~is_diffused).sum()
    binder_size = is_diffused.sum()

    origin = (hotspot_com * binder_size + target_com * target_size) / (target_size + binder_size)

    return origin
