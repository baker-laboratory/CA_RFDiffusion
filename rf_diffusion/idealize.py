import torch

from rf2aa.util_module import XYZConverter
from rf_diffusion.chemical import ChemicalData as ChemData



def calc_residue_rmsds(xyz1, xyz2, seq, eps=1e-6):
    '''
    Intended to calculate the overall and per residue rmsd of heavy atoms
    between two different conformations of the same molecule. The corrdinates
    for non-existent atoms should be NaN.
    
    Inputs
        xyz1, xyz2 (..., L, 36, 3)
        seq (..., L)
    
    Returns
        rmsd (float): The rmsd between all heavy atoms
        per_residue_rmsd (..., L): The rmsd of all heavy atoms for each residue
    '''    
    # Shape checks
    assert xyz1.shape == xyz2.shape
    assert seq.shape == xyz1.shape[:-2], f'{seq.shape=} {xyz1.shape=}'
    
    # Only calculate things over real valued coordinates
    is_canonically_resolved = ChemData().allatom_mask[seq]  # (..., L, 36)
    is_canonically_resolved[..., ChemData().NHEAVY:] = False
    xyz1_is_resolved = ~torch.isnan(xyz1).any(-1)
    xyz2_is_resolved = ~torch.isnan(xyz2).any(-1)
    is_heavy_atom_resolved = xyz1_is_resolved & xyz2_is_resolved & is_canonically_resolved

    # Pytorch doesn't like to backprop through nans, even when masked around
    xyz1 = torch.nan_to_num(xyz1)
    xyz2 = torch.nan_to_num(xyz2)

    # per atom squared error
    per_atom_se = (xyz1 - xyz2).pow(2).sum(-1)
    per_atom_se = torch.where(is_heavy_atom_resolved, per_atom_se, torch.tensor(0.))

    # per residue rmsd
    per_residue_rmsd = torch.sqrt(per_atom_se.sum(-1) / is_heavy_atom_resolved.sum(-1) + eps)

    # overall atom rmsd
    rmsd = torch.sqrt(per_atom_se.sum((-1, -2)) / is_heavy_atom_resolved.sum((-1, -2)) + eps)

    return rmsd, per_residue_rmsd

@torch.inference_mode(False) # Required to override any enclosing inference_mode(True) context i.e. if model is running in evaluation mode.
def idealize_pose(xyz, seq, steps=50, lr=1e-1):
    '''
    Adjusts torsion angles to minimize the heavy atom rmsd to the 
    xyz coordinates. Does not move backbone N, CA or C.

    Inputs
        xyz (B, L, 36, 3)
        seq (B, L)
        steps: Number of steps to take with the optimizer
        lr: Optimizer learning rate

    Return
        xyz_ideal (B, L, 36, 3)
        rmsd (B,)
        per_residue_rmsd (B, L)
    '''
    # Shape check
    assert xyz.shape[-2:] == (36, 3), f'{xyz.shape=}'
    assert xyz.shape[:2] == seq.shape

    # Since empty tensors as input to torch.einsum is not handled by some versions of torch, see https://github.com/pytorch/pytorch/issues/111757
    # exit early to avoid this corner case.  (Encountered in xyz_converter.compute_all_atom)
    B = xyz.shape[0]
    if torch.numel(xyz) == 0:
        return xyz, torch.full((B, 0), torch.nan), torch.full((B, 0), torch.nan), torch.full((B, 0), torch.nan)

    # Get the torsion angles initial guess
    xyz_converter = XYZConverter()
    torsions, torsions_alt, tors_mask, tors_planar = xyz_converter.get_torsions(xyz, seq)

    # SGD of torsions to minimize heavy atom rmsd
    if steps > 0:
        torsions = torsions.detach()  # to make a leaf
        torsions.requires_grad = True
        optimizer = torch.optim.Adam([torsions], lr=lr)

    losses = []
    for i in range(steps):
        # Calculate ideal coordinates
        RTframes, xyz_ideal = xyz_converter.compute_all_atom(seq, xyz, torsions)
        
        # Calc rmsd
        rmsd, per_residue_rmsd = calc_residue_rmsds(xyz, xyz_ideal, seq)
        loss = rmsd.sum()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    
    # Calculate ideal coordinates
    RTframes, xyz_ideal = xyz_converter.compute_all_atom(seq, xyz, torsions)
    
    # Calc rmsd
    rmsd, per_residue_rmsd = calc_residue_rmsds(xyz, xyz_ideal, seq)

    return xyz_ideal.detach(), rmsd.detach(), per_residue_rmsd.detach(), losses

