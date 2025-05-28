import torch
from icecream import ic
from rf_diffusion import aa_model
from omegaconf import OmegaConf
from rf_diffusion import conditioning


def extract_ori(indep: aa_model.Indep, pdb_fp: str):
    """
    Extracts the ORI token from the indep

    Args:
        indep (Indep): The holy Indep
        pdb_fp (str): The path to the PDB file

    Returns:
        origin (torch.tensor): The origin point [3,]
    """
    with open(pdb_fp, 'r') as fh:
        stream_orig = fh.readlines()

    # Extract ORI token and convert to origin token for indep
    stream = aa_model.remove_non_target_ligands(stream_orig, ['ORI'])
    # ORI may get connected to other atoms due to distance, but we can ignore
    stream_ori = aa_model.filter_het(stream, ['ORI'], covale_allowed=True, keep_conect=False)  
    target_feats_ori = aa_model.parse_pdb_lines_target(stream_ori, ['ORI'])
    ori_xyz = target_feats_ori['xyz_het']     
    N_ori = ori_xyz.shape[0]

    # If ORI tokens are found, parse
    origin = None
    if N_ori == 0:
        return origin

    if N_ori > 1:
        # Sample if multiple are found
        ic('Multiple ORI atoms found in the pdb, selecting one randomly. You better know what you are doing')
        rand_idx = torch.randint(low=0, high=N_ori, size=(1,))
        ori_xyz = ori_xyz[rand_idx,:]
    else:
        ori_xyz = ori_xyz[0]
    origin = torch.tensor(ori_xyz, dtype=indep.xyz.dtype, device=indep.xyz.device)  # [3,]        

    return origin


def validate_centering_strategy(origin: torch.Tensor, for_partial_diffusion: bool, conf: OmegaConf):
    """
    Checks if the ORI is set correct
    Args:
        origin (torch.Tensor, None): The origin point [3,]
        for_partial_diffusion (bool): Whether the current indep should undergo partial diffusion
        conf (OmegaConf) : Full config
    
    Raises:
        AssertionError: If the ORI is not set correctly
    """
    # Validate CenterPostTransform if it is used in the transform stack
    if 'CenterPostTransform' in conf.transforms.names:
        # Get the center type, or use default if not found
        center_type = getattr(conf.transforms.configs.CenterPostTransform, 'center_type', conditioning.CenterPostTransform().center_type)        
        if center_type not in ['all', 'is_diffused'] or for_partial_diffusion:
            return
        assert origin is not None, "ORI HETATM token is required for centering input correctly but was not provided"


def extract_centering_origin(indep: aa_model.Indep, pdb_fp: str, for_partial_diffusion: bool) -> torch.tensor:
    """
    Extracts the center of the indep as the origin

    Args:
        indep (Indep): The holy Indep
        pdb_fp (str): The path to the PDB file
        for_partial_diffusion (bool): Whether the current indep should undergo partial diffusion

    Returns:
        origin (torch.tensor): The origin point [3,]
    """
    origin = None
    if not for_partial_diffusion:        
        origin = extract_ori(indep, pdb_fp)

    return origin
