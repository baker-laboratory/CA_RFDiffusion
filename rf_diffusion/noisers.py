import logging
import torch
import copy

from openfold.utils import rigid_utils as ru

from rf_diffusion.frame_diffusion.data import se3_diffuser
from rf_diffusion.frame_diffusion.data import legacy_diffuser
FwdMargYieldsTMinusOne = legacy_diffuser.FwdMargYieldsTMinusOne

logger = logging.getLogger(__name__)

def get(noiser_conf):
    if 'type' not in noiser_conf or noiser_conf.type == 'diffusion':
        return se3_diffuser.SE3Diffuser(noiser_conf)
    elif noiser_conf.type == 'legacy': 
        return FwdMargYieldsTMinusOne( legacy_diffuser.LegacyDiffuser(noiser_conf) )
    elif noiser_conf.type == 'refine': 
        return legacy_diffuser.RefineDiffuser(noiser_conf)
    else:
        raise Exception(f'noiser type: {noiser_conf.type} not recognized')


class FakeR3Diffuser:
    def marginal_b_t(*args, **kwargs):
        return torch.tensor(1.0)
