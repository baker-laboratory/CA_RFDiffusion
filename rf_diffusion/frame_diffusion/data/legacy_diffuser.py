import numpy as np 
import logging 
import torch 
import copy 
from scipy.spatial.transform import Rotation as scipy_R

# this package 
from rf_diffusion.frame_diffusion.data import so3_diffuser, r3_diffuser
from rf_diffusion.frame_diffusion.data.se3_diffuser import SE3Diffuser, _assemble_rigid, _extract_trans_rots
from openfold.utils import rigid_utils as ru
from rf_diffusion.frame_diffusion.rf_score.model import calc_score
from rf_diffusion.frame_diffusion.data import all_atom





def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}, inference=False):
    """
    From Watson et al, 2023
    -----------------------
    Given a noise schedule type, create the beta schedule
    """
    assert schedule_type in ['linear', 'geometric', 'cosine']
    if T not in [1,2]: #  T=1|2 only used in testing
        assert T >= 15, "With discrete time and T < 15, the schedule is badly approximated"
        b0 *= (200 / T)
        bT *= (200 / T)

    # linear noise schedule
    if schedule_type == 'linear':
        schedule = torch.linspace(b0, bT, T)

    # geometric noise schedule
    elif schedule_type == 'geometric':
        raise NotImplementedError('geometric schedule not ready yet')

    # cosine noise schedule
    else:
        raise NotImplementedError('Cosine schedule has been disabled because variance with different T will need to be worked out')


    #get alphabar_t for convenience
    alpha_schedule = 1-schedule
    alphabar_t_schedule  = torch.cumprod(alpha_schedule, dim=0)

    if inference:
        print(f"With this beta schedule ({schedule_type} schedule, beta_0 = {b0}, beta_T = {bT}), alpha_bar_T = {alphabar_t_schedule[-1]}")

    return schedule, alpha_schedule, alphabar_t_schedule


def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    """
    From Watson et al, 2023
    -----------------------
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1
    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*beta_schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*beta_schedule[t_idx])/(1-alphabar_schedule[t_idx]))*px0_ca
    b = ((torch.sqrt(1-beta_schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))*xt_ca

    mu = a + b

    return mu, sigma


def get_next_ca(xt, px0, t, diffusion_mask, crd_scale, beta_schedule, alphabar_schedule, noise_scale=1.):
    """
    From Watson et al, 2023
    -----------------------
    Given full atom x0 prediction (xyz coordinates), move to x(t-1)
    
    Parameters:
        
        xt (L, 14/27, 3) set of coordinates
        px0 (L, 14/27, 3) set of coordinates
        t: time step. Note this is zero-index current time step, so are generating t-1    
        logits_aa (L x 20 ) amino acid probabilities at each position
        seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t
        diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused 
        noise_scale: scale factor for the noise being added

    """

    # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale 
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    # get mu(xt, x0)
    mu, sigma = get_mu_xt_x0(xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule)
    sampled_crds = torch.normal(mu, torch.sqrt(sigma*noise_scale))
    delta = sampled_crds - xt[:,1,:] 

    if diffusion_mask is not None:
        delta[diffusion_mask.squeeze(),...] = 0

    out_crds = xt + delta[:, None, :]

    return out_crds/crd_scale, delta/crd_scale


class EuclideanDiffuser():
    """
    From Watson et al, 2023
    -----------------------
    """

    def __init__(self,
                 T,
                 b_0,
                 b_T,
                 schedule_type='linear',
                 schedule_kwargs={},
                 ):

        self.T = T

        # make noise/beta schedule
        self.beta_schedule, _, self.alphabar_schedule  = get_beta_schedule(T, b_0, b_T, schedule_type, **schedule_kwargs)
        self.alpha_schedule = 1-self.beta_schedule


    def diffuse_translations(self, xyz, diffusion_mask=None, var_scale=1):
        return self.apply_kernel_recursive(xyz, diffusion_mask, var_scale)


    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """
        Applies a noising kernel to the points in x

        Parameters:
            x (torch.tensor, required): (N,3,3) set of backbone coordinates

            t (int, required): Which timestep

            noise_scale (float, required): scale for noise
        """
        t_idx = t-1 # bring from 1-indexed to 0-indexed

        assert len(x.shape) == 3
        L,_,_ = x.shape

        # c-alpha crds
        ca_xyz = x[:,1,:]


        b_t = self.beta_schedule[t_idx]


        # get the noise at timestep t
        mean  = torch.sqrt(1-b_t)*ca_xyz
        var   = torch.ones(L,3)*(b_t)*var_scale

        sampled_crds = torch.normal(mean, torch.sqrt(var))

        delta = sampled_crds - ca_xyz
        
        if diffusion_mask is not None:
            delta[diffusion_mask,...] = 0

        out_crds = x + delta[:,None,:]

        return out_crds, delta


    def apply_kernel_recursive(self, xyz, diffusion_mask=None, var_scale=1):
        """
        Repeatedly apply self.apply_kernel T times and return all crds
        """
        bb_stack = []
        T_stack  = []

        cur_xyz  = torch.clone(xyz)

        for t in range(1,self.T+1):
            cur_xyz, cur_T = self.apply_kernel(cur_xyz,
                                        t,
                                        var_scale=var_scale,
                                        diffusion_mask=diffusion_mask)
            bb_stack.append(cur_xyz)
            T_stack.append(cur_T)

        torch.save(torch.stack(bb_stack).transpose(0,1), 'bb_stack.pt')
        return torch.stack(bb_stack).transpose(0,1), torch.stack(T_stack).transpose(0,1)


class WrappedEuclideanDiffuser(EuclideanDiffuser, r3_diffuser.R3Diffuser):
    """
    Subclass to match the signatures of other diffusers.
    Written by DJ 
    """

    def __init__(self, conf):
        """
        Convert config into kwargs for EuclideanDiffuser 
        """
        EuclideanDiffuser.__init__(self, **{'T'               : conf.T,
                                            'b_0'             : conf.min_b,
                                            'b_T'             : conf.max_b,
                                            'schedule_kwargs' : conf.schedule_kwargs})
        
        r3_diffuser.R3Diffuser.__init__(self, conf)
        
        self.crd_scale = conf.coordinate_scaling
        self.noise_scale = conf.noise_scale


    def reverse(self, rigid_t, rigid_pred, t, diffuse_mask, **kwargs): 
        """
        Sample p(x_{t-1}|x_t) using EuclideanDiffuser

        Parameters: 
            rigid_t (openfold.utils.rigid_utils.Rigid): Structure at time t 
            rigid_pred (openfold.utils.rigid_utils.Rigid): Structure at time zero to interpolate towards
            t (float): continuous time in [0, 1].
            diffuse_mask (np.ndarray): Mask where True means apply update, False means no update. 
        """
        trans_t, _ = _extract_trans_rots(rigid_t)       # CAs at time t 
        trans_0, _ = _extract_trans_rots(rigid_pred)    # CAs at time 0
        L = len(trans_t.squeeze())

        ### build inputs for get_next_ca ###
        # Xt coordinates 
        xt = torch.zeros(L,14,3, dtype=torch.float) # (L,14,3)
        xt[:,1] = torch.from_numpy(trans_t.squeeze()) 

        # pX0 coordinates
        px0 = torch.zeros(L,14,3, dtype=torch.float) # (L,14,3)
        px0[:,1] = torch.from_numpy(trans_0)

        next_ca_kwargs = {  'xt'                  : xt,
                            'px0'                 : px0,
                            't'                   : int(t*self.T),
                            'diffusion_mask'      : diffuse_mask.bool(),
                            'crd_scale'           : self.crd_scale,
                            'beta_schedule'       : self.beta_schedule,       # from super 
                            'alphabar_schedule'   : self.alphabar_schedule,   # from super
                            'noise_scale'         : self.noise_scale}

        x_t_minus_1, _ = get_next_ca(**next_ca_kwargs)

        return x_t_minus_1[:,1,:]


class LegacyDiffuser(SE3Diffuser):
    """
    subclass of SE3Diffuser that uses the legacy (i.e., Watson et al) implementation 
    of euclidean diffuser. 
    Matches the signatures of updated noisers.

    Written by DJ
    """

    def __init__(self, se3_conf):
        self._log           = logging.getLogger(__name__)
        self._se3_conf      = se3_conf

        self._diffuse_rot   = se3_conf.diffuse_rot
        self._so3_diffuser  = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser   = WrappedEuclideanDiffuser(self._se3_conf.r3)

        self.crd_scale = se3_conf.r3.coordinate_scaling


    def reverse(self, **kwargs):
        """
        Legacy compatible reverse function.
        """
        # Get the t-1 rots however super does it. 
        # spoof no diffuse translations to save time in super call 
        orig_diffuse_trans = self._diffuse_trans
        self._diffuse_trans = False 
        super_so3_outs = super().reverse(**kwargs)
        rot_t_1 = super_so3_outs.get_rots().get_rotvec()
        self._diffuse_trans = orig_diffuse_trans # reset

        rigid_t    = kwargs['rigid_t']
        rigid_pred = kwargs['rigid_pred']
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        # reverse translations like EuclideanDiffuser
        if isinstance(kwargs['diffuse_mask'], np.ndarray): 
            kwargs['diffuse_mask'] = torch.from_numpy(kwargs['diffuse_mask'])

        trans_t_1 = self._r3_diffuser.reverse(rigid_t, 
                                              rigid_pred, 
                                              kwargs['t'], 
                                              ~(kwargs['diffuse_mask'].bool())) # Needs to go in True where NOT being denoised
                                                                                # and False where being denoised --> so invert diffuse_mask

        # use mask to prevent updates in specific locations if desired 
        # if kwargs['diffuse_mask'] is not None:
        #     trans_t_1 = self._apply_mask(
        #         trans_t_1, trans_t, kwargs['diffuse_mask'][..., None])
            
            # don't need to do it for rots because super does it
        return _assemble_rigid(rot_t_1.squeeze(), trans_t_1)



    def forward_marginal(self,
                         rigids_0       : ru.Rigid,
                         t              : float,
                         diffuse_mask   : np.ndarray=None,
                         as_tensor_7    : bool=True):
        """
        Samples from p(Xt|X0) using legacy diffuser. 

        Parameters:
            rigids_0 (openfold.utils.rigid_utils.Rigid): Series of openfold rigid objects 
            t (float): continuous time in [0, 1].
            diffuse_mask (np.ndarray): Mask denoting which tokens are subject to noising, and which are not. 
                                       NOTE: 1/True means 'revealed' (not diffused), 0/False means 'hidden' (diffused)
        """
        t_int = int(t*self._se3_conf.T) # convert back to integer 
        #### R3 Diffusion with Legacy Euclidean Diffuser #### 
        ##################################################### 

        # Only the CAs matter--spoof N/C
        L = len(rigids_0.get_trans())
        bb_crds = torch.zeros(L,3,3, dtype=torch.float) # (L,3,3) N,CA,C
        bb_crds[:,1] = rigids_0.get_trans()
        var_scale = self._se3_conf.r3.var_scale

        # scale the coordinates 
        bb_crds = bb_crds * self.crd_scale 

        # do the R3 noising 
        
        # Debugging -- check bb crds going into diffuse translations 
        _, deltas = self._r3_diffuser.diffuse_translations(bb_crds, ~(diffuse_mask.bool()), var_scale)
        cum_delta   = deltas.cumsum(dim=1) # (L,T,3) CA only 
        cum_delta_t = cum_delta[:,t_int-1] # zero indexed, so minus 1. (L,3)
        cum_delta_t = cum_delta_t / self.crd_scale # scale back
        
        bb_crds = bb_crds / self.crd_scale # scale back

        ###############################################
        #### SO3 Diffusion with stanard SO3Diffuser ### 
        # spoof no R3 diffusion since we did it above 
        orig_diffuse_trans  = self._diffuse_trans
        self._diffuse_trans = False
        so3_outs = super().forward_marginal(    rigids_0,
                                                t,
                                                diffuse_mask,
                                                as_tensor_7
                                         )
        self._diffuse_trans = orig_diffuse_trans # reset 

        # A rotation vector 
        rots_t = so3_outs['rigids_t'].get_rots().get_rotvec() # (L,3)


        # assemble output rigid like super does 
        trans_t = cum_delta_t + bb_crds[:,1] # (L,3)

        rigids_t = _assemble_rigid(rots_t, trans_t)

        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()

        outs = {'rigids_t'            :rigids_t,
                'trans_score'         :torch.tensor(0.0),
                'rot_score'           :torch.tensor(0.0),
                'trans_score_scaling' :torch.tensor(1.),
                'rot_score_scaling'   :torch.tensor(1.)}


        return outs
    

class FwdMargYieldsTMinusOne(): 
    """
    Wrapper for diffusers so that they output Xt minus one in the diffuser output as well. 
    """

    def __init__(self, diffuser):
        self.diffuser = diffuser


    def forward_marginal(self, rigids_0, **kwargs):
        """
        Wraps forward marginal to add on computation of the Xt-1 rigids. 

        Parameters:
            rigids_0 (openfold.utils.rigid_utils.Rigid): Rigids at time 0  
            kwargs['t'] (float): Current time
            kwargs['diffuse_mask'] (np.ndarray): Mask denoting which tokens are subject to noising, and which are not. 
                                                NOTE: 1/True means 'revealed' (not diffused), 0/False means 'hidden' (diffused)
        """
        diffuser_out = self.diffuser.forward_marginal(rigids_0, **kwargs)

        # Compute t-1 rigids from t rigids
        rigid_t = diffuser_out['rigids_t']
        rot_score_true, trans_score_true = calc_score(  rigids_0[None,None], 
                                                        rigid_t, 
                                                        self.diffuser, 
                                                        torch.tensor([kwargs['t']]))

        rigids_t_minus_1 = self.diffuser.reverse(   rigid_t      = rigid_t, 
                                                    rot_score    = rot_score_true,
                                                    trans_score  = trans_score_true,
                                                    t            = kwargs['t'],
                                                    dt           = 1/self.diffuser._se3_conf.T,
                                                    diffuse_mask = kwargs['diffuse_mask'].float(),
                                                    rigid_pred   = rigids_0
                                                )
        
        diffuser_out['rigids_t_minus_1'] = rigids_t_minus_1
        diffuser_out['x_t_minus_1'] = all_atom.atom37_from_rigid(rigids_t_minus_1)

        return diffuser_out


    def __getattr__(self, name):
        return getattr(self.diffuser, name)
    

    def __deepcopy__(self, memo):
        # ChatGPT wrote this one -- deal with deepcopy of EMA in train_multi_deep.py

        # Manually handle the deepcopy of the base_instance
        copied_base = copy.deepcopy(self.diffuser, memo)
        # Create a new Wrapper with the copied base instance
        copied_wrapper = FwdMargYieldsTMinusOne(copied_base)
        # Return the copied wrapper
        return copied_wrapper
    

class SingleGaussian(r3_diffuser.R3Diffuser):
    # class for perturbing points with a fixed amount of gaussian noise (not diffusion)

    def __init__(self,
                 conf
                 ):
        
        # ignore all classic diffusion parameters
        self.T          = conf.T
        self.variance   = conf.refine_variance

        r3_diffuser.R3Diffuser.__init__(self, conf)

    def diffuse_translations(self, xyz, diffusion_mask=None, crd_scale=1.0):
        assert len(xyz.shape) == 3, 'xyz must be (L,3,3)'
        return self.apply_gaussian_noise(xyz, diffusion_mask, crd_scale)

    def apply_gaussian_noise(self, xyz, diffusion_mask, crd_scale):
        """
        Applies gaussian noise to the points in x according to 
        self.mu and self.sigma

        Parameters: 

        xyz (torch.tensor, required): (L,3,3) set of backbone coordinates

        diffusion_mask (torch.tensor, optional):
        """
        mean = xyz[:,1,:]
        var = torch.ones_like(mean)*self.variance

        # variance is in A^2, given from command line 
        # --> scale to crd_scale because crds are already scaled
        sampled = torch.normal(mean, torch.sqrt(var)*crd_scale)
        delta = sampled - mean

        if diffusion_mask is not None:
            delta[diffusion_mask,...] = 0

        out_crds = xyz + delta[:,None,:].expand_as(xyz)

        return out_crds, delta
    

class RandomFrames(so3_diffuser.SO3Diffuser):
    """
    Produces completely random frames for all residues 
    """

    def __init__(self, conf):
        self.T = conf.T
        # self.use_cached_score=False
        # self.num_sigma=10
        # self.schedule='logarithmic'
        # self.max_sigma = 1.5
        # self.min_sigma = 0.1
        # self.cache_dir: './cache/'
        so3_diffuser.SO3Diffuser.__init__(self, conf)

    def diffuse_frames(self, xyz, t_list, diffusion_mask=None):
        return self.random_frames(xyz, diffusion_mask)
    
    def _get_random_rotations(self, L):
        """
        Get a random rotation matrix for each residue
        """
        return scipy_R.random(L)

    def random_frames(self, xyz, diffusion_mask=None):
        """
        produces a random frame for all AAs 
        """
        assert len(xyz.shape) == 3, 'xyz must be (L,3,3)'
        if torch.is_tensor(xyz):
            xyz = xyz.numpy()

        # R_rand = scipy_R.random(len(xyz))
        R_rand = self._get_random_rotations(len(xyz))

        # N  = torch.from_numpy(  xyz[None,:,0,:]  )
        Ca = torch.from_numpy(  xyz[None,:,1,:]  )
        # C  = torch.from_numpy(  xyz[None,:,2,:]  )

        rotated_crds = np.einsum('lij,laj->lai', 
                                 R_rand.as_matrix(), 
                                 xyz[:,:3,:] - Ca.squeeze()[:,None,...].numpy()) + Ca.squeeze()[:,None,...].numpy()
        
        rotated_crds = np.stack([rotated_crds]*self.T, axis=1)

        return rotated_crds
    

class RefineDiffuser(SE3Diffuser):
    """
    Diffuser for refinement.
    """

    def __init__(self, se3_conf):
        self._log           = logging.getLogger(__name__)
        self._se3_conf      = se3_conf

        self._diffuse_rot   = se3_conf.diffuse_rot
        self._so3_diffuser  = RandomFrames(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser   = SingleGaussian(self._se3_conf.r3)

        self.crd_scale = se3_conf.r3.coordinate_scaling


    def forward_marginal(self,
                         rigids_0       : ru.Rigid, 
                         t              : float,
                         diffuse_mask   : np.ndarray=None,
                         as_tensor_7    : bool=True):
        """
        Applies a single step of fixed noise to the input coordinates.

        Parameters:
            rigids_0 (openfold.utils.rigid_utils.Rigid): Series of openfold rigid objects 
            t (float): continuous time in [0, 1].
            diffuse_mask (np.ndarray): Mask denoting which tokens are subject to noising, and which are not. 
                                       NOTE: 1/True means 'revealed' (not diffused), 0/False means 'hidden' (diffused)
        """
        #### Translations #### 
        ######################

        # Only the CAs matter--spoof N/C
        L = len(rigids_0.get_trans())
        bb_crds = torch.zeros(L,3,3, dtype=torch.float) # (L,3,3) N,CA,C
        bb_crds[:,1] = rigids_0.get_trans()

        # scale the coordinates 
        bb_crds = bb_crds * self.crd_scale

        noised_crds, delta = self._r3_diffuser.diffuse_translations(bb_crds, 
                                                                    ~(diffuse_mask.bool()), 
                                                                    crd_scale=self.crd_scale)

        bb_crds = bb_crds / self.crd_scale # scale back
        delta = delta / self.crd_scale

        ####################################
        #### SO3 noise -- random frames ####
        rots_t = self._so3_diffuser._get_random_rotations(L).as_rotvec()

        trans_t = bb_crds[:,1] + delta

        # assemble output rigid 
        rigids_t = _assemble_rigid(rots_t, trans_t)


        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()

        outs = {'rigids_t'            :rigids_t,
                'trans_score'         :torch.tensor(0.0),
                'rot_score'           :torch.tensor(0.0),
                'trans_score_scaling' :torch.tensor(1.),
                'rot_score_scaling'   :torch.tensor(1.)}

        return outs