# script for diffusion protocols 
import torch 
import pickle
import numpy as np
import os







torch.set_printoptions(sci_mode=False)

def cosine_interp(T, eta_max, eta_min):
    """
    Cosine interpolation of some value between its max <eta_max> and its min <eta_min>

    from https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    
    Parameters:
        T (int, required): total number of steps 
        eta_max (float, required): Max value of some parameter eta 
        eta_min (float, required): Min value of some parameter eta 
    """
    
    t = torch.arange(T)
    out = eta_max + 0.5*(eta_min-eta_max)*(1+torch.cos((t/T)*np.pi))
    return out 

def get_chi_betaT(max_timestep=100, beta_0=0.01, abar_T=1e-3, method='cosine'):
    """
    Function to precalculate beta_T for chi angles (decoded at different time steps, so T in beta_T varies).
    Calculated empirically
    """

    name = f'./cached_schedules/T{max_timestep}_beta_0{beta_0}_abar_T{abar_T}_method_{method}.pkl'

    if not os.path.exists(name):
        print('Calculating chi_beta_T dictionary...')

        if method not in ['cosine', 'linear']:
            raise NotImplementedError("Only cosine and linear interpolations are implemented for chi angle beta schedule")
        beta_Ts = {1:1.}
        for timestep in range(2,101):
            best=999.99
            for i in torch.linspace(beta_0,0.999,5000): #sampling bT
                if method == 'cosine':
                    interp = cosine_interp(timestep, i, beta_0)
                elif method == 'linear':
                    interp = torch.linspace(beta_0, i, timestep)
                temp = torch.cumprod(1-interp, dim=0)
                if torch.abs(temp[-1] - abar_T) < best:
                    best = temp[-1] - abar_T
                    idx = i
            beta_Ts[timestep] = idx.item()

        # save cached schedule
        if not os.path.isdir('./cached_schedules/'):
            os.makedirs('./cached_schedules/')
        with open(name, 'wb') as fp:
            pickle.dump(beta_Ts, fp)

        print('Done calculating chi_beta_T dictionaries. They are now cached.')

    else:
        print('Using cached chi_beta_T dictionary.')
        with open(name, 'rb') as fp:
            beta_Ts = pickle.load(fp)


    print('Done calculating chi_beta_T, chi_alphas_T, and chi_abars_T dictionaries.')
    return beta_Ts

def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}, inference=False):
    """
    Given a noise schedule type, create the beta schedule 
    """
    assert schedule_type in ['linear', 'geometric', 'cosine']

    # linear noise schedule 
    if schedule_type == 'linear':
        schedule = torch.linspace(b0, bT, T) 

    # geometric noise schedule 
    elif schedule_type == 'geometric': 
        raise NotImplementedError('geometric schedule not ready yet')
    
    # cosine noise schedule 
    else:
        schedule = cosine_interp(T, bT, b0) 
    
    
    #get alphabar_t for convenience
    alpha_schedule = 1-schedule
    alphabar_t_schedule  = torch.cumprod(alpha_schedule, dim=0)
    
    if inference:
        print(f"With this beta schedule ({schedule_type} schedule, beta_0 = {b0}, beta_T = {bT}), alpha_bar_T = {alphabar_t_schedule[-1]}")

    return schedule, alpha_schedule, alphabar_t_schedule 


class EuclideanDiffuser():
    # class for diffusing points 

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

    
    # NOTE: this one seems fishy - doesn't match apply_kernel
    #def apply_kernel_closed(self, x0, t, var_scale=1, mask=None):
    #    """
    #    Applies a noising kernel to the points in x 

    #    Parameters:
    #        x0 (torch.tensor, required): (N,3,3) set of backbone coordinates from ORIGINAL backbone 

    #        t (int, required): Which timestep

    #        noise_scale (float, required): scale for noise 
    #    """
    #    t_idx = t-1 # bring from 1-indexed to 0-indexed

    #    assert len(x0.shape) == 3
    #    L,_,_ = x0.shape 

    #    # c-alpha crds 
    #    ca_xyz = x0[:,1,:]


    #    b_t = self.beta_schedule[t_idx]    
    #    a_t = self.alpha_schedule[t_idx]


    #    # get the noise at timestep t
    #    a_bar = torch.prod(self.alpha_schedule[:t_idx], dim=0)

    #    mean  = torch.sqrt(a_bar)*ca_xyz 
    #    var   = torch.ones(L,3)*(1-a_bar)*var_scale


    #    sampled_crds = torch.normal(mean, var)
    #    delta = sampled_crds - ca_xyz

    #    if mask != None:
    #        delta[mask,...] = 0

    #    out_crds = x0 + delta[:,None,:]

    #    return out_crds 


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
        

        return torch.stack(bb_stack).transpose(0,1), torch.stack(T_stack).transpose(0,1)

        
#TODO:  This class uses scipy+numpy for the slerping/matrix generation 
#       Probably could be much faster if everything was in torch

def write_pkl(save_path: str, pkl_data):
    """Serialize data into a pickle file."""
    with open(save_path, 'wb') as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f'Failed to read {read_path}')
            raise(e)
