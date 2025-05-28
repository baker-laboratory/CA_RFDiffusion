"""
Tests for CA rfdiffusion and 2d motif templating etc.. Written by DJ 
"""
import torch 
import unittest 
import pickle 
import hydra 
from hydra import initialize, compose 
import unittest.mock as mock 
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
import rf_diffusion.frame_diffusion
from rf_diffusion import data_loader
import train_multi_deep
import rf2aa 
import rf_diffusion.aa_model
from pathlib import Path 
from functools import partial 
import os

# Custom tolerance for coordinates with torch.testing.assert_close
th_assertclose_for_xyz = partial(torch.testing.assert_close, atol=5e-5, rtol=0.002)
relative_to_absolute = lambda rel: str(Path(__file__).parent / rel)

class ExitMockCall(Exception):
    # an exception we can look out for during mocking 
    pass 

def get_ca_config(overrides=[]):
    """
    Create a config for testing.
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    initialize(config_path="config/training")
    conf = compose(config_name='test_ca_rfd_unconditional_train.yaml', overrides=overrides, return_hydra_config=True)
    conf.dataloader.DATAPKL_AA = relative_to_absolute(conf.dataloader.DATAPKL_AA)
    return conf

def rfold_side_effect(*args, **kwargs): 
    # mocks the fwd in LegacyRosettaFoldModule 
    raise ExitMockCall('Captured forward inputs')


# Save the original functions for later in use mock side effect to avoid recursive calls 
get_ang_saved = rf2aa.kinematics.get_ang
get_dih_saved = rf2aa.kinematics.get_dih
wrap_featurize_saved = data_loader.wrap_featurize
get_mu_xt_x0_saved = rf_diffusion.frame_diffusion.data.legacy_diffuser.get_mu_xt_x0
get_next_ca_saved = rf_diffusion.frame_diffusion.data.legacy_diffuser.get_next_ca
diffuse_saved = rf_diffusion.aa_model.diffuse

PATCH_SAVE_DICT = {}

def diffuse_side_effect(*args, **kwargs):
    """
    Catch inputs/outputs of aa_model.diffuse
    """
    assert args[4] == 77
    new_args = list(args)
    new_args[4] = 78

    diffuse_return = diffuse_saved(*new_args, **kwargs)
    PATCH_SAVE_DICT['diffuse_return'] = diffuse_return
    PATCH_SAVE_DICT['diffuse_input_args'] = args # (conf, diffuser, indep, is_diffused, t)
    return diffuse_return


def get_next_ca_side_effect(*args, **kwargs):
    """
    Catch inoputs/outputs of get_next_ca.
    """
    o1, o2 = get_next_ca_saved(*args, **kwargs)
    PATCH_SAVE_DICT['get_next_ca_input_xt']  = kwargs['xt']
    PATCH_SAVE_DICT['get_next_ca_input_px0'] = kwargs['px0']
    PATCH_SAVE_DICT['get_next_ca_input_t']   = kwargs['t']


    PATCH_SAVE_DICT['get_next_ca_out_crds'] = o1
    PATCH_SAVE_DICT['get_next_ca_out_deltas'] = o2
    return o1, o2


def get_mu_xt_x0_side_effect(*args, **kwargs):
    """
    Catch inputs/outputs of get_mu_xt_x0
    """
    mu, sigma = get_mu_xt_x0_saved(*args, **kwargs)
    PATCH_SAVE_DICT['mu'] = mu
    PATCH_SAVE_DICT['sigma'] = sigma
    return mu, sigma


def wrap_feat_side_effect(**kwargs):
    """
    Catch inputs/outputs of wrap_featurize
    """
    t = kwargs['t']
    assert t == 77
    kwargs['t'] = 78

    rfi = wrap_featurize_saved(**kwargs)
    return rfi


def get_ang_side_effect(*args, **kwargs):
    kwargs['eps'] = 1e-6
    return get_ang_saved(*args, **kwargs)


def get_dih_side_effect(*args, **kwargs):
    kwargs['eps'] = 1e-6
    return get_dih_saved(*args, **kwargs)

     

class TestFeaturization(unittest.TestCase): 
    """
    Tests for featurization of inputs. Written by DJ
    """
    @classmethod
    @mock.patch('rf2aa.kinematics.get_dih',                                         side_effect=get_dih_side_effect)
    @mock.patch('rf2aa.kinematics.get_ang',                                         side_effect=get_ang_side_effect)
    @mock.patch('rf_diffusion.data_loader.wrap_featurize',                          side_effect=wrap_feat_side_effect)
    @mock.patch.object(LegacyRoseTTAFoldModule, 'forward',                          side_effect=rfold_side_effect)
    @mock.patch('rf_diffusion.aa_model.diffuse',                                    side_effect=diffuse_side_effect)
    @mock.patch('rf_diffusion.frame_diffusion.data.legacy_diffuser.get_next_ca',    side_effect=get_next_ca_side_effect)
    @mock.patch('rf_diffusion.frame_diffusion.data.legacy_diffuser.get_mu_xt_x0',   side_effect=get_mu_xt_x0_side_effect)
    @mock.patch('rf2aa.data.data_loader.sample_item')
    def setUpClass(cls, mock_sample_item, 
                        mock_get_mu_xt_x0, 
                        mock_get_next_ca, 
                        mock_diffuse, 
                        mock_rfold_fwd, 
                        mock_wrap_featurize, 
                        mock_get_ang,
                        mock_get_dih): 
        """
        Runs fake model training all the way until the first forward call.
        Analyzes inputs to the forward call and other side effects.
        """
        cls.conf = get_ca_config()
        cls.trainer = train_multi_deep.make_trainer(cls.conf)
        cls.load_goldens()
        os.environ['MASTER_PORT'] = '12009'
        os.environ['MASTER_ADDR'] = 'localhost'

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            cls.trainer.run_model_training(torch.cuda.device_count())
        except ExitMockCall:
            pass

        # A single call to forward should have been made
        assert len(mock_rfold_fwd.call_args_list) == 1
        cls.kall = mock_rfold_fwd.call_args_list[0]

        # all the things we caught with the side effects 
        cls.patch_caught = PATCH_SAVE_DICT

    @classmethod
    def tearDownClass(cls):
        hydra.core.global_hydra.GlobalHydra().clear()
        torch.distributed.destroy_process_group()

    @classmethod
    def load_goldens(cls):
        """
        Load all goldens here.
        """
        # RFIs 
        cls.rfi_77_golden = pickle.load(open(relative_to_absolute('goldens/ca_rfd_golden_rfi_t_dict.pkl'), 'rb'))
        cls.rfi_78_golden = pickle.load(open(relative_to_absolute('goldens/ca_rfd_golden_rfi_tp1_dict.pkl'), 'rb'))
    
        # diffusion stuff 
        cls.golden_xyz_into_diffuse     = torch.load(relative_to_absolute('goldens/ca_rfd_diffuse_xyz_in.pt'),       weights_only=True)
        cls.golden_indep_diffused_78    = torch.load(relative_to_absolute('goldens/ca_rfd_indep_diffused_78.pt'),    weights_only=True)
        cls.golden_indep_diffused_77    = torch.load(relative_to_absolute('goldens/ca_rfd_indep_diffused_77.pt'),    weights_only=True)
        cls.golden_mu_sigma             = torch.load(relative_to_absolute('goldens/ca_rfd_mu_sigma.pt'),             weights_only=True)
        cls.golden_get_next_ca_out_crds = torch.load(relative_to_absolute('goldens/ca_rfd_get_next_ca_out_crds.pt'), weights_only=True)


    def test_diffusion_inputs(self):
        """
        Test the inputs to diffusion.
        """
        want = self.golden_xyz_into_diffuse[:,1,:]
        got  = self.patch_caught['diffuse_input_args'][2].xyz[:,1,:] # the indep xyz which went into aa_model.diffuse 

        torch.testing.assert_close(got, want)


    # def test_mu_sigma(self):
    #     """
    #     Test the mu and sigma inputs to get_mu_xt_x0
    #     """
    #     want_mu = self.golden_mu_sigma['mu']
    #     want_sigma = self.golden_mu_sigma['sigma']

    #     got_mu = self.patch_caught['mu']
    #     got_sigma = self.patch_caught['sigma']

    #     torch.testing.assert_close(got_sigma, want_sigma)
    #     torch.testing.assert_close(got_mu, want_mu)


    # def test_next_ca_output(self): 
    #     """
    #     Test the output of get_next_ca
    #     """
    #     got_crds = self.patch_caught['get_next_ca_out_crds'][:,1,:]
    #     want_crds = self.golden_get_next_ca_out_crds[:,1,:]

    #     torch.testing.assert_close(got_crds, want_crds, atol=5e-5, rtol=0.0001)


    # def test_raw_diffusion_output(self): 
    #     """
    #     Test the output of the diffusion function. 
    #     """
    #     diffused_indep, diffuser_out = self.patch_caught['diffuse_return']
        
    #     got_diffused_xyz_78  = diffused_indep.xyz[:,1,:] # the xyz at t=78
    #     want_diffused_xyz_78 = self.golden_indep_diffused_78[:,1,:]

    #     # ensure t=78 outputs are same 
    #     th_assertclose_for_xyz(got_diffused_xyz_78, want_diffused_xyz_78)

    #     # ensure the t=78 output from diffusion is identical to what got passed into get_next_ca
    #     torch.testing.assert_close(got_diffused_xyz_78, 
    #                                self.patch_caught['get_next_ca_input_xt'][:,1])
        
    #     # ensure t=78 during get next ca 
    #     self.assertEqual(78, self.patch_caught['get_next_ca_input_t'])


    #     got_diffused_xyz_77 = diffuser_out['x_t_minus_1'][:,1,:] # the xyz at t=77
    #     want_diffused_xyz_77 = self.golden_indep_diffused_77[:,1,:]

    #     th_assertclose_for_xyz(got_diffused_xyz_77, want_diffused_xyz_77)
    
    ######################################
    # Testing the inputs to forward call #
    ###################################### 
            
    def test_msas(self):
        want_latent = self.rfi_78_golden['msa_latent']
        want_full   = self.rfi_78_golden['msa_full']

        got_latent = self.kall.kwargs['msa_latent']
        got_full   = self.kall.kwargs['msa_full']

        torch.testing.assert_close(got_latent, want_latent)
        torch.testing.assert_close(got_full, want_full)


    def test_seq(self):
        want = self.rfi_78_golden['seq']
        want_unmasked = self.rfi_78_golden['seq_unmasked']

        got = self.kall.kwargs['seq']
        got_unmasked = self.kall.kwargs['seq_unmasked']

        torch.testing.assert_close(got, want)
        torch.testing.assert_close(got_unmasked, want_unmasked)


    # def test_xyz(self):
    #     want = self.rfi_78_golden['xyz']
    #     nans_want = torch.isnan(want).sum()

    #     got = self.kall.kwargs['xyz']
    #     nans_got = torch.isnan(got).sum()

    #     # before proceeding with nan removal, ensure they both have the same positions marked nan 
    #     self.assertEqual(nans_got, nans_want)
        
    #     # remove nans so comparrison works 
    #     want = torch.nan_to_num(want)[0,:,1,:]
    #     got  = torch.nan_to_num(got)[0,:,1,:]

    #     th_assertclose_for_xyz(got, want)


    def test_sctors(self):
        want = self.rfi_78_golden['sctors']
        got  = self.kall.kwargs['sctors']

        torch.testing.assert_close(got, want)


    def test_idx(self): 
        want = self.rfi_78_golden['idx']
        got = self.kall.kwargs['idx']
        torch.testing.assert_close(got, want)


    def test_bond_feats(self): 
        want = self.rfi_78_golden['bond_feats']
        got  = self.kall.kwargs['bond_feats']
        torch.testing.assert_close(got, want)


    def test_dist_matrix(self):
        want = self.rfi_78_golden['dist_matrix']
        got = self.kall.kwargs['dist_matrix']
        torch.testing.assert_close(got, want)


    def test_chirals(self):
        want = self.rfi_78_golden['chirals']
        got = self.kall.kwargs['chirals']
        torch.testing.assert_close(got, want)


    def test_atom_frames(self):
        want = self.rfi_78_golden['atom_frames']
        got = self.kall.kwargs['atom_frames']
        torch.testing.assert_close(got.to(dtype=want.dtype), want)


    def test_t1d(self):
        want = self.rfi_78_golden['t1d']
        got = self.kall.kwargs['t1d']
        torch.testing.assert_close(got, want)


    def test_t2d_motif_only(self): 
        want = self.rfi_78_golden['t2d'][0,2] # motif template only
        got = self.kall.kwargs['t2d'][0,2]
        torch.testing.assert_close(got, want)


    # def test_t2d(self): 
    #     # whole t2d
    #     want = self.rfi_78_golden['t2d']
    #     got = self.kall.kwargs['t2d']
    #     torch.testing.assert_close(got, want, atol=3e-5, rtol=0.0007)


    # def test_xyz_t(self):
    #     want = self.rfi_78_golden['xyz_t'] # includes the motif template 
    #     got = self.kall.kwargs['xyz_t']
    #     th_assertclose_for_xyz(got, want)


    # def test_alpha_t(self): 
    #     want = self.rfi_78_golden['alpha_t']
    #     got = self.kall.kwargs['alpha_t']
    #     torch.testing.assert_close(got, want, atol=0.045, rtol=0.05) 
    

    def test_mask_t(self): 
        want = self.rfi_78_golden['mask_t']
        got = self.kall.kwargs['mask_t']
        torch.testing.assert_close(got, want)


    def test_same_chain(self): 
        want = self.rfi_78_golden['same_chain']
        got = self.kall.kwargs['same_chain']
        torch.testing.assert_close(got.to(dtype=want.dtype), want)


    def test_is_motif(self): 
        want = self.rfi_78_golden['is_motif']
        got = self.kall.kwargs['is_motif']
        torch.testing.assert_close(got, want)

    def test_prev(self): 
        want_msa_prev   = self.rfi_78_golden['msa_prev']
        want_pair_prev  = self.rfi_78_golden['pair_prev']
        want_state_prev = self.rfi_78_golden['state_prev']

        got_msa_prev = self.kall.kwargs['msa_prev']
        got_pair_prev = self.kall.kwargs['pair_prev']
        got_state_prev = self.kall.kwargs['state_prev']

        torch.testing.assert_close(got_msa_prev, want_msa_prev)
        torch.testing.assert_close(got_pair_prev, want_pair_prev)
        torch.testing.assert_close(got_state_prev, want_state_prev)

