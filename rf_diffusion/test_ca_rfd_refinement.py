# tests for ca RFdiffusion refinement 
import torch 
import unittest 
import pickle 
import hydra 
from hydra import initialize, compose 
from unittest import mock 
from pathlib import Path 
import pdb 
pydb=pdb
import torch.nn as nn

# rfdiffusion imports 
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
import run_inference 
from test_ca_rfd_sm_train import ExitMockCall
import aa_model

relative_to_absolute = lambda rel: str(Path(__file__).parent / rel)

def rfold_side_effect(*args, **kwargs): 
    # mocks the fwd in LegacyRosettaFoldModule 
    raise ExitMockCall('Captured forward inputs')


def get_ca_config(overrides=[]):
    """
    Create a config for testing.

    Args:
        overrides: list of overrides for config
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    initialize(config_path="config/inference")
    conf = compose(config_name='test_ca_rfd_refinement.yaml', overrides=overrides, return_hydra_config=True)
    conf.inference.ckpt_path = relative_to_absolute(conf.inference.ckpt_path)
    return conf

make_indep_saved = aa_model.make_indep
def make_indep_side_effect(pdb, *args, **kwargs):
    """Resolve absolute path of pdb
    
    Args:
        pdb (str): path to pdb file
    """
    cur_folder = Path(__file__).parent
    pdb_in = str(cur_folder / pdb)
    ret = make_indep_saved(pdb_in, *args, **kwargs)
    return ret

#### Side effects and data catching #### 
PATCH_SAVE_DICT = {} # save data here 


class TestCARFDFeaturization(unittest.TestCase):
    """
    Tests for CA RFdiffusion inference.
    """

    @classmethod
    @mock.patch('rf_diffusion.aa_model.make_indep', side_effect=make_indep_side_effect)
    @mock.patch.object(LegacyRoseTTAFoldModule, 'forward', side_effect=rfold_side_effect)
    @mock.patch.object(nn.Module, 'load_state_dict')
    def setUpClass(cls, mock_load_state_dict, mock_rfold_fwd, mock_make_indep): 
        """
        Load goldens, get config, run inference all the way up to fwd pass. 
        Then, compare inputs at fwd pass with goldens.
        """

        cls.load_goldens() # load goldens 
        cls.conf = get_ca_config() # get config 

        current_folder = Path(__file__).parent
        cls.conf.inference.input_pdb = str( current_folder / cls.conf.inference.input_pdb )

        try: 
            run_inference.main(cls.conf)
        except ExitMockCall: 
            pass 

        # save inputs to live fwd pass
        assert len(mock_rfold_fwd.call_args_list) == 1
        cls.kall = mock_rfold_fwd.call_args_list[0]

        # all the things we caught with side effects 
        cls.patch_caught = PATCH_SAVE_DICT

    @classmethod
    def load_goldens(cls):
        """
        Load goldens to compare with.
        """
        current_folder = Path(__file__).parent
        goldens_path = current_folder / 'goldens/ca_rfd_refine_rfi_031825.pkl'
        with open(goldens_path, 'rb') as f:
            cls.golden_rfi = pickle.load(f)
    
    """
    dict_keys([
    'msa_latent', 
    'msa_full', 
    'seq', 
    'seq_unmasked', 
    'xyz', 
    'sctors', 
    'idx', 
    'bond_feats', 
    'dist_matrix', 
    'chirals', 
    'atom_frames', 
    't1d', 
    't2d', 
    'xyz_t', 
    'alpha_t', 
    'mask_t', 
    'same_chain', 
    'is_motif', 
    'msa_prev', 
    'pair_prev', 
    'state_prev'])
    """

    def test_msas(self): 
        # NOTE: This fails for now due to supplying/not suppling motif seq difference
        want_latent = self.golden_rfi['msa_latent']
        want_full = self.golden_rfi['msa_full']
        
        got_latent = self.kall.kwargs['msa_latent'].to(want_latent.device)
        got_full = self.kall.kwargs['msa_full'].to(want_latent.device)
        torch.testing.assert_allclose(got_latent, want_latent)

        torch.testing.assert_allclose(want_full, got_full)

    def test_seq(self):
        # NOTE: This fails for now due to supplying/not suppling motif seq difference
        want = self.golden_rfi['seq']
        got = self.kall.kwargs['seq'].to(want.device)
        torch.testing.assert_allclose(got,want)

        want = self.golden_rfi['seq_unmasked']
        got = self.kall.kwargs['seq_unmasked'].to(want.device)
        torch.testing.assert_allclose(got,want)

        

    # skipping test_xyz
    
    def test_sctors(self):
        want = self.golden_rfi['sctors']
        got = self.kall.kwargs['sctors'].to(want.device)
        torch.testing.assert_allclose(got,want)
    
    def test_idx(self):
        want = self.golden_rfi['idx']
        got = self.kall.kwargs['idx'].to(want.device)
        torch.testing.assert_allclose(got,want)
    
    def test_bond_feats(self):
        want = self.golden_rfi['bond_feats']
        got = self.kall.kwargs['bond_feats'].to(want.device)
        torch.testing.assert_allclose(got,want)
    
    def test_dist_matrix(self):
        want = self.golden_rfi['dist_matrix']
        got = self.kall.kwargs['dist_matrix'].to(want.device)
        torch.testing.assert_allclose(got,want)
    
    def test_chirals(self):
        want = self.golden_rfi['chirals']
        got = self.kall.kwargs['chirals'].to(want.device)
        torch.testing.assert_allclose(got,want)
    
    def test_atom_frames(self):
        want = self.golden_rfi['atom_frames']
        got = self.kall.kwargs['atom_frames'].to(want.device)
        torch.testing.assert_allclose(got,want)
    
    def test_t1d(self):
        want = self.golden_rfi['t1d']
        got = self.kall.kwargs['t1d'].to(want.device)
        torch.testing.assert_allclose(got,want)

    def test_t2d_motif_only(self):
        # only testing the motif template
        want = self.golden_rfi['t2d'][0,-1] # last tmplt is motif
        got = self.kall.kwargs['t2d'][0,-1].to(want.device) # last tmplt is motif
        
        is_angle_entry = torch.zeros(69).bool()
        is_angle_entry[61:67] = True # 61,62,63,64,65,66 only
        not_angle_entry = ~is_angle_entry

        got1 = got[...,is_angle_entry]
        want1 = want[...,is_angle_entry]
        # rtol doesn't really matter here if atol is smol
        torch.testing.assert_close(got1, want1, atol=6e-4, rtol=1) 

        # non angles - gotta be dead on
        got2 = got[...,not_angle_entry]
        want2 = want[...,not_angle_entry]
        torch.testing.assert_close(got2, want2)

    def test_xyz_t_motif(self):
        want = self.golden_rfi['xyz_t'][0,2]
        got = self.kall.kwargs['xyz_t'][0,2].to(want.device)

        # NOTE: As long as test_mask_t is passing, we only 
        # need to test the coordinates ON MOTIF.
        # This is because when a distance map is made 
        # in the template embedding, mask_t is used to zero out 
        # contributions (distances) from non-motif (i,j) pairs

        is_motif = self.golden_rfi['is_motif']
        want = want[is_motif]
        got = got[is_motif]
        torch.testing.assert_close(got, want)

    def test_mask_t(self):
        want = self.golden_rfi['mask_t']
        got = self.kall.kwargs['mask_t'].to(want.device)
        torch.testing.assert_close(got,want)

    def test_is_motif(self):
        want = self.golden_rfi['is_motif']
        got = self.kall.kwargs['is_motif'].to(want.device)
        torch.testing.assert_close(got, want)

    def test_prev(self):
        g = self.golden_rfi
        k = self.kall.kwargs

        want = g['msa_prev']
        got = k['msa_prev']

        assert want == got # None

        want = g['pair_prev']
        got = k['pair_prev']
        assert want == got # None

        want = g['state_prev']
        got = k['state_prev']
        assert want == got # None