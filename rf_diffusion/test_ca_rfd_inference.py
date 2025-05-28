# tests for ca RFdiffusion inference
import torch 
import unittest 
import pickle 
import hydra 
from hydra import initialize, compose 
from unittest import mock 
import copy
from pathlib import Path
import torch.nn as nn

import run_inference 
from rf_diffusion.preprocess import add_motif_template
from test_ca_rfd_sm_train import ExitMockCall
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule

relative_to_absolute = lambda rel: str(Path(__file__).parent / rel)

def get_ca_config(overrides=[]):
    """
    Create a config for testing.
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    initialize(config_path="config/inference")
    conf = compose(config_name='test_ca_rfd_inference.yaml', overrides=overrides, return_hydra_config=True)
    conf.inference.ckpt_path = relative_to_absolute(conf.inference.ckpt_path)
    return conf

def rfold_side_effect(*args, **kwargs):
    raise ExitMockCall("Captured forward inputs")

def spy_side_effect(*args):
    return None

PATCH_SAVE = {} # place to save intercepted objects

add_motif_template_saved = add_motif_template
def add_motif_tmplt_side_effect(rfi, t2d_motif, xyz_t_motif, masks_1d):   
    PATCH_SAVE['AMT_rfi'] = copy.deepcopy(rfi)
    motif_template = {'t2d':copy.deepcopy(t2d_motif), 'xyz_t':copy.deepcopy(xyz_t_motif)}
    PATCH_SAVE['AMT_motif_template'] = motif_template
    PATCH_SAVE['AMT_masks_1d'] = copy.deepcopy(masks_1d)
    return add_motif_template_saved(rfi, t2d_motif, xyz_t_motif, masks_1d)


class TestCARFDInference(unittest.TestCase):
    """
    Tests for CA RFdiffusion inference.
    """

    @classmethod
    @mock.patch.object(LegacyRoseTTAFoldModule, 'forward', side_effect=rfold_side_effect)
    @mock.patch('rf_diffusion.inference.model_runners.spy', side_effect=spy_side_effect)
    @mock.patch('rf_diffusion.inference.data_loader.spy', side_effect=spy_side_effect)
    @mock.patch('rf_diffusion.conditioning.spy', side_effect=spy_side_effect)
    @mock.patch('rf_diffusion.data_loader.spy', side_effect=spy_side_effect)
    @mock.patch('rf_diffusion.preprocess.add_motif_template', side_effect=add_motif_tmplt_side_effect)
    @mock.patch.object(nn.Module, 'load_state_dict')
    def setUpClass(cls, mock_load_state_dict,
                        mock_add_motif_template, 
                        mock_dataloader_spy, 
                        mock_conditioning_spy, 
                        mock_inf_dataloader_spy, 
                        mock_mr_spy, 
                        mock_rfold_fwd): 
        """Do everything normally, as if inference were really running, then catch the inputs to RosettaFold fwd"""
        
        cls.conf = get_ca_config()
        cls.load_goldens()
        
        # modify inference.input_pdb to accomodate working dir
        current_dir = Path(__file__).parent
        cls.conf.inference.input_pdb = current_dir / cls.conf.inference.input_pdb

        try: 
            run_inference.main(cls.conf)
        except ExitMockCall: 
            pass 
            
        # A single call to forward should have been made 
        assert len(mock_rfold_fwd.call_args_list) == 1
        cls.kall = mock_rfold_fwd.call_args_list[0]

        ### Extract information from spy functions ###
        # [0][0] is args, [0][1] is kwargs
        spy_call_args = mock_mr_spy.call_args_list[0][0] 
        PATCH_SAVE['sample_init_out'] = {'indep':spy_call_args[0],
                                         'contig_map':spy_call_args[1],
                                         'atomizer':spy_call_args[2],
                                         't_step_input':spy_call_args[3]}


        # spy in inference.data_loader
        spy2_call1 = mock_inf_dataloader_spy.call_args_list[0][0]
        PATCH_SAVE['indep_getitem_inner_point_A'] = spy2_call1[0]['indep_getitem_inner_point_A']
        spy2_call2 = mock_inf_dataloader_spy.call_args_list[1][0]
        PATCH_SAVE['indep_getitem_inner_point_B'] = spy2_call2[0]['indep_getitem_inner_point_B'] 

        # spy in conditioning.py 
        spy3_call1 = mock_conditioning_spy.call_args_list[0][0]
        PATCH_SAVE['indep_into_ComputeMotifTemplate'] = spy3_call1[0]['indep_into_ComputeMotifTemplate']

        # spy in data_loader.py 
        spy4_call1 = mock_dataloader_spy.call_args_list[0][0]
        PATCH_SAVE['feats_into_transform_stack'] = spy4_call1[0]['feats_into_transform_stack']

    
    @classmethod 
    def load_goldens(cls):
        """Load inference goldens to compare against"""
        # RFI - siteC motif scaffolding example
        current_dir = Path(__file__).parent
        rfi_siteC = current_dir / "goldens/ca_rfd_inference_rfi_t50.pkl" 
        cls.rfi_golden = pickle.load(open(rfi_siteC, 'rb')) # the truth

        # 

    def load_golden_sample_init(self):
        current_dir = Path(__file__).parent
        sample_init_pkl = current_dir / 'goldens/ca_rfd_inf_sample_init_out.pkl'
        with open(sample_init_pkl, 'rb') as f:
            want_indep = pickle.load(f)
        return want_indep
    
    def load_golden_indep_orig(self):
        current_dir = Path(__file__).parent
        pkl = current_dir / 'goldens/ca_rfd_inf_indep_orig.pkl'
        with open(pkl, 'rb') as f:
            want_indep = pickle.load(f)
        return want_indep
    
    def load_golden_indep_after_insert(self): 
        current_dir = Path(__file__).parent
        pkl = current_dir / 'goldens/ca_rfd_inf_indep_after_insert_contig.pkl'
        with open(pkl, 'rb') as f:
            want_indep = pickle.load(f)
        return want_indep
    
    def load_golden_compute_motif_template(self):
        current_dir = Path(__file__).parent
        pkl = current_dir / 'goldens/ca_rfd_inf_compute_motif_template_ins_outs.pkl'
        with open(pkl, 'rb') as f:
            want = pickle.load(f)
        return want

    """
    The data loading steps are: 
    1. PDBLoaderDataset parsers the pdb file and makes initial indep
    2. Within PDBLoaderDataset, contig is inserted. 
    3. Then, indep with contig inserted passes through these transforms:
        rf_diffusion.conditioning.GetTemplatedMotifMasks
        rf_diffusion.conditioning.ComputeMotifTemplate
        rf_diffusion.conditioning.CenterOnCA
        rf_diffusion.conditioning.AddConditionalInputs
        InferenceDataset.__init__.<locals>.update_inference_state
        InferenceDataset.__init__.<locals>.diffuse
        InferenceDataset.__init__.<locals>.feature_tuple_from_feature_dict
    """

    def test_indeps(self):

        ####### indeps that we want (from OG repo)
        # Golden: initial indep, right after parsing 
        want_indep_orig = self.load_golden_indep_orig()

        # Golden: indep after inserting contig elements
        want_indep = self.load_golden_indep_after_insert()
        want_ismotif = want_indep['indep.seq'] != 21


        ############# some checks on indeps we've collected
        # check if the original indeps contain correct motif coordinates 
        isnan = torch.isnan(want_indep_orig['indep.xyz'])
        torch.testing.assert_close(want_indep_orig['indep.xyz'][~isnan], 
                                   PATCH_SAVE['indep_getitem_inner_point_A'].xyz[~isnan])

        ###  check if motif coordinates are good after insert contig
        got = PATCH_SAVE['indep_getitem_inner_point_B'].xyz[want_ismotif]
        motif_is_ligand = PATCH_SAVE['indep_getitem_inner_point_B'].is_sm[want_ismotif]
        want = want_indep['indep.xyz'][want_ismotif]

        torch.testing.assert_close(got[~motif_is_ligand,:3,:], want[~motif_is_ligand,:3,:]) # all three bb atoms should match
        torch.testing.assert_close(got[motif_is_ligand,1,:], want[motif_is_ligand,1,:]) # only 'ca' position for the ligand atoms

        #### feats right before transform stack
        feats = PATCH_SAVE['feats_into_transform_stack']
        got = feats['indep'].xyz[want_ismotif]
        # same want as above 
        torch.testing.assert_close(got[~motif_is_ligand,:3,:], want[~motif_is_ligand,:3,:])
        torch.testing.assert_close(got[motif_is_ligand,1,:], want[motif_is_ligand,1,:])


        # input into compute motif template call
        got = PATCH_SAVE['indep_into_ComputeMotifTemplate'].xyz[want_ismotif]
        torch.testing.assert_close(got[~motif_is_ligand,:3,:], want[~motif_is_ligand,:3,:])
        torch.testing.assert_close(got[motif_is_ligand,1,:], want[motif_is_ligand,1,:])


        # check on preprocess.add_motif_template
        got_motif_template = PATCH_SAVE['AMT_motif_template']
        xyz_t_motif = got_motif_template['xyz_t'][want_ismotif]
        torch.testing.assert_close(xyz_t_motif, want[:,1,:])

        got_t2d_motif = got_motif_template['t2d']
        got_xyz_t_motif = got_motif_template['xyz_t']

        want_t2d_data = self.load_golden_compute_motif_template()
        want_t2d_motif        = want_t2d_data['t2d_motif']
        want_xyz_t_motif = want_t2d_data['xyz_xt_w_motif']
        # want_is_motif_2d      = want_t2d_data['t2d_is_reavealed']

        ### NOTE: Details
        # The last 7 entries of t2d are 3 angles x 2 sin/cos, followed by 1 bool
        # The The other entries are binned distances
        # The sin/cos entries seem to want to be different by about 0.001, or ~0.75 degrees
        # I (DJ) consider this okay, so will set different tolerances for different regions
        # of the T2d. 
        # Angles can have slightly less precision, distances/bools must be dead on
        is_angle_entry = torch.zeros(68).bool()
        is_angle_entry[61:67] = True # 61,62,63,64,65,66
        not_angle_entry = ~is_angle_entry

        got = got_t2d_motif[...,is_angle_entry]
        want = want_t2d_motif[...,is_angle_entry]
        # rtol doesn't really matter here if atol is small
        torch.testing.assert_close(got, want, atol=6e-4, rtol=1)
        
        got = got_t2d_motif[...,not_angle_entry]
        want = want_t2d_motif[...,not_angle_entry]
        torch.testing.assert_close(got, want)

        ## Looking at xyz_xt_w_motif that we want 
        torch.testing.assert_close(want_xyz_t_motif[0,want_ismotif,1,:], 
                                   got_xyz_t_motif[want_ismotif,:])

    def get_key(self, key):
        want = self.rfi_golden[key]
        got = self.kall.kwargs[key]
        return want, got


    def test_msa_latent(self):
        want, got = self.get_key('msa_latent')
        torch.testing.assert_close(want,got)

    def test_msa_full(self):
        want,got = self.get_key('msa_full')
        torch.testing.assert_close(want,got)
    
    def test_seq(self):
        want,got = self.get_key('seq')
        torch.testing.assert_close(want,got)
    
    def test_seq_unmasked(self):
        want,got = self.get_key('seq_unmasked')
        torch.testing.assert_close(want,got)
    
    # def test_xyz(self):
    #     # this one will be challenging
    #     want,got = self.get_key('xyz')
    #     torch.testing.assert_close(want,got)

    def test_sctors(self):
        want,got = self.get_key('sctors')
        torch.testing.assert_close(want,got)
    
    def test_idx(self):
        want,got = self.get_key('idx')
        torch.testing.assert_close(want,got)
    
    def test_bond_feats(self):
        want,got = self.get_key('bond_feats')
        torch.testing.assert_close(want,got)
    
    def test_dist_matrix(self):
        want,got = self.get_key('dist_matrix')
        torch.testing.assert_close(want,got)
    
    def test_chirals(self):
        want,got = self.get_key('chirals')
        torch.testing.assert_close(want,got)
    
    def test_atom_frames(self):
        want,got = self.get_key('atom_frames')
        torch.testing.assert_close(want,got)
    
    def test_t1d(self):
        want,got = self.get_key('t1d')
        torch.testing.assert_close(want,got)

    def test_t2d_motif_only(self):
        want = self.rfi_golden['t2d'][0,2] # [0,2] is motif template only 
        got = self.kall.kwargs['t2d'][0,2]

        is_angle_entry = torch.zeros(69).bool()
        is_angle_entry[61:67] = True # 61,62,63,64,65,66 only
        not_angle_entry = ~is_angle_entry

        got1 = got[...,is_angle_entry]
        want1 = want[...,is_angle_entry]
        # rtol doesn't really matter here if atol is small
        torch.testing.assert_close(got1, want1, atol=6e-4, rtol=1)


        # non angles need to be essentially exact
        got2 = got[...,not_angle_entry]
        want2 = want[...,not_angle_entry]
        torch.testing.assert_close(got2, want2)

    # def test_t2d(self): 
    #     want = self.rfi_golden['t2d']
    #     got = self.kall.kwargs['t2d']
    #     torch.testing.assert_close(want, got) 

    def test_xyz_t_motif(self):
        want = self.rfi_golden['xyz_t'][0,2]
        got = self.kall.kwargs['xyz_t'][0,2]
        
        want_indep = self.load_golden_indep_after_insert()
        ismotif = want_indep['indep.seq'] != 21

        torch.testing.assert_close(want[ismotif], got[ismotif])
    
    # def test_alpha_t(self):
    #     want,got = self.get_key('alpha_t')
    #     torch.testing.assert_close(want,got)
    
    def test_mask_t(self):
        want,got = self.get_key('mask_t')
        torch.testing.assert_close(want,got)

    def test_same_chain(self):
        want,got = self.get_key('same_chain')
        torch.testing.assert_close(want,got)
    
    def test_is_motif(self):
        want,got = self.get_key('is_motif')
        torch.testing.assert_close(want,got)
    
    def test_prev(self):
        want_msa_prev, got_msa_prev = self.get_key('msa_prev')
        assert want_msa_prev == got_msa_prev

        want_pair_prev, got_pair_prev = self.get_key('pair_prev')
        assert want_pair_prev == got_pair_prev 

        want_state_prev, got_state_prev = self.get_key('state_prev')
        assert want_state_prev == got_state_prev

    
