import argparse
import pprint
from typing import OrderedDict
import mask_generator
import data_loader
import metrics
from icecream import ic




TRUNK_PARAMS = [
    'n_extra_block', 'n_main_block', 'n_ref_block', 'n_finetune_block',
    'd_msa', 'd_msa_full', 'd_pair', 'd_templ', 'n_head_msa', 'n_head_pair',
    'n_head_templ', 'd_hidden', 'd_hidden_templ', 'p_drop',
    'use_extra_l1', 'use_atom_frames', 'freeze_track_motif'
]
SE3_PARAMS = [
    'num_layers', 'num_channels', 'num_degrees', 'n_heads', 'div',
    'l0_in_features', 'l0_out_features', 'l1_in_features', 'l1_out_features',
    'num_edge_features'
]

def get_args(in_args=None):
    parser = argparse.ArgumentParser()

    # training parameters
    train_group = parser.add_argument_group("training parameters")
    train_group.add_argument("-model_name", default="BFF",
            help="model name for saving")
    train_group.add_argument('-ckpt_load_path', default=None, 
            help='Path for loading model checkpoint')
    train_group.add_argument('-batch_size', type=int, default=1,
            help="Batch size [1]")
    train_group.add_argument('-lr', type=float, default=1.0e-3, 
            help="Learning rate [1.0e-3]")
    train_group.add_argument('-num_epochs', type=int, default=200,
            help="Number of epochs [200]")
    train_group.add_argument("-port", type=int, default=12319,
            help="PORT for ddp training, should be randomized [12319]")
    train_group.add_argument("-seed", type=int, default=0,
            help="seed for random number, should be randomized for different training run [0]")
    train_group.add_argument("-accum", type=int, default=1,
            help="Gradient accumulation when it's > 1 [1]")
    train_group.add_argument("-interactive", action="store_true", default=False,
            help="Use interactive node")
    train_group.add_argument("-zero_weights", action="store_true", default=False,
            help="Train with all weights in the model set to zero")
    train_group.add_argument('-debug', default=False, action='store_true', 
            help="If true, will set script to debug mode")
    train_group.add_argument('-no_wandb', dest='wandb', default=True, action='store_false', 
            help="If passed, will not use wandb")
    train_group.add_argument('-epoch_size', default=25600, action='store', type=int,
            help='Number of examples per epoch (and thus between saving epochs')
    train_group.add_argument('-verbose_checks', default=False, action='store_true', 
            help="If true, will set model to do sanity checks on inputs")
    train_group.add_argument("-out_dir", default=None,
            help="output directory")

    # data-loading parameters
    data_group = parser.add_argument_group("data loading parameters")
    data_group.add_argument('-maxseq', type=int, default=1024,
            help="Maximum depth of subsampled MSA [1024]")
    data_group.add_argument('-maxlat', type=int, default=128,
            help="Maximum depth of subsampled MSA [128]")
    data_group.add_argument("-crop", type=int, default=256,
            help="Upper limit of crop size [256]")
    data_group.add_argument('-mintplt', type=int, default=0,
            help="Minimum number of templates to select [0]")
    data_group.add_argument('-maxtplt', type=int, default=4,
            help="maximum number of templates to select [4]")
    data_group.add_argument("-rescut", type=float, default=5.0,
            help="Resolution cutoff [5.0]")
    data_group.add_argument("-datcut", default="2020-Apr-30",
            help="PDB release date cutoff [2020-Apr-30]")
    data_group.add_argument('-plddtcut', type=float, default=70.0,
            help="pLDDT cutoff for distillation set [70.0]")
    data_group.add_argument('-seqid', type=float, default=99.0,
            help="maximum sequence identity cutoff for template selection [99.0]")
    data_group.add_argument('-maxcycle', type=int, default=4,
            help="maximum number of recycle [4]")
    data_group.add_argument("-hal_mask_low", type=int, default=10,
            help='Smallest number of residues to mask out for a hal example')
    data_group.add_argument("-hal_mask_high", type=int, default=35,
            help='Largest number of residues to mask out for a hal example')
    data_group.add_argument("-hal_mask_low_ar", type=int, default=20,
            help='Smallest number of residues to mask out for a hal_ar example')
    data_group.add_argument("-hal_mask_high_ar", type=int, default=50,
            help='Largest number of residues to mask out for a hal_ar example')
    data_group.add_argument("-complex_hal_mask_low", type=int, default=10,
            help='Smallest number of residues to mask out for a complex_hal example')
    data_group.add_argument("-complex_hal_mask_high", type=int, default=35,
            help='Largest number of residues to mask out for a complex_hal example')
    data_group.add_argument("-complex_hal_mask_low_ar", type=int, default=20,
            help='Smallest number of residues to mask out for a complex_hal_ar example')
    data_group.add_argument("-complex_hal_mask_high_ar", type=int, default=50,
            help='Largest number of residues to mask out for a complex_hal_ar example')
    data_group.add_argument('-flank_low',type=int, default=3,
            help = 'Smallest number of flanking residues to mask for a hal example')
    data_group.add_argument('-flank_high',type=int, default=6,
            help = 'Largest number of flanking residues to mask for a hal example')
    data_group.add_argument('-str2seq_full_low',type=float, default=0.9,
            help = 'Minimum fraction to be masked in str2seq_full task. Default=0.9')
    data_group.add_argument('-str2seq_full_high',type=float, default=1.0,
            help = 'Maximum fraction to be masked in str2seq_full task. Default=1.0')
    data_group.add_argument('-dataset',type=str, required=True,
            help = 'Select dataset(s) to use. No default so make up your mind. Options are ["cn","pdb","fb","complex"]. Specify in a list: e.g. -dataset cn,pdb,fb')
    data_group.add_argument('-dataset_prob',type=str, default=None,
            help = 'Select proportion of examples from each dataset. Default behaviour is uniform. Specify like 0.2,0.4,0.4. Must sum to 1')
    data_group.add_argument('-mask_min_proportion',type=float,default=0.2,
            help = 'When doing motif scaffolding in training, what is the minimum proportion of the protein you want to mask? Default is 0.2.')
    data_group.add_argument('-mask_max_proportion',type=float,default=1.0,
            help = 'When doing motif scaffodling in training, what is the maximum proportion of the protein you want to mask? Default is 1.0')   
    data_group.add_argument('-mask_broken_proportion',type=float,default=0.5,
            help = 'When doing motif scaffolding, what proportion of the time do you want the motif to be "broken" into two (mask in the middle), vs the mask being split over the termini. Default = 0.5')
    data_group.add_argument('-data_pkl', type=str, default='./dataset.pkl', 
            help='Path to pickled dataset to load for training on. If path doesn\'t exist, will write new pickle with that name.')
    data_group.add_argument('-data_pkl_aa', type=str, default='./all-atom-dataset.pkl', 
            help='Path to pickled dataset to load for training on. If path doesn\'t exist, will write new pickle with that name.')
    data_group.add_argument('-spoof_item', type=str, default='', 
            help='Path to pickled dataset to load for training on. If path doesn\'t exist, will write new pickle with that name.')
    data_group.add_argument('-mol_dir', type=str, default=data_loader.USE_DEFAULT)
    data_group.add_argument("-discontiguous_crop", default="True", choices=("True","False"))

    # Diffusion args 
    diff_group = parser.add_argument_group("diffusion parameters")
    def parse_mask_str(s):
        parsed = OrderedDict()
        for k,v in [e.split(':') for e in s.split(',')]:
            parsed[getattr(mask_generator, k)] = float(v)
        assert sum(parsed.values()) == 1, f'mask function probabilities must sum to 1, got: {parsed}'
        return parsed
    diff_group.add_argument('-diff_mask_probs', type=parse_mask_str, default={mask_generator.get_diffusion_mask_simple: 1.0},
        help='functions in mask generator to use for diffusion masking and their probabilities.  Example: get_double_contact:0.2,get_triple_contact:0.8')
    diff_group.add_argument('-diff_mask_low', type=int, default=20,
            help='Minimum number of residues to diffuse if doing diffusion. Default 20')
    diff_group.add_argument('-diff_mask_high', type=int, default=999, 
            help='Maximum number of residues to diffuse if doing diffusion. Default 999 (all)')
    diff_group.add_argument('-diff_b0', type=float, default=1e-2, 
            help='b_0 paramter for Euclidean diffuser.')
    diff_group.add_argument('-diff_bT', type=float, default=7e-2, 
            help='b_T parameter for Euclidean diffuser.')
    diff_group.add_argument('-diff_schedule_type', type=str, default='linear', 
            help='Type of schedule for (Euclidean) diffusion.')
    diff_group.add_argument('-diff_so3_type', type=str, default='igso3',
            help='Which type of SO3 diffusion to use. Default igso3')
    diff_group.add_argument('-diff_so3_schedule_type',type=str, default='linear',
            help='Which schedule type do you want for the igso3')
    diff_group.add_argument('-diff_min_b',type=float, default=1.5,
            help='min_b paramater for igso3 diffusion')
    diff_group.add_argument('-diff_max_b',type=float, default=2.5,
            help='max_b paramater for igso3 diffusion')
    diff_group.add_argument('-diff_min_sigma',type=float, default=0.02,
            help='min_sigma paramater for igso3 diffusion')
    diff_group.add_argument('-diff_max_sigma',type=float, default=1.5,
            help='max_sigma paramater for igso3 diffusion')
    diff_group.add_argument('-diff_chi_type', type=str, default='interp',
            help='Which type of chi angle diffusion to use. Default linear interpolation.')
    diff_group.add_argument('-diff_T', type=int, default=200, 
            help='Total number of diffusion steps for forward diffusion.')
    diff_group.add_argument('-aa_decode_steps', type=int, default=40, 
            help='Total number of steps to decode amino acid identities and chi angles over.')
    diff_group.add_argument('-predict_previous', action='store_true',
            help='If True, model predictions x_t-1 instead of x0')
    diff_group.add_argument('-seqdiff_b0', type=float, default=0.001,
            help='b_0 parameter for Sequence diffuser.')
    diff_group.add_argument('-seqdiff_bT', type=float, default=0.1,
            help='b_T parameter for Sequence diffuser.')
    diff_group.add_argument('-seqdiff_schedule_type', type=str, default='cosine',
            help='Type of schedule for Sequence diffusion')
    diff_group.add_argument('-seqdiff_loss_type', type=str, default='l2_loss',
            help='Type of loss to use with sequence diffusion {l2_loss, sigmoid}')
    diff_group.add_argument('-seqdiff', type=str, default=None,
            help='The type of sequence diffuser to use ["uniform", "continuous"]. Default: None (classic autoregressive decoding)')
    diff_group.add_argument('-seqdiff_lambda', type=float, default=1,
            help='Lamda parameter used to weight seq_aux and seq_vb for discrete sequence diffusion')
    diff_group.add_argument('-decode_mask_frac', type=float, default=0.0,
            help='Fraction of decoded+diffused residues exposed to potential mutations')
    diff_group.add_argument('-decode_corrupt_blosum', type=float, default=0.9, 
            help='Fraction of the time to mutate according to BLOSUM62 transitions.')
    diff_group.add_argument('-decode_corrupt_uniform', type=float, default=0.1,
            help='Fraction of the time to mutate according to uniform transitions')
    diff_group.add_argument('-diff_crd_scale',  type=float, default=1./15, 
            help='Coordinate scaling factor for diffusion')
    diff_group.add_argument('-randomize_frames',  default=False, action='store_true',
            help='If true, randomize all frames at each step')

    # Trunk module properties
    trunk_group = parser.add_argument_group("Trunk module parameters")
    trunk_group.add_argument('-n_extra_block', type=int, default=4,
            help="Number of iteration blocks for extra sequences [4]")
    trunk_group.add_argument('-n_main_block', type=int, default=8,
            help="Number of iteration blocks for main sequences [8]")
    trunk_group.add_argument('-n_ref_block', type=int, default=4,
            help="Number of refinement layers")
    trunk_group.add_argument('-n_finetune_block', type=int, default=0,
            help="Number of finetune layers" [0])
    trunk_group.add_argument('-d_msa', type=int, default=256,
            help="Number of MSA features [256]")
    trunk_group.add_argument('-d_msa_full', type=int, default=64,
            help="Number of MSA features [64]")
    trunk_group.add_argument('-d_pair', type=int, default=128,
            help="Number of pair features [128]")
    trunk_group.add_argument('-d_templ', type=int, default=64,
            help="Number of templ features [64]")
    trunk_group.add_argument('-n_head_msa', type=int, default=8,
            help="Number of attention heads for MSA2MSA [8]")
    trunk_group.add_argument('-n_head_pair', type=int, default=4,
            help="Number of attention heads for Pair2Pair [4]")
    trunk_group.add_argument('-n_head_templ', type=int, default=4,
            help="Number of attention heads for template [4]")
    trunk_group.add_argument("-d_hidden", type=int, default=32,
            help="Number of hidden features [32]")
    trunk_group.add_argument("-d_hidden_templ", type=int, default=64,
            help="Number of hidden features for templates [64]")
    trunk_group.add_argument("-p_drop", type=float, default=0.15,
            help="Dropout ratio [0.15]")
    trunk_group.add_argument("-no_extra_l1", dest='use_extra_l1', default='True', action='store_false',
            help="Turn off chirality and LJ grad inputs to SE3 layers (for backwards compatibility).")
    trunk_group.add_argument("-no_atom_frames", dest='use_atom_frames', default='True', action='store_false',
            help="Turn off l1 features from atom frames in SE3 layers (for backwards compatibility).")
    trunk_group.add_argument('-freeze_track_motif', default=False, action='store_true',
            help='If True, manually freezes updates to the motif structure in track module')
    trunk_group.add_argument('-assert_single_sequence_input', default=False, action='store_true',
            help='If True, assert expected shapes for single sequence input')

    # Structure module properties
    str_group = parser.add_argument_group("structure module parameters")
    str_group.add_argument('-num_layers', type=int, default=1,
            help="Number of equivariant layers in structure module block [1]")
    str_group.add_argument('-num_channels', type=int, default=32,
            help="Number of channels [32]")
    str_group.add_argument('-num_degrees', type=int, default=2,
            help="Number of degrees for SE(3) network [2]")
    str_group.add_argument('-l0_in_features', type=int, default=64,
            help="Number of type 0 input features [64]")
    str_group.add_argument('-l0_out_features', type=int, default=64,
            help="Number of type 0 output features [64]")
    str_group.add_argument('-l1_in_features', type=int, default=3,
            help="Number of type 1 input features [3]")
    str_group.add_argument('-l1_out_features', type=int, default=2,
            help="Number of type 1 output features [2]")
    str_group.add_argument('-num_edge_features', type=int, default=64,
            help="Number of edge features [64]")
    str_group.add_argument('-n_heads', type=int, default=4,
            help="Number of attention heads for SE3-Transformer [4]")
    str_group.add_argument("-div", type=int, default=4,
            help="Div parameter for SE3-Transformer [4]")
    str_group.add_argument('-ref_num_layers', type=int, default=1,
            help="Number of equivariant layers in structure module block [1]")
    str_group.add_argument('-ref_num_channels', type=int, default=32,
            help="Number of channels [32]")

    # Loss function parameters
    loss_group = parser.add_argument_group("loss parameters")
    loss_group.add_argument('-w_dist', type=float, default=1.0,
            help="Weight on distd in loss function [1.0]")
    loss_group.add_argument('-w_disp', type=float, default=0.0,
            help="Weight on L2 CA displacement error [0.0]")
    loss_group.add_argument('-w_ax_ang', type=float, default=0.0,
            help='Weight on squared L2 loss on axis-angle deviation error [0.0]')
    loss_group.add_argument('-w_frame_dist', type=float, default=0.0,
            help='Weight on squared L2 "frame distance" error') 
    loss_group.add_argument('-w_exp', type=float, default=0.0,
            help="Weight on experimental resolved in loss function [0.0]")
    loss_group.add_argument('-w_str', type=float, default=10.0,
            help="Weight on strd in loss function [10.0]")
    loss_group.add_argument('-w_lddt', type=float, default=0.1,
            help="Weight on predicted lddt loss [0.1]")
    loss_group.add_argument('-w_all', type=float, default=0.5,
            help="Weight on MSA masked token prediction loss [0.5]")
    loss_group.add_argument('-w_aa', type=float, default=3.0,
            help="Weight on MSA masked token prediction loss [3.0]")
    loss_group.add_argument('-w_blen', type=float, default=0.0,
            help="Weight on predicted blen loss [0.0]")
    loss_group.add_argument('-w_bang', type=float, default=0.0,
            help="Weight on predicted bang loss [0.0]")
    loss_group.add_argument('-w_lj', type=float, default=0.0,
            help="Weight on lj loss [0.0]")
    loss_group.add_argument('-w_hb', type=float, default=0.0,
            help="Weight on hb loss [0.0]")
    loss_group.add_argument('-lj_lin', type=float, default=0.75,
            help="switch from linear to 12-6 for lj potential [0.75]")
    loss_group.add_argument('-use_H', action='store_true', default=False,
            help="consider hydrogens for lj loss [False]")
    loss_group.add_argument('-use_tschedule', action='store_true', default=False,
            help='(Diffusion training) True, use loss scaling as a function of timestep')
    loss_group.add_argument('-scheduled_losses',type=list, 
            default=['aa_cce','tors', 'blen', 'bang', 'lj', 'hb', 'w_str'],
            help='Losses that you want schedule for')
    loss_group.add_argument('-scheduled_types',type=list,
            default=['sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','linear'],
            help='Loss types to be used for each of the schedules losses. This list must be the same length as -scheduled_losses')
    loss_group.add_argument('-scheduled_params',type=list,
            default=[{'sig_stretch':0.23, 'sig_shift':0.885, 'linear_start':1., 'linear_end':0.1}]*7,
            help='Parameters to be used for each loss schedule')
    loss_group.add_argument('-w_motif_disp', type=float, default=0.0,
            help="Weight on motif displacement")
    loss_group.add_argument('-backprop_non_displacement_on_given',action='store_true', default=False,
            help='True, apply all losses, not just the displacement loss on the given region')

    # other parameters
    parser.add_argument('-task_names', default='diff',
            help='Comma separated list of tasks to train')
    parser.add_argument('-task_p', default='1.0',type=str,
            help='Comma separated list of probabilities for each task')
    parser.add_argument('-max_length',type=int, default=260,
            help= 'Upper limit of a single chain (i.e. maximum length of protein to keep) - should be as long as possible with available GPU memory')
    parser.add_argument('-max_complex_chain',type=int, default=200,
            help= 'for fixbb tasks, keep one chain complete. This is the maximum length of that chain (should be <60ish residues from max_length, so there is enough of the other chain)')
    parser.add_argument('-wandb_prefix', type=str, required=True,
            help='Prefix for name of session on wandb. This MUST be specified - make it clear what general parameters were used')
    parser.add_argument('-metric', type=lambda m: getattr(metrics, m), action='append')
    parser.add_argument('-log_inputs', action='store_true', default=False)
    parser.add_argument('-n_write_pdb', type=int, default=100)
    parser.add_argument('-reinitialize_missing_params', action='store_true', default=False, help='If the checkpoint file is missing a network parameter, use the networks default initialization for that parameter')
    parser.add_argument('-saves_per_epoch', type=int, default=0, help='number of times to save the model per epoch')
    parser.add_argument('-resume', help='run ID of a wandb run to resume')
    
    # Preprocessing parameters
    preprocess_group = parser.add_argument_group("preprocess parameters")
    preprocess_group.add_argument("-sidechain_input", choices=("True","False"), required=True,
        help='Do you want to provide diffused sidechains to the model. No default - make up your mind')
    preprocess_group.add_argument("-motif_sidechain_input", default="True", choices=("True","False"),
        help = 'Do you want to provide sidechains of the motif to the model. Default = True')
    preprocess_group.add_argument("-sequence_decode", choices=("True","False"), default="True",
        help='Do you want to decode sequence. Overrides aa_decode_steps. Default=True')
    preprocess_group.add_argument('-d_t1d', type=int, default=21+1+1,
            help='dimension of t1d raw inputs')
    preprocess_group.add_argument('-d_t2d', type=int, default = 44,
            help = 'dimension of t2d raw inputs')
    diff_group.add_argument('-prob_self_cond', type=float, default=0,
            help='The probability the model will receive self conditioning information during training. Default=0')
    diff_group.add_argument('-new_self_cond', action="store_true", default=True,
            help='Whether to use the new (consistent frames) self conditioning or the old (inconsisent frames) self conditioning. Default=False (old version)')
    diff_group.add_argument('-str_self_cond', action="store_true", default=False,
            help='Whether to train the model with structure self conditioning information. Default=False')
    diff_group.add_argument('-seq_self_cond', action="store_true", default=False,
            help='Whether to train the model with sequence self conditioning information. Default=False')

    # parse arguments
    args = parser.parse_args(in_args)
    ic(args.sequence_decode)
    # parse boolean arguments
    args.sidechain_input = args.sidechain_input == 'True'
    args.motif_sidechain_input = args.motif_sidechain_input == 'True'
    args.sequence_decode = args.sequence_decode == 'True'
    args.discontiguous_crop = args.discontiguous_crop == 'True'
    ic(args.sequence_decode)
    # parse the task lists
    task_names = args.task_names.split(',')
    task_p = args.task_p.split(',')
    task_p = [float(a) for a in task_p]

    args.task_names = task_names
    args.task_p = task_p
    # sanity checks and warnings
    print("WARNING: If you have changed the -max_length or -max_complex_chain flags, you need to regenerate the dataset.pkl file")
    if args.max_complex_chain > args.crop + 60:
        print("WARNING: max_complex_chain flag is close (<60) residues to the crop length. Do you have enough residues in the second, cropped complex chain to be meaningful?")
    if args.crop != args.max_length:
        print("WARNING: -crop and -max_length flags are not the same. This will lead to cropping - do you want this?")

    # set up diffusion params
    diffusion_params = {}
    for param in [
                  'diff_mask_probs',
                  'diff_mask_low',
                  'diff_mask_high',
                  'diff_b0',
                  'diff_bT',
                  'diff_schedule_type',
                  'diff_so3_schedule_type',
                  'diff_so3_type',
                  'diff_chi_type',
                  'diff_T',
                  'aa_decode_steps',
                  'predict_previous',
                  'prob_self_cond',
                  'seqdiff_b0',
                  'seqdiff_bT',
                  'seqdiff_schedule_type',
                  'seqdiff_loss_type',
                  'seqdiff',
                  'seqdiff_lambda',
                  'decode_mask_frac',
                  'decode_corrupt_blosum',
                  'decode_corrupt_uniform',
                  'diff_crd_scale',
                  'diff_min_b',
                  'diff_max_b',
                  'diff_min_sigma',
                  'diff_max_sigma']:
        diffusion_params[param] = getattr(args, param)
    

    # Setup dataloader parameters:
    loader_param = data_loader.set_data_loader_params(args)

    ### TRUNK PARAMS 
    trunk_param = {}
    for param in TRUNK_PARAMS:
        trunk_param[param] = getattr(args, param)
    SE3_param = {}
    for param in SE3_PARAMS:
        if hasattr(args, param):
            SE3_param[param] = getattr(args, param)

    SE3_ref_param = SE3_param.copy()

    for param in SE3_PARAMS:
        if hasattr(args, 'ref_'+param):
            SE3_ref_param[param] = getattr(args, 'ref_'+param)

    trunk_param['SE3_param'] = SE3_param 
    trunk_param['SE3_ref_param'] = SE3_ref_param

    # Set loss params 
    loss_param = {}
    for param in ['w_frame_dist',\
                'w_ax_ang',\
                'w_dist',\
                'w_str',\
                'w_all',\
                'w_aa',\
                'w_lddt',\
                'w_blen',\
                'w_bang',\
                'w_lj',\
                'w_hb',\
                'lj_lin',\
                'use_H',\
                'w_disp',\
                'w_motif_disp',\
                'backprop_non_displacement_on_given',\
                'use_tschedule',\
                'scheduled_losses',\
                'scheduled_types',\
                'scheduled_params']:
        loss_param[param] = getattr(args, param)
    
    # Collect preprocess_params
    preprocess_param = {}
    for param in ['sidechain_input',
                  'motif_sidechain_input',
                  'sequence_decode',
                  'd_t1d',
                  'd_t2d',
                  'predict_previous',
                  'prob_self_cond',
                  'str_self_cond',
                  'seq_self_cond',
                  'new_self_cond',
                  'randomize_frames',
                  ]:
        preprocess_param[param] = getattr(args, param)
    if not preprocess_param['sequence_decode']:
        raise NotImplementedError("switching off sequence decoding still needs to be implemented")

    return args, trunk_param, loader_param, loss_param, diffusion_params, preprocess_param

if __name__ == '__main__':
    args = get_args()
    pprint.PrettyPrinter(indent=4).pprint(args)
