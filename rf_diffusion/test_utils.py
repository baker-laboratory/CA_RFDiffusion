import re
import torch
import json
from icecream import ic
import os
import subprocess
from pathlib import Path
import dataclasses
import pickle
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from rf2aa import tensor_util
from rf_diffusion.data_loader import default_dataset_configs, get_dataset_and_sampler
from torch.utils import data

import urllib
from rf_diffusion import parsers
from rf2aa.util import kabsch
import tree
import numpy as np
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.data_loader import no_batch_collate_fn
import rf_diffusion
from rf_diffusion import aa_model
from rf_diffusion import noisers

golden_dir = 'goldens'


def golden_path(name, processor_specific=False):
    processor_name = 'any'
    if processor_specific:
        processor_name = processor()
    return Path(os.path.join(golden_dir, f'{name}_{processor_name}'))

def get_golden_path(name):
    p = golden_path()
    if not p.exists():
        raise Exception(f'golden at {p} does not exist, please generate it')
    return p


def write(path, got):
    with open(path, 'wb') as fh:
        pickle.dump(got, fh)

def read(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def assert_matches_golden(t, name, got, rewrite=False, processor_specific=False, custom_comparator=None):
    p = golden_path(name, processor_specific=processor_specific)
    if rewrite:
        print(f'rewriting {p}')
        write(p, got)
        # p.write_text(got)
        return
    if not p.exists():
        raise Exception(f'golden at {p} does not exist, please generate it')
    # want = p.read_text()
    want = read(p)

    if isinstance(got, rf_diffusion.aa_model.Indep):

        # Don't compare new features that the old indeps don't have
        was_missing = []
        for could_be_missing in ['extra_t2d']:
            if not hasattr(want, could_be_missing):
                setattr(want, could_be_missing, None)
                was_missing.append(could_be_missing)

        # Remove metadata
        want = aa_model.Indep(**dataclasses.asdict(want))
        got = aa_model.Indep(**dataclasses.asdict(got))
        
        for key in was_missing:
            delattr(want, key)
            delattr(got, key)

    if custom_comparator:
        diff = custom_comparator(got, want)
        if not diff:
            return
        try:
            jsoned = diff.to_json()
            loaded = json.loads(jsoned)
            fail_msg = json.dumps(loaded, indent=4)
        except Exception as e:
            ic('failed to pretty print output', e)
            fail_msg = json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 
        t.fail(fail_msg)
    else:
        t.assertEqual(got, want)

    # seq: torch.Tensor # [L]
    # xyz: torch.Tensor # [L, 36?, 3]
    # idx: torch.Tensor

    # # SM specific
    # bond_feats: torch.Tensor
    # chirals: torch.Tensor
    # same_chain: torch.Tensor
    # terminus_type: torch.Tensor
    # extra_t1d: torch.Tensor = dataclasses.field(default_factory=lambda: None)

def indep_NA_comparator(got, want):
    if ~torch.eq(got.seq, want.seq):
        return True
    common_dim = min(got.xyz.shape[1], want.xyz.shape[1])
    # In rare cases, the goldens have 36 atoms, in which case the code should also produce 36 atoms
    if want.xyz.shape[1] >= 36 and got.xyz.shape[1] != want.xyz.shape[1]:
        return True
    if ~torch.eq(got.xyz[:,:common_dim], want.xyz[:,:common_dim]):
        return True
    if ~torch.eq(got.idx, want.idx):
        return True
    if ~torch.eq(got.is_sm, want.is_sm):
        return True
    if ~torch.eq(got.bond_feats, want.bond_feats):
        return True
    if ~torch.eq(got.chirals, want.chirals):
        return True
    if ~torch.eq(got.same_chain, want.same_chain):
        return True
    if ~torch.eq(got.terminus_type, want.terminus_type):
        return True
    if ~torch.eq(got.extra_t1d, want.extra_t1d):
        return True
    return False

def assertEqual(t, custom_comparator, got, want):
    diff = custom_comparator(got, want)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        fail_msg = json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        fail_msg = json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 
    t.fail(fail_msg)


def processor():
    o = subprocess.run(['lscpu | grep "Model name"'], capture_output=True, shell=True)
    s = o.stdout.decode("utf-8")
    s = s[len("Model name:"):].strip()
    ncpus = available_cpu_count()
    s = f'{s}-{ncpus}CPUs'
    return s

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

def cmp_pretty(got, want, **kwargs):
    diff = tensor_util.cmp(got, want, **kwargs)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        return json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        return json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff)

def assert_arrays_equal(got, want, **kwargs):
    diff = cmp_pretty(got, want, **kwargs)
    if diff:
        raise Exception(diff)

def where_nan(t):
    if torch.isnan(t).any():
        return (torch.isnan(t).nonzero(), torch.isnan(t).float().mean())
    return None

def assert_no_nan(t):
    msg = where_nan(t)
    if msg:
        raise Exception(msg)

def construct_conf(overrides=None, config_name='debug', train_config_name='train_covale_complex', inference=False):
    if not inference:
        return construct_conf_single(overrides=overrides, config_name=config_name, inference=inference)
    train_conf = construct_conf_single(config_name=train_config_name, inference=False)
    inference_conf = construct_conf_single(overrides=overrides, config_name=config_name, inference=inference)
    OmegaConf.set_struct(train_conf, False)
    OmegaConf.set_struct(inference_conf, False)
    conf = OmegaConf.merge(
        train_conf, inference_conf)
    return conf
    

def construct_conf_single(overrides=None, config_name='debug', inference=False):
    hydra.core.global_hydra.GlobalHydra().clear()
    overrides = overrides or []
    config_path = 'config/training'
    if inference:
        config_path = 'config/inference'
    initialize(version_base=None, config_path=config_path, job_name="test_app")
    ic(f'{config_name}.yaml', overrides, inference, config_path)
    conf = compose(config_name=f'{config_name}.yaml', overrides=overrides, return_hydra_config=True)
    return conf

def get_dataloader(conf: DictConfig, epoch=0) -> None:

    if conf.debug:
        ic.configureOutput(includeContext=True)
    diffuser = noisers.get(conf.diffuser)
    diffuser.T = conf.diffuser.T
    dataset_configs, homo = default_dataset_configs(conf.dataloader, debug=conf.debug)

    print('Making train sets')

    train_set, train_sampler = get_dataset_and_sampler(dataset_configs, conf.dataloader['DATASETS'], conf.dataloader['DATASET_PROB'], conf, diffuser,
                conf.epoch_size, 1, 0, homo,
                datahub_mode=conf.dataloader.get('DATAHUB_MODE', 'DATAHUB_IF_AVAILABLE')
                )

    train_sampler.epoch = epoch
    
    # mp.cpu_count()-1
    LOAD_PARAM = {
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True
    }
    train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=conf.batch_size, collate_fn=no_batch_collate_fn, **LOAD_PARAM)
    return train_loader

def loader_out_from_conf(conf, epoch=0):
    dataloader = get_dataloader(conf, epoch)
    for loader_out in dataloader:
        # indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        # indep.metadata = None
        return loader_out
    
def loader_out_from_conf_datahub(conf, epoch=0):
    load_param = {'shuffle': False,
                  'num_workers': 1,
                  'pin_memory': True}
    dataloader, _  = get_datahub_fallback_dataset_and_dataloader(conf, rank=0, world_size=1, LOAD_PARAM=load_param)
    for loader_out in dataloader:
        return loader_out


def loader_out_for_dataset(dataset, mask, overrides=[], epoch=0, config_name='debug'):
    '''
    Generate the output from the dataloader for a given dataset, mask, overrides, and config

    Args:
        dataset (str or None): The dataset to pull from. None if you want to rely on the config's default probabilities
        mask (str): The diffusion mask to use
        overrides (list[str]): Additional configs to pass through
        epoch (int): The epoch to pass into the loader
        config_name (str): The name of the config to use in configs/training/*.yaml

    Returns:
        loader_out (tuple): The standard return value from the dataloader
    '''
    if dataset is None:
        dataset_prob_part = []
    else:
        dataset_prob_part = [
            f'dataloader.DATASETS={dataset}',
            'dataloader.DATASET_PROB=[1.0]',
        ]
    conf = construct_conf([
        'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
        '+dataloader.ligands_to_remove=[]', # Remove this line when the subsampled dataset is remade
        '+dataloader.min_metal_contacts=0', # Remove this line when the subsampled dataset is remade        
        'dataloader.CROP=256',
        'dataloader.DIFF_MASK_PROBS=null',
        f'dataloader.DIFF_MASK_PROBS={{{mask}:1.0}}',
        'debug=True',
        'spoof_item=null',
    ] + dataset_prob_part + overrides, config_name=config_name)
    
    return loader_out_from_conf(conf, epoch=epoch)

def loader_out_for_datahub_dataset(config, overrides=[]):
    conf = construct_conf_single(overrides=[
        'datahub.n_fallback_retries=0',
        f'datahub.train.pdb.sub_datasets.pn_unit.dataset.transform.full_config_name={config}',
        f'datahub.train.pdb.sub_datasets.interface.dataset.transform.full_config_name={config}',
    ] + overrides, config_name=config)
    return loader_out_from_conf_datahub(conf)

def subset_target_feats(target_feats, is_subset):
    '''
    is_subset (iterable[bool]): True = Include this residue in the returned subset.
    '''
    xyz = target_feats['xyz'][is_subset]
    mask = target_feats['mask'][is_subset]
    idx = target_feats['idx'][is_subset]
    seq = target_feats['seq'][is_subset]
    pdb_idx = [x for b, x in zip(is_subset, target_feats['pdb_idx']) if b]

    subset = dict(
        xyz=xyz,
        mask=mask,
        idx=idx,
        seq=seq,
        pdb_idx=pdb_idx
    )

    return subset
    
def pdbid_to_feats(pdbid: str):
    stream = urllib.request.urlopen(f'https://files.rcsb.org/view/{pdbid.upper()}.pdb').read().decode('ascii').split('\n')
    target_feats = parsers.parse_pdb_lines_target(stream)
    return target_feats

def is_atom_resolved(xyz: torch.Tensor):
    '''
    xyz (..., n_atoms, 3)
    '''
    is_near_ori = (xyz < 1e-3).all(-1)
    is_nan = torch.isnan(xyz).any(-1)
    
    return ~(is_near_ori | is_nan)

def point_correspondance(xyz1, xyz2, atol=1e-2):
    '''
    Finds which points have the same nd coordinates within some tolerance.
    
    xyz1, xyz2 (..., n_dims)
    '''
    return ((xyz1[:, None] - xyz2[None, :]).abs() < atol).all(-1)

def compare_aa_atom_order(xyz_cmp, xyz_ref, aa_int):
    '''
    Mainly intended to detect if xyz_cmp is a permutation of xyz_ref.
    Does not do any alignment.

    Inputs
        xyz_cmp: (n_atoms, 3)
        xyz_ref: (n_atoms, 3)

    Returns
        True: If all the atoms are in the same order, based on their coordinates.
        False: Atoms in xyz_cmp are permuted compared to xyz_ref.
        None: Could not match all atoms, so no conclusion can be drawn.
    '''
    aa3 = ChemData().num2aa[aa_int]
    
    # Compare only resolved atoms
    is_resolved_cmp = is_atom_resolved(xyz_cmp)
    is_resolved_ref = is_atom_resolved(xyz_ref)
    if not (is_resolved_cmp.sum() == is_resolved_ref.sum()):
        msg = 'xyz_cmp and xyz_ref do not have the same number of resolved atoms. They cannot be compared.'
        return None, msg

    xyz_cmp = xyz_cmp[is_resolved_cmp]
    xyz_ref = xyz_ref[is_resolved_ref]
    same_xyz = point_correspondance(xyz_cmp, xyz_ref)

    n_atoms = xyz_ref.shape[0]
    if same_xyz.sum() != n_atoms:
        msg = (f'Not all {aa3} reference atoms found a comparison atoms with similar coordinates! '
              'No point in comparing the atom ordering')
        return None, msg

    out_of_order = same_xyz * ~torch.eye(*same_xyz.shape, dtype=bool)
    if out_of_order.any():
        msg = []
        # Find out what atom_names are not ordered correctly
        for cmp_idx0, ref_idx0 in zip(*torch.where(out_of_order)):
            atom_name = ChemData().aa2long[aa_int][ref_idx0]
            msg.append(f'For {aa3}, expected {atom_name} to be at position {ref_idx0} but was at position {cmp_idx0} instead!')

        return False, ' '.join(msg)
    else:
        msg = 'Ref and comparison atoms are in the same order.'
        return True, msg
    
def detect_permuted_aa_atoms(indep_train, item_context: dict):
    '''
    Uses atom coordinates to infer if amino acid atoms in the indep are
    ordered the same way as they are defined in rf2aa.chemical.
    
    Note: Currently could incorrectly identify an atom permutation because it
    does not account for symmetric side chains. So far this has
    not been a problem since the .pt files Ivan originally parsed largely have the 
    atom order as rf2aa for symmetric atoms.
    '''
    # Get "ground truth" features directly from rcsb
    pdbid, chain = item_context['sel_item']['CHAINID'].split('_')
    target_feats = pdbid_to_feats(pdbid)

    # Subset target_feats to the chain the dataloader used
    is_subset = [ch == chain.upper() for ch, resi in target_feats['pdb_idx']]
    target_feats = subset_target_feats(target_feats, is_subset)
    target_feats = tree.map_structure(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x, target_feats)

    # align only based on amino acids
    indep_aa_seq = torch.where(indep_train.seq < 20, indep_train.seq, torch.nan)
    target_aa_seq = torch.where(target_feats['seq'] < 20, target_feats['seq'], torch.nan)
    L_target = target_feats['seq'].shape[0]
    same_aa = indep_aa_seq[:, None] == target_aa_seq[None, :]

    n_aligned_res = torch.tensor([torch.diagonal(same_aa, offset).sum().item() for offset in range(L_target)])
    offset_max_aligned_res = n_aligned_res.argmax()

    results = {ChemData().num2aa[aa_int]: [] for aa_int in range(20)}
    if n_aligned_res[offset_max_aligned_res] < 5:
        print(f'Fewer than 5 consecutive residues could be aligned. Skipping {item_context}')

    else:
        # Find what ref residues correspond to which cmp residues
        is_aligned = torch.zeros(same_aa.shape, dtype=bool)
        torch.diagonal(is_aligned, offset=offset_max_aligned_res).fill_(True)
        is_aligned = is_aligned & same_aa
        indep_idx0, target_idx0 = torch.where(is_aligned)

        # Align indep xyz to target xyz based on CA xyz
        indep_ca_xyz = indep_train.xyz[indep_idx0, 1]
        target_ca_xyz = target_feats['xyz'][target_idx0, 1]
        rmsd, U = kabsch(indep_ca_xyz, target_ca_xyz)  # aligns indep to target xyz
        indep_xyz_aligned = (indep_train.xyz - indep_ca_xyz.mean(0)) @ U + target_ca_xyz.mean(0)

        # For each align amino acid, compare the atom coordinates to see if any of them are purmuted!
        for indep_i, target_i in zip(indep_idx0, target_idx0):
            xyz_cmp = indep_xyz_aligned[indep_i]
            xyz_ref = target_feats['xyz'][target_i]
            aa_int = target_feats['seq'][target_i]
            aa3 = ChemData().num2aa[aa_int]
            results[aa3].append(compare_aa_atom_order(xyz_cmp, xyz_ref, aa_int))

    return results

def full_count(aa_count, min_count):
    '''
    Have more than the minimum number of aa been checked?
    '''
    return all(map(lambda count: count >= min_count, aa_count.values()))
