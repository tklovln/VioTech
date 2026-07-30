import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor) 
from data_generator import MaestroDataset, Augmentor, Sampler, TestSampler, collate_fn
# from data_generator import MosaDataset, MosaSampler, MosaTestSampler
from data_generator import CustomDataset, CustomSampler,CustomTestSampler
from data_generator import RWCTechniqueDataset, RWCTechniqueSampler, RWCTechniqueTestSampler
from data_generator import RWCTechNoteWavDataset
from models_contrast import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN, AcousticModelCRnn8Dropout, NoteLevelTechniqueModel
from pytorch_utils import move_data_to_device
from losses import (
    get_loss_func,
    supcon_loss,
    all_pitch_supcon_loss,
    onset_binary_supcon_loss,
    offset_binary_supcon_loss,
    per_pitch_onoff_ctc_loss,
    technique_frame_ce_loss,
    viotech_technique_losses,
    moe_technique_losses,
    zone_moe_technique_losses,
    pertask_zone_moe_technique_losses,
    frame_multiscale_moe_losses,
)
from evaluate import SegmentEvaluator, evaluate_rwc_frame_moe
from moe_frame_multiscale import FrameMultiScaleMoEConfig
import config

from pytorch_metric_learning import losses, reducers, miners
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Reproducibility
    seed = getattr(args, 'seed', None)
    if seed is not None:
        set_seed(seed)
        print(f'Random seed fixed to {seed}')

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    num_workers = args.num_workers

    device = torch.device(f"cuda:{args.device}") if args.device >= 0 and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename
    dataset = args.dataset
    logdir = args.logdir
    model_tag = args.model_tag
    contrast_weight = args.contrast_weight
    ctc_weight = args.ctc_weight
    use_cosine_annealing_warm_restarts = args.use_cosine_annealing_warm_restarts
    use_aux_contrast_model = args.use_aux_contrast_model
    print('contrast weight: ', contrast_weight)
    print('use cosine annealing warm restarts: ', use_cosine_annealing_warm_restarts)
    print('use aux contrast model: ', use_aux_contrast_model)
    print('ctc weight: ', ctc_weight)
    technique_weight = getattr(args, 'technique_weight', 0.0)
    technique_moe_weight = getattr(args, 'technique_moe_weight', 0.0)
    moe_balance_coeff = getattr(args, 'moe_balance_coeff', 0.01)
    technique_moe_zone_weight = getattr(args, 'technique_moe_zone_weight', 0.0)
    moe_zone_balance_coeff = getattr(args, 'moe_zone_balance_coeff', 0.01)
    technique_moe_zone_pt_weight = getattr(args, 'technique_moe_zone_pt_weight', 0.0)
    moe_zone_pt_balance_coeff = getattr(args, 'moe_zone_pt_balance_coeff', 0.001)
    technique_frame_moe_weight = getattr(args, 'technique_frame_moe_weight', 0.0)
    frame_moe_balance_coeff = getattr(args, 'frame_moe_balance_coeff', 0.001)
    fmoe_spectral_expert = bool(getattr(args, 'fmoe_spectral_expert', 1))
    fmoe_expert_mask_str = getattr(args, 'fmoe_expert_mask', None)
    fmoe_expert_mask = (
        tuple(int(x) for x in fmoe_expert_mask_str.split(','))
        if fmoe_expert_mask_str else None
    )
    fmoe_top_k = getattr(args, 'fmoe_top_k', None)
    fmoe_shared_gate = bool(getattr(args, 'fmoe_shared_gate', 0))
    fmoe_uniform_routing = bool(getattr(args, 'fmoe_uniform_routing', 0))
    fmoe_aux_weight = getattr(args, 'fmoe_aux_weight', 0.0)
    fmoe_temporal_str = getattr(args, 'fmoe_temporal_experts', None)
    fmoe_temporal_experts = None
    if fmoe_temporal_str:
        fmoe_temporal_experts = tuple(
            tuple(int(x) for x in pair.split(':'))
            for pair in fmoe_temporal_str.split(',')
        )
    fmoe_use_cues = not bool(getattr(args, 'fmoe_no_cues', 0))
    fmoe_detach_cues = bool(getattr(args, 'fmoe_detach_cues', 0))
    freeze_transcription = bool(getattr(args, 'freeze_transcription', 0))
    if freeze_transcription:
        fmoe_detach_cues = True
    fmoe_direct_head = bool(getattr(args, 'fmoe_direct_head', 0))
    fmoe_hierarchical = bool(getattr(args, 'fmoe_hierarchical', 0))
    fmoe_hierarchical_multilevel = bool(getattr(args, 'fmoe_hierarchical_multilevel', 0))
    fmoe_hierarchical_moe = bool(getattr(args, 'fmoe_hierarchical_moe', 0))
    fmoe_multi_crnn = bool(getattr(args, 'fmoe_multi_crnn', 0))
    fmoe_crnn_moe = bool(getattr(args, 'fmoe_crnn_moe', 0))
    fmoe_shared_crnn_moe = bool(getattr(args, 'fmoe_shared_crnn_moe', 0))
    fmoe_mert_backbone = bool(getattr(args, 'fmoe_mert_backbone', 0))
    fmoe_mert_moe = bool(getattr(args, 'fmoe_mert_moe', 0))
    fmoe_mert_moe_hier = bool(getattr(args, 'fmoe_mert_moe_hierarchical', 0))
    if fmoe_mert_moe_hier:
        fmoe_mert_moe = True
    fmoe_mert_no_layer_weight = bool(getattr(args, 'fmoe_mert_no_layer_weight', 0))
    focal_gamma = getattr(args, 'focal_gamma', 0.0)
    print('technique weight: ', technique_weight)
    print('technique MoE weight: ', technique_moe_weight)
    print('technique MoE-Zone weight: ', technique_moe_zone_weight)
    print('technique MoE-Zone-PerTask weight: ', technique_moe_zone_pt_weight)
    print('technique Frame-MoE weight: ', technique_frame_moe_weight)
    print('Frame-MoE spectral expert: ', fmoe_spectral_expert)
    print('Frame-MoE expert mask: ', fmoe_expert_mask)
    print('Frame-MoE top-k: ', fmoe_top_k)
    print('Frame-MoE shared gate: ', fmoe_shared_gate)
    print('Frame-MoE uniform routing: ', fmoe_uniform_routing)
    print('Frame-MoE aux weight: ', fmoe_aux_weight)
    print('Frame-MoE temporal experts: ', fmoe_temporal_experts)
    print('Frame-MoE use cues: ', fmoe_use_cues)
    print('Frame-MoE detach cues: ', fmoe_detach_cues)
    print('Freeze transcription: ', freeze_transcription)
    print('Frame-MoE direct head: ', fmoe_direct_head)
    print('Frame-MoE hierarchical head: ', fmoe_hierarchical)
    print('Frame-MoE hierarchical multilevel: ', fmoe_hierarchical_multilevel)
    print('Frame-MoE hierarchical MoE: ', fmoe_hierarchical_moe)
    print('Frame-MoE multi-CRNN: ', fmoe_multi_crnn)
    print('Frame-MoE CRNN-MoE: ', fmoe_crnn_moe)
    print('Frame-MoE Shared-CRNN-MoE: ', fmoe_shared_crnn_moe)
    print('Frame-MoE MERT backbone: ', fmoe_mert_backbone)
    print('Frame-MoE MERT MoE: ', fmoe_mert_moe)
    print('Frame-MoE MERT MoE Hierarchical: ', fmoe_mert_moe_hier)
    print('focal loss gamma: ', focal_gamma)
    if dataset == 'viotech_mixed_mosavpt':
        print('mosavpt_ratio: ', getattr(args, 'mosavpt_ratio', 0.5))

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    # num_workers = 8

    # Loss function
    loss_func = get_loss_func(loss_type)

    # # Supervised contrastive loss
    # supcon_loss = losses.SupConLoss(temperature=0.07) # input: features, labels
    
    # TensorBoard writer
    logdir = os.path.join(logdir, args.model_tag)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    logging.info(f'TensorBoard logs will be saved to: {logdir}')

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)
    # Special handling for single-file RWC dataset: symlink the file into workspace/hdf5s/rwc
    if dataset == 'rwc':
        rwc_src = os.path.expanduser(getattr(args, 'rwc_h5_path', '~/data/rwc_processed_data.h5'))
        if not os.path.exists(rwc_src):
            raise FileNotFoundError(f"RWC h5 file not found at {rwc_src}. Set --rwc_h5_path correctly.")
        create_folder(hdf5s_dir)
        link_name = os.path.join(hdf5s_dir, os.path.basename(rwc_src))
        try:
            if os.path.islink(link_name):
                # Update link if it points elsewhere
                current_target = os.readlink(link_name)
                if current_target != rwc_src:
                    os.remove(link_name)
                    os.symlink(rwc_src, link_name)
            elif not os.path.exists(link_name):
                os.symlink(rwc_src, link_name)
        except OSError:
            # Fall back: copy if symlink not permitted
            if not os.path.exists(link_name):
                import shutil
                shutil.copyfile(rwc_src, link_name)


    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, model_tag,
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, model_tag,
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, model_tag,
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Model
    Model = eval(model_type)
    if technique_frame_moe_weight > 0:
        if fmoe_shared_crnn_moe:
            from moe_crnn_backbone import SharedCNNMoEConfig
            _sc_kwargs = dict(
                expert_mask=fmoe_expert_mask,
                use_cues=fmoe_use_cues,
            )
            if fmoe_top_k is not None:
                _sc_kwargs['top_k'] = fmoe_top_k
            frame_moe_cfg = SharedCNNMoEConfig(**_sc_kwargs)
        elif fmoe_crnn_moe:
            from moe_crnn_backbone import CRNNMoEConfig
            _cm_kwargs = dict(
                expert_mask=fmoe_expert_mask,
                use_cues=fmoe_use_cues,
            )
            if fmoe_top_k is not None:
                _cm_kwargs['top_k'] = fmoe_top_k
            frame_moe_cfg = CRNNMoEConfig(**_cm_kwargs)
        elif fmoe_multi_crnn:
            from moe_frame_multiscale import FrameMultiCRNNMoEConfig
            _mc_kwargs = dict(
                use_spectral_expert=fmoe_spectral_expert,
                expert_mask=fmoe_expert_mask,
                use_cues=fmoe_use_cues,
            )
            if fmoe_top_k is not None:
                _mc_kwargs['top_k'] = fmoe_top_k
            frame_moe_cfg = FrameMultiCRNNMoEConfig(**_mc_kwargs)
        elif fmoe_hierarchical_multilevel:
            from hierarchical_frame import FrameHierarchicalMultiLevelConfig
            frame_moe_cfg = FrameHierarchicalMultiLevelConfig(use_cues=fmoe_use_cues)
        elif fmoe_hierarchical_moe:
            from hierarchical_frame import FrameHierarchicalMoEConfig
            frame_moe_cfg = FrameHierarchicalMoEConfig(use_cues=fmoe_use_cues)
        elif fmoe_hierarchical:
            from hierarchical_frame import FrameHierarchicalConfig
            frame_moe_cfg = FrameHierarchicalConfig(use_cues=fmoe_use_cues)
        elif fmoe_mert_moe:
            from mert_moe_backbone import MERTMoEConfig
            _mm_kwargs = dict(
                use_cues=fmoe_use_cues,
                expert_mask=fmoe_expert_mask,
                hierarchical=fmoe_mert_moe_hier,
                use_layer_weight=(not fmoe_mert_no_layer_weight),
            )
            if fmoe_temporal_experts:
                _mm_kwargs['temporal_experts'] = fmoe_temporal_experts
            if fmoe_top_k is not None:
                _mm_kwargs['top_k'] = fmoe_top_k
            frame_moe_cfg = MERTMoEConfig(**_mm_kwargs)
        else:
            _fmoe_kwargs = dict(
                use_spectral_expert=fmoe_spectral_expert,
                expert_mask=fmoe_expert_mask,
                shared_gate=fmoe_shared_gate,
                uniform_routing=fmoe_uniform_routing,
                temporal_experts=fmoe_temporal_experts,
                use_cues=fmoe_use_cues,
            )
            if fmoe_top_k is not None:
                _fmoe_kwargs['top_k'] = fmoe_top_k
            frame_moe_cfg = FrameMultiScaleMoEConfig(**_fmoe_kwargs)
    else:
        frame_moe_cfg = None
    model = Model(
        frames_per_second=frames_per_second,
        classes_num=classes_num,
        output_features=True,
        predict_technique=(technique_weight > 0),
        predict_technique_moe=(technique_moe_weight > 0),
        predict_technique_moe_zone=(technique_moe_zone_weight > 0),
        predict_technique_moe_zone_pt=(technique_moe_zone_pt_weight > 0),
        predict_technique_frame_moe=(technique_frame_moe_weight > 0),
        frame_moe_config=frame_moe_cfg,
        detach_cues=fmoe_detach_cues,
        direct_head=fmoe_direct_head,
        hierarchical_head=fmoe_hierarchical,
        hierarchical_multilevel_head=fmoe_hierarchical_multilevel,
        hierarchical_moe_head=fmoe_hierarchical_moe,
        multi_crnn_head=fmoe_multi_crnn,
        crnn_moe_head=fmoe_crnn_moe,
        shared_crnn_moe_head=fmoe_shared_crnn_moe,
        mert_backbone=fmoe_mert_backbone,
        mert_moe_head=fmoe_mert_moe,
    )
    aux_contrast_onset_model = nn.Linear(512, 128)
    aux_contrast_offset_model = nn.Linear(512, 128)
    # aux_contrast_model = nn.Sequential(
    #     nn.Linear(512, 128),
    #     nn.BatchNorm1d(128),
    #     nn.ReLU(),
    # )
    # Using model's built-in technique head; remove auxiliary technique models


    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')
    
    # Dataset & Samplers
    if dataset in ['mosa', 'mosapt', 'mixed', 'rwc', 'rwc_tech', 'rwc_tech_note_wav', 'viotech', 'viotech_mixed_mosavpt']:
        if dataset == 'mixed':
            # Build MOSA and MOSAPT datasets
            hdf5s_dir_mosa = os.path.join(workspace, 'hdf5s', 'mosa')
            hdf5s_dir_mosapt = os.path.join(workspace, 'hdf5s', 'mosapt')

            train_dataset_mosa = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosa,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=max_note_shift,
                augmentor=augmentor,
                include_technique_label=False,
            )
            train_dataset_mosapt = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosapt,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=max_note_shift,
                augmentor=augmentor,
                include_technique_label=True,
            )

            # Independent samplers for underlying datasets
            train_sampler_mosa = CustomSampler(
                hdf5s_dir=hdf5s_dir_mosa,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            train_sampler_mosapt = CustomSampler(
                hdf5s_dir=hdf5s_dir_mosapt,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            # Mixed dataset wrapper dispatching metas to the correct base dataset
            class MixedDataset(object):
                def __init__(self, mosa_ds, mosapt_ds):
                    self.mosa_ds = mosa_ds
                    self.mosapt_ds = mosapt_ds

                def __getitem__(self, tagged_meta):
                    src, meta = tagged_meta  # src in {'mosa', 'mosapt'}
                    if src == 'mosa':
                        return self.mosa_ds[meta]
                    else:
                        return self.mosapt_ds[meta]

            train_dataset = MixedDataset(train_dataset_mosa, train_dataset_mosapt)

            # Batch sampler that yields tagged metas
            class MixedBatchSampler(object):
                def __init__(self, sampler_a, sampler_b, mosa_ratio=0.5):
                    self.sampler_a = sampler_a
                    self.sampler_b = sampler_b
                    self.mosa_ratio = mosa_ratio

                def __iter__(self):
                    iter_a = iter(self.sampler_a)
                    iter_b = iter(self.sampler_b)
                    while True:
                        use_b = (np.random.rand() < self.mosa_ratio)
                        if use_b:
                            try:
                                batch = next(iter_b)
                                yield [('mosapt', m) for m in batch]
                                continue
                            except StopIteration:
                                iter_b = iter(self.sampler_b)
                        try:
                            batch = next(iter_a)
                            yield [('mosa', m) for m in batch]
                        except StopIteration:
                            iter_a = iter(self.sampler_a)

                def __len__(self):
                    return -1

                def state_dict(self):
                    state = {}
                    if hasattr(self.sampler_a, 'state_dict'):
                        state['sampler_a'] = self.sampler_a.state_dict()
                    if hasattr(self.sampler_b, 'state_dict'):
                        state['sampler_b'] = self.sampler_b.state_dict()
                    return state

                def load_state_dict(self, state):
                    if 'sampler_a' in state and hasattr(self.sampler_a, 'load_state_dict'):
                        self.sampler_a.load_state_dict(state['sampler_a'])
                    if 'sampler_b' in state and hasattr(self.sampler_b, 'load_state_dict'):
                        self.sampler_b.load_state_dict(state['sampler_b'])

            train_sampler = MixedBatchSampler(train_sampler_mosa, train_sampler_mosapt, mosa_ratio=args.mosa_ratio)

            # Evaluation on both MOSA and MOSAPT
            evaluate_dataset_mosa = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosa,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=0,
                include_technique_label=False,
            )
            evaluate_dataset_mosapt = CustomDataset(
                hdf5s_dir=hdf5s_dir_mosapt,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=0,
                include_technique_label=True,
            )
            evaluate_dataset = MixedDataset(evaluate_dataset_mosa, evaluate_dataset_mosapt)

            evaluate_train_sampler_mosa = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosa,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            evaluate_train_sampler_mosapt = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosapt,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            # Finite mixed sampler for evaluation (no iterator reset)
            class MixedEvalBatchSampler(object):
                def __init__(self, sampler_a, sampler_b, mosa_ratio=0.5):
                    self.sampler_a = sampler_a
                    self.sampler_b = sampler_b
                    self.mosa_ratio = mosa_ratio


                def __iter__(self):
                    iter_a = iter(self.sampler_a)
                    iter_b = iter(self.sampler_b)

                    a_exhausted = False
                    b_exhausted = False

                    while True:
                        use_b = (np.random.rand() < self.mosa_ratio)
                        if use_b and not b_exhausted:
                            try:
                                batch = next(iter_b)
                                yield [('mosapt', m) for m in batch]
                                continue
                            except StopIteration:
                                b_exhausted = True
                        if not a_exhausted:
                            try:
                                batch = next(iter_a)
                                yield [('mosa', m) for m in batch]
                            except StopIteration:
                                a_exhausted = True

                        if a_exhausted and b_exhausted:
                            break

                def __len__(self):
                    return -1

            evaluate_train_sampler = MixedEvalBatchSampler(
                evaluate_train_sampler_mosa, evaluate_train_sampler_mosapt, mosa_ratio=0.5
            )

            evaluate_test_sampler_mosa = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosa,
                split='test',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            evaluate_test_sampler_mosapt = CustomTestSampler(
                hdf5s_dir=hdf5s_dir_mosapt,
                split='test',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data,
            )
            evaluate_test_sampler = MixedEvalBatchSampler(
                evaluate_test_sampler_mosa, evaluate_test_sampler_mosapt, mosa_ratio=0.5
            )
        else:
            if dataset == 'rwc_tech':
                # Technique-only training using RWC single H5
                rwc_src = os.path.expanduser(getattr(args, 'rwc_h5_path', '~/data/rwc_processed_data.h5'))
                if not os.path.exists(rwc_src):
                    raise FileNotFoundError(f"RWC h5 file not found at {rwc_src}. Set --rwc_h5_path correctly.")

                # Build deterministic file-level split
                import h5py
                with h5py.File(rwc_src, 'r') as f:
                    all_files = sorted([k for k in f.keys() if k.startswith('file_')])
                rng = np.random.RandomState(getattr(args, 'rwc_split_seed', 1234))
                rng.shuffle(all_files)
                max_files = getattr(args, 'rwc_max_files', 0)
                if max_files and max_files > 0:
                    all_files = all_files[:max_files]
                train_ratio = float(getattr(args, 'rwc_train_ratio', 0.9))
                split_idx = int(len(all_files) * train_ratio)
                rwc_train_files = all_files[:split_idx]
                rwc_test_files = all_files[split_idx:]

                # Datasets
                train_dataset = RWCTechniqueDataset(
                    rwc_h5_path=rwc_src,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                )
                evaluate_dataset = RWCTechniqueDataset(
                    rwc_h5_path=rwc_src,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    augmentor=None,
                )

                # Samplers
                train_sampler = RWCTechniqueSampler(
                    rwc_h5_path=rwc_src,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                    max_files=getattr(args, 'rwc_max_files', 0),
                    allowed_file_keys=rwc_train_files,
                )

                # Evaluation samplers (file-level split)
                evaluate_train_sampler = RWCTechniqueTestSampler(
                    rwc_h5_path=rwc_src,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                    max_files=getattr(args, 'rwc_max_files', 0),
                    allowed_file_keys=rwc_train_files,
                )
                evaluate_test_sampler = RWCTechniqueTestSampler(
                    rwc_h5_path=rwc_src,
                    split='test',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                    max_files=getattr(args, 'rwc_max_files', 0),
                    allowed_file_keys=rwc_test_files,
                )
            elif dataset == 'rwc_tech_note_wav':
                # Note-level technique dataset from exported WAVs
                notes_root = os.path.expanduser(getattr(args, 'rwc_notes_root', './rwc_notes'))
                train_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, return_logmel=True)
                evaluate_dataset = RWCTechNoteWavDataset(notes_root, frames_per_second=frames_per_second, return_logmel=True)
                # Simple sequential sampler using DataLoader default sampler (no batch_sampler)
                train_sampler = None
                evaluate_train_sampler = None
                evaluate_test_sampler = None
            elif dataset == 'viotech':
                # Similar to mosa/mosapt but for viotech
                print(f'Creating train and evaluate datasets for viotech...')
                train_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                    include_technique_label=True, # Use False for now as per "mimic mosa"
                )
                evaluate_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    include_technique_label=True,
                )

                # Sampler for training
                train_sampler = CustomSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )

                # Samplers for evaluation (single dataset case only)
                evaluate_train_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
                evaluate_test_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='test',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
            elif dataset == 'viotech_mixed_mosavpt':
                hdf5s_dir_viotech = os.path.join(workspace, 'hdf5s', 'viotech_paganini_kayser_with_openstring')
                hdf5s_dir_mosavpt = args.mosavpt_dir
                print(f'Creating mixed viotech + mosavpt datasets...')
                print(f'  viotech dir:   {hdf5s_dir_viotech}')
                print(f'  mosavpt dir: {hdf5s_dir_mosavpt}')

                train_dataset_vt = CustomDataset(
                    hdf5s_dir=hdf5s_dir_viotech,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                    include_technique_label=True,
                )
                train_dataset_fl = CustomDataset(
                    hdf5s_dir=hdf5s_dir_mosavpt,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                    include_technique_label=True,
                )

                train_sampler_vt = CustomSampler(
                    hdf5s_dir=hdf5s_dir_viotech,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
                train_sampler_fl = CustomSampler(
                    hdf5s_dir=hdf5s_dir_mosavpt,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )

                class MixedDatasetVF(object):
                    def __init__(self, vt_ds, mos_ds):
                        self.vt_ds = vt_ds
                        self.mos_ds = mos_ds
                    def __getitem__(self, tagged_meta):
                        src, meta = tagged_meta
                        if src == 'viotech':
                            return self.vt_ds[meta]
                        else:
                            return self.mos_ds[meta]

                train_dataset = MixedDatasetVF(train_dataset_vt, train_dataset_fl)

                class MixedBatchSamplerVF(object):
                    def __init__(self, sampler_vt, sampler_mos, ratio=0.5):
                        self.sampler_vt = sampler_vt
                        self.sampler_mos = sampler_mos
                        self.ratio = ratio
                    def __iter__(self):
                        iter_vt = iter(self.sampler_vt)
                        iter_mos = iter(self.sampler_mos)
                        while True:
                            use_mos = (np.random.rand() < self.ratio)
                            if use_mos:
                                try:
                                    batch = next(iter_mos)
                                    yield [('mosavpt', m) for m in batch]
                                    continue
                                except StopIteration:
                                    iter_mos = iter(self.sampler_mos)
                            try:
                                batch = next(iter_vt)
                                yield [('viotech', m) for m in batch]
                            except StopIteration:
                                iter_vt = iter(self.sampler_vt)
                    def __len__(self):
                        return -1
                    def state_dict(self):
                        state = {}
                        if hasattr(self.sampler_vt, 'state_dict'):
                            state['sampler_vt'] = self.sampler_vt.state_dict()
                        if hasattr(self.sampler_mos, 'state_dict'):
                            state['sampler_mos'] = self.sampler_mos.state_dict()
                        return state
                    def load_state_dict(self, state):
                        if 'sampler_vt' in state and hasattr(self.sampler_vt, 'load_state_dict'):
                            self.sampler_vt.load_state_dict(state['sampler_vt'])
                        if 'sampler_mos' in state and hasattr(self.sampler_mos, 'load_state_dict'):
                            self.sampler_mos.load_state_dict(state['sampler_mos'])

                train_sampler = MixedBatchSamplerVF(
                    train_sampler_vt, train_sampler_fl, ratio=args.mosavpt_ratio
                )

                evaluate_dataset_vt = CustomDataset(
                    hdf5s_dir=hdf5s_dir_viotech,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    include_technique_label=True,
                )
                evaluate_dataset_mos = CustomDataset(
                    hdf5s_dir=hdf5s_dir_mosavpt,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    include_technique_label=True,
                )
                evaluate_dataset = MixedDatasetVF(evaluate_dataset_vt, evaluate_dataset_mos)

                class MixedEvalBatchSamplerVF(object):
                    def __init__(self, sampler_vt, sampler_mos, mos_ratio=0.5):
                        self.sampler_vt = sampler_vt
                        self.sampler_mos = sampler_mos
                        self.mos_ratio = mos_ratio
                    def __iter__(self):
                        iter_vt = iter(self.sampler_vt)
                        iter_mos = iter(self.sampler_mos)
                        vt_done = False
                        mos_done = False
                        while True:
                            use_mos = (np.random.rand() < self.mos_ratio)
                            if use_mos and not mos_done:
                                try:
                                    batch = next(iter_mos)
                                    yield [('mosavpt', m) for m in batch]
                                    continue
                                except StopIteration:
                                    mos_done = True
                            if not vt_done:
                                try:
                                    batch = next(iter_vt)
                                    yield [('viotech', m) for m in batch]
                                except StopIteration:
                                    vt_done = True
                            if vt_done and mos_done:
                                break
                    def __len__(self):
                        return -1

                evaluate_train_sampler = MixedEvalBatchSamplerVF(
                    CustomTestSampler(
                        hdf5s_dir=hdf5s_dir_viotech, split='train',
                        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
                        batch_size=batch_size, mini_data=mini_data,
                    ),
                    CustomTestSampler(
                        hdf5s_dir=hdf5s_dir_mosavpt, split='train',
                        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
                        batch_size=batch_size, mini_data=mini_data,
                    ),
                    mos_ratio=0.5,
                )
                evaluate_test_sampler = MixedEvalBatchSamplerVF(
                    CustomTestSampler(
                        hdf5s_dir=hdf5s_dir_viotech, split='test',
                        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
                        batch_size=batch_size, mini_data=mini_data,
                    ),
                    CustomTestSampler(
                        hdf5s_dir=hdf5s_dir_mosavpt, split='test',
                        segment_seconds=segment_seconds, hop_seconds=hop_seconds,
                        batch_size=batch_size, mini_data=mini_data,
                    ),
                    mos_ratio=0.5,
                )
            else:
                train_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=max_note_shift,
                    augmentor=augmentor,
                    include_technique_label=(dataset == 'mosapt'),
                )
                evaluate_dataset = CustomDataset(
                    hdf5s_dir=hdf5s_dir,
                    segment_seconds=segment_seconds,
                    frames_per_second=frames_per_second,
                    max_note_shift=0,
                    include_technique_label=(dataset == 'mosapt'),
                )

                # Sampler for training
                train_sampler = CustomSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )

                # Samplers for evaluation (single dataset case only)
                evaluate_train_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='train',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
                evaluate_test_sampler = CustomTestSampler(
                    hdf5s_dir=hdf5s_dir,
                    split='test',
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                    batch_size=batch_size,
                    mini_data=mini_data,
                )
    else:
        train_dataset = MaestroDataset(
            hdf5s_dir=hdf5s_dir,
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
            max_note_shift=max_note_shift,
            augmentor=augmentor,
        )
        evaluate_dataset = MaestroDataset(
            hdf5s_dir=hdf5s_dir,
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
            max_note_shift=0,
        )

        # Sampler for training
        train_sampler = Sampler(
            hdf5s_dir=hdf5s_dir,
            split='train',
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            mini_data=mini_data,
        )

        # Samplers for evaluation
        evaluate_train_sampler = TestSampler(
            hdf5s_dir=hdf5s_dir,
            split='train',
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            mini_data=mini_data,
        )
        evaluate_test_sampler = TestSampler(
            hdf5s_dir=hdf5s_dir,
            split='test',
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            mini_data=mini_data,
        )


    # Dataloader
    if dataset == 'rwc_tech_note_wav':
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset,
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = evaluate_train_loader
    else:
        print(f'Creating train and evaluate loaders for {dataset}...')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
            batch_sampler=train_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)

        evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
            batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)
            
        test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
            batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # RWC evaluation dataloader (cross-dataset eval during training)
    rwc_eval_loader = None
    rwc_eval_h5 = getattr(args, 'rwc_eval_h5_path', None)
    if rwc_eval_h5:
        rwc_eval_h5 = os.path.expanduser(rwc_eval_h5)
        if os.path.isfile(rwc_eval_h5):
            from rwc_moe_utils import RWCMoEDataset, RWCMoETestSampler
            rwc_eval_dataset = RWCMoEDataset(
                rwc_h5_path=rwc_eval_h5,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
            )
            rwc_eval_sampler = RWCMoETestSampler(
                rwc_h5_path=rwc_eval_h5,
                split=getattr(args, 'rwc_eval_split', 'test'),
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                fold_id=getattr(args, 'rwc_eval_fold', 0),
            )
            rwc_max_iter = getattr(args, 'rwc_eval_max_iterations', 200)
            max_rwc = len(rwc_eval_sampler.segment_list) // max(batch_size, 1)
            if rwc_max_iter <= 0:
                rwc_eval_sampler.max_evaluate_iteration = max_rwc
            else:
                rwc_eval_sampler.max_evaluate_iteration = min(rwc_max_iter, max_rwc)
            rwc_eval_loader = torch.utils.data.DataLoader(
                dataset=rwc_eval_dataset,
                batch_sampler=rwc_eval_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
            )
            logging.info(f'RWC evaluation enabled: {rwc_eval_h5} '
                         f'(split={args.rwc_eval_split}, fold={args.rwc_eval_fold}, '
                         f'segments={len(rwc_eval_sampler.segment_list)}, '
                         f'max_batches={rwc_eval_sampler.max_evaluate_iteration})')
        else:
            logging.warning(f'RWC eval H5 not found: {rwc_eval_h5}, skipping RWC evaluation.')

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    # Optimize params: for rwc_tech with technique-only training, freeze non-technique branches
    if dataset == 'rwc_tech' and technique_weight > 0:
        for name, p in model.named_parameters():
            is_tech = ('technique_model' in name) or ('technique_fc' in name) or ('bn0' in name) or ('logmel_extractor' in name) or ('spectrogram_extractor' in name)
            # Keep front-end (spec/logmel) trainable if you want full finetune; here we freeze them
            if is_tech:
                p.requires_grad = True
            else:
                p.requires_grad = False
    if freeze_transcription:
        for p in model.parameters():
            p.requires_grad = False
        for sub in ['technique_acoustic_frame_moe', 'frame_moe_head',
                     'technique_acoustic', 'technique_head',
                     'technique_acoustic_moe', 'moe_technique_head']:
            if hasattr(model, sub):
                for p in getattr(model, sub).parameters():
                    p.requires_grad = True
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logging.info(f'Freeze transcription: {n_train}/{n_total} params trainable')
    params = [p for p in model.parameters() if p.requires_grad]
    if use_aux_contrast_model:
        params += list(aux_contrast_onset_model.parameters())
        params += list(aux_contrast_offset_model.parameters())

    optimizer = optim.Adam(params, lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    if use_cosine_annealing_warm_restarts:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=early_stop//2, T_mult=2, eta_min=0.05*learning_rate)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=early_stop, eta_min=0.05*learning_rate)

    # Load pretrained checkpoint if available
    if args.pretrain_path is not None and os.path.exists(args.pretrain_path):
        logging.info('Loading pretrained checkpoint from {}'.format(args.pretrain_path))
        checkpoint = torch.load(args.pretrain_path, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model'], strict=False)
        except TypeError:
            # For older torch versions without strict kw-only
            model.load_state_dict(checkpoint['model'])
        logging.info('Pretrained checkpoint loaded successfully')

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            model_type, 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
                '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # model.to("cuda")

    if 'cuda' in str(device):
        model.to(device)
        if use_aux_contrast_model:
            aux_contrast_onset_model.to(device)
            aux_contrast_offset_model.to(device)
    else:
        print('Using CPU.....')

    # Parallel
    # print('GPU number: {}'.format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)
    
    # Best-checkpoint tracking & patience-based early stopping
    patience = getattr(args, 'patience', 0)
    best_metric_name = getattr(args, 'best_metric', 'fmoe_loss_technique')
    best_metric_higher_better = best_metric_name.endswith('_ap') or best_metric_name.endswith('_acc')
    best_metric_val = float('-inf') if best_metric_higher_better else float('inf')
    best_iteration = 0
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
    patience_active = patience > 0
    if patience_active:
        logging.info(f'Patience-based early stopping enabled: '
                     f'metric={best_metric_name}, patience={patience}, '
                     f'higher_better={best_metric_higher_better}')

    print(f'Training model...')
    train_bgn_time = time.time()

    for batch_data_dict in train_loader:

        # Evaluation 
        if iteration % 1000 == 0:# and iteration > 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            logging.info('Evaluating on train set...')
            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            logging.info('Evaluating on test set...')
            test_statistics = evaluator.evaluate(test_loader)

            logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
            # logging.info('    Validation statistics: {}'.format(validate_statistics))
            logging.info('    Test statistics: {}'.format(test_statistics))

            # Log evaluation metrics to TensorBoard
            if evaluate_train_statistics:
                for key, value in evaluate_train_statistics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'train/{key}', value, iteration)
            
            if test_statistics:
                for key, value in test_statistics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'test/{key}', value, iteration)

            # RWC cross-dataset evaluation
            rwc_statistics = {}
            if rwc_eval_loader is not None:
                logging.info('Evaluating on RWC set...')
                rwc_statistics = evaluate_rwc_frame_moe(model, rwc_eval_loader, device)
                logging.info('    RWC statistics: {}'.format(rwc_statistics))
                for key, value in rwc_statistics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'rwc/{key}', value, iteration)

            statistics_container.append(iteration, evaluate_train_statistics, data_type='train')
            # statistics_container.append(iteration, validate_statistics, data_type='validation')
            statistics_container.append(iteration, test_statistics, data_type='test')
            if rwc_statistics:
                statistics_container.append(iteration, rwc_statistics, data_type='rwc')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            # Best-checkpoint tracking on test (validation) set
            if patience_active and test_statistics and best_metric_name in test_statistics:
                current_val = test_statistics[best_metric_name]
                improved = (current_val > best_metric_val if best_metric_higher_better
                            else current_val < best_metric_val)
                if improved:
                    best_metric_val = current_val
                    best_iteration = iteration
                    ckpt = {'iteration': iteration,
                            'model': model.state_dict(),
                            'sampler': train_sampler.state_dict(),
                            'best_metric': best_metric_name,
                            'best_metric_val': best_metric_val}
                    torch.save(ckpt, best_checkpoint_path)
                    logging.info(f'Best model saved: {best_metric_name}='
                                 f'{best_metric_val:.6f} at iter {iteration}')
                else:
                    steps_since = iteration - best_iteration
                    logging.info(f'No improvement for {steps_since} steps '
                                 f'(best {best_metric_name}={best_metric_val:.6f} '
                                 f'at iter {best_iteration}, patience={patience})')
                    if steps_since >= patience:
                        logging.info(f'Early stopping triggered at iter {iteration}')
                        break

            train_bgn_time = time.time()
        
        # Save model
        if iteration % 1000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Reduce learning rate
        # if iteration % reduce_iteration == 0 and iteration > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.9

        # Per PyTorch docs, call optimizer.step() before scheduler.step()
        # We log LR after stepping scheduler to reflect the updated value
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iteration)
        scheduler.step()
        
        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
         
        model.train()
        if dataset == 'rwc_tech_note_wav':
            # Build a forward dict compatible with technique loss: logits (B,1,C) and labels (B,1,C)
            # Prepare inputs
            if 'logmel' in batch_data_dict:
                logmel = move_data_to_device(batch_data_dict['logmel'], device)
                # Ensure shape (B, 1, T, F)
                if logmel.ndim == 3:
                    logmel = logmel[:, None, :, :]
            else:
                # If logmel not provided, compute from waveform (fallback)
                logmel = None
            # Build (or cache) note model on first iter
            if not hasattr(train, '_note_model'):
                train._note_model = NoteLevelTechniqueModel(classes_num=5, sample_rate=sample_rate, 
                                                            frames_per_second=frames_per_second, use_trans_features=use_trans_features).to(device)
            note_model = train._note_model
            logits = note_model(logmel)
            probs = torch.softmax(logits, dim=-1)
            B = probs.shape[0]
            batch_output_dict = {'technique_output': probs[:, None, :]}  # (B,1,C)
        else:
            # Build note_info for MoE technique branches if available
            note_info = None
            need_note_info = (technique_moe_weight > 0 or technique_moe_zone_weight > 0
                              or technique_moe_zone_pt_weight > 0)
            if need_note_info and 'note_onset_frames' in batch_data_dict:
                onset_f = batch_data_dict['note_onset_frames'].long()
                offset_f = batch_data_dict['note_offset_frames'].long()
                note_info = {
                    'onset_frames': onset_f,
                    'offset_frames': offset_f,
                    'num_notes': batch_data_dict['num_notes'].long(),
                }
                if 'note_pitches' in batch_data_dict:
                    note_info['pitches'] = batch_data_dict['note_pitches'].long()
                dur_frames = (offset_f - onset_f).clamp(min=1).float()
                note_info['durations'] = torch.log(dur_frames + 1.0)
            batch_output_dict = model(batch_data_dict['waveform'], note_info=note_info)
        if use_aux_contrast_model:
            aux_contrast_onset_model.train()
            aux_contrast_offset_model.train()
            batch_data_dict['reg_onset_features'] = aux_contrast_onset_model(batch_output_dict['reg_onset_features'])
            batch_data_dict['reg_offset_features'] = aux_contrast_offset_model(batch_output_dict['reg_offset_features'])

        print('batch_data_dict keys: ', batch_data_dict.keys())

        # Technique classification losses (3-head for viotech, legacy single-head otherwise)
        if technique_weight > 0 and ('tonal_technique' in batch_data_dict):
            loss_tonal, loss_artic, loss_legato = viotech_technique_losses(
                batch_output_dict, batch_data_dict, device=device)
            loss_technique = loss_tonal + loss_artic + loss_legato
        elif technique_weight > 0 and ('technique' in batch_data_dict):
            loss_technique = technique_frame_ce_loss(
                batch_output_dict, batch_data_dict, device=device)
            loss_tonal = loss_artic = loss_legato = None
        else:
            loss_technique = torch.tensor(0.0, device=device)
            loss_tonal = loss_artic = loss_legato = None

        # Note-level MoE technique losses
        loss_moe_technique = torch.tensor(0.0, device=device)
        loss_moe_balance = torch.tensor(0.0, device=device)
        moe_tonal = moe_artic = moe_legato = None
        if technique_moe_weight > 0 and 'note_tonal_logits' in batch_output_dict:
            moe_tonal, moe_artic, moe_legato, loss_moe_balance = \
                moe_technique_losses(batch_output_dict, batch_data_dict, device=device, focal_gamma=focal_gamma)
            loss_moe_technique = moe_tonal + moe_artic + moe_legato

        # Zone-Specialized MoE technique losses
        loss_zone_moe_technique = torch.tensor(0.0, device=device)
        loss_zone_moe_balance = torch.tensor(0.0, device=device)
        zone_tonal = zone_artic = zone_legato = None
        if technique_moe_zone_weight > 0 and 'zone_tonal_logits' in batch_output_dict:
            zone_tonal, zone_artic, zone_legato, loss_zone_moe_balance = \
                zone_moe_technique_losses(batch_output_dict, batch_data_dict, device=device, focal_gamma=focal_gamma)
            loss_zone_moe_technique = zone_tonal + zone_artic + zone_legato

        # Per-Task Gate Zone MoE technique losses
        loss_pt_moe_technique = torch.tensor(0.0, device=device)
        loss_pt_moe_balance = torch.tensor(0.0, device=device)
        pt_tonal = pt_artic = pt_legato = None
        if technique_moe_zone_pt_weight > 0 and 'pt_tonal_logits' in batch_output_dict:
            pt_tonal, pt_artic, pt_legato, loss_pt_moe_balance = \
                pertask_zone_moe_technique_losses(batch_output_dict, batch_data_dict, device=device, focal_gamma=focal_gamma)
            loss_pt_moe_technique = pt_tonal + pt_artic + pt_legato

        # Frame-level Multi-Scale MoE technique losses
        loss_frame_moe_technique = torch.tensor(0.0, device=device)
        loss_frame_moe_balance = torch.tensor(0.0, device=device)
        loss_fmoe_aux = torch.tensor(0.0, device=device)
        fmoe_tonal = fmoe_artic = fmoe_legato = None
        _has_fmoe_output = (technique_frame_moe_weight > 0
                            and 'fmoe_tonal_logits' in batch_output_dict)
        if _has_fmoe_output:
            _fmoe_result = frame_multiscale_moe_losses(
                batch_output_dict, batch_data_dict, device=device,
                focal_gamma=focal_gamma)
            fmoe_tonal, fmoe_artic, fmoe_legato, loss_frame_moe_balance, loss_fmoe_aux = _fmoe_result
            loss_frame_moe_technique = fmoe_tonal + fmoe_artic + fmoe_legato

        if contrast_weight > 0:
            loss_contrast = onset_binary_supcon_loss(batch_output_dict, batch_data_dict) + offset_binary_supcon_loss(batch_output_dict, batch_data_dict)
        else:
            loss_contrast = torch.tensor(0.0, device=device)

        if ctc_weight > 0:
            loss_ctc = per_pitch_onoff_ctc_loss(model, batch_output_dict, batch_data_dict)
        else:
            loss_ctc = torch.tensor(0.0, device=device)

        base_loss = loss_func(model, batch_output_dict, batch_data_dict)
        if ctc_weight > 0:
            loss_ctc = per_pitch_onoff_ctc_loss(model, batch_output_dict, batch_data_dict)
        else:
            loss_ctc = torch.tensor(0.0, device=device)

        loss = (base_loss
                + contrast_weight * loss_contrast
                + ctc_weight * loss_ctc
                + technique_weight * loss_technique
                + technique_moe_weight * loss_moe_technique
                + moe_balance_coeff * loss_moe_balance
                + technique_moe_zone_weight * loss_zone_moe_technique
                + moe_zone_balance_coeff * loss_zone_moe_balance
                + technique_moe_zone_pt_weight * loss_pt_moe_technique
                + moe_zone_pt_balance_coeff * loss_pt_moe_balance
                + technique_frame_moe_weight * loss_frame_moe_technique
                + frame_moe_balance_coeff * loss_frame_moe_balance
                + fmoe_aux_weight * loss_fmoe_aux)

        # Log loss to TensorBoard
        writer.add_scalar('train/loss', loss.item(), iteration)
        writer.add_scalar('train/loss_base', base_loss.item(), iteration)

        if contrast_weight > 0:
            writer.add_scalar('train/loss_contrast', loss_contrast.item(), iteration)
        if ctc_weight > 0:
            writer.add_scalar('train/loss_ctc', loss_ctc.item(), iteration)
        if technique_weight > 0:
            writer.add_scalar('train/loss_technique', loss_technique.item(), iteration)
            if loss_tonal is not None:
                writer.add_scalar('train/loss_tonal_technique', loss_tonal.item(), iteration)
            if loss_artic is not None:
                writer.add_scalar('train/loss_articulation', loss_artic.item(), iteration)
            if loss_legato is not None:
                writer.add_scalar('train/loss_legato', loss_legato.item(), iteration)
        if technique_moe_weight > 0:
            writer.add_scalar('train/loss_moe_technique', loss_moe_technique.item(), iteration)
            writer.add_scalar('train/loss_moe_balance', loss_moe_balance.item(), iteration)
            if moe_tonal is not None:
                writer.add_scalar('train/loss_moe_tonal', moe_tonal.item(), iteration)
            if moe_artic is not None:
                writer.add_scalar('train/loss_moe_artic', moe_artic.item(), iteration)
            if moe_legato is not None:
                writer.add_scalar('train/loss_moe_legato', moe_legato.item(), iteration)
        if technique_moe_zone_weight > 0:
            writer.add_scalar('train/loss_zone_moe_technique', loss_zone_moe_technique.item(), iteration)
            writer.add_scalar('train/loss_zone_moe_balance', loss_zone_moe_balance.item(), iteration)
            if zone_tonal is not None:
                writer.add_scalar('train/loss_zone_moe_tonal', zone_tonal.item(), iteration)
            if zone_artic is not None:
                writer.add_scalar('train/loss_zone_moe_artic', zone_artic.item(), iteration)
            if zone_legato is not None:
                writer.add_scalar('train/loss_zone_moe_legato', zone_legato.item(), iteration)
        if technique_moe_zone_pt_weight > 0:
            writer.add_scalar('train/loss_pt_moe_technique', loss_pt_moe_technique.item(), iteration)
            writer.add_scalar('train/loss_pt_moe_balance', loss_pt_moe_balance.item(), iteration)
            if pt_tonal is not None:
                writer.add_scalar('train/loss_pt_moe_tonal', pt_tonal.item(), iteration)
            if pt_artic is not None:
                writer.add_scalar('train/loss_pt_moe_artic', pt_artic.item(), iteration)
            if pt_legato is not None:
                writer.add_scalar('train/loss_pt_moe_legato', pt_legato.item(), iteration)
        if technique_frame_moe_weight > 0:
            writer.add_scalar('train/loss_frame_moe_technique', loss_frame_moe_technique.item(), iteration)
            writer.add_scalar('train/loss_frame_moe_balance', loss_frame_moe_balance.item(), iteration)
            if fmoe_aux_weight > 0:
                writer.add_scalar('train/loss_fmoe_aux', loss_fmoe_aux.item(), iteration)
            if fmoe_tonal is not None:
                writer.add_scalar('train/loss_fmoe_tonal', fmoe_tonal.item(), iteration)
            if fmoe_artic is not None:
                writer.add_scalar('train/loss_fmoe_artic', fmoe_artic.item(), iteration)
            if fmoe_legato is not None:
                writer.add_scalar('train/loss_fmoe_legato', fmoe_legato.item(), iteration)

        preview_loss = {
            'loss': loss.item(),
            'base_loss': base_loss.item(),
        }

        if ctc_weight > 0:
            preview_loss['loss_ctc'] = loss_ctc.item()
        if technique_weight > 0:
            preview_loss['loss_technique'] = loss_technique.item()
            if loss_tonal is not None:
                preview_loss['loss_tonal'] = loss_tonal.item()
            if loss_artic is not None:
                preview_loss['loss_artic'] = loss_artic.item()
            if loss_legato is not None:
                preview_loss['loss_legato'] = loss_legato.item()
        if technique_moe_weight > 0:
            preview_loss['loss_moe'] = loss_moe_technique.item()
            preview_loss['loss_moe_bal'] = loss_moe_balance.item()
        if technique_moe_zone_weight > 0:
            preview_loss['loss_zmoe'] = loss_zone_moe_technique.item()
            preview_loss['loss_zmoe_bal'] = loss_zone_moe_balance.item()
        if technique_moe_zone_pt_weight > 0:
            preview_loss['loss_ptmoe'] = loss_pt_moe_technique.item()
            preview_loss['loss_ptmoe_bal'] = loss_pt_moe_balance.item()
        if technique_frame_moe_weight > 0:
            preview_loss['loss_fmoe'] = loss_frame_moe_technique.item()
            preview_loss['loss_fmoe_bal'] = loss_frame_moe_balance.item()
            if fmoe_aux_weight > 0:
                preview_loss['loss_fmoe_aux'] = loss_fmoe_aux.item()
        if contrast_weight > 0:
            preview_loss['loss_contrast'] = loss_contrast.item()
        
        # flatten the string of preview_loss
        preview_loss_str = f'{iteration} | ' + '| '.join([f'{k}: {v:.6f}' for k, v in preview_loss.items()])
        print(preview_loss_str)

        # Backward
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1
    
    # Close TensorBoard writer
    writer.close()
    logging.info('Training completed. TensorBoard writer closed.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--device', type=int, default=None, help='-1 for CPU, 0 for GPU 0, 1 for GPU 1, etc.')
    parser_train.add_argument('--seed', type=int, default=2026,
        help='Random seed for reproducibility (None = non-deterministic)')
    parser_train.add_argument('--pretrain_path', type=str, default=None)
    parser_train.add_argument('--dataset', type=str, choices=['mosa', 'maestro', 'mosapt', 'mixed', 'rwc', 'rwc_tech', 'rwc_tech_note_wav', 'viotech', 'viotech_mixed_mosavpt'], default='mosa')
    parser_train.add_argument('--logdir', type=str, required=True)
    parser_train.add_argument('--model_tag', type=str, required=True)
    parser_train.add_argument('--contrast_weight', type=float, default=0.0)
    parser_train.add_argument('--ctc_weight', type=float, default=0.0)
    parser_train.add_argument('--use_aux_contrast_model', action='store_true', default=False)
    parser_train.add_argument('--use_cosine_annealing_warm_restarts', action='store_true', default=False)
    parser_train.add_argument('--technique_weight', type=float, default=0.0)
    parser_train.add_argument('--mosa_ratio', type=float, default=0.5)
    parser_train.add_argument('--mosavpt_dir', type=str, default='/root/VioPTT/hdf5s/mosapt',
                              help='Directory of mosavpt H5 files for viotech_mixed_mosavpt')
    parser_train.add_argument('--mosavpt_ratio', type=float, default=0.5,
                              help='Probability of sampling from mosavpt in viotech_mixed_mosavpt')
    parser_train.add_argument('--num_workers', type=int, default=8)
    parser_train.add_argument('--rwc_h5_path', type=str, default='~/data/rwc_processed_data.h5')
    parser_train.add_argument('--rwc_train_ratio', type=float, default=0.9)
    parser_train.add_argument('--rwc_split_seed', type=int, default=1234)
    parser_train.add_argument('--rwc_max_files', type=int, default=0)
    parser_train.add_argument('--rwc_notes_root', type=str, default='./rwc_notes')
    parser_train.add_argument('--technique_moe_weight', type=float, default=0.0,
        help='Weight for note-level MoE technique loss (0 = disabled)')
    parser_train.add_argument('--moe_balance_coeff', type=float, default=0.01,
        help='Coefficient for MoE load-balance auxiliary loss')
    parser_train.add_argument('--technique_moe_zone_weight', type=float, default=0.0,
        help='Weight for Zone-Specialized MoE technique loss (0 = disabled)')
    parser_train.add_argument('--moe_zone_balance_coeff', type=float, default=0.01,
        help='Coefficient for Zone MoE load-balance auxiliary loss')
    parser_train.add_argument('--technique_moe_zone_pt_weight', type=float, default=0.0,
        help='Weight for Per-Task Gate Zone MoE technique loss (0 = disabled)')
    parser_train.add_argument('--moe_zone_pt_balance_coeff', type=float, default=0.001,
        help='Coefficient for Per-Task Zone MoE load-balance auxiliary loss')
    parser_train.add_argument('--technique_frame_moe_weight', type=float, default=0.0,
        help='Weight for Frame-level Multi-Scale MoE technique loss (0 = disabled)')
    parser_train.add_argument('--frame_moe_balance_coeff', type=float, default=0.001,
        help='Coefficient for Frame MoE load-balance auxiliary loss')
    parser_train.add_argument('--fmoe_aux_weight', type=float, default=0.0,
        help='Weight for per-expert auxiliary loss (0 = disabled, 0.1 recommended)')
    parser_train.add_argument('--fmoe_spectral_expert', type=int, default=1, choices=[0, 1],
        help='Enable (1) or disable (0) spectral expert in Frame MoE (default: 1)')
    parser_train.add_argument('--fmoe_expert_mask', type=str, default=None,
        help='Comma-separated expert indices to keep (e.g. "0,1,2"). None = all available')
    parser_train.add_argument('--fmoe_top_k', type=int, default=None,
        help='Top-k sparse gating override (0=dense). None = use config default (2)')
    parser_train.add_argument('--fmoe_shared_gate', type=int, default=0, choices=[0, 1],
        help='Use single shared gate for all tasks (1) vs per-task gates (0)')
    parser_train.add_argument('--fmoe_uniform_routing', type=int, default=0, choices=[0, 1],
        help='Bypass gating with equal 1/E weights (1) vs learned gating (0)')
    parser_train.add_argument('--fmoe_temporal_experts', type=str, default=None,
        help='Parameterized temporal experts as "k1:d1,k2:d2,..." '
             '(e.g. "11:1,25:1,51:1,51:2,51:3" for 5-expert). None = legacy 3-expert')
    parser_train.add_argument('--fmoe_no_cues', type=int, default=0, choices=[0, 1],
        help='Disable transcription cue injection (1) vs use cues (0)')
    parser_train.add_argument('--fmoe_detach_cues', type=int, default=0, choices=[0, 1],
        help='Detach transcription cues from gradient graph (1) so technique loss '
             'does not update the transcription backbone (0 = end-to-end, default)')
    parser_train.add_argument('--freeze_transcription', type=int, default=0, choices=[0, 1],
        help='Freeze entire transcription backbone (1); only technique branch is trained. '
             'Implies --fmoe_detach_cues=1 (0 = default, all params trainable)')
    parser_train.add_argument('--fmoe_direct_head', type=int, default=0, choices=[0, 1],
        help='Replace MoE expert routing with a plain MLP head (1) to ablate '
             'whether temporal experts add value (0 = MoE, default)')
    parser_train.add_argument('--fmoe_hierarchical', type=int, default=0, choices=[0, 1],
        help='Use soft hierarchical head (1): binary detection then fine-grained '
             'classification, instead of MoE routing (0 = MoE, default)')
    parser_train.add_argument('--fmoe_hierarchical_multilevel', type=int, default=0, choices=[0, 1],
        help='Hierarchical + multi-level features (1): detector uses both GRU '
             'and CNN local features for improved transient detection')
    parser_train.add_argument('--fmoe_hierarchical_moe', type=int, default=0, choices=[0, 1],
        help='Hierarchical L1 + MoE L2 (1): binary detector + MoE fine-grained '
             'classifier with per-task routing')
    parser_train.add_argument('--fmoe_multi_crnn', type=int, default=0, choices=[0, 1],
        help='Use multi-CRNN MoE (1): shared CNN + per-expert GRU branches '
             'instead of shared CRNN + Conv1D experts (0 = standard, default)')
    parser_train.add_argument('--fmoe_crnn_moe', type=int, default=0, choices=[0, 1],
        help='Use CRNN-as-Expert MoE (1): 8 heterogeneous lightweight CRNNs '
             'with multi-scale temporal kernels as experts (0 = standard, default)')
    parser_train.add_argument('--fmoe_shared_crnn_moe', type=int, default=0, choices=[0, 1],
        help='Use Shared-CNN MoE (1): shared CNN blocks 1-3 + per-expert block4 '
             '& GRU branches with multi-scale temporal kernels (0 = standard, default)')
    parser_train.add_argument('--fmoe_mert_backbone', type=int, default=0, choices=[0, 1],
        help='Use frozen MERT-v1-330M as technique feature extractor (1) instead of CRNN')
    parser_train.add_argument('--fmoe_mert_moe', type=int, default=0, choices=[0, 1],
        help='MERT Multi-Scale MoE (1): per-expert MERT layer weighting + temporal '
             'pyramid Conv1D + per-expert GRU, fully integrated backbone+head')
    parser_train.add_argument('--fmoe_mert_moe_hierarchical', type=int, default=0, choices=[0, 1],
        help='MERT MoE + hierarchical prediction (1): binary detector L1 then '
             'fine-grained classifier L2, composed log-probs')
    parser_train.add_argument('--fmoe_mert_no_layer_weight', type=int, default=0, choices=[0, 1],
        help='Ablation (1): disable per-expert MERT layer weighting; each expert uses '
             'a fixed uniform average over MERT layers ("w/o MERT Layer Weight")')
    parser_train.add_argument('--focal_gamma', type=float, default=0.0,
        help='Focal loss gamma for technique CE losses (0 = standard CE, 2.0 recommended)')
    parser_train.add_argument('--patience', type=int, default=0,
        help='Patience for early stopping based on validation metric (0 = disabled, use --early_stop as max iter)')
    parser_train.add_argument('--best_metric', type=str, default='fmoe_loss_technique',
        help='Metric to monitor for best checkpoint (e.g. fmoe_loss_technique, frame_ap). '
             'Metrics ending with _ap or _acc are treated as higher-is-better.')
    parser_train.add_argument('--technique_label_config', type=str, default=None,
        help='Path to technique_label_config JSON (default: config/technique_label_config.json)')
    parser_train.add_argument('--rwc_eval_h5_path', type=str, default=None,
        help='Path to RWC H5 file for cross-dataset evaluation during training (None = disabled)')
    parser_train.add_argument('--rwc_eval_split', type=str, default='all',
        choices=['train', 'test', 'all'],
        help='RWC split for evaluation during training (all = entire dataset)')
    parser_train.add_argument('--rwc_eval_fold', type=int, default=0,
        help='RWC fold ID for evaluation during training')
    parser_train.add_argument('--rwc_eval_max_iterations', type=int, default=200,
        help='Max batches per RWC evaluation during training')
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        if args.technique_label_config:
            from technique_label_utils import get_technique_labels
            get_technique_labels(args.technique_label_config)
        train(args)

    else:
        raise Exception('Error argument!')