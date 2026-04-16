"""UMAP visualization of Frame-level MoE technique embeddings.

Extracts gate-mixed expert features from the frame MoE head via forward
hooks and creates UMAP scatter plots colored by technique labels
(tonal, articulation, legato — or binary flags).

Usage:
    python visualize_umap_frame_moe.py \
        --checkpoint_path <ckpt> --hdf5s_dir <dir> --output_dir <dir> \
        [same model args as evaluate_frame_moe.py] \
        [--n_per_class 1000] [--n_neighbors 30] [--min_dist 0.1] [--tag exp_name]
"""

import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import argparse
import numpy as np
import torch
import torch.utils.data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from models_contrast import Regress_onset_offset_frame_velocity_CRNN
from moe_frame_multiscale import FrameMultiScaleMoEConfig, FrameMultiCRNNMoEConfig
from pytorch_utils import move_data_to_device
from data_generator import CustomDataset, CustomTestSampler, collate_fn
from technique_label_utils import get_technique_labels
import config

_label_cfg = get_technique_labels()
TONAL_NAMES = _label_cfg.tonal_names
ARTIC_NAMES = _label_cfg.artic_names
LEGATO_NAMES = {0: 'sustained', 1: 'bow_change'}
BINARY_FLAG_NAMES = ['pizzicato', 'harmonic', 'spiccato', 'legato']


def _reload_label_config(config_path):
    global _label_cfg, TONAL_NAMES, ARTIC_NAMES
    _label_cfg = get_technique_labels(config_path)
    TONAL_NAMES = _label_cfg.tonal_names
    ARTIC_NAMES = _label_cfg.artic_names


# -----------------------------------------------------------------------
# Source / performer tracking
# -----------------------------------------------------------------------

def _extract_performer(hdf5_name):
    """'Kayser_Op20-01_AlexandrosIakovou.h5' → 'AlexandrosIakovou'"""
    base = os.path.splitext(hdf5_name)[0]
    parts = base.split('_')
    return parts[-1] if len(parts) >= 3 else base


class _DatasetWithSource:
    """Thin wrapper that injects an integer source_id into each data_dict."""

    def __init__(self, base_dataset, source_map):
        self.base = base_dataset
        self.source_map = source_map  # {hdf5_name: int}

    def __getitem__(self, meta):
        data_dict = self.base[meta]
        key = meta[0] if isinstance(meta, (list, tuple)) else str(meta)
        data_dict['source_id'] = np.int32(self.source_map.get(key, -1))
        return data_dict


def _build_source_map(segment_list):
    """Build source_map and id→name dict from a sampler's segment_list."""
    performer_to_id = {}
    source_map = {}
    for entry in segment_list:
        hdf5_name = entry[0]
        performer = _extract_performer(hdf5_name)
        if performer not in performer_to_id:
            performer_to_id[performer] = len(performer_to_id)
        source_map[hdf5_name] = performer_to_id[performer]
    source_names = {v: k for k, v in performer_to_id.items()}
    return source_map, source_names


# -----------------------------------------------------------------------
# Model building (mirrors evaluate_frame_moe.py logic)
# -----------------------------------------------------------------------

def build_model(args):
    device = (torch.device(f'cuda:{args.device}')
              if args.device >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))

    fps = config.frames_per_second

    fmoe_hierarchical = bool(getattr(args, 'fmoe_hierarchical', 0))
    fmoe_hierarchical_multilevel = bool(getattr(args, 'fmoe_hierarchical_multilevel', 0))
    fmoe_hierarchical_moe = bool(getattr(args, 'fmoe_hierarchical_moe', 0))
    fmoe_binary_technique = bool(getattr(args, 'fmoe_binary_technique', 0))
    fmoe_spectral = bool(getattr(args, 'fmoe_spectral_expert', 1))

    fmoe_mask_str = getattr(args, 'fmoe_expert_mask', None)
    fmoe_expert_mask = (
        tuple(int(x) for x in fmoe_mask_str.split(','))
        if fmoe_mask_str else None
    )
    fmoe_temporal_str = getattr(args, 'fmoe_temporal_experts', None)
    fmoe_temporal_experts = None
    if fmoe_temporal_str:
        fmoe_temporal_experts = tuple(
            tuple(int(x) for x in pair.split(':'))
            for pair in fmoe_temporal_str.split(',')
        )
    fmoe_use_cues = not bool(getattr(args, 'fmoe_no_cues', 0))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model']

    fmoe_multi_crnn = bool(getattr(args, 'fmoe_multi_crnn', 0))
    fmoe_crnn_moe = bool(getattr(args, 'fmoe_crnn_moe', 0))
    fmoe_shared_crnn_moe = bool(getattr(args, 'fmoe_shared_crnn_moe', 0))

    if fmoe_shared_crnn_moe:
        from moe_crnn_backbone import SharedCNNMoEConfig
        _kw = dict(expert_mask=fmoe_expert_mask, use_cues=fmoe_use_cues)
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _kw['top_k'] = fmoe_top_k
        frame_moe_cfg = SharedCNNMoEConfig(**_kw)
    elif fmoe_crnn_moe:
        from moe_crnn_backbone import CRNNMoEConfig
        _kw = dict(expert_mask=fmoe_expert_mask, use_cues=fmoe_use_cues)
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _kw['top_k'] = fmoe_top_k
        frame_moe_cfg = CRNNMoEConfig(**_kw)
    elif fmoe_multi_crnn:
        _kw = dict(
            use_spectral_expert=fmoe_spectral,
            expert_mask=fmoe_expert_mask,
            use_cues=fmoe_use_cues,
        )
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _kw['top_k'] = fmoe_top_k
        frame_moe_cfg = FrameMultiCRNNMoEConfig(**_kw)
    elif fmoe_hierarchical_multilevel:
        from hierarchical_frame import FrameHierarchicalMultiLevelConfig
        frame_moe_cfg = FrameHierarchicalMultiLevelConfig(use_cues=fmoe_use_cues)
    elif fmoe_hierarchical_moe:
        from hierarchical_frame import FrameHierarchicalMoEConfig
        frame_moe_cfg = FrameHierarchicalMoEConfig(use_cues=fmoe_use_cues)
    elif fmoe_hierarchical:
        from hierarchical_frame import FrameHierarchicalConfig
        frame_moe_cfg = FrameHierarchicalConfig(use_cues=fmoe_use_cues)
    else:
        if fmoe_temporal_experts is None and fmoe_use_cues:
            _e0_key = 'frame_moe_head.experts.0.net.0.weight'
            if _e0_key in state:
                _ckpt_in = state[_e0_key].shape[1]
                _P, _C = 128, 16
                _first_eid = min(fmoe_expert_mask) if fmoe_expert_mask else 0
                _legacy_dims = {0: _P + _C, 1: _P + _C, 2: _P + 3 * _C}
                _legacy_expected = _legacy_dims.get(_first_eid, _P + 3 * _C)
                if _ckpt_in != _legacy_expected and _ckpt_in == _P + 3 * _C:
                    fmoe_temporal_experts = ((11, 1), (51, 1), (51, 3))

        _kw = dict(
            use_spectral_expert=fmoe_spectral,
            expert_mask=fmoe_expert_mask,
            shared_gate=bool(getattr(args, 'fmoe_shared_gate', 0)),
            uniform_routing=bool(getattr(args, 'fmoe_uniform_routing', 0)),
            temporal_experts=fmoe_temporal_experts,
            use_cues=fmoe_use_cues,
            binary_technique=fmoe_binary_technique,
        )
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _kw['top_k'] = fmoe_top_k
        frame_moe_cfg = FrameMultiScaleMoEConfig(**_kw)

    fmoe_direct_head = bool(getattr(args, 'fmoe_direct_head', 0))

    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=fps,
        classes_num=config.classes_num,
        output_features=True,
        predict_technique=False,
        predict_technique_moe=False,
        predict_technique_moe_zone=False,
        predict_technique_moe_zone_pt=False,
        predict_technique_frame_moe=True,
        frame_moe_config=frame_moe_cfg,
        direct_head=fmoe_direct_head,
        hierarchical_head=fmoe_hierarchical,
        hierarchical_multilevel_head=fmoe_hierarchical_multilevel,
        hierarchical_moe_head=fmoe_hierarchical_moe,
        multi_crnn_head=fmoe_multi_crnn,
        crnn_moe_head=fmoe_crnn_moe,
        shared_crnn_moe_head=fmoe_shared_crnn_moe,
    )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f'[UMAP] Model loaded from {args.checkpoint_path}')

    return model, device, fmoe_binary_technique


# -----------------------------------------------------------------------
# Hook-based feature extraction
# -----------------------------------------------------------------------

def _register_hooks(model, binary_mode):
    """Register forward hooks on classification heads to capture their 128-D input."""
    captured = {}
    hooks = []
    head = model.frame_moe_head

    def _make_hook(key):
        def fn(module, input, output):
            captured[key] = input[0].detach()
        return fn

    if binary_mode and hasattr(head, 'heads'):
        for name in BINARY_FLAG_NAMES:
            if name in head.heads:
                hooks.append(
                    head.heads[name].register_forward_hook(_make_hook(name)))
    else:
        for attr, key in [('tonal_head', 'tonal'),
                          ('artic_head', 'artic'),
                          ('legato_head', 'legato')]:
            if hasattr(head, attr):
                hooks.append(
                    getattr(head, attr).register_forward_hook(_make_hook(key)))

    return captured, hooks


def collect_embeddings(model, loader, device, binary_mode=False, max_batches=0):
    """Run inference and collect per-frame embeddings + GT labels.

    max_batches=0 means unlimited (use all available batches).
    If the batch contains 'source_id' (injected by _DatasetWithSource),
    it is collected alongside features for per-performer marker plotting.
    """
    captured, hooks = _register_hooks(model, binary_mode)

    task_feats = {}
    task_labels = {}
    task_source = {}
    combined_feats = []
    combined_flag_labels = {f: [] for f in BINARY_FLAG_NAMES} if binary_mode else {}
    combined_mc_labels = {}
    combined_source_ids = []
    has_source = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            batch_source = batch.pop('source_id', None)
            for k in batch:
                batch[k] = move_data_to_device(batch[k], device)

            model(batch['waveform'])

            if not captured:
                continue

            ref = next(iter(captured.values()))
            T_feat = ref.shape[1]

            if 'frame_roll' in batch:
                active = (batch['frame_roll'].sum(dim=-1) > 0)
            else:
                active = torch.ones(ref.shape[0], T_feat,
                                    dtype=torch.bool, device=device)

            T = min(T_feat, active.shape[1])
            active = active[:, :T]

            B = active.shape[0]
            for b in range(B):
                m = active[b].cpu().numpy().astype(bool)
                n_active = m.sum()
                if n_active == 0:
                    continue

                feat_key = next(iter(captured.keys()))
                shared_feats = captured[feat_key][b, :T].cpu().numpy()[m]

                src_fill = None
                if batch_source is not None:
                    has_source = True
                    sid = int(batch_source[b])
                    src_fill = np.full(n_active, sid, dtype=np.int32)

                if binary_mode:
                    frame_feats = shared_feats
                    flag_row = {}
                    for flag in BINARY_FLAG_NAMES:
                        gt = batch.get(flag)
                        if gt is None:
                            continue
                        lab = (gt[b, :T] > 0.5).long().cpu().numpy()[m]
                        task_feats.setdefault(flag, []).append(frame_feats)
                        task_labels.setdefault(flag, []).append(lab)
                        if src_fill is not None:
                            task_source.setdefault(flag, []).append(src_fill)
                        flag_row[flag] = lab

                    if flag_row:
                        combined_feats.append(shared_feats)
                        if src_fill is not None:
                            combined_source_ids.append(src_fill)
                        for flag in BINARY_FLAG_NAMES:
                            if flag in flag_row:
                                combined_flag_labels[flag].append(flag_row[flag])
                else:
                    mc_row = {}
                    for src_key, task_key, is_soft in [
                        ('tonal_technique', 'tonal', False),
                        ('articulation', 'artic', False),
                        ('legato', 'legato', True),
                    ]:
                        gt = batch.get(src_key)
                        if gt is None:
                            continue
                        if is_soft:
                            lab = (gt[b, :T] > 0.5).long().cpu().numpy()[m]
                        else:
                            lab = gt[b, :T].cpu().numpy()[m].astype(int)

                        feat = captured.get(task_key)
                        if feat is None:
                            feat = next(iter(captured.values()))
                        frame_feats = feat[b, :T].cpu().numpy()[m]

                        task_feats.setdefault(task_key, []).append(frame_feats)
                        task_labels.setdefault(task_key, []).append(lab)
                        if src_fill is not None:
                            task_source.setdefault(task_key, []).append(src_fill)
                        mc_row[task_key] = lab

                    if mc_row:
                        combined_feats.append(shared_feats)
                        if src_fill is not None:
                            combined_source_ids.append(src_fill)
                        for k, v in mc_row.items():
                            combined_mc_labels.setdefault(k, []).append(v)

            if (batch_idx + 1) % 50 == 0:
                n = sum(len(x) for x in combined_feats)
                print(f'  [UMAP] batch {batch_idx + 1}: {n} active frames collected')

    for h in hooks:
        h.remove()

    # Per-task results (with optional source_ids)
    result = {}
    for key in task_feats:
        entry = {
            'features': np.concatenate(task_feats[key], axis=0),
            'labels': np.concatenate(task_labels[key], axis=0),
        }
        if key in task_source and task_source[key]:
            entry['source_ids'] = np.concatenate(task_source[key])
        result[key] = entry

    # Combined result
    if combined_feats:
        c_feats = np.concatenate(combined_feats, axis=0)
        if binary_mode:
            c_labels = _compose_binary_combined(
                {f: np.concatenate(v) for f, v in combined_flag_labels.items() if v})
        else:
            c_labels = _compose_multiclass_combined(
                {k: np.concatenate(v) for k, v in combined_mc_labels.items() if v})
        combined_entry = {'features': c_feats, 'labels': c_labels}
        if has_source and combined_source_ids:
            combined_entry['source_ids'] = np.concatenate(combined_source_ids)
        result['combined'] = combined_entry

    return result


# Combined technique label composers

COMBINED_BINARY_NAMES = {
    0: 'normal', 1: 'pizzicato', 2: 'harmonic', 3: 'spiccato', 4: 'legato',
}

def _compose_binary_combined(flag_arrays):
    """Compose a single categorical label from binary flag arrays.

    Priority: pizzicato > harmonic > spiccato > legato > normal.
    """
    n = len(next(iter(flag_arrays.values())))
    labels = np.zeros(n, dtype=int)
    for flag, cls_id in [('legato', 4), ('spiccato', 3),
                         ('harmonic', 2), ('pizzicato', 1)]:
        if flag in flag_arrays:
            labels[flag_arrays[flag] == 1] = cls_id
    return labels


COMBINED_MC_NAMES = {
    0: u'détaché', 1: 'harmonics', 2: 'pizzicato', 3: 'openstring',
    4: 'spiccato', 5: 'staccato', 6: 'release', 7: 'legato',
}

def _compose_multiclass_combined(mc_arrays):
    """Compose a single technique label from tonal + articulation + legato.

    Priority: tonal specials > articulation specials > legato > détaché.
    """
    n = len(next(iter(mc_arrays.values())))
    labels = np.zeros(n, dtype=int)

    tonal = mc_arrays.get('tonal')
    artic = mc_arrays.get('artic')
    legato = mc_arrays.get('legato')

    if legato is not None:
        labels[legato == 1] = 7
    if artic is not None:
        for artic_id, comb_id in [(3, 4), (2, 5), (1, 6)]:
            labels[artic == artic_id] = comb_id
    if tonal is not None:
        for tonal_id, comb_id in [(2, 1), (1, 2), (3, 3)]:
            labels[tonal == tonal_id] = comb_id

    return labels


# -----------------------------------------------------------------------
# Sampling, UMAP, plotting
# -----------------------------------------------------------------------

def sample_balanced(features, labels, n_per_class=1000, seed=42, source_ids=None):
    rng = np.random.RandomState(seed)
    idx_list = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        if len(ci) > n_per_class:
            ci = rng.choice(ci, n_per_class, replace=False)
        idx_list.append(ci)
    all_idx = np.sort(np.concatenate(idx_list))
    sampled_src = source_ids[all_idx] if source_ids is not None else None
    return features[all_idx], labels[all_idx], sampled_src


def run_umap(X, n_neighbors=30, min_dist=0.1):
    try:
        import umap
    except ImportError:
        import umap.umap_ as umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric='euclidean', random_state=42)
    return reducer.fit_transform(X)


def _shade(base_rgb, source_idx, n_sources):
    """Darken (idx=0) or lighten (idx=n-1) a base RGB color."""
    if n_sources <= 1:
        return base_rgb[:3]
    t = source_idx / (n_sources - 1)
    if t <= 0.5:
        f = 0.55 + 0.9 * t
        return tuple(c * f for c in base_rgb[:3])
    blend = (t - 0.5) * 0.8
    return tuple(c + (1 - c) * blend for c in base_rgb[:3])


def plot_umap(Z, labels, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    classes = sorted(np.unique(labels))
    cmap = plt.get_cmap('tab10')

    for i, c in enumerate(classes):
        mask = labels == c
        name = class_names.get(int(c), f'class_{c}')
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   s=6, alpha=0.45, color=cmap(i),
                   label=f'{name} (n={int(mask.sum()):,})',
                   rasterized=True)

    ax.legend(loc='best', fontsize=11, markerscale=4, framealpha=0.85)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_umap_with_source(Z, labels, class_names, source_ids, source_names,
                          title, out_path):
    """UMAP scatter: hue = technique class, shade = performer/source."""
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    classes = sorted(np.unique(labels))
    sources = sorted(np.unique(source_ids))
    n_src = len(sources)
    cmap = plt.get_cmap('tab10')

    for i, c in enumerate(classes):
        base = cmap(i % 10)
        for j, s in enumerate(sources):
            mask = (labels == c) & (source_ids == s)
            if not mask.any():
                continue
            color = _shade(base, j, n_src)
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       s=8, alpha=0.5, color=color, rasterized=True)

    tech_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i % 10),
               markersize=9, label=f'{class_names.get(int(c), f"class_{c}")} '
                                   f'(n={int((labels == c).sum()):,})')
        for i, c in enumerate(classes)
    ]
    src_handles = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=_shade((0.5, 0.5, 0.5), j, n_src),
               markeredgecolor=_shade((0.4, 0.4, 0.4), j, n_src),
               markersize=9, label=source_names.get(int(s), f'source_{s}'))
        for j, s in enumerate(sources)
    ]

    leg1 = ax.legend(handles=tech_handles, title='Technique',
                     loc='upper left', fontsize=10, title_fontsize=11,
                     framealpha=0.85, markerscale=1.2)
    ax.add_artist(leg1)
    ax.legend(handles=src_handles, title='Performer (shade)',
              loc='upper right', fontsize=10, title_fontsize=11,
              framealpha=0.85, markerscale=1.2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# -----------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------

def _effective_max_batches(args, max_possible):
    """Resolve max batches: 0 means unlimited."""
    if args.max_batches <= 0:
        return max_possible
    return min(args.max_batches, max_possible)


def _make_viotech_loader(args):
    fps = config.frames_per_second
    segment_seconds = config.segment_seconds

    dataset = CustomDataset(
        hdf5s_dir=args.hdf5s_dir,
        segment_seconds=segment_seconds,
        frames_per_second=fps,
        max_note_shift=0,
        augmentor=None,
        include_technique_label=True,
    )
    sampler = CustomTestSampler(
        hdf5s_dir=args.hdf5s_dir,
        split=args.split,
        segment_seconds=segment_seconds,
        hop_seconds=config.hop_seconds,
        batch_size=args.batch_size,
        mini_data=False,
    )
    max_possible = len(sampler.segment_list) // args.batch_size
    sampler.max_evaluate_iteration = _effective_max_batches(args, max_possible)
    print(f'  [Viotech] {len(sampler.segment_list)} segments, '
          f'{sampler.max_evaluate_iteration} batches')

    source_map, source_names = _build_source_map(sampler.segment_list)
    wrapped = _DatasetWithSource(dataset, source_map)
    print(f'  [Viotech] Performers: {source_names}')

    loader = torch.utils.data.DataLoader(
        dataset=wrapped,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader, source_names


def _make_rwc_loader(args):
    rwc_path = getattr(args, 'rwc_h5_path', None)
    if not rwc_path or not os.path.isfile(rwc_path):
        return None, None
    from rwc_moe_utils import RWCMoEDataset, RWCMoETestSampler

    fps = config.frames_per_second
    segment_seconds = config.segment_seconds
    rwc_dataset = RWCMoEDataset(
        rwc_h5_path=rwc_path,
        segment_seconds=segment_seconds,
        frames_per_second=fps,
    )
    rwc_sampler = RWCMoETestSampler(
        rwc_h5_path=rwc_path,
        split=args.rwc_split,
        segment_seconds=segment_seconds,
        hop_seconds=config.hop_seconds,
        batch_size=args.batch_size,
        fold_id=args.rwc_fold,
    )
    max_rwc = len(rwc_sampler.segment_list) // max(args.batch_size, 1)
    rwc_sampler.max_evaluate_iteration = _effective_max_batches(args, max_rwc)
    print(f'  [RWC] {len(rwc_sampler.segment_list)} segments, '
          f'{rwc_sampler.max_evaluate_iteration} batches')

    loader = torch.utils.data.DataLoader(
        dataset=rwc_dataset,
        batch_sampler=rwc_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader, None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def _task_configs(binary_mode):
    if binary_mode:
        binary_names = {0: 'negative', 1: 'positive'}
        per_flag = [(f, binary_names, f'Binary: {f}') for f in BINARY_FLAG_NAMES]
        combined = [('combined', COMBINED_BINARY_NAMES, 'All Techniques')]
        return per_flag, combined
    per_task = [
        ('tonal', TONAL_NAMES, 'Tonal Technique'),
        ('artic', ARTIC_NAMES, 'Articulation'),
        ('legato', LEGATO_NAMES, 'Legato'),
    ]
    combined = [('combined', COMBINED_MC_NAMES, 'All Techniques')]
    return per_task, combined


def _run_umap_for_tasks(data, task_list, args, dataset_tag, source_names=None):
    """Sample, run UMAP and plot for each task in task_list."""
    for task_key, class_names, task_title in task_list:
        if task_key not in data:
            continue
        feats = data[task_key]['features']
        labels = data[task_key]['labels']
        src_ids = data[task_key].get('source_ids')
        n_classes = len(np.unique(labels))
        print(f'  [{task_key}] {len(feats):,} frames, {n_classes} classes')

        if n_classes < 2:
            print(f'  [{task_key}] Skipping — only 1 class present')
            continue

        X_s, y_s, src_s = sample_balanced(
            feats, labels, args.n_per_class, source_ids=src_ids)
        print(f'  [{task_key}] After sampling: {len(X_s):,} points')

        Z = run_umap(X_s, args.n_neighbors, args.min_dist)

        fname = f'umap_{args.tag}_{dataset_tag}_{task_key}.png'
        out = os.path.join(args.output_dir, fname)
        title = f'{task_title} — {args.tag} ({dataset_tag})'

        if src_s is not None and source_names and len(np.unique(src_s)) > 1:
            plot_umap_with_source(Z, y_s, class_names, src_s, source_names,
                                  title, out)
        else:
            plot_umap(Z, y_s, class_names, title, out)


def _process_dataset(model, loader, device, binary_mode, args,
                     dataset_tag, per_task_cfgs, combined_cfgs,
                     source_names=None):
    print(f'\n[UMAP] Collecting embeddings from {dataset_tag}...')
    data = collect_embeddings(model, loader, device, binary_mode, args.max_batches)

    _run_umap_for_tasks(data, per_task_cfgs, args, dataset_tag, source_names)
    _run_umap_for_tasks(data, combined_cfgs, args, dataset_tag, source_names)


def main():
    parser = argparse.ArgumentParser(
        description='UMAP visualization of Frame-level MoE technique embeddings')

    # Model args (same as evaluate_frame_moe.py)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--hdf5s_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--fmoe_spectral_expert', type=int, default=1)
    parser.add_argument('--fmoe_expert_mask', type=str, default=None)
    parser.add_argument('--fmoe_top_k', type=int, default=None)
    parser.add_argument('--fmoe_shared_gate', type=int, default=0)
    parser.add_argument('--fmoe_uniform_routing', type=int, default=0)
    parser.add_argument('--fmoe_temporal_experts', type=str, default=None)
    parser.add_argument('--fmoe_no_cues', type=int, default=0)
    parser.add_argument('--fmoe_direct_head', type=int, default=0)
    parser.add_argument('--fmoe_hierarchical', type=int, default=0)
    parser.add_argument('--fmoe_hierarchical_multilevel', type=int, default=0)
    parser.add_argument('--fmoe_hierarchical_moe', type=int, default=0)
    parser.add_argument('--fmoe_multi_crnn', type=int, default=0)
    parser.add_argument('--fmoe_crnn_moe', type=int, default=0)
    parser.add_argument('--fmoe_shared_crnn_moe', type=int, default=0)
    parser.add_argument('--fmoe_binary_technique', type=int, default=0)
    parser.add_argument('--technique_label_config', type=str, default=None)

    # RWC
    parser.add_argument('--rwc_h5_path', type=str, default=None)
    parser.add_argument('--rwc_split', type=str, default='all')
    parser.add_argument('--rwc_fold', type=int, default=0)

    # UMAP-specific
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save UMAP plots')
    parser.add_argument('--tag', type=str, default='exp',
                        help='Experiment name used in filenames and titles')
    parser.add_argument('--n_per_class', type=int, default=1000,
                        help='Max points to sample per technique class')
    parser.add_argument('--n_neighbors', type=int, default=30)
    parser.add_argument('--min_dist', type=float, default=0.1)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='Max batches for feature collection (0=all)')

    args = parser.parse_args()

    if args.technique_label_config:
        _reload_label_config(args.technique_label_config)

    model, device, binary_mode = build_model(args)
    os.makedirs(args.output_dir, exist_ok=True)

    per_task_cfgs, combined_cfgs = _task_configs(binary_mode)

    # Viotech
    vio_loader, vio_source_names = _make_viotech_loader(args)
    _process_dataset(model, vio_loader, device, binary_mode, args,
                     'viotech', per_task_cfgs, combined_cfgs,
                     source_names=vio_source_names)

    # RWC (optional)
    rwc_loader, rwc_source_names = _make_rwc_loader(args)
    if rwc_loader is not None:
        _process_dataset(model, rwc_loader, device, binary_mode, args,
                         'rwc', per_task_cfgs, combined_cfgs,
                         source_names=rwc_source_names)

    print('\n[UMAP] Done.')


if __name__ == '__main__':
    main()
