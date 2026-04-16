"""Evaluate Frame-level Multi-Scale MoE technique classification.

Strictly end-to-end: uses transcription branch's onset/frame probabilities
as structural cues — NO ground-truth note boundaries at inference.

Expert roles:
  Expert 0 = Onset specialist  (short-range ~110ms + onset prob)
  Expert 1 = Note specialist   (medium-range ~510ms + frame activity)
  Expert 2 = Phrase specialist  (long-range ~1.5s + all cues)

Three per-task gates (tonal / articulation / legato).

Supports:
  1. Viotech  (--hdf5s_dir)  — default
  2. RWC      (--rwc_h5_path) — optional
"""

import os
import sys
import json

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import argparse
import numpy as np
import torch

from models_contrast import Regress_onset_offset_frame_velocity_CRNN
from moe_frame_multiscale import FrameMultiScaleMoEConfig, FrameMultiCRNNMoEConfig
from pytorch_utils import move_data_to_device
from data_generator import CustomDataset, CustomTestSampler, collate_fn
from technique_label_utils import get_technique_labels
import config

_label_cfg = get_technique_labels()
TONAL_NAMES = _label_cfg.tonal_names
ARTIC_NAMES = _label_cfg.artic_names
# Config class IDs: 0=bow_change, 1=sustained.  After Gaussian binarisation
# (>0.5) the convention flips: 1=bow_change, 0=sustained.  LEGATO_NAMES uses
# the *post-binarisation* convention which is what gt/pred arrays actually hold.
LEGATO_NAMES = {0: 'sustained', 1: 'bow_change'}
EXPERT_NAMES = {0: 'Onset', 1: 'Note', 2: 'Phrase', 3: 'Spectral'}  # overridden at runtime


BINARY_FLAG_NAMES = ['pizzicato', 'harmonic', 'spiccato', 'legato']


def _reload_label_config(config_path):
    """Re-initialise the label singleton and update module-level name dicts."""
    global _label_cfg, TONAL_NAMES, ARTIC_NAMES
    _label_cfg = get_technique_labels(config_path)
    TONAL_NAMES = _label_cfg.tonal_names
    ARTIC_NAMES = _label_cfg.artic_names


def _run_binary_eval_loop(model, test_loader, device, dataset_label='dataset'):
    """Frame-level evaluation loop for binary technique flags."""
    flag_names = BINARY_FLAG_NAMES
    all_gt = {name: [] for name in flag_names}
    all_pred = {name: [] for name in flag_names}
    total_frames = 0

    gate_global = {name: None for name in flag_names}
    gate_by_class = {name: {} for name in flag_names}

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            output_dict = model(batch_data_dict['waveform'])

            if 'fmoe_binary_probs' not in output_dict:
                continue

            if 'frame_roll' in batch_data_dict:
                active = (batch_data_dict['frame_roll'].sum(dim=-1) > 0)
            else:
                first_prob = next(iter(output_dict['fmoe_binary_probs'].values()))
                B, T_p = first_prob.shape[:2]
                active = torch.ones(B, T_p, dtype=torch.bool, device=device)

            probs_dict = output_dict['fmoe_binary_probs']
            gates_dict = output_dict.get('fmoe_binary_gates', {})

            first_prob = next(iter(probs_dict.values()))
            T_pred = first_prob.shape[1]
            T_active = active.shape[1]
            T = min(T_pred, T_active)
            active = active[:, :T]

            B = active.shape[0]
            for b in range(B):
                m = active[b].cpu().numpy().astype(bool)
                n_active = m.sum()
                if n_active == 0:
                    continue
                total_frames += n_active

                for name in flag_names:
                    prob = probs_dict[name][b, :T].squeeze(-1)
                    pred = (prob > 0.5).long().cpu().numpy()[m]
                    all_pred[name].extend(pred)

                    gt_tensor = batch_data_dict.get(name)
                    if gt_tensor is not None:
                        if name == 'legato':
                            gt = (gt_tensor[b, :T] > 0.5).long().cpu().numpy()[m]
                        else:
                            gt = (gt_tensor[b, :T] > 0.5).long().cpu().numpy()[m]
                        all_gt[name].extend(gt)

                        gp = gates_dict.get(name)
                        _accum_gate(name, gp, b, m, gt,
                                    gate_global, gate_by_class, T)

            if (batch_idx + 1) % 20 == 0:
                print(f'  [{dataset_label}] batch {batch_idx + 1}: '
                      f'{total_frames} active frames accumulated')

    result = {}
    for name in flag_names:
        result[name] = {
            'gt': np.array(all_gt[name], dtype=int),
            'pred': np.array(all_pred[name], dtype=int),
        }
    return result, gate_global, gate_by_class, total_frames


def _tonal_none_joint_from_binary(result):
    """GT without pizzicato and without harmonic → multiclass-style tonal 'none'.

    Correct prediction on those frames: both binary heads predict negative (0).
    Aligned frame order is guaranteed by _run_binary_eval_loop.
    """
    if 'pizzicato' not in result or 'harmonic' not in result:
        return None
    g_p = result['pizzicato']['gt']
    g_h = result['harmonic']['gt']
    p_p = result['pizzicato']['pred']
    p_h = result['harmonic']['pred']
    n = min(len(g_p), len(g_h), len(p_p), len(p_h))
    if n == 0:
        return None
    g_p, g_h = g_p[:n], g_h[:n]
    p_p, p_h = p_p[:n], p_h[:n]
    mask = (g_p == 0) & (g_h == 0)
    support = int(mask.sum())
    if support == 0:
        return {'accuracy': 0.0, 'support': 0}
    correct = ((p_p[mask] == 0) & (p_h[mask] == 0)).mean()
    return {'accuracy': round(float(correct), 4), 'support': support}


def _run_eval_loop(model, test_loader, device, dataset_label='dataset'):
    """Frame-level evaluation loop."""
    all_tonal_pred, all_tonal_gt = [], []
    all_artic_pred, all_artic_gt = [], []
    all_legato_pred, all_legato_gt = [], []
    total_frames = 0

    gate_global = {'tonal': None, 'artic': None, 'legato': None}
    gate_by_class = {'tonal': {}, 'artic': {}, 'legato': {}}

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            output_dict = model(batch_data_dict['waveform'])

            if 'fmoe_tonal_logits' not in output_dict:
                continue

            # Active-frame mask
            if 'frame_roll' in batch_data_dict:
                active = (batch_data_dict['frame_roll'].sum(dim=-1) > 0)  # (B, T)
            else:
                B, T = output_dict['fmoe_tonal_logits'].shape[:2]
                active = torch.ones(B, T, dtype=torch.bool, device=device)

            tonal_logits = output_dict['fmoe_tonal_logits']
            artic_logits = output_dict['fmoe_artic_logits']
            legato_prob = output_dict['fmoe_legato_prob'].squeeze(-1)

            T_pred = tonal_logits.shape[1]
            T_active = active.shape[1]
            T = min(T_pred, T_active)

            tonal_pred = tonal_logits[:, :T].argmax(dim=-1)
            artic_pred = artic_logits[:, :T].argmax(dim=-1)
            legato_pred = (legato_prob[:, :T] > 0.5).long()
            active = active[:, :T]

            # GT (legato is soft float [0,1] → binarize: 1=bow_change)
            tonal_gt = batch_data_dict.get('tonal_technique')
            artic_gt = batch_data_dict.get('articulation')
            legato_gt = batch_data_dict.get('legato')

            if tonal_gt is None:
                continue

            tonal_gt = tonal_gt[:, :T]
            artic_gt = artic_gt[:, :T]
            legato_gt = (legato_gt[:, :T] > 0.5).long()

            # Gate probs
            gp_tonal = output_dict.get('fmoe_gate_tonal')
            gp_artic = output_dict.get('fmoe_gate_artic')
            gp_legato = output_dict.get('fmoe_gate_legato')

            B = active.shape[0]
            for b in range(B):
                m = active[b].cpu().numpy().astype(bool)
                n_active = m.sum()
                if n_active == 0:
                    continue

                t_gt = tonal_gt[b].cpu().numpy()[m]
                a_gt = artic_gt[b].cpu().numpy()[m]
                l_gt = legato_gt[b].cpu().numpy()[m]

                all_tonal_pred.extend(tonal_pred[b].cpu().numpy()[m])
                all_tonal_gt.extend(t_gt)
                all_artic_pred.extend(artic_pred[b].cpu().numpy()[m])
                all_artic_gt.extend(a_gt)
                all_legato_pred.extend(legato_pred[b].cpu().numpy()[m])
                all_legato_gt.extend(l_gt)
                total_frames += n_active

                _accum_gate('tonal', gp_tonal, b, m, t_gt, gate_global, gate_by_class, T)
                _accum_gate('artic', gp_artic, b, m, a_gt, gate_global, gate_by_class, T)
                _accum_gate('legato', gp_legato, b, m, l_gt, gate_global, gate_by_class, T)

            if (batch_idx + 1) % 20 == 0:
                print(f'  [{dataset_label}] batch {batch_idx + 1}: '
                      f'{total_frames} active frames accumulated')

    return (np.array(all_tonal_gt, dtype=int), np.array(all_tonal_pred, dtype=int),
            np.array(all_artic_gt, dtype=int), np.array(all_artic_pred, dtype=int),
            np.array(all_legato_gt, dtype=int), np.array(all_legato_pred, dtype=int),
            gate_global, gate_by_class, total_frames)


def _accum_gate(task, gate_tensor, b, active_mask, gt_arr, gate_global, gate_by_class, T):
    if gate_tensor is None:
        return
    g = gate_tensor[b, :T].cpu().numpy()[active_mask]  # (n_active, E)
    if gate_global[task] is None:
        gate_global[task] = g.sum(axis=0)
    else:
        gate_global[task] += g.sum(axis=0)

    for cls_id in np.unique(gt_arr):
        cls_mask = gt_arr == cls_id
        s, c = gate_by_class[task].get(int(cls_id), (np.zeros(g.shape[1]), 0))
        gate_by_class[task][int(cls_id)] = (s + g[cls_mask].sum(axis=0), c + int(cls_mask.sum()))


def _print_binary_results(banner, result, gate_global, gate_by_class, total_frames):
    """Print results for binary technique flag evaluation."""
    print()
    print('=' * 70)
    print(f'  {banner} [Binary Flags] — Total active frames evaluated: {total_frames}')
    print('=' * 70)
    print()
    binary_cls_names = {0: 'negative', 1: 'positive'}
    for name in BINARY_FLAG_NAMES:
        if name not in result:
            continue
        gt = result[name]['gt']
        pred = result[name]['pred']
        _print_multiclass_report(f'Binary: {name}', gt, pred, binary_cls_names)
        _print_pertask_gate(name, gate_global.get(name),
                            gate_by_class.get(name, {}), binary_cls_names)
    joint = _tonal_none_joint_from_binary(result)
    if joint is not None and joint.get('support', 0) > 0:
        print('-' * 70)
        print('  Joint (multiclass-style): tonal none  [GT: no pizz & no harmonic]')
        print('-' * 70)
        print(f"    accuracy  {joint['accuracy']:.4f}  (support {joint['support']})")
        print('    (correct = both heads predict negative)')
        print()


def _build_binary_results_dict(result, gate_global, gate_by_class, total_frames):
    """Build JSON-serialisable results dict for binary technique flags."""
    binary_cls_names = {0: 'negative', 1: 'positive'}
    out = {'total_active_frames': int(total_frames), 'gate_distribution': {}}
    for name in BINARY_FLAG_NAMES:
        if name not in result:
            continue
        gt = result[name]['gt']
        pred = result[name]['pred']
        out[name] = _compute_metrics_dict(gt, pred, binary_cls_names)
        out['gate_distribution'][name] = _gate_to_dict(
            gate_global, gate_by_class, name, binary_cls_names)
    joint = _tonal_none_joint_from_binary(result)
    if joint is not None:
        out['tonal_none_joint'] = joint
    return out


def _print_results(banner, tonal_gt, tonal_pred, artic_gt, artic_pred,
                   legato_gt, legato_pred, gate_global, gate_by_class, total_frames):
    print()
    print('=' * 70)
    print(f'  {banner} — Total active frames evaluated: {total_frames}')
    print('=' * 70)
    print()

    _print_multiclass_report('Tonal Technique', tonal_gt, tonal_pred, TONAL_NAMES)
    _print_multiclass_report('Articulation', artic_gt, artic_pred, ARTIC_NAMES)
    _print_multiclass_report('Legato', legato_gt, legato_pred, LEGATO_NAMES)

    _print_pertask_gate('Tonal', gate_global.get('tonal'),
                        gate_by_class.get('tonal', {}), TONAL_NAMES)
    _print_pertask_gate('Articulation', gate_global.get('artic'),
                        gate_by_class.get('artic', {}), ARTIC_NAMES)
    _print_pertask_gate('Legato', gate_global.get('legato'),
                        gate_by_class.get('legato', {}), LEGATO_NAMES)


def _print_pertask_gate(task_title, global_sum, by_class, class_names):
    if global_sum is None:
        return
    num_experts = len(global_sum)
    pct = global_sum / global_sum.sum() * 100

    print('-' * 70)
    print(f'  Per-Task Gate — {task_title} (global avg routing weight)')
    print('-' * 70)
    for i, p in enumerate(pct):
        role = EXPERT_NAMES.get(i, f'Expert {i}')
        bar = '\u2588' * int(p / 2)
        print(f'  Expert {i} ({role:8s}): {p:5.1f}%  {bar}')
    print()

    if not by_class:
        return

    expert_hdr = ''.join(f'  E{i}({EXPERT_NAMES.get(i,"?"):>5s})' for i in range(num_experts))
    print(f'  {"Class":15s}  {"Frames":>8s}{expert_hdr}')
    for cls_id in sorted(by_class.keys()):
        s, c = by_class[cls_id]
        name = class_names.get(cls_id, f'class_{cls_id}')
        if c == 0:
            continue
        avg = s / c
        pcts = avg / avg.sum() * 100
        cells = ''.join(f'  {p:6.1f}%' for p in pcts)
        print(f'  {name:15s}  {c:8d}{cells}')
    print()

    for cls_id in sorted(by_class.keys()):
        s, c = by_class[cls_id]
        name = class_names.get(cls_id, f'class_{cls_id}')
        if c == 0:
            continue
        avg = s / c
        pcts = avg / avg.sum() * 100
        print(f'  {name}:')
        for i, p in enumerate(pcts):
            role = EXPERT_NAMES.get(i, '?')
            bar = '\u2588' * int(p / 2) + '\u2591' * (50 - int(p / 2))
            print(f'    E{i}({role:8s}): {bar} {p:5.1f}%')
        print()


def evaluate(args):
    device = (torch.device(f'cuda:{args.device}')
              if args.device >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))

    fps = config.frames_per_second
    segment_seconds = config.segment_seconds

    fmoe_hierarchical = bool(getattr(args, 'fmoe_hierarchical', 0))
    fmoe_hierarchical_multilevel = bool(getattr(args, 'fmoe_hierarchical_multilevel', 0))
    fmoe_hierarchical_moe = bool(getattr(args, 'fmoe_hierarchical_moe', 0))
    fmoe_binary_technique = bool(getattr(args, 'fmoe_binary_technique', 0))
    fmoe_mert_backbone = bool(getattr(args, 'fmoe_mert_backbone', 0))
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

    # Load checkpoint first for cue-mode auto-detection
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model']

    fmoe_multi_crnn = bool(getattr(args, 'fmoe_multi_crnn', 0))
    fmoe_crnn_moe = bool(getattr(args, 'fmoe_crnn_moe', 0))
    fmoe_shared_crnn_moe = bool(getattr(args, 'fmoe_shared_crnn_moe', 0))

    if fmoe_shared_crnn_moe:
        from moe_crnn_backbone import SharedCNNMoEConfig
        _sc_kw = dict(
            expert_mask=fmoe_expert_mask,
            use_cues=fmoe_use_cues,
        )
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _sc_kw['top_k'] = fmoe_top_k
        frame_moe_cfg = SharedCNNMoEConfig(**_sc_kw)
        print(f'Frame-MoE mode: Shared-CNN MoE (blocks 1-3 shared + per-expert block4 & GRU)')
        print(f'Frame-MoE expert mask: {fmoe_expert_mask}')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
    elif fmoe_crnn_moe:
        from moe_crnn_backbone import CRNNMoEConfig
        _cm_kw = dict(
            expert_mask=fmoe_expert_mask,
            use_cues=fmoe_use_cues,
        )
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _cm_kw['top_k'] = fmoe_top_k
        frame_moe_cfg = CRNNMoEConfig(**_cm_kw)
        print(f'Frame-MoE mode: CRNN-as-Expert MoE (8 heterogeneous CRNNs)')
        print(f'Frame-MoE expert mask: {fmoe_expert_mask}')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
    elif fmoe_multi_crnn:
        _mc_kw = dict(
            use_spectral_expert=fmoe_spectral,
            expert_mask=fmoe_expert_mask,
            use_cues=fmoe_use_cues,
        )
        fmoe_top_k = getattr(args, 'fmoe_top_k', None)
        if fmoe_top_k is not None:
            _mc_kw['top_k'] = fmoe_top_k
        frame_moe_cfg = FrameMultiCRNNMoEConfig(**_mc_kw)
        print(f'Frame-MoE mode: multi-CRNN (shared CNN + per-expert GRUs)')
        print(f'Frame-MoE spectral expert: {fmoe_spectral}')
        print(f'Frame-MoE expert mask: {fmoe_expert_mask}')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
    elif fmoe_hierarchical_multilevel:
        from hierarchical_frame import FrameHierarchicalMultiLevelConfig
        frame_moe_cfg = FrameHierarchicalMultiLevelConfig(use_cues=fmoe_use_cues)
        print(f'Frame-MoE mode: hierarchical multi-level (GRU + CNN local features)')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
    elif fmoe_hierarchical_moe:
        from hierarchical_frame import FrameHierarchicalMoEConfig
        frame_moe_cfg = FrameHierarchicalMoEConfig(use_cues=fmoe_use_cues)
        print(f'Frame-MoE mode: hierarchical L1 + MoE L2')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
    elif fmoe_hierarchical:
        from hierarchical_frame import FrameHierarchicalConfig
        frame_moe_cfg = FrameHierarchicalConfig(use_cues=fmoe_use_cues)
        print(f'Frame-MoE mode: hierarchical (soft coarse-to-fine)')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
    else:
        # Auto-detect legacy vs unified cue mode from checkpoint weights.
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
                    print(f'Auto-detected: checkpoint trained with unified cues, '
                          f'setting temporal_experts={fmoe_temporal_experts}')

        _fmoe_kw = dict(
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
            _fmoe_kw['top_k'] = fmoe_top_k
        frame_moe_cfg = FrameMultiScaleMoEConfig(**_fmoe_kw)
        print(f'Frame-MoE spectral expert: {fmoe_spectral}')
        print(f'Frame-MoE expert mask: {fmoe_expert_mask}')
        print(f'Frame-MoE temporal experts: {fmoe_temporal_experts}')
        print(f'Frame-MoE use cues: {fmoe_use_cues}')
        print(f'Frame-MoE shared gate: {_fmoe_kw["shared_gate"]}')
        print(f'Frame-MoE uniform routing: {_fmoe_kw["uniform_routing"]}')

    fmoe_direct_head = bool(getattr(args, 'fmoe_direct_head', 0))
    print(f'Frame-MoE direct head: {fmoe_direct_head}')
    print(f'Frame-MoE hierarchical: {fmoe_hierarchical}')
    print(f'Frame-MoE hierarchical multilevel: {fmoe_hierarchical_multilevel}')
    print(f'Frame-MoE hierarchical MoE: {fmoe_hierarchical_moe}')
    print(f'Frame-MoE multi-CRNN: {fmoe_multi_crnn}')
    print(f'Frame-MoE CRNN-MoE: {fmoe_crnn_moe}')
    print(f'Frame-MoE Shared-CRNN-MoE: {fmoe_shared_crnn_moe}')

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
        mert_backbone=fmoe_mert_backbone,
    )

    # Extract checkpoint metadata
    ckpt_info = {
        'checkpoint_path': args.checkpoint_path,
        'iteration': checkpoint.get('iteration', None),
        'best_metric': checkpoint.get('best_metric', None),
        'best_metric_val': checkpoint.get('best_metric_val', None),
    }
    if ckpt_info['iteration'] is not None:
        print(f'Checkpoint info: iteration={ckpt_info["iteration"]}, '
              f'{ckpt_info["best_metric"]}={ckpt_info["best_metric_val"]:.6f}')

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f'Model loaded from {args.checkpoint_path}')

    global EXPERT_NAMES
    if hasattr(model, 'frame_moe_head') and hasattr(model.frame_moe_head, '_expert_names'):
        EXPERT_NAMES = model.frame_moe_head._expert_names

    event_level = bool(getattr(args, 'event_level', False))

    output_json_path = getattr(args, 'output_json', None)
    json_output = {'checkpoint': ckpt_info, 'event_level': event_level}

    # ---- Viotech evaluation ----
    hdf5s_dir = args.hdf5s_dir

    dataset = CustomDataset(
        hdf5s_dir=hdf5s_dir,
        segment_seconds=segment_seconds,
        frames_per_second=fps,
        max_note_shift=0,
        augmentor=None,
        include_technique_label=True,
    )

    test_sampler = CustomTestSampler(
        hdf5s_dir=hdf5s_dir,
        split=args.split,
        segment_seconds=segment_seconds,
        hop_seconds=config.hop_seconds,
        batch_size=args.batch_size,
        mini_data=False,
    )
    max_possible = len(test_sampler.segment_list) // args.batch_size
    test_sampler.max_evaluate_iteration = min(args.max_iterations, max_possible)
    print(f'[Viotech] Test segments: {len(test_sampler.segment_list)}, '
          f'max batches: {test_sampler.max_evaluate_iteration}')

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    mode_str = 'Event-Level' if event_level else 'Frame-Level'
    print(f'Evaluating on {args.split} set from {hdf5s_dir}  [{mode_str}]')
    print(f'Device: {device}  |  batch_size: {args.batch_size}')
    print()

    if event_level:
        if fmoe_binary_technique:
            result_data, total_active, total_tc = _run_event_level_binary_eval_loop(
                model, test_loader, device, dataset_label='Viotech')
            if total_active > 0:
                _print_event_level_binary_results(
                    'Viotech Frame-MoE Binary (event-level)',
                    result_data, total_active, total_tc)
                json_output['viotech'] = _build_event_level_binary_results_dict(
                    result_data, total_active, total_tc)
            else:
                print('No active frames found in the viotech test set.')
        else:
            results = _run_event_level_eval_loop(
                model, test_loader, device, dataset_label='Viotech')
            (t_gt, t_p, a_gt, a_p, l_gt, l_p, total_active, total_tc) = results
            if total_active > 0:
                _print_event_level_results(
                    'Viotech Frame-MoE (event-level)',
                    t_gt, t_p, a_gt, a_p, l_gt, l_p, total_active, total_tc)
                json_output['viotech'] = _build_event_level_results_dict(
                    t_gt, t_p, a_gt, a_p, l_gt, l_p, total_active, total_tc)
            else:
                print('No active frames found in the viotech test set.')
    else:
        if fmoe_binary_technique:
            results = _run_binary_eval_loop(model, test_loader, device, dataset_label='Viotech')
            result_data, gate_global, gate_by_class, total_frames = results
            if total_frames > 0:
                _print_binary_results('Viotech Frame-MoE Binary (end-to-end)',
                                      result_data, gate_global, gate_by_class, total_frames)
                json_output['viotech'] = _build_binary_results_dict(
                    result_data, gate_global, gate_by_class, total_frames)
            else:
                print('No active frames found in the viotech test set.')
        else:
            results = _run_eval_loop(model, test_loader, device, dataset_label='Viotech')
            if results[-1] > 0:
                _print_results('Viotech Frame-MoE (end-to-end)', *results)
                json_output['viotech'] = _build_results_dict(*results)
            else:
                print('No active frames found in the viotech test set.')

    # ---- RWC evaluation (optional) ----
    rwc_path = getattr(args, 'rwc_h5_path', None)
    if rwc_path:
        rwc_path = os.path.expanduser(rwc_path)
        if not os.path.isfile(rwc_path):
            print(f'\n[RWC] H5 file not found: {rwc_path}, skipping.')
            return

        from rwc_moe_utils import RWCMoEDataset, RWCMoETestSampler

        print(f'\n{"=" * 70}')
        print(f'  RWC Evaluation (Frame-level Multi-Scale MoE) [{mode_str}]')
        print(f'{"=" * 70}')

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
        rwc_sampler.max_evaluate_iteration = min(args.max_iterations, max_rwc)

        rwc_loader = torch.utils.data.DataLoader(
            dataset=rwc_dataset,
            batch_sampler=rwc_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if event_level:
            if fmoe_binary_technique:
                result_data, total_active, total_tc = _run_event_level_binary_eval_loop(
                    model, rwc_loader, device, dataset_label='RWC')
                if total_active > 0:
                    _print_event_level_binary_results(
                        'RWC Frame-MoE Binary (event-level, cross-dataset)',
                        result_data, total_active, total_tc)
                    json_output['rwc'] = _build_event_level_binary_results_dict(
                        result_data, total_active, total_tc)
                else:
                    print('[RWC] No active frames found.')
            else:
                rwc_res = _run_event_level_eval_loop(
                    model, rwc_loader, device, dataset_label='RWC')
                (t_gt, t_p, a_gt, a_p, l_gt, l_p, total_active, total_tc) = rwc_res
                if total_active > 0:
                    _print_event_level_results(
                        'RWC Frame-MoE (event-level, cross-dataset)',
                        t_gt, t_p, a_gt, a_p, l_gt, l_p, total_active, total_tc)
                    json_output['rwc'] = _build_event_level_results_dict(
                        t_gt, t_p, a_gt, a_p, l_gt, l_p, total_active, total_tc)
                else:
                    print('[RWC] No active frames found.')
        else:
            if fmoe_binary_technique:
                rwc_results = _run_binary_eval_loop(model, rwc_loader, device, dataset_label='RWC')
                result_data, gate_global, gate_by_class, total_frames = rwc_results
                if total_frames > 0:
                    _print_binary_results('RWC Frame-MoE Binary (cross-dataset)',
                                          result_data, gate_global, gate_by_class, total_frames)
                    json_output['rwc'] = _build_binary_results_dict(
                        result_data, gate_global, gate_by_class, total_frames)
                else:
                    print('[RWC] No active frames found.')
            else:
                rwc_results = _run_eval_loop(model, rwc_loader, device, dataset_label='RWC')
                if rwc_results[-1] > 0:
                    _print_results('RWC Frame-MoE (cross-dataset)', *rwc_results)
                    json_output['rwc'] = _build_results_dict(*rwc_results)
                else:
                    print('[RWC] No active frames found.')

    # ---- Save JSON output ----
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f'\nResults saved to {output_json_path}')


def _print_multiclass_report(title, gt, pred, class_names):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score, f1_score)

    if len(gt) == 0:
        print(f'  {title}: no samples')
        return

    present_classes = sorted(set(gt) | set(pred))
    target_names = [class_names.get(c, f'class_{c}') for c in present_classes]

    acc = accuracy_score(gt, pred)
    macro_f1 = f1_score(gt, pred, labels=present_classes, average='macro', zero_division=0)
    weighted_f1 = f1_score(gt, pred, labels=present_classes, average='weighted', zero_division=0)

    print('-' * 70)
    print(f'  {title}')
    print('-' * 70)
    print(f'  Overall Accuracy : {acc:.4f}')
    print(f'  Macro F1         : {macro_f1:.4f}')
    print(f'  Weighted F1      : {weighted_f1:.4f}')
    print()
    print(classification_report(
        gt, pred,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    cm = confusion_matrix(gt, pred, labels=present_classes)
    col_w = max(12, max(len(n) for n in target_names) + 2)
    header = ''.ljust(col_w) + ''.join(n.ljust(col_w) for n in target_names)
    print('  Confusion Matrix (rows = GT, cols = Predicted):')
    print(f'  {header}')
    for i, row in enumerate(cm):
        cells = ''.join(str(v).ljust(col_w) for v in row)
        print(f'  {target_names[i].ljust(col_w)}{cells}')
    print()

    print('  Per-class Accuracy:')
    for i, cls in enumerate(present_classes):
        mask = gt == cls
        n_total = mask.sum()
        if n_total > 0:
            n_correct = (pred[mask] == cls).sum()
            cls_acc = n_correct / n_total
        else:
            n_correct = 0
            cls_acc = 0.0
        name = class_names.get(cls, f'class_{cls}')
        print(f'    {name:15s}  {n_correct:5d} / {n_total:5d}  = {cls_acc:.4f}')
    print()


def _compute_metrics_dict(gt, pred, class_names):
    """Compute classification metrics and return as a dict."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if len(gt) == 0:
        return {}

    present_classes = sorted(set(gt) | set(pred))
    acc = accuracy_score(gt, pred)
    macro_f1 = f1_score(gt, pred, labels=present_classes, average='macro', zero_division=0)
    weighted_f1 = f1_score(gt, pred, labels=present_classes, average='weighted', zero_division=0)
    macro_prec = precision_score(gt, pred, labels=present_classes, average='macro', zero_division=0)
    macro_rec = recall_score(gt, pred, labels=present_classes, average='macro', zero_division=0)

    per_class = {}
    for cls in present_classes:
        mask = gt == cls
        n_total = int(mask.sum())
        n_correct = int((pred[mask] == cls).sum()) if n_total > 0 else 0
        name = class_names.get(cls, f'class_{cls}')
        per_class[name] = {
            'accuracy': round(n_correct / n_total, 4) if n_total > 0 else 0.0,
            'support': n_total,
        }

    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_precision': round(macro_prec, 4),
        'macro_recall': round(macro_rec, 4),
        'per_class': per_class,
    }


def _gate_to_dict(gate_global, gate_by_class, task_key, class_names):
    """Convert gate distribution to a JSON-serialisable dict."""
    gs = gate_global.get(task_key)
    if gs is None:
        return {}
    pct = (gs / gs.sum() * 100).tolist()
    result = {
        'global': {EXPERT_NAMES.get(i, f'E{i}'): round(p, 2) for i, p in enumerate(pct)},
    }
    bc = gate_by_class.get(task_key, {})
    if bc:
        by_cls = {}
        for cls_id in sorted(bc.keys()):
            s, c = bc[cls_id]
            if c == 0:
                continue
            avg = s / c
            cls_pct = (avg / avg.sum() * 100).tolist()
            name = class_names.get(cls_id, f'class_{cls_id}')
            by_cls[name] = {EXPERT_NAMES.get(i, f'E{i}'): round(p, 2)
                            for i, p in enumerate(cls_pct)}
        result['by_class'] = by_cls
    return result


def _build_results_dict(tonal_gt, tonal_pred, artic_gt, artic_pred,
                        legato_gt, legato_pred, gate_global, gate_by_class,
                        total_frames):
    """Build a full results dict for JSON output."""
    return {
        'total_active_frames': int(total_frames),
        'tonal': _compute_metrics_dict(tonal_gt, tonal_pred, TONAL_NAMES),
        'articulation': _compute_metrics_dict(artic_gt, artic_pred, ARTIC_NAMES),
        'legato': _compute_metrics_dict(legato_gt, legato_pred, LEGATO_NAMES),
        'gate_distribution': {
            'tonal': _gate_to_dict(gate_global, gate_by_class, 'tonal', TONAL_NAMES),
            'articulation': _gate_to_dict(gate_global, gate_by_class, 'artic', ARTIC_NAMES),
            'legato': _gate_to_dict(gate_global, gate_by_class, 'legato', LEGATO_NAMES),
        },
    }


# =====================================================================
#  Event-level evaluation
#  A frame counts as correct ONLY when:
#    1. The transcription branch detects a note AND the pitch is correct
#       (predicted pitch set == GT pitch set for that frame)
#    2. The technique prediction is correct
#  Frames where the transcription is wrong get their technique prediction
#  set to a sentinel (-1) that never matches any GT class, so they
#  automatically become false-negatives for the GT technique class.
# =====================================================================

def _transcription_correct_mask(frame_roll, frame_output):
    """Return (B, T) bool mask: True where predicted pitches == GT pitches."""
    gt_pitches = (frame_roll > 0.5)
    pred_pitches = (frame_output > 0.5)
    return (pred_pitches == gt_pitches).all(dim=-1)


def _run_event_level_eval_loop(model, test_loader, device, dataset_label='dataset'):
    """Event-level evaluation loop for multi-class technique heads."""
    all_tonal_gt, all_tonal_pred = [], []
    all_artic_gt, all_artic_pred = [], []
    all_legato_gt, all_legato_pred = [], []
    total_active = 0
    total_trans_correct = 0

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict:
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            output_dict = model(batch_data_dict['waveform'])

            if 'fmoe_tonal_logits' not in output_dict:
                continue
            if 'frame_roll' not in batch_data_dict or 'frame_output' not in output_dict:
                continue

            frame_roll = batch_data_dict['frame_roll']
            frame_output = output_dict['frame_output']
            active = (frame_roll.sum(dim=-1) > 0)

            tonal_logits = output_dict['fmoe_tonal_logits']
            artic_logits = output_dict['fmoe_artic_logits']
            legato_prob = output_dict['fmoe_legato_prob'].squeeze(-1)

            T = min(frame_roll.shape[1], frame_output.shape[1], tonal_logits.shape[1])
            frame_roll_t = frame_roll[:, :T]
            frame_output_t = frame_output[:, :T]
            active = active[:, :T]

            trans_ok = _transcription_correct_mask(frame_roll_t, frame_output_t)

            tonal_pred = tonal_logits[:, :T].argmax(dim=-1)
            artic_pred = artic_logits[:, :T].argmax(dim=-1)
            legato_pred = (legato_prob[:, :T] > 0.5).long()

            tonal_gt = batch_data_dict.get('tonal_technique')
            artic_gt = batch_data_dict.get('articulation')
            legato_gt = batch_data_dict.get('legato')
            if tonal_gt is None:
                continue
            tonal_gt = tonal_gt[:, :T]
            artic_gt = artic_gt[:, :T]
            legato_gt = (legato_gt[:, :T] > 0.5).long()

            B = active.shape[0]
            for b in range(B):
                m_active = active[b].cpu().numpy().astype(bool)
                m_trans = trans_ok[b].cpu().numpy().astype(bool)
                n_active = int(m_active.sum())
                if n_active == 0:
                    continue
                total_active += n_active
                total_trans_correct += int((m_active & m_trans).sum())

                t_p = tonal_pred[b].cpu().numpy().copy()
                a_p = artic_pred[b].cpu().numpy().copy()
                l_p = legato_pred[b].cpu().numpy().copy()
                t_p[~m_trans] = -1
                a_p[~m_trans] = -1
                l_p[~m_trans] = -1

                all_tonal_gt.extend(tonal_gt[b].cpu().numpy()[m_active])
                all_tonal_pred.extend(t_p[m_active])
                all_artic_gt.extend(artic_gt[b].cpu().numpy()[m_active])
                all_artic_pred.extend(a_p[m_active])
                all_legato_gt.extend(legato_gt[b].cpu().numpy()[m_active])
                all_legato_pred.extend(l_p[m_active])

            if (batch_idx + 1) % 20 == 0:
                pct = total_trans_correct / max(total_active, 1) * 100
                print(f'  [{dataset_label}] batch {batch_idx + 1}: '
                      f'{total_active} active, {total_trans_correct} trans-ok ({pct:.1f}%)')

    return (np.array(all_tonal_gt, dtype=int), np.array(all_tonal_pred, dtype=int),
            np.array(all_artic_gt, dtype=int), np.array(all_artic_pred, dtype=int),
            np.array(all_legato_gt, dtype=int), np.array(all_legato_pred, dtype=int),
            total_active, total_trans_correct)


def _run_event_level_binary_eval_loop(model, test_loader, device, dataset_label='dataset'):
    """Event-level evaluation loop for binary technique flags."""
    flag_names = BINARY_FLAG_NAMES
    all_gt = {n: [] for n in flag_names}
    all_pred = {n: [] for n in flag_names}
    total_active = 0
    total_trans_correct = 0

    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(test_loader):
            for key in batch_data_dict:
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            output_dict = model(batch_data_dict['waveform'])

            if 'fmoe_binary_probs' not in output_dict:
                continue
            if 'frame_roll' not in batch_data_dict or 'frame_output' not in output_dict:
                continue

            frame_roll = batch_data_dict['frame_roll']
            frame_output = output_dict['frame_output']
            active = (frame_roll.sum(dim=-1) > 0)
            probs_dict = output_dict['fmoe_binary_probs']

            first_prob = next(iter(probs_dict.values()))
            T = min(frame_roll.shape[1], frame_output.shape[1],
                    first_prob.shape[1], active.shape[1])
            frame_roll_t = frame_roll[:, :T]
            frame_output_t = frame_output[:, :T]
            active = active[:, :T]

            trans_ok = _transcription_correct_mask(frame_roll_t, frame_output_t)

            B = active.shape[0]
            for b in range(B):
                m_active = active[b].cpu().numpy().astype(bool)
                m_trans = trans_ok[b].cpu().numpy().astype(bool)
                n_active = int(m_active.sum())
                if n_active == 0:
                    continue
                total_active += n_active
                total_trans_correct += int((m_active & m_trans).sum())

                for name in flag_names:
                    prob = probs_dict[name][b, :T].squeeze(-1)
                    pred = (prob > 0.5).long().cpu().numpy().copy()
                    pred[~m_trans] = -1

                    gt_tensor = batch_data_dict.get(name)
                    if gt_tensor is not None:
                        gt = (gt_tensor[b, :T] > 0.5).long().cpu().numpy()
                        all_gt[name].extend(gt[m_active])
                        all_pred[name].extend(pred[m_active])

            if (batch_idx + 1) % 20 == 0:
                pct = total_trans_correct / max(total_active, 1) * 100
                print(f'  [{dataset_label}] batch {batch_idx + 1}: '
                      f'{total_active} active, {total_trans_correct} trans-ok ({pct:.1f}%)')

    result = {}
    for name in flag_names:
        result[name] = {
            'gt': np.array(all_gt[name], dtype=int),
            'pred': np.array(all_pred[name], dtype=int),
        }
    return result, total_active, total_trans_correct


def _compute_event_level_metrics(gt, pred, class_names):
    """Compute event-level metrics.  pred may contain -1 sentinels for
    frames where transcription was wrong."""
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    if len(gt) == 0:
        return {}, []

    gt_classes = sorted(set(gt))
    event_correct = (gt == pred)
    event_acc = float(event_correct.mean())

    macro_f1 = float(f1_score(gt, pred, labels=gt_classes,
                               average='macro', zero_division=0))
    weighted_f1 = float(f1_score(gt, pred, labels=gt_classes,
                                  average='weighted', zero_division=0))
    macro_prec = float(precision_score(gt, pred, labels=gt_classes,
                                        average='macro', zero_division=0))
    macro_rec = float(recall_score(gt, pred, labels=gt_classes,
                                    average='macro', zero_division=0))

    per_class_f1 = f1_score(gt, pred, labels=gt_classes,
                             average=None, zero_division=0)
    per_class = {}
    per_class_f1_list = []
    for cls, f1_val in zip(gt_classes, per_class_f1):
        name = class_names.get(cls, f'class_{cls}')
        mask = gt == cls
        n_total = int(mask.sum())
        n_correct = int((pred[mask] == cls).sum()) if n_total > 0 else 0
        _acc = round(n_correct / n_total, 4) if n_total > 0 else 0.0
        per_class[name] = {
            'accuracy': _acc,
            'event_accuracy': _acc,
            'f1': round(float(f1_val), 4),
            'support': n_total,
        }
        per_class_f1_list.append(float(f1_val))

    metrics_dict = {
        'accuracy': round(event_acc, 4),
        'event_accuracy': round(event_acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_precision': round(macro_prec, 4),
        'macro_recall': round(macro_rec, 4),
        'per_class': per_class,
    }
    return metrics_dict, per_class_f1_list


def _print_event_level_head(title, gt, pred, class_names):
    """Print event-level metrics for one technique head."""
    from sklearn.metrics import classification_report, f1_score

    if len(gt) == 0:
        print(f'  {title}: no samples')
        return []

    gt_classes = sorted(set(gt))
    target_names = [class_names.get(c, f'class_{c}') for c in gt_classes]

    event_acc = (gt == pred).mean()
    macro_f1 = f1_score(gt, pred, labels=gt_classes, average='macro', zero_division=0)
    per_class_f1_vals = f1_score(gt, pred, labels=gt_classes, average=None, zero_division=0)

    n_trans_wrong = int((pred == -1).sum())
    pct_wrong = n_trans_wrong / max(len(pred), 1) * 100

    print('-' * 70)
    print(f'  {title}  (event-level)')
    print('-' * 70)
    print(f'  Event Accuracy    : {event_acc:.4f}')
    print(f'  Macro F1          : {macro_f1:.4f}')
    print(f'  Trans-wrong frames: {n_trans_wrong} ({pct_wrong:.1f}% of active)')
    print()

    all_labels = sorted(set(gt) | (set(pred) - {-1}))
    names = [class_names.get(c, f'class_{c}') for c in all_labels]
    print(classification_report(
        gt, pred, labels=all_labels, target_names=names,
        digits=4, zero_division=0))

    print('  Per-class Event Accuracy:')
    for i, cls in enumerate(gt_classes):
        mask = gt == cls
        n_total = mask.sum()
        n_correct = int((pred[mask] == cls).sum()) if n_total > 0 else 0
        cls_acc = n_correct / n_total if n_total > 0 else 0.0
        name = class_names.get(cls, f'class_{cls}')
        print(f'    {name:15s}  {n_correct:5d} / {n_total:5d}  = {cls_acc:.4f}'
              f'  (F1={per_class_f1_vals[i]:.4f})')
    print()
    return list(per_class_f1_vals.astype(float))


def _print_event_level_results(banner, tonal_gt, tonal_pred, artic_gt, artic_pred,
                               legato_gt, legato_pred, total_active, total_trans_correct):
    pct = total_trans_correct / max(total_active, 1) * 100
    print()
    print('=' * 70)
    print(f'  {banner} [Event-Level]')
    print(f'  Active frames: {total_active}  |  '
          f'Transcription correct: {total_trans_correct} ({pct:.1f}%)')
    print('=' * 70)

    all_f1 = []
    all_f1 += _print_event_level_head('Tonal Technique', tonal_gt, tonal_pred, TONAL_NAMES)
    all_f1 += _print_event_level_head('Articulation', artic_gt, artic_pred, ARTIC_NAMES)
    all_f1 += _print_event_level_head('Legato', legato_gt, legato_pred, LEGATO_NAMES)

    if all_f1:
        overall = np.mean(all_f1)
        print('=' * 70)
        print(f'  All-Technique Event-Level Macro F1: {overall:.4f}')
        print(f'  (mean of {len(all_f1)} per-class F1 scores across all heads)')
        print('=' * 70)
        print()


def _print_event_level_binary_results(banner, result, total_active, total_trans_correct):
    pct = total_trans_correct / max(total_active, 1) * 100
    print()
    print('=' * 70)
    print(f'  {banner} [Event-Level, Binary Flags]')
    print(f'  Active frames: {total_active}  |  '
          f'Transcription correct: {total_trans_correct} ({pct:.1f}%)')
    print('=' * 70)

    binary_cls = {0: 'negative', 1: 'positive'}
    all_f1 = []
    for name in BINARY_FLAG_NAMES:
        if name not in result:
            continue
        gt = result[name]['gt']
        pred = result[name]['pred']
        all_f1 += _print_event_level_head(f'Binary: {name}', gt, pred, binary_cls)

    if all_f1:
        overall = np.mean(all_f1)
        print('=' * 70)
        print(f'  All-Technique Event-Level Macro F1: {overall:.4f}')
        print(f'  (mean of {len(all_f1)} per-class F1 scores across all flags)')
        print('=' * 70)
        print()


def _build_event_level_results_dict(tonal_gt, tonal_pred, artic_gt, artic_pred,
                                    legato_gt, legato_pred,
                                    total_active, total_trans_correct):
    pct = round(total_trans_correct / max(total_active, 1) * 100, 2)
    all_f1 = []

    tonal_m, tonal_f1 = _compute_event_level_metrics(tonal_gt, tonal_pred, TONAL_NAMES)
    artic_m, artic_f1 = _compute_event_level_metrics(artic_gt, artic_pred, ARTIC_NAMES)
    legato_m, legato_f1 = _compute_event_level_metrics(legato_gt, legato_pred, LEGATO_NAMES)
    all_f1 = tonal_f1 + artic_f1 + legato_f1

    return {
        'total_active_frames': int(total_active),
        'transcription_correct_frames': int(total_trans_correct),
        'transcription_correct_pct': pct,
        'tonal': tonal_m,
        'articulation': artic_m,
        'legato': legato_m,
        'all_technique_macro_f1': round(float(np.mean(all_f1)), 4) if all_f1 else 0.0,
    }


def _build_event_level_binary_results_dict(result, total_active, total_trans_correct):
    pct = round(total_trans_correct / max(total_active, 1) * 100, 2)
    binary_cls = {0: 'negative', 1: 'positive'}
    out = {
        'total_active_frames': int(total_active),
        'transcription_correct_frames': int(total_trans_correct),
        'transcription_correct_pct': pct,
    }
    all_f1 = []
    for name in BINARY_FLAG_NAMES:
        if name not in result:
            continue
        m, f1_list = _compute_event_level_metrics(
            result[name]['gt'], result[name]['pred'], binary_cls)
        out[name] = m
        all_f1 += f1_list
    out['all_technique_macro_f1'] = round(float(np.mean(all_f1)), 4) if all_f1 else 0.0
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Frame-level Multi-Scale MoE technique (+ optional RWC)')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--hdf5s_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=500)
    parser.add_argument('--rwc_h5_path', type=str, default=None)
    parser.add_argument('--rwc_split', type=str, default='all',
                        choices=['train', 'test', 'all'])
    parser.add_argument('--rwc_fold', type=int, default=0)
    parser.add_argument('--fmoe_spectral_expert', type=int, default=1, choices=[0, 1],
                        help='Must match training config: 1=spectral expert, 0=no spectral')
    parser.add_argument('--fmoe_expert_mask', type=str, default=None,
                        help='Comma-separated expert indices (must match training config)')
    parser.add_argument('--fmoe_top_k', type=int, default=None,
                        help='Top-k override (must match training config)')
    parser.add_argument('--fmoe_shared_gate', type=int, default=0, choices=[0, 1],
                        help='Shared gate (must match training config)')
    parser.add_argument('--fmoe_uniform_routing', type=int, default=0, choices=[0, 1],
                        help='Uniform routing (must match training config)')
    parser.add_argument('--fmoe_temporal_experts', type=str, default=None,
                        help='Temporal experts as "k1:d1,k2:d2,..." (must match training)')
    parser.add_argument('--fmoe_no_cues', type=int, default=0, choices=[0, 1],
                        help='Disable cues (must match training config)')
    parser.add_argument('--fmoe_direct_head', type=int, default=0, choices=[0, 1],
                        help='Use direct MLP head instead of MoE (must match training)')
    parser.add_argument('--fmoe_hierarchical', type=int, default=0, choices=[0, 1],
                        help='Use soft hierarchical head (must match training)')
    parser.add_argument('--fmoe_hierarchical_multilevel', type=int, default=0, choices=[0, 1],
                        help='Hierarchical + multi-level features (must match training)')
    parser.add_argument('--fmoe_hierarchical_moe', type=int, default=0, choices=[0, 1],
                        help='Hierarchical L1 + MoE L2 (must match training)')
    parser.add_argument('--fmoe_multi_crnn', type=int, default=0, choices=[0, 1],
                        help='Use multi-CRNN MoE (must match training)')
    parser.add_argument('--fmoe_crnn_moe', type=int, default=0, choices=[0, 1],
                        help='Use CRNN-as-Expert MoE (must match training)')
    parser.add_argument('--fmoe_shared_crnn_moe', type=int, default=0, choices=[0, 1],
                        help='Use Shared-CNN MoE (must match training)')
    parser.add_argument('--fmoe_binary_technique', type=int, default=0, choices=[0, 1],
                        help='Binary technique decomposition (must match training)')
    parser.add_argument('--fmoe_mert_backbone', type=int, default=0, choices=[0, 1],
                        help='Use frozen MERT-v1-330M backbone (must match training)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save evaluation results as JSON')
    parser.add_argument('--technique_label_config', type=str, default=None,
                        help='Path to technique_label_config JSON (must match training)')
    parser.add_argument('--event_level', action='store_true', default=False,
                        help='Event-level mode: frame correct only when both '
                             'transcription (note+pitch) and technique are correct')
    args = parser.parse_args()

    if args.technique_label_config:
        _reload_label_config(args.technique_label_config)

    evaluate(args)
