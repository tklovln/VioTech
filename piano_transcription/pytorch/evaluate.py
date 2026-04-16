import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
import h5py
import time
import mir_eval
import librosa
import logging
from sklearn import metrics

from pytorch_utils import forward_dataloader, move_data_to_device
from losses import tonal_technique_loss, articulation_loss, legato_loss
from moe_frame_multiscale import frame_moe_technique_losses
from technique_label_utils import get_technique_labels

def mae(target, output, mask):
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)


class SegmentEvaluator(object):
    def __init__(self, model, batch_size):
        """Evaluate segment-wise metrics.

        Args:
          model: object
          batch_size: int
        """
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, dataloader):
        """Evaluate over a few mini-batches.

        Args:
          dataloader: object, used to generate mini-batches for evaluation.

        Returns:
          statistics: dict, e.g. {
            'frame_f1': 0.800, 
            (if exist) 'onset_f1': 0.500, 
            (if exist) 'offset_f1': 0.300, 
            ...}
        """

        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size)
        
        # Helper to align (N, T, C) along T
        def _align_time(a, b):
            T = min(a.shape[1], b.shape[1])
            return a[:, :T, ...], b[:, :T, ...]

        # Frame and onset evaluation (only if both prediction and target exist)
        if ('frame_output' in output_dict.keys()) and ('frame_roll' in output_dict.keys()):
            y_true, y_score = _align_time(output_dict['frame_roll'], output_dict['frame_output'])
            statistics['frame_ap'] = metrics.average_precision_score(
                y_true.flatten(), y_score.flatten(), average='macro')
        
        if ('onset_output' in output_dict.keys()) and ('onset_roll' in output_dict.keys()):
            y_true, y_score = _align_time(output_dict['onset_roll'], output_dict['onset_output'])
            statistics['onset_macro_ap'] = metrics.average_precision_score(
                y_true.flatten(), y_score.flatten(), average='macro')

        if ('offset_output' in output_dict.keys()) and ('offset_roll' in output_dict.keys()):
            y_true, y_score = _align_time(output_dict['offset_roll'], output_dict['offset_output'])
            statistics['offset_ap'] = metrics.average_precision_score(
                y_true.flatten(), y_score.flatten(), average='macro')

        if ('reg_onset_output' in output_dict.keys()) and ('reg_onset_roll' in output_dict.keys()):
            y_pred, y_true = _align_time(output_dict['reg_onset_output'], output_dict['reg_onset_roll'])
            mask = (np.sign(y_pred + y_true - 0.01) + 1) / 2
            statistics['reg_onset_mae'] = mae(y_pred, y_true, mask)

        if ('reg_offset_output' in output_dict.keys()) and ('reg_offset_roll' in output_dict.keys()):
            y_pred, y_true = _align_time(output_dict['reg_offset_output'], output_dict['reg_offset_roll'])
            mask = (np.sign(y_pred + y_true - 0.01) + 1) / 2
            statistics['reg_offset_mae'] = mae(y_pred, y_true, mask)

        if ('velocity_output' in output_dict.keys()) and ('velocity_roll' in output_dict.keys()) and ('onset_roll' in output_dict.keys()):
            # Align all to min time
            v_pred, v_true = _align_time(output_dict['velocity_output'], output_dict['velocity_roll'])
            _, on_mask = _align_time(output_dict['velocity_output'], output_dict['onset_roll'])
            statistics['velocity_mae'] = mae(v_pred, v_true / 128, on_mask)

        if 'reg_pedal_onset_output' in output_dict.keys():
            statistics['reg_pedal_onset_mae'] = mae(
                output_dict['reg_pedal_onset_roll'].flatten(), 
                output_dict['reg_pedal_onset_output'].flatten(), 
                mask=None)

        if 'reg_pedal_offset_output' in output_dict.keys():
            statistics['reg_pedal_offset_mae'] = mae(
                output_dict['reg_pedal_offset_output'].flatten(), 
                output_dict['reg_pedal_offset_roll'].flatten(), 
                mask=None)

        if 'pedal_frame_output' in output_dict.keys():
            statistics['pedal_frame_mae'] = mae(
                output_dict['pedal_frame_output'].flatten(), 
                output_dict['pedal_frame_roll'].flatten(), 
                mask=None)

        # Technique losses (viotech 3-head)
        _tech_keys_present = (
            'tonal_technique_output' in output_dict
            and 'tonal_technique' in output_dict
        )
        if _tech_keys_present:
            _to = lambda a: torch.from_numpy(a)
            _od = {k: _to(output_dict[k]) for k in output_dict if k.endswith('_output')}
            _td = {k: _to(output_dict[k]) for k in ('tonal_technique', 'articulation',
                    'legato', 'frame_roll') if k in output_dict}
            statistics['loss_tonal_technique'] = tonal_technique_loss(_od, _td).item()
            statistics['loss_articulation'] = articulation_loss(_od, _td).item()
            statistics['loss_legato'] = legato_loss(_od, _td).item()

        # Frame-level Multi-Scale MoE technique losses + per-class accuracy
        # Binary mode: fmoe_binary_probs is a dict of tensors,
        # skip the multi-class path and compute binary metrics directly.
        _binary_flag_names = ['pizzicato', 'harmonic', 'spiccato', 'legato']
        _fmoe_binary_present = any(
            f'fmoe_{fn}_prob' in output_dict for fn in _binary_flag_names[:3]
        )
        if _fmoe_binary_present:
            _to = lambda a: torch.from_numpy(a) if isinstance(a, np.ndarray) else a
            probs_dict = {}
            for fn in _binary_flag_names:
                k = f'fmoe_{fn}_prob'
                if k in output_dict:
                    probs_dict[fn] = output_dict[k]
            active_mask = None
            if 'frame_roll' in output_dict:
                fr = output_dict['frame_roll']
                if isinstance(fr, np.ndarray):
                    active_mask = fr.sum(axis=-1) > 0
                else:
                    active_mask = fr.sum(dim=-1) > 0

            total_loss = 0.0
            for flag_name, prob in probs_dict.items():
                if isinstance(prob, torch.Tensor):
                    prob_np = prob.cpu().numpy()
                else:
                    prob_np = prob
                pred = (prob_np.squeeze(-1) > 0.5).astype(np.int64)
                gt = output_dict.get(flag_name)
                if gt is None:
                    continue
                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                gt_bin = (gt > 0.5).astype(np.int64)
                T = min(pred.shape[1], gt_bin.shape[1])
                pred = pred[:, :T]
                gt_bin = gt_bin[:, :T]
                if active_mask is not None:
                    m = active_mask[:, :T]
                    if isinstance(m, torch.Tensor):
                        m = m.numpy()
                    pred_flat = pred[m]
                    gt_flat = gt_bin[m]
                else:
                    pred_flat = pred.flatten()
                    gt_flat = gt_bin.flatten()
                if len(gt_flat) > 0:
                    acc = (pred_flat == gt_flat).mean()
                    statistics[f'fmoe_{flag_name}_acc'] = float(acc)
                    pos_mask = gt_flat == 1
                    n_pos = pos_mask.sum()
                    if n_pos > 0:
                        statistics[f'fmoe_{flag_name}_recall'] = float(
                            (pred_flat[pos_mask] == 1).mean())
                    neg_mask = gt_flat == 0
                    n_neg = neg_mask.sum()
                    if n_neg > 0:
                        statistics[f'fmoe_{flag_name}_spec'] = float(
                            (pred_flat[neg_mask] == 0).mean())

                    p_t = _to(gt[:, :T].astype(np.float32))
                    p_p = _to(prob_np.squeeze(-1)[:, :T])
                    bce = torch.nn.functional.binary_cross_entropy(
                        p_p.float(), p_t.float(), reduction='mean')
                    statistics[f'fmoe_loss_{flag_name}'] = bce.item()
                    total_loss += bce.item()
            statistics['fmoe_loss_technique'] = total_loss

        _fmoe_present = (
            'fmoe_tonal_logits' in output_dict
            and output_dict.get('fmoe_tonal_logits') is not None
            and 'tonal_technique' in output_dict
        )
        if _fmoe_present:
            _to = lambda a: torch.from_numpy(a)
            _od = {k: _to(output_dict[k]) for k in output_dict
                   if isinstance(output_dict[k], np.ndarray)}
            _td = {}
            for k in ('tonal_technique', 'articulation', 'legato', 'frame_roll'):
                if k in output_dict:
                    _td[k] = _to(output_dict[k])

            fmoe_t, fmoe_a, fmoe_l, fmoe_bal, fmoe_aux = frame_moe_technique_losses(
                _od, _td, device=None)
            statistics['fmoe_loss_tonal'] = fmoe_t.item()
            statistics['fmoe_loss_artic'] = fmoe_a.item()
            statistics['fmoe_loss_legato'] = fmoe_l.item()
            statistics['fmoe_loss_balance'] = fmoe_bal.item()
            statistics['fmoe_loss_aux'] = fmoe_aux.item()
            statistics['fmoe_loss_technique'] = (fmoe_t + fmoe_a + fmoe_l).item()

            # Per-class accuracy for tonal technique & articulation
            active_mask = None
            if 'frame_roll' in output_dict:
                active_mask = output_dict['frame_roll'].sum(axis=-1) > 0  # (N, T)

            _lcfg = get_technique_labels()
            TONAL_NAMES = _lcfg.tonal_names
            ARTIC_NAMES = _lcfg.artic_names

            for logits_key, target_key, class_names, prefix in [
                ('fmoe_tonal_logits', 'tonal_technique', TONAL_NAMES, 'fmoe_tonal'),
                ('fmoe_artic_logits', 'articulation', ARTIC_NAMES, 'fmoe_artic'),
            ]:
                if logits_key not in output_dict or target_key not in output_dict:
                    continue
                logits = output_dict[logits_key]
                targets = output_dict[target_key].astype(np.int64)
                T = min(logits.shape[1], targets.shape[1])
                logits = logits[:, :T, :]
                targets = targets[:, :T]
                preds = logits.argmax(axis=-1)

                if active_mask is not None:
                    m = active_mask[:, :T]
                    preds_flat = preds[m]
                    tgts_flat = targets[m]
                else:
                    preds_flat = preds.flatten()
                    tgts_flat = targets.flatten()

                if len(tgts_flat) > 0:
                    statistics[f'{prefix}_acc'] = (preds_flat == tgts_flat).mean()
                    for cls_id, cls_name in class_names.items():
                        cls_mask = tgts_flat == cls_id
                        n_cls = cls_mask.sum()
                        if n_cls > 0:
                            statistics[f'{prefix}_acc_{cls_name}'] = (
                                preds_flat[cls_mask] == cls_id).mean()
                            statistics[f'{prefix}_n_{cls_name}'] = int(n_cls)

            # Legato accuracy
            if 'fmoe_legato_prob' in output_dict and 'legato' in output_dict:
                pred_l = (output_dict['fmoe_legato_prob'].squeeze(-1) > 0.5).astype(np.int64)
                tgt_l = (output_dict['legato'] > 0.5).astype(np.int64)
                T = min(pred_l.shape[1], tgt_l.shape[1])
                pred_l = pred_l[:, :T]
                tgt_l = tgt_l[:, :T]
                if active_mask is not None:
                    m = active_mask[:, :T]
                    pred_l = pred_l[m]
                    tgt_l = tgt_l[m]
                else:
                    pred_l = pred_l.flatten()
                    tgt_l = tgt_l.flatten()
                if len(tgt_l) > 0:
                    statistics['fmoe_legato_acc'] = (pred_l == tgt_l).mean()

        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)

        return statistics


def evaluate_rwc_frame_moe(model, dataloader, device):
    """Frame-level MoE technique evaluation on RWC during training.

    Reports macro F1 and per-technique accuracy for each head.
    Supports both multi-class (tonal/artic/legato) and binary
    (pizzicato/harmonic/spiccato/legato) modes.
    """
    from sklearn.metrics import f1_score, accuracy_score

    _lcfg = get_technique_labels()
    TONAL_NAMES = _lcfg.tonal_names
    ARTIC_NAMES = _lcfg.artic_names
    LEGATO_NAMES = {0: 'sustained', 1: 'bow_change'}
    BINARY_CLS = {0: 'negative', 1: 'positive'}

    _binary_mode = False
    all_tonal_pred, all_tonal_gt = [], []
    all_artic_pred, all_artic_gt = [], []
    all_legato_pred, all_legato_gt = [], []
    binary_flag_names = ['pizzicato', 'harmonic', 'spiccato', 'legato']
    all_binary_gt = {n: [] for n in binary_flag_names}
    all_binary_pred = {n: [] for n in binary_flag_names}
    total_frames = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_data_dict in dataloader:
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            output_dict = model(batch_data_dict['waveform'])

            is_binary = 'fmoe_binary_probs' in output_dict
            if is_binary:
                _binary_mode = True
            elif 'fmoe_tonal_logits' not in output_dict:
                continue

            if 'frame_roll' in batch_data_dict:
                active = (batch_data_dict['frame_roll'].sum(dim=-1) > 0)
            else:
                if is_binary:
                    first_p = next(iter(output_dict['fmoe_binary_probs'].values()))
                    B, T_p = first_p.shape[:2]
                else:
                    B, T_p = output_dict['fmoe_tonal_logits'].shape[:2]
                active = torch.ones(B, T_p, dtype=torch.bool, device=device)

            if is_binary:
                probs_dict = output_dict['fmoe_binary_probs']
                first_p = next(iter(probs_dict.values()))
                T = min(first_p.shape[1], active.shape[1])
                active = active[:, :T]
                B = active.shape[0]
                for b in range(B):
                    m = active[b].cpu().numpy().astype(bool)
                    n_active = int(m.sum())
                    if n_active == 0:
                        continue
                    total_frames += n_active
                    for name in binary_flag_names:
                        prob = probs_dict[name][b, :T].squeeze(-1)
                        pred = (prob > 0.5).long().cpu().numpy()[m]
                        all_binary_pred[name].extend(pred)
                        gt_tensor = batch_data_dict.get(name)
                        if gt_tensor is not None:
                            gt = (gt_tensor[b, :T] > 0.5).long().cpu().numpy()[m]
                            all_binary_gt[name].extend(gt)
            else:
                tonal_logits = output_dict['fmoe_tonal_logits']
                artic_logits = output_dict['fmoe_artic_logits']
                legato_prob = output_dict['fmoe_legato_prob'].squeeze(-1)

                T = min(tonal_logits.shape[1], active.shape[1])

                tonal_pred = tonal_logits[:, :T].argmax(dim=-1)
                artic_pred = artic_logits[:, :T].argmax(dim=-1)
                legato_pred = (legato_prob[:, :T] > 0.5).long()
                active = active[:, :T]

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
                    m = active[b].cpu().numpy().astype(bool)
                    n_active = int(m.sum())
                    if n_active == 0:
                        continue

                    all_tonal_pred.extend(tonal_pred[b].cpu().numpy()[m])
                    all_tonal_gt.extend(tonal_gt[b].cpu().numpy()[m])
                    all_artic_pred.extend(artic_pred[b].cpu().numpy()[m])
                    all_artic_gt.extend(artic_gt[b].cpu().numpy()[m])
                    all_legato_pred.extend(legato_pred[b].cpu().numpy()[m])
                    all_legato_gt.extend(legato_gt[b].cpu().numpy()[m])
                    total_frames += n_active

    if was_training:
        model.train()

    statistics = {'rwc_total_frames': total_frames}
    if total_frames == 0:
        return statistics

    def _compute_head_stats(gt_list, pred_list, class_names, prefix):
        gt_arr = np.array(gt_list, dtype=int)
        pred_arr = np.array(pred_list, dtype=int)
        present = sorted(set(gt_arr) | set(pred_arr))
        stats = {}
        stats[f'{prefix}_macro_f1'] = float(f1_score(
            gt_arr, pred_arr, labels=present, average='macro', zero_division=0))
        stats[f'{prefix}_acc'] = float(accuracy_score(gt_arr, pred_arr))
        for cls_id in present:
            name = class_names.get(cls_id, f'class_{cls_id}')
            mask = gt_arr == cls_id
            if mask.sum() > 0:
                stats[f'{prefix}_acc_{name}'] = float((pred_arr[mask] == cls_id).mean())
        return stats

    if _binary_mode:
        for name in binary_flag_names:
            if all_binary_gt[name]:
                statistics.update(_compute_head_stats(
                    all_binary_gt[name], all_binary_pred[name],
                    BINARY_CLS, f'rwc_{name}'))
    else:
        statistics.update(_compute_head_stats(
            all_tonal_gt, all_tonal_pred, TONAL_NAMES, 'rwc_tonal'))
        statistics.update(_compute_head_stats(
            all_artic_gt, all_artic_pred, ARTIC_NAMES, 'rwc_artic'))
        statistics.update(_compute_head_stats(
            all_legato_gt, all_legato_pred, LEGATO_NAMES, 'rwc_legato'))

    for key in statistics:
        if isinstance(statistics[key], float):
            statistics[key] = round(statistics[key], 4)

    return statistics