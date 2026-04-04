import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import losses, reducers, miners
from einops import rearrange


def focal_cross_entropy(logits, targets, gamma=2.0, weight=None, reduction='mean'):
    """Focal loss for multi-class classification (Lin et al., 2017).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Args:
        logits:  (N, C) raw logits
        targets: (N,)   int class indices
        gamma:   focusing parameter (0 = standard CE)
        weight:  optional (C,) per-class weight tensor
        reduction: 'mean' or 'sum'
    """
    log_p = F.log_softmax(logits, dim=-1)                   # (N, C)
    p = log_p.exp()                                          # (N, C)
    ce = F.nll_loss(log_p, targets, weight=weight, reduction='none')  # (N,)
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)       # (N,)
    focal_weight = (1.0 - p_t) ** gamma
    loss = focal_weight * ce
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()


def technique_frame_ce_loss(output_dict, target_dict, device=None, use_active_mask=True):
    """Legacy single-head technique loss (kept for backward compatibility)."""
    if ('technique_output' not in output_dict) or ('technique' not in target_dict):
        return torch.tensor(0.0, device=device if device is not None else None)

    probs = output_dict['technique_output']
    eps = 1e-6
    probs = probs.clamp(min=eps, max=1.0 - eps)
    logits = torch.log(probs) - torch.log(1.0 - probs)

    technique_labels = target_dict['technique']

    if logits.shape[1] != technique_labels.shape[1]:
        min_T = min(logits.shape[1], technique_labels.shape[1])
        logits = logits[:, :min_T, :]
        technique_labels = technique_labels[:, :min_T, :]

    target_indices = torch.argmax(technique_labels, dim=-1)
    logits_2d = logits.reshape(-1, logits.shape[-1])
    targets_1d = target_indices.reshape(-1)

    if use_active_mask and ('frame_roll' in target_dict):
        frame_activity = (target_dict['frame_roll'].sum(dim=-1) > 0)
        if frame_activity.shape[1] != logits.shape[1]:
            min_Tm = min(frame_activity.shape[1], logits.shape[1])
            frame_activity = frame_activity[:, :min_Tm]
            logits_2d = logits[:, :min_Tm, :].reshape(-1, logits.shape[-1])
            targets_1d = target_indices[:, :min_Tm].reshape(-1)
        active_mask = frame_activity.reshape(-1)
        if active_mask.any():
            logits_2d = logits_2d[active_mask]
            targets_1d = targets_1d[active_mask]
        else:
            return torch.tensor(0.0, device=logits.device if hasattr(logits, 'device') else device)
    ce = nn.CrossEntropyLoss(reduction='mean')
    loss = ce(logits_2d, targets_1d)
    return loss


def _get_active_mask(target_dict, T_pred):
    """Build a boolean active-frame mask from frame_roll, aligned to T_pred."""
    if 'frame_roll' not in target_dict:
        return None
    frame_activity = (target_dict['frame_roll'].sum(dim=-1) > 0)  # (B, T_fr)
    if frame_activity.shape[1] != T_pred:
        min_T = min(frame_activity.shape[1], T_pred)
        frame_activity = frame_activity[:, :min_T]
    return frame_activity


def tonal_technique_loss(output_dict, target_dict, device=None, use_active_mask=True):
    """Frame-level CE loss for tonal technique (4-class).

    output_dict['tonal_technique_output']: (B, T, 4) raw logits
    target_dict['tonal_technique']:        (B, T)    int labels in {0,1,2,3}
    """
    key_out, key_tgt = 'tonal_technique_output', 'tonal_technique'
    if (key_out not in output_dict) or (key_tgt not in target_dict):
        return torch.tensor(0.0, device=device)

    logits = output_dict[key_out]        # (B, T_pred, 4)
    targets = target_dict[key_tgt].long()  # (B, T_tgt)

    T_pred = logits.shape[1]
    T_tgt = targets.shape[1]
    if T_pred != T_tgt:
        min_T = min(T_pred, T_tgt)
        logits = logits[:, :min_T, :]
        targets = targets[:, :min_T]

    logits_2d = logits.reshape(-1, logits.shape[-1])
    targets_1d = targets.reshape(-1)

    if use_active_mask:
        mask = _get_active_mask(target_dict, logits.shape[1])
        if mask is not None:
            mask_1d = mask.reshape(-1)
            if mask_1d.any():
                logits_2d = logits_2d[mask_1d]
                targets_1d = targets_1d[mask_1d]
            else:
                return torch.tensor(0.0, device=logits.device)

    return nn.CrossEntropyLoss()(logits_2d, targets_1d)


def articulation_loss(output_dict, target_dict, device=None, use_active_mask=True):
    """Frame-level CE loss for articulation (4-class).

    output_dict['articulation_output']: (B, T, 4) raw logits
    target_dict['articulation']:        (B, T)    int labels in {0,1,2,3}
    """
    key_out, key_tgt = 'articulation_output', 'articulation'
    if (key_out not in output_dict) or (key_tgt not in target_dict):
        return torch.tensor(0.0, device=device)

    logits = output_dict[key_out]
    targets = target_dict[key_tgt].long()

    T_pred = logits.shape[1]
    T_tgt = targets.shape[1]
    if T_pred != T_tgt:
        min_T = min(T_pred, T_tgt)
        logits = logits[:, :min_T, :]
        targets = targets[:, :min_T]

    logits_2d = logits.reshape(-1, logits.shape[-1])
    targets_1d = targets.reshape(-1)

    if use_active_mask:
        mask = _get_active_mask(target_dict, logits.shape[1])
        if mask is not None:
            mask_1d = mask.reshape(-1)
            if mask_1d.any():
                logits_2d = logits_2d[mask_1d]
                targets_1d = targets_1d[mask_1d]
            else:
                return torch.tensor(0.0, device=logits.device)

    return nn.CrossEntropyLoss()(logits_2d, targets_1d)


def legato_loss(output_dict, target_dict, device=None, use_active_mask=True):
    """Positive-weighted BCE on soft bow_change regression target.

    output_dict['legato_output']: (B, T, 1) sigmoid probabilities (1=bow_change)
    target_dict['legato']:        (B, T)    float [0,1] soft regression target
    """
    key_out, key_tgt = 'legato_output', 'legato'
    if (key_out not in output_dict) or (key_tgt not in target_dict):
        return torch.tensor(0.0, device=device)

    pred = output_dict[key_out].squeeze(-1)  # (B, T_pred)
    targets = target_dict[key_tgt].float()   # (B, T_tgt)

    T_pred = pred.shape[1]
    T_tgt = targets.shape[1]
    if T_pred != T_tgt:
        min_T = min(T_pred, T_tgt)
        pred = pred[:, :min_T]
        targets = targets[:, :min_T]

    pred_1d = pred.reshape(-1)
    targets_1d = targets.reshape(-1)

    if use_active_mask:
        mask = _get_active_mask(target_dict, pred.shape[1])
        if mask is not None:
            mask_1d = mask.reshape(-1)
            if mask_1d.any():
                pred_1d = pred_1d[mask_1d]
                targets_1d = targets_1d[mask_1d]
            else:
                return torch.tensor(0.0, device=pred.device)

    bce = F.binary_cross_entropy(pred_1d, targets_1d, reduction='none')
    w = torch.where(targets_1d > 0.1, 5.0, 1.0)
    return (bce * w).mean()


def viotech_technique_losses(output_dict, target_dict, device=None, use_active_mask=True):
    """Compute all three technique losses and return them separately.

    Returns:
      (loss_tonal, loss_artic, loss_legato) — each a scalar tensor.
    """
    loss_tonal = tonal_technique_loss(output_dict, target_dict, device, use_active_mask)
    loss_artic = articulation_loss(output_dict, target_dict, device, use_active_mask)
    loss_leg = legato_loss(output_dict, target_dict, device, use_active_mask)
    return loss_tonal, loss_artic, loss_leg

def moe_technique_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
    """Note-level MoE technique losses (tonal + artic + legato + balance).

    Delegates to moe_technique.moe_note_technique_losses.
    Returns (loss_tonal, loss_artic, loss_legato, loss_balance).
    """
    from moe_technique import moe_note_technique_losses
    return moe_note_technique_losses(output_dict, target_dict, device=device,
                                     focal_gamma=focal_gamma)


def zone_moe_technique_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
    """Note-level Zone-Specialized MoE technique losses.

    Delegates to moe_zone_specialist.zone_moe_note_technique_losses.
    Returns (loss_tonal, loss_artic, loss_legato, loss_balance).
    """
    from moe_zone_specialist import zone_moe_note_technique_losses
    return zone_moe_note_technique_losses(output_dict, target_dict, device=device,
                                          focal_gamma=focal_gamma)


def pertask_zone_moe_technique_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
    """Note-level Per-Task Gate Zone MoE technique losses.

    Delegates to moe_zone_pertask.pertask_zone_moe_losses.
    Returns (loss_tonal, loss_artic, loss_legato, loss_balance).
    """
    from moe_zone_pertask import pertask_zone_moe_losses
    return pertask_zone_moe_losses(output_dict, target_dict, device=device,
                                   focal_gamma=focal_gamma)


def frame_multiscale_moe_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
    """Frame-level multi-scale MoE technique losses.

    Delegates to moe_frame_multiscale.frame_moe_technique_losses.
    Returns (loss_tonal, loss_artic, loss_legato, loss_balance, loss_aux).
    """
    from moe_frame_multiscale import frame_moe_technique_losses
    return frame_moe_technique_losses(output_dict, target_dict, device=device,
                                      focal_gamma=focal_gamma)


def technique_aux_frame_ce_loss(aux_technique_model, output_dict, target_dict, device=None):
    """Auxiliary frame-level technique classification loss.

    Args:
      aux_technique_model: nn.Module mapping (B, T, 512) -> (B, T, 4)
      output_dict: must contain 'frame_features' tensor of shape (B, T, 512)
      target_dict: must contain 'technique' one-hot tensor (B, T, C), where C=5 incl. 'no_technique'
      device: optional torch.device for creating zero tensors

    Returns:
      Scalar tensor loss (CrossEntropy over all frames using provided one-hot labels).
    """
    if ('frame_features' not in output_dict) or ('technique' not in target_dict):
        return torch.tensor(0.0, device=device if device is not None else None)

    technique_logits = aux_technique_model(output_dict['frame_features'])  # (B, T, C)
    technique_labels = target_dict['technique']  # (B, T, C) one-hot including 'no_technique'

    target_indices = torch.argmax(technique_labels, dim=-1)  # (B, T)
    logits_2d = technique_logits.reshape(-1, technique_logits.shape[-1])  # (B*T, C)
    targets_1d = target_indices.reshape(-1)  # (B*T,)
    ce = nn.CrossEntropyLoss(reduction='mean')
    loss = ce(logits_2d, targets_1d)

    return loss

def supcon_loss(output_dict, target_dict):
    """Supervised contrastive loss.
    """
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_onset_features']   # (B, T, D)
    labels = target_dict['reg_onset_roll']         # (B, T, 88)

    features = rearrange(features, 'b t d -> (b t) d')      # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')          # (B*T, 88)

    features = F.normalize(features, dim=1)

    total_loss = 0.0
    count = 0

    for pitch_class in range(labels.shape[1]):
        pitch_label = labels[:, pitch_class]   # (B*T,) binary
        mask = pitch_label > 0

        if mask.sum() < 2:
            print(pitch_class, 'pitch_class has less than 2 positive')
            continue  # 至少要有 2 個 positive

        selected_features = features[mask]  # shape (N, D)
        selected_labels = torch.zeros_like(pitch_label[mask], dtype=torch.long)

        # 所有 positive 給同一類標籤（class 0）
        loss = loss_func(selected_features, selected_labels)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)
    print(total_loss / count, 'total loss')
    return total_loss / count

def all_pitch_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_onset_features']  # (B, T, D)
    labels = target_dict['reg_onset_roll']        # (B, T, 88)

    features = rearrange(features, 'b t d -> (b t) d')   # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')       # (B*T, 88)

    features = F.normalize(features, dim=1)

    # 準備 feature 與 class label
    selected_features = []
    selected_labels = []

    for pitch in range(labels.shape[1]):
        mask = labels[:, pitch] > 0
        if mask.sum() < 2:
            continue
        selected_features.append(features[mask])
        selected_labels.append(torch.full((mask.sum(),), pitch, dtype=torch.long, device=features.device))

    if len(selected_features) == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    all_feats = torch.cat(selected_features, dim=0)
    all_labs = torch.cat(selected_labels, dim=0)

    loss = loss_func(all_feats, all_labs)
    print(loss, 'all pitch supcon loss')
    return loss

def onset_binary_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_onset_features']  # (B, T, D)
    labels = target_dict['reg_onset_roll']        # (B, T, 88)

    features = rearrange(features, 'b t d -> (b t) d')  # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')      # (B*T, 88)

    # 合併為 binary onset label：任一 pitch 為 1 即視為 onset
    binary_labels = (labels.sum(dim=1) > 0).long()  # shape: (B*T,)

    if binary_labels.sum() < 2 or (binary_labels == 0).sum() < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # 計算 positive 與 negative 的數量, 並隨機dropout掉數量較多的那一類（通常是negative）的labels與features，確保 positive 與 negative 的數量平衡
    positive_mask = binary_labels == 1
    negative_mask = binary_labels == 0
    
    num_positive = positive_mask.sum()
    num_negative = negative_mask.sum()
    
    # 確保平衡：取較小數量作為目標
    target_count = min(num_positive, num_negative)
    
    # 隨機選取 positive samples
    positive_indices = torch.where(positive_mask)[0]
    if len(positive_indices) > target_count:
        selected_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:target_count]]
    else:
        selected_positive_indices = positive_indices
    
    # 隨機選取 negative samples
    negative_indices = torch.where(negative_mask)[0]
    if len(negative_indices) > target_count:
        selected_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:target_count]]
    else:
        selected_negative_indices = negative_indices
    
    # 合併選取的 indices
    selected_indices = torch.cat([selected_positive_indices, selected_negative_indices])
    
    # 選取平衡後的 features 和 labels
    balanced_features = features[selected_indices]
    balanced_labels = binary_labels[selected_indices]

    balanced_features = F.normalize(balanced_features, dim=1)
    loss = loss_func(balanced_features, balanced_labels)
    print(loss, 'onset binary supcon loss')
    return loss

def offset_binary_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    features = output_dict['reg_offset_features']  # (B, T, D)
    labels = target_dict['reg_offset_roll']        # (B, T, 88)
    
    features = rearrange(features, 'b t d -> (b t) d')  # (B*T, D)
    labels = rearrange(labels, 'b t p -> (b t) p')      # (B*T, 88)
    
    # 合併為 binary offset label：任一 pitch 為 1 即視為 offset
    binary_labels = (labels.sum(dim=1) > 0).long()  # shape: (B*T,)

    if binary_labels.sum() < 2 or (binary_labels == 0).sum() < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    positive_mask = binary_labels == 1
    negative_mask = binary_labels == 0
    
    num_positive = positive_mask.sum()
    num_negative = negative_mask.sum()
    
    target_count = min(num_positive, num_negative)

    positive_indices = torch.where(positive_mask)[0]
    if len(positive_indices) > target_count:
        selected_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:target_count]]
    else:
        selected_positive_indices = positive_indices

    negative_indices = torch.where(negative_mask)[0]
    if len(negative_indices) > target_count:
        selected_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:target_count]]
    else:
        selected_negative_indices = negative_indices

    selected_indices = torch.cat([selected_positive_indices, selected_negative_indices])

    balanced_features = features[selected_indices]
    balanced_labels = binary_labels[selected_indices]

    balanced_features = F.normalize(balanced_features, dim=1)
    loss = loss_func(balanced_features, balanced_labels)
    print(loss, 'offset binary supcon loss')
    return loss

def onset_offset_binary_supcon_loss(output_dict, target_dict):
    loss_func = losses.SupConLoss(temperature=0.07)
    onset_features = output_dict['reg_onset_features']
    offset_features = output_dict['reg_offset_features']
    onset_labels = target_dict['reg_onset_roll']
    offset_labels = target_dict['reg_offset_roll']

    onset_features = rearrange(onset_features, 'b t d -> (b t) d')
    offset_features = rearrange(offset_features, 'b t d -> (b t) d')

    onset_labels = rearrange(onset_labels, 'b t p -> (b t) p')
    offset_labels = rearrange(offset_labels, 'b t p -> (b t) p')

    onset_features = F.normalize(onset_features, dim=1)
    offset_features = F.normalize(offset_features, dim=1)
    # 
def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

############ CTC losses ############
def per_pitch_onoff_ctc_loss(model, output_dict, target_dict):
    """Per-pitch CTC using existing onset/offset predictions.

    Each pitch is modeled as a 3-class CTC stream: {blank(0), onset(1), offset(2)}.

    Requires in output_dict:
      - 'reg_onset_output': (B, T, 88)
      - 'reg_offset_output': (B, T, 88)

    Requires in target_dict:
      - 'onset_roll': (B, T, 88)
      - 'offset_roll': (B, T, 88)
    """
    if ('reg_onset_output' not in output_dict) or ('reg_offset_output' not in output_dict):
        raise KeyError('per_pitch_onoff_ctc_loss requires reg_onset_output and reg_offset_output in output_dict')
    if ('onset_roll' not in target_dict) or ('offset_roll' not in target_dict):
        raise KeyError('per_pitch_onoff_ctc_loss requires onset_roll and offset_roll in target_dict')

    onset_prob = output_dict['reg_onset_output']  # (B, T, 88)
    offset_prob = output_dict['reg_offset_output']  # (B, T, 88)

    B, T, P = onset_prob.shape
    device = onset_prob.device

    eps = 1e-6
    onset_p = torch.clamp(onset_prob, eps, 1.0 - eps)
    offset_p = torch.clamp(offset_prob, eps, 1.0 - eps)
    blank_p = torch.clamp((1.0 - onset_p) * (1.0 - offset_p), eps, 1.0 - eps)

    # (B, T, P, 3) -> normalized log-probs
    log_probs = torch.stack((blank_p.log(), onset_p.log(), offset_p.log()), dim=-1)
    log_probs = torch.log_softmax(log_probs, dim=-1)
    # (T, B*P, 3)
    log_probs = log_probs.permute(1, 0, 2, 3).reshape(T, B * P, 3)

    onset_gt = target_dict['onset_roll']  # (B, T, 88)
    offset_gt = target_dict['offset_roll']  # (B, T, 88)

    target_tensors = []
    target_lengths = []
    selected_columns = []

    col_index = 0
    for b in range(B):
        for p in range(P):
            seq = []
            on_seq = onset_gt[b, :, p]
            off_seq = offset_gt[b, :, p]
            for t in range(T):
                if on_seq[t] > 0.5:
                    seq.append(1)
                if off_seq[t] > 0.5:
                    seq.append(2)
            if len(seq) > 0:
                target_tensors.append(torch.tensor(seq, dtype=torch.long, device=device))
                target_lengths.append(len(seq))
                selected_columns.append(col_index)
            col_index += 1

    if len(selected_columns) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    log_probs = log_probs[:, selected_columns, :]
    input_lengths = torch.full((len(selected_columns),), T, dtype=torch.long, device=device)
    targets = torch.cat(target_tensors, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)

    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    return loss

############ High-resolution regression loss ############
def regress_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def regress_pedal_bce(model, output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['reg_pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['reg_pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss

############ Google's onsets and frames system loss ############
def google_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano note loss. Only used for comparison.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def google_pedal_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano pedal loss. Only used for comparison.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'none':
        return lambda model, output_dict, target_dict: torch.tensor(0.0, device=next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else None)
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce

    elif loss_type == 'regress_pedal_bce':
        return regress_pedal_bce

    elif loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    elif loss_type == 'google_pedal_bce':
        return google_pedal_bce

    elif loss_type == 'per_pitch_onoff_ctc':
        return per_pitch_onoff_ctc_loss

    else:
        raise Exception('Incorrect loss_type!')