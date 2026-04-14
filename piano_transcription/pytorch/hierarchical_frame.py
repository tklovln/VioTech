"""Frame-level Soft Hierarchical technique classification.

Coarse-to-fine: for each dimension (tonal / articulation), first detect
whether a "special" technique is present (Level 1), then classify which
specific technique (Level 2).  Final probabilities are composed as:

    P(none)    = 1 - P(special)
    P(class_k) = P(special) * P(class_k | special)

This decomposition naturally handles extreme class imbalance (none >> special
techniques) by turning the hard multiclass problem into:
  1. A relatively balanced binary detection problem
  2. A fine-grained classification among rare classes only

Legato is already binary (sustained vs bow_change) and uses direct sigmoid.

The output is log-probabilities assembled in class-index order.  Since
sum(probs) == 1 by construction, feeding these to F.cross_entropy yields
the correct NLL loss, and argmax gives the correct predicted class.  This
makes the head a drop-in replacement for FrameMultiScaleMoEHead — existing
loss functions and evaluation pipelines work without modification.

Variants:
  - FrameHierarchicalHead           — GRU features only (original)
  - FrameHierarchicalMultiLevelHead — GRU + CNN local features for detector
  - FrameHierarchicalMoEHead        — hierarchical L1 + MoE L2 classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FrameHierarchicalConfig:
    feat_dim: int = 512
    proj_dim: int = 256
    cue_dim: int = 16
    hidden_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    tonal_none_id: int = 0
    artic_none_id: int = 0
    dropout: float = 0.2
    use_cues: bool = True


class FrameHierarchicalHead(nn.Module):
    """Soft hierarchical frame-level technique classification.

    Forward returns the same 9-tuple as FrameMultiScaleMoEHead for
    drop-in compatibility:
        tonal_logits       (B, T, num_tonal_classes)  — log-probs
        artic_logits       (B, T, num_artic_classes)  — log-probs
        bow_change_prob    (B, T, 1)
        gate_tonal         None
        gate_artic         None
        gate_legato        None
        aux_tonal          []
        aux_artic          []
        aux_legato         []
    """

    def __init__(self, cfg: FrameHierarchicalConfig = None):
        super().__init__()
        cfg = cfg or FrameHierarchicalConfig()
        self.cfg = cfg

        D = cfg.feat_dim
        P = cfg.proj_dim
        C = cfg.cue_dim
        H = cfg.hidden_dim
        self.use_cues = cfg.use_cues

        self.feat_proj = nn.Sequential(
            nn.Linear(D, P, bias=False),
            nn.LayerNorm(P),
            nn.GELU(),
        )

        cue_total = 3 * C if self.use_cues else 0
        if self.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        feat_total = P + cue_total

        K_tonal = cfg.num_tonal_classes - 1
        K_artic = cfg.num_artic_classes - 1

        # Level 1 — binary detection: "is this frame special?"
        self.tonal_detector = nn.Sequential(
            nn.Linear(feat_total, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )
        self.artic_detector = nn.Sequential(
            nn.Linear(feat_total, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )

        # Level 2 — fine-grained: "which special class?"
        # Conditioned on shared features + detector probability
        self.tonal_classifier = nn.Sequential(
            nn.Linear(feat_total + 1, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, K_tonal),
        )
        self.artic_classifier = nn.Sequential(
            nn.Linear(feat_total + 1, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, K_artic),
        )

        # Legato — direct binary (already ~80/20 split, no hierarchy needed)
        self.legato_head = nn.Sequential(
            nn.Linear(feat_total, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        feat = self.feat_proj(features)

        if self.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob))
            c_offset = F.gelu(self.offset_proj(offset_prob))
            c_frame = F.gelu(self.frame_proj(frame_prob))
            feat = torch.cat([feat, c_onset, c_offset, c_frame], dim=-1)

        det_tonal = self.tonal_detector(feat)
        det_artic = self.artic_detector(feat)

        p_tonal = torch.sigmoid(det_tonal)
        p_artic = torch.sigmoid(det_artic)

        cls_tonal = self.tonal_classifier(torch.cat([feat, p_tonal], dim=-1))
        cls_artic = self.artic_classifier(torch.cat([feat, p_artic], dim=-1))

        tonal_logits = _compose_log_probs(
            det_tonal, cls_tonal,
            self.cfg.num_tonal_classes, self.cfg.tonal_none_id)
        artic_logits = _compose_log_probs(
            det_artic, cls_artic,
            self.cfg.num_artic_classes, self.cfg.artic_none_id)

        legato_prob = torch.sigmoid(self.legato_head(feat))

        return (tonal_logits, artic_logits, legato_prob,
                None, None, None,
                [], [], [])


def _compose_log_probs(det_logit, cls_logits, num_classes, none_id):
    """Numerically stable hierarchical log-probability composition.

    log P(none)    = log sigmoid(-d)   = logsigmoid(-d)
    log P(class_k) = log sigmoid(d)    + log softmax(c)[k]
    """
    log_p_none = F.logsigmoid(-det_logit)
    log_p_special = F.logsigmoid(det_logit)
    log_cls = F.log_softmax(cls_logits, dim=-1)
    log_p_fine = log_p_special + log_cls

    parts = []
    j = 0
    for i in range(num_classes):
        if i == none_id:
            parts.append(log_p_none)
        else:
            parts.append(log_p_fine[:, :, j:j + 1])
            j += 1
    return torch.cat(parts, dim=-1)


# ===================================================================
# Variant 1 — Multi-level feature input (GRU + CNN local features)
# ===================================================================

@dataclass
class FrameHierarchicalMultiLevelConfig:
    feat_dim: int = 512          # GRU output dim
    cnn_feat_dim: int = 128      # conv_block4 channel dim
    cnn_freq_bins: int = 14      # freq bins after 4x pool (229 // 2^4)
    cnn_proj_dim: int = 64       # project CNN local features to this dim
    proj_dim: int = 256
    cue_dim: int = 16
    hidden_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    tonal_none_id: int = 0
    artic_none_id: int = 0
    dropout: float = 0.2
    use_cues: bool = True


class FrameHierarchicalMultiLevelHead(nn.Module):
    """Hierarchical head with multi-level features.

    The binary detector (Level 1) receives BOTH:
      - GRU output (512-dim) — contextual / sequential features
      - CNN conv_block4 output (128×14) — local spectral detail

    This gives the detector access to attack transient shape (from CNN)
    alongside note-level context (from GRU), improving spiccato detection.

    Level 2 fine-grained classifier uses GRU features only (context matters
    more for distinguishing among rare classes).

    Forward returns the same 9-tuple as FrameMultiScaleMoEHead.
    """

    def __init__(self, cfg: FrameHierarchicalMultiLevelConfig = None):
        super().__init__()
        cfg = cfg or FrameHierarchicalMultiLevelConfig()
        self.cfg = cfg

        P = cfg.proj_dim
        C = cfg.cue_dim
        H = cfg.hidden_dim
        CP = cfg.cnn_proj_dim
        self.use_cues = cfg.use_cues

        # GRU feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(cfg.feat_dim, P, bias=False),
            nn.LayerNorm(P),
            nn.GELU(),
        )

        # CNN local feature projection: (B, 128, T, 14) → (B, T, cnn_proj_dim)
        self.cnn_proj = nn.Sequential(
            nn.Linear(cfg.cnn_feat_dim, CP, bias=False),
            nn.LayerNorm(CP),
            nn.GELU(),
        )

        cue_total = 3 * C if self.use_cues else 0
        if self.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        # Detector input: GRU proj + CNN proj + cues
        det_in = P + CP + cue_total
        # Classifier input: GRU proj + cues + p_special (no CNN — context matters more)
        cls_in = P + cue_total + 1

        K_tonal = cfg.num_tonal_classes - 1
        K_artic = cfg.num_artic_classes - 1

        # Level 1 — binary detector (uses both GRU + CNN features)
        self.tonal_detector = nn.Sequential(
            nn.Linear(det_in, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )
        self.artic_detector = nn.Sequential(
            nn.Linear(det_in, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )

        # Level 2 — fine-grained classifier (GRU features + cues only)
        self.tonal_classifier = nn.Sequential(
            nn.Linear(cls_in, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, K_tonal),
        )
        self.artic_classifier = nn.Sequential(
            nn.Linear(cls_in, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, K_artic),
        )

        # Legato — uses both features (bow change is a local event)
        self.legato_head = nn.Sequential(
            nn.Linear(det_in, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        """
        Args:
            features:   (B, T, 512)  GRU output
            conv2d_map: (B, 128, T, 14) conv_block4 output
        """
        gru_feat = self.feat_proj(features)     # (B, T, P)

        # CNN local features: pool over frequency → project
        if conv2d_map is not None:
            # (B, 128, T, 14) → mean over freq → (B, 128, T) → (B, T, 128)
            cnn_local = conv2d_map.mean(dim=-1).transpose(1, 2)
            cnn_feat = self.cnn_proj(cnn_local)  # (B, T, CP)
        else:
            B, T = gru_feat.shape[:2]
            cnn_feat = torch.zeros(B, T, self.cfg.cnn_proj_dim,
                                   device=gru_feat.device)

        if self.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob))
            c_offset = F.gelu(self.offset_proj(offset_prob))
            c_frame = F.gelu(self.frame_proj(frame_prob))
            cues = torch.cat([c_onset, c_offset, c_frame], dim=-1)
            det_feat = torch.cat([gru_feat, cnn_feat, cues], dim=-1)
            cls_base = torch.cat([gru_feat, cues], dim=-1)
        else:
            det_feat = torch.cat([gru_feat, cnn_feat], dim=-1)
            cls_base = gru_feat

        # Level 1: binary detection (GRU + CNN + cues)
        det_tonal = self.tonal_detector(det_feat)
        det_artic = self.artic_detector(det_feat)

        p_tonal = torch.sigmoid(det_tonal)
        p_artic = torch.sigmoid(det_artic)

        # Level 2: fine-grained classification (GRU + cues + p_special)
        cls_tonal = self.tonal_classifier(torch.cat([cls_base, p_tonal], dim=-1))
        cls_artic = self.artic_classifier(torch.cat([cls_base, p_artic], dim=-1))

        tonal_logits = _compose_log_probs(
            det_tonal, cls_tonal,
            self.cfg.num_tonal_classes, self.cfg.tonal_none_id)
        artic_logits = _compose_log_probs(
            det_artic, cls_artic,
            self.cfg.num_artic_classes, self.cfg.artic_none_id)

        # Legato (uses GRU + CNN + cues — bow change is local)
        legato_prob = torch.sigmoid(self.legato_head(det_feat))

        return (tonal_logits, artic_logits, legato_prob,
                None, None, None,
                [], [], [])


# ===================================================================
# Variant 2 — Hierarchical L1 + MoE L2 hybrid
# ===================================================================

class _L2Gate(nn.Module):
    """Lightweight gate for Level-2 MoE routing."""
    def __init__(self, in_dim: int, num_experts: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_experts)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


@dataclass
class FrameHierarchicalMoEConfig:
    feat_dim: int = 512
    proj_dim: int = 256
    cue_dim: int = 16
    hidden_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    tonal_none_id: int = 0
    artic_none_id: int = 0
    dropout: float = 0.2
    use_cues: bool = True
    # MoE L2 settings
    num_l2_experts: int = 4
    l2_expert_dim: int = 64
    l2_top_k: int = 0        # 0 = use all experts (soft routing)


class FrameHierarchicalMoEHead(nn.Module):
    """Hierarchical Level-1 detector + MoE Level-2 classifier.

    Level 1 (hierarchical): P(special) — binary detector, generalizes well
    Level 2 (MoE):          P(k | special) — multiple expert MLPs for
                             fine-grained classification among rare classes

    The L2 MoE has per-task gates and per-task classification heads
    operating on shared expert features. This provides diversity for
    fine-grained discrimination while L1 ensures cross-dataset stability.

    Forward returns the same 9-tuple as FrameMultiScaleMoEHead.
    """

    def __init__(self, cfg: FrameHierarchicalMoEConfig = None):
        super().__init__()
        cfg = cfg or FrameHierarchicalMoEConfig()
        self.cfg = cfg

        D = cfg.feat_dim
        P = cfg.proj_dim
        C = cfg.cue_dim
        H = cfg.hidden_dim
        E = cfg.num_l2_experts
        ED = cfg.l2_expert_dim
        self.use_cues = cfg.use_cues

        self.feat_proj = nn.Sequential(
            nn.Linear(D, P, bias=False),
            nn.LayerNorm(P),
            nn.GELU(),
        )

        cue_total = 3 * C if self.use_cues else 0
        if self.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        feat_total = P + cue_total

        K_tonal = cfg.num_tonal_classes - 1
        K_artic = cfg.num_artic_classes - 1

        # Level 1 — binary detectors (same as original hierarchical)
        self.tonal_detector = nn.Sequential(
            nn.Linear(feat_total, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )
        self.artic_detector = nn.Sequential(
            nn.Linear(feat_total, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )

        # Level 2 — MoE experts (shared across tonal/artic tasks)
        l2_in = feat_total + 1   # features + cues + P(special)
        self.l2_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(l2_in, ED),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for _ in range(E)
        ])

        # Per-task L2 gates
        self.l2_gate_tonal = _L2Gate(l2_in, E, hidden=H // 2)
        self.l2_gate_artic = _L2Gate(l2_in, E, hidden=H // 2)

        # Per-task L2 classification heads (from expert dim to class logits)
        self.l2_tonal_head = nn.Linear(ED, K_tonal)
        self.l2_artic_head = nn.Linear(ED, K_artic)

        # Per-expert auxiliary heads (for load balance / specialization)
        self.aux_tonal_heads = nn.ModuleList([
            nn.Linear(ED, K_tonal) for _ in range(E)])
        self.aux_artic_heads = nn.ModuleList([
            nn.Linear(ED, K_artic) for _ in range(E)])

        # Legato — simple MLP (no MoE needed for binary)
        self.legato_head = nn.Sequential(
            nn.Linear(feat_total, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
        )

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        E = logits.shape[-1]
        k = self.cfg.l2_top_k
        if 0 < k < E:
            _, top_idx = logits.topk(k, dim=-1)
            mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1)

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        feat = self.feat_proj(features)

        if self.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob))
            c_offset = F.gelu(self.offset_proj(offset_prob))
            c_frame = F.gelu(self.frame_proj(frame_prob))
            feat = torch.cat([feat, c_onset, c_offset, c_frame], dim=-1)

        # === Level 1: binary detection ===
        det_tonal = self.tonal_detector(feat)
        det_artic = self.artic_detector(feat)

        p_tonal = torch.sigmoid(det_tonal)
        p_artic = torch.sigmoid(det_artic)

        # === Level 2: MoE fine-grained classification ===
        l2_in_tonal = torch.cat([feat, p_tonal], dim=-1)
        l2_in_artic = torch.cat([feat, p_artic], dim=-1)

        # Run all experts
        expert_outs = [e(l2_in_tonal) for e in self.l2_experts]
        expert_stack = torch.stack(expert_outs, dim=-2)  # (B, T, E, ED)

        # Per-task gating and mixing
        gp_tonal = self._get_gate_probs(self.l2_gate_tonal, l2_in_tonal)
        gp_artic = self._get_gate_probs(self.l2_gate_artic, l2_in_artic)

        mixed_tonal = (gp_tonal.unsqueeze(-1) * expert_stack).sum(dim=-2)
        mixed_artic = (gp_artic.unsqueeze(-1) * expert_stack).sum(dim=-2)

        cls_tonal = self.l2_tonal_head(mixed_tonal)
        cls_artic = self.l2_artic_head(mixed_artic)

        # Compose hierarchical log-probs
        tonal_logits = _compose_log_probs(
            det_tonal, cls_tonal,
            self.cfg.num_tonal_classes, self.cfg.tonal_none_id)
        artic_logits = _compose_log_probs(
            det_artic, cls_artic,
            self.cfg.num_artic_classes, self.cfg.artic_none_id)

        legato_prob = torch.sigmoid(self.legato_head(feat))

        # Per-expert auxiliary L2 logits (for specialization loss)
        aux_tonal = [h(expert_outs[i])
                     for i, h in enumerate(self.aux_tonal_heads)]
        aux_artic = [h(expert_outs[i])
                     for i, h in enumerate(self.aux_artic_heads)]

        return (tonal_logits, artic_logits, legato_prob,
                gp_tonal, gp_artic, None,
                aux_tonal, aux_artic, [])
