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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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
