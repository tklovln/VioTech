"""CRNN-based Mixture of Experts backbones for technique classification.

Two architectures:

1. CRNNMoEBackbone  — each expert is a full independent lightweight CRNN.
2. SharedCNNMoEBackbone — shared CNN blocks 1-3 + per-expert block4 & GRU.
   Solves the BN-instability / capacity-waste issues of (1) by reusing
   low-level spectral features while still diversifying high-level temporal
   processing through different block-4 temporal kernels and GRU configs.

SharedCNNMoEBackbone default 8-expert late branches:

  E0  Micro-Attack  — block4 tk=1 + 1L causal GRU       | CNN RF 130ms
  E1  Short-Bi      — block4 tk=3 + 1L BiGRU            | CNN RF 170ms
  E2  Medium-Causal — block4 tk=5 + 2L causal GRU       | CNN RF 210ms
  E3  Medium-Bi     — block4 tk=5 + 2L BiGRU            | CNN RF 210ms
  E4  Deep-Wide     — block4 tk=7 + 2L BiGRU h=192      | CNN RF 250ms
  E5  Wide-Spectral — block4 tk=3,ch=192 + 1L BiGRU h=96| CNN RF 170ms
  E6  Dilated-Conv  — block4 tk=3 + dilated Conv1D       | CNN RF 950ms
  E7  Spectral-Only — block4 tk=1,ch=192 + no temporal   | CNN RF 130ms

Forward signature matches FrameMultiScaleMoEHead for drop-in use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Expert configuration
# ---------------------------------------------------------------------------
@dataclass
class LightCRNNConfig:
    """Architecture of one lightweight CRNN expert."""
    channels: Tuple[int, ...] = (24, 32, 48, 64)
    time_kernels: Tuple[int, ...] = (3, 3, 3, 3)
    freq_pools: Tuple[int, ...] = (2, 2, 2, 2)
    fc_dim: int = 384
    gru_hidden: int = 128
    gru_layers: int = 1
    gru_bidirectional: bool = True
    post_type: str = 'gru'           # 'gru' | 'dilated_conv' | 'none'
    dilated_kernels: Tuple[Tuple[int, int], ...] = ((7, 1), (7, 3), (7, 9))
    dropout_cnn: float = 0.2
    dropout_fc: float = 0.5
    momentum: float = 0.01
    label: str = ''

    @property
    def output_dim(self) -> int:
        if self.post_type == 'gru' and self.gru_hidden > 0:
            return self.gru_hidden * (2 if self.gru_bidirectional else 1)
        return self.fc_dim


DEFAULT_EXPERT_CONFIGS: Tuple[LightCRNNConfig, ...] = (
    LightCRNNConfig(
        channels=(24, 32, 48, 64), time_kernels=(1, 1, 1, 3), fc_dim=384,
        gru_hidden=128, gru_layers=1, gru_bidirectional=False,
        post_type='gru', label='E0:Micro-Attack(causal)'),
    LightCRNNConfig(
        channels=(24, 32, 48, 64), time_kernels=(3, 3, 3, 3), fc_dim=384,
        gru_hidden=128, gru_layers=1, gru_bidirectional=True,
        post_type='gru', label='E1:Short-Bi'),
    LightCRNNConfig(
        channels=(32, 48, 64, 96), time_kernels=(3, 3, 5, 5), fc_dim=512,
        gru_hidden=128, gru_layers=2, gru_bidirectional=False,
        post_type='gru', label='E2:Medium-Causal'),
    LightCRNNConfig(
        channels=(32, 48, 64, 96), time_kernels=(3, 3, 5, 5), fc_dim=512,
        gru_hidden=128, gru_layers=2, gru_bidirectional=True,
        post_type='gru', label='E3:Medium-Bi'),
    LightCRNNConfig(
        channels=(24, 32, 48, 64), time_kernels=(5, 5, 5, 5), fc_dim=384,
        gru_hidden=192, gru_layers=2, gru_bidirectional=True,
        post_type='gru', label='E4:Deep-Narrow'),
    LightCRNNConfig(
        channels=(48, 64, 96, 128), time_kernels=(1, 1, 3, 3), fc_dim=512,
        freq_pools=(4, 2, 2, 1),
        gru_hidden=96, gru_layers=1, gru_bidirectional=True,
        post_type='gru', label='E5:Wide-Spectral'),
    LightCRNNConfig(
        channels=(24, 32, 48, 64), time_kernels=(3, 3, 3, 3), fc_dim=256,
        gru_hidden=0, post_type='dilated_conv',
        dilated_kernels=((7, 1), (7, 3), (7, 9)),
        label='E6:Dilated-Conv'),
    LightCRNNConfig(
        channels=(48, 64, 96, 128), time_kernels=(1, 1, 1, 1), fc_dim=256,
        gru_hidden=0, post_type='none',
        label='E7:Spectral-Only'),
)


@dataclass
class CRNNMoEConfig:
    """Top-level configuration for the CRNN-MoE backbone."""
    expert_configs: Optional[Tuple[LightCRNNConfig, ...]] = None
    shared_dim: int = 128
    expert_hidden: int = 256
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    cue_dim: int = 16
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.005
    top_k: int = 3
    use_cues: bool = True
    expert_mask: Optional[Tuple[int, ...]] = None
    router_dim: int = 128


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _LightConvBlock(nn.Module):
    """2-layer conv block with configurable temporal kernel."""

    def __init__(self, in_ch: int, out_ch: int, time_kernel: int = 3,
                 momentum: float = 0.01):
        super().__init__()
        t_pad = time_kernel // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, (time_kernel, 3),
                               padding=(t_pad, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=momentum)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (time_kernel, 3),
                               padding=(t_pad, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=momentum)
        self._init_weights()

    def _init_weights(self):
        from models_contrast import init_layer, init_bn
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(1, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if pool_size[0] > 1 or pool_size[1] > 1:
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class _DilatedResBlock(nn.Module):
    """Residual 1-D dilated conv (preserves sequence length)."""

    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        return x + F.gelu(self.bn(self.conv(x)))


class _Gate(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, hidden_dim: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Lightweight CRNN expert
# ---------------------------------------------------------------------------
class LightCRNN(nn.Module):
    """Single lightweight CRNN with configurable architecture.

    Input : logmel (B, 1, T, 229)
    Output: (B, T, output_dim)
    """

    def __init__(self, cfg: LightCRNNConfig):
        super().__init__()
        self.cfg = cfg

        chs = cfg.channels
        self.blocks = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(chs):
            self.blocks.append(
                _LightConvBlock(in_ch, out_ch, cfg.time_kernels[i], cfg.momentum))
            in_ch = out_ch

        freq = 229
        for p in cfg.freq_pools:
            freq = freq // p
        self._freq_bins = freq
        midfeat = chs[-1] * freq

        self.fc = nn.Linear(midfeat, cfg.fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(cfg.fc_dim, momentum=cfg.momentum)
        from models_contrast import init_layer, init_bn
        init_layer(self.fc)
        init_bn(self.bn)

        if cfg.post_type == 'gru' and cfg.gru_hidden > 0:
            self.gru = nn.GRU(
                cfg.fc_dim, cfg.gru_hidden,
                num_layers=cfg.gru_layers,
                batch_first=True,
                bidirectional=cfg.gru_bidirectional)
            from models_contrast import init_gru
            init_gru(self.gru)
        elif cfg.post_type == 'dilated_conv':
            self.dilated_stack = nn.Sequential(*[
                _DilatedResBlock(cfg.fc_dim, k, d)
                for k, d in cfg.dilated_kernels
            ])

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        x = logmel
        for i, block in enumerate(self.blocks):
            x = block(x, pool_size=(1, self.cfg.freq_pools[i]))
            x = F.dropout(x, p=self.cfg.dropout_cnn, training=self.training)

        x = x.transpose(1, 2).flatten(2)                           # (B, T, midfeat)
        x = F.relu(self.bn(self.fc(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=self.cfg.dropout_fc, training=self.training)

        if self.cfg.post_type == 'gru' and self.cfg.gru_hidden > 0:
            x, _ = self.gru(x)
            x = F.dropout(x, p=self.cfg.dropout_fc, training=self.training)
        elif self.cfg.post_type == 'dilated_conv':
            x = self.dilated_stack(x.transpose(1, 2)).transpose(1, 2)

        return x


# ---------------------------------------------------------------------------
# Router (lightweight CNN stem)
# ---------------------------------------------------------------------------
class _RouterStem(nn.Module):
    """Two small ConvBlocks → per-frame routing features."""

    def __init__(self, router_dim: int = 128, momentum: float = 0.01):
        super().__init__()
        self.conv1 = _LightConvBlock(1, 16, time_kernel=3, momentum=momentum)
        self.conv2 = _LightConvBlock(16, 32, time_kernel=3, momentum=momentum)
        freq_after = 229 // 4 // 4       # two (1,4) pools → 14
        self.proj = nn.Linear(32 * freq_after, router_dim, bias=False)
        self.norm = nn.LayerNorm(router_dim)

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        x = self.conv1(logmel, pool_size=(1, 4))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, pool_size=(1, 4))
        x = F.dropout(x, p=0.1, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        return F.gelu(self.norm(self.proj(x)))


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------
class CRNNMoEBackbone(nn.Module):
    """CRNN-as-Expert MoE backbone.

    Each active expert is a full LightCRNN that independently processes
    the log-mel spectrogram.  A lightweight router CNN + per-task gates
    decide how to combine expert outputs per frame.

    Forward returns the same 9-tuple as FrameMultiScaleMoEHead:
        tonal_logits, artic_logits, legato_prob,
        gate_tonal, gate_artic, gate_legato,
        aux_tonal_list, aux_artic_list, aux_legato_list
    """

    def __init__(self, cfg: CRNNMoEConfig = None):
        super().__init__()
        cfg = cfg or CRNNMoEConfig()
        self.cfg = cfg

        expert_cfgs = cfg.expert_configs or DEFAULT_EXPERT_CONFIGS

        if cfg.expert_mask is not None:
            active = sorted(cfg.expert_mask)
        else:
            active = list(range(len(expert_cfgs)))
        self.active_expert_ids = active
        num_experts = len(active)

        # CRNN experts + projection to shared_dim
        self.crnn_experts = nn.ModuleList()
        self.expert_projections = nn.ModuleList()
        self._expert_names = {}
        for idx, eid in enumerate(active):
            ecfg = expert_cfgs[eid]
            self.crnn_experts.append(LightCRNN(ecfg))
            out_dim = ecfg.output_dim
            self.expert_projections.append(nn.Sequential(
                nn.Linear(out_dim, cfg.shared_dim, bias=False),
                nn.LayerNorm(cfg.shared_dim),
                nn.GELU(),
            ))
            self._expert_names[eid] = ecfg.label or f'Expert{eid}'

        # Router
        self.router = _RouterStem(router_dim=cfg.router_dim)

        # Cue projections
        cue_total = 3 * cfg.cue_dim if cfg.use_cues else 0
        if cfg.use_cues:
            self.onset_proj = nn.Linear(1, cfg.cue_dim)
            self.offset_proj = nn.Linear(1, cfg.cue_dim)
            self.frame_proj = nn.Linear(1, cfg.cue_dim)

        # Per-expert refinement MLPs (CRNN features + cues → shared_dim)
        self.expert_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.shared_dim + cue_total, cfg.expert_hidden),
                nn.GELU(),
                nn.Dropout(cfg.expert_dropout),
                nn.Linear(cfg.expert_hidden, cfg.shared_dim),
            )
            for _ in range(num_experts)
        ])

        # Per-task gates
        gate_in = cfg.router_dim + cue_total
        gate_hidden = cfg.expert_hidden // 2
        self.gate_tonal = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_artic = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_legato = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)

        # Classification heads
        S = cfg.shared_dim
        self.ln_tonal = nn.LayerNorm(S)
        self.ln_artic = nn.LayerNorm(S)
        self.ln_legato = nn.LayerNorm(S)
        self.tonal_head = nn.Linear(S, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(S, cfg.num_artic_classes)
        self.legato_head = nn.Linear(S, 1)

        # Per-expert auxiliary heads
        self.aux_tonal_heads = nn.ModuleList([
            nn.Linear(S, cfg.num_tonal_classes) for _ in range(num_experts)])
        self.aux_artic_heads = nn.ModuleList([
            nn.Linear(S, cfg.num_artic_classes) for _ in range(num_experts)])
        self.aux_legato_heads = nn.ModuleList([
            nn.Linear(S, 1) for _ in range(num_experts)])

    # ---- gating helpers ----

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        num_e = logits.shape[-1]
        if 0 < self.cfg.top_k < num_e:
            return self._top_k_gating(logits, self.cfg.top_k)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def _top_k_gating(logits, k):
        _, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)

    # ---- forward ----

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        """
        Args:
            features:    ignored (kept for API compatibility)
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            logmel:      (B, 1, T, 229) BN-normalised log-mel spectrogram
        """
        assert logmel is not None, "CRNNMoEBackbone requires logmel input"

        # --- Run all CRNN experts ---
        expert_features = []
        for crnn, proj in zip(self.crnn_experts, self.expert_projections):
            feat = crnn(logmel)          # (B, T, expert_out_dim)
            feat = proj(feat)            # (B, T, shared_dim)
            expert_features.append(feat)

        T_feat = expert_features[0].shape[1]

        # --- Cue embeddings ---
        cue_cat = None
        if self.cfg.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob[:, :T_feat, :]))
            c_offset = F.gelu(self.offset_proj(offset_prob[:, :T_feat, :]))
            c_frame = F.gelu(self.frame_proj(frame_prob[:, :T_feat, :]))
            cue_cat = torch.cat([c_onset, c_offset, c_frame], dim=-1)

        # --- Expert refinement MLPs (inject cues) ---
        expert_list = []
        for i, feat in enumerate(expert_features):
            if cue_cat is not None:
                inp = torch.cat([feat, cue_cat], dim=-1)
            else:
                inp = feat
            expert_list.append(self.expert_mlps[i](inp))

        expert_outs = torch.stack(expert_list, dim=-2)  # (B, T, E, shared_dim)

        # --- Router ---
        router_feat = self.router(logmel)[:, :T_feat, :]
        if cue_cat is not None:
            gate_in = torch.cat([router_feat, cue_cat], dim=-1)
        else:
            gate_in = router_feat

        # --- Per-task gating ---
        gp_tonal = self._get_gate_probs(self.gate_tonal, gate_in)
        gp_artic = self._get_gate_probs(self.gate_artic, gate_in)
        gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
        artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
        legato_prob = torch.sigmoid(
            self.legato_head(_mix(gp_legato, self.ln_legato)))

        # --- Per-expert auxiliary logits ---
        aux_tonal = [h(expert_list[i])
                     for i, h in enumerate(self.aux_tonal_heads)]
        aux_artic = [h(expert_list[i])
                     for i, h in enumerate(self.aux_artic_heads)]
        aux_legato = [torch.sigmoid(h(expert_list[i]))
                      for i, h in enumerate(self.aux_legato_heads)]

        return (tonal_logits, artic_logits, legato_prob,
                gp_tonal, gp_artic, gp_legato,
                aux_tonal, aux_artic, aux_legato)


# ===================================================================
# Variant B — Shared early CNN + per-expert late branches
# ===================================================================

@dataclass
class LateBranchConfig:
    """Architecture of one per-expert late branch (block4 + post)."""
    block4_channels: int = 128
    block4_time_kernel: int = 3
    fc_dim: int = 768
    gru_hidden: int = 128
    gru_layers: int = 1
    gru_bidirectional: bool = True
    post_type: str = 'gru'           # 'gru' | 'dilated_conv' | 'none'
    dilated_kernels: Tuple[Tuple[int, int], ...] = ((7, 1), (7, 3), (7, 9))
    dropout: float = 0.5
    label: str = ''

    @property
    def output_dim(self) -> int:
        if self.post_type == 'gru' and self.gru_hidden > 0:
            return self.gru_hidden * (2 if self.gru_bidirectional else 1)
        return self.fc_dim


DEFAULT_BRANCH_CONFIGS: Tuple[LateBranchConfig, ...] = (
    LateBranchConfig(
        block4_channels=128, block4_time_kernel=1, fc_dim=768,
        gru_hidden=128, gru_layers=1, gru_bidirectional=False,
        post_type='gru', label='E0:Micro-Attack(causal)'),
    LateBranchConfig(
        block4_channels=128, block4_time_kernel=3, fc_dim=768,
        gru_hidden=128, gru_layers=1, gru_bidirectional=True,
        post_type='gru', label='E1:Short-Bi'),
    LateBranchConfig(
        block4_channels=128, block4_time_kernel=5, fc_dim=768,
        gru_hidden=128, gru_layers=2, gru_bidirectional=False,
        post_type='gru', label='E2:Medium-Causal'),
    LateBranchConfig(
        block4_channels=128, block4_time_kernel=5, fc_dim=768,
        gru_hidden=128, gru_layers=2, gru_bidirectional=True,
        post_type='gru', label='E3:Medium-Bi'),
    LateBranchConfig(
        block4_channels=128, block4_time_kernel=7, fc_dim=768,
        gru_hidden=192, gru_layers=2, gru_bidirectional=True,
        post_type='gru', label='E4:Deep-Wide'),
    LateBranchConfig(
        block4_channels=192, block4_time_kernel=3, fc_dim=768,
        gru_hidden=96, gru_layers=1, gru_bidirectional=True,
        post_type='gru', label='E5:Wide-Spectral'),
    LateBranchConfig(
        block4_channels=128, block4_time_kernel=3, fc_dim=512,
        gru_hidden=0, post_type='dilated_conv',
        dilated_kernels=((7, 1), (7, 3), (7, 9)),
        label='E6:Dilated-Conv'),
    LateBranchConfig(
        block4_channels=192, block4_time_kernel=1, fc_dim=512,
        gru_hidden=0, post_type='none',
        label='E7:Spectral-Only'),
)


@dataclass
class SharedCNNMoEConfig:
    """Config for shared-CNN + per-expert late branches MoE."""
    branch_configs: Optional[Tuple[LateBranchConfig, ...]] = None
    shared_dim: int = 128
    expert_hidden: int = 256
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    cue_dim: int = 16
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.005
    top_k: int = 3
    use_cues: bool = True
    expert_mask: Optional[Tuple[int, ...]] = None
    router_dim: int = 128


class _LateBranch(nn.Module):
    """One per-expert late branch: block4 (GroupNorm) + FC + GRU/post."""

    def __init__(self, cfg: LateBranchConfig, in_channels: int = 96):
        super().__init__()
        self.cfg = cfg
        t_pad = cfg.block4_time_kernel // 2
        ch = cfg.block4_channels

        self.conv1 = nn.Conv2d(in_channels, ch,
                               (cfg.block4_time_kernel, 3),
                               padding=(t_pad, 1), bias=False)
        self.gn1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch,
                               (cfg.block4_time_kernel, 3),
                               padding=(t_pad, 1), bias=False)
        self.gn2 = nn.GroupNorm(min(8, ch), ch)
        from models_contrast import init_layer
        init_layer(self.conv1)
        init_layer(self.conv2)

        midfeat = ch * 14       # after pool (1,2): 28 → 14
        self.fc = nn.Linear(midfeat, cfg.fc_dim, bias=False)
        self.ln = nn.LayerNorm(cfg.fc_dim)

        if cfg.post_type == 'gru' and cfg.gru_hidden > 0:
            self.gru = nn.GRU(
                cfg.fc_dim, cfg.gru_hidden,
                num_layers=cfg.gru_layers,
                batch_first=True,
                bidirectional=cfg.gru_bidirectional)
            from models_contrast import init_gru
            init_gru(self.gru)
        elif cfg.post_type == 'dilated_conv':
            self.dilated_stack = nn.Sequential(*[
                _DilatedResBlock(cfg.fc_dim, k, d)
                for k, d in cfg.dilated_kernels
            ])

    def forward(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_feat: (B, 96, T, 28) — output of shared CNN blocks 1-3
        Returns:
            (B, T, output_dim)
        """
        x = F.relu(self.gn1(self.conv1(shared_feat)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2))        # (B, ch, T, 14)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)               # (B, T, ch*14)
        x = F.relu(self.ln(self.fc(x)))
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        if self.cfg.post_type == 'gru' and self.cfg.gru_hidden > 0:
            x, _ = self.gru(x)
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        elif self.cfg.post_type == 'dilated_conv':
            x = self.dilated_stack(x.transpose(1, 2)).transpose(1, 2)

        return x


class SharedCNNMoEBackbone(nn.Module):
    """Shared early CNN (blocks 1-3) + per-expert late branches MoE.

    Solves the capacity-waste and BN-instability problems of CRNNMoEBackbone:
    - Blocks 1-3 are shared → stable BN, no redundant low-level learning
    - Block4 + GRU differ per expert → multi-scale temporal specialization
    - Per-expert block4 uses GroupNorm (batch-size invariant)
    - Per-expert FC uses LayerNorm (batch-size invariant)

    Forward returns the same 9-tuple as FrameMultiScaleMoEHead.
    """

    def __init__(self, cfg: SharedCNNMoEConfig = None):
        super().__init__()
        cfg = cfg or SharedCNNMoEConfig()
        self.cfg = cfg

        branch_cfgs = cfg.branch_configs or DEFAULT_BRANCH_CONFIGS

        if cfg.expert_mask is not None:
            active = sorted(cfg.expert_mask)
        else:
            active = list(range(len(branch_cfgs)))
        self.active_expert_ids = active
        num_experts = len(active)

        # --- Shared CNN backbone (blocks 1-3, same as AcousticModelCRnn8Dropout) ---
        from models_contrast import ConvBlock, init_bn
        momentum = 0.01
        self.conv_block1 = ConvBlock(1, 48, momentum)
        self.conv_block2 = ConvBlock(48, 64, momentum)
        self.conv_block3 = ConvBlock(64, 96, momentum)

        # --- Per-expert late branches ---
        self.branches = nn.ModuleList()
        self.expert_projections = nn.ModuleList()
        self._expert_names = {}
        for idx, eid in enumerate(active):
            bcfg = branch_cfgs[eid]
            self.branches.append(_LateBranch(bcfg, in_channels=96))
            out_dim = bcfg.output_dim
            self.expert_projections.append(nn.Sequential(
                nn.Linear(out_dim, cfg.shared_dim, bias=False),
                nn.LayerNorm(cfg.shared_dim),
                nn.GELU(),
            ))
            self._expert_names[eid] = bcfg.label or f'Expert{eid}'

        # --- Router from shared CNN features ---
        # shared CNN output (B, 96, T, 28) → pool freq → project
        self.router_proj = nn.Linear(96, cfg.router_dim, bias=False)
        self.router_norm = nn.LayerNorm(cfg.router_dim)

        # Cue projections
        cue_total = 3 * cfg.cue_dim if cfg.use_cues else 0
        if cfg.use_cues:
            self.onset_proj = nn.Linear(1, cfg.cue_dim)
            self.offset_proj = nn.Linear(1, cfg.cue_dim)
            self.frame_proj = nn.Linear(1, cfg.cue_dim)

        # Per-expert refinement MLPs
        self.expert_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.shared_dim + cue_total, cfg.expert_hidden),
                nn.GELU(),
                nn.Dropout(cfg.expert_dropout),
                nn.Linear(cfg.expert_hidden, cfg.shared_dim),
            )
            for _ in range(num_experts)
        ])

        # Per-task gates
        gate_in = cfg.router_dim + cue_total
        gate_hidden = cfg.expert_hidden // 2
        self.gate_tonal = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_artic = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_legato = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)

        # Classification heads
        S = cfg.shared_dim
        self.ln_tonal = nn.LayerNorm(S)
        self.ln_artic = nn.LayerNorm(S)
        self.ln_legato = nn.LayerNorm(S)
        self.tonal_head = nn.Linear(S, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(S, cfg.num_artic_classes)
        self.legato_head = nn.Linear(S, 1)

        # Per-expert auxiliary heads
        self.aux_tonal_heads = nn.ModuleList([
            nn.Linear(S, cfg.num_tonal_classes) for _ in range(num_experts)])
        self.aux_artic_heads = nn.ModuleList([
            nn.Linear(S, cfg.num_artic_classes) for _ in range(num_experts)])
        self.aux_legato_heads = nn.ModuleList([
            nn.Linear(S, 1) for _ in range(num_experts)])

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        num_e = logits.shape[-1]
        if 0 < self.cfg.top_k < num_e:
            return self._top_k_gating(logits, self.cfg.top_k)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def _top_k_gating(logits, k):
        _, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        """
        Args:
            features:    ignored (kept for API compatibility)
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            logmel:      (B, 1, T, 229) BN-normalised log-mel spectrogram
        """
        assert logmel is not None, "SharedCNNMoEBackbone requires logmel input"

        # --- Shared CNN blocks 1-3 ---
        x = self.conv_block1(logmel, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        shared_feat = x                                 # (B, 96, T, 28)

        # --- Router from shared features ---
        # Global-average-pool over frequency → (B, 96, T) → (B, T, 96)
        router_in = shared_feat.mean(dim=-1).transpose(1, 2)
        router_feat = F.gelu(self.router_norm(self.router_proj(router_in)))

        T_feat = router_feat.shape[1]

        # --- Per-expert late branches ---
        expert_features = []
        for branch, proj in zip(self.branches, self.expert_projections):
            feat = branch(shared_feat)   # (B, T, branch_out_dim)
            feat = proj(feat)            # (B, T, shared_dim)
            expert_features.append(feat)

        # --- Cue embeddings ---
        cue_cat = None
        if self.cfg.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob[:, :T_feat, :]))
            c_offset = F.gelu(self.offset_proj(offset_prob[:, :T_feat, :]))
            c_frame = F.gelu(self.frame_proj(frame_prob[:, :T_feat, :]))
            cue_cat = torch.cat([c_onset, c_offset, c_frame], dim=-1)

        # --- Expert refinement MLPs (inject cues) ---
        expert_list = []
        for i, feat in enumerate(expert_features):
            if cue_cat is not None:
                inp = torch.cat([feat, cue_cat], dim=-1)
            else:
                inp = feat
            expert_list.append(self.expert_mlps[i](inp))

        expert_outs = torch.stack(expert_list, dim=-2)  # (B, T, E, shared_dim)

        # --- Gating ---
        if cue_cat is not None:
            gate_in = torch.cat([router_feat, cue_cat], dim=-1)
        else:
            gate_in = router_feat

        gp_tonal = self._get_gate_probs(self.gate_tonal, gate_in)
        gp_artic = self._get_gate_probs(self.gate_artic, gate_in)
        gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
        artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
        legato_prob = torch.sigmoid(
            self.legato_head(_mix(gp_legato, self.ln_legato)))

        # Per-expert auxiliary logits
        aux_tonal = [h(expert_list[i])
                     for i, h in enumerate(self.aux_tonal_heads)]
        aux_artic = [h(expert_list[i])
                     for i, h in enumerate(self.aux_artic_heads)]
        aux_legato = [torch.sigmoid(h(expert_list[i]))
                      for i, h in enumerate(self.aux_legato_heads)]

        return (tonal_logits, artic_logits, legato_prob,
                gp_tonal, gp_artic, gp_legato,
                aux_tonal, aux_artic, aux_legato)
