"""Frame-level Multi-Scale MoE for technique classification.

Strictly end-to-end: no GT note boundaries at inference.
Uses the transcription branch's onset / frame-activity probabilities as
structural cues, combined with multi-scale temporal convolutions.

Expert 0 — Onset specialist (short-range ~110ms):
    Small 1D conv + onset probability injection.
    Captures attack transients (spiccato, pizzicato, bow-change).

Expert 1 — Note specialist (medium-range ~510ms):
    Medium 1D conv + frame-activity injection.
    Captures sustained note characteristics (vibrato, sustain).

Expert 2 — Phrase specialist (long-range ~1.5s):
    Large 1D conv (dilated for efficiency) + all cues.
    Captures legato transitions, phrase-level structure.

Expert 3 — Spectral specialist (frequency domain):
    2D CNN on logmel spectrogram + frame-activity cue.
    Captures harmonic energy distribution, spectral envelope differences
    (e.g. harmonics vs normal have different overtone patterns at same pitch).

Three independent per-task gates (tonal / articulation / legato), so each
task can learn its own routing pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class FrameMultiScaleMoEConfig:
    feat_dim: int = 512
    proj_dim: int = 128
    num_experts: int = 4
    short_kernel: int = 11      # ~110ms at 100fps  (legacy; use temporal_experts instead)
    note_kernel: int = 51       # ~510ms             (legacy)
    phrase_kernel: int = 51     # dilated×3 → ~1.5s  (legacy)
    phrase_dilation: int = 3    #                     (legacy)
    expert_hidden: int = 256
    shared_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    cue_dim: int = 16           # projected dim for each transcription cue
    spectral_proj_dim: int = 128     # spectral expert projected dim (→ SP)
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.001
    top_k: int = 2
    use_spectral_expert: bool = True
    # --- ablation knobs ---
    expert_mask: tuple = None       # e.g. (0,1,2) — None means all available
    shared_gate: bool = False       # single gate shared across tonal/artic/legato
    uniform_routing: bool = False   # equal-weight bypass (no learned gating)
    # --- parameterized temporal experts ---
    temporal_experts: tuple = None  # ((kernel, dilation), ...) None = legacy 3-expert
    use_cues: bool = True           # inject transcription cues into experts and gate
    # --- binary technique mode ---
    binary_technique: bool = False  # 4 independent binary heads (pizz/harm/spic/legato)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _TemporalCtx(nn.Module):
    """1D depthwise-separable conv for temporal context extraction."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dw = nn.Conv1d(in_dim, in_dim, kernel_size, padding=padding,
                            dilation=dilation, groups=in_dim, bias=False)
        self.pw = nn.Conv1d(in_dim, out_dim, 1, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, T, D)
        h = self.dw(x.transpose(1, 2))  # (B, D, T)
        h = self.pw(h).transpose(1, 2)  # (B, T, out_dim)
        return F.gelu(self.norm(h))


class _DedicatedSpectralCNN(nn.Module):
    """Dedicated 4-layer ConvBlock stack for spectral technique features.

    Same proven architecture as NoteLevelTechniqueModel's conv stack, but
    outputs **per-frame** features instead of a global-pooled summary.

    All parameters are trained exclusively by the technique loss — no
    transcription gradient contaminates these weights.  This gives the
    spectral expert filters that are specifically tuned for technique-
    discriminative spectral patterns (overtone distribution, attack shape,
    spectral envelope).

    Input : logmel (B, 1, T, 229)  — BN-normalised log-mel spectrogram
    Output: (B, T, out_dim)        — per-frame spectral descriptor
    """

    def __init__(self, out_dim: int = 128, momentum: float = 0.01):
        super().__init__()
        from models_contrast import ConvBlock
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        # 128 channels × 14 freq bins (229 / 2^4) = 1792
        self.proj = nn.Linear(1792, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, logmel):
        # logmel: (B, 1, T, 229)
        x = self.conv_block1(logmel, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # (B, 128, T, 14) → per-frame flatten + project
        x = x.transpose(1, 2).flatten(2)       # (B, T, 1792)
        return F.gelu(self.norm(self.proj(x)))  # (B, T, out_dim)


class _Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _Gate(nn.Module):
    def __init__(self, in_dim, num_experts, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------
class FrameMultiScaleMoEHead(nn.Module):
    """Frame-level MoE with multi-scale temporal experts + transcription cues.

    Forward returns:
        tonal_logits       (B, T, 4)
        artic_logits       (B, T, 4)
        bow_change_prob    (B, T, 1)  — 1.0 = bow_change, 0.0 = sustained
        gate_tonal         (B, T, E)
        gate_artic         (B, T, E)
        gate_legato        (B, T, E)
    """

    def __init__(self, cfg: FrameMultiScaleMoEConfig = None):
        super().__init__()
        cfg = cfg or FrameMultiScaleMoEConfig()
        self.cfg = cfg
        self.use_spectral = cfg.use_spectral_expert
        self.use_cues = cfg.use_cues
        D = cfg.feat_dim
        P = cfg.proj_dim
        C = cfg.cue_dim
        SP = cfg.spectral_proj_dim

        # Resolve temporal expert definitions
        # Legacy mode: temporal_experts=None → old per-expert cue assignment
        #   Expert 0: P+C (onset), Expert 1: P+C (frame), Expert 2: P+3C (all)
        # New mode: temporal_experts specified → unified all cues for every expert
        self._legacy_cue_mode = (cfg.temporal_experts is None and self.use_cues)
        if cfg.temporal_experts is not None:
            temporal_defs = tuple(cfg.temporal_experts)
        else:
            temporal_defs = (
                (cfg.short_kernel, 1),
                (cfg.note_kernel, 1),
                (cfg.phrase_kernel, cfg.phrase_dilation),
            )
        self.temporal_defs = temporal_defs
        N_temporal = len(temporal_defs)
        self.n_temporal = N_temporal

        # Spectral expert index = N_temporal (last)
        N_total = N_temporal + (1 if self.use_spectral else 0)

        # Resolve which experts are active
        if cfg.expert_mask is not None:
            mask = tuple(sorted(cfg.expert_mask))
        else:
            mask = tuple(range(N_total))
        spectral_idx = N_temporal
        if spectral_idx in mask and not self.use_spectral:
            raise ValueError(
                f"Expert {spectral_idx} (spectral) requires use_spectral_expert=True")
        self.active_expert_ids = list(mask)

        # Build expert name map for evaluation reporting
        self._expert_names = {}
        for i, (k, d) in enumerate(temporal_defs):
            eff_ms = k * d * 10  # at 100fps each frame = 10ms
            self._expert_names[i] = f'T{i}({k}x{d}={eff_ms}ms)'
        if self.use_spectral:
            self._expert_names[spectral_idx] = 'Spectral'

        # Transcription cue projections (only if use_cues)
        cue_total = 3 * C if self.use_cues else 0
        if self.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        # Build all temporal context extractors
        self.temporal_ctxs = nn.ModuleList([
            _TemporalCtx(D, P, k, d) for k, d in temporal_defs
        ])

        # Per-expert input dims
        if self._legacy_cue_mode:
            # Old per-expert cue assignment: E0=P+C, E1=P+C, E2=P+3C
            _exp_in = {0: P + C, 1: P + C, 2: P + 3 * C}
        else:
            _exp_in = {i: P + cue_total for i in range(N_temporal)}

        # Gate input: all N temporal contexts + 3 cues + (spectral if present)
        gate_in = N_temporal * P + cue_total

        if self.use_spectral:
            self.spectral_cnn = _DedicatedSpectralCNN(out_dim=SP)
            self.feat_proj = nn.Sequential(
                nn.Linear(D, P, bias=False),
                nn.LayerNorm(P),
                nn.GELU(),
            )
            if self._legacy_cue_mode:
                _exp_in[spectral_idx] = SP + P + C
            else:
                _exp_in[spectral_idx] = SP + P + cue_total
            gate_in += SP + P

        # Build expert MLPs for active experts
        experts = []
        for eid in self.active_expert_ids:
            experts.append(_Expert(_exp_in[eid], cfg.expert_hidden,
                                   cfg.shared_dim, cfg.expert_dropout))
        self.experts = nn.ModuleList(experts)
        num_experts = len(experts)

        # Gates & classification heads
        gate_hidden = cfg.expert_hidden // 2
        self._binary_mode = cfg.binary_technique

        if cfg.binary_technique:
            self._binary_flag_names = ['pizzicato', 'harmonic', 'spiccato', 'legato']
            n_flags = len(self._binary_flag_names)
            self.gates = nn.ModuleDict({
                name: _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
                for name in self._binary_flag_names
            })
            self.layer_norms = nn.ModuleDict({
                name: nn.LayerNorm(cfg.shared_dim)
                for name in self._binary_flag_names
            })
            self.heads = nn.ModuleDict({
                name: nn.Linear(cfg.shared_dim, 1)
                for name in self._binary_flag_names
            })
            self.aux_heads = nn.ModuleDict({
                name: nn.ModuleList([
                    nn.Linear(cfg.shared_dim, 1) for _ in range(num_experts)
                ])
                for name in self._binary_flag_names
            })
        else:
            if cfg.shared_gate:
                self.gate_shared = _Gate(gate_in, num_experts, gate_hidden,
                                         cfg.gate_dropout)
            else:
                self.gate_tonal = _Gate(gate_in, num_experts, gate_hidden,
                                        cfg.gate_dropout)
                self.gate_artic = _Gate(gate_in, num_experts, gate_hidden,
                                        cfg.gate_dropout)
                self.gate_legato = _Gate(gate_in, num_experts, gate_hidden,
                                         cfg.gate_dropout)

            self.ln_tonal = nn.LayerNorm(cfg.shared_dim)
            self.ln_artic = nn.LayerNorm(cfg.shared_dim)
            self.ln_legato = nn.LayerNorm(cfg.shared_dim)

            self.tonal_head = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
            self.artic_head = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
            self.legato_head = nn.Linear(cfg.shared_dim, 1)

            self.aux_tonal_heads = nn.ModuleList([
                nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
                for _ in range(num_experts)
            ])
            self.aux_artic_heads = nn.ModuleList([
                nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
                for _ in range(num_experts)
            ])
            self.aux_legato_heads = nn.ModuleList([
                nn.Linear(cfg.shared_dim, 1)
                for _ in range(num_experts)
            ])

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        num_e = logits.shape[-1]
        if self.cfg.top_k > 0 and self.cfg.top_k < num_e:
            return self._top_k_gating(logits, self.cfg.top_k)
        return F.softmax(logits, dim=-1)

    def _uniform_probs(self, ref_tensor):
        """Equal 1/E weights with shape (B, T, E)."""
        B, T = ref_tensor.shape[:2]
        E = len(self.active_expert_ids)
        return torch.full((B, T, E), 1.0 / E, device=ref_tensor.device)

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        """
        Args:
            features:    (B, T, D) acoustic features (GRU output) from dedicated CRNN
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            logmel:      (B, 1, T, 229) BN-normalised log-mel spectrogram
            conv2d_map:  (B, C_in, T, F) (unused, kept for backward compat)
        """
        # Compute all temporal contexts
        temporal_outs = [ctx(features) for ctx in self.temporal_ctxs]
        T_feat = temporal_outs[0].shape[1]

        # Cue embeddings (only if use_cues)
        c_onset = c_offset = c_frame = cue_cat = None
        if self.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob))
            c_offset = F.gelu(self.offset_proj(offset_prob))
            c_frame = F.gelu(self.frame_proj(frame_prob))
            cue_cat = torch.cat([c_onset, c_offset, c_frame], dim=-1)

        # Spectral features
        s_spec = f_proj = None
        if self.use_spectral:
            if logmel is not None:
                s_spec = self.spectral_cnn(logmel)
                T_spec = s_spec.shape[1]
                if T_spec != T_feat:
                    s_spec = s_spec[:, :T_feat, :] if T_spec > T_feat else \
                        F.pad(s_spec, (0, 0, 0, T_feat - T_spec))
            else:
                s_spec = torch.zeros(features.shape[0], T_feat,
                                     self.cfg.spectral_proj_dim,
                                     device=features.device)
            f_proj = self.feat_proj(features)

        # Build expert inputs for active experts
        if self._legacy_cue_mode:
            _inp_builders = {
                0: lambda: torch.cat([temporal_outs[0], c_onset], dim=-1),
                1: lambda: torch.cat([temporal_outs[1], c_frame], dim=-1),
                2: lambda: torch.cat([temporal_outs[2], c_onset, c_offset, c_frame], dim=-1),
            }
            if self.use_spectral:
                _inp_builders[self.n_temporal] = lambda: torch.cat(
                    [s_spec, f_proj, c_frame], dim=-1)

        expert_list = []
        for idx, eid in enumerate(self.active_expert_ids):
            if self._legacy_cue_mode:
                inp = _inp_builders[eid]()
            elif eid < self.n_temporal:
                parts = [temporal_outs[eid]]
                if cue_cat is not None:
                    parts.append(cue_cat)
                inp = torch.cat(parts, dim=-1)
            else:
                parts = [s_spec, f_proj]
                if cue_cat is not None:
                    parts.append(cue_cat)
                inp = torch.cat(parts, dim=-1)
            expert_list.append(self.experts[idx](inp))

        expert_outs = torch.stack(expert_list, dim=-2)  # (B, T, E, shared_dim)

        # Gate input: all temporal contexts + (spectral) + (cues)
        gate_parts = list(temporal_outs)
        if self.use_spectral:
            gate_parts.extend([s_spec, f_proj])
        if cue_cat is not None:
            gate_parts.append(cue_cat)
        gate_in = torch.cat(gate_parts, dim=-1)

        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        if self._binary_mode:
            gate_probs = {}
            probs = {}
            aux = {}
            for name in self._binary_flag_names:
                if self.cfg.uniform_routing:
                    gp = self._uniform_probs(features)
                else:
                    gp = self._get_gate_probs(self.gates[name], gate_in)
                gate_probs[name] = gp
                probs[name] = torch.sigmoid(
                    self.heads[name](_mix(gp, self.layer_norms[name])))
                aux[name] = [
                    torch.sigmoid(h(expert_list[i]))
                    for i, h in enumerate(self.aux_heads[name])
                ]
            return probs, gate_probs, aux

        if self.cfg.uniform_routing:
            gp_tonal = gp_artic = gp_legato = self._uniform_probs(features)
        elif self.cfg.shared_gate:
            gp_tonal = gp_artic = gp_legato = self._get_gate_probs(
                self.gate_shared, gate_in)
        else:
            gp_tonal = self._get_gate_probs(self.gate_tonal, gate_in)
            gp_artic = self._get_gate_probs(self.gate_artic, gate_in)
            gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
        artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
        legato_prob = torch.sigmoid(self.legato_head(_mix(gp_legato, self.ln_legato)))

        # Per-expert auxiliary logits (training only)
        aux_tonal = [h(expert_list[i]) for i, h in enumerate(self.aux_tonal_heads)]
        aux_artic = [h(expert_list[i]) for i, h in enumerate(self.aux_artic_heads)]
        aux_legato = [torch.sigmoid(h(expert_list[i]))
                      for i, h in enumerate(self.aux_legato_heads)]

        return (tonal_logits, artic_logits, legato_prob,
                gp_tonal, gp_artic, gp_legato,
                aux_tonal, aux_artic, aux_legato)

    @staticmethod
    def _top_k_gating(logits, k):
        top_vals, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)


@dataclass
class FrameMultiCRNNMoEConfig:
    """Configuration for multi-CRNN MoE (shared CNN + per-expert GRUs)."""
    expert_hidden: int = 256
    shared_dim: int = 128
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    cue_dim: int = 16
    spectral_proj_dim: int = 128
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.001
    top_k: int = 2
    use_spectral_expert: bool = False
    use_cues: bool = True
    expert_mask: tuple = None


class FrameMultiCRNNMoEHead(nn.Module):
    """Multi-CRNN MoE: shared CNN backbone with per-expert GRU branches.

    Replaces the shared-CRNN + post-hoc-Conv1D design with structurally
    different GRU experts that process the same CNN features independently:

      Expert 0 — Onset:  1-layer forward-only GRU (causal, 256-dim output)
      Expert 1 — Note:   2-layer BiGRU (512-dim output)
      Expert 2 — Phrase:  2-layer BiGRU (512-dim output, separate weights)
      Expert 3 — Spectral (optional): dedicated 2D CNN on logmel

    The shared CNN processes logmel once; each GRU branch then independently
    learns temporal dynamics suited to its scale.  This breaks the information
    bottleneck of funnelling all experts through one shared GRU.

    Forward signature matches FrameMultiScaleMoEHead for drop-in replacement.
    """

    _GRU_DEFS = [
        # (hidden_size, num_layers, bidirectional, label)
        (256, 1, False, 'Onset(fwd-GRU-1L)'),
        (256, 2, True,  'Note(bi-GRU-2L)'),
        (256, 2, True,  'Phrase(bi-GRU-2L)'),
    ]

    def __init__(self, cfg: FrameMultiCRNNMoEConfig = None):
        super().__init__()
        cfg = cfg or FrameMultiCRNNMoEConfig()
        self.cfg = cfg
        self.use_spectral = cfg.use_spectral_expert
        self.use_cues = cfg.use_cues
        P = cfg.shared_dim   # 128
        C = cfg.cue_dim      # 16
        SP = cfg.spectral_proj_dim

        N_temporal = len(self._GRU_DEFS)
        N_total = N_temporal + (1 if self.use_spectral else 0)

        if cfg.expert_mask is not None:
            mask = tuple(sorted(cfg.expert_mask))
        else:
            mask = tuple(range(N_total))
        spectral_idx = N_temporal
        if spectral_idx in mask and not self.use_spectral:
            raise ValueError(
                f"Expert {spectral_idx} (spectral) requires use_spectral_expert=True")
        self.active_expert_ids = list(mask)

        self._expert_names = {}
        for i, (_, _, _, label) in enumerate(self._GRU_DEFS):
            self._expert_names[i] = label
        if self.use_spectral:
            self._expert_names[spectral_idx] = 'Spectral'

        # --- Shared CNN backbone (same arch as AcousticModelCRnn8Dropout) ---
        from models_contrast import ConvBlock, init_gru
        momentum = 0.01
        self.conv_block1 = ConvBlock(1, 48, momentum)
        self.conv_block2 = ConvBlock(48, 64, momentum)
        self.conv_block3 = ConvBlock(64, 96, momentum)
        self.conv_block4 = ConvBlock(96, 128, momentum)
        midfeat = 1792  # 128 × 14
        fc_dim = 768
        self.shared_fc = nn.Linear(midfeat, fc_dim, bias=False)
        self.shared_bn = nn.BatchNorm1d(fc_dim, momentum=momentum)
        nn.init.xavier_uniform_(self.shared_fc.weight)
        self.shared_bn.bias.data.fill_(0.)
        self.shared_bn.weight.data.fill_(1.)

        # --- Per-expert GRU branches ---
        self.grus = nn.ModuleList()
        self._gru_out_dims = []
        for hidden, num_layers, bidirectional, _ in self._GRU_DEFS:
            gru = nn.GRU(fc_dim, hidden, num_layers=num_layers,
                         batch_first=True, bidirectional=bidirectional)
            init_gru(gru)
            self.grus.append(gru)
            self._gru_out_dims.append(hidden * (2 if bidirectional else 1))

        # Project each GRU output to P for gate input
        self.gate_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, P, bias=False), nn.LayerNorm(P), nn.GELU())
            for d in self._gru_out_dims
        ])

        # Cue projections
        cue_total = 3 * C if self.use_cues else 0
        if self.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        # Per-expert input dims: full GRU output + cues
        _exp_in = {i: self._gru_out_dims[i] + cue_total for i in range(N_temporal)}

        gate_in = N_temporal * P + cue_total

        if self.use_spectral:
            self.spectral_cnn = _DedicatedSpectralCNN(out_dim=SP)
            self.feat_proj = nn.Sequential(
                nn.Linear(self._gru_out_dims[1], P, bias=False),
                nn.LayerNorm(P),
                nn.GELU(),
            )
            _exp_in[spectral_idx] = SP + P + cue_total
            gate_in += SP + P

        # Expert MLPs (active experts only)
        experts = []
        for eid in self.active_expert_ids:
            experts.append(_Expert(_exp_in[eid], cfg.expert_hidden,
                                   cfg.shared_dim, cfg.expert_dropout))
        self.experts = nn.ModuleList(experts)
        num_experts = len(experts)

        # Per-task gates
        gate_hidden = cfg.expert_hidden // 2
        self.gate_tonal = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_artic = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)
        self.gate_legato = _Gate(gate_in, num_experts, gate_hidden, cfg.gate_dropout)

        # Classification heads
        self.ln_tonal = nn.LayerNorm(cfg.shared_dim)
        self.ln_artic = nn.LayerNorm(cfg.shared_dim)
        self.ln_legato = nn.LayerNorm(cfg.shared_dim)
        self.tonal_head = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
        self.legato_head = nn.Linear(cfg.shared_dim, 1)

        # Per-expert auxiliary heads
        self.aux_tonal_heads = nn.ModuleList([
            nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
            for _ in range(num_experts)
        ])
        self.aux_artic_heads = nn.ModuleList([
            nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
            for _ in range(num_experts)
        ])
        self.aux_legato_heads = nn.ModuleList([
            nn.Linear(cfg.shared_dim, 1)
            for _ in range(num_experts)
        ])

    def _get_gate_probs(self, gate_module, x):
        logits = gate_module(x)
        num_e = logits.shape[-1]
        if self.cfg.top_k > 0 and self.cfg.top_k < num_e:
            return self._top_k_gating(logits, self.cfg.top_k)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def _top_k_gating(logits, k):
        top_vals, top_idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        """
        Args:
            features:    IGNORED (None in multi-CRNN mode; kept for API compat)
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            logmel:      (B, 1, T, 229) BN-normalised log-mel spectrogram
        """
        assert logmel is not None, "FrameMultiCRNNMoEHead requires logmel input"

        # --- Shared CNN ---
        x = self.conv_block1(logmel, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # (B, 128, T, 14) → (B, T, 1792)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.shared_bn(self.shared_fc(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)
        # x: (B, T, 768)

        # --- Per-expert GRU branches ---
        gru_outputs = []
        for gru in self.grus:
            h, _ = gru(x)
            h = F.dropout(h, p=0.5, training=self.training)
            gru_outputs.append(h)
        T_feat = gru_outputs[0].shape[1]

        # Gate projections (all GRU outputs, even masked ones, feed the gate)
        gate_parts = [proj(g) for proj, g in zip(self.gate_projections, gru_outputs)]

        # Cue embeddings
        cue_cat = None
        if self.use_cues:
            c_onset = F.gelu(self.onset_proj(onset_prob))
            c_offset = F.gelu(self.offset_proj(offset_prob))
            c_frame = F.gelu(self.frame_proj(frame_prob))
            cue_cat = torch.cat([c_onset, c_offset, c_frame], dim=-1)

        # Spectral features (optional)
        N_temporal = len(self._GRU_DEFS)
        s_spec = f_proj = None
        if self.use_spectral:
            s_spec = self.spectral_cnn(logmel)
            T_spec = s_spec.shape[1]
            if T_spec != T_feat:
                s_spec = s_spec[:, :T_feat, :] if T_spec > T_feat else \
                    F.pad(s_spec, (0, 0, 0, T_feat - T_spec))
            f_proj = self.feat_proj(gru_outputs[1])

        # Build expert inputs for active experts
        expert_list = []
        for idx, eid in enumerate(self.active_expert_ids):
            if eid < N_temporal:
                parts = [gru_outputs[eid]]
                if cue_cat is not None:
                    parts.append(cue_cat)
                inp = torch.cat(parts, dim=-1)
            else:
                parts = [s_spec, f_proj]
                if cue_cat is not None:
                    parts.append(cue_cat)
                inp = torch.cat(parts, dim=-1)
            expert_list.append(self.experts[idx](inp))

        expert_outs = torch.stack(expert_list, dim=-2)  # (B, T, E, shared_dim)

        # Gate input
        gate_input_parts = list(gate_parts)
        if self.use_spectral:
            gate_input_parts.extend([s_spec, f_proj])
        if cue_cat is not None:
            gate_input_parts.append(cue_cat)
        gate_in = torch.cat(gate_input_parts, dim=-1)

        gp_tonal = self._get_gate_probs(self.gate_tonal, gate_in)
        gp_artic = self._get_gate_probs(self.gate_artic, gate_in)
        gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
        artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
        legato_prob = torch.sigmoid(self.legato_head(_mix(gp_legato, self.ln_legato)))

        # Per-expert auxiliary logits
        aux_tonal = [h(expert_list[i]) for i, h in enumerate(self.aux_tonal_heads)]
        aux_artic = [h(expert_list[i]) for i, h in enumerate(self.aux_artic_heads)]
        aux_legato = [torch.sigmoid(h(expert_list[i]))
                      for i, h in enumerate(self.aux_legato_heads)]

        return (tonal_logits, artic_logits, legato_prob,
                gp_tonal, gp_artic, gp_legato,
                aux_tonal, aux_artic, aux_legato)


class FrameDirectHead(nn.Module):
    """Direct CRNN features → classification heads, no MoE routing.

    Drop-in replacement for FrameMultiScaleMoEHead with the same forward
    signature and return format, so the rest of the pipeline is unchanged.
    Used to ablate whether temporal-expert MoE adds value over a plain MLP.
    """

    def __init__(self, cfg: FrameMultiScaleMoEConfig = None):
        super().__init__()
        cfg = cfg or FrameMultiScaleMoEConfig()
        D = cfg.feat_dim
        H = cfg.expert_hidden
        S = cfg.shared_dim
        C = cfg.cue_dim
        self.use_cues = cfg.use_cues

        cue_total = 3 * C if self.use_cues else 0
        if self.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        self.mlp = nn.Sequential(
            nn.Linear(D + cue_total, H),
            nn.GELU(),
            nn.Dropout(cfg.expert_dropout),
            nn.Linear(H, S),
        )

        self.ln_tonal = nn.LayerNorm(S)
        self.ln_artic = nn.LayerNorm(S)
        self.ln_legato = nn.LayerNorm(S)
        self.tonal_head = nn.Linear(S, cfg.num_tonal_classes)
        self.artic_head = nn.Linear(S, cfg.num_artic_classes)
        self.legato_head = nn.Linear(S, 1)

    def forward(self, features, onset_prob, offset_prob, frame_prob,
                logmel=None, conv2d_map=None):
        parts = [features]
        if self.use_cues:
            parts.append(F.gelu(self.onset_proj(onset_prob)))
            parts.append(F.gelu(self.offset_proj(offset_prob)))
            parts.append(F.gelu(self.frame_proj(frame_prob)))
        h = self.mlp(torch.cat(parts, dim=-1))

        tonal_logits = self.tonal_head(F.relu(self.ln_tonal(h)))
        artic_logits = self.artic_head(F.relu(self.ln_artic(h)))
        legato_prob = torch.sigmoid(self.legato_head(F.relu(self.ln_legato(h))))

        return (tonal_logits, artic_logits, legato_prob,
                None, None, None,
                [], [], [])


# ---------------------------------------------------------------------------
# Losses — frame-level with active-mask
# ---------------------------------------------------------------------------
# Legacy fixed names (used when temporal_experts is not specified)
EXPERT_NAMES = {0: 'Onset', 1: 'Note', 2: 'Phrase', 3: 'Spectral'}


def make_expert_names(temporal_defs, use_spectral=False):
    """Generate expert name dict from temporal definitions."""
    names = {}
    for i, (k, d) in enumerate(temporal_defs):
        eff_ms = k * d * 10
        names[i] = f'T{i}({k}x{d}={eff_ms}ms)'
    if use_spectral:
        names[len(temporal_defs)] = 'Spectral'
    return names


def frame_moe_technique_losses(output_dict, target_dict, device=None, focal_gamma=0.0):
    """Frame-level technique losses for the multi-scale MoE.

    Output keys: fmoe_tonal_logits, fmoe_artic_logits, fmoe_legato_prob,
                 fmoe_gate_tonal, fmoe_gate_artic, fmoe_gate_legato
    Target keys: tonal_technique (B, T), articulation (B, T),
                 legato (B, T) — soft float [0,1], 1.0=bow_change peak
    """
    dev = device

    # Active-frame mask: only compute loss on frames where a note is active
    active_mask = None
    if 'frame_roll' in target_dict:
        active_mask = (target_dict['frame_roll'].sum(dim=-1) > 0)  # (B, T)

    def _frame_ce(logits_key, target_key):
        if logits_key not in output_dict or target_key not in target_dict:
            return torch.tensor(0.0, device=dev)
        logits = output_dict[logits_key]
        targets = target_dict[target_key].long()
        T_pred, T_tgt = logits.shape[1], targets.shape[1]
        T = min(T_pred, T_tgt)
        logits = logits[:, :T, :]
        targets = targets[:, :T]
        logits_2d = logits.reshape(-1, logits.shape[-1])
        targets_1d = targets.reshape(-1)
        if active_mask is not None:
            m = active_mask[:, :T].reshape(-1)
            if m.any():
                logits_2d = logits_2d[m]
                targets_1d = targets_1d[m]
            else:
                return torch.tensor(0.0, device=dev)
        if focal_gamma > 0:
            from losses import focal_cross_entropy
            return focal_cross_entropy(logits_2d, targets_1d, gamma=focal_gamma)
        return F.cross_entropy(logits_2d, targets_1d)

    loss_tonal = _frame_ce('fmoe_tonal_logits', 'tonal_technique')
    loss_artic = _frame_ce('fmoe_artic_logits', 'articulation')

    # Legato — positive-weighted BCE on soft bow_change regression target
    # Target is a Gaussian-peaked float in [0,1]: 1.0 = bow_change, 0.0 = sustained
    loss_legato = torch.tensor(0.0, device=dev)
    if 'fmoe_legato_prob' in output_dict and 'legato' in target_dict:
        pred = output_dict['fmoe_legato_prob'].squeeze(-1)
        tgt = target_dict['legato'].float()
        T = min(pred.shape[1], tgt.shape[1])
        pred = pred[:, :T]
        tgt = tgt[:, :T]

        def _weighted_bce(p, t):
            bce = F.binary_cross_entropy(p, t, reduction='none')
            w = torch.where(t > 0.1, 5.0, 1.0)
            return (bce * w).mean()

        if active_mask is not None:
            m = active_mask[:, :T]
            if m.any():
                loss_legato = _weighted_bce(pred[m], tgt[m])
        else:
            loss_legato = _weighted_bce(pred.reshape(-1), tgt.reshape(-1))

    # Per-task balance losses
    bal_tonal = _frame_balance_loss(output_dict.get('fmoe_gate_tonal'), active_mask, dev)
    bal_artic = _frame_balance_loss(output_dict.get('fmoe_gate_artic'), active_mask, dev)
    bal_legato = _frame_balance_loss(output_dict.get('fmoe_gate_legato'), active_mask, dev)
    loss_balance = (bal_tonal + bal_artic + bal_legato) / 3.0

    # Per-expert auxiliary loss (each expert independently predicts all tasks)
    loss_aux = torch.tensor(0.0, device=dev)
    aux_tonal_list = output_dict.get('fmoe_aux_tonal')
    aux_artic_list = output_dict.get('fmoe_aux_artic')
    aux_legato_list = output_dict.get('fmoe_aux_legato')
    if aux_tonal_list and 'tonal_technique' in target_dict:
        E = len(aux_tonal_list)
        expert_losses = []
        for e in range(E):
            # Tonal CE
            l_t = _aux_frame_ce(aux_tonal_list[e], target_dict, 'tonal_technique',
                                active_mask, dev, focal_gamma)
            # Articulation CE
            l_a = _aux_frame_ce(aux_artic_list[e], target_dict, 'articulation',
                                active_mask, dev, focal_gamma)
            # Legato BCE
            l_l = _aux_legato_bce(aux_legato_list[e], target_dict, active_mask, dev)
            expert_losses.append(l_t + l_a + l_l)
        loss_aux = sum(expert_losses) / E

    return loss_tonal, loss_artic, loss_legato, loss_balance, loss_aux


def _aux_frame_ce(logits, target_dict, target_key, active_mask, dev, focal_gamma):
    """CE loss for a single expert's auxiliary head."""
    if target_key not in target_dict:
        return torch.tensor(0.0, device=dev)
    targets = target_dict[target_key].long()
    T = min(logits.shape[1], targets.shape[1])
    logits_2d = logits[:, :T, :].reshape(-1, logits.shape[-1])
    targets_1d = targets[:, :T].reshape(-1)
    if active_mask is not None:
        m = active_mask[:, :T].reshape(-1)
        if m.any():
            logits_2d = logits_2d[m]
            targets_1d = targets_1d[m]
        else:
            return torch.tensor(0.0, device=dev)
    return F.cross_entropy(logits_2d, targets_1d)


def _aux_legato_bce(pred_prob, target_dict, active_mask, dev):
    """Weighted BCE loss for a single expert's auxiliary legato head."""
    if 'legato' not in target_dict:
        return torch.tensor(0.0, device=dev)
    pred = pred_prob.squeeze(-1)
    tgt = target_dict['legato'].float()
    T = min(pred.shape[1], tgt.shape[1])
    pred = pred[:, :T]
    tgt = tgt[:, :T]
    if active_mask is not None:
        m = active_mask[:, :T]
        if not m.any():
            return torch.tensor(0.0, device=dev)
        pred = pred[m]
        tgt = tgt[m]
    else:
        pred = pred.reshape(-1)
        tgt = tgt.reshape(-1)
    bce = F.binary_cross_entropy(pred, tgt, reduction='none')
    w = torch.where(tgt > 0.1, 5.0, 1.0)
    return (bce * w).mean()


def _frame_balance_loss(gate_probs, active_mask, device):
    """Load-balance loss over active frames."""
    if gate_probs is None:
        return torch.tensor(0.0, device=device)
    if active_mask is not None:
        T = min(gate_probs.shape[1], active_mask.shape[1])
        masked = gate_probs[:, :T][active_mask[:, :T]]
    else:
        masked = gate_probs.reshape(-1, gate_probs.shape[-1])
    if masked.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    E = masked.shape[1]
    P = masked.mean(dim=0)
    return E * (P * P).sum()


# ---------------------------------------------------------------------------
# Binary technique losses
# ---------------------------------------------------------------------------

_BINARY_POS_WEIGHTS = {
    'pizzicato': 5.0,
    'harmonic': 5.0,
    'spiccato': 5.0,
    'legato': 5.0,
}


def frame_moe_binary_technique_losses(output_dict, target_dict, device=None):
    """Frame-level binary technique losses for the 4-flag decomposition.

    Target keys: pizzicato (B,T), harmonic (B,T), spiccato (B,T),
                 legato (B,T) — all float [0,1].
    Output keys: fmoe_binary_probs  {name: (B,T,1)},
                 fmoe_binary_gates  {name: (B,T,E)}.

    Returns (per_flag_losses: dict, loss_balance, loss_aux).
    """
    dev = device

    active_mask = None
    if 'frame_roll' in target_dict:
        active_mask = (target_dict['frame_roll'].sum(dim=-1) > 0)

    probs_dict = output_dict.get('fmoe_binary_probs', {})
    gates_dict = output_dict.get('fmoe_binary_gates', {})
    aux_dict = output_dict.get('fmoe_binary_aux', {})

    flag_names = list(probs_dict.keys())
    per_flag_losses = {}

    for name in flag_names:
        tgt_key = name
        if tgt_key not in target_dict:
            per_flag_losses[name] = torch.tensor(0.0, device=dev)
            continue

        pred = probs_dict[name].squeeze(-1)
        tgt = target_dict[tgt_key].float()
        T = min(pred.shape[1], tgt.shape[1])
        pred = pred[:, :T]
        tgt = tgt[:, :T]

        pw = _BINARY_POS_WEIGHTS.get(name, 5.0)

        if active_mask is not None:
            m = active_mask[:, :T]
            if m.any():
                bce = F.binary_cross_entropy(pred[m], tgt[m], reduction='none')
                w = torch.where(tgt[m] > 0.1, pw, 1.0)
                per_flag_losses[name] = (bce * w).mean()
            else:
                per_flag_losses[name] = torch.tensor(0.0, device=dev)
        else:
            bce = F.binary_cross_entropy(
                pred.reshape(-1), tgt.reshape(-1), reduction='none')
            w = torch.where(tgt.reshape(-1) > 0.1, pw, 1.0)
            per_flag_losses[name] = (bce * w).mean()

    # Balance losses across all gates
    bal_losses = []
    for name in flag_names:
        gp = gates_dict.get(name)
        bal_losses.append(_frame_balance_loss(gp, active_mask, dev))
    loss_balance = sum(bal_losses) / max(len(bal_losses), 1)

    # Per-expert auxiliary loss
    loss_aux = torch.tensor(0.0, device=dev)
    if aux_dict:
        first_key = next(iter(aux_dict))
        E = len(aux_dict[first_key])
        expert_losses = []
        for e in range(E):
            el = torch.tensor(0.0, device=dev)
            for name in flag_names:
                tgt_key = name
                if tgt_key not in target_dict or name not in aux_dict:
                    continue
                pred_e = aux_dict[name][e].squeeze(-1)
                tgt = target_dict[tgt_key].float()
                T = min(pred_e.shape[1], tgt.shape[1])
                pred_e = pred_e[:, :T]
                tgt = tgt[:, :T]
                if active_mask is not None:
                    m = active_mask[:, :T]
                    if not m.any():
                        continue
                    pred_e = pred_e[m]
                    tgt = tgt[m]
                else:
                    pred_e = pred_e.reshape(-1)
                    tgt = tgt.reshape(-1)
                pw = _BINARY_POS_WEIGHTS.get(name, 5.0)
                bce = F.binary_cross_entropy(pred_e, tgt, reduction='none')
                w = torch.where(tgt > 0.1, pw, 1.0)
                el = el + (bce * w).mean()
            expert_losses.append(el)
        loss_aux = sum(expert_losses) / E

    return per_flag_losses, loss_balance, loss_aux
