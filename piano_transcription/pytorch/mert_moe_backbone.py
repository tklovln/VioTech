"""MERT Multi-Scale MoE Backbone for violin technique classification.

Integrates frozen MERT-v1-330M with per-expert layer weighting, a temporal
pyramid of depthwise-separable Conv1D + per-expert GRU, and per-task gating.

Three levels of interpretability:
  1. Per-expert MERT layer weights  →  which pre-trained features each expert uses
  2. Per-expert temporal scale      →  which time resolution each expert operates at
  3. Per-task gate routing           →  which experts matter for each technique

Architecture per expert:
  MERT hidden states → learnable weighted sum → proj(1024→P) → interpolate →
  DepthSepConv1D(kernel=2^(i+1)) → GRU(P→H, bidirectional) → Expert MLP

Default 8-expert pyramid: k=2,4,8,16,32,64,128,256 @ 100fps
  = 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1280ms, 2560ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from dataclasses import dataclass
from typing import Optional, Tuple

from moe_frame_multiscale import _TemporalCtx, _Expert, _Gate


_DEFAULT_MODEL = "m-a-p/MERT-v1-330M"
_MERT_SR = 24_000


@dataclass
class MERTMoEConfig:
    """Configuration for MERT Multi-Scale MoE Backbone."""
    model_name: str = _DEFAULT_MODEL
    input_sr: int = 16_000
    target_fps: int = 100
    freeze_mert: bool = True
    temporal_experts: Tuple[Tuple[int, int], ...] = (
        (2, 1), (4, 1), (8, 1), (16, 1),
        (32, 1), (64, 1), (128, 1), (256, 1),
    )
    proj_dim: int = 128
    gru_hidden: int = 128
    gate_proj_dim: int = 64
    expert_hidden: int = 256
    shared_dim: int = 128
    cue_dim: int = 16
    gate_dropout: float = 0.0
    expert_dropout: float = 0.2
    balance_loss_coeff: float = 0.001
    top_k: int = 0
    use_cues: bool = True
    num_tonal_classes: int = 4
    num_artic_classes: int = 4
    expert_mask: Optional[Tuple[int, ...]] = None
    binary_technique: bool = False
    hierarchical: bool = False
    layer_init_sigma: float = 4.0


class MERTMoEBackbone(nn.Module):
    """MERT Multi-Scale MoE with per-expert layer weighting.

    Forward returns the same format as FrameMultiScaleMoEHead:
      Multiclass: (tonal_logits, artic_logits, legato_prob,
                   gp_tonal, gp_artic, gp_legato,
                   aux_tonal, aux_artic, aux_legato)
      Binary:     (probs_dict, gate_probs_dict, aux_dict)
    """

    def __init__(self, cfg: MERTMoEConfig = None):
        super().__init__()
        cfg = cfg or MERTMoEConfig()
        self.cfg = cfg

        temporal_defs = tuple(cfg.temporal_experts)
        N = len(temporal_defs)
        self.n_experts_total = N

        if cfg.expert_mask is not None:
            self.active_expert_ids = list(sorted(cfg.expert_mask))
        else:
            self.active_expert_ids = list(range(N))

        self._expert_names = {}
        for i, (k, d) in enumerate(temporal_defs):
            eff_ms = k * d * 10
            self._expert_names[i] = f'M{i}(k={k},d={d},{eff_ms}ms)'

        P = cfg.proj_dim
        H = cfg.gru_hidden
        GP = cfg.gate_proj_dim
        C = cfg.cue_dim
        gru_out = H * 2  # bidirectional
        cue_total = 3 * C if cfg.use_cues else 0

        # ---------- Frozen MERT ----------
        self.resampler = T.Resample(cfg.input_sr, _MERT_SR)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.model_name, trust_remote_code=True)
        self.mert = AutoModel.from_pretrained(
            cfg.model_name, trust_remote_code=True, use_safetensors=True)
        if cfg.freeze_mert:
            for p in self.mert.parameters():
                p.requires_grad = False
            self.mert.eval()
        hidden_dim = self.mert.config.hidden_size
        num_layers = self.mert.config.num_hidden_layers + 1
        self._freeze_mert = cfg.freeze_mert
        self._num_mert_layers = num_layers

        # ---------- Per-expert MERT layer weights ----------
        self.layer_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(num_layers)) for _ in range(N)
        ])
        self._init_layer_weights(num_layers, N, cfg.layer_init_sigma)

        # ---------- Per-expert projection (1024 → P) ----------
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, P, bias=False), nn.LayerNorm(P))
            for _ in range(N)
        ])

        # ---------- Per-expert temporal Conv1D ----------
        self.temporal_ctxs = nn.ModuleList([
            _TemporalCtx(P, P, k, d) for k, d in temporal_defs
        ])

        # ---------- Per-expert GRU ----------
        self.grus = nn.ModuleList([
            nn.GRU(P, H, num_layers=1, batch_first=True, bidirectional=True)
            for _ in range(N)
        ])
        for gru in self.grus:
            _init_gru(gru)

        # ---------- Per-expert gate projection ----------
        self.gate_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(gru_out, GP, bias=False),
                          nn.LayerNorm(GP), nn.GELU())
            for _ in range(N)
        ])

        # ---------- Cue projections ----------
        if cfg.use_cues:
            self.onset_proj = nn.Linear(1, C)
            self.offset_proj = nn.Linear(1, C)
            self.frame_proj = nn.Linear(1, C)

        # ---------- Expert MLPs (active experts only) ----------
        expert_in = gru_out + cue_total
        experts = []
        for _ in self.active_expert_ids:
            experts.append(_Expert(expert_in, cfg.expert_hidden,
                                   cfg.shared_dim, cfg.expert_dropout))
        self.experts = nn.ModuleList(experts)
        num_active = len(experts)

        # ---------- Gates & heads ----------
        gate_in = N * GP + cue_total
        gate_hidden = cfg.expert_hidden // 2
        self._binary_mode = cfg.binary_technique
        self._hierarchical = cfg.hierarchical

        if cfg.binary_technique:
            self._binary_flag_names = ['pizzicato', 'harmonic', 'spiccato', 'legato']
            self.gates = nn.ModuleDict({
                name: _Gate(gate_in, num_active, gate_hidden, cfg.gate_dropout)
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
                    nn.Linear(cfg.shared_dim, 1) for _ in range(num_active)
                ]) for name in self._binary_flag_names
            })
        elif cfg.hierarchical:
            S = cfg.shared_dim
            Hh = cfg.expert_hidden // 2
            K_tonal = cfg.num_tonal_classes - 1
            K_artic = cfg.num_artic_classes - 1
            self._tonal_none_id = 0
            self._artic_none_id = 0

            self.gate_tonal = _Gate(gate_in, num_active, gate_hidden,
                                    cfg.gate_dropout)
            self.gate_artic = _Gate(gate_in, num_active, gate_hidden,
                                    cfg.gate_dropout)
            self.gate_legato = _Gate(gate_in, num_active, gate_hidden,
                                     cfg.gate_dropout)
            self.ln_tonal = nn.LayerNorm(S)
            self.ln_artic = nn.LayerNorm(S)
            self.ln_legato = nn.LayerNorm(S)

            self.tonal_detector = nn.Sequential(
                nn.Linear(S, Hh), nn.GELU(), nn.Dropout(cfg.expert_dropout),
                nn.Linear(Hh, 1))
            self.tonal_classifier = nn.Sequential(
                nn.Linear(S + 1, Hh), nn.GELU(), nn.Dropout(cfg.expert_dropout),
                nn.Linear(Hh, K_tonal))
            self.artic_detector = nn.Sequential(
                nn.Linear(S, Hh), nn.GELU(), nn.Dropout(cfg.expert_dropout),
                nn.Linear(Hh, 1))
            self.artic_classifier = nn.Sequential(
                nn.Linear(S + 1, Hh), nn.GELU(), nn.Dropout(cfg.expert_dropout),
                nn.Linear(Hh, K_artic))
            self.legato_head = nn.Sequential(
                nn.Linear(S, Hh), nn.GELU(), nn.Dropout(cfg.expert_dropout),
                nn.Linear(Hh, 1))

            self.aux_tonal_heads = nn.ModuleList([
                nn.Linear(S, cfg.num_tonal_classes) for _ in range(num_active)])
            self.aux_artic_heads = nn.ModuleList([
                nn.Linear(S, cfg.num_artic_classes) for _ in range(num_active)])
            self.aux_legato_heads = nn.ModuleList([
                nn.Linear(S, 1) for _ in range(num_active)])
        else:
            self.gate_tonal = _Gate(gate_in, num_active, gate_hidden,
                                    cfg.gate_dropout)
            self.gate_artic = _Gate(gate_in, num_active, gate_hidden,
                                    cfg.gate_dropout)
            self.gate_legato = _Gate(gate_in, num_active, gate_hidden,
                                     cfg.gate_dropout)
            self.ln_tonal = nn.LayerNorm(cfg.shared_dim)
            self.ln_artic = nn.LayerNorm(cfg.shared_dim)
            self.ln_legato = nn.LayerNorm(cfg.shared_dim)
            self.tonal_head = nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
            self.artic_head = nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
            self.legato_head = nn.Linear(cfg.shared_dim, 1)
            self.aux_tonal_heads = nn.ModuleList([
                nn.Linear(cfg.shared_dim, cfg.num_tonal_classes)
                for _ in range(num_active)
            ])
            self.aux_artic_heads = nn.ModuleList([
                nn.Linear(cfg.shared_dim, cfg.num_artic_classes)
                for _ in range(num_active)
            ])
            self.aux_legato_heads = nn.ModuleList([
                nn.Linear(cfg.shared_dim, 1)
                for _ in range(num_active)
            ])

    # ------------------------------------------------------------------
    def _init_layer_weights(self, num_layers, num_experts, sigma):
        """Gaussian-biased initialization so each expert starts attending to
        different MERT layers (low→acoustic, high→semantic)."""
        for i in range(num_experts):
            center = i * (num_layers - 1) / max(num_experts - 1, 1)
            x = torch.arange(num_layers, dtype=torch.float32)
            init = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
            self.layer_weights[i].data = torch.log(init + 1e-8)

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_mert:
            self.mert.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _preprocess(self, waveform_24k: torch.Tensor) -> torch.Tensor:
        device = waveform_24k.device
        batch = []
        for w in waveform_24k:
            out = self.processor(
                w.cpu().numpy(), sampling_rate=_MERT_SR,
                return_tensors="pt")
            batch.append(out.input_values.squeeze(0))
        return torch.stack(batch).to(device)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(self, waveform, onset_prob, offset_prob, frame_prob,
                target_T=None, **kwargs):
        """
        Args:
            waveform:    (B, data_length) raw audio at input_sr Hz
            onset_prob:  (B, T, 1) max-pooled onset probability
            offset_prob: (B, T, 1) max-pooled offset probability
            frame_prob:  (B, T, 1) max-pooled frame activity probability
            target_T:    exact number of output frames (from transcription branch)
        """
        if target_T is None:
            target_T = round(waveform.shape[1] / self.cfg.input_sr
                             * self.cfg.target_fps)

        # 1) Resample + preprocess + frozen MERT
        wav_24k = self.resampler(waveform)
        wav_24k = self._preprocess(wav_24k)
        with torch.no_grad():
            outputs = self.mert(wav_24k, output_hidden_states=True)
        all_hidden = torch.stack(outputs.hidden_states)  # (L, B, T_mert, D)

        # 2) Per-expert: weighted sum → proj → interpolate → Conv1D → GRU
        gru_outputs = []
        gate_parts = []
        for i in range(self.n_experts_total):
            w = F.softmax(self.layer_weights[i], dim=0)
            feat = torch.einsum("l b t d, l -> b t d", all_hidden, w)
            feat = self.projections[i](feat)
            if feat.shape[1] != target_T:
                feat = F.interpolate(
                    feat.permute(0, 2, 1), size=target_T,
                    mode="linear", align_corners=False
                ).permute(0, 2, 1)
            h = self.temporal_ctxs[i](feat)
            h, _ = self.grus[i](h)
            h = F.dropout(h, p=0.3, training=self.training)
            gru_outputs.append(h)
            gate_parts.append(self.gate_projections[i](h))

        # 3) Cue embeddings — align T to match Conv1D output
        #    Even-sized kernels in _TemporalCtx drop 1 frame; trim cues to match.
        T_feat = gru_outputs[0].shape[1]
        cue_cat = None
        if self.cfg.use_cues:
            o = onset_prob[:, :T_feat, :]
            off = offset_prob[:, :T_feat, :]
            fr = frame_prob[:, :T_feat, :]
            c_onset = F.gelu(self.onset_proj(o))
            c_offset = F.gelu(self.offset_proj(off))
            c_frame = F.gelu(self.frame_proj(fr))
            cue_cat = torch.cat([c_onset, c_offset, c_frame], dim=-1)

        # 4) Expert MLPs (active experts)
        expert_list = []
        for idx, eid in enumerate(self.active_expert_ids):
            parts = [gru_outputs[eid]]
            if cue_cat is not None:
                parts.append(cue_cat)
            expert_list.append(self.experts[idx](torch.cat(parts, dim=-1)))

        expert_outs = torch.stack(expert_list, dim=-2)  # (B, T, E, shared_dim)

        # 5) Gate input
        gate_input_parts = list(gate_parts)
        if cue_cat is not None:
            gate_input_parts.append(cue_cat)
        gate_in = torch.cat(gate_input_parts, dim=-1)

        # 6) Gating + classification
        def _mix(gp, ln):
            h = (gp.unsqueeze(-1) * expert_outs).sum(dim=-2)
            return F.relu(ln(h))

        if self._binary_mode:
            gate_probs = {}
            probs = {}
            aux = {}
            for name in self._binary_flag_names:
                gp = self._get_gate_probs(self.gates[name], gate_in)
                gate_probs[name] = gp
                probs[name] = torch.sigmoid(
                    self.heads[name](_mix(gp, self.layer_norms[name])))
                aux[name] = [
                    torch.sigmoid(h(expert_list[i]))
                    for i, h in enumerate(self.aux_heads[name])
                ]
            return probs, gate_probs, aux

        gp_tonal = self._get_gate_probs(self.gate_tonal, gate_in)
        gp_artic = self._get_gate_probs(self.gate_artic, gate_in)
        gp_legato = self._get_gate_probs(self.gate_legato, gate_in)

        if self._hierarchical:
            mixed_tonal = _mix(gp_tonal, self.ln_tonal)
            mixed_artic = _mix(gp_artic, self.ln_artic)
            mixed_legato = _mix(gp_legato, self.ln_legato)

            det_tonal = self.tonal_detector(mixed_tonal)
            p_tonal = torch.sigmoid(det_tonal)
            cls_tonal = self.tonal_classifier(
                torch.cat([mixed_tonal, p_tonal], dim=-1))
            tonal_logits = _compose_hierarchical_log_probs(
                det_tonal, cls_tonal,
                self.cfg.num_tonal_classes, self._tonal_none_id)

            det_artic = self.artic_detector(mixed_artic)
            p_artic = torch.sigmoid(det_artic)
            cls_artic = self.artic_classifier(
                torch.cat([mixed_artic, p_artic], dim=-1))
            artic_logits = _compose_hierarchical_log_probs(
                det_artic, cls_artic,
                self.cfg.num_artic_classes, self._artic_none_id)

            legato_prob = torch.sigmoid(self.legato_head(mixed_legato))
        else:
            tonal_logits = self.tonal_head(_mix(gp_tonal, self.ln_tonal))
            artic_logits = self.artic_head(_mix(gp_artic, self.ln_artic))
            legato_prob = torch.sigmoid(
                self.legato_head(_mix(gp_legato, self.ln_legato)))

        aux_tonal = [h(expert_list[i])
                     for i, h in enumerate(self.aux_tonal_heads)]
        aux_artic = [h(expert_list[i])
                     for i, h in enumerate(self.aux_artic_heads)]
        aux_legato = [torch.sigmoid(h(expert_list[i]))
                      for i, h in enumerate(self.aux_legato_heads)]

        return (tonal_logits, artic_logits, legato_prob,
                gp_tonal, gp_artic, gp_legato,
                aux_tonal, aux_artic, aux_legato)


def _compose_hierarchical_log_probs(det_logit, cls_logits, num_classes, none_id):
    """Numerically stable hierarchical log-probability composition.

    log P(none)    = logsigmoid(-d)
    log P(class_k) = logsigmoid(d) + log_softmax(c)[k]
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


def _init_gru(gru):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            param.data.fill_(0.)
