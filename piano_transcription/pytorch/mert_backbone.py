"""Frozen MERT-v1-330M backbone for technique feature extraction.

Replaces AcousticModelCRnn8Dropout as the feature extractor feeding
the FrameMultiScaleMoEHead.  MERT parameters are frozen; only the
learnable layer-weight aggregation and the output projection are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor


_DEFAULT_MODEL = "m-a-p/MERT-v1-330M"
_MERT_SR = 24_000
_MERT_FEATURE_RATE = 75  # Hz


class MERTBackbone(nn.Module):
    """Frozen MERT feature extractor with learnable layer weighting.

    Takes raw waveform at ``input_sr`` (default 16 kHz), resamples to 24 kHz,
    extracts all hidden states from the frozen MERT transformer, computes a
    learnable softmax-weighted sum, projects to ``proj_dim``, and interpolates
    the frame axis to ``target_fps``.

    The output is ``(B, T, proj_dim)`` where *T* matches the transcription
    system's frame count (100 fps by default).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        proj_dim: int = 512,
        input_sr: int = 16_000,
        target_fps: int = 100,
        freeze: bool = True,
    ):
        super().__init__()
        self.input_sr = input_sr
        self.target_fps = target_fps

        # Compatibility attributes expected by models_contrast.py forward path
        self.output_features = True
        self.output_conv2d_map = False

        # Resampler 16 kHz → 24 kHz
        self.resampler = T.Resample(input_sr, _MERT_SR)

        # HuggingFace preprocessor (normalisation stats)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True,
        )

        # MERT transformer (use_safetensors avoids torch.load CVE-2025-32434 check)
        self.mert = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, use_safetensors=True,
        )
        if freeze:
            for p in self.mert.parameters():
                p.requires_grad = False
            self.mert.eval()

        hidden_dim = self.mert.config.hidden_size          # 1024
        num_layers = self.mert.config.num_hidden_layers + 1  # 25 (incl. embedding)

        # Learnable weighted average over transformer layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Projection to match FrameMultiScaleMoEHead.feat_dim
        self.proj = nn.Linear(hidden_dim, proj_dim)

        self._freeze = freeze

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Keep the frozen MERT in eval mode regardless of model.train()."""
        super().train(mode)
        if self._freeze:
            self.mert.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _preprocess(self, waveform_24k: torch.Tensor) -> torch.Tensor:
        """Apply the Wav2Vec2FeatureExtractor normalisation on-device.

        ``waveform_24k``: already resampled to 24 kHz, shape ``(B, S)``.
        Returns ``(B, S)`` normalised tensor on the same device.
        """
        device = waveform_24k.device
        batch = []
        for w in waveform_24k:
            out = self.processor(
                w.cpu().numpy(),
                sampling_rate=_MERT_SR,
                return_tensors="pt",
            )
            batch.append(out.input_values.squeeze(0))
        return torch.stack(batch).to(device)

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor, target_T: int | None = None) -> torch.Tensor:
        """
        Args:
            waveform: ``(B, data_length)`` raw audio at ``input_sr`` Hz.
            target_T: exact number of output frames (from the transcription
                branch).  When *None*, estimated from waveform length.

        Returns:
            features: ``(B, target_T, proj_dim)``.
        """
        if target_T is None:
            target_T = round(waveform.shape[1] / self.input_sr * self.target_fps)

        # 1) Resample 16 kHz → 24 kHz
        wav_24k = self.resampler(waveform)               # (B, S_24k)

        # 2) Processor normalisation
        wav_24k = self._preprocess(wav_24k)               # (B, S_24k)

        # 3) Run frozen MERT
        with torch.no_grad():
            outputs = self.mert(wav_24k, output_hidden_states=True)
        # outputs.hidden_states: tuple of (B, T_mert, 1024) × 25
        all_hidden = torch.stack(outputs.hidden_states)  # (25, B, T_mert, D)

        # 4) Learnable weighted sum across layers
        w = F.softmax(self.layer_weights, dim=0)          # (25,)
        feat = torch.einsum("l b t d, l -> b t d", all_hidden, w)  # (B, T_mert, D)

        # 5) Project 1024 → proj_dim
        feat = self.proj(feat)                             # (B, T_mert, proj_dim)

        # 6) Interpolate T_mert → T_target (75 Hz → 100 fps)
        if feat.shape[1] != target_T:
            feat = feat.permute(0, 2, 1)                   # (B, C, T_mert)
            feat = F.interpolate(feat, size=target_T, mode="linear", align_corners=False)
            feat = feat.permute(0, 2, 1)                   # (B, T_target, C)

        return feat
