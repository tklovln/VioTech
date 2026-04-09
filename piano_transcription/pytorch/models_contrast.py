import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_utils import move_data_to_device

try:
    from multilabel_technique_model import NoteWiseTechniqueHead
except ImportError:
    NoteWiseTechniqueHead = None

from note_slicer import NoteSlicer, NoteSlicerConfig
from moe_technique import MoETechniqueHead, MoEConfig
from moe_zone_specialist import ZoneMoETechniqueHead, ZoneMoEConfig
from moe_zone_pertask import PerTaskZoneMoEHead, PerTaskZoneMoEConfig
from moe_frame_multiscale import FrameMultiScaleMoEHead, FrameMultiScaleMoEConfig, FrameDirectHead
from hierarchical_frame import FrameHierarchicalHead, FrameHierarchicalConfig


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum, output_features=False,
                 output_conv2d_map=False):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()
        self.output_features = output_features
        self.output_conv2d_map = output_conv2d_map

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
          If output_features: also returns GRU features (B, T, 512)
          If output_conv2d_map: also returns conv_block4 2D map (B, 128, T, freq_bins//16)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        conv2d_map = x if self.output_conv2d_map else None  # (B, 128, T, 14)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))

        if self.output_features and self.output_conv2d_map:
            return output, x, conv2d_map
        elif self.output_features:
            return output, x
        elif self.output_conv2d_map:
            return output, conv2d_map
        else:
            return output


class FrameLevelTechniqueHead(nn.Module):
    """Three parallel FC heads for frame-level technique classification.

    Expects pre-extracted acoustic features (e.g. 512-dim GRU output from
    AcousticModelCRnn8Dropout) and produces:
      - tonal_technique: (B, T, num_tonal) logits
      - articulation:    (B, T, num_artic) logits
      - legato:          (B, T, 1)         sigmoid probability
    """
    def __init__(self, in_features=512, num_tonal_classes=4, num_artic_classes=4):
        super(FrameLevelTechniqueHead, self).__init__()

        self.tonal_fc = nn.Linear(in_features, num_tonal_classes, bias=True)
        self.artic_fc = nn.Linear(in_features, num_artic_classes, bias=True)
        self.legato_fc = nn.Linear(in_features, 1, bias=True)

        init_layer(self.tonal_fc)
        init_layer(self.artic_fc)
        init_layer(self.legato_fc)

    def forward(self, x):
        """
        Args:
          x: (B, T, in_features) — acoustic features from dedicated CRNN branch

        Returns:
          tonal_output:  (B, T, num_tonal_classes) raw logits
          artic_output:  (B, T, num_artic_classes) raw logits
          legato_output: (B, T, 1)                 sigmoid probability
        """
        tonal_output = self.tonal_fc(x)
        artic_output = self.artic_fc(x)
        legato_output = torch.sigmoid(self.legato_fc(x))

        return tonal_output, artic_output, legato_output


class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num, output_features=False,
                 predict_technique=False, predict_technique_moe=False,
                 predict_technique_moe_zone=False,
                 predict_technique_moe_zone_pt=False,
                 predict_technique_frame_moe=False,
                 moe_config=None, zone_moe_config=None,
                 pt_zone_moe_config=None, frame_moe_config=None,
                 slicer_config=None,
                 detach_cues=False,
                 direct_head=False,
                 hierarchical_head=False):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()
        self.detach_cues = detach_cues

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01
        technique_classes_num = 5
        self.predict_technique = predict_technique
        self.predict_technique_moe = predict_technique_moe
        self.predict_technique_moe_zone = predict_technique_moe_zone
        self.predict_technique_moe_zone_pt = predict_technique_moe_zone_pt
        self.predict_technique_frame_moe = predict_technique_frame_moe

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)
        self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum, output_features)

        # Frame-level technique branch (original)
        if self.predict_technique:
            self.technique_acoustic = AcousticModelCRnn8Dropout(
                classes_num, midfeat, momentum, output_features=True)
            self.technique_head = FrameLevelTechniqueHead(
                in_features=512,
                num_tonal_classes=4,
                num_artic_classes=4,
            )

        # Note-level MoE technique branch
        if self.predict_technique_moe:
            self.technique_acoustic_moe = AcousticModelCRnn8Dropout(
                classes_num, midfeat, momentum, output_features=True)
            self.note_slicer = NoteSlicer(slicer_config or NoteSlicerConfig())
            self.moe_technique_head = MoETechniqueHead(moe_config or MoEConfig())

        # Zone-Specialized MoE technique branch
        if self.predict_technique_moe_zone:
            self.technique_acoustic_zone = AcousticModelCRnn8Dropout(
                classes_num, midfeat, momentum, output_features=True)
            self.note_slicer_zone = NoteSlicer(slicer_config or NoteSlicerConfig())
            self.zone_moe_head = ZoneMoETechniqueHead(zone_moe_config or ZoneMoEConfig())

        # Per-Task Gate Zone MoE technique branch
        if self.predict_technique_moe_zone_pt:
            self.technique_acoustic_zone_pt = AcousticModelCRnn8Dropout(
                classes_num, midfeat, momentum, output_features=True)
            self.note_slicer_zone_pt = NoteSlicer(slicer_config or NoteSlicerConfig())
            self.pt_zone_moe_head = PerTaskZoneMoEHead(
                pt_zone_moe_config or PerTaskZoneMoEConfig())

        # Frame-level Multi-Scale MoE technique branch (end-to-end, no GT boundaries)
        if self.predict_technique_frame_moe:
            self.technique_acoustic_frame_moe = AcousticModelCRnn8Dropout(
                classes_num, midfeat, momentum,
                output_features=True)
            if hierarchical_head:
                _cfg = (frame_moe_config
                        if isinstance(frame_moe_config, FrameHierarchicalConfig)
                        else FrameHierarchicalConfig())
                self.frame_moe_head = FrameHierarchicalHead(_cfg)
            elif direct_head:
                _cfg = frame_moe_config or FrameMultiScaleMoEConfig()
                self.frame_moe_head = FrameDirectHead(_cfg)
            else:
                _cfg = frame_moe_config or FrameMultiScaleMoEConfig()
                self.frame_moe_head = FrameMultiScaleMoEHead(_cfg)

        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()
        if output_features:
            self.output_features_dict = {}
            self.output_features = True
        else:
            self.output_features = False
        

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input, note_info=None):
        """
        Args:
          input: (batch_size, data_length)
          note_info: optional dict for MoE technique branch, with keys
              'onset_frames' (B, N), 'offset_frames' (B, N), 'num_notes' (B,)

        Outputs:
          output_dict: dict
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.predict_technique or self.predict_technique_moe or self.predict_technique_moe_zone or self.predict_technique_moe_zone_pt or self.predict_technique_frame_moe:
            logmel = x

        if self.output_features:
            frame_output, frame_features = self.frame_model(x)
            reg_onset_output, reg_onset_features = self.reg_onset_model(x)
            reg_offset_output, reg_offset_features = self.reg_offset_model(x)
            velocity_output, velocity_features = self.velocity_model(x)
        else:
            frame_output = self.frame_model(x)
            reg_onset_output = self.reg_onset_model(x)
            reg_offset_output = self.reg_offset_model(x)
            velocity_output = self.velocity_model(x)

        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))

        # Frame-level technique branch
        if self.predict_technique:
            _, tech_features = self.technique_acoustic(logmel)
            tonal_output, artic_output, legato_output = self.technique_head(tech_features)

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output,
        }
        if self.output_features:
            output_dict['frame_features'] = frame_features
            output_dict['reg_onset_features'] = reg_onset_features
            output_dict['reg_offset_features'] = reg_offset_features
            output_dict['velocity_features'] = velocity_features
        
        if self.predict_technique:
            output_dict['tonal_technique_output'] = tonal_output
            output_dict['articulation_output'] = artic_output
            output_dict['legato_output'] = legato_output

        # Note-level MoE technique branch
        if self.predict_technique_moe and note_info is not None:
            _, moe_features = self.technique_acoustic_moe(logmel)  # (B, T, 512)
            N = note_info['onset_frames'].shape[1]
            device = moe_features.device
            note_mask = (
                torch.arange(N, device=device).unsqueeze(0)
                < note_info['num_notes'].unsqueeze(1)
            )
            zone_feats = self.note_slicer(
                moe_features,
                note_info['onset_frames'],
                note_info['offset_frames'],
                note_mask,
            )
            tonal_logits, artic_logits, legato_prob, gate_probs = \
                self.moe_technique_head(zone_feats)

            output_dict['note_tonal_logits'] = tonal_logits
            output_dict['note_artic_logits'] = artic_logits
            output_dict['note_legato_prob']  = legato_prob
            output_dict['moe_gate_probs']    = gate_probs

        # Zone-Specialized MoE technique branch
        if self.predict_technique_moe_zone and note_info is not None:
            _, zone_features = self.technique_acoustic_zone(logmel)
            N_z = note_info['onset_frames'].shape[1]
            dev = zone_features.device
            zone_mask = (
                torch.arange(N_z, device=dev).unsqueeze(0)
                < note_info['num_notes'].unsqueeze(1)
            )
            zone_feats = self.note_slicer_zone(
                zone_features,
                note_info['onset_frames'],
                note_info['offset_frames'],
                zone_mask,
            )
            pitches = note_info.get('pitches', None)
            durations = note_info.get('durations', None)
            zt_logits, za_logits, zl_prob, zg_probs = \
                self.zone_moe_head(zone_feats, pitches=pitches, durations=durations)

            output_dict['zone_tonal_logits'] = zt_logits
            output_dict['zone_artic_logits'] = za_logits
            output_dict['zone_legato_prob']  = zl_prob
            output_dict['zone_gate_probs']   = zg_probs

        # Per-Task Gate Zone MoE technique branch
        if self.predict_technique_moe_zone_pt and note_info is not None:
            _, pt_features = self.technique_acoustic_zone_pt(logmel)
            N_pt = note_info['onset_frames'].shape[1]
            dev_pt = pt_features.device
            pt_mask = (
                torch.arange(N_pt, device=dev_pt).unsqueeze(0)
                < note_info['num_notes'].unsqueeze(1)
            )
            pt_zone_feats = self.note_slicer_zone_pt(
                pt_features,
                note_info['onset_frames'],
                note_info['offset_frames'],
                pt_mask,
            )
            pitches_pt = note_info.get('pitches', None)
            durations_pt = note_info.get('durations', None)
            pt_t, pt_a, pt_l, gp_t, gp_a, gp_l = \
                self.pt_zone_moe_head(pt_zone_feats,
                                      pitches=pitches_pt, durations=durations_pt)

            output_dict['pt_tonal_logits'] = pt_t
            output_dict['pt_artic_logits'] = pt_a
            output_dict['pt_legato_prob']  = pt_l
            output_dict['pt_gate_tonal']   = gp_t
            output_dict['pt_gate_artic']   = gp_a
            output_dict['pt_gate_legato']  = gp_l

        # Frame-level Multi-Scale MoE technique branch (end-to-end)
        if self.predict_technique_frame_moe:
            _, fmoe_features = self.technique_acoustic_frame_moe(logmel)
            # fmoe_features: (B, T, 512)
            # Transcription structural cues — max-pool across pitches
            reg_onset = output_dict['reg_onset_output']
            reg_offset = output_dict['reg_offset_output']
            frame = output_dict['frame_output']
            if self.detach_cues:
                reg_onset = reg_onset.detach()
                reg_offset = reg_offset.detach()
                frame = frame.detach()
            onset_cue = reg_onset.max(dim=-1, keepdim=True)[0]   # (B, T, 1)
            offset_cue = reg_offset.max(dim=-1, keepdim=True)[0]
            frame_cue = frame.max(dim=-1, keepdim=True)[0]

            (fm_t, fm_a, fm_l, fg_t, fg_a, fg_l,
             aux_t, aux_a, aux_l) = self.frame_moe_head(
                fmoe_features, onset_cue, offset_cue, frame_cue,
                logmel=logmel)

            output_dict['fmoe_tonal_logits'] = fm_t
            output_dict['fmoe_artic_logits'] = fm_a
            output_dict['fmoe_legato_prob']  = fm_l
            output_dict['fmoe_gate_tonal']   = fg_t
            output_dict['fmoe_gate_artic']   = fg_a
            output_dict['fmoe_gate_legato']  = fg_l
            output_dict['fmoe_aux_tonal']    = aux_t
            output_dict['fmoe_aux_artic']    = aux_a
            output_dict['fmoe_aux_legato']   = aux_l

        return output_dict


class Regress_pedal_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_pedal_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.reg_pedal_onset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_offset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_frame_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        
    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        reg_pedal_onset_output = self.reg_pedal_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_pedal_offset_output = self.reg_pedal_offset_model(x)  # (batch_size, time_steps, classes_num)
        pedal_frame_output = self.reg_pedal_frame_model(x)  # (batch_size, time_steps, classes_num)
        
        output_dict = {
            'reg_pedal_onset_output': reg_pedal_onset_output, 
            'reg_pedal_offset_output': reg_pedal_offset_output,
            'pedal_frame_output': pedal_frame_output}

        return output_dict


# This model is not trained, but is combined from the trained note and pedal models.
class Note_pedal(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        """The combination of note and pedal model.
        """
        super(Note_pedal, self).__init__()

        self.note_model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second, classes_num)
        self.pedal_model = Regress_pedal_CRNN(frames_per_second, classes_num)

    def load_state_dict(self, m, strict=False):
        self.note_model.load_state_dict(m['note_model'], strict=strict)
        self.pedal_model.load_state_dict(m['pedal_model'], strict=strict)

    def forward(self, input):
        note_output_dict = self.note_model(input)
        pedal_output_dict = self.pedal_model(input)

        full_output_dict = {}
        full_output_dict.update(note_output_dict)
        full_output_dict.update(pedal_output_dict)
        return full_output_dict


class LocalTechniqueFeatureConvModel(nn.Module):
    def __init__(
        self,
        classes_num: int = 5,
        window_size: int = 10,
        proj_size: int = 256,
        hidden_size: int = 128,
        momentum: float = 0.01,
        output_features: bool = False,
        block_time_kernels: Optional[Sequence[int]] = [3, 3, 3, 3],
    ):
        """Technique model that takes precomputed logmel features (B, 1, T, F) and limits
        temporal context to ~window_size frames via Conv1d.

        This mirrors `AcousticModelCRnn8Dropout`'s channel progression but without temporal GRUs,
        and ensures strictly local temporal dependence using a temporal Conv1d head.
        """
        super(LocalTechniqueFeatureConvModel, self).__init__()

        # Configure temporal kernel sizes for the 4 front conv blocks.
        # By default uses [3, 3, 3, 3] which contributes (k-1) per block -> 8 frames total.
        if block_time_kernels is None:
            block_time_kernels = [3, 3, 3, 3]
        if len(block_time_kernels) != 4:
            raise ValueError("block_time_kernels must have length 4")
        for k in block_time_kernels:
            if k < 1:
                raise ValueError("Temporal kernel must be >= 1")
            if k % 2 == 0:
                raise ValueError("Temporal kernels must be odd to preserve length with symmetric padding")

        def freq_only_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
            )

        def freq_time_block(in_ch: int, out_ch: int, time_kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(time_kernel, 3),
                    padding=(time_kernel // 2, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch, momentum),
                nn.ReLU(inplace=True),
            )

        self.conv_block1 = freq_time_block(1, 48, block_time_kernels[0])
        self.conv_block2 = freq_time_block(48, 64, block_time_kernels[1])
        self.conv_block3 = freq_time_block(64, 96, block_time_kernels[2])
        self.conv_block4 = freq_time_block(96, 128, block_time_kernels[3])

        midfeat = 1792  # 128 channels * ~14 mel bins after pooling
        self.fc_proj = nn.Linear(midfeat, proj_size, bias=False)
        self.bn_proj = nn.BatchNorm1d(proj_size, momentum=momentum)

        # Keep total temporal receptive field within window_size.
        # Front-end temporal contribution is sum_over_blocks(k_i - 1).
        front_added = sum(max(0, k - 1) for k in block_time_kernels)
        # Head kernel covers the remaining context.
        head_kernel = max(1, window_size - front_added)
        left_pad = head_kernel // 2
        right_pad = head_kernel - 1 - left_pad
        self.temporal_pad = nn.ConstantPad1d((left_pad, right_pad), 0.0)
        self.temporal_conv = nn.Conv1d(
            in_channels=proj_size,
            out_channels=hidden_size,
            kernel_size=head_kernel,
            padding=0,
            bias=True,
        )
        self.head_bn = nn.BatchNorm1d(hidden_size, momentum=momentum)
        self.head_act = nn.ReLU(inplace=True)
        self.head_dropout = nn.Dropout(p=0.2)
        self.pointwise = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=classes_num,
            kernel_size=1,
            bias=True,
        )

        self.output_features = output_features

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
          x: precomputed logmel tensor (B, 1, T, F)
        Returns:
          probs: (B, T, classes_num) in [0, 1]
        """
        # Frequency-only conv blocks with freq pooling
        x = self.conv_block1(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        # (B, C, T, F') -> (B, T, C*F')
        x = x.transpose(1, 2).flatten(2)

        # Frame-wise projection
        x = self.fc_proj(x)
        x = self.bn_proj(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        # Local temporal head
        x = x.transpose(1, 2)
        x = self.temporal_pad(x)
        x = self.temporal_conv(x)
        x = self.head_bn(x)
        x = self.head_act(x)
        x = self.head_dropout(x)
        x = self.pointwise(x)
        x = torch.sigmoid(x)
        probs = x.transpose(1, 2)

        if self.output_features:
            return probs, probs  # keep API parity (features not used); placeholder
        else:
            return probs


class NoteLevelTechniqueModel(nn.Module):
    def __init__(self, classes_num: int = 5, sample_rate: int = 16000, frames_per_second: int = 100,
                 trans_feat_dim: int = 88 * 4, mel_bins: int = 229, use_trans_features: bool = True,
                 output_features: bool = False):
        """Note-level technique classifier with 2D CNN over Logmel, like LocalTechniqueFeatureConvModel.

        Inputs:
          - waveform: (B, L) mono note audio (fixed length)
          - trans_features: (B, T, 88*4) optional transcription probs (onset/offset/frame/velocity)

        Output: (B, classes_num)
        """
        super(NoteLevelTechniqueModel, self).__init__()
        self.sample_rate = sample_rate
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.output_features = output_features

        # STFT + Logmel inside model
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
            hop_length=hop_size, win_length=window_size, window='hann',
            center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
            n_fft=window_size, n_mels=mel_bins, fmin=30, fmax=sample_rate // 2,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum=0.01)

        # 2D conv stack (reuse ConvBlock)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=0.01)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=0.01)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=0.01)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=0.01)

        # Audio projection after global pooling
        self.audio_fc = nn.Linear(128, 128)

        # Transcription features projection (onset/offset/frame concatenated along last dim)
        self.use_trans_features = use_trans_features
        self.trans_proj = nn.Sequential(
            nn.Linear(trans_feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )

        # Fusion and head
        self.fusion = nn.Sequential(
            nn.Linear(128 + (128 if self.use_trans_features else 0), 128), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(128, classes_num)
        )

    @torch.no_grad()
    def extract_transcription_features(self, transcriptor: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
        """Run frozen transcription model to get time-aligned (onset, offset, frame) probs.
        Returns (B, T, 88*4)."""
        transcriptor.eval()
        out = transcriptor(waveform)
        onset = out.get('reg_onset_output')
        offset = out.get('reg_offset_output')
        frame = out.get('frame_output')
        velocity = out.get('velocity_output')
        feats = torch.cat([onset, offset, frame, velocity], dim=-1)
        return feats

    def forward(self, waveform: torch.Tensor, trans_features: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Compute logmel from waveform
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # 2D conv with pooling
        x = self.conv_block1(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global average pooling over time and freq -> (B, C)
        x = x.mean(dim=[2, 3])
        x = F.relu(self.audio_fc(x))

        # Transcription features path
        if self.use_trans_features and trans_features is not None:
            tp = self.trans_proj(trans_features).mean(dim=1)
            # max pooling with amax
            # tp = self.trans_proj(trans_features.amax(dim=1))
            z = torch.cat([x, tp], dim=-1)
        else:
            z = x

        logits = self.fusion(z)
        if self.output_features:
            return logits, z
        else:
            return logits