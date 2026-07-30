"""RWC dataset utilities for MoE technique evaluation.

Provides a dataset and sampler that loads the RWC single-H5 format and
produces note-level data compatible with the MoE evaluation loop:
  waveform, note_onset_frames, note_offset_frames, note_pitches,
  note_tonal_technique, note_articulation, note_legato, num_notes

Label mapping (RWC technique → viotech MoE labels, from technique_label_config.json):
  flageolet  →  tonal=2(harmonics),  artic=0(none),     legato=0(bow_change)
  normal     →  tonal=0(none),       artic=0(none),     legato=0(bow_change)
  pizzicato  →  tonal=1(pizzicato),  artic=0(none),     legato=0(bow_change)
  spiccato   →  tonal=0(none),       artic=3(spiccato), legato=0(bow_change)
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import config
from technique_label_utils import get_technique_labels

MAX_NOTES_PER_SEGMENT = 128

_label_cfg = get_technique_labels()
RWC_TECHNIQUE_MAP = _label_cfg.rwc_technique_map
RWC_DEFAULT_MAP = {'tonal': 0, 'artic': 0, 'legato': 0}


def _get_file_split(fold_id=0):
    """Technique-balanced file split for RWC (3-fold CV)."""
    fl = ['file_006', 'file_031', 'file_032']
    nm = ['file_001', 'file_002', 'file_004', 'file_005', 'file_007',
          'file_010', 'file_019', 'file_022', 'file_024', 'file_025',
          'file_029', 'file_030']
    pz = ['file_000', 'file_008', 'file_009', 'file_015', 'file_016',
          'file_017', 'file_018', 'file_023', 'file_028']
    sp = ['file_003', 'file_011', 'file_012', 'file_013', 'file_014',
          'file_020', 'file_021', 'file_026', 'file_027']

    fold_tests = {
        0: ['file_006', 'file_001', 'file_005', 'file_019', 'file_025',
            'file_000', 'file_015', 'file_018', 'file_003', 'file_013', 'file_021'],
        1: ['file_031', 'file_002', 'file_007', 'file_022', 'file_029',
            'file_008', 'file_016', 'file_023', 'file_011', 'file_014', 'file_026'],
        2: ['file_032', 'file_004', 'file_010', 'file_024', 'file_030',
            'file_009', 'file_017', 'file_028', 'file_012', 'file_020', 'file_027'],
    }
    all_files = set(fl + nm + pz + sp)
    k = fold_id if fold_id in fold_tests else 0
    test_files = sorted(fold_tests[k])
    train_files = sorted(list(all_files - set(test_files)))
    return train_files, test_files


class RWCMoETestSampler:
    """Test sampler for RWC that produces [file_key, start_time] metas."""

    def __init__(self, rwc_h5_path, split='test', segment_seconds=10.0,
                 hop_seconds=1.0, batch_size=4, fold_id=0):
        import h5py
        self.batch_size = batch_size
        self.max_evaluate_iteration = 9999

        if split == 'all':
            allowed = None
        else:
            train_files, test_files = _get_file_split(fold_id)
            allowed = set(train_files if split == 'train' else test_files)

        self.segment_list = []
        with h5py.File(rwc_h5_path, 'r') as f:
            for fk in sorted(f.keys()):
                if not fk.startswith('file_') or (allowed is not None and fk not in allowed):
                    continue
                g = f[fk]
                sr = int(g.attrs.get('sample_rate', 44100))
                wkey = 'full_waveform' if 'full_waveform' in g else 'waveform'
                duration = g[wkey].shape[0] / float(sr)
                t = 0.0
                while t + segment_seconds <= duration:
                    self.segment_list.append([fk, t])
                    t += hop_seconds

        logging.info(f'RWCMoETestSampler ({split}): {len(self.segment_list)} segments')

    def __iter__(self):
        pointer = 0
        iteration = 0
        while pointer < len(self.segment_list) and iteration < self.max_evaluate_iteration:
            batch = []
            for _ in range(self.batch_size):
                if pointer >= len(self.segment_list):
                    break
                batch.append(self.segment_list[pointer])
                pointer += 1
            if batch:
                yield batch
            iteration += 1

    def __len__(self):
        return -1


class RWCMoEDataset:
    """Dataset that loads RWC segments with note-level labels for MoE evaluation.

    Resamples audio from native sample rate to 16kHz (config.sample_rate).
    Extracts note boundaries within each segment and maps file-level technique
    to tonal/articulation/legato labels.
    """

    def __init__(self, rwc_h5_path, segment_seconds=10.0, frames_per_second=100):
        self.rwc_h5_path = rwc_h5_path
        self.segment_seconds = segment_seconds
        self.fps = frames_per_second
        self.target_sr = config.sample_rate
        self.segment_samples = int(self.target_sr * self.segment_seconds)
        self.max_notes = MAX_NOTES_PER_SEGMENT

    def __getitem__(self, meta):
        import h5py
        try:
            import librosa
        except ImportError:
            import scipy.signal as _sig
            librosa = None

        file_key, start_time = meta
        data_dict = {}

        with h5py.File(self.rwc_h5_path, 'r') as f:
            g = f[file_key]
            native_sr = int(g.attrs.get('sample_rate', 44100))
            tech_name = g.attrs.get('pt_name', 'no_technique')
            if isinstance(tech_name, (bytes, bytearray)):
                tech_name = tech_name.decode()
            tech_name = tech_name.lower()

            wkey = 'full_waveform' if 'full_waveform' in g else 'waveform'
            native_start = int(start_time * native_sr)
            native_end = native_start + int(self.segment_seconds * native_sr)
            total_len = g[wkey].shape[0]
            if native_end > total_len:
                native_start = max(0, total_len - int(self.segment_seconds * native_sr))
                native_end = native_start + int(self.segment_seconds * native_sr)
            segment = g[wkey][native_start:native_end].astype(np.float32)

            # Resample to target_sr
            if native_sr != self.target_sr:
                if librosa is not None:
                    segment = librosa.resample(
                        segment, orig_sr=native_sr, target_sr=self.target_sr)
                else:
                    num_out = int(len(segment) * self.target_sr / native_sr)
                    segment = _sig.resample(segment, num_out).astype(np.float32)

            # Pad/truncate to exact segment_samples
            if len(segment) < self.segment_samples:
                segment = np.pad(segment, (0, self.segment_samples - len(segment)))
            else:
                segment = segment[:self.segment_samples]

            max_abs = np.max(np.abs(segment)) if segment.size > 0 else 1.0
            if max_abs > 0:
                segment = segment / (max_abs + 1e-8)
            data_dict['waveform'] = segment

            # Extract note boundaries within this segment
            seg_end_time = start_time + self.segment_seconds
            frames_num = int(self.segment_seconds * self.fps)

            MN = self.max_notes
            note_onset_frames = np.zeros(MN, dtype=np.int32)
            note_offset_frames = np.zeros(MN, dtype=np.int32)
            note_pitches = np.zeros(MN, dtype=np.int32)
            note_tonal = np.zeros(MN, dtype=np.int32)
            note_artic = np.zeros(MN, dtype=np.int32)
            note_legato = np.zeros(MN, dtype=np.int32)
            note_count = 0

            label_map = RWC_TECHNIQUE_MAP.get(tech_name, RWC_DEFAULT_MAP)

            if 'notes' in g:
                notes_g = g['notes']
                for nk in sorted(notes_g.keys()):
                    if not nk.startswith('note_'):
                        continue
                    ng = notes_g[nk]
                    n_start = float(ng.attrs.get('start_time', 0.0))
                    n_dur = float(ng.attrs.get('duration', 0.0))
                    n_end = n_start + n_dur

                    ovl_start = max(n_start, start_time)
                    ovl_end = min(n_end, seg_end_time)
                    if ovl_end <= ovl_start:
                        continue

                    local_onset = ovl_start - start_time
                    local_offset = ovl_end - start_time
                    f0 = int(round(local_onset * self.fps))
                    f1 = int(round(local_offset * self.fps))
                    f0 = max(0, min(frames_num - 1, f0))
                    f1 = max(f0 + 1, min(frames_num, f1))

                    if note_count < MN:
                        note_onset_frames[note_count] = f0
                        note_offset_frames[note_count] = f1
                        note_pitches[note_count] = 60
                        note_tonal[note_count] = label_map['tonal']
                        note_artic[note_count] = label_map['artic']
                        note_legato[note_count] = label_map['legato']
                        note_count += 1

            data_dict['note_onset_frames'] = note_onset_frames
            data_dict['note_offset_frames'] = note_offset_frames
            data_dict['note_pitches'] = note_pitches
            data_dict['note_tonal_technique'] = note_tonal
            data_dict['note_articulation'] = note_artic
            data_dict['note_legato'] = note_legato
            data_dict['num_notes'] = np.int32(note_count)

            # Build frame-level rolls from note-level labels (for frame-level eval)
            tonal_roll = np.zeros(frames_num, dtype=np.int32)
            artic_roll = np.zeros(frames_num, dtype=np.int32)
            # Legato uses Gaussian-peaked float [0,1] to match Viotech
            # convention: 1.0 = bow_change onset, 0.0 = sustained.
            # Config class ID 0 = bow_change → collect those onsets.
            legato_roll = np.zeros(frames_num, dtype=np.float32)
            bow_change_onsets = []
            frame_roll = np.zeros((frames_num, 88), dtype=np.float32)
            for i in range(note_count):
                f0 = int(note_onset_frames[i])
                f1 = int(note_offset_frames[i])
                if f1 > f0:
                    tonal_roll[f0:f1] = note_tonal[i]
                    artic_roll[f0:f1] = note_artic[i]
                    if note_legato[i] == 0:  # 0 = bow_change in config
                        bow_change_onsets.append(f0)
                    pitch_idx = max(0, min(87, note_pitches[i] - 21))
                    frame_roll[f0:f1, pitch_idx] = 1.0
            if bow_change_onsets:
                _sigma = 2.0
                _radius = int(3 * _sigma)
                for cf in bow_change_onsets:
                    t0 = max(0, cf - _radius)
                    t1 = min(frames_num, cf + _radius + 1)
                    ts = np.arange(t0, t1)
                    vals = np.exp(-0.5 * ((ts - cf) / _sigma) ** 2)
                    legato_roll[t0:t1] = np.maximum(legato_roll[t0:t1], vals)
            data_dict['tonal_technique'] = tonal_roll
            data_dict['articulation'] = artic_roll
            data_dict['legato'] = legato_roll
            data_dict['frame_roll'] = frame_roll

        return data_dict
