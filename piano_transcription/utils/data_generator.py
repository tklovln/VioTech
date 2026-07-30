import os
import sys
import numpy as np
import h5py
import csv
import time
import collections
import librosa
import sox
import logging
import audiomentations as A
import soundfile as sf

from utilities import (create_folder, int16_to_float32, traverse_folder, 
    pad_truncate_sequence, TargetProcessor, write_events_to_midi, 
    plot_waveform_midi_targets)
import config


class MaestroDataset(object):
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, 
        max_note_shift=0, augmentor=None):
        """This class takes the meta of an audio segment as input, and return 
        the waveform and targets of the audio segment. This class is used by 
        DataLoader. 
        
        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
          augmentor: object
        """
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.max_note_shift = max_note_shift
        self.begin_note = config.begin_note
        self.classes_num = config.classes_num
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.augmentor = augmentor

        self.random_state = np.random.RandomState(1234)

        self.target_processor = TargetProcessor(self.segment_seconds, 
            self.frames_per_second, self.begin_note, self.classes_num)
        """Used for processing MIDI events to target."""

    def __getitem__(self, meta):
        """Prepare input and target of a segment for training.
        
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)
         
        data_dict = {}

        note_shift = self.random_state.randint(low=-self.max_note_shift, 
            high=self.max_note_shift + 1)

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])

            if self.augmentor:
                waveform = self.augmentor.augment(waveform)

            if note_shift != 0:
                """Augment pitch"""
                waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate, 
                    n_steps=note_shift, bins_per_octave=12)

            # scale waveform to -1.0 to 1.0
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

            data_dict['waveform'] = waveform

            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

            # Process MIDI events to target
            (target_dict, note_events, pedal_events) = \
                self.target_processor.process(start_time, midi_events_time, 
                    midi_events, extend_pedal=True, note_shift=note_shift)

        # Combine input and target
        for key in target_dict.keys():
            data_dict[key] = target_dict[key]

        debugging = False
        if debugging:
            plot_waveform_midi_targets(data_dict, start_time, note_events)
            exit()

        return data_dict

import pedalboard as pb
from pedalboard import Pedalboard, PitchShift, Gain, Reverb, LadderFilter
import numpy as np

class PedalboardEqualizerApprox:
    def __init__(self, rng=np.random.RandomState(42)):
        self.rng = rng

    def make_filter(self):
        freq = float(np.exp(self.rng.uniform(np.log(32), np.log(4096))))
        resonance = float(self.rng.uniform(1, 2))  # approximate Q
        # Note: no gain_db in LadderFilter

        return LadderFilter(
            mode=LadderFilter.Mode.BPF12,
            cutoff_hz=freq,
            resonance=resonance
        )
    
class PedalboardAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(42)
        self.board = Pedalboard([
            PitchShift(semitones=0.1),                   # Slight pitch shift
            Gain(gain_db=5),                             # Gain boost
            self._random_ladder_filter(),
            self._random_ladder_filter(),
            Reverb(room_size=0.35),                      # Reverb effect
        ])
        

    def _random_ladder_filter(self):
        freq = float(np.exp(self.rng.uniform(np.log(32), np.log(4096))))  # log-uniform
        resonance = float(self.rng.uniform(0.0, 1.0))  # Q-like behavior

        return LadderFilter(
            mode=LadderFilter.Mode.BPF12,
            cutoff_hz=freq,
            resonance=resonance
        )

    def augment(self, x):
        return self.board.process(x, sample_rate=self.sample_rate)

class ModernAugmentor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        
        self.pipeline = A.Compose([
            A.PitchShift(min_semitones=-1.2, max_semitones=1.2, p=1.0),  # pitch
            A.Gain(min_gain_db=-30, max_gain_db=10, p=1.0),              # contrast equivalent
            A.PeakingFilter(min_center_freq=32, max_center_freq=4096, 
                          min_gain_db=-30, max_gain_db=10, p=1.0),       # equalizer 1
            A.PeakingFilter(min_center_freq=32, max_center_freq=4096, 
                          min_gain_db=-30, max_gain_db=10, p=1.0),       # equalizer 2  
            A.RoomSimulator(
                # Room dimensions (meters)
                min_size_x=3.0, max_size_x=8.0,    # width
                min_size_y=3.0, max_size_y=8.0,    # depth  
                min_size_z=2.4, max_size_z=4.0,    # height
                # Room acoustics - varying from dry to reverberant
                min_absorption_value=0.1, max_absorption_value=0.7,
                # Source position (meters from origin)
                min_source_x=0.5, max_source_x=7.0,
                min_source_y=0.5, max_source_y=7.0, 
                min_source_z=1.0, max_source_z=2.5,
                # Microphone distance from source
                min_mic_distance=0.3, max_mic_distance=3.0,
                # Microphone angle (radians) - full sphere
                min_mic_azimuth=-np.pi, max_mic_azimuth=np.pi,
                min_mic_elevation=-np.pi/2, max_mic_elevation=np.pi/2,
                # Simulation settings
                calculation_mode="absorption",  # faster than "rt60"
                use_ray_tracing=False,          # faster, less accurate
                max_order=3,                    # reflection order
                leave_length_unchanged=True,    # keep original audio length
                p=1.0                          # always apply
            ),
        ])
    
    def augment(self, x):
        return self.pipeline(samples=x, sample_rate=self.sample_rate)
      
class Augmentor(object):
    def __init__(self):
        """Data augmentor."""
        
        self.sample_rate = config.sample_rate
        self.random_state = np.random.RandomState(1234)

    def augment(self, x):
        clip_samples = len(x)

        # logger = logging.getLogger('sox')
        # logger.propagate = False

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        tfm.pitch(self.random_state.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.random_state.uniform(0, 100, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0], 
            width_q=self.random_state.uniform(1, 2, 1)[0], 
            gain_db=self.random_state.uniform(-30, 10, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0], 
            width_q=self.random_state.uniform(1, 2, 1)[0], 
            gain_db=self.random_state.uniform(-30, 10, 1)[0])
        
        tfm.reverb(reverberance=self.random_state.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate)
        aug_x = pad_truncate_sequence(aug_x, clip_samples)
        
        return aug_x

    def loguniform(self, low, high, size):
        return np.exp(self.random_state.uniform(np.log(low), np.log(high), size))


class Sampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if 'split' not in hf.attrs:
                    continue
                attr = hf.attrs['split']
                if isinstance(attr, bytes):
                    attr = attr.decode()
                if attr == split:
                    audio_name = hdf5_path.split('/')[-1]
                    year = hf.attrs['year'].decode()
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([year, audio_name, start_time])
                        start_time += self.hop_seconds
                    
                    n += 1
                    if mini_data and n == 10:
                        break
        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

        logging.info('{} segments: {}'.format(split, len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class TestSampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler for testing.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 20    # Number of mini-batches to validate

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if 'split' not in hf.attrs:
                    continue
                attr = hf.attrs['split']
                if isinstance(attr, bytes):
                    attr = attr.decode()
                if attr == split:
                    audio_name = hdf5_path.split('/')[-1]
                    year = hf.attrs['year'].decode()
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([year, audio_name, start_time])
                        start_time += self.hop_seconds
                    
                    n += 1
                    if mini_data and n == 10:
                        break
        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

        logging.info('Evaluate {} segments: {}'.format(split, len(self.segment_list)))

        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        pointer = 0
        iteration = 0

        while True:
            if iteration == self.max_evaluate_iteration:
                break

            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1

            yield batch_segment_list

    def __len__(self):
        return -1


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict
    

class CustomSampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if 'split' not in hf.attrs:
                    continue
                attr = hf.attrs['split']
                if isinstance(attr, bytes):
                    attr = attr.decode()
                if attr == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([audio_name, start_time])
                        start_time += self.hop_seconds
                    
                    n += 1
                    if mini_data and n == 10:
                        break
        """self.segment_list looks like:
        [['MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""
        
        logging.info('{} segments: {}'.format(split, len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']

class CustomTestSampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler for testing.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 20    # Number of mini-batches to validate

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if 'split' not in hf.attrs:
                    continue
                attr = hf.attrs['split']
                if isinstance(attr, bytes):
                    attr = attr.decode()
                if attr == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([ audio_name, start_time])
                        start_time += self.hop_seconds
                    
                    n += 1
                    if mini_data and n == 10:
                        break
        """self.segment_list looks like:
        [['MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

        logging.info('Evaluate {} segments: {}'.format(split, len(self.segment_list)))

        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)
        
    def __iter__(self):
        pointer = 0
        iteration = 0

        while True:
            if iteration == self.max_evaluate_iteration:
                break

            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1

            yield batch_segment_list

    def __len__(self):
        return -1


MAX_NOTES_PER_SEGMENT = 128


class CustomDataset(object):
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, 
        max_note_shift=0, augmentor=None, include_technique_label=False):
        """This class takes the meta of an audio segment as input, and return 
        the waveform and targets of the audio segment. This class is used by 
        DataLoader. 
        
        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
          augmentor: object
        """
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.max_note_shift = max_note_shift
        self.begin_note = config.begin_note
        self.classes_num = config.classes_num
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.augmentor = augmentor
        self.include_technique_label = include_technique_label
        self.max_notes = MAX_NOTES_PER_SEGMENT
        # Fixed order of technique classes for one-hot encoding (5 classes including no_technique)
        # self.technique_classes = ['legato', 'pizzicato', 'spiccato', 'sustain', 'no_technique']
        self.technique_classes = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']

        from technique_label_utils import get_technique_labels
        self._label_cfg = get_technique_labels()

        self.random_state = np.random.RandomState(1234)

        self.target_processor = TargetProcessor(self.segment_seconds, 
            self.frames_per_second, self.begin_note, self.classes_num)
        """Used for processing MIDI events to target."""

    def __getitem__(self, meta):
        """Prepare input and target of a segment for training.
        
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)

        data_dict = {}

        note_shift = self.random_state.randint(low=-self.max_note_shift, 
            high=self.max_note_shift + 1)

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])

            if self.augmentor:
                waveform = self.augmentor.augment(waveform)

            if note_shift != 0:
                """Augment pitch"""
                waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate, 
                    n_steps=note_shift, bins_per_octave=12)

            # scale waveform to -1.0 to 1.0
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

            data_dict['waveform'] = waveform

            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

            # Process MIDI events to target
            (target_dict, note_events, pedal_events) = \
                self.target_processor.process(start_time, midi_events_time, 
                    midi_events, extend_pedal=True, note_shift=note_shift)

        # Combine input and target
        for key in target_dict.keys():
            data_dict[key] = target_dict[key]

        if self.include_technique_label:
            frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
            end_time = start_time + self.segment_seconds

            with h5py.File(hdf5_path, 'r') as hf:
                has_viotech = all(k in hf for k in ('tonalTechnique', 'articulation', 'legato'))
                if has_viotech:
                    evt_times = hf['midi_event_time'][:]
                    evt_strings = [e.decode() for e in hf['midi_event'][:]]
                    evt_tonal = hf['tonalTechnique'][:]
                    evt_artic = hf['articulation'][:]
                    evt_legato = hf['legato'][:]

            if has_viotech:
                tonal_roll = np.zeros(frames_num, dtype=np.int32)
                artic_roll = np.zeros(frames_num, dtype=np.int32)
                legato_roll = np.zeros(frames_num, dtype=np.float32)
                bow_change_onsets = []

                # Note-level arrays (padded to max_notes)
                MN = self.max_notes
                note_onset_frames  = np.zeros(MN, dtype=np.int32)
                note_offset_frames = np.zeros(MN, dtype=np.int32)
                note_pitches       = np.zeros(MN, dtype=np.int32)
                note_tonal         = np.zeros(MN, dtype=np.int32)
                note_artic         = np.zeros(MN, dtype=np.int32)
                note_leg           = np.zeros(MN, dtype=np.int32)
                note_count = 0

                pending = {}
                for i, evt_str in enumerate(evt_strings):
                    if 'note_on' in evt_str:
                        for tok in evt_str.split():
                            if tok.startswith('note='):
                                pitch = int(tok.split('=')[1])
                                pending[pitch] = (evt_times[i], evt_tonal[i], evt_artic[i], evt_legato[i])
                                break
                    elif 'note_off' in evt_str:
                        for tok in evt_str.split():
                            if tok.startswith('note='):
                                pitch = int(tok.split('=')[1])
                                break
                        else:
                            continue
                        if pitch in pending:
                            onset_t, tv, av, lv = pending.pop(pitch)
                            tv = self._label_cfg.remap_tonal(tv)
                            av = self._label_cfg.remap_artic(av)
                            lv = self._label_cfg.remap_legato(lv)
                            offset_t = evt_times[i]
                            if offset_t > start_time and onset_t < end_time:
                                local_on = max(0.0, onset_t - start_time)
                                local_off = min(self.segment_seconds, offset_t - start_time)
                                f0 = max(0, min(frames_num, int(round(local_on * self.frames_per_second))))
                                f1 = max(0, min(frames_num, int(round(local_off * self.frames_per_second))))
                                if f1 > f0:
                                    tonal_roll[f0:f1] = tv
                                    artic_roll[f0:f1] = av
                                    if lv == 0:
                                        bow_change_onsets.append(f0)
                                    if note_count < MN:
                                        note_onset_frames[note_count] = f0
                                        note_offset_frames[note_count] = f1
                                        note_pitches[note_count] = pitch
                                        note_tonal[note_count] = tv
                                        note_artic[note_count] = av
                                        note_leg[note_count] = lv
                                        note_count += 1

                # Soft regression target for bow_change detection (Gaussian peaks)
                if bow_change_onsets:
                    _sigma = 2.0
                    _radius = int(3 * _sigma)
                    for cf in bow_change_onsets:
                        t0 = max(0, cf - _radius)
                        t1 = min(frames_num, cf + _radius + 1)
                        ts = np.arange(t0, t1)
                        vals = np.exp(-0.5 * ((ts - cf) / _sigma) ** 2)
                        legato_roll[t0:t1] = np.maximum(legato_roll[t0:t1], vals)

                # Frame-level rolls (backward compatible)
                data_dict['tonal_technique'] = tonal_roll
                data_dict['articulation'] = artic_roll
                data_dict['legato'] = legato_roll

                # Note-level arrays for MoE technique head
                data_dict['note_onset_frames']    = note_onset_frames
                data_dict['note_offset_frames']   = note_offset_frames
                data_dict['note_pitches']         = note_pitches
                data_dict['note_tonal_technique'] = note_tonal
                data_dict['note_articulation']    = note_artic
                data_dict['note_legato']          = note_leg
                data_dict['num_notes']            = np.int32(note_count)
            else:
                # Legacy mosapt: single technique from filename as one-hot
                technique_name = self._extract_technique_name(hdf5_name)
                technique_one_hot = self._technique_one_hot(technique_name)
                if 'frame_roll' in data_dict:
                    frames_num = data_dict['frame_roll'].shape[0]
                technique_roll = np.tile(technique_one_hot, (frames_num, 1))
                frame_activity = np.any(target_dict['frame_roll'] > 0, axis=1)
                no_idx = self.technique_classes.index('no_technique')
                technique_roll[~frame_activity, :] = 0.0
                technique_roll[~frame_activity, no_idx] = 1.0
                data_dict['technique'] = technique_roll.astype(np.float32)

        return data_dict

    def _extract_technique_name(self, hdf5_name):
        """Extract technique name from filename e.g. '*_legato.h5' -> 'legato'."""
        base = os.path.splitext(os.path.basename(hdf5_name))[0]
        parts = base.split('_')
        return parts[-1].lower() if parts else 'unknown'

    def _technique_one_hot(self, technique_name):
        """Return one-hot numpy vector (len=5) for the technique.
        Unknown techniques map to 'no_technique'.
        Order: ['legato', 'pizzicato', 'spiccato', 'sustain', 'no_technique']
        """
        one_hot = np.zeros(len(self.technique_classes), dtype=np.float32)
        if technique_name in self.technique_classes:
            idx = self.technique_classes.index(technique_name)
        else:
            idx = self.technique_classes.index('no_technique')
        one_hot[idx] = 1.0
        return one_hot


class RWCTechniqueSampler(object):
    def __init__(self, rwc_h5_path, split, segment_seconds, hop_seconds,
            batch_size, mini_data, random_seed=1234, max_files=0, allowed_file_keys=None):
        """Sampler for RWC technique-only training from a single H5 file.

        Args:
          rwc_h5_path: str, path to a single H5 containing groups 'file_*'
          split: str, ignored (kept for API compatibility)
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool
          max_files: int, limit number of file_* groups to use (0 for all)
        """
        assert split in ['train', 'validation', 'test']
        import h5py
        self.rwc_h5_path = rwc_h5_path
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        self.segment_list = []  # [file_key, start_time]
        with h5py.File(self.rwc_h5_path, 'r') as f:
            file_keys = [k for k in f.keys() if k.startswith('file_')]
            if allowed_file_keys is not None:
                allowed_set = set(allowed_file_keys)
                file_keys = [k for k in file_keys if k in allowed_set]
            else:
                # Use fixed balanced split by default
                train_list, test_list = get_rwc_fixed_file_split()
                default_allowed = set(train_list if split == 'train' else test_list)
                # Intersect with actually present keys
                intersected = [k for k in file_keys if k in default_allowed]
                if len(intersected) > 0:
                    file_keys = intersected
            if mini_data:
                file_keys = file_keys[:10]
            if max_files and max_files > 0:
                file_keys = file_keys[:max_files]
            for fk in file_keys:
                g = f[fk]
                # Prefer 'full_waveform', else 'waveform'
                if 'full_waveform' in g:
                    length = g['full_waveform'].shape[0]
                elif 'waveform' in g:
                    length = g['waveform'].shape[0]
                else:
                    continue
                duration = length / float(self.sample_rate)
                start_time = 0.0
                while (start_time + self.segment_seconds < duration):
                    self.segment_list.append([fk, start_time])
                    start_time += self.hop_seconds
        logging.info('RWC segments: {}'.format(len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return -1

    def state_dict(self):
        state = {
            'pointer': self.pointer,
            'segment_indexes': self.segment_indexes,
        }
        return state

    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class RWCTechniqueTestSampler(object):
    def __init__(self, rwc_h5_path, split, segment_seconds, hop_seconds,
            batch_size, mini_data, random_seed=1234, max_files=0, allowed_file_keys=None):
        assert split in ['train', 'validation', 'test']
        import h5py
        self.rwc_h5_path = rwc_h5_path
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 20

        self.segment_list = []
        with h5py.File(self.rwc_h5_path, 'r') as f:
            file_keys = [k for k in f.keys() if k.startswith('file_')]
            if allowed_file_keys is not None:
                allowed_set = set(allowed_file_keys)
                file_keys = [k for k in file_keys if k in allowed_set]
            else:
                # Use fixed balanced split by default
                train_list, test_list = get_rwc_fixed_file_split()
                default_allowed = set(train_list if split == 'train' else test_list)
                intersected = [k for k in file_keys if k in default_allowed]
                if len(intersected) > 0:
                    file_keys = intersected
            if mini_data:
                file_keys = file_keys[:10]
            if max_files and max_files > 0:
                file_keys = file_keys[:max_files]
            for fk in file_keys:
                g = f[fk]
                if 'full_waveform' in g:
                    length = g['full_waveform'].shape[0]
                elif 'waveform' in g:
                    length = g['waveform'].shape[0]
                else:
                    continue
                duration = length / float(self.sample_rate)
                start_time = 0.0
                while (start_time + self.segment_seconds < duration):
                    self.segment_list.append([fk, start_time])
                    start_time += self.hop_seconds

        logging.info('Evaluate RWC segments: {}'.format(len(self.segment_list)))
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        pointer = 0
        iteration = 0
        while True:
            if iteration == self.max_evaluate_iteration:
                break
            batch_segment_list = []
            i = 0
            while i < self.batch_size and pointer < len(self.segment_indexes):
                index = self.segment_indexes[pointer]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
                i += 1
            iteration += 1
            if batch_segment_list:
                yield batch_segment_list
            else:
                break

    def __len__(self):
        return -1


class RWCTechniqueDataset(object):
    def __init__(self, rwc_h5_path, segment_seconds, frames_per_second,
        max_note_shift=0, augmentor=None):
        """Technique-only dataset for RWC single H5 structure.

        Yields waveform segment and frame-wise technique one-hot labels.
        """
        import h5py
        self.rwc_h5_path = rwc_h5_path
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.classes_num = config.classes_num
        self.max_note_shift = max_note_shift
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.augmentor = augmentor

        self.technique_classes = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']
        self.random_state = np.random.RandomState(1234)

    def __getitem__(self, meta):
        import h5py
        [file_key, start_time] = meta
        data_dict = {}

        note_shift = self.random_state.randint(low=-self.max_note_shift,
            high=self.max_note_shift + 1)

        with h5py.File(self.rwc_h5_path, 'r') as f:
            g = f[file_key]
            if 'full_waveform' in g:
                wave = g['full_waveform']
            else:
                wave = g['waveform']
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples
            if end_sample >= wave.shape[0]:
                start_sample = max(0, wave.shape[0] - self.segment_samples)
                end_sample = start_sample + self.segment_samples
            # Convert to float32 in [-1, 1]
            segment = wave[start_sample:end_sample]
            if segment.dtype == np.int16:
                segment = (segment.astype(np.float32) / np.max(np.abs(segment)))
            else:
                segment = segment.astype(np.float32)

            if self.augmentor:
                segment = self.augmentor.augment(segment)

            if note_shift != 0:
                segment = librosa.effects.pitch_shift(segment, sr=self.sample_rate,
                    n_steps=note_shift, bins_per_octave=12)

            # Normalize to [-1, 1]
            max_abs = np.max(np.abs(segment)) if segment.size > 0 else 1.0
            if max_abs > 0:
                segment = segment / (max_abs + 1e-8)
            data_dict['waveform'] = segment

            # Technique one-hot for all frames
            frames_num = int(self.segment_seconds * self.frames_per_second)
            tech_name = g.attrs.get('pt_name', 'no_technique')
            if isinstance(tech_name, (bytes, bytearray)):
                tech_name = tech_name.decode()
            tech_name = tech_name.lower()
            if tech_name not in self.technique_classes:
                tech_name = 'no_technique'

            one_hot = np.zeros(len(self.technique_classes), dtype=np.float32)
            one_hot[self.technique_classes.index(tech_name)] = 1.0
            technique_roll = np.tile(one_hot, (frames_num, 1))
            data_dict['technique'] = technique_roll

            # Build activity mask from note annotations to exclude silent frames in loss
            activity = np.zeros(frames_num, dtype=bool)
            if 'notes' in g and isinstance(g['notes'], h5py.Group):
                notes_g = g['notes']
                for nk in notes_g.keys():
                    if not nk.startswith('note_'):
                        continue
                    ng = notes_g[nk]
                    n_start = float(ng.attrs.get('start_time', 0.0))
                    n_dur = float(ng.attrs.get('duration', 0.0))
                    n_end = n_start + n_dur
                    seg_start = start_time
                    seg_end = start_time + self.segment_seconds
                    # Overlap of [n_start, n_end) and [seg_start, seg_end)
                    ovl_start = max(n_start, seg_start)
                    ovl_end = min(n_end, seg_end)
                    if ovl_end > ovl_start:
                        local_start = ovl_start - seg_start
                        local_end = ovl_end - seg_start
                        f0 = int(np.floor(local_start * self.frames_per_second))
                        f1 = int(np.ceil(local_end * self.frames_per_second))
                        f0 = max(0, min(frames_num, f0))
                        f1 = max(0, min(frames_num, f1))
                        if f1 > f0:
                            activity[f0:f1] = True
            # Fallback: simple energy-based activity if no notes found
            if not activity.any():
                # frame hop in samples (approx) and frame size not strictly needed; use simple block energy
                hop = max(1, int(self.sample_rate / self.frames_per_second))
                energies = []
                for t in range(frames_num):
                    s0 = t * hop
                    s1 = min(len(segment), s0 + hop)
                    if s1 > s0:
                        energies.append(float(np.mean(segment[s0:s1] ** 2)))
                    else:
                        energies.append(0.0)
                energies = np.array(energies)
                thr = max(1e-8, 0.001 * float(np.max(energies)) if energies.size > 0 else 1e-8)
                activity = energies > thr

            # Create frame_roll binary mask (any-activity as positive across pitches)
            frame_roll = np.zeros((frames_num, self.classes_num), dtype=np.float32)
            if activity.any():
                frame_roll[activity, :] = 1.0
            data_dict['frame_roll'] = frame_roll

        return data_dict


class RWCTechNoteSampler(object):
    def __init__(self, rwc_h5_path, split, batch_size, mini_data, random_seed=1234, allowed_file_keys=None, fold_id=0):
        assert split in ['train', 'validation', 'test']
        import h5py
        self.rwc_h5_path = rwc_h5_path
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        self.note_list = []  # [file_key, note_key]
        with h5py.File(self.rwc_h5_path, 'r') as f:
            file_keys = [k for k in f.keys() if k.startswith('file_')]
            if allowed_file_keys is not None:
                allowed_set = set(allowed_file_keys)
                file_keys = [k for k in file_keys if k in allowed_set]
            else:
                train_list, test_list = get_rwc_fixed_file_split(fold_id)
                default_allowed = set(train_list if split == 'train' else test_list)
                file_keys = [k for k in file_keys if k in default_allowed]

            for fk in file_keys:
                g = f[fk]
                if 'notes' in g and isinstance(g['notes'], h5py.Group):
                    for nk in g['notes'].keys():
                        if nk.startswith('note_'):
                            self.note_list.append([fk, nk])

        if mini_data:
            self.note_list = self.note_list[: min(100, len(self.note_list))]

        logging.info('RWC note samples ({}): {}'.format(split, len(self.note_list)))

        self.pointer = 0
        self.indexes = np.arange(len(self.note_list))
        self.random_state.shuffle(self.indexes)

    def __iter__(self):
        while True:
            batch = []
            i = 0
            while i < self.batch_size:
                idx = self.indexes[self.pointer]
                self.pointer += 1
                if self.pointer >= len(self.indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                batch.append(self.note_list[idx])
                i += 1
            yield batch

    def __len__(self):
        return -1

    def state_dict(self):
        return {'pointer': self.pointer, 'indexes': self.indexes}

    def load_state_dict(self, state):
        self.pointer = state.get('pointer', 0)
        self.indexes = state.get('indexes', self.indexes)


class RWCTechNoteDataset(object):
    def __init__(self, rwc_h5_path, frames_per_second):
        import h5py
        self.rwc_h5_path = rwc_h5_path
        self.sample_rate = config.sample_rate
        self.frames_per_second = frames_per_second
        self.technique_classes = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']

    def __getitem__(self, meta):
        import h5py
        file_key, note_key = meta
        with h5py.File(self.rwc_h5_path, 'r') as f:
            g = f[file_key]
            ng = g['notes'][note_key]
            wav = ng['waveform'][:]
            if wav.dtype == np.int16:
                wav = wav.astype(np.float32) / 32768.0
            else:
                wav = wav.astype(np.float32)
            if wav.size > 0:
                wav = wav / (np.max(np.abs(wav)) + 1e-8)

            data = {'waveform': wav}

            tech_name = g.attrs.get('pt_name', 'no_technique')
            if isinstance(tech_name, (bytes, bytearray)):
                tech_name = tech_name.decode()
            tech_name = tech_name.lower()
            if tech_name not in self.technique_classes:
                tech_name = 'no_technique'
            one_hot = np.zeros(len(self.technique_classes), dtype=np.float32)
            one_hot[self.technique_classes.index(tech_name)] = 1.0
            # Note-level label as 1 frame for compatibility: (T=1, C)
            data['technique'] = one_hot[None, :]
        return data


class RWCTechNoteWavDataset(object):
    def __init__(self, root_dir, frames_per_second, segment_seconds=2.0, return_logmel=False, split: str = 'train', 
    allowed_file_keys=None, augmentor=None, fold_id=0):
        """Dataset for note-level WAVs exported by rwc_export_note_dataset.py

        Expects filenames like:
          file_000_note_012_tech-pizzicato_st-1.234_dur-0.456.wav
        """
        import glob
        self.root_dir = root_dir
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.segment_seconds = segment_seconds
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.technique_classes = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']
        self.return_logmel = False  # compute mel in model, not here
        self.augmentor = augmentor

        # Find all wavs recursively
        pattern = os.path.join(root_dir, '**', '*.wav')
        all_paths = sorted(glob.glob(pattern, recursive=True))

        # Split by file_key using fixed mapping or provided allowed list
        def _extract_file_key(pth: str) -> str:
            base = os.path.basename(pth)
            if '_note_' in base:
                return base.split('_note_')[0]
            # fallback: use immediate parent dir if it looks like file_XXX
            parent = os.path.basename(os.path.dirname(pth))
            return parent if parent.startswith('file_') else base

        if allowed_file_keys is None:
            try:
                from data_generator import get_rwc_fixed_file_split  # type: ignore
                train_keys, test_keys = get_rwc_fixed_file_split(fold_id)
                key_set = set(train_keys if split == 'train' else test_keys)
            except Exception:
                key_set = None
        else:
            key_set = set(allowed_file_keys)

        if key_set is not None:
            self.paths = [p for p in all_paths if _extract_file_key(p) in key_set]
        else:
            self.paths = all_paths

        logging.info('RWCTechNoteWavDataset: found {} wavs under {} (split={})'.format(len(self.paths), root_dir, split))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import soundfile as sf
        path = self.paths[idx]
        wav, sr = sf.read(path)
        if sr != self.sample_rate:
            # simple resample with librosa if mismatch
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=self.sample_rate)
        wav = wav.astype(np.float32)
        # pad / truncate to fixed length
        wav = pad_truncate_sequence(wav, self.segment_samples)
        # optional augmentation on waveform
        if self.augmentor is not None:
            try:
                wav = self.augmentor.augment(wav)
                wav = pad_truncate_sequence(wav, self.segment_samples)
            except Exception:
                pass
        if wav.size > 0:
            wav = wav / (np.max(np.abs(wav)) + 1e-8)

        # Parse technique from filename
        base = os.path.basename(path)
        tech = 'no_technique'
        if 'tech-' in base:
            try:
                tech = base.split('tech-')[1].split('_')[0].lower()
            except Exception:
                pass
        if tech not in self.technique_classes:
            tech = 'no_technique'
        one_hot = np.zeros(len(self.technique_classes), dtype=np.float32)
        one_hot[self.technique_classes.index(tech)] = 1.0

        data = {
            'waveform': wav,
            'technique': one_hot[None, :],  # (T=1, C)
        }
        # do NOT compute logmel here; computed in model
        return data


class MOSAPTNoteDataset(object):
    def __init__(self, hdf5s_dir, frames_per_second, fixed_seconds=None, augmentor=None, split: str = 'train', allowed_file_keys=None, mini_data=False):
        """Read MOSAPT as single-note samples for faster training.

        Supports two modes automatically:
        - WAV mode: reads pre-exported note WAVs under a root like `mosapt_notes/`.
        - H5 mode: falls back to slicing notes online from MOSAPT HDF5s (previous behavior).

        Returns items with keys: 'waveform' (float32) and 'technique' (one-hot, shape (1, C)).
        If fixed_seconds is set, audio is padded/truncated to that length.
        """
        import glob
        import h5py

        self.root_dir = hdf5s_dir
        self.sample_rate = config.sample_rate
        self.frames_per_second = frames_per_second
        self.fixed_seconds = fixed_seconds
        self.segment_samples = int(self.sample_rate * fixed_seconds) if fixed_seconds is not None else None
        self.augmentor = augmentor
        self.technique_classes = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']

        # Detect mode by presence of .wav files
        wav_paths = sorted(glob.glob(os.path.join(self.root_dir, '**', '*.wav'), recursive=True))
        self.mode = 'wav' if len(wav_paths) > 0 else 'h5'

        if self.mode == 'wav':
            def _base_key(p: str) -> str:
                base = os.path.basename(p)
                if '_note_' in base:
                    return base.split('_note_')[0]
                parent = os.path.basename(os.path.dirname(p))
                return parent

            # Filter by allowed file keys if provided (keys should match H5 base names)
            if allowed_file_keys is not None:
                key_set = set(allowed_file_keys)
                def _allowed(p: str) -> bool:
                    base = os.path.splitext(os.path.basename(p))[0]
                    before_note = base.split('_note_')[0] if '_note_' in base else base
                    return (_base_key(p) in key_set) or (base in key_set) or (before_note in key_set)
                wav_paths = [p for p in wav_paths if _allowed(p)]

            if mini_data:
                wav_paths = wav_paths[: min(1000, len(wav_paths))]

            self.paths = wav_paths
            logging.info('MOSAPTNoteDataset[WAV]: found {} wavs under {}'.format(len(self.paths), self.root_dir))

        else:
            # H5 fallback mode (kept for compatibility)
            self.items = []  # list of (h5_path, onset_s, offset_s, technique_name)
            (hdf5_names, hdf5_paths) = traverse_folder(self.root_dir)
            if allowed_file_keys is not None:
                allowed = set(allowed_file_keys)
                hdf5_paths = [p for p in hdf5_paths if os.path.splitext(os.path.basename(p))[0] in allowed]

            for p in sorted(hdf5_paths):
                try:
                    with h5py.File(p, 'r') as hf:
                        if ('midi_event' not in hf) or ('midi_event_time' not in hf) or ('waveform' not in hf):
                            continue
                        midi_events = [e.decode() for e in hf['midi_event'][:]]
                        midi_times = hf['midi_event_time'][:]
                        duration_sec = float(hf.attrs['duration']) if 'duration' in hf.attrs else (len(hf['waveform']) / self.sample_rate)
                        tp = TargetProcessor(duration_sec, self.frames_per_second, config.begin_note, config.classes_num)
                        _, note_events, _ = tp.process(0.0, midi_times, midi_events, extend_pedal=True)
                        tech = self._extract_technique_name(os.path.basename(p))
                        for ev in note_events:
                            onset = float(ev['onset_time']); offset = float(ev['offset_time'])
                            if offset <= onset:
                                continue
                            self.items.append((p, onset, offset, tech))
                            if mini_data and len(self.items) >= 200:
                                break
                except Exception:
                    continue
                if mini_data and len(self.items) >= 200:
                    break

            logging.info(f"MOSAPTNoteDataset[H5]: total notes = {len(self.items)} from {self.root_dir}")

    def __len__(self):
        return len(self.paths) if getattr(self, 'mode', 'h5') == 'wav' else len(self.items)

    def _extract_technique_name(self, name: str) -> str:
        base = os.path.splitext(os.path.basename(name))[0]
        # Prefer 'tech-XYZ' pattern when available
        if 'tech-' in base:
            try:
                tech = base.split('tech-')[1].split('_')[0].lower()
            except Exception:
                tech = ''
        else:
            parts = base.split('_')
            tech = parts[-1].lower() if parts else ''
        if tech not in self.technique_classes:
            tech = 'no_technique'
        return tech

    def __getitem__(self, idx):
        if getattr(self, 'mode', 'h5') == 'wav':
            path = self.paths[idx]
            wav, sr = sf.read(path)
            if sr != self.sample_rate:
                wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=self.sample_rate)
            wav = wav.astype(np.float32)
            if self.segment_samples is not None:
                wav = pad_truncate_sequence(wav, self.segment_samples)
            if self.augmentor is not None:
                try:
                    wav = self.augmentor.augment(wav)
                    if self.segment_samples is not None:
                        wav = pad_truncate_sequence(wav, self.segment_samples)
                except Exception:
                    pass
            if wav.size > 0:
                wav = wav / (np.max(np.abs(wav)) + 1e-8)

            tech = self._extract_technique_name(os.path.basename(path))
            one_hot = np.zeros(len(self.technique_classes), dtype=np.float32)
            one_hot[self.technique_classes.index(tech)] = 1.0
            return {
                'waveform': wav,
                'technique': one_hot[None, :],
            }

        # H5 fallback
        import h5py
        h5_path, onset_s, offset_s, tech = self.items[idx]
        with h5py.File(h5_path, 'r') as hf:
            start = int(onset_s * self.sample_rate)
            end = int(offset_s * self.sample_rate)
            if (self.fixed_seconds is not None) and (self.segment_samples is not None):
                center = (start + end) // 2
                half = self.segment_samples // 2
                s0 = max(0, center - half)
                s1 = s0 + self.segment_samples
                if s1 > hf['waveform'].shape[0]:
                    s0 = max(0, hf['waveform'].shape[0] - self.segment_samples)
                    s1 = s0 + self.segment_samples
                segment = int16_to_float32(hf['waveform'][s0:s1])
            else:
                segment = int16_to_float32(hf['waveform'][start:end])

        if self.augmentor is not None:
            try:
                segment = self.augmentor.augment(segment)
            except Exception:
                pass

        if self.segment_samples is not None:
            segment = pad_truncate_sequence(segment, self.segment_samples)
        if segment.size > 0:
            segment = segment / (np.max(np.abs(segment)) + 1e-8)

        one_hot = np.zeros(len(self.technique_classes), dtype=np.float32)
        one_hot[self.technique_classes.index(tech)] = 1.0
        return {
            'waveform': segment.astype(np.float32),
            'technique': one_hot[None, :],
        }

def get_rwc_fixed_file_split(fold_id: int = 0):
    """Return fixed, technique-balanced split for RWC files.

    If fold_id in {0,1,2} is provided, returns the corresponding 3-fold CV split
    (balanced per technique). Otherwise defaults to fold 0 behavior.

    Returns:
      (train_files, test_files): two lists of 'file_XXX' keys
    """
    # Technique buckets
    fl = ['file_006', 'file_031', 'file_032']
    nm = ['file_001', 'file_002', 'file_004', 'file_005', 'file_007', 'file_010', 'file_019', 'file_022', 'file_024', 'file_025', 'file_029', 'file_030']
    pz = ['file_000', 'file_008', 'file_009', 'file_015', 'file_016', 'file_017', 'file_018', 'file_023', 'file_028']
    sp = ['file_003', 'file_011', 'file_012', 'file_013', 'file_014', 'file_020', 'file_021', 'file_026', 'file_027']

    # Define 3 folds test sets (balanced per technique):
    # - flageolet: 3 files -> 1 per fold
    # - normal:   12 files -> 4 per fold
    # - pizzicato: 9 files -> 3 per fold
    # - spiccato:  9 files -> 3 per fold
    fold_tests = {
        0: [
            # fl
            'file_006',
            # nm (4)
            'file_001', 'file_005', 'file_019', 'file_025',
            # pz (3)
            'file_000', 'file_015', 'file_018',
            # sp (3)
            'file_003', 'file_013', 'file_021',
        ],
        1: [
            # fl
            'file_031',
            # nm (4)
            'file_002', 'file_007', 'file_022', 'file_029',
            # pz (3)
            'file_008', 'file_016', 'file_023',
            # sp (3)
            'file_011', 'file_014', 'file_026',
        ],
        2: [
            # fl
            'file_032',
            # nm (4)
            'file_004', 'file_010', 'file_024', 'file_030',
            # pz (3)
            'file_009', 'file_017', 'file_028',
            # sp (3)
            'file_012', 'file_020', 'file_027',
        ],
    }

    if fold_id == -1:
        return fl + nm + pz + sp, fl + nm + pz + sp

    # All files set
    all_files = set(fl + nm + pz + sp)
    k = fold_id if fold_id in fold_tests else 0
    test_files = sorted(fold_tests[k])
    train_files = sorted(list(all_files.difference(test_files)))
    return train_files, test_files

''' Test CustomDataset and CustomSampler 

run python utils/data_generator.py 

'''
if __name__ == '__main__':
    """Test function for CustomDataset"""
    
    def test_custom_dataset():
        """Test the CustomDataset functionality"""
        print("Testing CustomDataset...")
        
        # Test parameters
       #  hdf5s_dir = '/mnt/gestalt/home/tkwang/ViolinMamba/hdf5s/mosa'  # Adjust this path as needed
        hdf5s_dir = '/home/yuehpo/coding/violin-mamba/hdf5s/mosapt'
        segment_seconds = 100.0
        frames_per_second = 100
        hop_seconds = 1
        batch_size = 16
        max_note_shift = 0
        mini_data = True
        
        try:
            # Test CustomSampler
            print(f"Creating CustomSampler with:")
            print(f"  hdf5s_dir: {hdf5s_dir}")
            print(f"  segment_seconds: {segment_seconds}")
            print(f"  hop_seconds: {hop_seconds}")
            print(f"  batch_size: {batch_size}")
            
            sampler = CustomSampler(
                hdf5s_dir=hdf5s_dir,
                split='train',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data
            )
            
            print(f"✓ CustomSampler created successfully")
            print(f"  Total segments: {len(sampler.segment_list)}")
            
            # Test CustomDataset
            print(f"\nCreating CustomDataset with:")
            print(f"  max_note_shift: {max_note_shift}")
            
            dataset = CustomDataset(
                hdf5s_dir=hdf5s_dir,
                segment_seconds=segment_seconds,
                frames_per_second=frames_per_second,
                max_note_shift=max_note_shift,
                augmentor=None,
                include_technique_label=True
            )
            
            print(f"✓ CustomDataset created successfully")
            
            # Test data loading
            print(f"\nTesting data loading...")
            sampler_iter = iter(sampler)
            batch_meta = next(sampler_iter)
            
            print(f"  Got batch with {len(batch_meta)} samples")
            print(f"  First sample meta: {batch_meta[0]}")
            print(f"  Second sample meta: {batch_meta[1]}")
            print(f"  Third sample meta: {batch_meta[6]}")
            print(f"  Fourth sample meta: {batch_meta[10]}")
            
            # Test getting data from dataset
            data_dict = dataset[batch_meta[0]]
            second_data_dict = dataset[batch_meta[1]]
            third_data_dict = dataset[batch_meta[6]]
            fourth_data_dict = dataset[batch_meta[10]]

            print(f"  First sample data_dict: {data_dict}")
            
            print(f"\n✓ Successfully loaded data sample:")
            print(f"  Waveform shape: {data_dict['waveform'].shape}")
            print(f"  Sample rate: {dataset.sample_rate}")
            print(f"  Segment samples: {dataset.segment_samples}")
            
            # Print available keys in data_dict
            print(f"  Available data keys: {list(data_dict.keys())}")
            
            # Test shapes of different arrays
            for key, value in data_dict.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")
                else:
                    print(f"    {key}: {type(value)}")
            
            # Test collate function
            print(f"\nTesting collate function...")
            list_data_dict = []
            for meta in batch_meta[:2]:  # Test with 2 samples
                list_data_dict.append(dataset[meta])
            
            batch_data = collate_fn(list_data_dict)
            print(f"✓ Collate function successful")
            print(f"  Batch waveform shape: {batch_data['waveform'].shape}")
            
            
            print(f"\n🎉 All CustomDataset tests passed!")

            # save 4 audio sample for debugging
            
            audio_path1 = f'/home/yuehpo/coding/violin-mamba/debug_audio_1.wav'
            audio_path2 = f'/home/yuehpo/coding/violin-mamba/debug_audio_2.wav'
            audio_path3 = f'/home/yuehpo/coding/violin-mamba/debug_audio_3.wav'
            audio_path4 = f'/home/yuehpo/coding/violin-mamba/debug_audio_4.wav'
            sf.write(audio_path1, data_dict['waveform'], samplerate=config.sample_rate)
            sf.write(audio_path2, second_data_dict['waveform'], samplerate=config.sample_rate)
            sf.write(audio_path3, third_data_dict['waveform'], samplerate=config.sample_rate)
            sf.write(audio_path4, fourth_data_dict['waveform'], samplerate=config.sample_rate)

        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print(f"   Please check that the hdf5s directory exists and contains data files")
            print(f"   Current test directory: {hdf5s_dir}")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    def test_custom_test_sampler():
        """Test the CustomTestSampler functionality"""
        print("\n" + "="*50)
        print("Testing CustomTestSampler...")
        
        # hdf5s_dir = '/mnt/gestalt/home/tkwang/ViolinMamba/hdf5s/mosa'
        hdf5s_dir = '/home/yuehpo/coding/violin-mamba/hdf5s/mosa'
        segment_seconds = 10.0
        hop_seconds = 1
        batch_size = 2
        mini_data = True
        
        try:
            test_sampler = CustomTestSampler(
                hdf5s_dir=hdf5s_dir,
                split='test',
                segment_seconds=segment_seconds,
                hop_seconds=hop_seconds,
                batch_size=batch_size,
                mini_data=mini_data
            )
            
            print(f"✓ CustomTestSampler created successfully")
            print(f"  Max evaluate iterations: {test_sampler.max_evaluate_iteration}")
            
            # Test iteration
            iteration_count = 0
            for batch_meta in test_sampler:
                iteration_count += 1
                print(f"  Iteration {iteration_count}: batch size {len(batch_meta)}")
                if iteration_count >= 3:  # Test first 3 iterations
                    break
            
            print(f"✓ CustomTestSampler iteration test passed")
            
        except Exception as e:
            print(f"❌ CustomTestSampler test failed: {e}")
    
    # Run the tests
    test_custom_dataset()

    a=input("Press Enter to test_custom_test_sampler ...")

    test_custom_test_sampler()
    
    print(f"\n" + "="*50)
    print("Test completed!")
    print("Note: If you get file not found errors, please:")
    print("1. Check that the hdf5s/maestro directory exists")
    print("2. Ensure it contains properly formatted HDF5 files")
    print("3. Adjust the hdf5s_dir path in the test function if needed")
    
    
    
