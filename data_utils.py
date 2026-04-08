import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat


def get_song_folders_bach10():
    """Get all song folders in Bach10 dataset."""
    dataset_dir = Path(os.getenv("BACH10_DIR", "/mnt/hdd/Bach10"))
    song_folders = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            # Check if this folder contains the expected audio and midi files
            song_name = item
            audio_pattern = os.path.join(item_path, f"{song_name}-violin.wav")
            midi_pattern = os.path.join(item_path, f"{song_name}-GTNotes.mat")
            
            if os.path.exists(audio_pattern) and os.path.exists(midi_pattern):
                song_folders.append({
                    'song_name': song_name,
                    'audio_path': audio_pattern,
                    'midi_path': midi_pattern,
                    'align_path': None
                })
            else:
                print(f"Warning: Missing files in {item_path}")
        
                
    return song_folders

def get_song_folders_bvd(): # Bach-Violin-Dataset

    bvd_path = Path("/mnt/gestalt/home/tkwang/ViolinMamba/bach-violin-dataset/bach-violin")
    bvd_audio_path = bvd_path / "audio"
    bvd_align_path = bvd_path / "alignments"
    bvd_note_path = bvd_path / "notes"

    # iterate over all directory in audio_path and get the name of the directory
    # use the name of the directory to get the alignment file

    # Iterate over all directories in the audio path


    bvd_audio_dirs = ["/".join(audio_file.parent.parts[-1:] + (audio_file.name,)) for audio_file in bvd_audio_path.rglob("*.mp3")]
    print(bvd_audio_dirs)

    song_folders = []
    for audio_name in bvd_audio_dirs:
        # Find the corresponding alignment file
        
        audio_file = bvd_audio_path / audio_name
        align_file = bvd_align_path / audio_name.replace('.mp3', '.csv')
        note_file = bvd_note_path / audio_name.replace('.mp3', '.csv')


        
        if audio_file.exists() and align_file.exists() and note_file.exists():
            
            
            
            song_folders.append({
                    'song_name': audio_file.stem,
                    'audio_path': audio_file,
                    'midi_path': note_file,
                    'align_path': align_file
                })

        else:
            print(f"Warning: Missing files in {audio_file}")

    return song_folders

def get_song_folders_urmp(): # URMP Dataset
    urmp_path = Path(os.getenv("URMP_DIR", "/mnt/hdd/URMP_Violin/Processed/Dataset"))

    # Recursively find all files matching Notes_*.txt in parent directories using pathlib
    note_files = list(urmp_path.rglob("Notes_*.txt"))
    tatal_song = len(note_files)
    total_song_get = 0
    vn_song_note_files = []
    vn_song_audio_files = []
    # split the folder 01_Jupiter_vn_vc and split by "_"

    for note_file in note_files:
        song_name = note_file.name
        song_name_parts = song_name.split("_")
        # print(song_name_parts[:2])
        # print(song_name_parts[2:])

        if song_name_parts[2] == "vn" :
            total_song_get += 1
            vn_song_note_files.append(note_file)

            # e.g. AuSep_1_vn_01_Jupiter.wav
            vn_song_audio_files.append(note_file.parent / note_file.name.replace("Notes", "AuSep").replace(".txt", ".wav"))

    song_folders = []
    for audio_path, note_path in zip(vn_song_audio_files, vn_song_note_files):
        # Find the corresponding alignment file
        if audio_path.exists() and note_path.exists():
            
            song_folders.append({
                    'song_name': audio_path.stem,
                    'audio_path': audio_path,
                    'midi_path': note_path,
                    'align_path': None
                })

        else:
            print(f"Warning: Missing somethings")

    return song_folders

def get_ref_note_events(midi_align_path):
    # load txt line-by-line
    with open(midi_align_path, 'r') as f:
        midi_audio_lines = f.readlines()


    # convert all to a list of ints
    midi_audio_lines = [line.split('\t') for line in midi_audio_lines]
    midi_audio_lines = [[int(x) for x in line] for line in midi_audio_lines]

    # print the list
    # filter all items that the last column is 1
    midi_audio_lines = [line for line in midi_audio_lines if line[-1] == 1]

    # dismiss the last column
    midi_audio_lines = [line[:-1] for line in midi_audio_lines]

    # extract the list column to form another list, called ref_pitches, and the rest to form another list, called ref_onsets
    ref_pitches = [line[2] for line in midi_audio_lines]
    ref_on_off_pairs = [line[:2] for line in midi_audio_lines]

    # convert on off time from ms to second
    ref_on_off_pairs = np.array(ref_on_off_pairs) / 1000

    assert len(ref_pitches) == len(ref_on_off_pairs)

    return np.array(ref_on_off_pairs), np.array(ref_pitches)

def get_ref_note_events_bach10(midi_align_path):

    HOP_MS = 10

    ref_on_off_pairs = []
    ref_pitches = []

    mat = loadmat(midi_align_path)
    gt_notes = mat['GTNotes']  # shape: 1 x 4 cell array for each instrument
    violin_notes = gt_notes[0, 0]  # violin = index 0
    for i in range(violin_notes.shape[0]):
        note = violin_notes[i][0]  # extract the 2-row matrix
        onset_frame = note[0][0]
        offset_frame = note[0][-1]
        pitch = note[1][0]

        onset_ms_in_sec = onset_frame * HOP_MS / 1000
        offset_ms_in_sec = offset_frame * HOP_MS / 1000

        ref_on_off_pairs.append([onset_ms_in_sec, offset_ms_in_sec])
        ref_pitches.append(pitch)

    ref_on_off_pairs = np.array(ref_on_off_pairs)
    ref_pitches = np.array(ref_pitches)

    return np.array(ref_on_off_pairs), np.array(ref_pitches), None

def get_ref_note_events_bvd(note_path, align_path):
    # load csv , column = ['onset', 'offset', 'pitch', 'velocity']

    note_df = pd.read_csv(note_path)
    align_df = pd.read_csv(align_path)


    assert len(align_df) == len(note_df)

    # replace onset and offset in note_df with align_df's start and end
    note_df['onset'] = align_df['start']
    note_df['offset'] = align_df['end']

    ref_on_off_pairs = note_df[['onset', 'offset']].values
    ref_pitches = note_df['pitch'].values
    ref_velocities = note_df['velocity'].values

    return np.array(ref_on_off_pairs), np.array(ref_pitches), np.array(ref_velocities)

def get_ref_note_events_urmp(note_path):
    """
    Parse a URMP note file (txt) where each line is: onset, pitch, duration
    Returns:
        ref_on_off_pairs: np.ndarray of shape (N, 2), onset and offset times in seconds
        ref_pitches: np.ndarray of shape (N,), MIDI note numbers
        ref_velocities: np.ndarray of shape (N,), dummy velocities (set to 100)
    """
    import numpy as np

    onsets = []
    offsets = []
    pitches = []
    velocities = []

    with open(note_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try splitting by comma or whitespace
            if ',' in line:
                parts = [x.strip() for x in line.split(',')]
            else:
                parts = line.split()
            if len(parts) < 3:
                continue  # skip malformed lines
            onset = float(parts[0])
            pitch = float(parts[1])
            duration = float(parts[2])
            offset = onset + duration
            onsets.append(onset)
            offsets.append(offset)
            pitches.append(pitch)
            velocities.append(80)  # URMP txt does not have velocity, use dummy

    ref_on_off_pairs = np.stack([onsets, offsets], axis=1)
    ref_pitches = np.array(pitches)
    ref_velocities = np.array(velocities)
    return ref_on_off_pairs, ref_pitches, ref_velocities


def get_song_folders_mosapt(prefix='ba3'):
    """Get song folders for MOSAPT dataset from HDF5 files."""
    hdf5s_dir = "/home/yuehpo/coding/VioPTT/data/hdf5s/mosapt_ssv_4"
    song_folders = []

    for item in os.listdir(hdf5s_dir):
        if item.startswith(prefix) and item.endswith('.h5'):
            item_path = os.path.join(hdf5s_dir, item)
            song_name = os.path.splitext(item)[0]

            song_folders.append({
                'song_name': song_name,
                'audio_path': item_path,  # Use H5 path as audio_path
                'midi_path': item_path,   # Use H5 path as midi_path
                'align_path': None
            })

    return song_folders


def get_ref_note_events_mosapt(h5_path):
    """Extract reference note events from MOSAPT HDF5 file."""
    import h5py
    from utilities import TargetProcessor
    import config

    with h5py.File(h5_path, 'r') as hf:
        midi_events = [e.decode() if isinstance(e, bytes) else e for e in hf['midi_event'][:]]
        midi_times = hf['midi_event_time'][:]

        # Determine duration
        if 'duration' in hf.attrs:
            duration_sec = float(hf.attrs['duration'])
        else:
            sample_rate = config.sample_rate
            duration_sec = len(hf['waveform']) / sample_rate

        tp = TargetProcessor(
            segment_seconds=duration_sec,
            frames_per_second=config.frames_per_second,
            begin_note=config.begin_note,
            classes_num=config.classes_num
        )

        # Process MIDI events to get note events
        _, note_events, _ = tp.process(0.0, midi_times, midi_events, extend_pedal=True)

        ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
        ref_midi_notes = np.array([event['midi_note'] for event in note_events])
        ref_velocities = np.array([event['velocity'] for event in note_events])

    return ref_on_off_pairs, ref_midi_notes, ref_velocities
