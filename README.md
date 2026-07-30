# VioTech — Frame-Level Violin Playing Technique Detection with Multi-Scale Mixture-of-Experts

> Official code release for our **ISMIR 2026** paper:
> *VioTech: Frame-Level Violin Playing Technique Detection with Multi-Scale Mixture-of-Experts.*
> Ting-Kang Wang, Vincent K.M. Cheung, Nan-Tien Lai, Li Su.

VioTech detects violin playing techniques **directly at the frame level** (10 ms / frame), removing
the note pre-segmentation stage and its cascading onset-detection errors. It predicts three
technique dimensions in parallel — **tonal technique** (none / pizzicato / harmonics),
**articulation** (none / spiccato), and **legato** (bow-change vs. sustained, framed as a binary
regression problem).

The proposed model couples an **HRPT transcription backbone** (providing onset/offset/frame cues)
with a **frozen MERT-v1-330M** acoustic backbone, on top of which **N multi-scale temporal experts**
(default `N = 8`, receptive fields 20 ms – 2560 ms) are routed by **per-task gating**.

The full method, architecture, and result tables are in the paper PDF and
[`ISMIR_PAPER_REFERENCE.md`](ISMIR_PAPER_REFERENCE.md).

---

## Highlights

- **Frame-level, cascade-free** technique detection — no note onset/offset segmentation required.
- **Multi-scale Mixture-of-Experts** over a frozen MERT-v1-330M backbone; each expert keeps
  independent MERT layer weights and its own temporal receptive field.
- **Per-task gating** for tonal / articulation / legato, with dense soft routing.
- **First computational model of legato bow-change timing**, via a Gaussian-peak binary regression target.
- **VioTech120 dataset** — 120 manually annotated violin solo etudes (Paganini Op.1, Wohlfahrt Op.35,
  Kayser Op.20), with a dedicated annotation interface (see below).

---

## Environment

- Python 3.10, CUDA 11.8, PyTorch 2.6.0. A single GPU suffices.
- `transformers` + `torchaudio` are used to load the frozen **MERT-v1-330M** backbone
  (`m-a-p/MERT-v1-330M`), which is downloaded automatically from the Hugging Face Hub on first run.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact package versions used for all reported experiments.

### Checkpoints

Only one local checkpoint is needed under `checkpoints/`:

| File | Purpose |
| --- | --- |
| `transcriptor_model.pth` | HRPT transcription backbone (pretrained on violin data). Used as the fine-tuning starting point; the transcription branch is jointly fine-tuned during technique training. |

The MERT-v1-330M encoder is **frozen** and fetched from Hugging Face — it is not stored in this repo.

---

## Data Preparation

Training combines **VioTech120** with **MOSA-VPT** (8:2 ratio); **RWC** is used for cross-dataset
evaluation. Raw annotations (`annotation_revised.csv` + audio) are packed into HDF5 first.

```bash
# VioTech120 → HDF5  (also writes viotech_dataset_summary.txt)
python3 prepare_viotech.py --data_dir /path/to/viotech --output_dir ./hdf5s/viotech

# MOSA-VPT   → HDF5  (auxiliary dataset for mixed training)
python3 prepare_mosavpt.py --data_dir /path/to/MOSA_VPT --output_dir ./hdf5s/mosavpt \
                           --mapping_file mapping_config.json

# RWC note-level clips for cross-dataset evaluation
python3 rwc_export_note_dataset.py
```

Label definitions live in `config/technique_label_config*.json` (`_reduced` is the merged class set
used for training) and `mapping_config.json`.

Two HDF5-packing helpers for the auxiliary corpora are provided as shell wrappers:
`scripts/run_MOSA_preprocessing.sh` and `scripts/run_MUSC_preprocessing.sh`.

---

## Training

The proposed model is the `mert_moe` experiment. Training and all paper ablations are driven by a
single launcher:

```bash
# Proposed VioTech model (frozen MERT-v1-330M + 8-expert Multi-Scale MoE)
bash scripts/ablation_expert_composition_v2.sh mert_moe

# Component ablations (Tables 1 & 2)
bash scripts/ablation_expert_composition_v2.sh mert_no_layer_weight  # w/o MERT layer weight
bash scripts/ablation_expert_composition_v2.sh mert_no_cues          # w/o transcription cue
bash scripts/ablation_expert_composition_v2.sh mert_no_moe           # w/o MoE

# Expert-count scaling (Fig. per-class F1 vs. #experts); repeat 3x with different seeds
bash scripts/ablation_expert_composition_v2.sh mert_moe_n2   # 2 experts
bash scripts/ablation_expert_composition_v2.sh mert_moe_n4   # 4 experts
bash scripts/ablation_expert_composition_v2.sh mert_moe_n8   # 8 experts (= mert_moe)
bash scripts/ablation_expert_composition_v2.sh mert_moe_n16  # 16 experts
bash scripts/ablation_expert_composition_v2.sh mert_moe_n32  # 32 experts

# Run every ablation sequentially (optionally pass device and seed)
bash scripts/ablation_expert_composition_v2.sh all 0 2026
```

Under the hood this calls `piano_transcription/pytorch/main_contrast.py train --fmoe_mert_moe=1`,
which sets the 8-expert temporal pyramid `2:1,4:1,8:1,16:1,32:1,64:1,128:1,256:1` and loads the
transcription backbone from `checkpoints/transcriptor_model.pth`.

Key training settings (paper §5.2): Adam, `lr=5e-4`, `batch_size=8`, LR reduction every 1000 iters,
early stopping with `patience=5000` (max 20000 iters), best checkpoint by validation technique
macro-F1, pitch-shift ±2 semitones augmentation, 10-second segments, seed `2026`.

---

## Evaluation

Evaluate a trained checkpoint (frame-level metrics, per-class accuracy/F1, gate-routing analysis,
and — for MERT-MoE — UMAP embedding plots), including cross-dataset RWC:

```bash
# Frame-level evaluation of the proposed model
bash scripts/ablation_eval_expert_composition_v2.sh mert_moe

# UMAP embeddings only
bash scripts/ablation_eval_expert_composition_v2.sh --umap mert_moe

# Evaluate all experiments and build a summary table
bash scripts/ablation_eval_expert_composition_v2.sh all
```

This calls `piano_transcription/pytorch/evaluate_frame_moe.py` (metrics + JSON export) and
`piano_transcription/pytorch/visualize_umap_frame_moe.py` (UMAP). Per-experiment JSON results are
written under `results/`.

### Figures

```bash
python3 scripts/export_figures.py <results_subfolder>   # e.g. ablation_expert_composition_v2
```

Pre-computed MERT-MoE experiment outputs (JSON + figures) for the paper are kept under
`results/ablation_expert_composition_v2/` and `results/ablation_expert_composition_v2_10000/`.

---

## Results (summary)

Per-class accuracy on the RWC cross-dataset test set (paper Table 2). Columns: Tonal none/pizz/harm
(TN/TP/TH), Articulation none/spiccato (AN/AS), Legato sustained/bow-change (LS/LB).

| Model | TN | TP | TH | AN | AS | LS | LB | Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **VioTech (Ours)** | 79.07 | **89.60** | **88.84** | 97.78 | **97.16** | 96.49 | 56.73 | **86.52** |
| w/o MoE | **85.98** | 40.85 | 69.62 | 94.43 | 61.41 | **98.72** | 68.96 | 74.28 |
| w/o Layer Weight | 86.11 | 93.68 | 65.44 | 98.74 | 83.66 | 97.00 | 57.65 | 83.18 |
| w/o Transcription Cue | 79.83 | 95.28 | 83.28 | 99.57 | 64.51 | 97.00 | 64.21 | 83.38 |

The MoE routing yields large gains on the expressive, less frequent classes (pizzicato, harmonics,
spiccato), especially under cross-dataset transfer.

---

## Annotation Interface

VioTech120 was labelled with a purpose-built piano-roll web tool that lets violinists assign
note-level technique labels (and violin position / fingering) against synchronized audio, producing
the `annotation_revised.csv` format used here.

**→ https://github.com/tklovln/audio-midi-marker**

---

## Project Structure

```
VioTech/
├── config/                              # technique label configs + string→class mapping
│   ├── technique_label_config.json          # full 6-class definition
│   └── technique_label_config_reduced.json  # merged classes used for training
├── scripts/
│   ├── ablation_expert_composition_v2.sh    # ★ training launcher (mert_moe + ablations)
│   ├── ablation_eval_expert_composition_v2.sh  # ★ evaluation + UMAP launcher
│   ├── export_figures.py                    # render result figures
│   ├── run_MOSA_preprocessing.sh            # pack MOSA-VPT → HDF5
│   └── run_MUSC_preprocessing.sh            # pack MuSC → HDF5
├── piano_transcription/
│   ├── pytorch/
│   │   ├── main_contrast.py             # training entry point
│   │   ├── models_contrast.py           # model factory (HRPT backbone + technique heads)
│   │   ├── mert_moe_backbone.py         # ★ MERT Multi-Scale MoE (proposed model)
│   │   ├── mert_backbone.py             # frozen-MERT backbone (w/o MoE ablation)
│   │   ├── moe_frame_multiscale.py      # shared MoE building blocks (_TemporalCtx/_Expert/_Gate) + losses
│   │   ├── losses.py                    # loss functions
│   │   ├── note_slicer.py               # note slicing utility
│   │   ├── evaluate.py                  # in-training evaluator (SegmentEvaluator, RWC eval)
│   │   ├── evaluate_frame_moe.py        # frame/event-level evaluation + JSON export
│   │   ├── visualize_umap_frame_moe.py  # UMAP of frame-level embeddings
│   │   ├── rwc_moe_utils.py             # RWC cross-dataset dataset/sampler
│   │   └── pytorch_utils.py
│   └── utils/                           # config, data_generator, features, utilities, labels, piano_vad
├── prepare_viotech.py                   # VioTech120 → HDF5
├── prepare_mosavpt.py                   # MOSA-VPT → HDF5
├── rwc_export_note_dataset.py           # RWC note-level clips
├── mapping_config.json                  # CSV string → class-id mapping
├── requirements.txt                     # pinned Python environment
├── checkpoints/transcriptor_model.pth   # pretrained transcription backbone
├── results/                             # MERT-MoE experiment outputs (JSON + figures)
├── ISMIR_PAPER_REFERENCE.md             # extended method / results reference
└── LICENSE
```

## License

See [`LICENSE`](LICENSE).
