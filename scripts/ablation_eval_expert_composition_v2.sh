#!/bin/bash
#
# Expert Composition Ablation v2 — Evaluation (Unified Cues)
#
# Evaluates v2 checkpoints (ablation_fmoe_v2_*) which all use unified cue injection.
# Prints checkpoint info, saves per-experiment JSON, and summary table.
#
# Usage:
#   bash scripts/ablation_eval_expert_composition_v2.sh <experiment_id> [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh all [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh <id> [device] <checkpoint>
#   bash scripts/ablation_eval_expert_composition_v2.sh --umap <experiment_id> [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh --umap all [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh --event_level <experiment_id> [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh --event_level all [device]

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

UMAP_ONLY=0
EVENT_LEVEL=0
if [ "${1:-}" = "--umap" ]; then
    UMAP_ONLY=1
    shift
elif [ "${1:-}" = "--event_level" ]; then
    EVENT_LEVEL=1
    shift
fi

EXPERIMENT="${1:?Usage: $0 [--umap|--event_level] <experiment_id> [device] [checkpoint]}"
DEVICE="${2:-0}"
MANUAL_CKPT="${3:-}"

RWC_H5="/mnt/hdd/rwc_processed_data.h5"
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech_paganini_kayser_with_openstring"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced_openstring.json"
BINARY_LABEL_CFG="${WORKSPACE}/config/technique_label_config_binary.json"
SPLIT="test"
RESULTS_DIR="${WORKSPACE}/results/ablation_expert_composition_v2"
TEMPORAL_EXPERTS="11:1,51:1,51:3"
mkdir -p "$RESULTS_DIR"

resolve_experiment() {
    local exp="$1"
    DIRECT_HEAD=0
    HIERARCHICAL=0
    MULTI_CRNN=0
    CRNN_MOE=0
    SHARED_CRNN_MOE=0
    HIER_MULTILEVEL=0
    HIER_MOE=0
    BINARY_TECHNIQUE=0
    MERT_BACKBONE=0
    EFFECTIVE_LABEL_CFG="$LABEL_CFG"
    case "$exp" in
        baseline)    TAG="ablation_fmoe_openstring_baseline_e012";         EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        full)        TAG="ablation_fmoe_openstring_full_e0123_spectral";   EXPERT_MASK="0,1,2,3"; SPECTRAL=1 ;;
        # no_onset)    TAG="ablation_fmoe_openstring_no_onset_e12";          EXPERT_MASK="1,2";   SPECTRAL=0 ;;
        # no_note)     TAG="ablation_fmoe_openstring_no_note_e02";           EXPERT_MASK="0,2";   SPECTRAL=0 ;;
        # no_phrase)   TAG="ablation_fmoe_openstring_no_phrase_e01";         EXPERT_MASK="0,1";   SPECTRAL=0 ;;
        # only_onset)  TAG="ablation_fmoe_openstring_only_onset_e0";         EXPERT_MASK="0";     SPECTRAL=0 ;;
        # only_note)   TAG="ablation_fmoe_openstring_only_note_e1";          EXPERT_MASK="1";     SPECTRAL=0 ;;
        # only_phrase) TAG="ablation_fmoe_openstring_only_phrase_e2";        EXPERT_MASK="2";     SPECTRAL=0 ;;
        tech0.1)     TAG="ablation_fmoe_tech0.1_baseline_e012";           EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        detach_cues) TAG="ablation_fmoe_tech0.1_detach_cues_e012";        EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        freeze_trans) TAG="ablation_fmoe_tech0.1_freeze_trans_e012";      EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        no_moe)      TAG="ablation_fmoe_tech0.1_no_moe_direct";           EXPERT_MASK="0,1,2"; SPECTRAL=0; DIRECT_HEAD=1 ;;
        hierarchical) TAG="ablation_fmoe_tech0.1_hierarchical";           EXPERT_MASK="";      SPECTRAL=0; HIERARCHICAL=1 ;;
        multi_crnn)  TAG="ablation_fmoe_tech0.1_multi_crnn_e012";         EXPERT_MASK="0,1,2"; SPECTRAL=0; MULTI_CRNN=1 ;;
        crnn_moe)    TAG="ablation_fmoe_tech0.1_crnn_moe_e01234567";      EXPERT_MASK="";      SPECTRAL=0; CRNN_MOE=1 ;;
        shared_crnn_moe) TAG="ablation_fmoe_tech0.1_shared_crnn_moe_e01234567"; EXPERT_MASK=""; SPECTRAL=0; SHARED_CRNN_MOE=1 ;;
        hier_multilevel) TAG="ablation_fmoe_tech0.1_hier_multilevel"; EXPERT_MASK=""; SPECTRAL=0; HIER_MULTILEVEL=1 ;;
        hier_moe)        TAG="ablation_fmoe_tech0.1_hier_moe";        EXPERT_MASK=""; SPECTRAL=0; HIER_MOE=1 ;;
        binary_baseline) TAG="ablation_fmoe_tech0.1_binary_baseline_e012"; EXPERT_MASK="0,1,2"; SPECTRAL=0; BINARY_TECHNIQUE=1; EFFECTIVE_LABEL_CFG="$BINARY_LABEL_CFG" ;;
        binary_full)     TAG="ablation_fmoe_tech0.1_binary_full_e0123";    EXPERT_MASK="0,1,2,3"; SPECTRAL=1; BINARY_TECHNIQUE=1; EFFECTIVE_LABEL_CFG="$BINARY_LABEL_CFG" ;;
        binary_no_moe)   TAG="ablation_fmoe_tech0.1_binary_no_moe";       EXPERT_MASK="0,1,2"; SPECTRAL=0; DIRECT_HEAD=1; BINARY_TECHNIQUE=1; EFFECTIVE_LABEL_CFG="$BINARY_LABEL_CFG" ;;
        binary_detach_cues) TAG="ablation_fmoe_tech0.1_binary_detach_cues_e012"; EXPERT_MASK="0,1,2"; SPECTRAL=0; BINARY_TECHNIQUE=1; EFFECTIVE_LABEL_CFG="$BINARY_LABEL_CFG" ;;
        mert_baseline) TAG="ablation_fmoe_tech0.1_mert_baseline_e012"; EXPERT_MASK="0,1,2"; SPECTRAL=0; MERT_BACKBONE=1 ;;
        mert_binary) TAG="ablation_fmoe_tech0.1_mert_binary_e012"; EXPERT_MASK="0,1,2"; SPECTRAL=0; MERT_BACKBONE=1; BINARY_TECHNIQUE=1; EFFECTIVE_LABEL_CFG="$BINARY_LABEL_CFG" ;;
        *)
            echo "Unknown experiment: $exp"
            return 1 ;;
    esac
}

find_checkpoint() {
    local tag="$1"
    local base="${WORKSPACE}/checkpoints/main_contrast/${tag}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2"

    local ckpt_dir
    for bs in 4 8; do
        ckpt_dir="${base}/batch_size=${bs}"

        if [ -f "${ckpt_dir}/best_model.pth" ]; then
            echo "${ckpt_dir}/best_model.pth"
            return 0
        fi

        local latest
        latest=$(ls -t "${ckpt_dir}"/*_iterations.pth 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "$latest"
            return 0
        fi
    done

    echo ""
    return 1
}

_resolve_checkpoint_and_args() {
    local exp="$1"
    resolve_experiment "$exp" || return 1

    if [ -n "$MANUAL_CKPT" ]; then
        CHECKPOINT="$MANUAL_CKPT"
    else
        CHECKPOINT=$(find_checkpoint "$TAG")
        if [ -z "$CHECKPOINT" ]; then
            echo "[SKIP] No checkpoint found for ${exp} (${TAG})"
            return 1
        fi
    fi

    CKPT_TYPE="latest"
    if [[ "$CHECKPOINT" == *"best_model.pth"* ]]; then
        CKPT_TYPE="best"
    fi

    EXTRA_ARGS=""
    [ -n "$EXPERT_MASK" ] && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"
    [ "$DIRECT_HEAD" = "1" ] && EXTRA_ARGS+=" --fmoe_direct_head=1"
    [ "$HIERARCHICAL" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical=1"
    [ "$MULTI_CRNN" = "1" ] && EXTRA_ARGS+=" --fmoe_multi_crnn=1"
    [ "$CRNN_MOE" = "1" ] && EXTRA_ARGS+=" --fmoe_crnn_moe=1"
    [ "$SHARED_CRNN_MOE" = "1" ] && EXTRA_ARGS+=" --fmoe_shared_crnn_moe=1"
    [ "$HIER_MULTILEVEL" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical_multilevel=1"
    [ "$HIER_MOE" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical_moe=1"
    [ "$BINARY_TECHNIQUE" = "1" ] && EXTRA_ARGS+=" --fmoe_binary_technique=1"
    [ "$MERT_BACKBONE" = "1" ] && EXTRA_ARGS+=" --fmoe_mert_backbone=1"

    RWC_ARGS=""
    [ -f "$RWC_H5" ] && RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split all"
}

run_umap() {
    local exp="$1"
    _resolve_checkpoint_and_args "$exp" || return 0

    echo ""
    echo "============================================"
    echo "  UMAP only: ${exp}"
    echo "============================================"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "============================================"

    cd "${WORKSPACE}/piano_transcription"

    local UMAP_DIR="${RESULTS_DIR}/figures"
    mkdir -p "$UMAP_DIR"

    local UMAP_ARGS=""
    UMAP_ARGS+=" --checkpoint_path $CHECKPOINT"
    UMAP_ARGS+=" --hdf5s_dir $HDF5S_DIR"
    UMAP_ARGS+=" --split $SPLIT"
    UMAP_ARGS+=" --device $DEVICE"
    UMAP_ARGS+=" --batch_size 4"
    UMAP_ARGS+=" --num_workers 4"
    UMAP_ARGS+=" --fmoe_spectral_expert=${SPECTRAL}"
    UMAP_ARGS+=" --fmoe_temporal_experts=${TEMPORAL_EXPERTS}"
    UMAP_ARGS+=" --fmoe_shared_gate=0"
    UMAP_ARGS+=" --fmoe_uniform_routing=0"
    UMAP_ARGS+=" --technique_label_config=${EFFECTIVE_LABEL_CFG}"
    UMAP_ARGS+=" --output_dir $UMAP_DIR"
    UMAP_ARGS+=" --tag $exp"
    UMAP_ARGS+=" --n_per_class 1000"
    UMAP_ARGS+=" --max_batches 0"

    echo ""
    echo "  Generating UMAP plots for ${exp}..."
    python3 pytorch/visualize_umap_frame_moe.py \
        $UMAP_ARGS \
        $EXTRA_ARGS \
        $RWC_ARGS

    echo ""
    echo "--------------------------------------------"
    echo "  ${exp}: UMAP done"
    echo "  UMAP → ${UMAP_DIR}/umap_${exp}_*.png"
    echo "--------------------------------------------"
}

run_eval() {
    local exp="$1"
    _resolve_checkpoint_and_args "$exp" || return 0

    local MODE_STR="Frame-Level"
    local EVENT_LEVEL_ARG=""
    local JSON_SUFFIX=""
    if [ "$EVENT_LEVEL" = "1" ]; then
        MODE_STR="Event-Level"
        EVENT_LEVEL_ARG="--event_level"
        JSON_SUFFIX="_event_level"
    fi

    echo ""
    echo "============================================"
    echo "  Expert Ablation v2 Eval [${MODE_STR}]: ${exp}"
    echo "============================================"
    echo "  Tag        : ${TAG}"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "  Expert mask: ${EXPERT_MASK}"
    echo "  Spectral   : ${SPECTRAL}"
    echo "  Temporal   : ${TEMPORAL_EXPERTS}"
    echo "  Split      : ${SPLIT}"
    echo "  Device     : ${DEVICE}"
    echo "============================================"

    # Print best checkpoint info
    python3 -c "
import torch
ckpt = torch.load('${CHECKPOINT}', map_location='cpu', weights_only=False)
it = ckpt.get('iteration', 'N/A')
bm = ckpt.get('best_metric', 'N/A')
bv = ckpt.get('best_metric_val', None)
bv_str = f'{bv:.6f}' if bv is not None else 'N/A'
print(f'  Best step     : {it}')
print(f'  Best metric   : {bm} = {bv_str}')
"

    local JSON_OUT="${RESULTS_DIR}/${exp}${JSON_SUFFIX}.json"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/evaluate_frame_moe.py \
        --checkpoint_path "$CHECKPOINT" \
        --hdf5s_dir "$HDF5S_DIR" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --batch_size 4 \
        --num_workers 4 \
        --max_iterations 500 \
        --fmoe_spectral_expert=${SPECTRAL} \
        --fmoe_temporal_experts="${TEMPORAL_EXPERTS}" \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --technique_label_config="${EFFECTIVE_LABEL_CFG}" \
        --output_json "$JSON_OUT" \
        $EVENT_LEVEL_ARG \
        $EXTRA_ARGS \
        $RWC_ARGS

    # UMAP visualization (skip in event-level mode)
    if [ "$EVENT_LEVEL" = "0" ]; then
        run_umap "$exp"
    fi

    echo ""
    echo "--------------------------------------------"
    echo "  ${exp}: done [${MODE_STR}] (${CKPT_TYPE} checkpoint)"
    echo "  Results → ${JSON_OUT}"
    echo "--------------------------------------------"
}

ALL_EXPERIMENTS="baseline full no_onset no_note no_phrase \
    only_onset only_note only_phrase \
    detach_cues freeze_trans no_moe hierarchical \
    hier_multilevel hier_moe \
    multi_crnn crnn_moe shared_crnn_moe \
    binary_baseline binary_full binary_no_moe binary_detach_cues \
    mert_baseline mert_binary"

if [ "$UMAP_ONLY" = "1" ]; then
    if [ "$EXPERIMENT" = "all" ]; then
        echo "Running UMAP only for all experiments..."
        for e in $ALL_EXPERIMENTS; do
            run_umap "$e"
        done
    else
        run_umap "$EXPERIMENT"
    fi
    echo ""
    echo "======== UMAP generation complete ========"
    exit 0
fi

if [ "$EXPERIMENT" = "all" ]; then
    MODE_LABEL="Frame-Level"
    JSON_SUFFIX=""
    if [ "$EVENT_LEVEL" = "1" ]; then
        MODE_LABEL="Event-Level"
        JSON_SUFFIX="_event_level"
    fi
    echo "Running v2 evaluation [${MODE_LABEL}] for all expert composition experiments..."
    echo ""
    for e in $ALL_EXPERIMENTS; do
        run_eval "$e"
    done

    # Merge individual JSONs into a summary
    SUMMARY="${RESULTS_DIR}/summary${JSON_SUFFIX}.json"
    python3 -c "
import json, glob, os
event_level = ${EVENT_LEVEL}
suffix = '_event_level' if event_level else ''
results = {}
for f in sorted(glob.glob('${RESULTS_DIR}/*' + suffix + '.json')):
    name = os.path.splitext(os.path.basename(f))[0]
    if name.startswith('summary'):
        continue
    with open(f) as fp:
        results[name] = json.load(fp)
with open('${SUMMARY}', 'w') as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print(f'Summary saved to ${SUMMARY}')
print()

if event_level:
    # Event-level summary table
    mc_results = {k: v for k, v in results.items() if 'binary' not in k}
    bin_results = {k: v for k, v in results.items() if 'binary' in k}

    if mc_results:
        header = f'{\"Experiment\":<28s} {\"Step\":>6s}  {\"Trans%\":>7s} {\"Tonal F1\":>9s} {\"Artic F1\":>9s} {\"Leg F1\":>8s} {\"All F1\":>8s}'
        print(header)
        print('-' * len(header))
        for name, data in mc_results.items():
            step = data.get('checkpoint', {}).get('iteration', '?')
            vio = data.get('viotech', {})
            tc = vio.get('transcription_correct_pct', 0)
            t = vio.get('tonal', {})
            a = vio.get('articulation', {})
            l = vio.get('legato', {})
            af1 = vio.get('all_technique_macro_f1', 0)
            print(f'{name:<28s} {str(step):>6s}  {tc:7.1f} {t.get(\"macro_f1\",0):9.4f} {a.get(\"macro_f1\",0):9.4f} {l.get(\"macro_f1\",0):8.4f} {af1:8.4f}')
        print()

    if bin_results:
        bh = f'{\"Experiment\":<28s} {\"Step\":>6s}  {\"Trans%\":>7s} {\"Pizz F1\":>8s} {\"Harm F1\":>8s} {\"Spic F1\":>8s} {\"Leg F1\":>8s} {\"All F1\":>8s}'
        print(bh)
        print('-' * len(bh))
        for name, data in bin_results.items():
            step = data.get('checkpoint', {}).get('iteration', '?')
            vio = data.get('viotech', {})
            tc = vio.get('transcription_correct_pct', 0)
            p = vio.get('pizzicato', {})
            h = vio.get('harmonic', {})
            s = vio.get('spiccato', {})
            l = vio.get('legato', {})
            af1 = vio.get('all_technique_macro_f1', 0)
            print(f'{name:<28s} {str(step):>6s}  {tc:7.1f} {p.get(\"macro_f1\",0):8.4f} {h.get(\"macro_f1\",0):8.4f} {s.get(\"macro_f1\",0):8.4f} {l.get(\"macro_f1\",0):8.4f} {af1:8.4f}')

else:
    mc_results = {k: v for k, v in results.items() if 'binary' not in k}
    bin_results = {k: v for k, v in results.items() if 'binary' in k}

    if mc_results:
        header = f'{\"Experiment\":<20s} {\"Step\":>6s}  {\"Tonal Acc\":>9s} {\"Tonal F1\":>9s} {\"Artic Acc\":>9s} {\"Artic F1\":>9s} {\"Legato Acc\":>10s} {\"Legato F1\":>10s}'
        print(header)
        print('-' * len(header))
        for name, data in mc_results.items():
            step = data.get('checkpoint', {}).get('iteration', '?')
            vio = data.get('viotech', {})
            t = vio.get('tonal', {})
            a = vio.get('articulation', {})
            l = vio.get('legato', {})
            print(f'{name:<20s} {str(step):>6s}  {t.get(\"accuracy\",0):9.4f} {t.get(\"macro_f1\",0):9.4f} {a.get(\"accuracy\",0):9.4f} {a.get(\"macro_f1\",0):9.4f} {l.get(\"accuracy\",0):10.4f} {l.get(\"macro_f1\",0):10.4f}')
        print()

    if bin_results:
        bh = f'{\"Experiment\":<20s} {\"Step\":>6s}  {\"Pizz Acc\":>9s} {\"Pizz F1\":>9s} {\"Harm Acc\":>9s} {\"Harm F1\":>9s} {\"Spic Acc\":>9s} {\"Spic F1\":>9s} {\"Leg Acc\":>9s} {\"Leg F1\":>9s}'
        print(bh)
        print('-' * len(bh))
        for name, data in bin_results.items():
            step = data.get('checkpoint', {}).get('iteration', '?')
            vio = data.get('viotech', {})
            p = vio.get('pizzicato', {})
            h = vio.get('harmonic', {})
            s = vio.get('spiccato', {})
            l = vio.get('legato', {})
            print(f'{name:<20s} {str(step):>6s}  {p.get(\"accuracy\",0):9.4f} {p.get(\"macro_f1\",0):9.4f} {h.get(\"accuracy\",0):9.4f} {h.get(\"macro_f1\",0):9.4f} {s.get(\"accuracy\",0):9.4f} {s.get(\"macro_f1\",0):9.4f} {l.get(\"accuracy\",0):9.4f} {l.get(\"macro_f1\",0):9.4f}')
"

    echo ""
    echo "======== All v2 expert composition evaluations [${MODE_LABEL}] complete ========"
    echo "Results dir: ${RESULTS_DIR}/"
else
    run_eval "$EXPERIMENT"
fi
