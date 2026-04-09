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

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [device] [checkpoint]}"
DEVICE="${2:-0}"
MANUAL_CKPT="${3:-}"

RWC_H5="/mnt/hdd/rwc_processed_data.h5"
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech_paganini_kayser_with_openstring"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced_openstring.json"
SPLIT="test"
RESULTS_DIR="${WORKSPACE}/results/ablation_expert_composition_v2"
TEMPORAL_EXPERTS="11:1,51:1,51:3"
mkdir -p "$RESULTS_DIR"

resolve_experiment() {
    local exp="$1"
    DIRECT_HEAD=0
    HIERARCHICAL=0
    case "$exp" in
        baseline)    TAG="ablation_fmoe_openstring_baseline_e012";         EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        full)        TAG="ablation_fmoe_openstring_full_e0123_spectral";   EXPERT_MASK="0,1,2,3"; SPECTRAL=1 ;;
        no_onset)    TAG="ablation_fmoe_openstring_no_onset_e12";          EXPERT_MASK="1,2";   SPECTRAL=0 ;;
        no_note)     TAG="ablation_fmoe_openstring_no_note_e02";           EXPERT_MASK="0,2";   SPECTRAL=0 ;;
        no_phrase)   TAG="ablation_fmoe_openstring_no_phrase_e01";         EXPERT_MASK="0,1";   SPECTRAL=0 ;;
        only_onset)  TAG="ablation_fmoe_openstring_only_onset_e0";         EXPERT_MASK="0";     SPECTRAL=0 ;;
        only_note)   TAG="ablation_fmoe_openstring_only_note_e1";          EXPERT_MASK="1";     SPECTRAL=0 ;;
        only_phrase) TAG="ablation_fmoe_openstring_only_phrase_e2";        EXPERT_MASK="2";     SPECTRAL=0 ;;
        tech0.1)     TAG="ablation_fmoe_tech0.1_baseline_e012";           EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        detach_cues) TAG="ablation_fmoe_tech0.1_detach_cues_e012";        EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        freeze_trans) TAG="ablation_fmoe_tech0.1_freeze_trans_e012";      EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        no_moe)      TAG="ablation_fmoe_tech0.1_no_moe_direct";           EXPERT_MASK="0,1,2"; SPECTRAL=0; DIRECT_HEAD=1 ;;
        hierarchical) TAG="ablation_fmoe_tech0.1_hierarchical";           EXPERT_MASK="";      SPECTRAL=0; HIERARCHICAL=1 ;;
        *)
            echo "Unknown experiment: $exp"
            return 1 ;;
    esac
}

find_checkpoint() {
    local tag="$1"
    local ckpt_dir="${WORKSPACE}/checkpoints/main_contrast/${tag}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4"

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

    echo ""
    return 1
}

run_eval() {
    local exp="$1"
    resolve_experiment "$exp" || return 1

    local CHECKPOINT
    if [ -n "$MANUAL_CKPT" ]; then
        CHECKPOINT="$MANUAL_CKPT"
    else
        CHECKPOINT=$(find_checkpoint "$TAG")
        if [ -z "$CHECKPOINT" ]; then
            echo "[SKIP] No checkpoint found for ${exp} (${TAG})"
            return 0
        fi
    fi

    local CKPT_TYPE="latest"
    if [[ "$CHECKPOINT" == *"best_model.pth"* ]]; then
        CKPT_TYPE="best"
    fi

    echo ""
    echo "============================================"
    echo "  Expert Ablation v2 Eval: ${exp}"
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

    local EXTRA_ARGS=""
    [ -n "$EXPERT_MASK" ] && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"
    [ "$DIRECT_HEAD" = "1" ] && EXTRA_ARGS+=" --fmoe_direct_head=1"
    [ "$HIERARCHICAL" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical=1"

    local RWC_ARGS=""
    [ -f "$RWC_H5" ] && RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split all"

    local JSON_OUT="${RESULTS_DIR}/${exp}.json"

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
        --technique_label_config="${LABEL_CFG}" \
        --output_json "$JSON_OUT" \
        $EXTRA_ARGS \
        $RWC_ARGS

    echo ""
    echo "--------------------------------------------"
    echo "  ${exp}: done (${CKPT_TYPE} checkpoint)"
    echo "  Results → ${JSON_OUT}"
    echo "--------------------------------------------"
}

if [ "$EXPERIMENT" = "all" ]; then
    echo "Running v2 evaluation for all expert composition experiments..."
    echo ""
    for e in baseline full no_onset no_note no_phrase \
             only_onset only_note only_phrase \
             detach_cues freeze_trans no_moe hierarchical; do
        run_eval "$e"
    done

    # Merge individual JSONs into a summary
    SUMMARY="${RESULTS_DIR}/summary.json"
    python3 -c "
import json, glob, os
results = {}
for f in sorted(glob.glob('${RESULTS_DIR}/*.json')):
    name = os.path.splitext(os.path.basename(f))[0]
    if name == 'summary':
        continue
    with open(f) as fp:
        results[name] = json.load(fp)
with open('${SUMMARY}', 'w') as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print(f'Summary saved to ${SUMMARY}')
print()
# Print comparison table
header = f'{\"Experiment\":<15s} {\"Step\":>6s}  {\"Tonal Acc\":>9s} {\"Tonal F1\":>9s} {\"Artic Acc\":>9s} {\"Artic F1\":>9s} {\"Legato Acc\":>10s} {\"Legato F1\":>10s}'
print(header)
print('-' * len(header))
for name, data in results.items():
    step = data.get('checkpoint', {}).get('iteration', '?')
    vio = data.get('viotech', {})
    t = vio.get('tonal', {})
    a = vio.get('articulation', {})
    l = vio.get('legato', {})
    print(f'{name:<15s} {str(step):>6s}  {t.get(\"accuracy\",0):9.4f} {t.get(\"macro_f1\",0):9.4f} {a.get(\"accuracy\",0):9.4f} {a.get(\"macro_f1\",0):9.4f} {l.get(\"accuracy\",0):10.4f} {l.get(\"macro_f1\",0):10.4f}')
"

    echo ""
    echo "======== All v2 expert composition evaluations complete ========"
    echo "Results dir: ${RESULTS_DIR}/"
else
    run_eval "$EXPERIMENT"
fi
