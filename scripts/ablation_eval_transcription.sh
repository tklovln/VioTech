#!/bin/bash
#
# Expert Composition Ablation v2 — Transcription Evaluation
#
# Evaluates note transcription (onset/offset/pitch via mir_eval) for
# v2 checkpoints (ablation_fmoe_v2_*) that were trained with unified cues.
# Reuses the same checkpoint resolution as ablation_eval_expert_composition_v2.sh.
#
# Usage:
#   bash scripts/ablation_eval_transcription.sh <experiment_id> <dataset> [device]
#   bash scripts/ablation_eval_transcription.sh all <dataset> [device]
#   bash scripts/ablation_eval_transcription.sh <id> <dataset> [device] <checkpoint>
#
# Datasets: bach10, urmp, mosapt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id|all> <dataset> [device] [checkpoint]}"
DATASET="${2:?Usage: $0 <experiment_id|all> <dataset> [device] [checkpoint]}"
DEVICE="${3:-0}"
MANUAL_CKPT="${4:-}"

RESULTS_DIR="${WORKSPACE}/results/ablation_transcription"
mkdir -p "$RESULTS_DIR"

# ── Experiment definitions (same as expert_composition_v2) ──────────────
resolve_experiment() {
    local exp="$1"
    case "$exp" in
        baseline)    TAG="ablation_fmoe_openstring_baseline_e012"       ;;
        full)        TAG="ablation_fmoe_openstring_full_e0123_spectral" ;;
        no_onset)    TAG="ablation_fmoe_openstring_no_onset_e12"        ;;
        no_note)     TAG="ablation_fmoe_openstring_no_note_e02"         ;;
        no_phrase)   TAG="ablation_fmoe_openstring_no_phrase_e01"       ;;
        only_onset)  TAG="ablation_fmoe_openstring_only_onset_e0"       ;;
        only_note)   TAG="ablation_fmoe_openstring_only_note_e1"        ;;
        only_phrase) TAG="ablation_fmoe_openstring_only_phrase_e2"      ;;
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

# ── Single-experiment runner ────────────────────────────────────────────
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
    echo "  Transcription Eval: ${exp}"
    echo "============================================"
    echo "  Tag        : ${TAG}"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "  Dataset    : ${DATASET}"
    echo "  Device     : ${DEVICE}"
    echo "============================================"

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

    cd "${WORKSPACE}"

    # evaluate.py creates: {output_dir}/{dataset}/{model_tag}/eval.json
    python3 evaluate.py \
        --dataset "$DATASET" \
        --checkpoint_path "$CHECKPOINT" \
        --model_type Regress_onset_offset_frame_velocity_CRNN \
        --device "$DEVICE" \
        --post_processor_type regression \
        --model_tag "${exp}" \
        --output_dir "$RESULTS_DIR"

    local JSON_OUT="${RESULTS_DIR}/${DATASET}/${exp}/eval.json"

    echo ""
    echo "--------------------------------------------"
    echo "  ${exp}: done (${CKPT_TYPE} checkpoint)"
    echo "  Results → ${JSON_OUT}"
    echo "--------------------------------------------"
}

# ── Main ────────────────────────────────────────────────────────────────
if [ "$EXPERIMENT" = "all" ]; then
    echo "Running transcription evaluation (${DATASET}) for all expert composition experiments..."
    echo ""
    for e in baseline full no_onset no_note no_phrase \
             only_onset only_note only_phrase; do
        run_eval "$e"
    done

    # Merge per-experiment results into summary
    EVAL_DIR="${RESULTS_DIR}/${DATASET}"
    SUMMARY="${EVAL_DIR}/summary.json"
    python3 -c "
import json, os

eval_dir = '${EVAL_DIR}'
results = {}
for name in ['baseline', 'full', 'no_onset', 'no_note', 'no_phrase',
             'only_onset', 'only_note', 'only_phrase']:
    json_path = os.path.join(eval_dir, name, 'eval.json')
    if not os.path.isfile(json_path):
        continue
    with open(json_path) as fp:
        results[name] = json.load(fp)

with open('${SUMMARY}', 'w') as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)

print(f'Summary saved to ${SUMMARY}')
print()

# Print comparison table
header = (f'{\"Experiment\":<15s} {\"P\":>7s} {\"R\":>7s} {\"F1\":>7s} '
          f'{\"F1(no off)\":>10s} {\"#Ref\":>7s} {\"#Est\":>7s} {\"#Songs\":>6s}')
print(header)
print('-' * len(header))
for name, data in results.items():
    p  = data.get('overall_precision', 0)
    r  = data.get('overall_recall', 0)
    f1 = data.get('overall_f1', 0)
    f1n = data.get('overall_f1_wo_offset', 0)
    nr = data.get('total_ref_notes', 0)
    ne = data.get('total_est_notes', 0)
    ns = data.get('num_songs', 0)
    print(f'{name:<15s} {p:7.4f} {r:7.4f} {f1:7.4f} {f1n:10.4f} {nr:7d} {ne:7d} {ns:6d}')
"

    echo ""
    echo "======== All transcription evaluations complete (${DATASET}) ========"
    echo "Results dir: ${EVAL_DIR}/"
else
    run_eval "$EXPERIMENT"
fi
