#!/bin/bash
# ============================================================
# Evaluate checkpoint-100 for all 3 languages
# Runs sequentially on GPU 0 (or whichever GPU you free up)
# Can run while training continues on other GPUs
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

EVAL_GPU="${1:-0}"  # Pass GPU id as arg, default 0
echo "============================================================"
echo "  Evaluating checkpoint-100 on GPU ${EVAL_GPU}"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# Evaluate each language checkpoint
# ──────────────────────────────────────────────────────────────
declare -A LANG_WHISPER
LANG_WHISPER[zh]="zh"
LANG_WHISPER[fr]="fr"
LANG_WHISPER[ar]="ar"

for LANG in zh fr ar; do
    CKPT="./exp/omnivoice_finetuned_${LANG}/checkpoint-100"
    OUT_DIR="./eval_results/checkpoint-100/${LANG}"

    if [ ! -d "${CKPT}" ]; then
        echo "  ⚠ Skipping ${LANG}: ${CKPT} not found"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  🔍 Evaluating ${LANG} (checkpoint-100)"
    echo "  Model: ${CKPT}"
    echo "  Output: ${OUT_DIR}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${EVAL_GPU} python evaluation/evaluate_omnivoice.py \
        --model-name "${CKPT}" \
        --whisper-lang "${LANG_WHISPER[${LANG}]}" \
        --output-dir "${OUT_DIR}" \
        --max-samples 30 \
        --resume

    echo "  ✅ ${LANG} eval done → ${OUT_DIR}/eval_summary.json"
done

# ──────────────────────────────────────────────────────────────
# Also evaluate the base model for comparison
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🔍 Evaluating BASE OmniVoice (no LoRA) for comparison"
echo "============================================================"

for LANG in zh fr ar; do
    OUT_DIR="./eval_results/baseline/${LANG}"

    echo "  --> ${LANG} baseline..."
    CUDA_VISIBLE_DEVICES=${EVAL_GPU} python evaluation/evaluate_omnivoice.py \
        --model-name "k2-fsa/OmniVoice" \
        --whisper-lang "${LANG_WHISPER[${LANG}]}" \
        --output-dir "${OUT_DIR}" \
        --max-samples 30 \
        --resume

    echo "  ✅ ${LANG} baseline done"
done

# ──────────────────────────────────────────────────────────────
# Summary comparison
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  📊  COMPARISON: Baseline vs Checkpoint-100"
echo "============================================================"

python3 - << 'PYEOF'
import json, os, sys

langs = ["zh", "fr", "ar"]
labels = {"zh": "Chinese", "fr": "French", "ar": "Arabic"}

print(f"{'Lang':<10} {'Model':<15} {'WER':>8} {'CER':>8} {'SimScore':>10}")
print("-" * 55)

for lang in langs:
    for tag, path_tpl in [("Baseline", "baseline"), ("Ckpt-100", "checkpoint-100")]:
        summary_path = f"./eval_results/{path_tpl}/{lang}/eval_summary.json"
        if not os.path.exists(summary_path):
            print(f"{labels[lang]:<10} {tag:<15} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue
        with open(summary_path) as f:
            data = json.load(f)
        wer = data.get("WER", {}).get("mean", float("nan"))
        cer = data.get("CER", {}).get("mean", float("nan"))
        sim = data.get("Similarity", {}).get("mean", float("nan"))
        print(f"{labels[lang]:<10} {tag:<15} {wer:>8.4f} {cer:>8.4f} {sim:>10.4f}")
    print()

print("=" * 55)
PYEOF
