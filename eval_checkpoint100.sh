#!/bin/bash
# ============================================================
# Evaluate checkpoint-100 on CPU (100 samples per language)
# Won't touch GPU training at all
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=""  # Force CPU — no GPU usage
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

SAMPLES=100

echo "============================================================"
echo "  Evaluating checkpoint-100 on CPU (${SAMPLES} samples)"
echo "  Training continues undisturbed on GPU"
echo "============================================================"

declare -A LANG_WHISPER
LANG_WHISPER[zh]="zh"
LANG_WHISPER[fr]="fr"
LANG_WHISPER[ar]="ar"

# ── Evaluate finetuned checkpoints ──
for LANG in zh fr ar; do
    CKPT="./exp/omnivoice_finetuned_${LANG}/checkpoint-100"
    OUT_DIR="./eval_results/checkpoint-100/${LANG}"

    if [ ! -d "${CKPT}" ]; then
        echo "  ⚠ Skipping ${LANG}: ${CKPT} not found"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  🔍 Evaluating ${LANG} (checkpoint-100) on CPU"
    echo "============================================================"

    python evaluation/evaluate_omnivoice.py \
        --model-name "${CKPT}" \
        --whisper-lang "${LANG_WHISPER[${LANG}]}" \
        --output-dir "${OUT_DIR}" \
        --max-samples ${SAMPLES} \
        --resume

    echo "  ✅ ${LANG} eval done"
done

# ── Evaluate baseline for comparison ──
for LANG in zh fr ar; do
    OUT_DIR="./eval_results/baseline/${LANG}"

    echo ""
    echo "  --> ${LANG} baseline (CPU)..."
    python evaluation/evaluate_omnivoice.py \
        --model-name "k2-fsa/OmniVoice" \
        --whisper-lang "${LANG_WHISPER[${LANG}]}" \
        --output-dir "${OUT_DIR}" \
        --max-samples ${SAMPLES} \
        --resume

    echo "  ✅ ${LANG} baseline done"
done

# ── Comparison table ──
echo ""
echo "============================================================"
echo "  📊  COMPARISON: Baseline vs Checkpoint-100"
echo "============================================================"

python3 - << 'PYEOF'
import json, os

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
