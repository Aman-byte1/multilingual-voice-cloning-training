#!/bin/bash
# ============================================================
# Sequential Full Evaluation for IWSLT 2026
# Evaluates Baseline vs Step 200 vs Step 300 vs Step 400
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="0"  # Use first GPU for maximum speed
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo "  SEQUENTIAL EVALUATION: Baseline vs 200 vs 300 vs 400"
echo "  GPU: 0 | Split: Full Eval Set"
echo "============================================================"

# Steps to evaluate
STEPS=("100" "200" "300" "400")

LANGS=("zh" "fr" "ar")
DECLARE_WHISPER=("zh" "fr" "ar")

BASE_OUT_ROOT="./eval_results/full_test_final"

# 1. Evaluate Baseline (Skipping as already done)
# for i in "${!LANGS[@]}"; do
#     LANG="${LANGS[$i]}"
#     OUT_DIR="${BASE_OUT_ROOT}/baseline/${LANG}"
#     
#     echo ""
#     echo "------------------------------------------------------------"
#     echo "  🔍 Evaluating ${LANG} BASELINE"
#     echo "------------------------------------------------------------"
#     
#     python evaluation/evaluate_omnivoice.py \
#         --model-name "k2-fsa/OmniVoice" \
#         --whisper-lang "${LANG}" \
#         --output-dir "${OUT_DIR}" \
#         --resume
# done


# 2. Evaluate Checkpoints from Hugging Face
for LANG in "${LANGS[@]}"; do
    for STEP in "${STEPS[@]}"; do
        # Step 100 is in the original repo (no suffix)
        # Steps 200, 300, 400 have the -step suffix
        if [ "${STEP}" == "100" ]; then
            HF_REPO="amanuelbyte/omnivoice-lora-${LANG}"
        else
            HF_REPO="amanuelbyte/omnivoice-lora-${LANG}-${STEP}"
        fi
        
        OUT_DIR="${BASE_OUT_ROOT}/step-${STEP}/${LANG}"
        
        echo ""
        echo "------------------------------------------------------------"
        echo "  🔍 Evaluating ${LANG} — STEP ${STEP} (from Hugging Face)"
        echo "  Repo: ${HF_REPO}"
        echo "------------------------------------------------------------"

        python evaluation/evaluate_omnivoice.py \
            --model-name "${HF_REPO}" \
            --whisper-lang "${LANG}" \
            --output-dir "${OUT_DIR}" \
            --resume
    done
done



# 3. Generate Comparison Table
echo ""
echo "============================================================"
echo "  📊 FINAL REPORT GENERATION"
echo "============================================================"

python3 - << 'PYEOF'
import json, os

langs = ["zh", "fr", "ar"]
steps = ["200", "300", "400"]
labels = {"zh": "Chinese", "fr": "French", "ar": "Arabic"}
root = "./eval_results/full_test_final"

print(f"{'Language':<10} {'Checkpoint':<12} {'WER':>8} {'CER':>8} {'SimScore':>10}")
print("-" * 60)

for lang in langs:
    # Baseline
    summary_path = os.path.join(root, "baseline", lang, "eval_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        wer = data.get("WER", {}).get("mean", 0.0)
        cer = data.get("CER", {}).get("mean", 0.0)
        sim = data.get("Similarity", {}).get("mean", 0.0)
        print(f"{labels[lang]:<10} {'Baseline':<12} {wer:>8.4f} {cer:>8.4f} {sim:>10.4f}")
    
    # Checkpoints
    for step in steps:
        summary_path = os.path.join(root, f"step-{step}", lang, "eval_summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)
        wer = data.get("WER", {}).get("mean", 0.0)
        cer = data.get("CER", {}).get("mean", 0.0)
        sim = data.get("Similarity", {}).get("mean", 0.0)
        print(f"{labels[lang]:<10} {f'Step {step}':<12} {wer:>8.4f} {cer:>8.4f} {sim:>10.4f}")
    print()

print("=" * 60)
PYEOF
