#!/bin/bash
# ============================================================
# Blind Test Evaluation Pipeline for Fine-Tuned OmniVoice
# ============================================================
set -euo pipefail

echo "============================================================"
echo "📦 1. Installing Evaluation Dependencies..."
echo "============================================================"
pip install -q jiwer faster-whisper speechbrain soundfile tqdm huggingface_hub peft accelerate

if ! python -c "import omnivoice" &>/dev/null; then
    echo "📦 OmniVoice is not installed in this environment. Installing it..."
    if [ -d "OmniVoice" ]; then
        pip install -e OmniVoice
    else
        echo "⚠️ OmniVoice folder not found in current directory. Trying fallback install..."
        pip install omnivoice
    fi
fi

echo "============================================================"
echo "🔍 2. Verifying Generated Audios in temp_submission/..."
echo "============================================================"
for lang in fr ar zh; do
    if [ -d "temp_submission/$lang" ]; then
        count=$(find "temp_submission/$lang" -name "*.wav" | wc -l)
        echo "  ✓ temp_submission/$lang: found $count generated files"
    else
        echo "  ✗ temp_submission/$lang not found!"
    fi
done

echo "============================================================"
echo "📈 3. Running A/B Evaluation (Base vs. Fine-Tuned LoRA)..."
echo "============================================================"
for lang in fr ar zh; do
    if [ -d "temp_submission/$lang" ]; then
        echo "🚀 Evaluating $lang..."
        python evaluate_lang.py --lang "$lang" "$@"
    else
        echo "⏭ Skipping $lang (no generated audio)"
    fi
done

echo "============================================================"
echo "✅ Evaluation Pipeline Completed!"
echo "============================================================"
