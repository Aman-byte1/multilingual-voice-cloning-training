#!/bin/bash
# ============================================================
# Full Reproduction: Setup & Eval from Hugging Face Adapters
# ============================================================
set -euo pipefail

export HF_TOKEN="${1:-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}"
export CUDA_VISIBLE_DEVICES="0" # Use first GPU if available, or remove to use CPU

echo "============================================================"
echo "  🛠️  Setting up environment..."
echo "============================================================"
# Clone OmniVoice base code if missing
if [ ! -d "OmniVoice" ]; then
    echo "Cloning OmniVoice base repository..."
    git clone https://github.com/k2-fsa/OmniVoice.git
fi

# Install core dependencies
pip install -q "datasets<2.21.0" "multidict<7.0.0" soundfile librosa jiwer faster-whisper speechbrain peft
# Ensure tokenizer is available
pip install -q git+https://github.com/k2-fsa/higgs-audio-v2-tokenizer.git

# Clone and install OmniVoice base code
if [ ! -d "OmniVoice" ]; then
    echo "Cloning OmniVoice base repository..."
    git clone https://github.com/k2-fsa/OmniVoice.git
fi
echo "Installing OmniVoice package..."
cd OmniVoice && pip install -q -e . && cd ..

# Setup PYTHONPATH (Absolute path)
export PYTHONPATH="$(pwd)/OmniVoice:${PYTHONPATH:-}"



LANGS=("zh" "fr" "ar")
HF_USER="amanuelbyte"

for LANG in "${LANGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  🔍 Evaluating ${LANG} from Hugging Face adapter"
    echo "  Repo: ${HF_USER}/omnivoice-lora-${LANG}"
    echo "============================================================"
    
    python evaluation/evaluate_omnivoice.py \
        --model-name "${HF_USER}/omnivoice-lora-${LANG}" \
        --whisper-lang "${LANG}" \
        --output-dir "./eval_results_hf/${LANG}" \
        --max-samples 100 \
        --resume
done

echo "✅ Evaluation complete. Results in ./eval_results_hf/"
