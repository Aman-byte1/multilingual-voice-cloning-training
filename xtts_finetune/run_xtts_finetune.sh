#!/usr/bin/env bash
# ==============================================================================
# XTTS v2 Fine-Tuning Pipeline
# Creates an isolated environment to prevent dependency conflicts with OmniVoice.
# ==============================================================================

set -e

WORKSPACE_DIR="/workspace/multilingual-voice-cloning-training"
VENV_DIR="$WORKSPACE_DIR/.venv-xtts"
XTTS_DIR="$WORKSPACE_DIR/xtts_finetune"

cd "$WORKSPACE_DIR"

echo "================================================================="
echo " 1. Setting up Isolated XTTS Virtual Environment"
echo "================================================================="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✅ Created $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install Coqui TTS and matching dependencies
echo "📦 Installing TTS==0.22.0 and locking dependencies..."
pip install --upgrade pip
pip install TTS==0.22.0 "numpy>=1.24.3,<2.0.0" soundfile librosa pandas datasets
# PyTorch matching the Coqui TTS requirement (usually Torch 2.1 or 2.2 is safe, but we'll let TTS figure it out, just force flash-attn if needed)

echo "================================================================="
echo " 2. Preparing Dataset (Resampling to 22.05kHz)"
echo "================================================================="
python "$XTTS_DIR/prep_xtts_dataset.py" \
    --output-dir "$WORKSPACE_DIR/data/xtts_dataset"

echo "================================================================="
echo " 3. Starting XTTS v2 Fine-Tuning"
echo "================================================================="
python "$XTTS_DIR/train_xtts.py" \
    --dataset-path "$WORKSPACE_DIR/data/xtts_dataset" \
    --output-path "$WORKSPACE_DIR/exp/xtts_finetuned" \
    --languages "fr,ar,zh" \
    --epochs 10

echo "🎉 XTTS Fine-Tuning completed successfully!"
