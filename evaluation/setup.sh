#!/bin/bash
# ============================================================
# Evaluation Environment Setup
# Run once on a fresh server before running eval.py
# Usage: bash evaluation/setup.sh
# ============================================================
set -e

echo "============================================"
echo "  Setting up evaluation environment"
echo "============================================"

# 1. Install system dependencies (tmux)
if command -v apt-get &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# 2. Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "Detected CUDA: $CUDA_VER"
    
    # Map to PyTorch index
    case "$CUDA_VER" in
        11.8*) CUDA_TAG="cu118" ;;
        12.1*) CUDA_TAG="cu121" ;;
        12.4*) CUDA_TAG="cu124" ;;
        12.6*) CUDA_TAG="cu124" ;;
        12.8*) CUDA_TAG="cu124" ;; # Use cu124 for CUDA 12.8
        *)     CUDA_TAG="cu121" ; echo "Unknown CUDA $CUDA_VER, defaulting to cu121" ;;
    esac
else
    CUDA_TAG="cpu"
    echo "No GPU detected, using CPU"
fi

echo "Using PyTorch index: $CUDA_TAG"

# 3. Install PyTorch (matched set — torch + torchaudio + torchvision)
echo ""
echo "Installing PyTorch..."
# Using 2.5.1 as it is widely available in all indices
pip install --force-reinstall torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
    --extra-index-url https://download.pytorch.org/whl/$CUDA_TAG

# 4. Install pinned dependencies for chatterbox-tts
echo ""
echo "Installing dependencies..."
pip install "numpy<2.0"
pip install transformers==5.2.0
pip install "datasets<4.0" soundfile
pip install chatterbox-tts==0.1.7
pip install faster-whisper speechbrain jiwer tqdm

echo ""
echo "============================================"
echo "  Setup complete! Ready to run eval.py"
echo "============================================"
