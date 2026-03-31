#!/bin/bash
# =============================================================================
# Chatterbox FR Fine-Tuning — Server Setup & Run Script
# =============================================================================
# Usage:
#   bash run.sh              # Full pipeline (setup + train + convert + test)
#   bash run.sh --train-only # Skip setup, just train
#   bash run.sh --test-only  # Skip training, just test
# =============================================================================

set -e

# ---- Configuration ----
VENV_DIR="./venv"
OUTPUT_DIR="./chatterbox_fr_finetuned"
REF_AUDIO=""  # Set this to a reference speaker WAV, or leave empty to skip test

# ---- Parse args ----
TRAIN_ONLY=false
TEST_ONLY=false
for arg in "$@"; do
    case $arg in
        --train-only) TRAIN_ONLY=true ;;
        --test-only)  TEST_ONLY=true ;;
    esac
done

echo "============================================================"
echo "  Chatterbox Multilingual TTS — French LoRA Fine-Tuning"
echo "============================================================"

# ---- Step 1: Environment setup ----
if [ "$TRAIN_ONLY" = false ] && [ "$TEST_ONLY" = false ]; then
    echo ""
    echo ">>> Step 1: Setting up Python environment …"

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        echo "  Created virtual environment at $VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install dependencies
    echo "  Installing dependencies …"
    pip install -r requirements.txt

    echo "  Environment ready ✓"
else
    # Activate existing venv
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    fi
fi

# ---- Step 2: Data preparation ----
echo ""
echo ">>> Step 2: Preparing Sparse Dataset (1 ref per target) …"

if [ "$TEST_ONLY" = false ]; then
    if [ -f "./audio_data/metadata.csv" ] && [ -n "$TRAIN_ONLY" ]; then
        echo "  metadata.csv exists, skipping data prep (TRAIN_ONLY)."
    else
        # Create the sparse dataset on HF
        echo "  Creating sparse dataset on HuggingFace ..."
        python training/create_sparse_dataset.py
        
        echo "  Preparing local training directory ..."
        python finetune_chatterbox_fr.py --mode prepare-only --skip-filter
    fi

    # ---- Step 3: Training ----
    echo ""
    echo ">>> Step 3: Training with Optimized LoRA (Rank 16, All Modules) …"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

    # Fast Sparse Training: ~1500 steps total (3 epochs * 500 steps)
    python finetune_chatterbox_fr.py \
        --mode train \
        --output-dir "$OUTPUT_DIR" \
        --use-lora \
        --lora-rank 16 \
        --batch-size 8 \
        --epochs 3 \
        --lr 1e-5 \
        --gradient-accumulation-steps 2 \
        --fp16 \
        --cloning-mode cross_lingual \
        --skip-filter

    # ---- Step 4: Convert merged model ----
    echo ""
    echo ">>> Step 4: Converting merged model to safetensors …"
    if [ -d "$OUTPUT_DIR/merged_model" ]; then
        python fix_merged_model.py --model-dir "$OUTPUT_DIR/merged_model"
    else
        echo "  Skipping merge (no merged_model found)."
    fi
fi

# ---- Step 5: Test inference ----
echo ""
echo ">>> Step 5: Testing fine-tuned model …"

if [ -n "$REF_AUDIO" ] && [ -f "$REF_AUDIO" ]; then
    python test_finetuned.py \
        --model-dir "$OUTPUT_DIR" \
        --ref-audio "$REF_AUDIO" \
        --output-dir ./test_outputs \
        --baseline
    echo "  Test outputs saved to ./test_outputs/"
else
    echo "  Skipping inference test (no REF_AUDIO set)."
    echo "  To test, run:"
    echo "    python test_finetuned.py --model-dir $OUTPUT_DIR --ref-audio YOUR_FILE.wav"
fi

echo ""
echo "============================================================"
echo "  Done! ✓"
echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  Merged model: $OUTPUT_DIR/merged_model/"
echo "  Training plot: $OUTPUT_DIR/training_metrics.png"
echo "============================================================"
