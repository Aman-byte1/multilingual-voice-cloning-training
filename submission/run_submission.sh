#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# IWSLT 2026 — Full Submission Pipeline
# ═══════════════════════════════════════════════════════════════
# Run this on the GPU server (A40/4090 with 24GB+ VRAM).
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export TEAM_NAME="your_team"
#   bash run_submission.sh [--languages fr] [--models qwen cosyvoice]
#
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── DEFAULTS ──────────────────────────────────────────────────
TEAM_NAME="${TEAM_NAME:-tcd}"
LANGUAGES="${LANGUAGES:-fr ar zh}"
MODELS="${MODELS:-qwen cosyvoice}"
BLIND_TEST_DIR="./blind_test"
REF_AUDIO_DIR="./blind_test/ref_audio"
OUTPUT_BASE="./outputs"
FINAL_DIR="./final"
EVAL_DIR="./eval_comparison"

echo "═══════════════════════════════════════════════════════════════"
echo "  IWSLT 2026 — Cross-Lingual Voice Cloning Pipeline"
echo "  Team: ${TEAM_NAME}"
echo "  Languages: ${LANGUAGES}"
echo "  Models: ${MODELS}"
echo "═══════════════════════════════════════════════════════════════"

# ─── STEP 0: Environment Setup ────────────────────────────────
echo ""
echo "📦 Step 0: Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q qwen-tts transformers accelerate flash-attn
pip install -q TTS  # XTTS-v2
pip install -q faster-whisper soundfile librosa numpy tqdm
pip install -q jiwer sacrebleu speechbrain pesq pystoi pymcd
pip install -q datasets huggingface_hub scipy scikit-learn

# CosyVoice3 (from source)
if [ ! -d "CosyVoice" ]; then
    echo "  Cloning CosyVoice3..."
    git clone https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice && pip install -e . && cd ..
fi

echo "  ✅ Dependencies installed"

# ─── STEP 1: Verify Blind Data ────────────────────────────────
echo ""
echo "📂 Step 1: Verifying blind test data..."

if [ ! -d "${BLIND_TEST_DIR}" ]; then
    echo "  ❌ Blind test data not found at ${BLIND_TEST_DIR}"
    echo "  Please download from IWSLT and place files as follows:"
    echo "    ${BLIND_TEST_DIR}/ref_audio/*.wav"
    echo "    ${BLIND_TEST_DIR}/arabic.txt"
    echo "    ${BLIND_TEST_DIR}/chinese.txt"
    echo "    ${BLIND_TEST_DIR}/french.txt"
    exit 1
fi

for lang in ${LANGUAGES}; do
    case $lang in
        ar) textfile="arabic.txt" ;;
        zh) textfile="chinese.txt" ;;
        fr) textfile="french.txt" ;;
    esac
    if [ ! -f "${BLIND_TEST_DIR}/${textfile}" ]; then
        echo "  ⚠ Missing: ${BLIND_TEST_DIR}/${textfile}"
    else
        lines=$(wc -l < "${BLIND_TEST_DIR}/${textfile}")
        echo "  ✓ ${textfile}: ${lines} lines"
    fi
done

ref_count=$(ls -1 "${REF_AUDIO_DIR}"/*.wav 2>/dev/null | wc -l)
echo "  ✓ Reference audio: ${ref_count} files"

# ─── STEP 2: Generate with each model ─────────────────────────
for model in ${MODELS}; do
    echo ""
    echo "🎙  Step 2: Generating with ${model^^}..."
    
    model_output="${OUTPUT_BASE}/${model}"
    mkdir -p "${model_output}"

    case $model in
        qwen)
            python submission/generate_qwen.py \
                --languages ${LANGUAGES} \
                --ref-dir "${REF_AUDIO_DIR}" \
                --text-dir "${BLIND_TEST_DIR}" \
                --output-dir "${model_output}"
            ;;
        cosyvoice)
            python submission/generate_cosyvoice.py \
                --languages ${LANGUAGES} \
                --ref-dir "${REF_AUDIO_DIR}" \
                --text-dir "${BLIND_TEST_DIR}" \
                --output-dir "${model_output}"
            ;;
        xtts)
            echo "  XTTS generation needs custom runner (see evaluate_xtts.py)"
            ;;
    esac
    
    echo "  ✅ ${model^^} generation complete"
done

# ─── STEP 3: Evaluate on dev data (optional) ──────────────────
echo ""
echo "📊 Step 3: Running dev evaluation..."
python submission/evaluate_models.py \
    --models ${MODELS} \
    --languages ${LANGUAGES} \
    --max-samples 50 \
    --output-dir "${EVAL_DIR}" \
    2>&1 | tee eval_log.txt

echo "  ✅ Evaluation complete — check ${EVAL_DIR}/comparison.json"

# ─── STEP 4: Pick best model & package ─────────────────────────
echo ""
echo "📦 Step 4: Packaging submissions..."

# Default: use qwen (user can change after comparing)
BEST_MODEL="${BEST_MODEL:-qwen}"

python submission/package_submission.py \
    --team "${TEAM_NAME}" \
    --model-dir "${OUTPUT_BASE}/${BEST_MODEL}" \
    --languages ${LANGUAGES} \
    --source-dir "${BLIND_TEST_DIR}" \
    --ref-dir "${REF_AUDIO_DIR}" \
    --output-dir "${FINAL_DIR}"

# ─── STEP 5: Final validation ─────────────────────────────────
echo ""
echo "🔍 Step 5: Validating submissions..."
for lang in ${LANGUAGES}; do
    zip_file="${FINAL_DIR}/${TEAM_NAME}_${lang}.zip"
    if [ -f "${zip_file}" ]; then
        case $lang in
            ar) textfile="arabic.txt" ;;
            zh) textfile="chinese.txt" ;;
            fr) textfile="french.txt" ;;
        esac
        echo "  Validating ${zip_file}..."
        # Run official validator if available
        if [ -f "verify_submission_naming.py" ]; then
            python verify_submission_naming.py "${zip_file}" \
                --language "${lang}" \
                --source-file "${BLIND_TEST_DIR}/${textfile}" \
                --reference-dir "${REF_AUDIO_DIR}" || true
        else
            # Quick manual check
            count=$(unzip -l "${zip_file}" | grep "\.wav" | wc -l)
            echo "  ✓ ${zip_file}: ${count} WAV files"
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ PIPELINE COMPLETE"
echo "  Submissions ready in: ${FINAL_DIR}/"
echo "  Submit before: April 15, 2026 11:59PM UTC-12"
echo "═══════════════════════════════════════════════════════════════"
