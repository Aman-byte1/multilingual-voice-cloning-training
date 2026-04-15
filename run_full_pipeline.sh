#!/bin/bash
# ============================================================
# FULL END-TO-END PIPELINE
# Clone → Install → Synthesize → Fine-tune → Upload
# ============================================================
# Usage (one-liner on A100):
#   curl -sL https://raw.githubusercontent.com/Aman-byte1/multilingual-voice-cloning-training/main/run_full_pipeline.sh | bash
#   OR:
#   git clone https://github.com/Aman-byte1/multilingual-voice-cloning-training.git && cd multilingual-voice-cloning-training && bash run_full_pipeline.sh
# ============================================================

set -euo pipefail

HF_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN env var first — export HF_TOKEN=hf_xxx}"
HF_REPO_DATASET="amanuelbyte/omnivoice-best-of-n-training"
HF_REPO_MODEL="amanuelbyte/omnivoice-finetuned-iwslt2026"
WORKDIR="/workspace/multilingual-voice-cloning-training"

echo "============================================================"
echo "  FULL PIPELINE: Clone → Install → Synth → Train → Upload"
echo "  Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# STEP 0: Clone repo
# ---------------------------------------------------------------
echo ""
echo "📦 STEP 0: Cloning repository..."

if [ ! -d "${WORKDIR}" ]; then
    git clone https://github.com/Aman-byte1/multilingual-voice-cloning-training.git "${WORKDIR}"
fi
cd "${WORKDIR}"
git pull

# ---------------------------------------------------------------
# STEP 1: Install all dependencies
# ---------------------------------------------------------------
echo ""
echo "📦 STEP 1: Installing dependencies..."

pip install --upgrade pip

# System deps (sox for Qwen3-TTS, ffmpeg for audio processing)
apt-get update -qq && apt-get install -y -qq sox ffmpeg > /dev/null 2>&1 || echo "   ⚠ Could not install sox/ffmpeg (non-root?)"

# Install OmniVoice FIRST — it pins the correct transformers version
pip install omnivoice datasets==2.20.0

# Record the transformers version omnivoice needs
TF_VER=$(python3 -c "import transformers; print(transformers.__version__)")
echo "   OmniVoice installed with transformers==${TF_VER}"

# Pre-install numpy and build deps for pkuseg (chatterbox dependency)
pip install "numpy<1.26" cython setuptools wheel

# Install other TTS models WITHOUT upgrading transformers
pip install resemble-perth pkuseg s3tokenizer omegaconf diffusers==0.29.0 gradio==5.44.1 --no-build-isolation
pip install voxcpm --no-deps 2>/dev/null || pip install voxcpm
pip install chatterbox-tts --no-deps 2>/dev/null || pip install chatterbox-tts

# Qwen3-TTS — use qwen-tts library
pip install qwen-tts 2>/dev/null || echo "   qwen-tts not available, will try transformers fallback"
pip install accelerate

# Re-pin transformers to the version omnivoice needs
pip install "transformers==${TF_VER}"

# Evaluation + scoring
pip install faster-whisper jiwer speechbrain torchaudio soundfile

# Dataset + upload
pip install datasets huggingface_hub

# OmniVoice training deps
pip install "omnivoice[train]" 2>/dev/null || pip install webdataset

# ---------------------------------------------------------------
# STEP 1.5: Fix Python compatibility issues in third-party libraries
# ---------------------------------------------------------------
# Fix qwen_tts compatibility with transformers 5.x
# (check_model_inputs changed from decorator factory to plain decorator)
QWEN_TTS_FILE=$(python3 -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py'))" 2>/dev/null || echo "")
if [ -n "${QWEN_TTS_FILE}" ] && [ -f "${QWEN_TTS_FILE}" ]; then
    sed -i 's/@check_model_inputs()/@check_model_inputs/g' "${QWEN_TTS_FILE}"
    echo "   🔧 Patched qwen_tts for transformers compat"
fi

# ---------------------------------------------------------------
# Verify critical imports
# ---------------------------------------------------------------
python3 -c "
import omnivoice; print(f'   ✅ omnivoice {omnivoice.__version__}')
import transformers; print(f'   ✅ transformers {transformers.__version__}')
try:
    import voxcpm; print('   ✅ voxcpm')
except: print('   ⚠ voxcpm not available')
try:
    from chatterbox.tts import ChatterboxTTS; print('   ✅ chatterbox')
except Exception as e: print(f'   ⚠ chatterbox failed: {e}')
try:
    import qwen_tts; print('   ✅ qwen_tts')
except Exception as e: print(f'   ⚠ qwen_tts failed: {e}')
"

echo "   ✅ All dependencies installed"

# ---------------------------------------------------------------
# STEP 2: Login to HuggingFace
# ---------------------------------------------------------------
echo ""
echo "🔑 STEP 2: HuggingFace login..."
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

# ---------------------------------------------------------------
# STEP 3: Best-of-N synthesis (all 3 languages)
# ---------------------------------------------------------------
echo ""
echo "🎙  STEP 3: Best-of-N synthesis..."

for lang in fr ar zh; do
    echo ""
    echo "  ── Synthesizing ${lang} ──"
    python evaluation/synthesize_dev_best_of_n.py \
        --lang ${lang} \
        --dataset ymoslem/acl-6060 \
        --split dev \
        --output-dir ./dev_synth \
        --cache-dir ./data_cache \
        --hf-token "${HF_TOKEN}"

    # Upload this language's results immediately
    echo "  📤 Uploading ${lang} results to HuggingFace..."
    python evaluation/upload_best_of_n_dataset.py \
        --dev-synth-dir ./dev_synth \
        --repo-id "${HF_REPO_DATASET}" \
        --hf-token "${HF_TOKEN}" 2>/dev/null || echo "  ⚠ Upload for ${lang} had warnings (non-fatal)"
    echo "  ✅ ${lang} done and uploaded"
done

echo "   ✅ All 3 languages synthesized and uploaded"

# ---------------------------------------------------------------
# STEP 4: Final dataset upload (merged manifest)
# ---------------------------------------------------------------
echo ""
echo "📤 STEP 4: Final dataset upload (merged train_all.jsonl)..."

python evaluation/upload_best_of_n_dataset.py \
    --dev-synth-dir ./dev_synth \
    --repo-id "${HF_REPO_DATASET}" \
    --hf-token "${HF_TOKEN}"

echo "   ✅ Dataset uploaded"

# ---------------------------------------------------------------
# STEP 5: Clone OmniVoice repo (for training scripts)
# ---------------------------------------------------------------
echo ""
echo "📦 STEP 5: Setting up OmniVoice training..."

if [ ! -d "./OmniVoice" ]; then
    git clone https://github.com/k2-fsa/OmniVoice.git
    cd OmniVoice && pip install -e ".[train]" 2>/dev/null || pip install -e . && cd ..
fi

# Fix: flex_attention crashes on GPUs with <128KB shared memory (T4, A40, etc.).
# Run the full OmniVoice patch unconditionally so the builder and BlockMask
# paths stay in sync even if the repo was partially patched already.
python patch_omnivoice_attention.py --omnivoice-dir ./OmniVoice

export PYTHONPATH="./OmniVoice:${PYTHONPATH:-}"

# ---------------------------------------------------------------
# STEP 6: Merge JSONLs + split train/dev
# ---------------------------------------------------------------
echo ""
echo "📋 STEP 6: Preparing training data..."

mkdir -p data/finetune
MERGED="data/finetune/merged_all.jsonl"
TRAIN_JSONL="data/finetune/train.jsonl"
DEV_JSONL="data/finetune/dev.jsonl"

cat /dev/null > "${MERGED}"
for lang in fr ar zh; do
    SRC="dev_synth/train_${lang}.jsonl"
    if [ -f "${SRC}" ]; then
        echo "  Adding ${SRC} ($(wc -l < ${SRC}) samples)"
        cat "${SRC}" >> "${MERGED}"
    fi
done

TOTAL=$(wc -l < "${MERGED}")
echo "  Total: ${TOTAL} samples"

python3 -c "
import random
with open('${MERGED}') as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.85)
with open('${TRAIN_JSONL}', 'w') as f:
    f.writelines(lines[:split])
with open('${DEV_JSONL}', 'w') as f:
    f.writelines(lines[split:])
print(f'  Train: {split} | Dev: {len(lines) - split}')
"

# ---------------------------------------------------------------
# STEP 7: Tokenize audio
# ---------------------------------------------------------------
echo ""
echo "🔧 STEP 7: Tokenizing audio..."

TOKEN_DIR="data/finetune/tokens"
TOKENIZER="eustlb/higgs-audio-v2-tokenizer"

for split_name in train dev; do
    if [ "${split_name}" = "train" ]; then
        SPLIT_JSONL="${TRAIN_JSONL}"
    else
        SPLIT_JSONL="${DEV_JSONL}"
    fi

    echo "  Tokenizing ${split_name}..."
    CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
        --input_jsonl "${SPLIT_JSONL}" \
        --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
        --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
        --tokenizer_path "${TOKENIZER}" \
        --nj_per_gpu 3 \
        --shuffle True
done

# ---------------------------------------------------------------
# STEP 8: Create configs
# ---------------------------------------------------------------
echo ""
echo "📝 STEP 8: Creating training configs..."

CONFIG_DIR="data/finetune/config"
mkdir -p "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/train_config.json" << 'EOF'
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,
    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.3, 0.7],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,
    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 500,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.05,
    "warmup_steps": 0,
    "batch_tokens": 4096,
    "gradient_accumulation_steps": 16,
    "num_workers": 4,
    "mixed_precision": "bf16",
    "allow_tf32": true,
    "logging_steps": 25,
    "eval_steps": 100,
    "save_steps": 100,
    "keep_last_n_checkpoints": 3
}
EOF

cat > "${CONFIG_DIR}/data_config.json" << EOF
{
    "train": [{"manifest_path": ["${TOKEN_DIR}/train/data.lst"]}],
    "dev": [{"manifest_path": ["${TOKEN_DIR}/dev/data.lst"]}]
}
EOF

# ---------------------------------------------------------------
# STEP 9: Fine-tune
# ---------------------------------------------------------------
echo ""
echo "🚀 STEP 9: Fine-tuning OmniVoice..."
echo "  This will take a while..."

OUTPUT_DIR="./exp/omnivoice_finetuned"

# Disable torch.compile/dynamo as a safety net — flex_attention and inductor
# can fail on GPUs with limited shared memory.  Eager works without compilation.
export TORCHDYNAMO_DISABLE=1

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config "${CONFIG_DIR}/train_config.json" \
    --data_config "${CONFIG_DIR}/data_config.json" \
    --output_dir "${OUTPUT_DIR}"

echo "   ✅ Fine-tuning complete!"

# ---------------------------------------------------------------
# STEP 10: Upload fine-tuned model to HuggingFace
# ---------------------------------------------------------------
echo ""
echo "📤 STEP 10: Uploading fine-tuned model to HuggingFace..."

python3 << PYEOF
from huggingface_hub import HfApi, login
login(token="${HF_TOKEN}")
api = HfApi()

api.create_repo(
    repo_id="${HF_REPO_MODEL}",
    repo_type="model",
    exist_ok=True,
    private=False,
)

# Upload all checkpoints
api.upload_folder(
    folder_path="${OUTPUT_DIR}",
    repo_id="${HF_REPO_MODEL}",
    repo_type="model",
)

# Create model card
readme = """---
license: apache-2.0
base_model: k2-fsa/OmniVoice
tags:
  - voice-cloning
  - text-to-speech
  - iwslt-2026
language:
  - fr
  - ar
  - zh
datasets:
  - ${HF_REPO_DATASET}
  - ymoslem/acl-6060
---

# OmniVoice Fine-tuned for IWSLT 2026

Fine-tuned version of [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) for cross-lingual voice cloning.

## Training

- **Base model**: k2-fsa/OmniVoice (600+ languages, Qwen3-0.6B backbone)
- **Training data**: Best-of-N ensemble selection from OmniVoice, Qwen3-TTS, VoxCPM2, and Chatterbox
- **Dataset**: [${HF_REPO_DATASET}](https://huggingface.co/datasets/${HF_REPO_DATASET})
- **Languages**: French, Arabic, Chinese
- **Hardware**: A100 80GB
- **Method**: Full fine-tuning, 3000 steps, lr=1e-5, bf16

## Usage

\\\`\\\`\\\`python
from omnivoice import OmniVoice
import torch, torchaudio

model = OmniVoice.from_pretrained(
    "${HF_REPO_MODEL}",
    device_map="cuda:0",
    dtype=torch.float16
)
audio = model.generate(text="Bonjour!", ref_audio="ref.wav")
torchaudio.save("out.wav", audio[0], 24000)
\\\`\\\`\\\`
"""

api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id="${HF_REPO_MODEL}",
    repo_type="model",
)

print(f"✅ Model uploaded to: https://huggingface.co/${HF_REPO_MODEL}")
PYEOF

# ---------------------------------------------------------------
# DONE
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  🎉 PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "  📊 Dataset:  https://huggingface.co/datasets/${HF_REPO_DATASET}"
echo "  🤖 Model:    https://huggingface.co/${HF_REPO_MODEL}"
echo ""
echo "  To evaluate the fine-tuned model:"
echo "    python evaluation/evaluate_omnivoice.py \\"
echo "      --model-name ${OUTPUT_DIR} \\"
echo "      --whisper-lang fr --output-dir ./eval_results_omnivoice_ft/fr"
echo ""
