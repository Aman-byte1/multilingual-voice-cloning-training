#!/bin/bash
# ============================================================
# FULL DEV+EVAL PIPELINE — Clone → Install → Synth → Train → Upload
# ============================================================
# Usage (one-liner on A100/H100):
#   export HF_TOKEN=hf_xxx
#   curl -sL https://raw.githubusercontent.com/Aman-byte1/multilingual-voice-cloning-training/main/run_dev_eval_pipeline.sh | bash
#
# OR:
#   git clone https://github.com/Aman-byte1/multilingual-voice-cloning-training.git && \
#   cd multilingual-voice-cloning-training && \
#   bash run_dev_eval_pipeline.sh
# ============================================================

set -euo pipefail

# ====== Configuration ======
HF_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN env var first — export HF_TOKEN=hf_xxx}"
HF_USERNAME="amanuelbyte"
HF_REPO_DATASET="${HF_USERNAME}/omnivoice-best-of-n-dev-eval"
WORKDIR="$(pwd)"

LANGUAGES=("fr" "ar" "zh")
LORA_RANK=32
LORA_ALPHA=64
STEPS=600          # Increased from 400 — ~2x more data needs more steps
LEARNING_RATE="1e-4"
GPU_IDS="0"
NUM_GPUS=1
MIN_SCORE="0.65"
# ===========================

echo "============================================================"
echo "  DEV+EVAL PIPELINE: Synth → Train → Upload"
echo "  Started: $(date)"
echo "  Training splits: dev + eval (884 samples)"
echo "  Evaluation: blind test set"
echo "============================================================"

# ---------------------------------------------------------------
# STEP 1: Install all dependencies
# ---------------------------------------------------------------
echo ""
echo "📦 STEP 1: Installing dependencies..."

pip install --upgrade pip

# System deps (sox for Qwen3-TTS, ffmpeg for audio)
apt-get update -qq && apt-get install -y -qq sox ffmpeg > /dev/null 2>&1 || echo "   ⚠ Could not install sox/ffmpeg (non-root?)"

# Install OmniVoice FIRST — it pins the correct transformers version
pip install omnivoice datasets==2.20.0

# Record the transformers version omnivoice needs
TF_VER=$(python3 -c "import transformers; print(transformers.__version__)")
echo "   OmniVoice installed with transformers==${TF_VER}"

# Pre-install numpy and build deps for pkuseg (chatterbox dependency)
pip install "numpy<1.26" cython setuptools wheel

# Install other TTS models WITHOUT upgrading transformers
pip install resemble-perth pkuseg s3tokenizer omegaconf diffusers==0.29.0 --no-build-isolation
pip install voxcpm --no-deps 2>/dev/null || pip install voxcpm
pip install chatterbox-tts --no-deps 2>/dev/null || pip install chatterbox-tts

# Re-pin transformers
pip install "transformers==${TF_VER}"

# Evaluation + scoring
pip install faster-whisper jiwer speechbrain torchaudio soundfile accelerate

# OmniVoice training deps
pip install "omnivoice[train]" 2>/dev/null || pip install webdataset
pip install peft safetensors

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
"

echo "   ✅ All dependencies installed"

# ---------------------------------------------------------------
# STEP 2: Login to HuggingFace
# ---------------------------------------------------------------
echo ""
echo "🔑 STEP 2: HuggingFace login..."
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

# ---------------------------------------------------------------
# STEP 3: Best-of-N synthesis (dev + eval splits)
# ---------------------------------------------------------------
echo ""
echo "🎙  STEP 3: Best-of-N synthesis on dev+eval splits..."
echo "   This synthesizes with multiple models and picks the best per sample."

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "  ── Synthesizing ${lang} (dev+eval) ──"
    python evaluation/synthesize_dev_best_of_n.py \
        --lang ${lang} \
        --dataset ymoslem/acl-6060 \
        --splits dev eval \
        --output-dir ./dev_eval_synth \
        --cache-dir ./data_cache \
        --hf-token "${HF_TOKEN}"
    echo "  ✅ ${lang} synthesis complete"
done

echo "   ✅ All 3 languages synthesized from dev+eval"

# ---------------------------------------------------------------
# STEP 4: Upload synthesized dataset to HuggingFace
# ---------------------------------------------------------------
echo ""
echo "📤 STEP 4: Uploading synthesized dataset to HuggingFace..."

python evaluation/upload_best_of_n_dataset.py \
    --dev-synth-dir ./dev_eval_synth \
    --repo-id "${HF_REPO_DATASET}" \
    --hf-token "${HF_TOKEN}" 2>/dev/null || echo "  ⚠ Upload had warnings (non-fatal)"

echo "   ✅ Dataset uploaded to ${HF_REPO_DATASET}"

# ---------------------------------------------------------------
# STEP 5: Clone OmniVoice repo (for training scripts)
# ---------------------------------------------------------------
echo ""
echo "📦 STEP 5: Setting up OmniVoice training environment..."

OMNIVOICE_DIR="./OmniVoice"
if [ ! -d "${OMNIVOICE_DIR}" ]; then
    git clone https://github.com/k2-fsa/OmniVoice.git "${OMNIVOICE_DIR}"
    cd "${OMNIVOICE_DIR}" && pip install -e ".[train]" 2>/dev/null || pip install -e . && cd ..
fi

# Patch flex_attention for GPU compat
python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------
# STEP 6: Per-language LoRA fine-tuning
# ---------------------------------------------------------------
echo ""
echo "🚀 STEP 6: Per-language LoRA fine-tuning on dev+eval data..."

export TORCHDYNAMO_DISABLE=1

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  [START] Fine-Tuning: ${LANG^^} (dev+eval, ${STEPS} steps)"
    echo "============================================================"

    DATA_DIR="./data/finetune_${LANG}"
    OUTPUT_DIR="./exp/omnivoice_lora_${LANG}_deveval"
    TOKEN_DIR="${DATA_DIR}/tokens"
    CONFIG_DIR="${DATA_DIR}/config"
    mkdir -p "${CONFIG_DIR}"

    # --- Prepare JSONL manifest ---
    MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
    TRAIN_JSONL="${DATA_DIR}/train.jsonl"
    DEV_JSONL="${DATA_DIR}/dev.jsonl"

    # Copy the Best-of-N synthesized JSONL
    SRC_JSONL="./dev_eval_synth/train_${LANG}.jsonl"
    if [ -f "${SRC_JSONL}" ]; then
        cp "${SRC_JSONL}" "${MERGED_JSONL}"
        echo "  Using synthesized manifest: ${SRC_JSONL} ($(wc -l < ${MERGED_JSONL}) samples)"
    else
        echo "  ❌ Missing ${SRC_JSONL} — skipping ${LANG}"
        continue
    fi

    # 90/10 train/dev split
    python3 -c "
import random
with open('${MERGED_JSONL}') as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.90)
with open('${TRAIN_JSONL}', 'w') as f:
    f.writelines(lines[:split])
with open('${DEV_JSONL}', 'w') as f:
    f.writelines(lines[split:])
print(f'  Train: {split} | Dev: {len(lines) - split}')
"

    # --- Tokenize audio ---
    TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"
    for split_name in train dev; do
        SPLIT_JSONL="${TRAIN_JSONL}"
        [ "${split_name}" = "dev" ] && SPLIT_JSONL="${DEV_JSONL}"

        if [ -f "${TOKEN_DIR}/${split_name}/data.lst" ] && [ -s "${TOKEN_DIR}/${split_name}/data.lst" ]; then
            echo "  Tokens already exist for ${split_name} (${LANG}). Skipping."
            continue
        fi

        echo "  Tokenizing ${split_name} for ${LANG}..."
        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            python extract_audio_tokens_compat.py \
            --input_jsonl "${SPLIT_JSONL}" \
            --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 1 \
            --shuffle True
    done

    # --- Create configs ---
    cat > "${CONFIG_DIR}/train_config.json" << EOF
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,
    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.05,
    "prompt_ratio_range": [0.3, 0.7],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 1.0,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,
    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",
    "learning_rate": ${LEARNING_RATE},
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": ${STEPS},
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.05,
    "batch_tokens": 4096,
    "gradient_accumulation_steps": 16,
    "num_workers": 2,
    "mixed_precision": "no",
    "allow_tf32": false,
    "logging_steps": 10,
    "eval_steps": 50,
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

    # --- Launch fine-tuning ---
    echo "  Launching LoRA fine-tuning for ${LANG}..."
    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        finetune_omnivoice_per_lang.py \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --train_config "${CONFIG_DIR}/train_config.json" \
        --data_config "${CONFIG_DIR}/data_config.json" \
        --output_dir "${OUTPUT_DIR}" \
        --use_rslora \
        --steps ${STEPS}

    echo "============================================================"
    echo "  [DONE] ${LANG^^} — checkpoints: ${OUTPUT_DIR}"
    echo "============================================================"
done

echo ""
echo "🎉 All languages fine-tuned!"

# ---------------------------------------------------------------
# STEP 7: Upload LoRA checkpoints to HuggingFace
# ---------------------------------------------------------------
echo ""
echo "📤 STEP 7: Uploading LoRA checkpoints to HuggingFace..."

python3 << 'PYEOF'
import os, glob, json
from huggingface_hub import HfApi, login

TOKEN = os.environ["HF_TOKEN"]
USERNAME = "amanuelbyte"
login(token=TOKEN)
api = HfApi()

LANGUAGES = ["fr", "ar", "zh"]
STEPS = int(os.environ.get("STEPS", "600"))

for lang in LANGUAGES:
    exp_dir = f"./exp/omnivoice_lora_{lang}_deveval"
    if not os.path.isdir(exp_dir):
        print(f"  ⚠ Skipping {lang}: {exp_dir} not found")
        continue

    # Find the best checkpoint (latest step)
    checkpoints = sorted(glob.glob(os.path.join(exp_dir, "checkpoint-*")),
                         key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else 0)

    # Upload the final_lora adapter and all checkpoints
    for ckpt_dir in checkpoints:
        step_num = ckpt_dir.split("-")[-1]
        repo_id = f"{USERNAME}/omnivoice-lora-{lang}-deveval-{step_num}"

        print(f"  📤 Uploading {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=ckpt_dir,
            repo_id=repo_id,
            repo_type="model",
        )

        # Create model card
        readme = f"""---
license: apache-2.0
base_model: k2-fsa/OmniVoice
tags:
  - voice-cloning
  - text-to-speech
  - iwslt-2026
  - lora
language:
  - {lang}
datasets:
  - ymoslem/acl-6060
  - {USERNAME}/omnivoice-best-of-n-dev-eval
---

# OmniVoice LoRA — {lang.upper()} (Dev+Eval, Step {step_num})

LoRA adapter for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
fine-tuned on **dev + eval** splits of ACL-6060 using Best-of-N ensemble distillation.

## Training Details
- **Base model**: k2-fsa/OmniVoice (Qwen3-0.6B backbone)
- **Training data**: Best-of-N selection from OmniVoice, Chatterbox, VoxCPM2
- **Splits used**: dev (468) + eval (416) = 884 samples
- **LoRA rank**: 32, alpha: 64, RSLoRA
- **Steps**: {step_num}
- **Language**: {lang}
"""
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✅ {repo_id} uploaded")

    # Also upload final_lora if it exists
    final_lora = os.path.join(exp_dir, "final_lora")
    if os.path.isdir(final_lora):
        repo_id = f"{USERNAME}/omnivoice-lora-{lang}-deveval-final"
        print(f"  📤 Uploading {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=final_lora, repo_id=repo_id, repo_type="model")
        print(f"  ✅ {repo_id} uploaded")

print("\n✅ All models uploaded!")
PYEOF

# ---------------------------------------------------------------
# STEP 8: Evaluate on blind test set
# ---------------------------------------------------------------
echo ""
echo "📊 STEP 8: Evaluating fine-tuned models on blind test set..."

# We need the blind test files
if [ ! -d "./blind_test" ]; then
    echo "  ⚠ blind_test/ directory not found. Skipping evaluation."
    echo "  To run evaluation later, place blind test data in ./blind_test/{text,audio}/"
else
    for LANG in "${LANGUAGES[@]}"; do
        echo "  Evaluating ${LANG}..."
        CUDA_VISIBLE_DEVICES=${GPU_IDS} python evaluation/evaluate_blind_single.py \
            --model omnivoice_finetuned \
            --lang ${LANG} \
            --output-dir ./eval_blind_deveval \
            > logs/blind_eval_${LANG}_deveval.log 2>&1 || echo "  ⚠ Eval for ${LANG} had issues (check logs)"
    done

    # Print summary
    python3 - << 'SUMEOF'
import json, os, glob

root = "./eval_blind_deveval"
summaries = glob.glob(os.path.join(root, "*/*/summary.json"))

if not summaries:
    print("No evaluation results found!")
    exit()

print(f"\n  {'Model':<20} {'Lang':<5} {'WER↓':>8} {'CER↓':>8} {'SIM↑':>8}")
print("  " + "-" * 55)

for path in sorted(summaries):
    with open(path) as f:
        d = json.load(f)
    print(f"  {d['model']:<20} {d['lang']:<5} {d['WER']:>8.4f} {d['CER']:>8.4f} {d['Similarity']:>8.4f}")

print("  " + "=" * 55)
SUMEOF
fi

# ---------------------------------------------------------------
# DONE
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  🎉 FULL PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "  📊 Dataset: https://huggingface.co/datasets/${HF_REPO_DATASET}"
echo "  🤖 Models:  https://huggingface.co/${HF_USERNAME} (omnivoice-lora-*-deveval-*)"
echo ""
echo "  Training: dev+eval splits (884 samples)"
echo "  Evaluation: blind test set"
echo ""
