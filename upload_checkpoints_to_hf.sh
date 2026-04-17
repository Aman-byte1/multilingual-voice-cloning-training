#!/bin/bash
# ============================================================
# Upload OmniVoice LoRA checkpoints to Hugging Face
# ============================================================
set -euo pipefail

HF_USER="amanuelbyte"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "❌ HF_TOKEN not set. Run: export HF_TOKEN=your_token_here"
    exit 1
fi

# Install huggingface_hub CLI if needed
pip install -q huggingface_hub

# Login
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

LANGUAGES=("zh" "fr" "ar")
LANG_NAMES=("chinese" "french" "arabic")

echo "============================================================"
echo "  Uploading OmniVoice LoRA checkpoints to HuggingFace"
echo "  Account: ${HF_USER}"
echo "============================================================"

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    LANG_NAME="${LANG_NAMES[$i]}"
    CKPT_DIR="./exp/omnivoice_finetuned_${LANG}"
    
    # Find all checkpoints
    CHECKPOINTS=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n)
    
    if [ -z "${CHECKPOINTS}" ]; then
        echo "  ⚠ No checkpoints found for ${LANG}, skipping"
        continue
    fi
    
    for CKPT in ${CHECKPOINTS}; do
        STEP=$(basename "${CKPT}" | sed 's/checkpoint-//')
        REPO_NAME="${HF_USER}/omnivoice-lora-${LANG}-${STEP}"
        
        echo ""
        echo "------------------------------------------------------------"
        echo "  📤 Uploading ${LANG} (${LANG_NAME}) — Step ${STEP}"
        echo "     Path: ${CKPT}"
        echo "     Repo: ${REPO_NAME}"
        echo "------------------------------------------------------------"
        
        # Create model card
        cat > "${CKPT}/README.md" << EOF
---
language:
- ${LANG}
license: apache-2.0
tags:
- omnivoice
- voice-cloning
- lora
- speech-synthesis
- tts
base_model: k2-fsa/OmniVoice
library_name: peft
---

# OmniVoice LoRA — ${LANG_NAME^} (${LANG}) — Step ${STEP}

Fine-tuned LoRA adapter for [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) to improve zero-shot voice cloning quality for **${LANG_NAME^}**.

## Training Details

- **Base model:** k2-fsa/OmniVoice (Qwen3-0.6B backbone)
- **Method:** LoRA (rank=32, alpha=64, RSLoRA)
- **Target modules:** Self-attention + audio projection layers
- **Training data:** Best-of-N distilled ${LANG_NAME^} speech samples
- **Steps:** ${STEP} / 400
- **Precision:** bf16
- **Hardware:** NVIDIA A40 (48GB)

## Usage

\`\`\`python
from omnivoice import OmniVoice
from peft import PeftModel
import torch

# Load base model
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)

# Load LoRA adapter
model.llm = PeftModel.from_pretrained(model.llm, "${REPO_NAME}")
model.llm = model.llm.merge_and_unload()

# Generate
audio = model.generate(
    text="Your ${LANG_NAME} text here",
    ref_audio="path/to/reference.wav",
)
\`\`\`

## Projects

This adapter was developed for the IWSLT 2026 shared task.
EOF

        # Upload to HF
        python3 -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = '${REPO_NAME}'
try:
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False, token='${HF_TOKEN}')
    api.upload_folder(
        folder_path='${CKPT}',
        repo_id=repo_id,
        commit_message='Upload OmniVoice LoRA ${LANG_NAME} step ${STEP}',
        token='${HF_TOKEN}',
    )
    print('  ✅ Uploaded to https://huggingface.co/' + repo_id)
except Exception as e:
    print(f'  ❌ Error uploading {repo_id}: {e}')
"
    done
done

echo ""
echo "============================================================"
echo "  🎉 All checkpoints uploaded to individual repos!"
echo "============================================================"

