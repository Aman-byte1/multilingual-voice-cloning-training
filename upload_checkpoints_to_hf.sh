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
    REPO_NAME="${HF_USER}/omnivoice-lora-${LANG}"
    
    # Find the latest checkpoint
    CKPT_DIR="./exp/omnivoice_finetuned_${LANG}"
    LATEST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    
    if [ -z "${LATEST_CKPT}" ]; then
        echo "  ⚠ No checkpoint found for ${LANG}, skipping"
        continue
    fi
    
    echo ""
    echo "------------------------------------------------------------"
    echo "  📤 Uploading ${LANG} (${LANG_NAME})"
    echo "     Checkpoint: ${LATEST_CKPT}"
    echo "     Repo: ${REPO_NAME}"
    echo "------------------------------------------------------------"
    
    # Create model card
    STEP=$(basename "${LATEST_CKPT}" | sed 's/checkpoint-//')
    cat > "${LATEST_CKPT}/README.md" << EOF
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

# OmniVoice LoRA — ${LANG_NAME^} (${LANG})

Fine-tuned LoRA adapter for [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) to improve zero-shot voice cloning quality for **${LANG_NAME^}**.

## Training Details

- **Base model:** k2-fsa/OmniVoice (Qwen3-0.6B backbone)
- **Method:** LoRA (rank=32, alpha=64, RSLoRA)
- **Target modules:** Self-attention + audio projection layers
- **Training data:** Best-of-N distilled ${LANG_NAME^} speech samples
- **Steps:** ${STEP}
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

## IWSLT 2026

This adapter was developed for the IWSLT 2026 shared task on cross-lingual voice cloning.
EOF

    # Upload to HF
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${REPO_NAME}', exist_ok=True, private=False, token='${HF_TOKEN}')
api.upload_folder(
    folder_path='${LATEST_CKPT}',
    repo_id='${REPO_NAME}',
    commit_message='Upload OmniVoice LoRA checkpoint-${STEP} for ${LANG_NAME}',
    token='${HF_TOKEN}',
)
print('  ✅ Uploaded to https://huggingface.co/${REPO_NAME}')
"
done

echo ""
echo "============================================================"
echo "  🎉 All checkpoints uploaded!"
echo "  ${HF_USER}/omnivoice-lora-zh"
echo "  ${HF_USER}/omnivoice-lora-fr"  
echo "  ${HF_USER}/omnivoice-lora-ar"
echo "============================================================"
