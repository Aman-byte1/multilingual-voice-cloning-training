# OmniVoice Training & Best-of-N Synthesis

This repository contains tools for multilingual voice cloning, including fine-tuning OmniVoice with LoRA and synthesizing datasets using a Best-of-N strategy across multiple models (OmniVoice, Chatterbox, VoxCPM).

## 📥 Setup & Dependencies

To set up the environment (optimized for CUDA/A40 environments):

```bash
# Basic setup
pip install omnivoice datasets==2.20.0 "numpy<1.26" cython setuptools wheel conformer
TF_VER=$(python3 -c "import transformers; print(transformers.__version__)")

# TTS Model dependencies
pip install resemble-perth pkuseg s3tokenizer omegaconf diffusers==0.29.0 --no-build-isolation
pip install voxcpm --no-deps 2>/dev/null || pip install voxcpm
pip install chatterbox-tts --no-deps 2>/dev/null || pip install chatterbox-tts

# Re-pin transformers and install eval/scoring tools
pip install "transformers==${TF_VER}"
pip install faster-whisper jiwer speechbrain torchaudio soundfile huggingface_hub

# Login to Hugging Face
export HF_TOKEN=your_token_here
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
```

## 🎙 Best-of-N Data Synthesis (dev + eval)

Synthesize the combined `dev` and `eval` splits to create a high-quality training dataset.

### 1. Run French (fr) and Arabic (ar) Synthesis
```bash
for lang in fr ar; do
    TORCHDYNAMO_DISABLE=1 python evaluation/synthesize_dev_best_of_n.py \
        --lang $lang \
        --splits dev eval \
        --output-dir ./dev_eval_synth \
        --cache-dir ./data_cache \
        --hf-token "${HF_TOKEN}"
done

# Push FR and AR results to HuggingFace
python evaluation/upload_best_of_n_dataset.py \
    --dev-synth-dir ./dev_eval_synth \
    --repo-id amanuelbyte/omnivoice-best-of-n-dev-eval \
    --hf-token "${HF_TOKEN}"
```

### 2. Run Chinese (zh) Synthesis
```bash
TORCHDYNAMO_DISABLE=1 python evaluation/synthesize_dev_best_of_n.py \
    --lang zh \
    --splits dev eval \
    --output-dir ./dev_eval_synth \
    --cache-dir ./data_cache \
    --hf-token "${HF_TOKEN}"

# Final Push (includes all 3 languages)
python evaluation/upload_best_of_n_dataset.py \
    --dev-synth-dir ./dev_eval_synth \
    --repo-id amanuelbyte/omnivoice-best-of-n-dev-eval \
    --hf-token "${HF_TOKEN}"
```

> [!TIP]
> Since synthesis is partially CPU-bound, you can run multiple languages in parallel in different terminals to maximize GPU utilization on high-VRAM cards like the A40.

## 🚀 Training Pipeline

To run the full pipeline including synthesis, upload, LoRA fine-tuning, and blind evaluation:

```bash
bash run_dev_eval_pipeline.sh
```
