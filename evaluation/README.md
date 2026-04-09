# Cross-Lingual Voice Cloning — Evaluation

Evaluation pipeline for Chatterbox (base & LoRA) on the [ymoslem/acl-6060](https://huggingface.co/datasets/ymoslem/acl-6060) dataset.

**Metrics:** WER, CER, Speaker Similarity, Inference Time, RTF

---

## Setup & Installation

### 1. Fresh Server Setup
If you are on a new server or hit environment errors (like `torchvision::nms` issues), run the setup script:

```bash
bash evaluation/setup.sh
```

### 2. Manual Installation (Optional)
If you prefer manual installation (confirmed working for CUDA 12.8):
```bash
# Aggressive clean
pip uninstall -y torch torchvision torchaudio

# Install matched set
pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# All other dependencies
pip install "datasets<4.0" soundfile
pip install -r evaluation/requirements.txt
```

## Running Evaluation

Use the `--resume` flag to avoid re-generating samples if the process was interrupted (e.g., OOM or crash).

### Chatterbox (Base)
```bash
python evaluation/eval.py \
  --model-type chatterbox \
  --dataset ymoslem/acl-6060 --split eval --whisper-lang zh \
  --skip-lora --output-dir ./eval_results/zh_chatterbox \
  --resume
```

### Qwen3-TTS (Base)
```bash
python evaluation/eval.py \
  --model-type qwen \
  --dataset ymoslem/acl-6060 --split eval --whisper-lang zh \
  --output-dir ./eval_results/zh_qwen \
  --resume
```

### Chatterbox (LoRA)
```bash
python evaluation/eval.py \
  --model-type chatterbox \
  --dataset ymoslem/acl-6060 --split eval --whisper-lang fr \
  --repo-id amanuelbyte/chatterbox-fr-lora \
  --lora-file best_lora_adapter.pt \
  --cfg-weight 0.0 \
  --whisper-model large-v3 \
  --output-dir ./eval_results/fr_lora \
  --cache-dir ./data_cache
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ymoslem/acl-6060` | HuggingFace dataset name |
| `--split` | `eval` | Dataset split to evaluate |
| `--whisper-lang` | `fr` | Target language code (`fr`, `ar`, `zh`, etc.) |
| `--whisper-model` | `large-v3` | Faster-whisper model size |
| `--whisper-beam` | `5` | Beam size for ASR decoding |
| `--skip-lora` | `false` | Use base Chatterbox (no LoRA) |
| `--repo-id` | `amanuelbyte/chatterbox-fr-lora` | HF repo for LoRA weights |
| `--lora-file` | `best_lora_adapter.pt` | LoRA checkpoint filename |
| `--cfg-weight` | `0.0` | Classifier-free guidance weight |
| `--max-samples` | `None` | Limit number of samples (all if unset) |
| `--output-dir` | `./eval_results` | Where to save outputs |
| `--cache-dir` | `./data_cache` | Dataset cache directory |

---

## Output

Results are saved to `<output-dir>/`:

- `synth_XXXXX.wav` — Generated audio files
- `eval_summary.json` — Aggregated metrics (mean, std, valid count)

---

## Troubleshooting

### `torchvision::nms does not exist`
**Cause:** Mismatched `torch` / `torchvision` versions.  
**Fix:** Reinstall as a matched set:
```bash
pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
```

### `ImportError: please install 'torchcodec'`
**Cause:** `datasets>=4.0` requires `torchcodec` for audio decoding.  
**Fix:** Downgrade datasets:
```bash
pip install "datasets<4.0" soundfile
```

### `numpy` version conflicts
**Cause:** `chatterbox-tts` requires `numpy<2.0`.  
**Fix:**
```bash
pip install "numpy<2.0"
```
