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
If you prefer manual installation:
```bash
# Install PyTorch with correct CUDA (auto-detected in setup.sh)
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121

# All other dependencies
pip install -r evaluation/requirements.txt
```

## Running Evaluation

Use the `--resume` flag to avoid re-generating samples if the process was interrupted (e.g., OOM or crash).

### French (fr) — Base model
```bash
python evaluation/eval.py \
  --dataset ymoslem/acl-6060 --split eval --whisper-lang fr \
  --skip-lora --cfg-weight 0.0 --whisper-model large-v3 \
  --output-dir ./eval_results/fr_base --cache-dir ./data_cache \
  --resume
```

### Chinese (zh) — Base model
```bash
python evaluation/eval.py \
  --dataset ymoslem/acl-6060 --split eval --whisper-lang zh \
  --skip-lora --cfg-weight 0.0 --whisper-model large-v3 \
  --output-dir ./eval_results/zh_base --cache-dir ./data_cache \
  --resume
```

#### French (fr) — LoRA fine-tuned

```bash
python evaluation/eval.py \
  --dataset ymoslem/acl-6060 \
  --split eval \
  --whisper-lang fr \
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
