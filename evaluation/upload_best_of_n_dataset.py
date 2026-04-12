#!/usr/bin/env python3
"""
Upload Best-of-N curated dataset to Hugging Face Hub
=====================================================
Run this after synthesize_dev_best_of_n.py completes for all 3 languages.

Creates a HuggingFace dataset with:
  - Audio files (best output per sentence)
  - JSONL metadata (text, language, source model, quality scores)

Usage:
  python evaluation/upload_best_of_n_dataset.py \
    --repo-id Aman-byte1/omnivoice-best-of-n-training \
    --hf-token $HF_TOKEN
"""

import os
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser(description="Upload best-of-N dataset to HF Hub")
    parser.add_argument("--dev-synth-dir", default="./dev_synth",
                        help="Directory containing scores_*.csv and train_*.jsonl")
    parser.add_argument("--repo-id", default="Aman-byte1/omnivoice-best-of-n-training",
                        help="HuggingFace repo ID")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()
    synth_dir = Path(args.dev_synth_dir)

    # Create the repo
    print(f"📦 Creating/updating repo: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
        token=token,
    )

    # Merge all JSONL manifests into one
    merged = []
    for lang in ["fr", "ar", "zh"]:
        jsonl_path = synth_dir / f"train_{lang}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    merged.append(entry)
            print(f"  {lang}: {sum(1 for e in merged if e.get('language_id') == lang)} samples")

    # Write merged manifest
    merged_path = synth_dir / "train_all.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n📤 Uploading to {args.repo_id}...")

    # Upload JSONL manifests
    for lang in ["fr", "ar", "zh"]:
        for fname in [f"train_{lang}.jsonl", f"scores_{lang}.csv"]:
            fpath = synth_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  ✅ Uploaded {fname}")

    # Upload merged manifest
    api.upload_file(
        path_or_fileobj=str(merged_path),
        path_in_repo="train_all.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )
    print(f"  ✅ Uploaded train_all.jsonl ({len(merged)} samples)")

    # Upload audio files per model per language
    for lang in ["fr", "ar", "zh"]:
        for model_dir in (synth_dir).iterdir():
            audio_dir = model_dir / lang
            if audio_dir.is_dir() and any(audio_dir.glob("synth_*.wav")):
                print(f"  Uploading {model_dir.name}/{lang}/ ...")
                api.upload_folder(
                    folder_path=str(audio_dir),
                    path_in_repo=f"audio/{model_dir.name}/{lang}",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  ✅ Uploaded {model_dir.name}/{lang}/")

    # Upload ref audio
    ref_dir = synth_dir / "ref_audio"
    if ref_dir.exists():
        print(f"  Uploading ref_audio/ ...")
        api.upload_folder(
            folder_path=str(ref_dir),
            path_in_repo="audio/ref",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"  ✅ Uploaded ref_audio/")

    # Create README
    readme = f"""---
license: apache-2.0
task_categories:
  - text-to-speech
language:
  - fr
  - ar
  - zh
tags:
  - voice-cloning
  - best-of-n
  - iwslt-2026
size_categories:
  - 1K<n<10K
---

# Best-of-N Voice Cloning Training Data

Curated training dataset for fine-tuning [OmniVoice](https://github.com/k2-fsa/OmniVoice).

## Dataset Description

For each sentence in the dev split of `ymoslem/acl-6060` (468 samples × 3 languages),
we synthesized audio with the **top 3 voice cloning models** per language and selected
the best output per sentence based on a combined quality score (50% CER + 50% speaker similarity).

### Models used per language

| Language | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| French | OmniVoice | Qwen3-TTS | Chatterbox |
| Arabic | OmniVoice | Chatterbox | VoxCPM2 |
| Chinese | OmniVoice | VoxCPM2 | Qwen3-TTS |

### Total samples: ~{len(merged)}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{args.repo_id}")
```

## Quality Metrics

See `scores_{{lang}}.csv` for per-sentence quality scores from each model.
"""
    readme_path = synth_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"\n🎉 Dataset uploaded to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
