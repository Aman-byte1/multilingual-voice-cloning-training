#!/usr/bin/env python3
"""
Upload Best-of-N curated dataset to Hugging Face Hub
=====================================================
Creates a proper HuggingFace dataset with PLAYABLE audio columns
so you can listen to all synthesized voices directly in the browser.

Usage:
  python evaluation/upload_best_of_n_dataset.py \
    --repo-id amanuelbyte/omnivoice-best-of-n-training \
    --hf-token $HF_TOKEN
"""

import os
import json
import argparse
from pathlib import Path

from datasets import Dataset, Audio, Features, Value
from huggingface_hub import HfApi, login


def build_dataset_for_lang(synth_dir: Path, lang: str):
    """Build a list of dicts for one language with audio paths."""
    scores_path = synth_dir / f"scores_{lang}.csv"
    jsonl_path = synth_dir / f"train_{lang}.jsonl"

    if not jsonl_path.exists():
        print(f"  ⚠ No train_{lang}.jsonl found, skipping {lang}")
        return []

    # Load the JSONL manifest
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    # Load scores CSV for richer metadata
    scores_by_idx = {}
    if scores_path.exists():
        import csv
        with open(scores_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row.get("idx", "")
                scores_by_idx[idx] = row

    rows = []
    for entry in entries:
        audio_path = entry.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            continue

        # Extract the sample index from the ID (e.g. "fr_00042" -> "42")
        sample_id = entry.get("id", "")
        idx_str = sample_id.split("_")[-1].lstrip("0") or "0"

        # Find the reference audio
        ref_path = str(synth_dir / "ref_audio" / f"ref_{int(idx_str):05d}.wav")
        if not os.path.exists(ref_path):
            ref_path = None

        # Get scores metadata
        score_row = scores_by_idx.get(idx_str, {})
        best_model = score_row.get("best_model", "")
        best_score = score_row.get("best_score", "")

        row = {
            "id": sample_id,
            "language": lang,
            "text": entry.get("text", ""),
            "best_model": best_model,
            "best_score": float(best_score) if best_score else 0.0,
            "best_audio": audio_path,
        }

        if ref_path:
            row["ref_audio"] = ref_path

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Upload best-of-N dataset to HF Hub")
    parser.add_argument("--dev-synth-dir", default="./dev_synth",
                        help="Directory containing scores_*.csv and train_*.jsonl")
    parser.add_argument("--repo-id", default="amanuelbyte/omnivoice-best-of-n-training",
                        help="HuggingFace repo ID")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    synth_dir = Path(args.dev_synth_dir)

    # ---------------------------------------------------------------
    # Collect all rows across languages
    # ---------------------------------------------------------------
    all_rows = []
    for lang in ["fr", "ar", "zh"]:
        lang_rows = build_dataset_for_lang(synth_dir, lang)
        print(f"  {lang}: {len(lang_rows)} samples")
        all_rows.extend(lang_rows)

    if not all_rows:
        print("❌ No samples found! Make sure synthesize_dev_best_of_n.py has completed.")
        return

    # Ensure all rows have ref_audio (fill missing with None)
    for row in all_rows:
        if "ref_audio" not in row:
            row["ref_audio"] = None

    print(f"\n  Total: {len(all_rows)} samples across all languages")

    # ---------------------------------------------------------------
    # Create a proper HuggingFace Dataset with Audio columns
    # ---------------------------------------------------------------
    print(f"\n🔧 Building HuggingFace Dataset with playable audio...")

    ds = Dataset.from_list(all_rows)

    # Cast audio columns so HF viewer shows playable audio players
    ds = ds.cast_column("best_audio", Audio())
    ds = ds.cast_column("ref_audio", Audio())

    print(f"   Dataset: {ds}")
    print(f"   Features: {ds.features}")

    # ---------------------------------------------------------------
    # Push to HuggingFace Hub
    # ---------------------------------------------------------------
    print(f"\n📤 Pushing dataset to {args.repo_id}...")

    ds.push_to_hub(
        args.repo_id,
        token=token,
        private=args.private,
    )

    print(f"   ✅ Dataset pushed!")

    # ---------------------------------------------------------------
    # Upload supplementary files (scores CSVs, summaries)
    # ---------------------------------------------------------------
    api = HfApi()

    for lang in ["fr", "ar", "zh"]:
        for fname in [f"scores_{lang}.csv", f"summary_{lang}.json"]:
            fpath = synth_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"metadata/{fname}",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print(f"   ✅ Uploaded metadata/{fname}")

    # ---------------------------------------------------------------
    # Create README
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # Create Detailed Results Tables per Language
    # ---------------------------------------------------------------
    detailed_stats = ""
    for lang in ["fr", "ar", "zh"]:
        summary_path = synth_dir / f"summary_{lang}.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    data = json.load(f)
                
                detailed_stats += f"### {lang.upper()} Results\n\n"
                detailed_stats += "| Model | Rows Won | % | Avg CER | Avg Sim | Avg Score |\n"
                detailed_stats += "|---|---|---|---|---|---|\n"
                
                # Sort models by Avg Score
                results = data.get("results", [])
                results.sort(key=lambda x: x.get("avg_score", 0), reverse=True)
                
                for r in results:
                    detailed_stats += (
                        f"| {r['model']} | {r['rows_won']} | {r['pct_won']:.1f}% | "
                        f"{r['avg_cer']:.4f} | {r['avg_sim']:.4f} | {r['avg_score']:.4f} |\n"
                    )
                
                best_n = data.get("best_of_n", {})
                detailed_stats += (
                    f"| **BEST-OF-N** | **{best_n.get('total', 0)}** | **100%** | "
                    f"— | — | **{best_n.get('avg_score', 0):.4f}** |\n\n"
                )
            except Exception as e:
                print(f"  ⚠ Could not parse {summary_path}: {e}")

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
  - omnivoice
size_categories:
  - 1K<n<10K
---

# 🎙 Best-of-N Voice Cloning Training Data

Curated training dataset for fine-tuning [OmniVoice](https://github.com/k2-fsa/OmniVoice) for IWSLT 2026.

## 🎧 Listen to the Audio

This dataset has **playable audio columns** — click on any row in the dataset viewer
to listen to both the **reference audio** (original speaker) and the **best synthesized audio**
(selected by quality score).

## Dataset Description

For each sentence in the dev split of `ymoslem/acl-6060` (884 samples × 3 languages),
we synthesized audio with the **top voice cloning models** per language and selected
the best output based on: **Combined = 0.5 × (1 - CER) + 0.5 × Speaker_Similarity**

### Columns

| Column | Description |
|---|---|
| `ref_audio` | 🔊 Original reference speaker audio (playable) |
| `best_audio` | 🔊 Best synthesized voice clone (playable) |
| `text` | Target text that was synthesized |
| `language` | Language code (fr, ar, zh) |
| `best_model` | Which model won for this sample |
| `best_score` | Combined quality score (0-1, higher=better) |

### Detailed Results by Language

{detailed_stats}

### Total samples: {len(all_rows)}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{args.repo_id}")

# Play the first sample
print(ds["train"][0]["text"])
print(ds["train"][0]["best_model"])
# Audio is accessible as ds["train"][0]["best_audio"]["array"]
```

## Quality Metrics

See `metadata/scores_{{lang}}.csv` for per-sentence quality scores from each model.
"""

    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"\n🎉 Dataset live at: https://huggingface.co/datasets/{args.repo_id}")
    print(f"   Open it and click any row to play the audio! 🔊")


if __name__ == "__main__":
    main()
