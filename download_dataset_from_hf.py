#!/usr/bin/env python3
"""
Download and filter Best-of-N dataset from HuggingFace for OmniVoice training.

This script exports WAV files and a JSONL manifest expected by
`omnivoice.scripts.extract_audio_tokens`.

Default behavior keeps only high-quality rows with best_score >= 0.6.
"""

import argparse
import json
import os
from typing import Iterable, List, Set

import soundfile as sf
from datasets import Audio, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export filtered HF dataset to OmniVoice JSONL manifest"
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", "amanuelbyte/omnivoice-best-of-n-training"),
        help="Hugging Face dataset repo id",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Keep only rows where best_score >= min-score",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=["ar", "zh", "fr"],
        help="Language filter (space-separated), e.g. --languages ar zh fr",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/finetune/wavs",
        help="Directory where WAV files are written",
    )
    parser.add_argument(
        "--jsonl-path",
        default="./data/finetune/merged_all.jsonl",
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not overwrite already exported wav files",
    )
    return parser.parse_args()


def should_keep_row(row: dict, min_score: float, languages: Set[str]) -> bool:
    lang = str(row.get("language", "")).strip().lower()
    if languages and lang not in languages:
        return False

    score = row.get("best_score", None)
    if score is None:
        return False

    try:
        return float(score) >= float(min_score)
    except (TypeError, ValueError):
        return False


def main():
    args = parse_args()
    languages = {x.strip().lower() for x in args.languages if x.strip()}

    print(f"Loading dataset: {args.repo_id} [{args.split}]")
    ds = load_dataset(args.repo_id, split=args.split)

    # Avoid runtime dependency on torchcodec when reading audio fields.
    # We only need raw bytes/path and will decode with soundfile when needed.
    if "best_audio" in ds.column_names:
        ds = ds.cast_column("best_audio", Audio(decode=False))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.jsonl_path) or ".", exist_ok=True)

    kept_rows: List[dict] = []
    for row in ds:
        if should_keep_row(row, args.min_score, languages):
            kept_rows.append(row)

    print(
        f"Filtering done: kept {len(kept_rows)}/{len(ds)} rows "
        f"with best_score >= {args.min_score} for languages={sorted(languages)}"
    )

    n_written = 0
    manifest_entries = []

    for row in kept_rows:
        audio_info = row.get("best_audio")
        if not isinstance(audio_info, dict):
            continue

        audio_array = audio_info.get("array")
        sample_rate = audio_info.get("sampling_rate")

        # decode=False returns either bytes/path or already-decoded array/sr.
        if audio_array is None or sample_rate is None:
            if audio_info.get("bytes") is not None:
                import io

                audio_array, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]))
            elif audio_info.get("path"):
                audio_array, sample_rate = sf.read(audio_info["path"])
        text = (row.get("text") or "").strip()
        language = (row.get("language") or "").strip().lower()
        sample_id = (row.get("id") or f"sample_{n_written:05d}").strip()

        if audio_array is None or sample_rate is None or not text:
            continue

        out_wav = os.path.join(args.output_dir, f"{sample_id}.wav")
        if not (args.keep_existing and os.path.exists(out_wav)):
            sf.write(out_wav, audio_array, sample_rate)

        manifest_entries.append(
            {
                "id": sample_id,
                "audio_path": os.path.abspath(out_wav),
                "text": text,
                "language_id": language,
            }
        )
        n_written += 1

    with open(args.jsonl_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Export complete: {n_written} audio files")
    print(f"Manifest: {args.jsonl_path}")


if __name__ == "__main__":
    main()
