#!/usr/bin/env python3
"""
Build VoxCPM train/val JSONL manifests from a Hugging Face dataset.

Output schema matches OpenBMB VoxCPM training loader:
- text: str
- audio: path/to/target_audio.wav
- ref_audio: path/to/reference_audio.wav (optional but recommended)
- dataset_id: int
"""

import argparse
import json
import os
import random
from typing import Iterable, Optional

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def first_non_empty(row: dict, keys: Iterable[str]):
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return None


def save_wav(audio_field: dict, out_path: str) -> None:
    arr = np.asarray(audio_field["array"], dtype=np.float32)
    sr = int(audio_field["sampling_rate"])
    wav = torch.from_numpy(arr).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, wav, sample_rate=sr)


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VoxCPM manifests from HF dataset")
    parser.add_argument("--dataset", default="amanuelbyte/omnivoice-best-of-n-training")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="./data/voxcpm_zh")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-text-len", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--score-fields", default="score,similarity")
    parser.add_argument(
        "--reject-missing-score",
        action="store_true",
        help="If set, samples with no score field will be rejected when --min-score is used",
    )
    parser.add_argument(
        "--text-fields",
        default="trg_zh_text,text_zh,text",
        help="Comma-separated candidate text fields in priority order",
    )
    parser.add_argument(
        "--target-audio-fields",
        default="trg_zh_voice,audio_zh,trg_voice,audio,tts_audio",
        help="Comma-separated candidate target audio fields in priority order",
    )
    parser.add_argument(
        "--ref-audio-fields",
        default="ref_en_voice,audio_en,ref_audio,prompt_audio",
        help="Comma-separated candidate reference audio fields in priority order",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    text_fields = parse_csv_list(args.text_fields)
    target_audio_fields = parse_csv_list(args.target_audio_fields)
    ref_audio_fields = parse_csv_list(args.ref_audio_fields)
    score_fields = parse_csv_list(args.score_fields)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)

    out_dir = os.path.abspath(args.output_dir)
    tgt_dir = os.path.join(out_dir, "target_wavs")
    ref_dir = os.path.join(out_dir, "ref_wavs")
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    rows = []
    rejected = {
        "text": 0,
        "target_audio": 0,
        "ref_audio": 0,
        "score": 0,
        "missing_score": 0,
    }

    for idx in tqdm(range(len(ds)), desc="Preparing"):
        item = ds[idx]

        score: Optional[float] = None
        for sf in score_fields:
            v = item.get(sf)
            if v is not None:
                try:
                    score = float(v)
                    break
                except Exception:
                    pass
        if score is None:
            rejected["missing_score"] += 1
            if args.reject_missing_score:
                rejected["score"] += 1
                continue
            # Keep sample when score is unavailable unless explicitly requested.
            score = args.min_score

        if score < args.min_score:
            rejected["score"] += 1
            continue

        text = first_non_empty(item, text_fields)
        text = (text or "").strip()
        if len(text) < args.min_text_len:
            rejected["text"] += 1
            continue

        target_audio = first_non_empty(item, target_audio_fields)
        if not isinstance(target_audio, dict) or "array" not in target_audio:
            rejected["target_audio"] += 1
            continue

        ref_audio = first_non_empty(item, ref_audio_fields)
        if not isinstance(ref_audio, dict) or "array" not in ref_audio:
            rejected["ref_audio"] += 1
            continue

        uid = f"{len(rows):06d}"
        tgt_path = os.path.join(tgt_dir, f"tgt_{uid}.wav")
        ref_path = os.path.join(ref_dir, f"ref_{uid}.wav")

        try:
            save_wav(target_audio, tgt_path)
            save_wav(ref_audio, ref_path)
        except Exception as exc:
            print(f"Skip sample {idx}: audio save failed: {exc}")
            continue

        rows.append(
            {
                "text": text,
                "audio": tgt_path,
                "ref_audio": ref_path,
                "dataset_id": 0,
            }
        )

        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    random.shuffle(rows)
    split_idx = int(len(rows) * (1.0 - args.val_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    train_jsonl = os.path.join(out_dir, "train.jsonl")
    val_jsonl = os.path.join(out_dir, "val.jsonl")

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "total_input": len(ds),
        "kept": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "rejected": rejected,
        "text_fields": text_fields,
        "target_audio_fields": target_audio_fields,
        "ref_audio_fields": ref_audio_fields,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Done")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
