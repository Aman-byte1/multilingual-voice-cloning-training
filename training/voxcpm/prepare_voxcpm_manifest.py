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
from datasets import Audio, load_dataset
from tqdm import tqdm


def first_non_empty(row: dict, keys: Iterable[str]):
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return None


def _looks_like_audio(v) -> bool:
    if isinstance(v, dict):
        if "array" in v and "sampling_rate" in v:
            return True
        # Path-only dicts from HF can be unresolved ids (e.g. "fr_00001").
        # Treat as audio only if the path is actually resolvable on disk.
        if "path" in v and v["path"]:
            p = str(v["path"])
            if os.path.isabs(p) and os.path.exists(p):
                return True
    if isinstance(v, str) and v:
        return os.path.isabs(v) and os.path.exists(v)
    return False


def pick_audio_field(row: dict, candidates: list[str], role: str):
    # 1) Try explicit user-provided candidates first.
    for k in candidates:
        if _looks_like_audio(row.get(k)):
            return row.get(k), k

    # 2) Fallback heuristic scan for unknown schemas.
    role_tokens = ["ref", "prompt", "source", "src"] if role == "ref" else ["trg", "target", "tts", "gen"]
    for k, v in row.items():
        lk = str(k).lower()
        if not _looks_like_audio(v):
            continue
        if "audio" in lk or "voice" in lk or lk.endswith("_wav"):
            if role == "target" and any(tok in lk for tok in role_tokens):
                return v, k
            if role == "ref" and any(tok in lk for tok in role_tokens):
                return v, k

    # 3) Last resort: any audio-like field.
    for k, v in row.items():
        if _looks_like_audio(v):
            return v, k

    return None, None


def maybe_cast_audio_columns(ds, columns: list[str]):
    casted = []
    for col in columns:
        if col not in ds.column_names:
            continue
        try:
            ds = ds.cast_column(col, Audio())
            casted.append(col)
        except Exception:
            pass
    return ds, casted


def save_wav(audio_field, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if isinstance(audio_field, dict) and "array" in audio_field and "sampling_rate" in audio_field:
        arr = np.asarray(audio_field["array"], dtype=np.float32)
        sr = int(audio_field["sampling_rate"])
        wav = torch.from_numpy(arr).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(out_path, wav, sample_rate=sr)
        return

    src_path = None
    if isinstance(audio_field, dict) and audio_field.get("path"):
        src_path = audio_field["path"]
    elif isinstance(audio_field, str):
        src_path = audio_field

    if src_path is None:
        raise ValueError(f"Unsupported audio field format: {type(audio_field)}")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Audio path does not exist: {src_path}")

    wav, sr = torchaudio.load(src_path)
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
    parser.add_argument("--language-field", default="", help="Optional language column name")
    parser.add_argument("--language-value", default="", help="Optional language value to keep (e.g. zh)")
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

    # Decode likely audio columns up front to materialize {array, sampling_rate}
    # and avoid unresolved path ids (e.g. "fr_00001").
    candidate_audio_cols = list(dict.fromkeys(target_audio_fields + ref_audio_fields))
    ds, casted_cols = maybe_cast_audio_columns(ds, candidate_audio_cols)

    out_dir = os.path.abspath(args.output_dir)
    tgt_dir = os.path.join(out_dir, "target_wavs")
    ref_dir = os.path.join(out_dir, "ref_wavs")
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    rows = []
    chosen_target_fields = {}
    chosen_ref_fields = {}
    rejected = {
        "language": 0,
        "text": 0,
        "target_audio": 0,
        "ref_audio": 0,
        "score": 0,
        "missing_score": 0,
    }

    for idx in tqdm(range(len(ds)), desc="Preparing"):
        item = ds[idx]

        if args.language_field and args.language_value:
            lang = str(item.get(args.language_field, "")).strip().lower()
            if lang != args.language_value.strip().lower():
                rejected["language"] += 1
                continue

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

        target_audio, target_key = pick_audio_field(item, target_audio_fields, role="target")
        if target_audio is None:
            rejected["target_audio"] += 1
            continue
        chosen_target_fields[target_key or "<unknown>"] = chosen_target_fields.get(target_key or "<unknown>", 0) + 1

        ref_audio, ref_key = pick_audio_field(item, ref_audio_fields, role="ref")
        if ref_audio is None:
            rejected["ref_audio"] += 1
            continue
        chosen_ref_fields[ref_key or "<unknown>"] = chosen_ref_fields.get(ref_key or "<unknown>", 0) + 1

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
        "language_field": args.language_field,
        "language_value": args.language_value,
        "total_input": len(ds),
        "kept": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "rejected": rejected,
        "text_fields": text_fields,
        "target_audio_fields": target_audio_fields,
        "ref_audio_fields": ref_audio_fields,
        "chosen_target_fields": chosen_target_fields,
        "chosen_ref_fields": chosen_ref_fields,
        "casted_audio_columns": casted_cols,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Done")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
