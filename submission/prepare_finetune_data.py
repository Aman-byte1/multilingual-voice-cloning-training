#!/usr/bin/env python3
"""
🔧 IWSLT 2026 — Fine-Tuning Data Preparation
===============================================
Prepares ACL 60/60 dataset for model fine-tuning on voice cloning.
Based on the sparse dataset pipeline from create_sparse_dataset.py.

This creates training data in a model-agnostic format that can be used
for fine-tuning Qwen3-TTS, CosyVoice3, or other models.

Output format:
    - WAV files organized by speaker
    - Metadata JSON with text, language, speaker ID, durations
    - Train/val/test splits

Usage:
    python prepare_finetune_data.py --dataset amanuelbyte/acl-voice-cloning-fr-data \
                                     --output-dir ./finetune_data \
                                     --languages fr
"""

import os
import sys
import gc
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as TaT
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("DataPrep")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── QUALITY FILTERS (from create_sparse_dataset.py) ──────────
MIN_DUR_SEC = 1.0
MAX_DUR_SEC = 20.0
MIN_SNR_DB = 12.0
MAX_SILENCE = 0.60
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 500
TARGET_SR = 16000


def gpu_snr(wav_t: torch.Tensor) -> float:
    frame = int(TARGET_SR * 0.025)
    T = wav_t.shape[0]
    n_frames = T // frame
    if n_frames < 4:
        return 60.0
    frames = wav_t[: n_frames * frame].reshape(n_frames, frame)
    energies = frames.pow(2).mean(dim=1)
    sorted_e, _ = energies.sort()
    n_noise = max(1, n_frames // 10)
    noise_power = sorted_e[:n_noise].mean()
    signal_power = sorted_e.mean()
    if noise_power < 1e-12:
        return 60.0
    return float(10.0 * torch.log10(signal_power / noise_power.clamp(min=1e-12)))


def gpu_silence_ratio(wav_t: torch.Tensor) -> float:
    frame = int(TARGET_SR * 0.025)
    T = wav_t.shape[0]
    n_frames = T // frame
    if n_frames < 2:
        return 1.0
    THRESHOLD = 10 ** (-40.0 / 10)
    frames = wav_t[: n_frames * frame].reshape(n_frames, frame)
    energies = frames.pow(2).mean(dim=1)
    return float((energies < THRESHOLD).float().mean())


def resample_to_target(arr: np.ndarray, sr: int) -> torch.Tensor:
    t = torch.from_numpy(arr).float().to(DEVICE)
    if sr != TARGET_SR:
        t = TaT.Resample(sr, TARGET_SR).to(DEVICE)(t.unsqueeze(0)).squeeze(0)
    return t


def passes_quality_filter(wav_t: torch.Tensor, text: str) -> bool:
    """Apply quality filters from create_sparse_dataset.py."""
    dur_sec = wav_t.shape[0] / TARGET_SR
    if not (MIN_DUR_SEC <= dur_sec <= MAX_DUR_SEC):
        return False
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return False
    if gpu_snr(wav_t) < MIN_SNR_DB:
        return False
    if gpu_silence_ratio(wav_t) > MAX_SILENCE:
        return False
    return True


def save_wav(wav_t: torch.Tensor, sr: int, path: str):
    """Save tensor to WAV file."""
    wav_cpu = wav_t.cpu()
    if wav_cpu.dim() == 1:
        wav_cpu = wav_cpu.unsqueeze(0)
    torchaudio.save(path, wav_cpu, sr)


def process_dataset(
    dataset_name: str,
    output_dir: str,
    max_per_speaker: int = 100,
    refs_per_target: int = 2,
    split_ratios: tuple = (0.8, 0.1, 0.1),
):
    """Process dataset into fine-tuning format.

    Creates:
        output_dir/
        ├── wavs/
        │   ├── speaker_001/
        │   │   ├── ref_000.wav      (EN reference)
        │   │   ├── target_000.wav   (FR target)
        │   │   └── ...
        │   └── speaker_002/
        │       └── ...
        ├── metadata_train.json
        ├── metadata_val.json
        └── metadata_test.json
    """
    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # Process each split
    for split_name in ["train", "test"]:
        if split_name not in ds:
            log.warning(f"Split '{split_name}' not found, skipping")
            continue
            
        split_ds = ds[split_name]
        log.info(f"\nProcessing split: {split_name} ({len(split_ds)} rows)")

        # Group by speaker
        speaker_data = defaultdict(list)
        filtered_out = 0

        for i in tqdm(range(len(split_ds)), desc=f"Filtering {split_name}"):
            row = split_ds[i]
            speaker_id = row.get("speaker_id", f"spk_{i:04d}")
            text_fr = (row.get("text_fr") or row.get("trg_fr_text") or "").strip()
            text_en = (row.get("text_en") or row.get("ref_en_text") or "").strip()

            # Get audio data
            ref_audio = row.get("audio_en") or row.get("ref_en_voice")
            target_audio = row.get("cloned_audio_fr") or row.get("trg_fr_voice")

            if not text_fr or not ref_audio or not target_audio:
                filtered_out += 1
                continue

            # Quality filter on target audio
            arr = np.asarray(target_audio["array"], dtype=np.float32)
            sr = target_audio["sampling_rate"]
            wav_t = resample_to_target(arr, sr)

            if not passes_quality_filter(wav_t, text_fr):
                filtered_out += 1
                continue

            speaker_data[speaker_id].append({
                "index": i,
                "text_fr": text_fr,
                "text_en": text_en,
                "ref_audio": ref_audio,
                "target_audio": target_audio,
            })

        log.info(f"Kept: {sum(len(v) for v in speaker_data.values())} | "
                 f"Filtered: {filtered_out} | Speakers: {len(speaker_data)}")

        # Create pairs and save WAVs
        metadata = []
        pair_idx = 0

        for spk_id, entries in tqdm(speaker_data.items(), desc="Creating pairs"):
            spk_dir = os.path.join(wav_dir, str(spk_id))
            os.makedirs(spk_dir, exist_ok=True)

            # Limit per speaker
            if max_per_speaker and len(entries) > max_per_speaker:
                entries = random.sample(entries, max_per_speaker)

            for target_entry in entries:
                # Get reference entries (different from target)
                potential_refs = [e for e in entries if e["index"] != target_entry["index"]]
                if not potential_refs:
                    continue

                selected_refs = random.sample(
                    potential_refs, min(refs_per_target, len(potential_refs))
                )

                for ref_entry in selected_refs:
                    # Save reference WAV
                    ref_arr = np.asarray(ref_entry["ref_audio"]["array"], dtype=np.float32)
                    ref_sr = ref_entry["ref_audio"]["sampling_rate"]
                    ref_wav = resample_to_target(ref_arr, ref_sr)
                    ref_path = os.path.join(spk_dir, f"ref_{pair_idx:05d}.wav")
                    save_wav(ref_wav, TARGET_SR, ref_path)

                    # Save target WAV
                    tgt_arr = np.asarray(target_entry["target_audio"]["array"], dtype=np.float32)
                    tgt_sr = target_entry["target_audio"]["sampling_rate"]
                    tgt_wav = resample_to_target(tgt_arr, tgt_sr)
                    tgt_path = os.path.join(spk_dir, f"target_{pair_idx:05d}.wav")
                    save_wav(tgt_wav, TARGET_SR, tgt_path)

                    metadata.append({
                        "id": pair_idx,
                        "speaker_id": str(spk_id),
                        "ref_audio_path": os.path.relpath(ref_path, output_dir),
                        "target_audio_path": os.path.relpath(tgt_path, output_dir),
                        "ref_text_en": ref_entry["text_en"],
                        "target_text_fr": target_entry["text_fr"],
                        "ref_duration_sec": ref_wav.shape[0] / TARGET_SR,
                        "target_duration_sec": tgt_wav.shape[0] / TARGET_SR,
                    })
                    pair_idx += 1

        log.info(f"Created {len(metadata)} training pairs from {split_name}")

        # Split into train/val/test
        random.shuffle(metadata)
        n = len(metadata)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        splits = {
            f"{split_name}_train": metadata[:n_train],
            f"{split_name}_val": metadata[n_train:n_train + n_val],
            f"{split_name}_test": metadata[n_train + n_val:],
        }

        for split_key, split_data in splits.items():
            meta_path = os.path.join(output_dir, f"metadata_{split_key}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            log.info(f"  {split_key}: {len(split_data)} pairs → {meta_path}")

    # Summary stats
    total_wavs = sum(1 for _ in Path(wav_dir).rglob("*.wav"))
    total_size_mb = sum(f.stat().st_size for f in Path(wav_dir).rglob("*.wav")) / (1024*1024)
    log.info(f"\n{'='*60}")
    log.info(f"  DATA PREPARATION COMPLETE")
    log.info(f"  Total WAVs: {total_wavs}")
    log.info(f"  Total size: {total_size_mb:.1f} MB")
    log.info(f"  Output: {output_dir}")
    log.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="IWSLT 2026 — Fine-Tuning Data Preparation"
    )
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-data")
    parser.add_argument("--output-dir", default="./finetune_data")
    parser.add_argument("--max-per-speaker", type=int, default=100)
    parser.add_argument("--refs-per-target", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    process_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_per_speaker=args.max_per_speaker,
        refs_per_target=args.refs_per_target,
    )


if __name__ == "__main__":
    main()
