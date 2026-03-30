#!/usr/bin/env python3
"""
GPU-Accelerated dataset filter & HuggingFace upload.

Uses batched GPU tensor operations for maximum speed on RTX 4090.
All SNR/silence/trim computations run on CUDA, not CPU.

Usage:
    python training/filter_and_upload.py
"""

import numpy as np
import torch
import torchaudio.transforms as TaT
import os, logging, sys
from datasets import load_dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("GPUFilter")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SRC_DATASET  = "amanuelbyte/acl-voice-cloning-fr-expandedtry"
DST_DATASET  = "amanuelbyte/acl-voice-cloning-fr-cleaned-v2"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# Filter thresholds — loosened for more data volume
MIN_DUR_SEC  = 1.0
MAX_DUR_SEC  = 20.0
MIN_SNR_DB   = 10.0    # was 15.0 — include moderately noisy audio
MAX_SILENCE  = 0.65    # was 0.50 — allow more pauses
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 500

# GPU batch size — higher = faster but more VRAM. 4090 can handle 512 easily.
GPU_BATCH    = 512
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Running on: {DEVICE}")
if DEVICE.type == "cuda":
    log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True


def gpu_snr_batch(wavs: torch.Tensor) -> torch.Tensor:
    """
    Compute SNR for a batch of fixed-length tensors on GPU.
    wavs: (B, T) float32 CUDA tensor
    Returns: (B,) float32 SNR in dB
    """
    frame = int(16000 * 0.025)  # 400 samples
    B, T = wavs.shape
    n_frames = T // frame
    if n_frames < 4:
        return torch.full((B,), 60.0, device=DEVICE)
    # (B, n_frames, frame)
    frames = wavs[:, :n_frames * frame].reshape(B, n_frames, frame)
    energies = frames.pow(2).mean(dim=2)  # (B, n_frames)
    sorted_e, _ = energies.sort(dim=1)
    n_noise = max(1, n_frames // 10)
    noise_power = sorted_e[:, :n_noise].mean(dim=1)
    signal_power = sorted_e.mean(dim=1)
    safe_noise = noise_power.clamp(min=1e-12)
    snr = 10.0 * torch.log10(signal_power / safe_noise)
    snr = torch.where(noise_power < 1e-12, torch.full_like(snr, 60.0), snr)
    return snr


def gpu_silence_batch(wavs: torch.Tensor) -> torch.Tensor:
    """
    Compute silence ratio for a batch on GPU.
    Returns: (B,) float32 in [0, 1]
    """
    frame = int(16000 * 0.025)
    B, T = wavs.shape
    n_frames = T // frame
    if n_frames < 2:
        return torch.ones(B, device=DEVICE)
    THRESHOLD = 10 ** (-40.0 / 10)
    frames = wavs[:, :n_frames * frame].reshape(B, n_frames, frame)
    energies = frames.pow(2).mean(dim=2)  # (B, n_frames)
    silent = (energies < THRESHOLD).float().mean(dim=1)
    return silent


def filter_batch_gpu(batch: dict) -> dict:
    """
    HuggingFace .map() callback: receives a batch dict, returns keep mask.
    All heavy audio math is done on GPU.
    """
    n = len(batch["trg_fr_text"])
    keep = [True] * n

    # ── 1. Text filters (CPU, trivial) ──
    for i in range(n):
        text = (batch["trg_fr_text"][i] or "").strip()
        if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
            keep[i] = False

    # ── 2. Audio filters (GPU) ──
    # Collect indices still alive after text filter
    alive_idx = [i for i in range(n) if keep[i]]
    if not alive_idx:
        batch["_keep"] = keep
        return batch

    # Pad audio arrays to the same length so we can stack into a GPU tensor
    arrays = []
    durations = []
    max_samples = int(MAX_DUR_SEC * 16000)

    for i in alive_idx:
        audio_field = batch["trg_fr_voice"][i]
        if not isinstance(audio_field, dict):
            keep[i] = False
            arrays.append(None)
            durations.append(0.0)
            continue
        arr = np.asarray(audio_field["array"], dtype=np.float32)
        sr  = audio_field["sampling_rate"]
        if sr != 16000:
            # Resample on GPU for speed
            t = torch.from_numpy(arr).to(DEVICE).unsqueeze(0)
            arr = TaT.Resample(sr, 16000).to(DEVICE)(t).squeeze(0).cpu().numpy()
        durations.append(len(arr) / 16000)
        # Truncate to max duration
        arr = arr[:max_samples]
        arrays.append(arr)

    # Duration filter (CPU, fast)
    for j, i in enumerate(alive_idx):
        if arrays[j] is None:
            continue
        if not (MIN_DUR_SEC <= durations[j] <= MAX_DUR_SEC):
            keep[i] = False
            arrays[j] = None

    # GPU filter: pad remaining arrays and batch them
    valid_j = [j for j in range(len(alive_idx)) if arrays[j] is not None]
    if valid_j:
        max_len = max(len(arrays[j]) for j in valid_j)
        padded = torch.zeros(len(valid_j), max_len, device=DEVICE)
        for k, j in enumerate(valid_j):
            arr = arrays[j]
            padded[k, :len(arr)] = torch.from_numpy(arr).to(DEVICE)

        snrs = gpu_snr_batch(padded)       # (K,)
        silences = gpu_silence_batch(padded)  # (K,)

        for k, j in enumerate(valid_j):
            i = alive_idx[j]
            if snrs[k].item() < MIN_SNR_DB:
                keep[i] = False
            elif silences[k].item() > MAX_SILENCE:
                keep[i] = False

    batch["_keep"] = keep
    return batch


def main():
    log.info(f"Loading source dataset: {SRC_DATASET}")
    ds = load_dataset(SRC_DATASET, token=HF_TOKEN)
    log.info(f"Splits: {list(ds.keys())}")

    cleaned = {}
    for split, data in ds.items():
        orig_len = len(data)
        log.info(f"[{split}] GPU-filtering {orig_len} samples in batches of {GPU_BATCH} ...")

        # Add keep column via GPU batched map (no multiprocessing — GPU can't fork)
        data = data.map(
            filter_batch_gpu,
            batched=True,
            batch_size=GPU_BATCH,
            desc=f"GPU filter [{split}]",
        )
        # Apply the mask and drop helper column
        filtered = data.filter(lambda x: x["_keep"])
        filtered = filtered.remove_columns(["_keep"])

        kept = len(filtered)
        log.info(f"[{split}] Kept {kept}/{orig_len} ({100*kept/orig_len:.1f}%)")
        cleaned[split] = filtered

    cleaned_ds = DatasetDict(cleaned)

    log.info(f"Pushing cleaned dataset to: {DST_DATASET}")
    cleaned_ds.push_to_hub(
        DST_DATASET,
        token=HF_TOKEN,
        max_shard_size="500MB",
    )
    log.info(f"✅ Done! https://huggingface.co/datasets/{DST_DATASET}")


if __name__ == "__main__":
    main()
