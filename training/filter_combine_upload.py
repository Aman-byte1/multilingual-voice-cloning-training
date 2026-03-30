#!/usr/bin/env python3
"""
GPU-accelerated pipeline: Filter source → Combine pairs → Upload to HF.

Filters the small 800-row source dataset on GPU, then creates all
(ref, target) pairs per speaker and uploads shards incrementally.

Usage:
    HF_TOKEN=hf_... python training/filter_combine_upload.py
"""

import os, gc, time, shutil, sys, logging
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torchaudio.transforms as TaT
from datasets import load_dataset, Dataset, Features, Value, Audio
from huggingface_hub import login, HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("FilterCombine")

# ─── CONFIG ────────────────────────────────────────────────────
TOKEN     = os.environ.get("HF_TOKEN", "")
SRC       = "amanuelbyte/acl-voice-cloning-fr-data"
DST       = "amanuelbyte/acl-voice-cloning-fr-cleaned-v2"
WORK      = "/tmp/vc_shards"
REF_BATCH = 20   # ref segments per shard

# Filter thresholds (loosened for more data)
MIN_DUR_SEC  = 1.0
MAX_DUR_SEC  = 20.0
MIN_SNR_DB   = 10.0
MAX_SILENCE  = 0.65
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 500

# ─── GPU SETUP ─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True

# ─── GPU FILTER FUNCTIONS ──────────────────────────────────────

def gpu_snr(wav_t: torch.Tensor) -> float:
    """SNR estimate for a single waveform on GPU."""
    frame = int(16000 * 0.025)
    T = wav_t.shape[0]
    n_frames = T // frame
    if n_frames < 4:
        return 60.0
    frames = wav_t[:n_frames * frame].reshape(n_frames, frame)
    energies = frames.pow(2).mean(dim=1)
    sorted_e, _ = energies.sort()
    n_noise = max(1, n_frames // 10)
    noise_power = sorted_e[:n_noise].mean()
    signal_power = sorted_e.mean()
    if noise_power < 1e-12:
        return 60.0
    return float(10.0 * torch.log10(signal_power / noise_power.clamp(min=1e-12)))


def gpu_silence_ratio(wav_t: torch.Tensor) -> float:
    """Silence ratio for a single waveform on GPU."""
    frame = int(16000 * 0.025)
    T = wav_t.shape[0]
    n_frames = T // frame
    if n_frames < 2:
        return 1.0
    THRESHOLD = 10 ** (-40.0 / 10)
    frames = wav_t[:n_frames * frame].reshape(n_frames, frame)
    energies = frames.pow(2).mean(dim=1)
    return float((energies < THRESHOLD).float().mean())


def resample_to_16k_gpu(arr: np.ndarray, sr: int) -> torch.Tensor:
    """Resample to 16kHz on GPU, returns CUDA tensor."""
    t = torch.from_numpy(arr).float().to(DEVICE)
    if sr != 16000:
        t = TaT.Resample(sr, 16000).to(DEVICE)(t.unsqueeze(0)).squeeze(0)
    return t


def passes_filter(audio_field: dict, text: str) -> bool:
    """Check if a single sample passes all quality filters. All audio ops on GPU."""
    # Text filter (instant)
    text = (text or "").strip()
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return False

    if not isinstance(audio_field, dict):
        return False

    arr = np.asarray(audio_field["array"], dtype=np.float32)
    sr = audio_field["sampling_rate"]

    # Duration filter
    dur = len(arr) / sr
    if not (MIN_DUR_SEC <= dur <= MAX_DUR_SEC):
        return False

    # GPU filters
    wav_t = resample_to_16k_gpu(arr, sr)

    if gpu_snr(wav_t) < MIN_SNR_DB:
        return False
    if gpu_silence_ratio(wav_t) > MAX_SILENCE:
        return False

    return True


# ─── OUTPUT SCHEMA ─────────────────────────────────────────────
FEATURES = Features({
    "speaker_id":    Value("string"),
    "speaker_name":  Value("string"),
    "ref_en_text":   Value("string"),
    "ref_fr_text":   Value("string"),
    "ref_en_voice":  Audio(),
    "ref_fr_voice":  Audio(),
    "trg_en_text":   Value("string"),
    "trg_fr_text":   Value("string"),
    "trg_en_voice":  Audio(),
    "trg_fr_voice":  Audio(),
})

# ─── HELPERS ───────────────────────────────────────────────────
def clean_cache():
    for d in ["generator", "parquet"]:
        p = Path.home() / ".cache" / "huggingface" / "datasets" / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    gc.collect()


def make_gen(ref_indices, all_rows, n_total):
    def gen():
        for ri in ref_indices:
            r = all_rows[ri]
            for ti in range(n_total):
                if ri == ti:
                    continue
                t = all_rows[ti]
                yield {
                    "speaker_id":   r["speaker_id"],
                    "speaker_name": r["speaker_name"],
                    "ref_en_text":  r["text_en"],
                    "ref_fr_text":  r["text_fr"],
                    "ref_en_voice": r["audio_en"],
                    "ref_fr_voice": r["cloned_audio_fr"],
                    "trg_en_text":  t["text_en"],
                    "trg_fr_text":  t["text_fr"],
                    "trg_en_voice": t["audio_en"],
                    "trg_fr_voice": t["cloned_audio_fr"],
                }
    return gen


# ─── STEP 1: FILTER SOURCE DATASET ON GPU ─────────────────────
def filter_source(split_ds, split_name):
    """Filter rows using GPU-accelerated quality checks. Returns kept indices."""
    total = len(split_ds)
    kept_indices = []
    rejected_reasons = {"text": 0, "audio": 0, "duration": 0, "snr": 0, "silence": 0}

    log.info(f"[{split_name}] GPU-filtering {total} source rows ...")

    for i in range(total):
        row = split_ds[i]
        text = (row.get("text_fr") or "").strip()
        audio = row.get("cloned_audio_fr")

        if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
            rejected_reasons["text"] += 1
            continue

        if not isinstance(audio, dict):
            rejected_reasons["audio"] += 1
            continue

        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        dur = len(arr) / sr

        if not (MIN_DUR_SEC <= dur <= MAX_DUR_SEC):
            rejected_reasons["duration"] += 1
            continue

        # GPU-accelerated quality checks
        wav_t = resample_to_16k_gpu(arr, sr)

        if gpu_snr(wav_t) < MIN_SNR_DB:
            rejected_reasons["snr"] += 1
            continue

        if gpu_silence_ratio(wav_t) > MAX_SILENCE:
            rejected_reasons["silence"] += 1
            continue

        kept_indices.append(i)

    kept = len(kept_indices)
    log.info(f"[{split_name}] Kept {kept}/{total} ({100*kept/total:.1f}%)")
    log.info(f"  Rejected: {rejected_reasons}")
    return kept_indices


# ─── STEP 2: COMBINE PAIRS & UPLOAD ───────────────────────────
def combine_and_upload(split_ds, kept_indices, split_name, api):
    """Create all (ref, target) pairs from kept rows and upload shards."""
    # Group by speaker
    groups = OrderedDict()
    for i in kept_indices:
        sid = split_ds[i]["speaker_id"]
        groups.setdefault(sid, []).append(i)

    total_pairs = 0
    for sid, idxs in groups.items():
        n = len(idxs)
        p = n * (n - 1)
        total_pairs += p
        sname = split_ds[idxs[0]]["speaker_name"]
        log.info(f"    {sid} ({sname:>12s})  {n:>4d} segs → {p:>7,} pairs")
    log.info(f"    {'TOTAL':>38s}  {total_pairs:>7,}")

    shard_num = 0
    uploaded_rows = 0

    for sid, idxs in groups.items():
        n = len(idxs)
        t_spk = time.time()
        log.info(f"  🔧  {sid}  ({n} segs → {n*(n-1):,} pairs)")

        # Pre-fetch all rows for this speaker
        speaker_rows = [split_ds[i] for i in idxs]

        for b_start in range(0, n, REF_BATCH):
            b_end = min(b_start + REF_BATCH, n)
            ref_sl = list(range(b_start, b_end))
            n_pairs = len(ref_sl) * (n - 1)

            clean_cache()
            batch_ds = Dataset.from_generator(
                make_gen(ref_sl, speaker_rows, n),
                features=FEATURES,
            )

            fname = f"{split_name}-{shard_num:05d}.parquet"
            local = os.path.join(WORK, fname)
            batch_ds.to_parquet(local)
            fsize = os.path.getsize(local) / 1e6

            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=f"data/{fname}",
                repo_id=DST,
                repo_type="dataset",
            )

            os.remove(local)
            del batch_ds
            clean_cache()

            uploaded_rows += n_pairs
            shard_num += 1
            log.info(f"      shard {shard_num:>3d}  ({n_pairs:>5,} rows, {fsize:>7.1f} MB)  ✅")

        del speaker_rows
        gc.collect()
        log.info(f"  ✅  {sid} done  ({time.time()-t_spk:.0f}s)")

    return uploaded_rows, shard_num


# ─── MAIN ──────────────────────────────────────────────────────
def main():
    login(token=TOKEN)
    api = HfApi()
    create_repo(DST, repo_type="dataset", exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    t0 = time.time()
    log.info(f"Loading source: {SRC}")
    ds = load_dataset(SRC)
    log.info(f"  train: {len(ds['train'])}  |  test: {len(ds['test'])}")

    # ── TRAIN ──
    log.info("=" * 55 + "  TRAIN")
    train_kept = filter_source(ds["train"], "train")
    train_rows, train_shards = combine_and_upload(ds["train"], train_kept, "train", api)

    # ── TEST ──
    log.info("=" * 55 + "  TEST")
    test_kept = filter_source(ds["test"], "test")
    test_rows, test_shards = combine_and_upload(ds["test"], test_kept, "test", api)

    # ── README ──
    readme = f"""---
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: test
        path: data/test-*.parquet
---

# ACL Voice Cloning FR — Cleaned & Expanded (V2)

Source filtered with GPU-accelerated quality checks (SNR≥{MIN_SNR_DB}dB, silence≤{int(MAX_SILENCE*100)}%),
then expanded into all (reference, target) pairs per speaker.

| Filter | Threshold |
|---|---|
| Duration | {MIN_DUR_SEC}-{MAX_DUR_SEC}s |
| SNR | ≥ {MIN_SNR_DB} dB |
| Silence | ≤ {int(MAX_SILENCE*100)}% |
| Text length | {MIN_TEXT_LEN}-{MAX_TEXT_LEN} chars |

| Split | Source kept | Expanded pairs | Shards |
|---|---|---|---|
| train | {len(train_kept)} | {train_rows:,} | {train_shards} |
| test  | {len(test_kept)} | {test_rows:,} | {test_shards} |
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=DST,
        repo_type="dataset",
    )

    elapsed = time.time() - t0
    log.info(f"✅ ALL DONE ({elapsed/60:.1f} min)")
    log.info(f"  https://huggingface.co/datasets/{DST}")
    log.info(f"  train: {train_rows:,}  |  test: {test_rows:,}")


if __name__ == "__main__":
    main()
