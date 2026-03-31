#!/usr/bin/env python3
"""
GPU-accelerated pipeline: Sparse Dataset Creation.
Limits redundancy by creating only 2 random reference pairs per target segment.
"""

import os, gc, time, shutil, sys, logging, random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torchaudio.transforms as TaT
from datasets import load_dataset, Dataset, Features, Value, Audio
from huggingface_hub import login, HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("SparseDataset")

# ─── CONFIG ────────────────────────────────────────────────────
TOKEN     = os.environ.get("HF_TOKEN", "")
SRC       = "amanuelbyte/acl-voice-cloning-fr-data"
DST       = "amanuelbyte/acl-voice-cloning-fr-sparse-v1"
WORK      = "/tmp/vc_sparse_shards"
REFS_PER_TARGET = 2 # 2 refs * ~740 source rows = ~1500 training rows

# Filter thresholds
MIN_DUR_SEC  = 1.0
MAX_DUR_SEC  = 20.0
MIN_SNR_DB   = 12.0 # Slightly stricter for sparse
MAX_SILENCE  = 0.60 # Slightly stricter
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ─── GPU FILTER FUNCTIONS ──────────────────────────────────────

def gpu_snr(wav_t: torch.Tensor) -> float:
    frame = int(16000 * 0.025)
    T = wav_t.shape[0]
    n_frames = T // frame
    if n_frames < 4: return 60.0
    frames = wav_t[:n_frames * frame].reshape(n_frames, frame)
    energies = frames.pow(2).mean(dim=1)
    sorted_e, _ = energies.sort()
    n_noise = max(1, n_frames // 10)
    noise_power = sorted_e[:n_noise].mean()
    signal_power = sorted_e.mean()
    if noise_power < 1e-12: return 60.0
    return float(10.0 * torch.log10(signal_power / noise_power.clamp(min=1e-12)))

def gpu_silence_ratio(wav_t: torch.Tensor) -> float:
    frame = int(16000 * 0.025)
    T = wav_t.shape[0]
    n_frames = T // frame
    if n_frames < 2: return 1.0
    THRESHOLD = 10 ** (-40.0 / 10)
    frames = wav_t[:n_frames * frame].reshape(n_frames, frame)
    energies = frames.pow(2).mean(dim=1)
    return float((energies < THRESHOLD).float().mean())

def resample_to_16k_gpu(arr: np.ndarray, sr: int) -> torch.Tensor:
    t = torch.from_numpy(arr).float().to(DEVICE)
    if sr != 16000:
        t = TaT.Resample(sr, 16000).to(DEVICE)(t.unsqueeze(0)).squeeze(0)
    return t

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

def make_gen_sparse(kept_indices, all_rows):
    """Creates only 1-2 random pairs per target segment."""
    def gen():
        # Group indices by speaker
        groups = {}
        for idx in kept_indices:
            sid = all_rows[idx]["speaker_id"]
            groups.setdefault(sid, []).append(idx)
        
        for sid, idxs in groups.items():
            for ti in idxs:
                # Potential references: all other segments of same speaker
                potential_refs = [ri for ri in idxs if ri != ti]
                if not potential_refs: continue
                
                # Pick N random references
                selected_refs = random.sample(potential_refs, min(REFS_PER_TARGET, len(potential_refs)))
                
                for ri in selected_refs:
                    r = all_rows[ri]
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

# ─── MAIN ──────────────────────────────────────────────────────
def filter_source(split_ds, split_name):
    total = len(split_ds)
    kept = []
    log.info(f"[{split_name}] Filtering {total} rows ...")
    for i in range(total):
        row = split_ds[i]
        text = (row.get("text_fr") or "").strip()
        audio = row.get("cloned_audio_fr")
        if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN): continue
        if not isinstance(audio, dict): continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if not (MIN_DUR_SEC <= len(arr)/sr <= MAX_DUR_SEC): continue
        
        wav_t = resample_to_16k_gpu(arr, sr)
        if gpu_snr(wav_t) < MIN_SNR_DB: continue
        if gpu_silence_ratio(wav_t) > MAX_SILENCE: continue
        kept.append(i)
    log.info(f"[{split_name}] Kept {len(kept)}/{total}")
    return kept

def main():
    if not TOKEN:
        log.error("HF_TOKEN missing!")
        return
    login(token=TOKEN)
    api = HfApi()
    create_repo(DST, repo_type="dataset", exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    ds = load_dataset(SRC)
    
    for split in ["train", "test"]:
        kept = filter_source(ds[split], split)
        sparse_ds = Dataset.from_generator(make_gen_sparse(kept, ds[split]), features=FEATURES)
        
        fname = f"{split}-sparse.parquet"
        local = os.path.join(WORK, fname)
        sparse_ds.to_parquet(local)
        
        log.info(f"Uploading {split} ({len(sparse_ds)} rows) ...")
        api.upload_file(path_or_fileobj=local, path_in_repo=f"data/{fname}", repo_id=DST, repo_type="dataset")
        
    # README
    readme = f"---\nconfigs:\n  - config_name: default\n    data_files:\n      - split: train\n        path: data/train-sparse.parquet\n      - split: test\n        path: data/test-sparse.parquet\n---\n# Sparse French Voice Cloning Dataset\nReduced redundancy (1 ref per target) for fast LoRA experimentation."
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=DST, repo_type="dataset")
    log.info(f"✅ Sparse dataset uploaded: https://huggingface.co/datasets/{DST}")

if __name__ == "__main__":
    main()
