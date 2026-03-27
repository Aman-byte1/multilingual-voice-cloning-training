#!/usr/bin/env python3
"""
Fine-tune Chatterbox Multilingual TTS for French Voice Cloning
==============================================================
LoRA on the T3 text→semantic transformer.

Key fixes vs previous version:
  - Warmup scaled to actual training length (ratio, not absolute)
  - LoRA rank reduced, targets reduced (attention-only)
  - Save/load format fixed (proper state_dict keys)
  - Full dataset used (no 20% sampling)
  - Data quality filtering added
  - Double resampling eliminated (16kHz throughout)
  - Periodic actions scaled to actual step count
  - Early stopping with patience
"""

import os
import sys
import gc
import csv
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# amp imports handled via torch.amp directly
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_chatterbox_fr.log"),
    ],
)
logger = logging.getLogger("ChatterboxLoRA")

SEP = '\u2550' * 50
STAR = '\u2605'


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """All hyperparameters and paths for fine-tuning."""

    # ── Dataset ──
    dataset_name: str = "amanuelbyte/acl-voice-cloning-fr-cleaned"
    dataset_split: str = "train"
    sample_fraction: float = 1.0
    cache_dir: str = "./data_cache"
    audio_data_dir: str = "./audio_data"

    # ── Audio ──
    # Store everything at 16kHz (VE & S3 tokenizer native rate)
    # Eliminates double resampling: was 16k→24k→16k
    native_sr: int = 16000
    max_audio_duration_sec: float = 20.0
    min_audio_duration_sec: float = 1.0  # was 0.5, too short
    ref_audio_max_duration_sec: float = 15.0  # was 10, longer = better

    # ── Data filtering ──
    filter_min_snr_db: float = 15.0
    filter_max_silence_ratio: float = 0.5
    filter_min_text_len: int = 5  # was 2
    filter_max_text_len: int = 500  # was 1000

    # ── Training ──
    output_dir: str = "./chatterbox_fr_finetuned"
    num_epochs: int = 10       # more passes on smaller clean dataset
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5  # higher LR: data is clean, 7.7k samples
    min_learning_rate: float = 3e-6
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03   # less warmup on smaller dataset
    fp16: bool = True
    compile_model: bool = True
    seed: int = 42
    weight_decay: float = 0.01

    # LoRA config — rank 16 for 8-speaker coverage
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05   # less dropout: clean data, less regularization
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ── Freezing ──
    freeze_ve: bool = True
    freeze_s3gen: bool = True

    # ── Validation / checkpointing ──
    val_split_ratio: float = 0.10
    eval_every_ratio: float = 0.10
    save_every_ratio: float = 0.20
    log_every_n_steps: int = 5
    test_every_ratio: float = 0.25
    num_test_samples: int = 3
    patience: int = 8   # more tolerance on smaller dataset

    # ── Language ──
    language_id: str = "fr"
    cloning_mode: str = "cross_lingual"

    # ── HuggingFace upload ──
    hf_token: str = ""
    hf_username: str = "amanuelbyte"
    hf_repo_name: str = "chatterbox-fr-lora"
    hf_repo_id: Optional[str] = None # Auto-generated if None
    skip_dataset_upload: bool = False

    def __post_init__(self):
        for d in [
            self.output_dir,
            self.cache_dir,
            os.path.join(self.output_dir, "checkpoints"),
            os.path.join(self.output_dir, "samples"),
            os.path.join(self.audio_data_dir, "audio"),
            os.path.join(self.audio_data_dir, "ref_audio"),
        ]:
            os.makedirs(d, exist_ok=True)


# ============================================================================
# LoRA IMPLEMENTATION
# ============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer wrapping nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f, out_f = original.in_features, original.out_features
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original_layer(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora


def inject_lora(model: nn.Module, targets: List[str], rank: int,
                alpha: float, dropout: float = 0.0) -> List[str]:
    """Inject LoRA layers into matching nn.Linear modules.
    Returns list of injected module names (for save/load matching)."""
    injected_names = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name for target in targets):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        lora = LoRALayer(module, rank, alpha, dropout)
        setattr(parent, parts[-1], lora)
        injected_names.append(name)
    logger.info(f"Injected LoRA into {len(injected_names)} layers (rank={rank}, alpha={alpha})")
    for n in injected_names:
        logger.info(f"  LoRA → {n}")
    return injected_names


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA parameters using PROPER state_dict key names.
    Keys match model.state_dict() format so load_state_dict works."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    return state


def save_lora(model: nn.Module, path: str, config: dict = None):
    """Save LoRA weights with proper state_dict keys + config metadata."""
    payload = {
        "lora_state_dict": get_lora_state_dict(model),
        "config": config or {},
    }
    torch.save(payload, path)
    n_params = sum(v.numel() for v in payload["lora_state_dict"].values())
    logger.info(f"LoRA saved → {path} ({n_params:,} params)")


def load_lora(model: nn.Module, path: str, device: str = "cuda"):
    """Load LoRA weights using proper state_dict keys."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if "lora_state_dict" in payload:
        lora_sd = payload["lora_state_dict"]
    else:
        lora_sd = payload
    current_sd = model.state_dict()
    loaded = 0
    for key, value in lora_sd.items():
        if key in current_sd:
            current_sd[key] = value.to(device)
            loaded += 1
        else:
            logger.warning(f"  Key not found in model: {key}")
    model.load_state_dict(current_sd, strict=False)
    logger.info(f"LoRA loaded ← {path} ({loaded} tensors)")
    return payload.get("config", {})


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all trainable LoRA parameters by walking model modules."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            params.extend([module.lora_A, module.lora_B])
    return params


def merge_lora(model: nn.Module):
    """Merge LoRA weights into base linear layers permanently."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            with torch.no_grad():
                module.original_layer.weight.data += (
                    module.lora_B @ module.lora_A) * module.scaling
            count += 1
    logger.info(f"Merged {count} LoRA layers into base model")


# ============================================================================
# DATA FILTERING
# ============================================================================

def compute_audio_snr(audio: np.ndarray, sr: int, frame_len: float = 0.025) -> float:
    """Estimate SNR — fully vectorized with numpy reshape."""
    frame_samples = int(sr * frame_len)
    n_frames = len(audio) // frame_samples
    if n_frames < 4:
        return 0.0
    # Reshape into (n_frames, frame_samples) and compute energy per frame
    frames = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.mean(frames ** 2, axis=1)
    sorted_e = np.sort(energies)
    n_noise = max(1, len(sorted_e) // 10)
    noise_power = np.mean(sorted_e[:n_noise])
    signal_power = np.mean(sorted_e)
    if noise_power < 1e-12:
        return 60.0
    return float(10 * np.log10(signal_power / noise_power))


def compute_silence_ratio(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> float:
    """Fraction of frames below energy threshold — vectorized."""
    frame_len = int(sr * 0.025)
    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return 1.0
    threshold_energy = 10 ** (threshold_db / 10)
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)
    return float(np.mean(energies < threshold_energy))


def trim_trailing_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    """Trim silence from the END of audio — vectorized energy scan."""
    frame_len = int(sr * 0.01)  # 10ms frames
    padding_len = int(sr * 0.2) # 200ms safety padding
    threshold_energy = 10 ** (threshold_db / 10)
    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return audio
    # Vectorized: compute energy of all frames at once
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)
    # Find last frame above threshold (searching from the end)
    above = np.where(energies > threshold_energy)[0]
    if len(above) == 0:
        return audio
    last_active_frame = above[-1]
    last_idx = min(len(audio), (last_active_frame + 1) * frame_len + padding_len)
    if last_idx < sr * 0.5:  # Don't trim to less than 0.5s
        return audio
    return audio[:last_idx]


def filter_sample(audio: np.ndarray, sr: int, text: str,
                  config: TrainingConfig) -> Tuple[bool, str]:
    """Returns (keep, reason). True = keep the sample."""
    if not text or len(text.strip()) < config.filter_min_text_len:
        return False, "text_too_short"
    if len(text.strip()) > config.filter_max_text_len:
        return False, "text_too_long"
    dur = len(audio) / sr
    if dur < config.min_audio_duration_sec:
        return False, f"too_short_{dur:.1f}s"
    if dur > config.max_audio_duration_sec:
        return False, f"too_long_{dur:.1f}s"
    snr = compute_audio_snr(audio, sr)
    if snr < config.filter_min_snr_db:
        return False, f"low_snr_{snr:.1f}dB"
    silence = compute_silence_ratio(audio, sr)
    if silence > config.filter_max_silence_ratio:
        return False, f"too_silent_{silence:.1%}"
    return True, "ok"


# ============================================================================
# DATA PREPARATION  — HuggingFace → WAV files + metadata.csv
# ============================================================================

def prepare_dataset(config: TrainingConfig) -> List[Dict]:
    """Download dataset, filter for quality, extract audio at 16kHz."""
    logger.info(f"Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name, split=config.dataset_split,
                      cache_dir=config.cache_dir)
    logger.info(f"Full dataset size: {len(ds)} rows")

    # ── Stratified sampling ──
    speaker_ids = ds["speaker_id"]
    by_speaker: Dict[str, List[int]] = {}
    for idx, sid in enumerate(speaker_ids):
        by_speaker.setdefault(sid, []).append(idx)

    rng = np.random.RandomState(config.seed)
    sampled_indices = []
    for speaker, indices in sorted(by_speaker.items()):
        n = max(1, int(len(indices) * config.sample_fraction))
        chosen = rng.choice(indices, size=n, replace=False).tolist()
        sampled_indices.extend(chosen)
        logger.info(f"  {speaker}: {len(indices)} total -> {n} sampled")

    sampled_indices.sort()
    ds_sampled = ds.select(sampled_indices)
    logger.info(f"Sampled dataset: {len(ds_sampled)} rows")

    audio_dir = os.path.join(config.audio_data_dir, "audio")
    ref_dir = os.path.join(config.audio_data_dir, "ref_audio")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    rows = []
    filter_stats = {
        "total": len(ds_sampled),
        "kept": 0,
        "ref_types": {"en": 0, "fr": 0, "none": 0},
        "reasons": {}
    }
    resampler_cache: Dict[int, T.Resample] = {}

    def resample_to_16k(array: np.ndarray, sr: int) -> np.ndarray:
        if sr == 16000:
            return array
        if sr not in resampler_cache:
            resampler_cache[sr] = T.Resample(sr, 16000)
        wav_t = torch.from_numpy(array).unsqueeze(0)
        return resampler_cache[sr](wav_t).squeeze(0).numpy()

    for i in tqdm(range(len(ds_sampled)), desc="Processing"):
        row = ds_sampled[i]
        filter_stats["total"] += 1

        text = (row.get("trg_fr_text") or "").strip()
        speaker_id = row.get("speaker_id", "unknown")

        trg = row.get("trg_fr_voice")
        if trg is None or not isinstance(trg, dict):
            filter_stats["reasons"]["no_audio"] = filter_stats["reasons"].get("no_audio", 0) + 1
            continue

        trg_array = np.asarray(trg["array"], dtype=np.float32)
        sr = trg["sampling_rate"]
        trg_16k = resample_to_16k(trg_array, sr)

        # Trim trailing silence
        trg_16k = trim_trailing_silence(trg_16k, 16000)

        # Quality filter
        keep, reason = filter_sample(trg_16k, 16000, text, config)
        if not keep:
            filter_stats["reasons"][reason] = filter_stats["reasons"].get(reason, 0) + 1
            continue

        # Save target audio at 16kHz (soundfile is faster than torchaudio for raw numpy)
        trg_file = f"{speaker_id}_{i:06d}.wav"
        trg_path = os.path.join(audio_dir, trg_file)
        
        # Robustness check: NaN/Inf (pure numpy, no torch overhead)
        if np.any(np.isnan(trg_16k)) or np.any(np.isinf(trg_16k)):
            filter_stats["reasons"]["nan_or_inf"] = filter_stats["reasons"].get("nan_or_inf", 0) + 1
            continue

        try:
            sf.write(trg_path, trg_16k, 16000)
        except Exception as e:
            logger.error(f"Failed to save {trg_path}: {e}")
            filter_stats["reasons"]["write_error"] = filter_stats["reasons"].get("write_error", 0) + 1
            continue

        # Reference audio
        ref_file = ""
        actual_ref_type = "none"
        ref_key = "ref_en_voice" if config.cloning_mode == "cross_lingual" else "ref_fr_voice"
        alt_key = "ref_fr_voice" if config.cloning_mode == "cross_lingual" else "ref_en_voice"
        ref_data = row.get(ref_key)
        if ref_data is not None and isinstance(ref_data, dict):
            actual_ref_type = "en" if "en" in ref_key else "fr"
        else:
            ref_data = row.get(alt_key)
            if ref_data is not None and isinstance(ref_data, dict):
                actual_ref_type = "fr" if "fr" in alt_key else "en"

        if ref_data is not None and isinstance(ref_data, dict):
            filter_stats["ref_types"][actual_ref_type] += 1
            ref_array = np.asarray(ref_data["array"], dtype=np.float32)
            ref_sr = ref_data["sampling_rate"]
            ref_16k = resample_to_16k(ref_array, ref_sr)
            max_ref = int(config.ref_audio_max_duration_sec * 16000)
            ref_16k = ref_16k[:max_ref]
            ref_file = f"ref_{speaker_id}_{i:06d}.wav"
            ref_path = os.path.join(ref_dir, ref_file)
            
            if not (np.any(np.isnan(ref_16k)) or np.any(np.isinf(ref_16k))):
                try:
                    sf.write(ref_path, ref_16k, 16000)
                except Exception as e:
                    logger.warning(f"Failed to save reference {ref_path}: {e}")
                    ref_file = ""
            else:
                ref_file = ""
        else:
            filter_stats["ref_types"]["none"] += 1

        rows.append({
            "file_name": f"audio/{trg_file}",
            "ref_file": f"ref_audio/{ref_file}" if ref_file else "",
            "transcription": text,
            "duration_seconds": round(len(trg_16k) / 16000, 3),
            "speaker_id": speaker_id,
            "language_id": config.language_id,
        })

        if i % 5000 == 0:
            gc.collect()

    # Write metadata.csv
    if not rows:
        logger.error("No samples passed filtering!")
        return rows
    csv_path = os.path.join(config.audio_data_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Save filter report
    filter_path = os.path.join(config.audio_data_dir, "filter_report.json")
    with open(filter_path, "w") as f:
        json.dump(filter_stats, f, indent=2)

    logger.info(f"Data filtering: {filter_stats['kept']}/{filter_stats['total']} kept")
    ref_en = filter_stats["ref_types"].get("en", 0)
    ref_fr = filter_stats["ref_types"].get("fr", 0)
    logger.info(f"  Ref source: {ref_en} EN (cross-lingual), {ref_fr} FR (fallback/mono)")
    for reason, count in sorted(filter_stats["reasons"].items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}")
    logger.info(f"Metadata -> {csv_path}")
    del ds, ds_sampled
    gc.collect()
    return rows


# ============================================================================
# DATASET & DATALOADER
# ============================================================================

class ChatterboxFrDataset(Dataset):
    """PyTorch Dataset — loads pre-saved 16kHz WAV files."""

    def __init__(self, metadata: List[Dict], base_dir: str,
                 max_dur: float = 20.0):
        self.meta = metadata
        self.base = base_dir
        self.max_dur = max_dur

    def __len__(self):
        return len(self.meta)

    def _load_16k(self, path: str) -> np.ndarray:
        """Load audio file, ensure mono, return numpy at 16kHz."""
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = T.Resample(sr, 16000)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav.squeeze(0).numpy()

    def __getitem__(self, idx):
        row = self.meta[idx]
        try:
            wav_16k = self._load_16k(
                os.path.join(self.base, row["file_name"]))
            ref_16k = None
            if row.get("ref_file"):
                ref_path = os.path.join(self.base, row["ref_file"])
                if os.path.exists(ref_path):
                    ref_16k = self._load_16k(ref_path)
            return {
                "wav_16k": wav_16k,
                "ref_16k": ref_16k,
                "text": row["transcription"],
                "speaker_id": row.get("speaker_id", ""),
                "language_id": row.get("language_id", "fr"),
                "duration": float(row.get("duration_seconds", 0)),
            }
        except Exception as e:
            logger.warning(f"Bad sample {idx}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return batch if batch else None


def load_metadata(csv_path: str) -> List[Dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_train_val(meta: List[Dict], val_ratio: float = 0.1,
                    seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified split by speaker — properly randomized."""
    rng = np.random.RandomState(seed)
    by_speaker: Dict[str, List[Dict]] = {}
    for r in meta:
        by_speaker.setdefault(r.get("speaker_id", "?"), []).append(r)

    train, val = [], []
    for speaker, rows in by_speaker.items():
        n_val = max(1, int(len(rows) * val_ratio))
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for rank, idx in enumerate(indices):
            if rank < n_val:
                val.append(rows[idx])
            else:
                train.append(rows[idx])

    logger.info(f"Split → {len(train)} train, {len(val)} val "
                f"({len(by_speaker)} speakers)")
    return train, val


# ============================================================================
# TRAINING METRICS
# ============================================================================

class Metrics:
    def __init__(self):
        self.train_losses, self.val_losses = [], []
        self.lrs, self.grad_norms, self.step_times = [], [], []
        self.steps, self.val_steps = [], []

    def log_train(self, step, loss, lr, grad_norm, dt):
        self.steps.append(step)
        self.train_losses.append(loss)
        self.lrs.append(lr)
        self.grad_norms.append(grad_norm)
        self.step_times.append(dt)

    def log_val(self, step, loss):
        self.val_steps.append(step)
        self.val_losses.append(loss)

    def save(self, path: str):
        """Save raw training curves as JSON for later analysis."""
        data = {
            "steps": self.steps,
            "train_losses": self.train_losses,
            "val_steps": self.val_steps,
            "val_losses": self.val_losses,
            "lrs": self.lrs,
            "grad_norms": self.grad_norms,
            "step_times": self.step_times,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Training curves → {path}")

    def plot(self, path="training_metrics.png"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Chatterbox FR LoRA Fine-Tuning", fontsize=14, fontweight="bold")

        # Train loss
        ax = axes[0, 0]
        ax.plot(self.steps, self.train_losses, "b-", alpha=0.3, label="Raw")
        if len(self.train_losses) > 20:
            w = min(50, len(self.train_losses) // 4)
            sm = np.convolve(self.train_losses, np.ones(w)/w, mode="valid")
            ax.plot(self.steps[w-1:], sm, "b-", lw=2, label="Smoothed")
        ax.set(xlabel="Step", ylabel="Loss", title="Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Val loss
        ax = axes[0, 1]
        if self.val_losses:
            ax.plot(self.val_steps, self.val_losses, "ro-", lw=2, ms=4)
        ax.set(xlabel="Step", ylabel="Loss", title="Validation Loss")
        ax.grid(True, alpha=0.3)

        # LR
        ax = axes[0, 2]
        ax.plot(self.steps, self.lrs, "g-", lw=1.5)
        ax.set(xlabel="Step", ylabel="LR", title="Learning Rate")
        ax.grid(True, alpha=0.3)

        # Grad norm
        ax = axes[1, 0]
        ax.plot(self.steps, self.grad_norms, "m-", alpha=0.5)
        ax.set(xlabel="Step", ylabel="Norm", title="Gradient Norms")
        ax.grid(True, alpha=0.3)

        # Step time
        ax = axes[1, 1]
        ax.plot(self.steps, self.step_times, "c-", alpha=0.5)
        ax.set(xlabel="Step", ylabel="s", title="Step Time")
        ax.grid(True, alpha=0.3)

        # Recent loss histogram
        ax = axes[1, 2]
        recent = self.train_losses[-100:]
        if recent:
            ax.hist(recent, bins=30, color="steelblue", ec="white", alpha=0.8)
            ax.axvline(np.mean(recent), color="red", ls="--",
                       label=f"μ={np.mean(recent):.4f}")
        ax.set(xlabel="Loss", ylabel="Count", title="Recent Loss (last 100)")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================================
# LR SCHEDULE
# ============================================================================

def cosine_lr(step: int, warmup: int, total: int,
              base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# TRAINER
# ============================================================================

class ChatterboxFrTrainer:
    """
    Fine-tuning trainer for Chatterbox Multilingual TTS on French.

    Internally uses the real Chatterbox model components:
      - model.t3   : T3 text→semantic transformer (Llama backbone)
      - model.s3gen: S3 semantic→acoustic generator
      - model.ve   : Voice encoder (speaker conditioning)
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)
        self.metrics = Metrics()
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.total_steps = 0
        self.warmup_steps = 0
        self.eval_every = 0
        self.save_every = 0
        self.test_every = 0

        self._set_seed(config.seed)

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()} "
                        f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    @staticmethod
    def _set_seed(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # RTX 4090 optimizations: enable TF32 for ~2x matmul speedup
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    # ---- Model ----

    def load_model(self):
        logger.info("Loading ChatterboxMultilingualTTS …")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        logger.info("Model loaded ✓")

        # Freeze VE and S3gen completely
        if self.cfg.freeze_ve:
            for p in self.model.ve.parameters():
                p.requires_grad = False
            logger.info("Voice encoder frozen ✓")
        if self.cfg.freeze_s3gen:
            for p in self.model.s3gen.parameters():
                p.requires_grad = False
            logger.info("S3gen frozen ✓")

        if self.cfg.use_lora:
            # First freeze ALL T3 parameters, then inject LoRA
            for p in self.model.t3.parameters():
                p.requires_grad = False
            self.lora_names = inject_lora(
                self.model.t3,
                self.cfg.target_modules,
                self.cfg.lora_rank,
                self.cfg.lora_alpha,
                self.cfg.lora_dropout,
            )

        total = sum(p.numel() for p in self.model.t3.parameters())
        trainable = sum(p.numel() for p in self.model.t3.parameters() if p.requires_grad)
        logger.info(f"T3: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")
        all_total = sum(p.numel() for p in self.model.parameters())
        all_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Full model: {all_total:,} total, {all_train:,} trainable ({100*all_train/all_total:.3f}%)")

        # torch.compile for kernel fusion (RTX 4090 Ampere+)
        if self.cfg.compile_model and hasattr(torch, 'compile'):
            try:
                self.model.t3 = torch.compile(self.model.t3, mode="reduce-overhead")
                logger.info("torch.compile applied to T3 (reduce-overhead mode)")
            except Exception as e:
                logger.warning(f"torch.compile failed (non-fatal): {e}")

    # ---- Data ----

    def prepare_data(self):
        csv_path = os.path.join(self.cfg.audio_data_dir, "metadata.csv")
        if not os.path.exists(csv_path):
            prepare_dataset(self.cfg)
        meta = load_metadata(csv_path)
        logger.info(f"Loaded {len(meta)} samples from metadata")
        self.train_meta, self.val_meta = split_train_val(
            meta, self.cfg.val_split_ratio, self.cfg.seed)
        self.train_loader = DataLoader(
            ChatterboxFrDataset(
                self.train_meta, self.cfg.audio_data_dir,
                self.cfg.max_audio_duration_sec),
            batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=4, collate_fn=collate_fn,
            pin_memory=True, drop_last=True, prefetch_factor=3,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            ChatterboxFrDataset(
                self.val_meta, self.cfg.audio_data_dir,
                self.cfg.max_audio_duration_sec),
            batch_size=4, shuffle=False,
            num_workers=2, collate_fn=collate_fn,
            persistent_workers=True,
        )

        # Compute step counts and ratio-based intervals
        steps_per_epoch = len(self.train_loader) // self.cfg.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.cfg.num_epochs
        self.warmup_steps = max(1, int(self.total_steps * self.cfg.warmup_ratio))
        self.eval_every = max(1, int(self.total_steps * self.cfg.eval_every_ratio))
        self.save_every = max(1, int(self.total_steps * self.cfg.save_every_ratio))
        self.test_every = max(1, int(self.total_steps * self.cfg.test_every_ratio))

        logger.info(f"Steps/epoch:  {steps_per_epoch}")
        logger.info(f"Total steps:  {self.total_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        logger.info(f"Eval every:   {self.eval_every} steps")
        logger.info(f"Save every:   {self.save_every} steps")
        logger.info(f"Test every:   {self.test_every} steps")

    # ---- Optimizer ----

    def setup_optimizer(self):
        if self.cfg.use_lora:
            params = get_lora_params(self.model.t3)
        else:
            params = [p for p in self.model.t3.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        logger.info(f"Optimizer: {n_params:,} parameters")
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.95),
        )

    # ---- Loss ----

    def compute_loss(self, sample: Dict) -> Optional[torch.Tensor]:
        """
        Forward pass through the real Chatterbox pipeline:
          1. VE:  speaker embedding from reference audio  (16 kHz numpy)
          2. S3 tokenizer: extract ground-truth speech tokens (16 kHz numpy)
          3. T3:  teacher-forced forward  → speech_logits
          4. Cross-entropy loss on next-token prediction

        Audio is already pre-resampled to 16 kHz by the DataLoader workers.
        """
        from chatterbox.models.t3.t3 import T3Cond

        text = sample["text"]
        lang = sample.get("language_id", self.cfg.language_id)

        # ── Audio already at 16 kHz numpy from DataLoader ──
        wav_16k = sample["wav_16k"]
        ref_16k = sample.get("ref_16k")
        if ref_16k is None:
            ref_16k = wav_16k

        with torch.no_grad():
            # 1. Speaker embedding via voice encoder
            ve_embed = torch.from_numpy(
                self.model.ve.embeds_from_wavs([ref_16k], sample_rate=16000)
            ).mean(axis=0, keepdim=True).to(self.device)

            # 2. Ground-truth speech tokens from target audio
            s3_tokzr = self.model.s3gen.tokenizer
            speech_tokens, _ = s3_tokzr.forward([wav_16k])
            speech_tokens = torch.atleast_2d(speech_tokens).to(self.device)

            # 3. Conditioning prompt tokens from reference audio
            plen = self.model.t3.hp.speech_cond_prompt_len
            enc_cond_len = getattr(self.model, "ENC_COND_LEN", len(ref_16k))
            cond_tokens, _ = s3_tokzr.forward(
                [ref_16k[:enc_cond_len]], max_len=plen
            )
            cond_tokens = torch.atleast_2d(cond_tokens).to(self.device)

        # 4. Build T3Cond
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=cond_tokens,
            emotion_adv=0.5 * torch.ones(1, 1, 1),
        ).to(device=self.device)

        # 5. Tokenize text
        text_tokens = self.model.tokenizer.text_to_tokens(
            text, language_id=lang
        ).to(self.device)
        sot = self.model.t3.hp.start_text_token
        eot = self.model.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)

        # 6. T3 forward (teacher-forced)
        text_token_lens = torch.tensor([text_tokens.size(1)], device=self.device)
        speech_token_lens = torch.tensor([speech_tokens.size(1)], device=self.device)

        out = self.model.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )

        # 7. Next-token prediction loss on speech logits
        speech_logits = out.speech_logits          # (B, len_speech, vocab)
        target = speech_tokens[:, 1:].contiguous()
        logits = speech_logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-1,
        )
        return loss if not (torch.isnan(loss) or torch.isinf(loss)) else None

    # ---- Validation ----

    @torch.no_grad()
    def validate(self) -> float:
        self.model.t3.eval()
        losses = []
        for batch in self.val_loader:
            if batch is None:
                continue
            for sample in batch:
                try:
                    with torch.amp.autocast("cuda", enabled=self.cfg.fp16):
                        loss = self.compute_loss(sample)
                    if loss is not None:
                        losses.append(loss.item())
                except Exception:
                    continue
        self.model.t3.train()
        return np.mean(losses) if losses else float("inf")

    # ---- Sample generation ----

    @torch.no_grad()
    def generate_samples(self, step: int):
        self.model.t3.eval()
        out_dir = os.path.join(self.cfg.output_dir, "samples", f"step_{step}")
        os.makedirs(out_dir, exist_ok=True)

        texts = [
            "Bonjour, comment allez-vous aujourd'hui ? Je suis ravi de vous rencontrer.",
            "L'intelligence artificielle transforme notre façon de communiquer.",
            "La recherche en traitement automatique du langage naturel progresse rapidement.",
        ]

        # Pick random references from val set for variety
        # Use English reference voice (ref_file) for cross-lingual evaluation
        import shutil
        try:
            for i, txt in enumerate(texts[:self.cfg.num_test_samples]):
                ref_sample = random.choice(self.val_meta)
                spk_name = ref_sample.get("speaker_name", ref_sample.get("speaker_id", "?"))

                # Prefer English reference voice for cross-lingual evaluation
                ref_file = ref_sample.get("ref_file", "")
                if ref_file:
                    ref_path = os.path.join(self.cfg.audio_data_dir, ref_file)
                else:
                    ref_path = os.path.join(self.cfg.audio_data_dir, ref_sample["file_name"])
                if not os.path.exists(ref_path):
                    ref_path = os.path.join(self.cfg.audio_data_dir, ref_sample["file_name"])

                wav = self.model.generate(
                    txt, audio_prompt_path=ref_path,
                    language_id=self.cfg.language_id,
                )
                if isinstance(wav, torch.Tensor):
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    # Save generated FR sample
                    torchaudio.save(
                        os.path.join(out_dir, f"sample_{i}.wav"),
                        wav.cpu(), self.model.sr,
                    )
                    # Save EN reference voice for A/B comparison
                    ref_dest = os.path.join(out_dir, f"ref_{i}_{spk_name}_EN.wav")
                    shutil.copy2(ref_path, ref_dest)
                    logger.info(f"  Generated sample_{i}.wav (FR) from ref: {spk_name} (EN) + ref saved")
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")

        self.model.t3.train()

    # ---- Checkpoint ----

    def save_checkpoint(self, step: int, val_loss: float = None, is_best=False):
        ckpt_dir = os.path.join(self.cfg.output_dir, "checkpoints")
        lora_config = {
            "rank": self.cfg.lora_rank, "alpha": self.cfg.lora_alpha,
            "dropout": self.cfg.lora_dropout, "target_modules": self.cfg.target_modules,
        }
        payload = {
            "step": step, "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "lora_config": lora_config,
            "lora_state_dict": get_lora_state_dict(self.model.t3),
        }
        torch.save(payload, os.path.join(ckpt_dir, f"checkpoint_step{step}.pt"))
        logger.info(f"Checkpoint saved (step {step})")
        if is_best:
            save_lora(self.model.t3,
                      os.path.join(ckpt_dir, "best_lora_adapter.pt"),
                      config=lora_config)

    def load_checkpoint(self, path: str):
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt.get("step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "lora_state_dict" in ckpt:
            current_sd = self.model.t3.state_dict()
            loaded = 0
            for k, v in ckpt["lora_state_dict"].items():
                if k in current_sd:
                    current_sd[k] = v.to(self.device)
                    loaded += 1
            self.model.t3.load_state_dict(current_sd, strict=False)
            logger.info(f"  Loaded {loaded} LoRA tensors")
        logger.info(f"Resumed at step {self.global_step}, best_val={self.best_val_loss:.4f}")

    # ---- Main loop ----

    def train(self, resume_from: Optional[str] = None):
        logger.info("=" * 70)
        logger.info("Chatterbox Multilingual LoRA Fine-Tuning (French)")
        logger.info("=" * 70)
        logger.info(f"  Language:        {self.cfg.language_id}")
        logger.info(f"  Cloning mode:    {self.cfg.cloning_mode}")
        logger.info(f"  Epochs:          {self.cfg.num_epochs}")
        logger.info(f"  Batch size:      {self.cfg.batch_size}")
        logger.info(f"  Grad accum:      {self.cfg.gradient_accumulation_steps}")
        logger.info(f"  Effective BS:    {self.cfg.batch_size * self.cfg.gradient_accumulation_steps}")
        logger.info(f"  LR:              {self.cfg.learning_rate}")
        logger.info(f"  LoRA:          rank={self.cfg.lora_rank}, alpha={self.cfg.lora_alpha}")
        logger.info(f"  Targets:       {self.cfg.target_modules}")
        logger.info(f"  Total steps:   {self.total_steps}")
        logger.info(f"  Warmup steps:  {self.warmup_steps}")
        logger.info(f"  Patience:      {self.cfg.patience} eval cycles")
        logger.info(f"  FP16:          {self.cfg.fp16}")
        logger.info("=" * 70)

        with open(os.path.join(self.cfg.output_dir, "training_config.json"), "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2, default=str)

        if resume_from:
            self.load_checkpoint(resume_from)

        self.model.t3.train()
        accum_loss = 0.0
        accum_count = 0
        step_t0 = time.time()

        # Cache LoRA params list to avoid re-walking the module tree every step
        cached_lora_params = get_lora_params(self.model.t3)
        logger.info(f"Cached {len(cached_lora_params)} LoRA parameter tensors")

        for epoch in range(self.cfg.num_epochs):
            logger.info(f"\n{SEP}  Epoch {epoch+1}/{self.cfg.num_epochs}  {SEP}")
            epoch_losses = []
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=True)

            for batch_idx, batch in enumerate(progress):
                if batch is None:
                    continue

                batch_loss = 0.0
                valid = 0

                for sample in batch:
                    try:
                        with torch.amp.autocast("cuda", enabled=self.cfg.fp16):
                            loss = self.compute_loss(sample)
                        if loss is not None:
                            scaled = loss / self.cfg.gradient_accumulation_steps
                            self.scaler.scale(scaled).backward()
                            batch_loss += loss.item()
                            valid += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("OOM \u2013 skipping")
                            torch.cuda.empty_cache()
                        else:
                            logger.warning(f"Error: {e}")
                        continue

                if valid > 0:
                    accum_loss += batch_loss / valid
                    accum_count += 1

                is_accum_boundary = (
                    (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(self.train_loader)
                )
                if not is_accum_boundary or accum_count == 0:
                    continue

                self.scaler.unscale_(self.optimizer)
                gn = torch.nn.utils.clip_grad_norm_(cached_lora_params, self.cfg.max_grad_norm)

                lr = cosine_lr(self.global_step, self.warmup_steps,
                               self.total_steps, self.cfg.learning_rate,
                               self.cfg.min_learning_rate)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                avg_loss = accum_loss / accum_count
                dt = time.time() - step_t0
                epoch_losses.append(avg_loss)
                gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
                self.metrics.log_train(self.global_step, avg_loss, lr, gn_val, dt)

                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=self.global_step)

                if self.global_step % self.cfg.log_every_n_steps == 0:
                    logger.info(
                        f"  Step {self.global_step:5d} | Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e} | Grad {gn_val:.4f} | {dt:.1f}s")

                self.global_step += 1
                accum_loss = 0.0
                accum_count = 0
                step_t0 = time.time()

                # ── Periodic: Validate ──
                if self.global_step > 0 and self.global_step % self.eval_every == 0:
                    vl = self.validate()
                    self.metrics.log_val(self.global_step, vl)
                    is_best = vl < self.best_val_loss
                    if is_best:
                        self.best_val_loss = vl
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    self.save_checkpoint(self.global_step, vl, is_best)
                    star_text = f" {STAR} best!" if is_best else ""
                    logger.info(
                        f"  Val loss: {vl:.4f}{star_text}"
                        f" (patience: {self.patience_counter}/{self.cfg.patience})")
                    if self.patience_counter >= self.cfg.patience:
                        logger.info(f"Early stopping at step {self.global_step}")
                        self._finish_training()
                        return

                # ── Periodic: Save ──
                if self.global_step > 0 and self.global_step % self.save_every == 0:
                    self.save_checkpoint(self.global_step)
                    self.metrics.plot(os.path.join(self.cfg.output_dir, "training_metrics.png"))
                    self.metrics.save(os.path.join(self.cfg.output_dir, "training_metrics.json"))

                # ── Periodic: Generate samples ──
                if self.global_step > 0 and self.global_step % self.test_every == 0:
                    self.generate_samples(self.global_step)

            # Epoch summary
            if epoch_losses:
                logger.info(f"Epoch {epoch+1}: mean={np.mean(epoch_losses):.4f} "
                            f"min={np.min(epoch_losses):.4f}")

            # End-of-epoch validation
            vl = self.validate()
            self.metrics.log_val(self.global_step, vl)
            is_best = vl < self.best_val_loss
            if is_best:
                self.best_val_loss = vl
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            star_text = f" {STAR} best!" if is_best else ""
            logger.info(f"Epoch {epoch+1} val: {vl:.4f}{star_text}"
                        f" (patience: {self.patience_counter}/{self.cfg.patience})")

        self._finish_training()

    def _finish_training(self):
        """Save final artifacts and optionally upload."""
        logger.info("\n" + "=" * 70)
        logger.info("Training complete!")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        ckpt_dir = os.path.join(self.cfg.output_dir, "checkpoints")
        lora_config = {
            "rank": self.cfg.lora_rank, "alpha": self.cfg.lora_alpha,
            "dropout": self.cfg.lora_dropout, "target_modules": self.cfg.target_modules,
        }
        save_lora(self.model.t3,
                  os.path.join(ckpt_dir, "final_lora_adapter.pt"),
                  config=lora_config)
        self.generate_samples(self.global_step)
        self.metrics.plot(os.path.join(self.cfg.output_dir, "training_metrics.png"))
        self.metrics.save(os.path.join(self.cfg.output_dir, "training_metrics.json"))

        if self.cfg.hf_token:
            self.upload_to_hf()

        logger.info("All done ✓")

    # ---- HuggingFace Upload ----

    def upload_to_hf(self):
        """Upload LoRA adapter, merged model, samples, config to HuggingFace."""
        from huggingface_hub import HfApi, create_repo

        repo_id = self.cfg.hf_repo_id or f"{self.cfg.hf_username}/{self.cfg.hf_repo_name}"
        
        fraction_str = (f"{self.cfg.sample_fraction:.0%} stratified sample" 
                        if self.cfg.sample_fraction < 1.0 
                        else "full dataset")
        
        readme = f"""---
license: apache-2.0
base_model: chatterbox-multilingual
tags:
- tts
- voice-cloning
- multilingual
- french
- lora
---
# Chatterbox French LoRA Fine-Tuned

This is a LoRA adapter for the Chatterbox Multilingual TTS model, fine-tuned specifically for French voice cloning.

## Model Details
- **Base Model**: Chatterbox Multilingual
- **Language**: French (`fr`)
- **Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: {self.cfg.dataset_name} ({fraction_str})

## How to use
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
# 1. Load base
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
# 2. Add LoRA (inject + load_lora)
# (Self-describing adapter reads rank/alpha/targets from metadata)
```
"""
        logger.info(f"Uploading to HF: {repo_id} ...")
        api = HfApi(token=self.cfg.hf_token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            
            # Upload checkpoints
            api.upload_folder(
                folder_path=os.path.join(self.cfg.output_dir, "checkpoints"),
                path_in_repo="checkpoints",
                repo_id=repo_id,
                repo_type="model",
            )
            
            # Upload training dataset for reproducibility
            if not self.cfg.skip_dataset_upload:
                data_dir = self.cfg.audio_data_dir
                if os.path.exists(data_dir):
                    logger.info(f"  Uploading training dataset (audio_data) ...")
                    api.upload_folder(
                        folder_path=data_dir,
                        path_in_repo="training_dataset",
                        repo_id=repo_id,
                        repo_type="model",
                    )
            else:
                logger.info("  Skipping dataset upload (already on HF)")
            
            # Create README
            api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(f"  Uploaded: README.md")
        except Exception as e:
            logger.warning(f"  HuggingFace upload failed: {e}")

        logger.info(f"✅ Model uploaded to https://huggingface.co/{repo_id}")


# ============================================================================
# INFERENCE HELPER
# ============================================================================

class ChatterboxFrInference:
    """Inference with the fine-tuned model."""

    def __init__(self, model_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model = None

    def load(self, use_lora: bool = True):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        logger.info("Loading base model + LoRA adapter …")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        if use_lora:
            lora_path = os.path.join(
                self.model_dir, "checkpoints", "best_lora_adapter.pt")
            if os.path.exists(lora_path):
                # Read config from saved adapter
                payload = torch.load(lora_path, map_location=self.device, weights_only=True)
                cfg = payload.get("config", {})
                inject_lora(
                    self.model.t3,
                    cfg.get("target_modules", ["q_proj", "v_proj"]),
                    cfg.get("rank", 8),
                    cfg.get("alpha", 16.0),
                    cfg.get("dropout", 0.1),
                )
                load_lora(self.model.t3, lora_path, self.device)
            else:
                logger.warning(f"No LoRA adapter found at {lora_path}")

        self.model.eval() if hasattr(self.model, 'eval') else None
        return self.model

    def synthesize(self, text: str, ref_audio_path: str,
                   output_path: str = "output.wav",
                   exaggeration: float = 0.5,
                   cfg_weight: float = 0.5) -> str:
        assert self.model is not None, "Call .load() first"
        with torch.no_grad():
            wav = self.model.generate(
                text, audio_prompt_path=ref_audio_path,
                language_id="fr",
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            torchaudio.save(output_path, wav.cpu(), self.model.sr)
            logger.info(f"Saved → {output_path}")
        return output_path

    def batch_synthesize(self, texts: List[str], ref_audio_path: str,
                         output_dir: str = "./outputs") -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, text in enumerate(tqdm(texts, desc="Synthesizing")):
            p = os.path.join(output_dir, f"output_{i:04d}.wav")
            self.synthesize(text, ref_audio_path, p)
            paths.append(p)
        return paths


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Chatterbox Multilingual TTS for French")
    p.add_argument("--mode", choices=["train", "prepare-only", "inference", "evaluate"],
                   default="train")
    p.add_argument("--output-dir", default="./chatterbox_fr_finetuned")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--sample-fraction", type=float, default=None)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--cloning-mode", choices=["cross_lingual", "monolingual"],
                   default="cross_lingual")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token for uploading model after training")
    p.add_argument("--skip-dataset-upload", action="store_true",
                   help="Skip uploading the 60GB dataset if already on HF")
    # Inference
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--ref-audio", type=str, default=None)
    p.add_argument("--out-audio", type=str, default="output.wav")
    return p.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(output_dir=args.output_dir, cloning_mode=args.cloning_mode)
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.epochs is not None: config.num_epochs = args.epochs
    if args.lr is not None: config.learning_rate = args.lr
    if args.lora_rank is not None: config.lora_rank = args.lora_rank
    if args.sample_fraction is not None: config.sample_fraction = args.sample_fraction
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.use_lora = args.use_lora
    config.fp16 = args.fp16
    if args.hf_token is not None: config.hf_token = args.hf_token
    config.skip_dataset_upload = args.skip_dataset_upload

    if args.mode == "prepare-only":
        prepare_dataset(config)
        if config.hf_token:
            trainer = ChatterboxFrTrainer(config)
            trainer.upload_to_hf()
        logger.info("Data preparation complete.")
        return

    if args.mode == "train":
        trainer = ChatterboxFrTrainer(config)
        trainer.load_model()
        trainer.prepare_data()
        trainer.setup_optimizer()
        trainer.train(resume_from=args.resume)

    elif args.mode == "inference":
        assert args.text, "Provide --text"
        assert args.ref_audio, "Provide --ref-audio"
        inf = ChatterboxFrInference(args.output_dir)
        inf.load(use_lora=config.use_lora)
        inf.synthesize(args.text, args.ref_audio, args.out_audio)

    elif args.mode == "evaluate":
        trainer = ChatterboxFrTrainer(config)
        trainer.load_model()
        trainer.prepare_data()
        trainer.setup_optimizer()

        best = os.path.join(config.output_dir, "checkpoints", "best_lora_adapter.pt")
        if os.path.exists(best):
            logger.info(f"Loading best LoRA adapter from {best}")
            # Use the robust load_lora function
            load_lora(trainer.model.t3, best, str(trainer.device))
        else:
            logger.warning(f"Best adapter not found at {best}. Checking for newest checkpoint...")
            import glob
            ckpts = sorted(glob.glob(os.path.join(config.output_dir, "checkpoints", "checkpoint_step*.pt")))
            if ckpts:
                trainer.load_checkpoint(ckpts[-1])
            else:
                logger.error("No checkpoints found to evaluate!")
                return

        vl = trainer.validate()
        logger.info(f"Evaluation loss: {vl:.4f}")
        trainer.generate_samples(step=0)


if __name__ == "__main__":
    main()
