#!/usr/bin/env python3
"""
Fine-tune Chatterbox Multilingual TTS for French Voice Cloning
==============================================================
Uses LoRA (Low-Rank Adaptation) on the T3 (text→semantic) transformer
of ChatterboxMultilingualTTS, trained on 20% stratified sample of
`amanuelbyte/acl-voice-cloning-fr-expanded2`.

Supports:
  - Cross-lingual mode (EN reference → FR output)
  - Monolingual mode (FR reference → FR output)
  - Mixed-precision (FP16) training
  - Gradient accumulation
  - Checkpoint resume
  - Periodic sample generation

Usage:
    python finetune_chatterbox_fr.py --mode train
    python finetune_chatterbox_fr.py --mode train --use-lora --lora-rank 32
    python finetune_chatterbox_fr.py --mode train --resume checkpoints/checkpoint_step400.pt
    python finetune_chatterbox_fr.py --mode prepare-only
    python finetune_chatterbox_fr.py --mode inference --text "Bonjour" --ref-audio ref.wav
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
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import torchaudio.transforms as T
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_chatterbox_fr.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """All hyperparameters and paths for fine-tuning."""

    # ---- Dataset ----
    dataset_name: str = "amanuelbyte/acl-voice-cloning-fr-expanded2"
    dataset_split: str = "train"
    sample_fraction: float = 0.20          # Use 20% of training set
    cache_dir: str = "./data_cache"
    audio_data_dir: str = "./audio_data"

    # ---- Audio ----
    sample_rate: int = 24000               # Chatterbox native SR
    max_audio_duration_sec: float = 20.0
    min_audio_duration_sec: float = 0.5
    ref_audio_max_duration_sec: float = 10.0

    # ---- Training ----
    output_dir: str = "./chatterbox_fr_finetuned"
    num_epochs: int = 10
    batch_size: int = 2                    # Per-GPU batch size
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    fp16: bool = True
    seed: int = 42

    # ---- LoRA ----
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ---- Freezing (full fine-tune mode) ----
    freeze_encoder: bool = True
    freeze_decoder_layers: int = 0

    # ---- Validation / checkpointing ----
    val_split_ratio: float = 0.10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 200
    log_every_n_steps: int = 10
    test_every_n_steps: int = 1000
    num_test_samples: int = 3

    # ---- Language ----
    language_id: str = "fr"
    cloning_mode: str = "cross_lingual"    # "cross_lingual" or "monolingual"
    num_speakers: int = 8

    def __post_init__(self):
        for d in [
            self.output_dir,
            self.cache_dir,
            os.path.join(self.output_dir, "checkpoints"),
            os.path.join(self.output_dir, "samples"),
            os.path.join(self.audio_data_dir, "audio"),
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
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
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
                alpha: float, dropout: float = 0.0) -> List[LoRALayer]:
    """Inject LoRA layers into matching nn.Linear modules."""
    layers = []
    for name, module in list(model.named_modules()):
        for target in targets:
            if target in name and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                lora = LoRALayer(module, rank, alpha, dropout)
                setattr(parent, parts[-1], lora)
                layers.append(lora)
                break
    logger.info(f"Injected LoRA into {len(layers)} layers (rank={rank}, alpha={alpha})")
    return layers


def get_lora_params(layers: List[LoRALayer]) -> List[nn.Parameter]:
    params = []
    for layer in layers:
        params.extend([layer.lora_A, layer.lora_B])
    return params


def save_lora_state(layers: List[LoRALayer], path: str):
    state = {}
    for i, layer in enumerate(layers):
        state[f"layer_{i}_A"] = layer.lora_A.data.cpu()
        state[f"layer_{i}_B"] = layer.lora_B.data.cpu()
        state[f"layer_{i}_rank"] = layer.rank
        state[f"layer_{i}_alpha"] = layer.alpha
    torch.save(state, path)
    logger.info(f"LoRA adapter saved → {path}")


def load_lora_state(layers: List[LoRALayer], path: str, device: str = "cuda"):
    state = torch.load(path, map_location=device, weights_only=True)
    for i, layer in enumerate(layers):
        if f"layer_{i}_A" in state:
            layer.lora_A.data = state[f"layer_{i}_A"].to(device)
            layer.lora_B.data = state[f"layer_{i}_B"].to(device)
    logger.info(f"LoRA adapter loaded ← {path}")


def merge_lora(layers: List[LoRALayer]):
    """Merge LoRA weights into original linear layers permanently."""
    for layer in layers:
        with torch.no_grad():
            layer.original_layer.weight.data += (layer.lora_B @ layer.lora_A) * layer.scaling
    logger.info("LoRA weights merged into base model")


# ============================================================================
# DATA PREPARATION  — HuggingFace → WAV files + metadata.csv
# ============================================================================

def prepare_dataset(config: TrainingConfig) -> List[Dict]:
    """
    Download HF dataset, stratified-sample per speaker, extract French
    target audio as WAV files, create metadata.csv.
    """
    logger.info(f"Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name, split=config.dataset_split,
                      cache_dir=config.cache_dir)
    logger.info(f"Full dataset size: {len(ds)} rows")

    # ── Stratified 20 % sampling ──
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
        logger.info(f"  {speaker}: {len(indices)} total → {n} sampled")

    sampled_indices.sort()
    ds_sampled = ds.select(sampled_indices)
    logger.info(f"Sampled dataset: {len(ds_sampled)} rows")

    # ── Extract audio + build metadata ──
    audio_dir = os.path.join(config.audio_data_dir, "audio")
    ref_dir = os.path.join(config.audio_data_dir, "ref_audio")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    rows = []
    skipped = 0

    for i in tqdm(range(len(ds_sampled)), desc="Extracting audio"):
        sample = ds_sampled[i]
        text = (sample.get("trg_fr_text") or "").strip()
        if not text or len(text) < 2 or len(text) > 1000:
            skipped += 1; continue

        # ── Target French audio ──
        trg = sample.get("trg_fr_voice")
        if trg is None or not isinstance(trg, dict):
            skipped += 1; continue
        wav = torch.from_numpy(np.asarray(trg["array"], dtype=np.float32))
        sr = trg["sampling_rate"]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        dur = wav.shape[-1] / sr
        if dur < config.min_audio_duration_sec or dur > config.max_audio_duration_sec:
            skipped += 1; continue
        if sr != config.sample_rate:
            wav = T.Resample(sr, config.sample_rate)(wav)

        trg_file = f"{sample['speaker_id']}_{i:06d}.wav"
        torchaudio.save(os.path.join(audio_dir, trg_file), wav, config.sample_rate)

        # ── Reference audio (for voice conditioning) ──
        # cross-lingual: prefer EN reference; monolingual: prefer FR
        if config.cloning_mode == "cross_lingual":
            ref_data = sample.get("ref_en_voice") or sample.get("ref_fr_voice")
        else:
            ref_data = sample.get("ref_fr_voice") or sample.get("ref_en_voice")

        ref_file = ""
        if ref_data is not None and isinstance(ref_data, dict):
            ref_wav = torch.from_numpy(np.asarray(ref_data["array"], dtype=np.float32))
            ref_sr = ref_data["sampling_rate"]
            if ref_wav.dim() == 1:
                ref_wav = ref_wav.unsqueeze(0)
            if ref_sr != config.sample_rate:
                ref_wav = T.Resample(ref_sr, config.sample_rate)(ref_wav)
            # Trim reference
            max_ref = int(config.ref_audio_max_duration_sec * config.sample_rate)
            ref_wav = ref_wav[..., :max_ref]
            ref_file = f"ref_{sample['speaker_id']}_{i:06d}.wav"
            torchaudio.save(os.path.join(ref_dir, ref_file), ref_wav, config.sample_rate)

        rows.append({
            "file_name": f"audio/{trg_file}",
            "ref_file": f"ref_audio/{ref_file}" if ref_file else "",
            "transcription": text,
            "duration_seconds": round(wav.shape[-1] / config.sample_rate, 3),
            "speaker_id": sample["speaker_id"],
            "speaker_name": sample.get("speaker_name", ""),
            "language_id": config.language_id,
        })

        if (i + 1) % 2000 == 0:
            gc.collect()

    # ── Write metadata.csv ──
    csv_path = os.path.join(config.audio_data_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    logger.info(f"Prepared {len(rows)} samples ({skipped} skipped) → {csv_path}")
    del ds, ds_sampled; gc.collect()
    return rows


# ============================================================================
# DATASET & DATALOADER
# ============================================================================

class ChatterboxFrDataset(Dataset):
    """PyTorch Dataset for Chatterbox fine-tuning."""

    def __init__(self, metadata: List[Dict], base_dir: str,
                 sr: int = 24000, max_dur: float = 20.0):
        self.meta = metadata
        self.base = base_dir
        self.sr = sr
        self.max_dur = max_dur

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta[idx]
        try:
            wav, sr = torchaudio.load(os.path.join(self.base, row["file_name"]))
            if sr != self.sr:
                wav = T.Resample(sr, self.sr)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

            # Load reference audio if available
            ref_wav = None
            if row.get("ref_file"):
                ref_path = os.path.join(self.base, row["ref_file"])
                if os.path.exists(ref_path):
                    ref_wav, ref_sr = torchaudio.load(ref_path)
                    if ref_sr != self.sr:
                        ref_wav = T.Resample(ref_sr, self.sr)(ref_wav)
                    if ref_wav.shape[0] > 1:
                        ref_wav = ref_wav.mean(0, keepdim=True)
                    ref_wav = ref_wav.squeeze(0)

            return {
                "waveform": wav.squeeze(0),
                "ref_waveform": ref_wav,
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
    """Stratified split by speaker."""
    rng = np.random.RandomState(seed)
    by_speaker: Dict[str, List[Dict]] = {}
    for r in meta:
        by_speaker.setdefault(r.get("speaker_id", "?"), []).append(r)

    train, val = [], []
    for rows in by_speaker.values():
        n_val = max(1, int(len(rows) * val_ratio))
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        for i, r in enumerate(rows):
            (val if i < n_val else train).append(r)

    logger.info(f"Split → {len(train)} train, {len(val)} val")
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

def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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
        self.lora_layers: List[LoRALayer] = []
        self.optimizer = None
        self.scaler = GradScaler(enabled=config.fp16)
        self.metrics = Metrics()
        self.global_step = 0
        self.best_val_loss = float("inf")

        self._set_seed(config.seed)

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()} "
                        f"({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")

    @staticmethod
    def _set_seed(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ---- Model ----

    def load_model(self):
        logger.info("Loading ChatterboxMultilingualTTS …")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        logger.info("Model loaded ✓")

        if self.cfg.use_lora:
            self._apply_lora()
        else:
            self._apply_freezing()

        total = sum(p.numel() for p in self.model.t3.parameters())
        trainable = sum(p.numel() for p in self.model.t3.parameters() if p.requires_grad)
        logger.info(f"T3 params: {total:,} total, {trainable:,} trainable "
                    f"({100*trainable/total:.2f}%)")

    def _apply_lora(self):
        logger.info("Injecting LoRA into T3 transformer …")
        self.lora_layers = inject_lora(
            self.model.t3,
            self.cfg.target_modules,
            self.cfg.lora_rank,
            self.cfg.lora_alpha,
            self.cfg.lora_dropout,
        )

    def _apply_freezing(self):
        """Selective freezing for full fine-tune mode."""
        frozen = 0
        for name, p in self.model.named_parameters():
            if any(kw in name.lower()
                   for kw in ["ve.", "s3gen.", "encoder", "spk_encoder"]):
                p.requires_grad = False
                frozen += 1
        logger.info(f"Froze {frozen} non-T3 parameters")

    # ---- Data ----

    def prepare_data(self):
        csv_path = os.path.join(self.cfg.audio_data_dir, "metadata.csv")
        if not os.path.exists(csv_path):
            prepare_dataset(self.cfg)
        meta = load_metadata(csv_path)
        logger.info(f"Loaded {len(meta)} samples from metadata")
        self.train_meta, self.val_meta = split_train_val(
            meta, self.cfg.val_split_ratio, self.cfg.seed
        )
        self.train_loader = DataLoader(
            ChatterboxFrDataset(self.train_meta, self.cfg.audio_data_dir, self.cfg.sample_rate),
            batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=2, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            ChatterboxFrDataset(self.val_meta, self.cfg.audio_data_dir, self.cfg.sample_rate),
            batch_size=1, shuffle=False,
            num_workers=1, collate_fn=collate_fn,
        )

    # ---- Optimizer ----

    def setup_optimizer(self):
        if self.cfg.use_lora:
            params = get_lora_params(self.lora_layers)
        else:
            params = [p for p in self.model.t3.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.95),
        )
        self.total_steps = (
            len(self.train_loader) * self.cfg.num_epochs
            // self.cfg.gradient_accumulation_steps
        )
        logger.info(f"Optimizer ready – {self.total_steps} total steps")

    # ---- Loss ----

    def compute_loss(self, sample: Dict) -> Optional[torch.Tensor]:
        """
        Forward pass through the real Chatterbox pipeline:
          1. VE: extract speaker conditioning from reference (or target) audio
          2. S3gen: extract ground-truth semantic tokens from target audio
          3. T3: predict semantic tokens from text + conditioning
          4. Cross-entropy loss on the semantic token predictions
        """
        wav = sample["waveform"].unsqueeze(0).to(self.device)       # [1, T_audio]
        text = sample["text"]
        lang = sample.get("language_id", self.cfg.language_id)

        # Use reference audio for speaker conditioning if available,
        # otherwise fall back to target audio (single-speaker scenario)
        ref = sample.get("ref_waveform")
        if ref is not None:
            ref = ref.unsqueeze(0).to(self.device)
        else:
            ref = wav

        with torch.no_grad():
            # Speaker conditioning vector
            conds = self.model.ve(ref)
            # Ground-truth semantic tokens from target audio
            s3_tokens = self.model.s3gen.extract_s3_tokens(wav)

        # Tokenize text
        text_tokens = self.model.t3.tokenize_text(text, lang_id=lang)
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.to(self.device)
        else:
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)

        # T3 forward (teacher-forced)
        logits = self.model.t3.forward_train(
            text_tokens=text_tokens,
            s3_tokens=s3_tokens,
            conds=conds,
            language_id=lang,
        )
        if isinstance(logits, tuple):
            logits = logits[0]

        # Next-token prediction loss
        target = s3_tokens[:, 1:].contiguous()
        logits = logits[:, :target.shape[1], :].contiguous()
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
                    with autocast(enabled=self.cfg.fp16, device_type="cuda"):
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

        # Pick a reference from val set
        try:
            ref_sample = self.val_meta[0]
            ref_path = os.path.join(self.cfg.audio_data_dir, ref_sample["file_name"])

            for i, txt in enumerate(texts[:self.cfg.num_test_samples]):
                wav = self.model.generate(
                    txt, audio_prompt_path=ref_path,
                    language_id=self.cfg.language_id,
                )
                if isinstance(wav, torch.Tensor):
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    torchaudio.save(
                        os.path.join(out_dir, f"sample_{i}.wav"),
                        wav.cpu(), self.model.sr,
                    )
                    logger.info(f"  Generated sample_{i}.wav")
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")

        self.model.t3.train()

    # ---- Checkpoint ----

    def save_checkpoint(self, step: int, val_loss: float = None, is_best=False):
        ckpt_dir = os.path.join(self.cfg.output_dir, "checkpoints")
        payload = {
            "step": step,
            "config": self.cfg.__dict__,
            "val_loss": val_loss,
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.cfg.use_lora:
            lora_st = {}
            for i, layer in enumerate(self.lora_layers):
                lora_st[f"layer_{i}_A"] = layer.lora_A.data.cpu()
                lora_st[f"layer_{i}_B"] = layer.lora_B.data.cpu()
            payload["lora_state"] = lora_st
            torch.save(payload, os.path.join(ckpt_dir, f"checkpoint_step{step}.pt"))
            if is_best:
                save_lora_state(self.lora_layers,
                                os.path.join(ckpt_dir, "best_lora_adapter.pt"))
        else:
            payload["model_state"] = {
                n: p.cpu().detach()
                for n, p in self.model.t3.named_parameters() if p.requires_grad
            }
            torch.save(payload, os.path.join(ckpt_dir, f"checkpoint_step{step}.pt"))
            if is_best:
                torch.save(payload, os.path.join(ckpt_dir, "best_model.pt"))

    def load_checkpoint(self, path: str):
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf")) or float("inf")
        if self.cfg.use_lora and "lora_state" in ckpt:
            for i, layer in enumerate(self.lora_layers):
                if f"layer_{i}_A" in ckpt["lora_state"]:
                    layer.lora_A.data = ckpt["lora_state"][f"layer_{i}_A"].to(self.device)
                    layer.lora_B.data = ckpt["lora_state"][f"layer_{i}_B"].to(self.device)
        elif "model_state" in ckpt:
            for n, p in self.model.t3.named_parameters():
                if n in ckpt["model_state"]:
                    p.data = ckpt["model_state"][n].to(self.device)
        logger.info(f"Resumed at step {self.global_step}")

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
        logger.info(f"  LoRA:            {self.cfg.use_lora} (rank={self.cfg.lora_rank})")
        logger.info(f"  FP16:            {self.cfg.fp16}")
        logger.info(f"  Output:          {self.cfg.output_dir}")
        logger.info("=" * 70)

        # Save config
        with open(os.path.join(self.cfg.output_dir, "training_config.json"), "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2, default=str)

        if resume_from:
            self.load_checkpoint(resume_from)

        self.model.t3.train()
        accum_loss = 0.0
        accum_count = 0
        step_t0 = time.time()

        for epoch in range(self.cfg.num_epochs):
            logger.info(f"\n{'═'*50}  Epoch {epoch+1}/{self.cfg.num_epochs}  {'═'*50}")
            epoch_losses = []
            progress = tqdm(self.train_loader,
                            desc=f"Epoch {epoch+1}", leave=True)

            for batch_idx, batch in enumerate(progress):
                if batch is None:
                    continue

                batch_loss = 0.0
                valid = 0

                for sample in batch:
                    try:
                        with autocast(enabled=self.cfg.fp16, device_type="cuda"):
                            loss = self.compute_loss(sample)

                        if loss is not None:
                            scaled = loss / self.cfg.gradient_accumulation_steps
                            self.scaler.scale(scaled).backward()
                            batch_loss += loss.item()
                            valid += 1

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("OOM – skipping sample")
                            torch.cuda.empty_cache()
                        else:
                            logger.warning(f"Error: {e}")
                        continue

                if valid > 0:
                    accum_loss += batch_loss / valid
                    accum_count += 1

                # Optimizer step at accumulation boundary
                if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0 or \
                   (batch_idx + 1) == len(self.train_loader):
                    if accum_count > 0:
                        self.scaler.unscale_(self.optimizer)
                        params = (get_lora_params(self.lora_layers) if self.cfg.use_lora
                                  else [p for p in self.model.t3.parameters() if p.requires_grad])
                        gn = torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)

                        lr = cosine_lr(self.global_step, self.cfg.warmup_steps,
                                       self.total_steps, self.cfg.learning_rate)
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = lr

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        avg_loss = accum_loss / accum_count
                        dt = time.time() - step_t0
                        epoch_losses.append(avg_loss)
                        self.metrics.log_train(
                            self.global_step, avg_loss, lr,
                            gn.item() if isinstance(gn, torch.Tensor) else gn, dt)

                        progress.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            lr=f"{lr:.2e}",
                            step=self.global_step,
                        )

                        if self.global_step % self.cfg.log_every_n_steps == 0:
                            logger.info(
                                f"  Step {self.global_step:5d} | "
                                f"Loss {avg_loss:.4f} | LR {lr:.2e} | "
                                f"Grad {gn:.4f} | {dt:.1f}s"
                            )

                        self.global_step += 1
                        accum_loss = 0.0
                        accum_count = 0
                        step_t0 = time.time()

                    # ── Periodic actions ──
                    if self.global_step > 0 and self.global_step % self.cfg.save_every_n_steps == 0:
                        self.save_checkpoint(self.global_step)
                        self.metrics.plot(
                            os.path.join(self.cfg.output_dir, "training_metrics.png"))

                    if self.global_step > 0 and self.global_step % self.cfg.eval_every_n_steps == 0:
                        vl = self.validate()
                        self.metrics.log_val(self.global_step, vl)
                        is_best = vl < self.best_val_loss
                        if is_best:
                            self.best_val_loss = vl
                        self.save_checkpoint(self.global_step, vl, is_best)
                        logger.info(f"  Val loss: {vl:.4f}"
                                    f"{' ★ new best!' if is_best else ''}")

                    if self.global_step > 0 and self.global_step % self.cfg.test_every_n_steps == 0:
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
            self.save_checkpoint(self.global_step, vl, is_best)
            logger.info(f"Epoch {epoch+1} val loss: {vl:.4f}"
                        f"{' ★ new best!' if is_best else ''}")

            self.metrics.plot(
                os.path.join(self.cfg.output_dir, "training_metrics.png"))

        # ── Final save ──
        logger.info("\n" + "=" * 70)
        logger.info("Training complete!")
        save_lora_state(self.lora_layers,
                        os.path.join(self.cfg.output_dir, "checkpoints",
                                     "final_lora_adapter.pt"))

        # Merge and export
        logger.info("Merging LoRA → base model …")
        merge_lora(self.lora_layers)
        merged_dir = os.path.join(self.cfg.output_dir, "merged_model")
        os.makedirs(merged_dir, exist_ok=True)
        torch.save(self.model.t3.state_dict(),
                    os.path.join(merged_dir, "t3_mtl23ls_v2.pt"))
        logger.info(f"Merged model → {merged_dir}")
        logger.info("Run `python fix_merged_model.py` to convert to safetensors")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("All done ✓")


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

        # Try loading merged model first
        merged = os.path.join(self.model_dir, "merged_model")
        if os.path.isdir(merged) and os.path.exists(
                os.path.join(merged, "t3_mtl23ls_v2.safetensors")):
            logger.info("Loading merged fine-tuned model …")
            self.model = ChatterboxMultilingualTTS.from_local(merged, device=self.device)
        else:
            # Load base + LoRA
            logger.info("Loading base model + LoRA adapter …")
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            if use_lora:
                lora_path = os.path.join(
                    self.model_dir, "checkpoints", "best_lora_adapter.pt")
                if os.path.exists(lora_path):
                    layers = inject_lora(
                        self.model.t3,
                        TrainingConfig().target_modules,
                        TrainingConfig().lora_rank,
                        TrainingConfig().lora_alpha,
                        TrainingConfig().lora_dropout,
                    )
                    load_lora_state(layers, lora_path, self.device)
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

    if args.mode == "prepare-only":
        prepare_dataset(config)
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
        best = os.path.join(config.output_dir, "checkpoints", "best_model.pt")
        if os.path.exists(best):
            trainer.setup_optimizer()
            trainer.load_checkpoint(best)
        vl = trainer.validate()
        logger.info(f"Evaluation loss: {vl:.4f}")
        trainer.generate_samples(step=0)


if __name__ == "__main__":
    main()
