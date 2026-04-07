#!/usr/bin/env python3
"""
Cross-Lingual Voice Cloning Evaluation Pipeline
================================================
Chatterbox base vs. LoRA fine-tuned model evaluation.

Metrics (9 total):
  Content:  WER, chrF++
  Speaker:  ECAPA Cosine Similarity, EER
  Quality:  PESQ, STOI, MCD, UTMOS
  Prosody:  Pitch Correlation (F0)

ASR: faster-whisper large-v3 (CTranslate2 backend)

Usage:
  python eval.py --skip-lora --output-dir ./eval_base   # baseline
  python eval.py --output-dir ./eval_lora                # LoRA
"""

import os
import sys
import math
import csv
import json
import argparse
import warnings
import tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")


# ===================================================================
# LoRA helpers (must match training code)
# ===================================================================

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        self.original_layer = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = original.in_features, original.out_features
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.lora_dropout = (nn.Dropout(dropout) if dropout > 0
                             else nn.Identity())
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original_layer(x)
        lora = (self.lora_dropout(x) @ self.lora_A.T
                @ self.lora_B.T * self.scaling)
        return base + lora


# ===================================================================
# Audio I/O helpers
# ===================================================================

def save_temp_wav(audio_array: np.ndarray, sr: int,
                  prefix: str = "eval") -> str:
    """Save numpy audio to a temp wav file. Returns path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path


def load_audio_16k(path: str) -> Tuple[np.ndarray, int]:
    """Load audio and resample to 16kHz. Returns (numpy_array, 16000)."""
    wav, sr = librosa.load(path, sr=16000)
    return wav, 16000


def load_audio_tensor_16k(path: str, device: str = "cpu") -> torch.Tensor:
    """Load audio as torch tensor at 16kHz. Shape: [1, T]."""
    signal, fs = torchaudio.load(path)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.to(device)


# ===================================================================
# Metric functions (each returns Optional[float] — None on failure)
# ===================================================================

def compute_pesq(ref_path: str, syn_path: str) -> Optional[float]:
    """PESQ: Perceptual speech quality. Range: -0.5 to 4.5, higher=better."""
    try:
        from pesq import pesq as pesq_fn
        ref, _ = load_audio_16k(ref_path)
        syn, _ = load_audio_16k(syn_path)
        min_len = min(len(ref), len(syn))
        if min_len < 8000:  # <0.5s too short for PESQ
            return None
        return float(pesq_fn(16000, ref[:min_len], syn[:min_len], "wb"))
    except Exception:
        return None


def compute_stoi(ref_path: str, syn_path: str) -> Optional[float]:
    """STOI: Short-Time Objective Intelligibility. Range: 0-1, higher=better."""
    try:
        from pystoi import stoi as stoi_fn
        ref, _ = load_audio_16k(ref_path)
        syn, _ = load_audio_16k(syn_path)
        min_len = min(len(ref), len(syn))
        if min_len < 8000:
            return None
        return float(stoi_fn(ref[:min_len], syn[:min_len], 16000,
                             extended=False))
    except Exception:
        return None


def compute_mcd(ref_path: str, syn_path: str) -> Optional[float]:
    """MCD: Mel Cepstral Distortion. Lower=better. Good range: 3-10."""
    try:
        from pymcd.mcd import Calculate_MCD
        mcd_calc = Calculate_MCD(MCD_mode="plain")
        return float(mcd_calc.calculate_mcd(ref_path, syn_path))
    except Exception:
        return None


def compute_pitch_correlation(ref_path: str,
                              syn_path: str) -> Optional[float]:
    """F0 Pitch Correlation: Pearson corr of pitch contours on voiced frames.
    
    Range: -1 to 1, closer to 1 = better prosody preservation.
    Uses pyin for robust F0 extraction.
    """
    try:
        ref, _ = load_audio_16k(ref_path)
        syn, _ = load_audio_16k(syn_path)

        f0_ref, voiced_ref, _ = librosa.pyin(
            ref, fmin=50, fmax=500, sr=16000)
        f0_syn, voiced_syn, _ = librosa.pyin(
            syn, fmin=50, fmax=500, sr=16000)

        min_len = min(len(f0_ref), len(f0_syn))
        f0_ref = f0_ref[:min_len]
        f0_syn = f0_syn[:min_len]
        voiced_ref = voiced_ref[:min_len]
        voiced_syn = voiced_syn[:min_len]

        both_voiced = voiced_ref & voiced_syn
        if np.sum(both_voiced) < 10:
            return None

        corr = np.corrcoef(f0_ref[both_voiced],
                           f0_syn[both_voiced])[0, 1]
        return float(corr) if not np.isnan(corr) else None
    except Exception:
        return None


# ===================================================================
# UTMOS: Neural MOS predictor (non-intrusive — no reference needed)
# ===================================================================

def load_utmos(device: str = "cuda"):
    """Load UTMOS model via torch.hub. Returns predictor or None."""
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
            trust_repo=True
        )
        predictor = predictor.to(device)
        predictor.eval()
        print("   UTMOS loaded ✓")
        return predictor
    except Exception as e:
        print(f"   ⚠ UTMOS failed to load: {e}. Skipping UTMOS.")
        return None


def compute_utmos(syn_path: str, predictor,
                  device: str = "cuda") -> Optional[float]:
    """UTMOS: Neural Mean Opinion Score. Range: 1-5, higher=better.
    
    Non-intrusive — evaluates synthesized audio quality without reference.
    """
    if predictor is None:
        return None
    try:
        wav = load_audio_tensor_16k(syn_path, device=device)
        with torch.no_grad():
            score = predictor(wav, sr=16000)
        return float(score.item())
    except Exception:
        return None


# ===================================================================
# Speaker model & EER
# ===================================================================

def load_speaker_model(device: str = "cuda"):
    """Load ECAPA-TDNN speaker verification model."""
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )


def extract_speaker_embedding(wav_path: str, verifier,
                               device: str = "cuda") -> Optional[torch.Tensor]:
    """Extract ECAPA-TDNN speaker embedding. Returns [192] tensor or None."""
    try:
        signal = load_audio_tensor_16k(wav_path, device=device)
        embedding = verifier.encode_batch(signal)
        return embedding.squeeze().cpu()  # [192]
    except Exception:
        return None


def compute_eer(genuine_scores: List[float],
                impostor_scores: List[float]) -> Optional[float]:
    """Compute Equal Error Rate from genuine/impostor similarity scores.
    
    Returns EER as a fraction (0-1). Lower = better speaker preservation.
    """
    if len(genuine_scores) < 5 or len(impostor_scores) < 5:
        return None
    try:
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import roc_curve

        labels = ([1] * len(genuine_scores)
                  + [0] * len(impostor_scores))
        scores = list(genuine_scores) + list(impostor_scores)

        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x),
                      0.0, 1.0)
        return float(eer)
    except Exception:
        return None


# ===================================================================
# Aggregation helpers
# ===================================================================

def safe_mean(values: list) -> float:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else float("nan")


def safe_std(values: list) -> float:
    valid = [v for v in values if v is not None]
    return float(np.std(valid)) if len(valid) > 1 else 0.0


def safe_count(values: list) -> int:
    return sum(1 for v in values if v is not None)


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Lingual Voice Cloning Evaluation")
    parser.add_argument("--dataset",
                        default="amanuelbyte/acl-voice-cloning-fr-expandedtry")
    parser.add_argument("--repo-id",
                        default="amanuelbyte/chatterbox-fr-lora")
    parser.add_argument("--lora-file",
                        default="best_lora_adapter.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3",
                        help="faster-whisper model size")
    parser.add_argument("--cfg-weight", type=float, default=0.0,
                        help="Chatterbox CFG weight (0.0 recommended cross-lingual)")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Evaluate base model (no LoRA)")
    parser.add_argument("--skip-utmos", action="store_true",
                        help="Skip UTMOS computation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    mode = "BASE" if args.skip_lora else "LoRA"

    print("=" * 60)
    print(f"  CROSS-LINGUAL VOICE CLONING EVALUATION — {mode}")
    print("=" * 60)

    # ==============================================================
    # PHASE 1: Load dataset
    # ==============================================================
    print(f"\n📥 Phase 1: Loading test data from {args.dataset}")
    ds_test = load_dataset(
        args.dataset,
        data_files={"test": "data/test-*.parquet"},
        split="test",
        cache_dir=args.cache_dir
    )
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)
    print(f"   Samples: {total}")

    # ==============================================================
    # PHASE 2: Load Chatterbox + optional LoRA
    # ==============================================================
    print(f"\n🔧 Phase 2: Loading Chatterbox model ({mode})")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if not args.skip_lora:
        lora_path = hf_hub_download(
            repo_id=args.repo_id, filename=args.lora_file)
        payload = torch.load(lora_path, map_location=device, weights_only=True)
        # Read config from self-describing adapter
        lora_cfg = payload.get("config", {})
        targets = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        rank = lora_cfg.get("rank", 8)
        alpha = lora_cfg.get("alpha", 16.0)
        dropout = lora_cfg.get("dropout", 0.1)
        lora_count = 0
        for name, module in model.t3.named_modules():
            if isinstance(module, nn.Linear) and any(
                    k in name for k in targets):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.t3.get_submodule(parent_name)
                setattr(parent, child_name,
                        LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout))
                lora_count += 1
        # Load with proper state_dict keys
        lora_sd = payload.get("lora_state_dict", payload)
        current_sd = model.t3.state_dict()
        loaded = 0
        for key, value in lora_sd.items():
            if key in current_sd:
                current_sd[key] = value.to(device)
                loaded += 1
        model.t3.load_state_dict(current_sd, strict=False)
        print(f"   LoRA injected: {lora_count} layers, {loaded} tensors loaded ✓")

    model.t3.eval()
    print(f"   Sample rate: {model.sr} Hz")

    # ==============================================================
    # PHASE 3: Generate audio
    # ==============================================================
    print(f"\n🎙  Phase 3: Generating {total} audio samples")
    samples = []
    skipped = 0

    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        # Support both 'amanuelbyte' and 'ymoslem' schemas
        text_fr = (row.get("trg_fr_text") or row.get("text_fr") or "").strip()
        text_en = (row.get("ref_en_text") or row.get("text_en") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get("ref_fr_voice") or row.get("audio_en") or row.get("audio")
        gt_data = row.get("trg_fr_voice")

        if not ref_data or not text_fr:
            print(f"\n   ⚠ Debug: sample {i} skipped! text_fr is {bool(text_fr)}, ref_data is {bool(ref_data)}")
            skipped += 1
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"], "ref_")

        gt_path = None
        if gt_data is not None:
             gt_path = save_temp_wav(
                 np.asarray(gt_data["array"], dtype=np.float32),
                 gt_data["sampling_rate"], "gt_")

        try:
            with torch.inference_mode():
                wav = model.generate(
                    text_fr,
                    audio_prompt_path=ref_path,
                    language_id="fr",
                    cfg_weight=args.cfg_weight
                )
        except Exception as e:
            print(f"\n   ⚠ Generation failed sample {i}: {e}")
            if ref_path and os.path.exists(ref_path): os.remove(ref_path)
            if gt_path and os.path.exists(gt_path): os.remove(gt_path)
            skipped += 1
            continue

        # Keep ref_path for speaker similarity in Phase 5

        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        wav_out = wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu()
        torchaudio.save(syn_path, wav_out, model.sr)

        samples.append({
            "idx": i,
            "syn_path": syn_path,
            "gt_path": gt_path,
            "ref_path": ref_path,
            "text_fr": text_fr,
            "text_en": text_en,
            "speaker_id": row.get("speaker_id", "unknown"),
        })

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")

    if not samples:
        print("❌ No samples generated. Exiting.")
        sys.exit(1)

    # Free generation model GPU memory
    del model
    torch.cuda.empty_cache()

    # ==============================================================
    # PHASE 4: ASR transcription (faster-whisper)
    # ==============================================================
    print(f"\n🗣  Phase 4: Transcribing with faster-whisper "
          f"{args.whisper_model}")
    from faster_whisper import WhisperModel as FasterWhisperModel

    whisper = FasterWhisperModel(
        args.whisper_model,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    transcripts = []
    for s in tqdm(samples, desc="Transcribing"):
        try:
            segments, info = whisper.transcribe(
                s["syn_path"],
                language="fr",
                beam_size=args.whisper_beam,
                vad_filter=True
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"\n   ⚠ ASR failed sample {s['idx']}: {e}")
            text = ""
        transcripts.append(text)

    del whisper
    torch.cuda.empty_cache()
    print(f"   Transcribed: {len(transcripts)} samples")

    # ==============================================================
    # PHASE 5: Acoustic metrics + speaker embeddings
    # ==============================================================
    print(f"\n📊 Phase 5: Acoustic metrics & speaker embeddings")

    verifier = load_speaker_model(device=device)
    utmos_predictor = (None if args.skip_utmos
                       else load_utmos(device=device))

    import jiwer
    import sacrebleu

    wer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    results = []
    # Storage for EER computation
    synth_embeddings = []  # (embedding_tensor, speaker_id)
    gt_embeddings = []     # (embedding_tensor, speaker_id)

    for s, tx in tqdm(
            zip(samples, transcripts),
            total=len(samples), desc="Computing metrics"):

        gt_path = s["gt_path"]
        ref_path = s["ref_path"]
        syn_path = s["syn_path"]
        spk = s["speaker_id"]

        # Quality metrics (Require Ground Truth)
        if gt_path is not None:
            pesq_val = compute_pesq(gt_path, syn_path)
            stoi_val = compute_stoi(gt_path, syn_path)
            mcd_val = compute_mcd(gt_path, syn_path)
            pitch_val = compute_pitch_correlation(gt_path, syn_path)
        else:
            pesq_val = stoi_val = mcd_val = pitch_val = None
            
        utmos_val = compute_utmos(syn_path, utmos_predictor, device)

        # --- Speaker embeddings (using original English reference) ---
        syn_emb = extract_speaker_embedding(syn_path, verifier, device)
        ref_emb = extract_speaker_embedding(ref_path, verifier, device)

        if syn_emb is not None and ref_emb is not None:
            sim = float(F.cosine_similarity(
                syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
            synth_embeddings.append((syn_emb, spk))
            gt_embeddings.append((ref_emb, spk))
        else:
            sim = None

        # --- WER ---
        try:
            if tx.strip():
                ref_clean = wer_transforms(s["text_fr"])
                hyp_clean = wer_transforms(tx)
                
                if isinstance(ref_clean, list) and len(ref_clean) > 0 and isinstance(ref_clean[0], list):
                    ref_clean = " ".join(ref_clean[0])
                if isinstance(hyp_clean, list) and len(hyp_clean) > 0 and isinstance(hyp_clean[0], list):
                    hyp_clean = " ".join(hyp_clean[0])
                elif isinstance(ref_clean, list):
                    ref_clean = " ".join(ref_clean)
                if isinstance(hyp_clean, list):
                    hyp_clean = " ".join(hyp_clean)

                w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
            else:
                w = 1.0
        except Exception as e:
            print(f"WER Failed: {e}")
            w = None

        # --- chrF++ ---
        try:
            chr_val = float(
                sacrebleu.sentence_chrf(tx, [s["text_fr"]]).score)
        except Exception:
            chr_val = None

        # Cleanup temp files
        if gt_path is not None and os.path.exists(gt_path):
            os.remove(gt_path)
        if os.path.exists(ref_path):
            os.remove(ref_path)

        results.append({
            "idx": s["idx"],
            "speaker": spk,
            "WER": w,
            "chrF": chr_val,
            "PESQ": pesq_val,
            "STOI": stoi_val,
            "MCD": mcd_val,
            "UTMOS": utmos_val,
            "PitchCorr": pitch_val,
            "Similarity": sim,
            "transcript": tx,
            "reference": s["text_fr"],
        })

    del verifier
    if utmos_predictor is not None:
        del utmos_predictor
    torch.cuda.empty_cache()

    # ==============================================================
    # PHASE 6: EER computation
    # ==============================================================
    print(f"\n🔐 Phase 6: Computing EER from {len(synth_embeddings)} "
          f"embeddings")

    genuine_scores = []
    impostor_scores = []

    for i, (syn_emb_i, spk_i) in enumerate(synth_embeddings):
        for j, (gt_emb_j, spk_j) in enumerate(gt_embeddings):
            sim_ij = float(F.cosine_similarity(
                syn_emb_i.unsqueeze(0),
                gt_emb_j.unsqueeze(0)
            ).item())
            if spk_i == spk_j:
                genuine_scores.append(sim_ij)
            else:
                impostor_scores.append(sim_ij)

    eer_value = compute_eer(genuine_scores, impostor_scores)
    print(f"   Genuine trials:  {len(genuine_scores)}")
    print(f"   Impostor trials: {len(impostor_scores)}")
    print(f"   EER: {eer_value:.4f}" if eer_value is not None
          else "   EER: could not compute")

    # ==============================================================
    # PHASE 7: Summary & export
    # ==============================================================
    print(f"\n📋 Phase 7: Generating summary")

    metric_keys = ["WER", "chrF", "Similarity",
                   "PESQ", "STOI", "MCD", "UTMOS", "PitchCorr"]

    # Overall stats
    overall = {}
    for k in metric_keys:
        vals = [r[k] for r in results]
        overall[k] = {
            "mean": safe_mean(vals),
            "std": safe_std(vals),
            "valid": safe_count(vals),
            "total": len(vals),
        }

    # Per-speaker stats
    speakers = sorted(set(r["speaker"] for r in results))
    per_speaker = {}
    for spk in speakers:
        spk_r = [r for r in results if r["speaker"] == spk]
        per_speaker[spk] = {"count": len(spk_r)}
        for k in metric_keys:
            vals = [r[k] for r in spk_r]
            per_speaker[spk][k] = {
                "mean": safe_mean(vals),
                "std": safe_std(vals),
                "valid": safe_count(vals),
            }

    # Build output
    output = {
        "config": {
            "mode": mode,
            "dataset": args.dataset,
            "lora_file": (args.lora_file if not args.skip_lora
                          else None),
            "whisper_model": args.whisper_model,
            "num_evaluated": len(results),
            "num_skipped": skipped,
        },
        "overall": {k: v["mean"] for k, v in overall.items()},
        "overall_std": {k: v["std"] for k, v in overall.items()},
        "valid_counts": {k: v["valid"] for k, v in overall.items()},
        "EER": eer_value,
        "eer_details": {
            "genuine_trials": len(genuine_scores),
            "impostor_trials": len(impostor_scores),
        },
        "per_speaker": per_speaker,
        "results": results,
    }

    # Save JSON
    json_path = os.path.join(args.output_dir, f"eval_{mode.lower()}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"eval_{mode.lower()}.csv")
    csv_fields = (["idx", "speaker"] + metric_keys
                  + ["transcript", "reference"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields,
                                extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # ── Print summary ─────────────────────────────────────────
    direction = {
        "WER": "↓", "chrF": "↑",
        "Similarity": "↑", "PESQ": "↑", "STOI": "↑",
        "MCD": "↓", "UTMOS": "↑", "PitchCorr": "↑",
    }

    print("\n" + "=" * 64)
    print(f"  EVALUATION COMPLETE — {mode} MODEL")
    print(f"  ASR: faster-whisper {args.whisper_model}")
    print(f"  Samples: {len(results)} evaluated, {skipped} skipped")
    print("=" * 64)
    print(f"  {'Metric':<14} {'Dir':>3} {'Mean':>9} {'± Std':>9}"
          f"  {'Valid':>6}")
    print("-" * 64)
    for k in metric_keys:
        d = direction.get(k, " ")
        m = overall[k]["mean"]
        s = overall[k]["std"]
        v = overall[k]["valid"]
        t = overall[k]["total"]
        m_str = f"{m:.4f}" if not np.isnan(m) else "  N/A"
        s_str = f"±{s:.4f}" if not np.isnan(m) else "     "
        print(f"  {k:<14} {d:>3} {m_str:>9} {s_str:>9}  {v:>3}/{t}")

    print("-" * 64)
    if eer_value is not None:
        print(f"  {'EER':<14} {'↓':>3} {eer_value:>9.4f}"
              f"           ({len(genuine_scores)}g/"
              f"{len(impostor_scores)}i)")
    else:
        print(f"  {'EER':<14} {'↓':>3}       N/A")
    print("=" * 64)

    # Per-speaker
    print(f"\n  Per-Speaker Breakdown:")
    for spk in speakers:
        n = per_speaker[spk]["count"]
        print(f"\n  ┌─ Speaker: {spk} ({n} samples)")
        for k in metric_keys:
            v = per_speaker[spk][k]["mean"]
            c = per_speaker[spk][k]["valid"]
            v_str = f"{v:.4f}" if not np.isnan(v) else "N/A"
            print(f"  │  {k:<14}: {v_str:>9}  ({c} valid)")
        print(f"  └─")

    print(f"\n✅ JSON  → {json_path}")
    print(f"✅ CSV   → {csv_path}")
    print(f"✅ Audio → {args.output_dir}/synth_*.wav")


if __name__ == "__main__":
    main()
