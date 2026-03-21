#!/usr/bin/env python3
"""
End-to-end evaluation of the Chatterbox FR LoRA fine-tuned model.

Pipeline:
  1. Downloads the TEST split from `amanuelbyte/acl-voice-cloning-fr-expandedtry`
  2. Loads base ChatterboxMultilingualTTS + LoRA adapter from HuggingFace
  3. Runs inference on each test sample (cross-lingual: EN ref → FR synthesis)
  4. Computes three metrics comparing synthesized vs ground-truth FR audio:
     - Speaker Similarity (ECAPA-TDNN cosine similarity)
     - Mel Cepstral Distortion (MCD)
     - Perceptual Evaluation of Speech Quality (PESQ)
     - Word Error Rate (WER) using Whisper
  5. Outputs per-sample CSV + aggregate summary

Prerequisites:
    pip install pesq pymcd speechbrain librosa datasets huggingface_hub torchaudio jiwer transformers

Usage:
    python evaluation/evaluate_metrics.py
    python evaluation/evaluate_metrics.py --max-samples 100
    python evaluation/evaluate_metrics.py --lora-file final_lora_adapter.pt
"""

import os
import sys
import math
import csv
import json
import argparse
import warnings
import tempfile
from typing import List, Optional

import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LoRA helpers (must match training code exactly)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
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
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original_layer(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora


def inject_lora(model, targets, rank, alpha, dropout=0.0):
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
    return layers


def load_lora_state(layers, path, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=True)
    for i, layer in enumerate(layers):
        if f"layer_{i}_A" in state:
            layer.lora_A.data = state[f"layer_{i}_A"].to(device)
            layer.lora_B.data = state[f"layer_{i}_B"].to(device)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_pesq_score(ref_wav_path: str, synth_wav_path: str) -> Optional[float]:
    """PESQ wideband (16 kHz)."""
    try:
        from pesq import pesq as pesq_fn, NoUtterancesError
        ref, _ = librosa.load(ref_wav_path, sr=16000)
        syn, _ = librosa.load(synth_wav_path, sr=16000)
        min_len = min(len(ref), len(syn))
        if min_len < 16000 * 0.5:
            return None
        return float(pesq_fn(16000, ref[:min_len], syn[:min_len], "wb"))
    except Exception:
        return None


def compute_mcd_score(ref_wav_path: str, synth_wav_path: str) -> Optional[float]:
    """Mel Cepstral Distortion."""
    try:
        from pymcd.mcd import Calculate_MCD
        mcd_calc = Calculate_MCD(MCD_mode="plain")
        return float(mcd_calc.calculate_mcd(ref_wav_path, synth_wav_path))
    except Exception:
        return None


def load_speaker_model():
    """Load ECAPA-TDNN speaker verification model."""
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError:
        from speechbrain.pretrained import SpeakerRecognition

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def compute_speaker_similarity(verifier, ref_wav_path: str, synth_wav_path: str) -> Optional[float]:
    """Cosine similarity of ECAPA-TDNN embeddings."""
    try:
        score, _ = verifier.verify_files(ref_wav_path, synth_wav_path)
        return float(score.item())
    except Exception:
        return None


def compute_wer_score(reference_text: str, synth_wav_path: str, asr_pipe) -> Optional[float]:
    """Word Error Rate using Whisper."""
    try:
        import jiwer
        import librosa
        # Bypass FFmpeg by loading the audio natively first
        audio, _ = librosa.load(synth_wav_path, sr=16000)
        transcription = asr_pipe(audio, generate_kwargs={"language": "french"})["text"]
        
        # Clean up text (lowercase, remove punctuation)
        ref_clean = jiwer.RemovePunctuation()(reference_text.lower())
        hyp_clean = jiwer.RemovePunctuation()(transcription.lower())
        
        if not ref_clean:
            return None
        return float(jiwer.wer(ref_clean, hyp_clean))
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Helpers — save numpy audio to a temp WAV
# ---------------------------------------------------------------------------

def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval") -> str:
    """Save a numpy array to a temporary WAV file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chatterbox FR LoRA on the HF test split")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-expandedtry",
                        help="HuggingFace dataset ID")
    parser.add_argument("--repo-id", default="amanuelbyte/chatterbox-fr-lora",
                        help="HuggingFace model repo for LoRA weights")
    parser.add_argument("--lora-file", default="best_lora_adapter.pt",
                        help="LoRA weight filename in the HF repo")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples to evaluate (useful for quick runs)")
    parser.add_argument("--output-dir", default="./eval_results",
                        help="Directory to save results CSV and summary")
    parser.add_argument("--cache-dir", default="./data_cache",
                        help="HF dataset cache directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the test split from HuggingFace
    # ------------------------------------------------------------------
    print(f"📥 Loading TEST split from {args.dataset} (skipping train split) ...")
    # Explictly specify data_files to prevent `datasets` from downloading the massive train split
    ds_test = load_dataset(
        args.dataset, 
        data_files={"test": "data/test-*.parquet"},
        split="test", 
        cache_dir=args.cache_dir
    )
    total = len(ds_test)
    print(f"   Test set size: {total} rows")

    if args.max_samples and args.max_samples < total:
        # Deterministic subsample
        indices = list(range(0, total, total // args.max_samples))[:args.max_samples]
        ds_test = ds_test.select(indices)
        total = len(ds_test)
        print(f"   Subsampled to {total} rows")

    # ------------------------------------------------------------------
    # 2. Load base model + LoRA adapter
    # ------------------------------------------------------------------
    print(f"\n🔧 Downloading LoRA weights '{args.lora_file}' from {args.repo_id} ...")
    lora_path = hf_hub_download(repo_id=args.repo_id, filename=args.lora_file)

    print("🔧 Loading base ChatterboxMultilingualTTS ...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    print("🔧 Injecting LoRA ...")
    targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layers = inject_lora(model.t3, targets=targets, rank=32, alpha=64.0, dropout=0.0)
    load_lora_state(layers, lora_path, device=device)
    if hasattr(model, 'eval'):
        model.eval()
    elif hasattr(model.t3, 'eval'):
        model.t3.eval()
    print(f"   LoRA injected into {len(layers)} layers ✓\n")

    # ------------------------------------------------------------------
    # 3. Load speaker verification model
    # ------------------------------------------------------------------
    print("🔧 Loading ECAPA-TDNN speaker verification model ...")
    try:
        verifier = load_speaker_model()
    except Exception as e:
        print(f"   ⚠ Could not load speaker model: {e}")
        verifier = None

    # ------------------------------------------------------------------
    # 4. Load Whisper ASR model for WER
    # ------------------------------------------------------------------
    print("🔧 Loading Whisper ASR model for WER ...")
    try:
        from transformers import pipeline
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=device, chunk_length_s=30)
    except Exception as e:
        print(f"   ⚠ Could not load Whisper ASR: {e}")
        asr_pipe = None

    # ------------------------------------------------------------------
    # 5. Run inference + metrics on each test sample
    # ------------------------------------------------------------------
    results = []
    pesq_scores, mcd_scores, spk_scores, wer_scores = [], [], [], []

    print(f"\n🚀 Running evaluation on {total} test samples ...\n")

    for i in tqdm(range(total), desc="Evaluating"):
        row = ds_test[i]
        text_fr = (row.get("trg_fr_text") or "").strip()
        speaker_id = row.get("speaker_id", "unknown")
        speaker_name = row.get("speaker_name", "unknown")

        if not text_fr or len(text_fr) < 2:
            continue

        temp_files = []
        try:
            # --- Get reference audio (English voice for cross-lingual) ---
            ref_data = row.get("ref_en_voice")
            if ref_data is None or not isinstance(ref_data, dict):
                ref_data = row.get("ref_fr_voice")
            if ref_data is None or not isinstance(ref_data, dict):
                continue

            ref_array = np.asarray(ref_data["array"], dtype=np.float32)
            ref_sr = ref_data["sampling_rate"]
            ref_wav_path = save_temp_wav(ref_array, ref_sr, prefix="ref_")
            temp_files.append(ref_wav_path)

            # --- Get ground-truth target French audio ---
            gt_data = row.get("trg_fr_voice")
            if gt_data is None or not isinstance(gt_data, dict):
                continue

            gt_array = np.asarray(gt_data["array"], dtype=np.float32)
            gt_sr = gt_data["sampling_rate"]
            gt_wav_path = save_temp_wav(gt_array, gt_sr, prefix="gt_")
            temp_files.append(gt_wav_path)

            # --- Run inference: generate French speech ---
            with torch.no_grad():
                wav = model.generate(
                    text_fr,
                    audio_prompt_path=ref_wav_path,
                    language_id="fr",
                )

            if not isinstance(wav, torch.Tensor):
                continue

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            synth_wav_path = os.path.join(
                args.output_dir, f"synth_{speaker_id}_{i:05d}.wav"
            )
            torchaudio.save(synth_wav_path, wav.cpu(), model.sr)

            # --- Compute metrics (synthesized vs ground-truth) ---
            p = compute_pesq_score(gt_wav_path, synth_wav_path)
            m = compute_mcd_score(gt_wav_path, synth_wav_path)
            s = compute_speaker_similarity(verifier, gt_wav_path, synth_wav_path) if verifier else None
            w = compute_wer_score(text_fr, synth_wav_path, asr_pipe) if asr_pipe else None

            if p is not None:
                pesq_scores.append(p)
            if m is not None:
                mcd_scores.append(m)
            if s is not None:
                spk_scores.append(s)
            if w is not None:
                wer_scores.append(w)

            results.append({
                "index": i,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "text": text_fr[:80],
                "PESQ": f"{p:.4f}" if p is not None else "N/A",
                "MCD": f"{m:.4f}" if m is not None else "N/A",
                "Speaker_Similarity": f"{s:.4f}" if s is not None else "N/A",
                "WER": f"{w:.4f}" if w is not None else "N/A",
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"   ⚠ OOM on sample {i}, skipping")
            else:
                print(f"   ⚠ Error on sample {i}: {e}")
            continue
        except Exception as e:
            print(f"   ⚠ Error on sample {i}: {e}")
            continue
        finally:
            # Clean up temp files
            for tf in temp_files:
                try:
                    os.remove(tf)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    # Per-sample CSV
    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\n📄 Per-sample results saved to: {csv_path}")

    # Summary
    summary = {
        "dataset": args.dataset,
        "split": "test",
        "lora_repo": args.repo_id,
        "lora_file": args.lora_file,
        "total_test_samples": total,
        "evaluated_samples": len(results),
        "metrics": {
            "PESQ": {
                "mean": float(np.mean(pesq_scores)) if pesq_scores else None,
                "std": float(np.std(pesq_scores)) if pesq_scores else None,
                "min": float(np.min(pesq_scores)) if pesq_scores else None,
                "max": float(np.max(pesq_scores)) if pesq_scores else None,
                "count": len(pesq_scores),
            },
            "MCD": {
                "mean": float(np.mean(mcd_scores)) if mcd_scores else None,
                "std": float(np.std(mcd_scores)) if mcd_scores else None,
                "min": float(np.min(mcd_scores)) if mcd_scores else None,
                "max": float(np.max(mcd_scores)) if mcd_scores else None,
                "count": len(mcd_scores),
            },
            "Speaker_Similarity": {
                "mean": float(np.mean(spk_scores)) if spk_scores else None,
                "std": float(np.std(spk_scores)) if spk_scores else None,
                "min": float(np.min(spk_scores)) if spk_scores else None,
                "max": float(np.max(spk_scores)) if spk_scores else None,
                "count": len(spk_scores),
            },
            "WER": {
                "mean": float(np.mean(wer_scores)) if wer_scores else None,
                "std": float(np.std(wer_scores)) if wer_scores else None,
                "min": float(np.min(wer_scores)) if wer_scores else None,
                "max": float(np.max(wer_scores)) if wer_scores else None,
                "count": len(wer_scores),
            },
        },
    }

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Dataset:    {args.dataset} (test split)")
    print(f"  Model:      {args.repo_id} / {args.lora_file}")
    print(f"  Evaluated:  {len(results)} / {total} samples")
    print("-" * 60)

    if pesq_scores:
        print(f"  PESQ:               {np.mean(pesq_scores):.4f} ± {np.std(pesq_scores):.4f}")
    else:
        print("  PESQ:               N/A (install: pip install pesq)")

    if mcd_scores:
        print(f"  MCD:                {np.mean(mcd_scores):.4f} ± {np.std(mcd_scores):.4f}")
    else:
        print("  MCD:                N/A (install: pip install pymcd)")

    if spk_scores:
        print(f"  Speaker Similarity: {np.mean(spk_scores):.4f} ± {np.std(spk_scores):.4f}")
    else:
        print("  Speaker Similarity: N/A (install: pip install speechbrain)")

    if wer_scores:
        print(f"  WER:                {np.mean(wer_scores):.4f} ± {np.std(wer_scores):.4f}")
    else:
        print("  WER:                N/A (install: pip install jiwer transformers)")

    print("=" * 60)
    print(f"\n📄 Full results:  {csv_path}")
    print(f"📄 Summary JSON:  {summary_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
