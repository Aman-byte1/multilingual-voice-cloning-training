#!/usr/bin/env python3
"""
Chatterbox Raw HuggingFace — Multilingual Voice Cloning Evaluation
==================================================================
Evaluates the upstream `resemble-ai/chatterbox` model (zero-shot, no LoRA)
across dynamic cross-lingual targets leveraging CER tracking.

Metrics (4 total):
  Content:  WER, CER
  Speaker:  ECAPA Cosine Similarity
  Timing:   Inference Latency (s), Real-Time Factor (RTF)

ASR: faster-whisper large-v3
"""

import os
import sys
import csv
import json
import time
import argparse
import warnings
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from eval import (
    save_temp_wav,
    load_speaker_model,
    extract_speaker_embedding,
    safe_mean,
    safe_std,
    safe_count,
)

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")

MODEL_ID = "resemble-ai/chatterbox"

def audio_duration_s(wav_tensor: torch.Tensor, sr: int) -> float:
    n_samples = wav_tensor.numel()
    return n_samples / sr

def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox Raw (HuggingFace) Evaluation — uniform pipeline")
    parser.add_argument("--dataset",
                        default="ymoslem/acl-6060",
                        help="HuggingFace dataset repo")
    parser.add_argument("--split", default="eval",
                        help="Dataset split to evaluate")
    parser.add_argument("--data-files", default=None,
                        help="Optional glob for parquet shards")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of evaluated samples (stratified)")
    parser.add_argument("--exl", type=float, default=0.5,
                        help="Chatterbox exaggeration level (0–1)")
    parser.add_argument("--cfg-weight", type=float, default=0.0,
                        help="Chatterbox CFG weight (0.0 recommended for cross-lingual)")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="faster-whisper model size")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr",
                        help="Language hint for Whisper (ISO 639-1)")
    parser.add_argument("--output-dir", default="./eval_chatterbox_raw")
    parser.add_argument("--cache-dir",  default="./data_cache")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    target_lang = args.whisper_lang.strip().lower()

    wall_start = time.perf_counter()

    print("=" * 64)
    print("  CHATTERBOX RAW (HuggingFace) — ZERO-SHOT EVALUATION")
    print(f"  Model : {MODEL_ID}")
    print(f"  Device: {device}")
    print(f"  Target: {target_lang}")
    print("=" * 64)

    # ── PHASE 1 · Dataset
    print(f"\n📥 Phase 1: Loading '{args.split}' split from {args.dataset}")
    load_kwargs: Dict = dict(cache_dir=args.cache_dir)
    if args.data_files:
        load_kwargs["data_files"] = {args.split: args.data_files}
    ds_test = load_dataset(args.dataset, split=args.split, **load_kwargs)

    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[: args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)
    print(f"   Samples to evaluate: {total}")

    # ── PHASE 2 · Load Chatterbox
    print(f"\n🔧 Phase 2: Loading Chatterbox from '{MODEL_ID}'")
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except Exception as e:
        sys.exit(f"❌  Failed to import ChatterboxMultilingualTTS. The error was:\n{e}")

    torch.backends.cudnn.enabled = False
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model_sr = model.sr
    print(f"   Sample rate : {model_sr} Hz")
    print(f"   Exaggeration: {args.exl}  |  CFG weight: {args.cfg_weight}")

    # ── PHASE 3 · Generate audio
    print(f"\n🎙  Phase 3: Generating {total} audio samples")
    samples: List[Dict] = []
    skipped = 0
    inference_times: List[float] = []
    audio_durations:  List[float] = []

    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        
        # Dynamic Schema Support
        text_target = (row.get(f"trg_{target_lang}_text") or row.get(f"text_{target_lang}") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get(f"ref_{target_lang}_voice") or row.get("audio_en") or row.get("audio")

        if not ref_data or not text_target:
            print(f"\n   ⚠ Debug: sample {i} skipped! text_{target_lang} is {bool(text_target)}, ref is {bool(ref_data)}")
            skipped += 1
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"], "ref_")

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                wav = model.generate(
                    text_target,
                    language_id=target_lang,
                    audio_prompt_path=ref_path,
                    exaggeration=args.exl,
                    cfg_weight=args.cfg_weight,
                )
            t1 = time.perf_counter()
        except Exception as e:
            tqdm.write(f"\n   ⚠ Generation failed sample {i}: {e}")
            if ref_path and os.path.exists(ref_path): os.remove(ref_path)
            skipped += 1
            continue

        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        wav_out = wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu()
        torchaudio.save(syn_path, wav_out, model_sr)

        elapsed   = t1 - t0
        audio_dur = audio_duration_s(wav_out, model_sr)
        inference_times.append(elapsed)
        audio_durations.append(audio_dur)

        samples.append({
            "idx":          i,
            "syn_path":     syn_path,
            "ref_path":     ref_path,
            "text_target":  text_target,
            "speaker_id":   row.get("speaker_id", "unknown"),
            "inference_s":  elapsed,
            "audio_dur_s":  audio_dur,
            "rtf":          elapsed / audio_dur if audio_dur > 0 else None,
        })

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")
    if not samples:
        print("❌ No samples generated. Exiting.")
        sys.exit(1)

    del model
    torch.cuda.empty_cache()

    # ── PHASE 4 · ASR transcription
    print(f"\n🗣  Phase 4: Transcribing with faster-whisper {args.whisper_model}")
    from faster_whisper import WhisperModel as FasterWhisperModel

    whisper = FasterWhisperModel(
        args.whisper_model,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    transcripts: List[str] = []
    for s in tqdm(samples, desc="Transcribing"):
        try:
            segments, _ = whisper.transcribe(
                s["syn_path"],
                language=target_lang,
                beam_size=args.whisper_beam,
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            tqdm.write(f"\n   ⚠ ASR failed sample {s['idx']}: {e}")
            text = ""
        transcripts.append(text)

    del whisper
    torch.cuda.empty_cache()
    print(f"   Transcribed: {len(transcripts)} samples")

    # ── PHASE 5 · Acoustic metrics & embedding
    print(f"\n📊 Phase 5: Acoustic metrics & speaker embeddings")
    verifier = load_speaker_model(device=device)

    import jiwer
    wer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    results: List[Dict] = []
    
    for s, tx in tqdm(
            zip(samples, transcripts),
            total=len(samples), desc="Computing metrics"):

        ref_path = s["ref_path"]
        syn_path = s["syn_path"]
        spk      = s["speaker_id"]

        syn_emb = extract_speaker_embedding(syn_path, verifier, device)
        ref_emb = extract_speaker_embedding(ref_path,  verifier, device)
        if syn_emb is not None and ref_emb is not None:
            sim = float(F.cosine_similarity(
                syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
        else:
            sim = None

        # WER & CER
        try:
            if tx.strip():
                ref_clean = wer_transforms(s["text_target"])
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
                c = float(jiwer.cer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
            else:
                w = 1.0
                c = 1.0
        except Exception as e:
            print(f"Metrics Failed: {e}")
            w = None
            c = None

        if os.path.exists(ref_path):
            os.remove(ref_path)

        results.append({
            "idx":         s["idx"],
            "speaker":     spk,
            "WER":         w,
            "CER":         c,
            "Similarity":  sim,
            "InferenceS":  s["inference_s"],
            "AudioDurS":   s["audio_dur_s"],
            "RTF":         s["rtf"],
            "transcript":  tx,
            "reference":   s["text_target"],
        })

    del verifier
    torch.cuda.empty_cache()

    # ── PHASE 6 · Summary & export
    print(f"\n📋 Phase 6: Generating summary")

    wall_total   = time.perf_counter() - wall_start
    mean_inf     = safe_mean([r["InferenceS"] for r in results])
    mean_dur     = safe_mean([r["AudioDurS"]  for r in results])
    mean_rtf     = safe_mean([r["RTF"]        for r in results])
    total_inf    = sum(r["InferenceS"] for r in results)

    metric_keys = ["WER", "CER", "Similarity", "InferenceS", "AudioDurS", "RTF"]

    overall: Dict = {}
    for k in metric_keys:
        vals = [r[k] for r in results]
        overall[k] = {
            "mean":  safe_mean(vals),
            "std":   safe_std(vals),
            "valid": safe_count(vals),
            "total": len(vals),
        }

    output = {
        "config": {
            "model":          MODEL_ID,
            "mode":           "raw",
            "dataset":        args.dataset,
            "split":          args.split,
            "whisper_model":  args.whisper_model,
            "language":       target_lang,
            "exaggeration":   args.exl,
            "cfg_weight":     args.cfg_weight,
            "num_evaluated":  len(results),
            "num_skipped":    skipped,
        },
        "overall":      {k: v["mean"]  for k, v in overall.items()},
        "overall_std":  {k: v["std"]   for k, v in overall.items()},
        "valid_counts": {k: v["valid"] for k, v in overall.items()},
        "timing_summary": {
            "wall_time_s":        round(wall_total, 2),
            "total_inference_s":  round(total_inf,  2),
            "mean_inference_s":   round(mean_inf,   4),
            "mean_audio_dur_s":   round(mean_dur,   4),
            "mean_rtf":           round(mean_rtf,   4),
        },
        "results":     results,
    }

    json_path = os.path.join(args.output_dir, "eval_chatterbox_raw.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    csv_path   = os.path.join(args.output_dir, "eval_chatterbox_raw.csv")
    csv_fields = (["idx", "speaker"] + metric_keys + ["transcript", "reference"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    direction = {
        "WER":        "↓", "CER":        "↓",
        "Similarity": "↑", "InferenceS": "↓",
        "AudioDurS":  " ", "RTF":        "↓",
    }
    label = {
        "InferenceS": "Inf. Time (s)",
        "AudioDurS":  "Audio Dur.(s)",
        "RTF":        "RTF",
    }

    print("\n" + "=" * 62)
    print("  CHATTERBOX RAW — EVALUATION COMPLETE")
    print(f"  ASR: faster-whisper {args.whisper_model} ({target_lang})")
    print(f"  Samples: {len(results)} evaluated, {skipped} skipped")
    print("=" * 62)
    print(f"  {'Metric':<16} {'Dir':>3} {'Mean':>9} {'± Std':>9}  {'Valid':>6}")
    print("-" * 62)
    for k in metric_keys:
        d    = direction.get(k, " ")
        name = label.get(k, k)
        m    = overall[k]["mean"]
        s    = overall[k]["std"]
        v    = overall[k]["valid"]
        t    = overall[k]["total"]
        m_s  = f"{m:.4f}" if not np.isnan(m) else "   N/A"
        s_s  = f"±{s:.4f}" if not np.isnan(m) else "      "
        print(f"  {name:<16} {d:>3} {m_s:>9} {s_s:>9}  {v:>3}/{t}")

    print("=" * 62)
    print("\n  ⏱  Timing Summary")
    print(f"  {'Wall-clock time':<22}: {wall_total:.1f} s  ({wall_total/60:.1f} min)")
    print(f"  {'Mean audio duration':<22}: {mean_dur:.3f} s")
    print(f"  {'Mean RTF':<22}: {mean_rtf:.4f}")

    print(f"\n✅ JSON  → {json_path}")
    print(f"✅ CSV   → {csv_path}")

if __name__ == "__main__":
    main()
