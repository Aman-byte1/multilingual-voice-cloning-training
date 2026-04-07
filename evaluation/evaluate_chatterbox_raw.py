#!/usr/bin/env python3
"""
Chatterbox Raw HuggingFace — Voice Cloning Evaluation
======================================================
Evaluates the upstream `resemble-ai/chatterbox` model (zero-shot, no LoRA)
as a fair baseline, using the identical metric pipeline as eval.py.

Metrics (10 total):
  Content:  WER, chrF++
  Speaker:  ECAPA Cosine Similarity, EER
  Quality:  PESQ, STOI, MCD, UTMOS
  Prosody:  Pitch Correlation (F0)
  Timing:   Inference Latency (s), Real-Time Factor (RTF)

ASR: faster-whisper large-v3 (CTranslate2 backend)

Usage:
  # Basic run — evaluates raw chatterbox on your test split
  python evaluate_chatterbox_raw.py \\
      --dataset amanuelbyte/acl-voice-cloning-fr-expandedtry \\
      --output-dir ./eval_chatterbox_raw

  # Limit samples (quick sanity check)
  python evaluate_chatterbox_raw.py --max-samples 20

  # Skip UTMOS if UTMOS predictor is unavailable
  python evaluate_chatterbox_raw.py --skip-utmos
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

# ── shared helpers from eval.py ──────────────────────────────────────────────
# We import directly so metrics stay 100 % identical across all eval scripts.
sys.path.insert(0, os.path.dirname(__file__))
from eval import (
    save_temp_wav,
    load_audio_16k,
    load_audio_tensor_16k,
    compute_pesq,
    compute_stoi,
    compute_mcd,
    compute_pitch_correlation,
    load_utmos,
    compute_utmos,
    load_speaker_model,
    extract_speaker_embedding,
    compute_eer,
    safe_mean,
    safe_std,
    safe_count,
)

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")

MODEL_ID = "resemble-ai/chatterbox"   # upstream HuggingFace checkpoint


# ── timing helpers ────────────────────────────────────────────────────────────

def audio_duration_s(wav_tensor: torch.Tensor, sr: int) -> float:
    """Return duration of a 1-D or [1,T] tensor in seconds."""
    n_samples = wav_tensor.numel()
    return n_samples / sr


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox Raw (HuggingFace) Evaluation — uniform pipeline")
    parser.add_argument("--dataset",
                        default="amanuelbyte/acl-voice-cloning-fr-expandedtry",
                        help="HuggingFace dataset repo")
    parser.add_argument("--split", default="test",
                        help="Dataset split to evaluate")
    parser.add_argument("--data-files", default=None,
                        help="Optional glob for parquet shards, e.g. 'data/test-*.parquet'")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of evaluated samples (stratified)")
    parser.add_argument("--exl", type=float, default=0.5,
                        help="Chatterbox exaggeration level (0–1)")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="Chatterbox CFG weight")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="faster-whisper model size")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr",
                        help="Language hint for Whisper (ISO 639-1)")
    parser.add_argument("--output-dir", default="./eval_chatterbox_raw")
    parser.add_argument("--cache-dir",  default="./data_cache")
    parser.add_argument("--skip-utmos", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    wall_start = time.perf_counter()

    print("=" * 64)
    print("  CHATTERBOX RAW (HuggingFace) — ZERO-SHOT EVALUATION")
    print(f"  Model : {MODEL_ID}")
    print(f"  Device: {device}")
    print("=" * 64)

    # ── PHASE 1 · Dataset ────────────────────────────────────────────────────
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

    # ── PHASE 2 · Load Chatterbox ────────────────────────────────────────────
    print(f"\n🔧 Phase 2: Loading Chatterbox from '{MODEL_ID}'")
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except Exception as e:
        sys.exit(f"❌  Failed to import ChatterboxMultilingualTTS. The error was:\n{e}\n\nPlease verify your chatterbox installation.")

    # Even though we are evaluating the raw model, we use the Multilingual 
    # class to support French (`language_id="fr"`) generation. It defaults to the correct HuggingFace repo.
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model_sr = model.sr
    print(f"   Sample rate : {model_sr} Hz")
    print(f"   Exaggeration: {args.exl}  |  CFG weight: {args.cfg_weight}")

    # ── PHASE 3 · Generate audio ──────────────────────────────────────────────
    print(f"\n🎙  Phase 3: Generating {total} audio samples")
    samples: List[Dict] = []
    skipped = 0

    # Timing accumulators
    inference_times: List[float] = []   # seconds of wall-clock per sample
    audio_durations:  List[float] = []   # seconds of generated audio per sample

    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        text_fr = (row.get("trg_fr_text") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get("ref_fr_voice")
        gt_data  = row.get("trg_fr_voice")

        if not ref_data or not gt_data or not text_fr:
            skipped += 1
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"], "ref_")
        gt_path = save_temp_wav(
            np.asarray(gt_data["array"],  dtype=np.float32),
            gt_data["sampling_rate"],  "gt_")

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                wav = model.generate(
                    text_fr,
                    audio_prompt_path=ref_path,
                    exaggeration=args.exl,
                    cfg_weight=args.cfg_weight,
                )
            t1 = time.perf_counter()
        except Exception as e:
            tqdm.write(f"\n   ⚠ Generation failed sample {i}: {e}")
            if os.path.exists(ref_path): os.remove(ref_path)
            if os.path.exists(gt_path): os.remove(gt_path)
            skipped += 1
            continue

        # Keep ref_path for speaker similarity in Phase 5
        # Persist synthesised audio
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
            "gt_path":      gt_path,
            "ref_path":     ref_path,
            "text_fr":      text_fr,
            "text_en":      (row.get("ref_en_text") or "").strip(),
            "speaker_id":   row.get("speaker_id", "unknown"),
            "inference_s":  elapsed,
            "audio_dur_s":  audio_dur,
            "rtf":          elapsed / audio_dur if audio_dur > 0 else None,
        })

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")

    if not samples:
        print("❌ No samples generated. Exiting.")
        sys.exit(1)

    # Free GPU memory before loading Whisper
    del model
    torch.cuda.empty_cache()

    # ── PHASE 4 · ASR transcription ───────────────────────────────────────────
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
                language=args.whisper_lang,
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

    # ── PHASE 5 · Acoustic metrics + speaker embeddings ───────────────────────
    print(f"\n📊 Phase 5: Acoustic metrics & speaker embeddings")
    verifier        = load_speaker_model(device=device)
    utmos_predictor = None if args.skip_utmos else load_utmos(device=device)

    import jiwer
    import sacrebleu

    wer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    results: List[Dict] = []
    synth_embeddings: List[Tuple[torch.Tensor, str]] = []
    gt_embeddings:    List[Tuple[torch.Tensor, str]] = []

    for s, tx in tqdm(
            zip(samples, transcripts),
            total=len(samples), desc="Computing metrics"):

        gt_path  = s["gt_path"]
        ref_path = s["ref_path"]
        syn_path = s["syn_path"]
        spk      = s["speaker_id"]

        # Quality metrics
        pesq_val  = compute_pesq(gt_path, syn_path)
        stoi_val  = compute_stoi(gt_path, syn_path)
        mcd_val   = compute_mcd(gt_path, syn_path)
        pitch_val = compute_pitch_correlation(gt_path, syn_path)
        utmos_val = compute_utmos(syn_path, utmos_predictor, device)

        # Speaker similarity (against ORIGINAL source voice)
        syn_emb = extract_speaker_embedding(syn_path, verifier, device)
        ref_emb = extract_speaker_embedding(ref_path,  verifier, device)
        if syn_emb is not None and ref_emb is not None:
            sim = float(F.cosine_similarity(
                syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
            synth_embeddings.append((syn_emb, spk))
            gt_embeddings.append((ref_emb,  spk))
        else:
            sim = None

        # WER
        try:
            w = float(jiwer.wer(
                s["text_fr"], tx,
                truth_transform=wer_transforms,
                hypothesis_transform=wer_transforms,
            )) if tx.strip() else 1.0
        except Exception:
            w = None

        # chrF++
        try:
            chr_val = float(
                sacrebleu.sentence_chrf(tx, [s["text_fr"]]).score)
        except Exception:
            chr_val = None

        # Cleanup temp files
        if os.path.exists(gt_path):
            os.remove(gt_path)
        if os.path.exists(ref_path):
            os.remove(ref_path)

        results.append({
            "idx":         s["idx"],
            "speaker":     spk,
            "WER":         w,
            "chrF":        chr_val,
            "PESQ":        pesq_val,
            "STOI":        stoi_val,
            "MCD":         mcd_val,
            "UTMOS":       utmos_val,
            "PitchCorr":   pitch_val,
            "Similarity":  sim,
            # ── Timing metrics ──────────────────────
            "InferenceS":  s["inference_s"],
            "AudioDurS":   s["audio_dur_s"],
            "RTF":         s["rtf"],
            # ── Transcription ───────────────────────
            "transcript":  tx,
            "reference":   s["text_fr"],
        })

    del verifier
    if utmos_predictor is not None:
        del utmos_predictor
    torch.cuda.empty_cache()

    # ── PHASE 6 · EER ─────────────────────────────────────────────────────────
    print(f"\n🔐 Phase 6: Computing EER from {len(synth_embeddings)} embeddings")
    genuine_scores:  List[float] = []
    impostor_scores: List[float] = []

    for syn_emb_i, spk_i in synth_embeddings:
        for gt_emb_j, spk_j in gt_embeddings:
            sim_ij = float(F.cosine_similarity(
                syn_emb_i.unsqueeze(0),
                gt_emb_j.unsqueeze(0),
            ).item())
            if spk_i == spk_j:
                genuine_scores.append(sim_ij)
            else:
                impostor_scores.append(sim_ij)

    eer_value = compute_eer(genuine_scores, impostor_scores)
    print(f"   Genuine:  {len(genuine_scores)} | Impostor: {len(impostor_scores)}")
    print(f"   EER: {eer_value:.4f}" if eer_value is not None
          else "   EER: could not compute (need ≥5 genuine + impostor pairs)")

    # ── PHASE 7 · Summary & export ────────────────────────────────────────────
    print(f"\n📋 Phase 7: Generating summary")

    wall_total   = time.perf_counter() - wall_start
    mean_inf     = safe_mean([r["InferenceS"] for r in results])
    mean_dur     = safe_mean([r["AudioDurS"]  for r in results])
    mean_rtf     = safe_mean([r["RTF"]        for r in results])
    total_inf    = sum(r["InferenceS"] for r in results)

    metric_keys = [
        "WER", "chrF", "Similarity",
        "PESQ", "STOI", "MCD", "UTMOS", "PitchCorr",
        "InferenceS", "AudioDurS", "RTF",
    ]

    overall: Dict = {}
    for k in metric_keys:
        vals = [r[k] for r in results]
        overall[k] = {
            "mean":  safe_mean(vals),
            "std":   safe_std(vals),
            "valid": safe_count(vals),
            "total": len(vals),
        }

    speakers = sorted(set(r["speaker"] for r in results))
    per_speaker: Dict = {}
    for spk in speakers:
        spk_r = [r for r in results if r["speaker"] == spk]
        per_speaker[spk] = {"count": len(spk_r)}
        for k in metric_keys:
            vals = [r[k] for r in spk_r]
            per_speaker[spk][k] = {
                "mean":  safe_mean(vals),
                "std":   safe_std(vals),
                "valid": safe_count(vals),
            }

    output = {
        "config": {
            "model":          MODEL_ID,
            "mode":           "raw",
            "dataset":        args.dataset,
            "split":          args.split,
            "whisper_model":  args.whisper_model,
            "exaggeration":   args.exl,
            "cfg_weight":     args.cfg_weight,
            "num_evaluated":  len(results),
            "num_skipped":    skipped,
        },
        "overall":      {k: v["mean"]  for k, v in overall.items()},
        "overall_std":  {k: v["std"]   for k, v in overall.items()},
        "valid_counts": {k: v["valid"] for k, v in overall.items()},
        "EER":          eer_value,
        "eer_details": {
            "genuine_trials":  len(genuine_scores),
            "impostor_trials": len(impostor_scores),
        },
        "timing_summary": {
            "wall_time_s":        round(wall_total, 2),
            "total_inference_s":  round(total_inf,  2),
            "mean_inference_s":   round(mean_inf,   4),
            "mean_audio_dur_s":   round(mean_dur,   4),
            "mean_rtf":           round(mean_rtf,   4),
        },
        "per_speaker": per_speaker,
        "results":     results,
    }

    # JSON
    json_path = os.path.join(args.output_dir, "eval_chatterbox_raw.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # CSV
    csv_path   = os.path.join(args.output_dir, "eval_chatterbox_raw.csv")
    csv_fields = (["idx", "speaker"] + metric_keys
                  + ["transcript", "reference"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields,
                                extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # ── Console summary ────────────────────────────────────────────────────────
    direction = {
        "WER":        "↓", "chrF":       "↑",
        "Similarity": "↑", "PESQ":       "↑",
        "STOI":       "↑", "MCD":        "↓",
        "UTMOS":      "↑", "PitchCorr":  "↑",
        "InferenceS": "↓", "AudioDurS":  " ",
        "RTF":        "↓",
    }
    label = {
        "InferenceS": "Inf. Time (s)",
        "AudioDurS":  "Audio Dur.(s)",
        "RTF":        "RTF",
    }

    print("\n" + "=" * 68)
    print("  CHATTERBOX RAW — EVALUATION COMPLETE")
    print(f"  ASR: faster-whisper {args.whisper_model}")
    print(f"  Samples: {len(results)} evaluated, {skipped} skipped")
    print("=" * 68)
    print(f"  {'Metric':<16} {'Dir':>3} {'Mean':>9} {'± Std':>9}  {'Valid':>6}")
    print("-" * 68)
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

    print("-" * 68)
    if eer_value is not None:
        print(f"  {'EER':<16} {'↓':>3} {eer_value:>9.4f}"
              f"            ({len(genuine_scores)}g/{len(impostor_scores)}i)")
    else:
        print(f"  {'EER':<16} {'↓':>3}       N/A")
    print("=" * 68)

    print("\n  ⏱  Timing Summary")
    print(f"  {'Wall-clock time':<22}: {wall_total:.1f} s  "
          f"({wall_total/60:.1f} min)")
    print(f"  {'Total inference time':<22}: {total_inf:.1f} s")
    print(f"  {'Mean inference / sample':<22}: {mean_inf:.3f} s")
    print(f"  {'Mean audio duration':<22}: {mean_dur:.3f} s")
    print(f"  {'Mean RTF':<22}: {mean_rtf:.4f}  "
          f"({'faster' if mean_rtf < 1 else 'slower'} than real-time)")

    print(f"\n  Per-Speaker Breakdown:")
    spk_metric_keys = [k for k in metric_keys
                       if k not in ("AudioDurS",)]
    for spk in speakers:
        n = per_speaker[spk]["count"]
        print(f"\n  ┌─ Speaker: {spk} ({n} samples)")
        for k in spk_metric_keys:
            v = per_speaker[spk][k]["mean"]
            c = per_speaker[spk][k]["valid"]
            v_s = f"{v:.4f}" if not np.isnan(v) else "N/A"
            print(f"  │  {label.get(k, k):<16}: {v_s:>9}  ({c} valid)")
        print("  └─")

    print(f"\n✅ JSON  → {json_path}")
    print(f"✅ CSV   → {csv_path}")
    print(f"✅ Audio → {args.output_dir}/synth_*.wav")


if __name__ == "__main__":
    main()
