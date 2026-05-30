#!/usr/bin/env python3
"""
Re-compute all evaluation metrics (Transcription + WER + CER + Similarity) 
from the already generated audio files (synth_XXXXX.wav).
This saves hours of re-generation time!

Usage: python evaluation/recompute_metrics.py --output-dir ./eval_results/zh_base --whisper-lang zh
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from datasets import load_dataset


def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    sb_device = "cuda:0" if device == "cuda" else device
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": sb_device}
    )

def extract_speaker_embedding_from_wav(wav_path, model, device="cuda"):
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).detach()
    except Exception as e:
        print(f"   ⚠ Failed to extract embedding from wav {wav_path}: {e}")
        return None

def extract_speaker_embedding_from_array(audio_array, sr, model, device="cuda"):
    try:
        wav = torch.from_numpy(audio_array).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).detach()
    except Exception as e:
        print(f"   ⚠ Failed to extract embedding from array: {e}")
        return None

def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def safe_std(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.std(v)) if v else float('nan')

def safe_count(vals):
    return len([x for x in vals if x is not None and not np.isnan(x)])


def main():
    parser = argparse.ArgumentParser(description="Recompute metrics from generated audio files")
    parser.add_argument("--output-dir", default="./eval_results/zh_base", help="Output directory where audio files are located")
    parser.add_argument("--dataset", default="ymoslem/acl-6060", help="HF dataset name")
    parser.add_argument("--split", default="eval", help="Dataset split")
    parser.add_argument("--cache-dir", default="./data_cache", help="Dataset cache dir")
    parser.add_argument("--whisper-lang", default="zh", help="Language code (zh, fr, etc.)")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--whisper-beam", type=int, default=5, help="Beam size for transcription")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_lang = args.whisper_lang.strip().lower()

    print("=" * 64)
    print(f"  RE-COMPUTING METRICS ({target_lang})")
    print("=" * 64)

    # 1. Load dataset
    print("\n📥 Loading dataset...")
    ds_test = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total_samples = len(ds_test)
    print(f"   Dataset samples: {total_samples}")

    # 2. Check which synthesized audio files exist
    existing_audios = []
    for i in range(total_samples):
        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        if os.path.exists(syn_path):
            existing_audios.append((i, syn_path))
    
    print(f"   Found {len(existing_audios)} generated audio files in {args.output_dir}.")
    if not existing_audios:
        print("❌ No synthesized wav files found! Please make sure --output-dir is correct.")
        return

    # 3. Load Models
    print("\n🔎 Loading Whisper-large-v3...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device=device, compute_type="float16" if device == "cuda" else "int8")

    print("🔎 Loading SpeechBrain ECAPA-TDNN...")
    verifier = load_speaker_model(device=device)

    # Text normalization
    import jiwer
    if target_lang in ("zh", "ar", "ja", "ko"):
        wer_transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
    else:
        wer_transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
        ])

    results = []

    # 4. Transcribe and compute metrics
    for idx, syn_path in tqdm(existing_audios, desc="Computing metrics"):
        row = ds_test[idx]
        text_target = (row.get(f"trg_{target_lang}_text") or row.get(f"text_{target_lang}") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get(f"ref_{target_lang}_voice") or row.get("audio_en") or row.get("audio")

        # Get audio duration
        try:
            info = torchaudio.info(syn_path)
            audio_dur = info.num_frames / info.sample_rate
        except Exception:
            audio_dur = 0.0

        # Transcribe
        transcript = ""
        try:
            segments, _ = whisper.transcribe(syn_path, language=target_lang, beam_size=args.whisper_beam, vad_filter=True)
            transcript = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"   ⚠ Transcription failed for sample {idx}: {e}")

        # Speaker Similarity
        sim = None
        if ref_data:
            ref_emb = extract_speaker_embedding_from_array(
                np.asarray(ref_data["array"], dtype=np.float32),
                ref_data["sampling_rate"],
                verifier,
                device
            )
            syn_emb = extract_speaker_embedding_from_wav(
                syn_path,
                verifier,
                device
            )
            if ref_emb is not None and syn_emb is not None:
                sim = float(F.cosine_similarity(ref_emb.unsqueeze(0), syn_emb.unsqueeze(0)).item())

        # WER / CER
        w = c = None
        try:
            if transcript.strip() and text_target.strip():
                ref_clean = wer_transforms(text_target)
                hyp_clean = wer_transforms(transcript)
                w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
                c = float(jiwer.cer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
            else:
                w = c = 1.0
        except Exception as e:
            print(f"   ⚠ Text metrics failed for sample {idx}: {e}")

        results.append({
            "idx": idx, "WER": w, "CER": c, "Similarity": sim,
            "InferenceS": 0.0, "AudioDurS": audio_dur, "RTF": 0.0,
            "transcript": transcript, "reference": text_target
        })

    # 5. Summarize and Save
    metric_keys = ["WER", "CER", "Similarity"]
    overall = {k: {"mean": safe_mean([r[k] for r in results]), "std": safe_std([r[k] for r in results]), "valid": safe_count([r[k] for r in results])} for k in metric_keys}

    print("\n" + "=" * 62)
    print("  EVALUATION COMPLETE")
    print(f"  Target: {target_lang} | Samples: {len(results)}")
    print("=" * 62)
    print(f"  {'Metric':<16} {'Mean':>9} {'± Std':>9}  {'Valid':>6}")
    print("-" * 62)
    for k in metric_keys:
        m, s, v = overall[k]["mean"], overall[k]["std"], overall[k]["valid"]
        print(f"  {k:<16} {m:>9.4f} {f'±{s:.4f}' if not np.isnan(s) else '':>9}  {v:>3}/{len(results)}")
    print("=" * 62)

    # Save summary
    with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "eval_per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "WER", "CER", "Similarity", "InferenceS", "AudioDurS", "RTF", "reference", "transcript"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Per-sample results saved to {csv_path}")
    print(f"✅ Summary results saved to {os.path.join(args.output_dir, 'eval_summary.json')}")


if __name__ == "__main__":
    main()
