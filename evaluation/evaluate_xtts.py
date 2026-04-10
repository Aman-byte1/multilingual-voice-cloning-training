#!/usr/bin/env python3
"""
Cross-Lingual Voice Cloning Evaluation Pipeline — XTTS-v2
==========================================================
Zero-shot Coqui XTTS-v2 evaluation on ymoslem/acl-6060.
Supports 17 languages including Arabic, Chinese, French.
Metrics: WER, CER, Speaker Similarity, Inference Time, RTF.

ASR: faster-whisper large-v3
"""

import os
import sys
import csv
import json
import time
import argparse
import warnings
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

sys.path.insert(0, os.path.dirname(__file__))

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")

# ---- Monkey-patch for coqui-tts 0.27.5 + transformers>=4.57 compat ----
# coqui-tts imports `isin_mps_friendly` which was removed in transformers 4.57
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = torch.isin

# ===================================================================
# Language config — XTTS-v2 uses ISO codes (with zh-cn for Chinese)
# Supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn,
#           ja, hu, ko, hi
# ===================================================================
XTTS_LANG_MAP = {
    "fr": "fr",
    "ar": "ar",
    "zh": "zh-cn",   # XTTS uses "zh-cn" not "zh"
    "en": "en",
    "es": "es",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ja": "ja",
    "hu": "hu",
    "ko": "ko",
    "hi": "hi",
}


# ===================================================================
# Helper functions
# ===================================================================

def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval", output_dir: str = None) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{prefix}.wav")
    else:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
        os.close(fd)

    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path

def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )

def extract_speaker_embedding(wav_path, model, device="cuda"):
    """Extract speaker embedding, resampling to 16kHz (ECAPA-TDNN requirement)."""
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0).to(device)  # mono
        emb = model.encode_batch(wav.unsqueeze(0))
        return emb.squeeze(0).squeeze(0).detach()
    except Exception:
        return None

def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.mean(v) if v else np.nan

def safe_std(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.std(v) if v else np.nan

def safe_count(vals):
    return len([x for x in vals if x is not None and not np.isnan(x)])


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Lingual Voice Cloning Evaluation — XTTS-v2")
    parser.add_argument("--dataset",
                        default="ymoslem/acl-6060")
    parser.add_argument("--model-name",
                        default="tts_models/multilingual/multi-dataset/xtts_v2",
                        help="Coqui TTS model name")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr",
                        help="Language code (fr, ar, zh, etc.)")
    parser.add_argument("--output-dir", default="./eval_results_xtts")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--resume", action="store_true", help="Skip generation if audio already exists")
    parser.add_argument("--hf-token", default=None, help="Hugging Face API token for authentication")
    args = parser.parse_args()

    if args.hf_token:
        print("🔑 Authenticating with Hugging Face...")
        login(token=args.hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    target_lang = args.whisper_lang.strip().lower()

    if target_lang not in XTTS_LANG_MAP:
        print(f"\n❌ ERROR: '{target_lang}' is NOT supported by XTTS-v2.")
        print(f"   Supported: {', '.join(sorted(XTTS_LANG_MAP.keys()))}")
        sys.exit(1)

    xtts_lang = XTTS_LANG_MAP[target_lang]

    print("=" * 64)
    print(f"  XTTS-v2 EVALUATION ({target_lang} / {xtts_lang})")
    print("=" * 64)

    # Ensure temp dir for references (on disk, not RAM)
    temp_ref_dir = os.path.join(args.output_dir, "temp_ref")
    os.makedirs(temp_ref_dir, exist_ok=True)

    # PHASE 1: Load dataset and pre-extract to disk
    print(f"\n📥 Phase 1: Loading dataset and extracting to disk")
    ds_test = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)
    print(f"   Samples: {total}")

    # Pre-extract all text and ref audio to disk, then free the dataset
    print(f"   Extracting references to {temp_ref_dir} ...")
    manifest = []
    for i in tqdm(range(total), desc="Extracting"):
        row = ds_test[i]
        text_target = (row.get(f"trg_{target_lang}_text") or row.get(f"text_{target_lang}") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get(f"ref_{target_lang}_voice") or row.get("audio_en") or row.get("audio")

        if not ref_data or not text_target:
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"],
            f"ref_{i:05d}",
            output_dir=temp_ref_dir
        )
        manifest.append({
            "idx": i,
            "text_target": text_target,
            "ref_path": ref_path,
            "speaker_id": row.get("speaker_id", "unknown"),
        })

    # FREE the dataset to save RAM
    del ds_test
    gc.collect()
    print(f"   Extracted {len(manifest)} samples. Dataset freed from RAM.")

    # PHASE 2: Load XTTS-v2
    print(f"\n🔧 Phase 2: Loading XTTS-v2 model")
    from TTS.api import TTS
    tts = TTS(args.model_name, gpu=(device == "cuda"))
    print("   XTTS-v2 loaded ✓")

    XTTS_SR = 24000  # XTTS-v2 native sample rate

    # PHASE 3: Generate (using pre-extracted manifest)
    print(f"\n🎙  Phase 3: Generating {len(manifest)} samples")
    samples = []
    skipped = 0
    inference_times = []
    audio_durations = []

    for entry in tqdm(manifest, desc="Generating"):
        i = entry["idx"]
        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        ref_path = entry["ref_path"]
        text_target = entry["text_target"]

        # Resume logic: skip generation if file exists
        if args.resume and os.path.exists(syn_path):
            try:
                wav_info = torchaudio.info(syn_path)
                audio_dur = wav_info.num_frames / wav_info.sample_rate
                samples.append({
                    "idx": i, "syn_path": syn_path, "ref_path": ref_path,
                    "text_target": text_target, "speaker_id": entry["speaker_id"],
                    "inference_s": 0, "audio_dur_s": audio_dur, "rtf": 0
                })
                continue
            except Exception:
                pass  # If file is corrupt, re-generate

        try:
            t0 = time.perf_counter()
            tts.tts_to_file(
                text=text_target,
                speaker_wav=ref_path,
                language=xtts_lang,
                file_path=syn_path
            )
            t1 = time.perf_counter()
        except Exception as e:
            print(f"   ⚠ Sample {i} generation failed: {e}")
            skipped += 1
            continue

        elapsed = t1 - t0
        try:
            info = torchaudio.info(syn_path)
            audio_dur = info.num_frames / info.sample_rate
        except Exception:
            audio_dur = 0

        inference_times.append(elapsed)
        audio_durations.append(audio_dur)

        samples.append({
            "idx": i,
            "syn_path": syn_path,
            "ref_path": ref_path,
            "text_target": text_target,
            "speaker_id": entry["speaker_id"],
            "inference_s": elapsed,
            "audio_dur_s": audio_dur,
            "rtf": elapsed / audio_dur if audio_dur > 0 else 0
        })

        # Proactive OOM prevention
        if i % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")
    del tts
    gc.collect()
    torch.cuda.empty_cache()

    # PHASE 4: ASR
    print(f"\n🗣  Phase 4: Transcribing ({target_lang})")
    from faster_whisper import WhisperModel as FasterWhisperModel
    whisper = FasterWhisperModel(args.whisper_model, device=device, compute_type="float16" if device == "cuda" else "int8")

    transcripts = []
    for s in tqdm(samples, desc="Transcribing"):
        try:
            segments, _ = whisper.transcribe(s["syn_path"], language=target_lang, beam_size=args.whisper_beam, vad_filter=True)
            transcripts.append(" ".join(seg.text for seg in segments).strip())
        except Exception:
            transcripts.append("")

    del whisper
    gc.collect()
    torch.cuda.empty_cache()

    # PHASE 5: Metrics
    print(f"\n📊 Phase 5: Computing Metrics")
    verifier = load_speaker_model(device=device)
    import jiwer

    # Language-appropriate text normalization
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
    for s, tx in tqdm(zip(samples, transcripts), total=len(samples), desc="Metrics"):
        # Speaker Similarity
        syn_emb = extract_speaker_embedding(s["syn_path"], verifier, device)
        ref_emb = extract_speaker_embedding(s["ref_path"], verifier, device)
        sim = float(F.cosine_similarity(syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()) if (syn_emb is not None and ref_emb is not None) else None

        # WER / CER
        try:
            if tx.strip():
                ref_clean = wer_transforms(s["text_target"])
                hyp_clean = wer_transforms(tx)
                w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
                c = float(jiwer.cer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
            else:
                w = c = 1.0
        except Exception as e:
            print(f"   ⚠ Metrics failed for sample {s['idx']}: {e}")
            w = c = None

        if os.path.exists(s["ref_path"]): os.remove(s["ref_path"])

        results.append({
            "idx": s["idx"], "WER": w, "CER": c, "Similarity": sim,
            "InferenceS": s["inference_s"], "AudioDurS": s["audio_dur_s"], "RTF": s["rtf"],
            "transcript": tx, "reference": s["text_target"]
        })

    # Summary
    metric_keys = ["WER", "CER", "Similarity", "InferenceS", "RTF"]
    overall = {k: {"mean": safe_mean([r[k] for r in results]), "std": safe_std([r[k] for r in results]), "valid": safe_count([r[k] for r in results])} for k in metric_keys}

    print("\n" + "=" * 62)
    print("  XTTS-v2 EVALUATION COMPLETE")
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

    # Save per-sample CSV
    csv_path = os.path.join(args.output_dir, "eval_per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "WER", "CER", "Similarity", "InferenceS", "AudioDurS", "RTF", "reference", "transcript"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Per-sample results saved to {csv_path}")

    # Final cleanup
    if os.path.exists(temp_ref_dir):
        import shutil
        shutil.rmtree(temp_ref_dir)

if __name__ == "__main__":
    main()
