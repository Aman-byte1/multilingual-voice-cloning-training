#!/usr/bin/env python3
"""
Cross-Lingual Voice Cloning Evaluation Pipeline — Qwen3-TTS
============================================================
Zero-shot Qwen3-TTS evaluation on ymoslem/acl-6060.
Supports dynamic language targets (Arabic, Chinese, French, etc.)
and streamlined metrics (Time, WER, CER, Similarity).

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
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

sys.path.insert(0, os.path.dirname(__file__))

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")

# ===================================================================
# Language config — Qwen uses full language names
# Qwen3-TTS supported: chinese, english, french, german, italian,
#                       japanese, korean, portuguese, russian, spanish
# NOTE: Arabic is NOT supported by Qwen3-TTS
# ===================================================================
QWEN_LANG_MAP = {
    "fr": "French",
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


# ===================================================================
# Helper functions (shared with eval.py)
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
        description="Cross-Lingual Voice Cloning Evaluation — Qwen3-TTS")
    parser.add_argument("--dataset",
                        default="ymoslem/acl-6060")
    parser.add_argument("--model-name",
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Qwen3-TTS model name or path")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr",
                        help="Language code (fr, ar, zh, etc.)")
    parser.add_argument("--output-dir", default="./eval_results_qwen")
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

    if target_lang not in QWEN_LANG_MAP:
        print(f"\n❌ ERROR: '{target_lang}' is NOT supported by Qwen3-TTS.")
        print(f"   Supported: {', '.join(sorted(QWEN_LANG_MAP.keys()))}")
        print(f"   Use Chatterbox or XTTS-v2 for {target_lang} instead.")
        sys.exit(1)

    qwen_lang = QWEN_LANG_MAP[target_lang]

    print("=" * 64)
    print(f"  QWEN3-TTS EVALUATION ({target_lang} / {qwen_lang})")
    print("=" * 64)

    # Ensure temp dir for references (on disk, not RAM)
    temp_ref_dir = os.path.join(args.output_dir, "temp_ref")
    os.makedirs(temp_ref_dir, exist_ok=True)

    # PHASE 1: Load dataset
    print(f"\n📥 Phase 1: Loading dataset from {args.dataset}")
    ds_test = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)
    print(f"   Samples: {total}")

    # PHASE 2: Load Qwen3-TTS
    print(f"\n🔧 Phase 2: Loading Qwen3-TTS model: {args.model_name}")
    try:
        from qwen_tts import Qwen3TTSModel
        try:
            import flash_attn  # noqa
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            print("   flash-attn not installed, using SDPA attention")
        model = Qwen3TTSModel.from_pretrained(
            args.model_name,
            device_map=device,
            dtype=torch.float32,
            attn_implementation=attn_impl,
        )
        print("   Qwen3-TTS loaded via qwen_tts ✓")
    except ImportError:
        print("   qwen_tts not found, trying transformers...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        print("   Qwen3-TTS loaded via transformers ✓")

    QWEN_SR = 24000  # Qwen3-TTS native sample rate

    # PHASE 3: Generate
    print(f"\n🎙  Phase 3: Generating {total} samples")
    samples = []
    skipped = 0
    inference_times = []
    audio_durations = []

    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        text_target = (row.get(f"trg_{target_lang}_text") or row.get(f"text_{target_lang}") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get(f"ref_{target_lang}_voice") or row.get("audio_en") or row.get("audio")
        ref_text = (row.get("ref_en_text") or row.get("text_en") or "").strip()

        if not ref_data or not text_target or not ref_text:
            skipped += 1
            continue

        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"],
            f"ref_{i:05d}",
            output_dir=temp_ref_dir
        )

        # Resume logic: skip generation if file exists and --resume is set
        if args.resume and os.path.exists(syn_path):
            try:
                wav_info = torchaudio.info(syn_path)
                audio_dur = wav_info.num_frames / wav_info.sample_rate
                samples.append({
                    "idx": i, "syn_path": syn_path, "ref_path": ref_path,
                    "text_target": text_target, "speaker_id": row.get("speaker_id", "unknown"),
                    "inference_s": 0, "audio_dur_s": audio_dur, "rtf": 0
                })
                continue
            except Exception:
                pass  # If file is corrupt, re-generate

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                # Try qwen_tts API first, then fallback
                if hasattr(model, 'generate_voice_clone'):
                    wavs, sr = model.generate_voice_clone(
                        text=text_target,
                        language=qwen_lang,
                        ref_audio=ref_path,
                        ref_text=ref_text,
                    )
                    wav_np = wavs[0] if isinstance(wavs, list) else wavs
                    if isinstance(wav_np, torch.Tensor):
                        wav_np = wav_np.cpu().numpy()
                    if wav_np.ndim > 1:
                        wav_np = wav_np.squeeze()
                    sf.write(syn_path, wav_np, sr)
                    actual_sr = sr
                elif hasattr(model, 'generate'):
                    wavs = model.generate(
                        text=text_target,
                        ref_audio_path=ref_path,
                        ref_text=ref_text,
                        language=qwen_lang,
                    )
                    if isinstance(wavs, torch.Tensor):
                        wav_np = wavs.cpu().numpy()
                        if wav_np.ndim > 1:
                            wav_np = wav_np.squeeze()
                        sf.write(syn_path, wav_np, QWEN_SR)
                    else:
                        sf.write(syn_path, wavs[0], QWEN_SR)
                    actual_sr = QWEN_SR
                else:
                    raise RuntimeError("Model has no generate_voice_clone or generate method")

            t1 = time.perf_counter()
        except Exception as e:
            print(f"   ⚠ Sample {i} generation failed: {e}")
            if ref_path and os.path.exists(ref_path): os.remove(ref_path)
            skipped += 1
            continue

        elapsed = t1 - t0
        # Read back to get exact duration
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
            "speaker_id": row.get("speaker_id", "unknown"),
            "inference_s": elapsed,
            "audio_dur_s": audio_dur,
            "rtf": elapsed / audio_dur if audio_dur > 0 else 0
        })

        # Proactive OOM prevention: clear cache every 25 samples
        if i % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")
    del model
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
    # RemovePunctuation destroys CJK characters — only use for Latin scripts
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
    print("  QWEN3-TTS EVALUATION COMPLETE")
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

    # Save per-sample CSV for analysis
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
