#!/usr/bin/env python3
"""
Blind Test Evaluation — All Baseline Models × 4 Selected Voices
================================================================
Evaluates: OmniVoice (base), Chatterbox, Qwen3, XTTS-v2, VoxCPM2
Voices:    193 (Asian female), 14 (Western female),
           810 (Western male),  289 (Asian male)
Metrics:   WER, CER, Speaker Similarity (ECAPA-TDNN)
ASR:       faster-whisper large-v3

Usage:
    python evaluation/evaluate_blind_all_models.py --lang fr
    python evaluation/evaluate_blind_all_models.py --lang ar
    python evaluation/evaluate_blind_all_models.py --lang zh
    python evaluation/evaluate_blind_all_models.py --lang all
    python evaluation/evaluate_blind_all_models.py --lang fr --models omnivoice chatterbox
"""

import os, sys, gc, csv, json, time, argparse, warnings, glob
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# ── Selected voices ──────────────────────────────────────────────
SELECTED_VOICES = ["193", "14", "810", "289"]
VOICE_META = {
    "193": "Asian Female",
    "14":  "Western Female",
    "810": "Western Male",
    "289": "Asian Male",
}

LANG_TEXT_MAP = {
    "zh": "chinese.txt",
    "ar": "arabic.txt",
    "fr": "french.txt",
}

# Models that support each language
MODEL_LANG_SUPPORT = {
    "omnivoice":  ["fr", "ar", "zh"],
    "chatterbox": ["fr", "ar", "zh"],
    "qwen3":      ["fr", "zh"],          # No Arabic support
    "xtts":       ["fr", "ar", "zh"],
    "voxcpm":     ["fr", "ar", "zh"],
}

QWEN_LANG_MAP = {"fr": "French", "zh": "Chinese"}
XTTS_LANG_MAP = {"fr": "fr", "ar": "ar", "zh": "zh-cn"}

# ── Helpers ──────────────────────────────────────────────────────

def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device},
    )

def extract_speaker_embedding(wav_path, model, device="cuda"):
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).detach()
    except Exception:
        return None

def safe_mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")

def safe_std(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(v)) if v else float("nan")

def find_ref_audio(audio_dir, voice_id):
    """Find reference audio file for a voice ID, trying multiple naming patterns."""
    patterns = [
        f"{voice_id}.wav",
        f"speaker_{voice_id}.wav",
        f"spk_{voice_id}.wav",
        f"voice_{voice_id}.wav",
    ]
    for pat in patterns:
        p = os.path.join(audio_dir, pat)
        if os.path.exists(p):
            return p
    # Glob fallback
    matches = glob.glob(os.path.join(audio_dir, f"*{voice_id}*.wav"))
    if matches:
        return matches[0]
    return None

def get_best_reference(ref_path, duration=10.0):
    """Extract clean speech segment for voice cloning."""
    waveform, sr = torchaudio.load(str(ref_path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    window_size = int(sr * 0.05)
    stride = window_size // 2
    windows = waveform.unfold(-1, window_size, stride)
    energy = torch.sum(windows**2, dim=-1).squeeze(0)
    threshold = torch.max(energy) * 0.05
    active_frames = (energy > threshold).nonzero()
    target_samples = int(duration * sr)
    start_idx = active_frames[0].item() * stride if len(active_frames) > 0 else 0
    end_idx = start_idx + target_samples
    best_chunk = waveform[:, start_idx:end_idx]
    return (best_chunk, sr)


# ── Model Generators ─────────────────────────────────────────────

def generate_omnivoice(text, ref_path, ref_tuple, lang, device, model):
    """Generate audio with OmniVoice base model."""
    with torch.no_grad():
        res = model.generate(text=text, ref_audio=ref_tuple, temperature=0.8, top_p=0.9)
        audio = res[0] if isinstance(res, tuple) else res
        if isinstance(audio, (list, tuple)):
            audio = torch.from_numpy(np.array(audio))
        elif not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return audio.cpu().float(), 24000

def generate_chatterbox(text, ref_path, ref_tuple, lang, device, model):
    """Generate audio with Chatterbox."""
    with torch.inference_mode():
        wav = model.generate(
            text, language_id=lang, audio_prompt_path=ref_path,
            exaggeration=0.5, cfg_weight=0.0,
        )
    wav_out = wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu()
    return wav_out, model.sr

def generate_qwen3(text, ref_path, ref_tuple, lang, device, model):
    """Generate audio with Qwen3-TTS."""
    import soundfile as sf
    qwen_lang = QWEN_LANG_MAP[lang]
    # Need ref_text for Qwen3 — use empty string if unavailable
    if hasattr(model, 'generate_voice_clone'):
        wavs, sr = model.generate_voice_clone(
            text=text, language=qwen_lang, ref_audio=ref_path, ref_text="",
        )
        wav_np = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(wav_np, torch.Tensor):
            wav_np = wav_np.cpu().numpy()
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()
        wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
        return wav_t, sr
    elif hasattr(model, 'generate'):
        wavs = model.generate(
            text=text, ref_audio_path=ref_path, ref_text="", language=qwen_lang,
        )
        if isinstance(wavs, torch.Tensor):
            wav_np = wavs.cpu().numpy()
        else:
            wav_np = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(wav_np, np.ndarray):
            if wav_np.ndim > 1:
                wav_np = wav_np.squeeze()
            wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
        else:
            wav_t = wav_np.unsqueeze(0) if wav_np.dim() == 1 else wav_np
        return wav_t.cpu().float(), 24000

def generate_xtts(text, ref_path, ref_tuple, lang, device, model):
    """Generate audio with XTTS-v2."""
    import tempfile
    xtts_lang = XTTS_LANG_MAP.get(lang, lang)
    fd, syn_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    model.tts_to_file(text=text, speaker_wav=ref_path, language=xtts_lang, file_path=syn_path)
    wav, sr = torchaudio.load(syn_path)
    os.remove(syn_path)
    return wav, sr

def generate_voxcpm(text, ref_path, ref_tuple, lang, device, model):
    """Generate audio with VoxCPM2."""
    import soundfile as sf
    wav = model.generate(
        text=text, reference_wav_path=ref_path, cfg_value=2.0, inference_timesteps=10,
    )
    sr = model.tts_model.sample_rate
    wav_t = torch.from_numpy(wav).float().unsqueeze(0) if isinstance(wav, np.ndarray) else wav
    if isinstance(wav_t, torch.Tensor) and wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    return wav_t if isinstance(wav_t, torch.Tensor) else torch.from_numpy(np.array(wav_t)).float().unsqueeze(0), sr


# ── Model Loaders ────────────────────────────────────────────────

def load_model(model_name, device):
    """Load a model and return (model_obj, generate_fn)."""
    print(f"\n🔧 Loading {model_name}...")

    if model_name == "omnivoice":
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
        model.to(device).eval()
        return model, generate_omnivoice

    elif model_name == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        torch.backends.cudnn.enabled = False
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return model, generate_chatterbox

    elif model_name == "qwen3":
        try:
            from qwen_tts import Qwen3TTSModel
            try:
                import flash_attn
                attn = "flash_attention_2"
            except ImportError:
                attn = "sdpa"
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device, dtype=torch.float32, attn_implementation=attn,
            )
        except ImportError:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device, torch_dtype=torch.float32, trust_remote_code=True,
            )
        return model, generate_qwen3

    elif model_name == "xtts":
        from TTS.api import TTS
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        return model, generate_xtts

    elif model_name == "voxcpm":
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        return model, generate_voxcpm

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Main Pipeline ────────────────────────────────────────────────

def evaluate_model_on_blind(model_name, lang, text_lines, voice_refs, output_root, device, whisper, verifier):
    """Run generation + evaluation for one model × one language × all voices."""
    import jiwer

    if lang not in MODEL_LANG_SUPPORT.get(model_name, []):
        print(f"   ⏭ {model_name} does not support {lang}, skipping.")
        return None

    out_dir = os.path.join(output_root, model_name, lang)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, gen_fn = load_model(model_name, device)

    # Text normalization for metrics
    if lang in ("zh", "ar", "ja", "ko"):
        wer_transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    else:
        wer_transforms = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.RemovePunctuation(),
        ])

    results = []
    total = len(text_lines) * len(voice_refs)

    with tqdm(total=total, desc=f"{model_name}/{lang}") as pbar:
        for voice_id, ref_path in voice_refs.items():
            ref_tuple = get_best_reference(ref_path, duration=10.0)

            for idx, text in enumerate(text_lines):
                syn_path = os.path.join(out_dir, f"{lang}_{idx:03d}_{voice_id}.wav")

                # Generate (skip if already exists for resume)
                if not os.path.exists(syn_path):
                    try:
                        t0 = time.perf_counter()
                        wav_out, sr = gen_fn(text, ref_path, ref_tuple, lang, device, model)
                        t1 = time.perf_counter()
                        torchaudio.save(syn_path, wav_out, sr)
                        inf_time = t1 - t0
                    except Exception as e:
                        tqdm.write(f"   ⚠ {model_name} failed on voice {voice_id}, line {idx}: {e}")
                        pbar.update(1)
                        continue
                else:
                    inf_time = 0.0

                # ASR transcription
                try:
                    segs, _ = whisper.transcribe(syn_path, language=lang, beam_size=5, vad_filter=True)
                    hyp = " ".join(s.text for s in segs).strip()
                except Exception:
                    hyp = ""

                # WER / CER
                try:
                    if hyp.strip():
                        ref_clean = wer_transforms(text)
                        hyp_clean = wer_transforms(hyp)
                        wer = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
                        cer = float(jiwer.cer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
                    else:
                        wer = cer = 1.0
                except Exception:
                    wer = cer = 1.0

                # Speaker similarity
                syn_emb = extract_speaker_embedding(syn_path, verifier, device)
                ref_emb = extract_speaker_embedding(ref_path, verifier, device)
                if syn_emb is not None and ref_emb is not None:
                    sim = float(F.cosine_similarity(syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
                else:
                    sim = None

                results.append({
                    "model": model_name, "lang": lang, "voice_id": voice_id,
                    "voice_desc": VOICE_META.get(voice_id, ""),
                    "text_idx": idx, "WER": wer, "CER": cer,
                    "Similarity": sim, "InferenceS": inf_time,
                })
                pbar.update(1)

                # Periodic cleanup
                if idx % 20 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

    # Unload model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save per-model results
    if results:
        csv_path = os.path.join(out_dir, "results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

        summary = {
            "model": model_name, "lang": lang, "n_samples": len(results),
            "WER": safe_mean([r["WER"] for r in results]),
            "CER": safe_mean([r["CER"] for r in results]),
            "Similarity": safe_mean([r["Similarity"] for r in results]),
            "InferenceS": safe_mean([r["InferenceS"] for r in results]),
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"   ✅ {model_name}/{lang}: WER={summary['WER']:.4f}  CER={summary['CER']:.4f}  SIM={summary['Similarity']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Blind Test — All Models × 4 Voices")
    parser.add_argument("--lang", default="all", help="fr, ar, zh, or all")
    parser.add_argument("--models", nargs="+",
                        default=["omnivoice", "chatterbox", "qwen3", "xtts", "voxcpm"],
                        help="Models to evaluate")
    parser.add_argument("--text-dir", default="./blind_test/text")
    parser.add_argument("--audio-dir", default="./blind_test/audio")
    parser.add_argument("--output-dir", default="./eval_blind_test")
    parser.add_argument("--voices", nargs="+", default=SELECTED_VOICES,
                        help="Voice IDs to evaluate")
    parser.add_argument("--whisper-model", default="large-v3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = ["fr", "ar", "zh"] if args.lang == "all" else [args.lang]

    print("=" * 64)
    print("  BLIND TEST EVALUATION — ALL MODELS × SELECTED VOICES")
    print(f"  Languages: {langs}")
    print(f"  Models:    {args.models}")
    print(f"  Voices:    {args.voices}")
    print(f"  Device:    {device}")
    print("=" * 64)

    # Resolve voice reference audio files
    voice_refs = {}
    for vid in args.voices:
        ref = find_ref_audio(args.audio_dir, vid)
        if ref:
            voice_refs[vid] = ref
            print(f"  ✓ Voice {vid} ({VOICE_META.get(vid, '?')}): {ref}")
        else:
            print(f"  ✗ Voice {vid}: NOT FOUND in {args.audio_dir}")
    if not voice_refs:
        print("❌ No voice references found. Check --audio-dir path.")
        sys.exit(1)

    # Load shared eval models (Whisper + ECAPA) — loaded once, reused across all models
    print(f"\n🔎 Loading Whisper {args.whisper_model} + ECAPA-TDNN...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device=device,
                           compute_type="float16" if device == "cuda" else "int8")
    verifier = load_speaker_model(device=device)

    all_results = []

    for lang in langs:
        # Load text lines
        text_file = os.path.join(args.text_dir, LANG_TEXT_MAP.get(lang, f"{lang}.txt"))
        if not os.path.exists(text_file):
            print(f"⚠ Text file not found: {text_file}, skipping {lang}")
            continue
        with open(text_file, "r", encoding="utf-8") as f:
            text_lines = [l.strip() for l in f if l.strip()]
        print(f"\n📝 {lang.upper()}: {len(text_lines)} text lines × {len(voice_refs)} voices")

        for model_name in args.models:
            results = evaluate_model_on_blind(
                model_name, lang, text_lines, voice_refs,
                args.output_dir, device, whisper, verifier,
            )
            if results:
                all_results.extend(results)

    # ── Final comparison table ───────────────────────────────────
    if all_results:
        print("\n" + "=" * 80)
        print("  📊 FINAL COMPARISON TABLE")
        print("=" * 80)
        print(f"  {'Model':<14} {'Lang':<5} {'WER↓':>8} {'CER↓':>8} {'SIM↑':>8} {'Time(s)↓':>10}")
        print("-" * 80)

        # Group by model+lang
        from collections import defaultdict
        groups = defaultdict(list)
        for r in all_results:
            groups[(r["model"], r["lang"])].append(r)

        for (m, l), recs in sorted(groups.items()):
            wer = safe_mean([r["WER"] for r in recs])
            cer = safe_mean([r["CER"] for r in recs])
            sim = safe_mean([r["Similarity"] for r in recs])
            inf = safe_mean([r["InferenceS"] for r in recs])
            print(f"  {m:<14} {l:<5} {wer:>8.4f} {cer:>8.4f} {sim:>8.4f} {inf:>10.3f}")

        print("=" * 80)

        # Per-voice breakdown
        print("\n  📊 PER-VOICE BREAKDOWN")
        print("-" * 80)
        print(f"  {'Model':<14} {'Lang':<5} {'Voice':<6} {'Desc':<16} {'WER↓':>8} {'CER↓':>8} {'SIM↑':>8}")
        print("-" * 80)
        voice_groups = defaultdict(list)
        for r in all_results:
            voice_groups[(r["model"], r["lang"], r["voice_id"])].append(r)
        for (m, l, v), recs in sorted(voice_groups.items()):
            wer = safe_mean([r["WER"] for r in recs])
            cer = safe_mean([r["CER"] for r in recs])
            sim = safe_mean([r["Similarity"] for r in recs])
            desc = VOICE_META.get(v, "")
            print(f"  {m:<14} {l:<5} {v:<6} {desc:<16} {wer:>8.4f} {cer:>8.4f} {sim:>8.4f}")
        print("=" * 80)

        # Save master CSV
        master_csv = os.path.join(args.output_dir, "blind_test_all_results.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        print(f"\n✅ Master CSV → {master_csv}")

        # Save master summary JSON
        master_json = os.path.join(args.output_dir, "blind_test_summary.json")
        summary_data = {}
        for (m, l), recs in groups.items():
            key = f"{m}_{l}"
            summary_data[key] = {
                "model": m, "lang": l, "n_samples": len(recs),
                "WER": safe_mean([r["WER"] for r in recs]),
                "CER": safe_mean([r["CER"] for r in recs]),
                "Similarity": safe_mean([r["Similarity"] for r in recs]),
                "InferenceS": safe_mean([r["InferenceS"] for r in recs]),
            }
        with open(master_json, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"✅ Master JSON → {master_json}")


if __name__ == "__main__":
    main()
