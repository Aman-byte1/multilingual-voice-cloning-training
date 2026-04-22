#!/usr/bin/env python3
"""
Blind Test Evaluation — Single Model Runner
=============================================
Evaluates ONE model at a time on the blind test dataset × selected voices.
This avoids dependency conflicts between models.

Usage:
    python evaluation/evaluate_blind_single.py --model chatterbox --lang fr
    python evaluation/evaluate_blind_single.py --model xtts --lang all
    python evaluation/evaluate_blind_single.py --model omnivoice --lang all
    python evaluation/evaluate_blind_single.py --model voxcpm --lang all
    python evaluation/evaluate_blind_single.py --model qwen3 --lang fr
"""

import os, sys, gc, csv, json, time, argparse, warnings, glob
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# ── Voices ───────────────────────────────────────────────────────
SELECTED_VOICES = ["193", "14", "810", "289"]
VOICE_META = {
    "193": "Asian Female",
    "14":  "Western Female",
    "810": "Western Male",
    "289": "Asian Male",
}

LANG_TEXT_MAP = {"zh": "chinese.txt", "ar": "arabic.txt", "fr": "french.txt"}

MODEL_LANG_SUPPORT = {
    "omnivoice":  ["fr", "ar", "zh"],
    "chatterbox": ["fr", "ar", "zh"],
    "qwen3":      ["fr", "zh"],
    "xtts":       ["fr", "ar", "zh"],
    "voxcpm":     ["fr", "ar", "zh"],
}

XTTS_LANG_MAP = {"fr": "fr", "ar": "ar", "zh": "zh-cn"}
QWEN_LANG_MAP = {"fr": "French", "zh": "Chinese"}


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

def find_ref_audio(audio_dir, voice_id):
    patterns = [f"{voice_id}.wav", f"speaker_{voice_id}.wav", f"spk_{voice_id}.wav"]
    for pat in patterns:
        p = os.path.join(audio_dir, pat)
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(audio_dir, f"*{voice_id}*.wav"))
    return matches[0] if matches else None

def get_best_reference(ref_path, duration=10.0):
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
    return (waveform[:, start_idx:end_idx], sr)


# ── Model-Specific Generation ───────────────────────────────────

def run_chatterbox(text, ref_path, lang, device, model):
    with torch.inference_mode():
        wav = model.generate(
            text, language_id=lang, audio_prompt_path=ref_path,
            exaggeration=0.5, cfg_weight=0.0,
        )
    wav_out = wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu()
    return wav_out, model.sr

def run_omnivoice(text, ref_path, lang, device, model, ref_tuple):
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

def run_xtts(text, ref_path, lang, device, model):
    import tempfile
    xtts_lang = XTTS_LANG_MAP.get(lang, lang)
    fd, syn_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    model.tts_to_file(text=text, speaker_wav=ref_path, language=xtts_lang, file_path=syn_path)
    wav, sr = torchaudio.load(syn_path)
    os.remove(syn_path)
    return wav, sr

def run_voxcpm(text, ref_path, lang, device, model):
    wav = model.generate(
        text=text, reference_wav_path=ref_path, cfg_value=2.0, inference_timesteps=10,
    )
    sr = model.tts_model.sample_rate
    if isinstance(wav, np.ndarray):
        wav_t = torch.from_numpy(wav).float()
    else:
        wav_t = wav
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    return wav_t, sr

# Cache for reference transcripts to avoid re-transcribing same voice
REF_TRANSCRIPT_CACHE = {}

def run_qwen3(text, ref_path, lang, device, model, whisper_model):
    import soundfile as sf
    qwen_lang = QWEN_LANG_MAP[lang]
    
    with torch.inference_mode():
        if hasattr(model, 'generate_voice_clone'):
            # Use x_vector mode to avoid needing ref_text (pure audio-based cloning)
            wavs, sr = model.generate_voice_clone(
                text=text, language=qwen_lang, ref_audio=ref_path, x_vector_only_mode=True,
            )
            
            wav_np = wavs[0] if isinstance(wavs, list) else wavs
            if isinstance(wav_np, torch.Tensor):
                wav_np = wav_np.cpu().numpy()
            if wav_np.ndim > 1:
                wav_np = wav_np.squeeze()
            return torch.from_numpy(wav_np).float().unsqueeze(0), sr
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
                return torch.from_numpy(wav_np).float().unsqueeze(0), 24000
            return wavs.unsqueeze(0).cpu().float() if wavs.dim() == 1 else wavs.cpu().float(), 24000


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blind Test — Single Model Evaluation")
    parser.add_argument("--model", required=True, choices=["omnivoice", "chatterbox", "qwen3", "xtts", "voxcpm"])
    parser.add_argument("--lang", nargs="+", default=["all"], help="fr, ar, zh, or all")
    parser.add_argument("--text-dir", default="./blind_test/text")
    parser.add_argument("--audio-dir", default="./blind_test/audio")
    parser.add_argument("--output-dir", default="./eval_blind_test")
    parser.add_argument("--voices", nargs="+", default=SELECTED_VOICES)
    parser.add_argument("--whisper-model", default="large-v3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model
    
    if args.lang == ["all"]:
        langs = ["fr", "ar", "zh"]
    else:
        langs = args.lang

    print("=" * 64)
    print(f"  BLIND TEST — {model_name.upper()}")
    print(f"  Languages: {langs} | Voices: {args.voices}")
    print("=" * 64)

    # Resolve voice refs
    voice_refs = {}
    for vid in args.voices:
        ref = find_ref_audio(args.audio_dir, vid)
        if ref:
            voice_refs[vid] = ref
            print(f"  ✓ Voice {vid} ({VOICE_META.get(vid, '?')}): {ref}")
        else:
            print(f"  ✗ Voice {vid}: NOT FOUND")
    if not voice_refs:
        print("❌ No voice references found.")
        sys.exit(1)

    # Load shared eval tools
    print(f"\n🔎 Loading Whisper {args.whisper_model} + ECAPA-TDNN...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device=device,
                           compute_type="float16" if device == "cuda" else "int8")
    verifier = load_speaker_model(device=device)

    # Load model
    print(f"\n🔧 Loading {model_name}...")
    if model_name == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        torch.backends.cudnn.enabled = False
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_chatterbox(text, ref, lang, dev, model)

    elif model_name == "omnivoice":
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
        model.to(device).eval()
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_omnivoice(text, ref, lang, dev, model, ref_tuple)

    elif model_name == "xtts":
        from TTS.api import TTS
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_xtts(text, ref, lang, dev, model)

    elif model_name == "voxcpm":
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device=device)
        # Force the internal model to the device if it stuck on CPU
        if hasattr(model, 'tts_model'):
            model.tts_model.to(device)
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_voxcpm(text, ref, lang, dev, model)

    elif model_name == "qwen3":
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
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_qwen3(text, ref, lang, dev, model, whisper)

    print(f"  ✅ {model_name} loaded")

    # Text normalization
    import jiwer
    all_results = []

    for lang in langs:
        if lang not in MODEL_LANG_SUPPORT.get(model_name, []):
            print(f"\n⏭ {model_name} does not support {lang}, skipping.")
            continue

        text_file = os.path.join(args.text_dir, LANG_TEXT_MAP.get(lang, f"{lang}.txt"))
        if not os.path.exists(text_file):
            print(f"⚠ Text file not found: {text_file}")
            continue
        with open(text_file, "r", encoding="utf-8") as f:
            text_lines = [l.strip() for l in f if l.strip()]

        if lang in ("zh", "ar", "ja", "ko"):
            wer_transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        else:
            wer_transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.RemovePunctuation()])

        out_dir = os.path.join(args.output_dir, model_name, lang)
        os.makedirs(out_dir, exist_ok=True)

        total = len(text_lines) * len(voice_refs)
        print(f"\n📝 {lang.upper()}: {len(text_lines)} lines × {len(voice_refs)} voices = {total} samples")

        with tqdm(total=total, desc=f"{model_name}/{lang}") as pbar:
            for voice_id, raw_ref_path in voice_refs.items():
                # Extract and save the best 10s segment to a temporary file
                # This ensures consistent audio/transcript for all models
                ref_wav, ref_sr = get_best_reference(raw_ref_path, duration=10.0)
                temp_ref_path = os.path.join(out_dir, f"temp_ref_{voice_id}.wav")
                torchaudio.save(temp_ref_path, ref_wav, ref_sr)
                
                ref_tuple = (ref_wav, ref_sr)

                for idx, text in enumerate(text_lines):
                    syn_path = os.path.join(out_dir, f"{lang}_{idx:03d}_{voice_id}.wav")

                    # Generate (resume-friendly)
                    if not os.path.exists(syn_path):
                        try:
                            t0 = time.perf_counter()
                            # Use temp_ref_path for all models now
                            wav_out, sr = gen_fn(text, temp_ref_path, lang, device, ref_tuple)
                            t1 = time.perf_counter()
                            torchaudio.save(syn_path, wav_out, sr)
                            inf_time = t1 - t0
                        except Exception as e:
                            tqdm.write(f"   ⚠ {voice_id}/line{idx}: {e}")
                            pbar.update(1)
                            continue
                    else:
                        inf_time = 0.0

                    # ASR
                    try:
                        segs, _ = whisper.transcribe(syn_path, language=lang, beam_size=5, vad_filter=True)
                        hyp = " ".join(s.text for s in segs).strip()
                    except Exception:
                        hyp = ""

                    # WER/CER
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

                    # Speaker Similarity
                    syn_emb = extract_speaker_embedding(syn_path, verifier, device)
                    ref_emb = extract_speaker_embedding(temp_ref_path, verifier, device)
                    if syn_emb is not None and ref_emb is not None:
                        sim = float(F.cosine_similarity(syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
                    else:
                        sim = None

                    all_results.append({
                        "model": model_name, "lang": lang, "voice_id": voice_id,
                        "voice_desc": VOICE_META.get(voice_id, ""),
                        "text_idx": idx, "WER": wer, "CER": cer,
                        "Similarity": sim, "InferenceS": inf_time,
                    })
                    pbar.update(1)

                    if idx % 20 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                # Clear cache between voices
                gc.collect()
                torch.cuda.empty_cache()

        # Per-language summary
        lang_results = [r for r in all_results if r["lang"] == lang]
        if lang_results:
            summary = {
                "model": model_name, "lang": lang, "n_samples": len(lang_results),
                "WER": safe_mean([r["WER"] for r in lang_results]),
                "CER": safe_mean([r["CER"] for r in lang_results]),
                "Similarity": safe_mean([r["Similarity"] for r in lang_results]),
                "InferenceS": safe_mean([r["InferenceS"] for r in lang_results]),
            }
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"   ✅ {model_name}/{lang}: WER={summary['WER']:.4f}  CER={summary['CER']:.4f}  SIM={summary['Similarity']:.4f}")

    # Save master CSV for this model
    if all_results:
        master_csv = os.path.join(args.output_dir, model_name, "all_results.csv")
        os.makedirs(os.path.dirname(master_csv), exist_ok=True)
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        print(f"\n✅ Results saved → {master_csv}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
