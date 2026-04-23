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

import os
import gc
import json
import csv
import time
import argparse
import torch
import torchaudio
# Monkey-patch torchaudio to bypass torchcodec globally
_orig_ta_load = torchaudio.load
_orig_ta_save = torchaudio.save

def _patched_ta_load(filepath, *args, **kwargs):
    try:
        return _orig_ta_load(filepath, *args, **kwargs)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(filepath)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        return torch.from_numpy(data).float(), sr

def _patched_ta_save(filepath, src, sample_rate, *args, **kwargs):
    try:
        return _orig_ta_save(filepath, src, sample_rate, *args, **kwargs)
    except Exception:
        import soundfile as sf
        if isinstance(src, torch.Tensor):
            data = src.detach().cpu().numpy()
        else:
            data = src
        if len(data.shape) == 2:
            data = data.T
        sf.write(filepath, data, sample_rate)

torchaudio.load = _patched_ta_load
torchaudio.save = _patched_ta_save
import numpy as np
from tqdm import tqdm
import warnings
import glob
import torch.nn.functional as F

# --- Monkey Patch HuggingFace Hub ---
import huggingface_hub
if not hasattr(huggingface_hub, "is_offline_mode"):
    huggingface_hub.is_offline_mode = lambda: os.environ.get("HF_HUB_OFFLINE", "0") == "1"

# Fix SpeechBrain passing the removed 'use_auth_token' argument
_orig_download = huggingface_hub.hf_hub_download
def _patched_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_download

# Fix PyTorch 2.6+ weights_only security error when loading XTTS/Coqui models
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ------------------------------------

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
    "omnivoice_finetuned":  ["fr", "ar", "zh"],
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
        wav, sr = safe_load_audio(wav_path)
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

def safe_load_audio(path):
    """Fallback loader to bypass broken torchaudio/torchcodec versions."""
    import soundfile as sf
    try:
        # Try standard torchaudio first
        return torchaudio.load(path)
    except Exception:
        # Fallback to soundfile + torch conversion
        data, sr = sf.read(path)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T # (T, C) -> (C, T)
        return torch.from_numpy(data).float(), sr

def safe_save_audio(path, wav, sr):
    """Fallback saver to bypass broken torchaudio/torchcodec versions."""
    import soundfile as sf
    try:
        # Try standard torchaudio first
        return torchaudio.save(path, wav, sr)
    except Exception:
        # Fallback to soundfile
        if isinstance(wav, torch.Tensor):
            data = wav.detach().cpu().numpy()
        else:
            data = wav
        if len(data.shape) == 2:
            data = data.T # (C, T) -> (T, C)
        sf.write(path, data, sr)

def get_best_reference(ref_path, duration=10.0):
    waveform, sr = safe_load_audio(str(ref_path))
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
    parser.add_argument("--model", required=True, choices=["omnivoice", "omnivoice_finetuned", "chatterbox", "qwen3", "xtts", "voxcpm"])
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
    if args.voices == ["all"]:
        # Auto-discover all speakers from audio directory
        import re
        for wav_file in sorted(glob.glob(os.path.join(args.audio_dir, "*.wav"))):
            basename = os.path.basename(wav_file)
            # Extract voice ID: last number segment before .wav
            # e.g., "2023.acl-long.810.wav" → "810"
            numbers = re.findall(r'(\d+)', basename)
            if numbers:
                vid = numbers[-1]  # Take the LAST number as voice ID
                if vid not in voice_refs:
                    voice_refs[vid] = wav_file
        print(f"  🔍 Auto-discovered {len(voice_refs)} speakers from {args.audio_dir}")
        for vid, ref in voice_refs.items():
            print(f"  ✓ Voice {vid} ({VOICE_META.get(vid, '?')}): {ref}")
    else:
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

    # ════════════════════════════════════════════════════════════
    # PHASE 1: Generate all audio (only TTS model loaded)
    # ════════════════════════════════════════════════════════════
    print(f"\n🔧 Loading {model_name}...")

    # We need whisper for qwen3 only (it uses it internally)
    whisper = None
    if model_name == "qwen3":
        print(f"\n🔎 Loading Whisper {args.whisper_model} (required by Qwen3)...")
        from faster_whisper import WhisperModel
        whisper = WhisperModel(args.whisper_model, device=device,
                               compute_type="float16" if device == "cuda" else "int8")

    if model_name == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        torch.backends.cudnn.enabled = False
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        # Force internal models to GPU (wrapper class doesn't support .to())
        if device == "cuda":
            for attr_name in dir(model):
                attr = getattr(model, attr_name, None)
                if isinstance(attr, torch.nn.Module):
                    attr.to("cuda")
                    print(f"  📍 Moved {attr_name} → cuda")
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_chatterbox(text, ref, lang, dev, model)

    elif model_name in ["omnivoice", "omnivoice_finetuned"]:
        from omnivoice import OmniVoice
        model = None
        gen_fn = None

    elif model_name == "xtts":
        from TTS.api import TTS
        use_gpu = device == "cuda" or str(device).startswith("cuda")
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=use_gpu)
        if use_gpu and hasattr(model, 'synthesizer') and hasattr(model.synthesizer, 'tts_model'):
            model.synthesizer.tts_model.cuda()
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_xtts(text, ref, lang, dev, model)

    elif model_name == "voxcpm":
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        # Move all internal nn.Module subcomponents to GPU
        if device == "cuda":
            for attr_name in dir(model):
                attr = getattr(model, attr_name, None)
                if isinstance(attr, torch.nn.Module):
                    attr.to("cuda")
                    print(f"  📍 Moved {attr_name} → cuda")
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_voxcpm(text, ref, lang, dev, model)

    elif model_name == "qwen3":
        from qwen_tts import Qwen3TTSModel
        try:
            from transformers import BitsAndBytesConfig
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device, dtype=torch.float32, attn_implementation=attn,
        )
        gen_fn = lambda text, ref, lang, dev, ref_tuple: run_qwen3(text, ref, lang, dev, model, whisper)

    print(f"  ✅ {model_name} loaded")

    # --- Generate all audio files ---
    generated_files = {}  # {lang: {(voice_id, idx): syn_path}}

    for lang in langs:
        if lang not in MODEL_LANG_SUPPORT.get(model_name, []):
            print(f"\n⏭ {model_name} does not support {lang}, skipping.")
            continue

        if model_name in ["omnivoice", "omnivoice_finetuned"]:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            print(f"\n🧩 Loading OmniVoice base model and LoRA for {lang.upper()}...")
            # Clean up old model if it exists
            if 'model' in locals() and model is not None:
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
            
            if model_name == "omnivoice_finetuned":
                # Map languages to LoRA repo (using step 400 for all)
                BEST_MODELS = {
                    "zh": "amanuelbyte/omnivoice-lora-zh-400",
                    "ar": "amanuelbyte/omnivoice-lora-ar-400",
                    "fr": "amanuelbyte/omnivoice-lora-fr-400",
                }
                repo_id = BEST_MODELS.get(lang)
                if repo_id:
                    print(f"  📥 Merging LoRA weights from {repo_id}...")
                    try:
                        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
                        sd = load_file(weights_path)
                        merged_sd = {}
                        processed_bases = set()
                        scaling = 64 / 32
                        for k in sd.keys():
                            if ".base_layer.weight" in k:
                                base_key = k.replace("llm.base_model.model.", "llm.")
                                clean_key = base_key.replace(".base_layer", "")
                                la_key = k.replace(".base_layer.weight", ".lora_A.default.weight")
                                lb_key = k.replace(".base_layer.weight", ".lora_B.default.weight")
                                if la_key in sd and lb_key in sd:
                                    A = sd[la_key].to(torch.float32)
                                    B = sd[lb_key].to(torch.float32)
                                    base = sd[k].to(torch.float32)
                                    merged = base + (B @ A) * scaling
                                    merged_sd[clean_key] = merged.to(sd[k].dtype)
                                    processed_bases.add(k)
                                    processed_bases.add(la_key)
                                    processed_bases.add(lb_key)
                        for k in sd.keys():
                            if k not in processed_bases:
                                clean_key = k.replace("llm.base_model.model.", "llm.")
                                merged_sd[clean_key] = sd[k]
                        model.load_state_dict(merged_sd, strict=False)
                        print("  ✅ LoRA Merge successful.")
                    except Exception as e:
                        print(f"  ❌ Failed to merge LoRA: {e}")
            else:
                print(f"  📥 Using Base OmniVoice (no LoRA)...")
            
            model.to(device).eval()
            gen_fn = lambda text, ref, l, dev, ref_tuple, m=model: run_omnivoice(text, ref, l, dev, m, ref_tuple)

        text_file = os.path.join(args.text_dir, LANG_TEXT_MAP.get(lang, f"{lang}.txt"))
        if not os.path.exists(text_file):
            print(f"⚠ Text file not found: {text_file}")
            continue
        with open(text_file, "r", encoding="utf-8") as f:
            text_lines = [l.strip() for l in f if l.strip()]

        out_dir = os.path.join(args.output_dir, model_name, lang)
        os.makedirs(out_dir, exist_ok=True)

        total = len(text_lines) * len(voice_refs)
        print(f"\n📝 {lang.upper()}: {len(text_lines)} lines × {len(voice_refs)} voices = {total} samples")

        generated_files[lang] = {}

        with tqdm(total=total, desc=f"{model_name}/{lang} [GEN]") as pbar:
            for voice_id, raw_ref_path in voice_refs.items():
                ref_wav, ref_sr = get_best_reference(raw_ref_path, duration=10.0)
                temp_ref_path = os.path.join(out_dir, f"temp_ref_{voice_id}.wav")
                safe_save_audio(temp_ref_path, ref_wav, ref_sr)
                ref_tuple = (ref_wav, ref_sr)

                for idx, text in enumerate(text_lines):
                    syn_path = os.path.join(out_dir, f"{lang}_{idx:03d}_{voice_id}.wav")
                    generated_files[lang][(voice_id, idx)] = (syn_path, text, temp_ref_path)

                    if not os.path.exists(syn_path):
                        try:
                            wav_out, sr = gen_fn(text, temp_ref_path, lang, device, ref_tuple)
                            safe_save_audio(syn_path, wav_out, sr)
                        except Exception as e:
                            tqdm.write(f"   ⚠ {voice_id}/line{idx}: {e}")
                    pbar.update(1)

                    if idx % 20 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                gc.collect()
                torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════
    # PHASE 2: Unload TTS model, load eval tools, score everything
    # ════════════════════════════════════════════════════════════
    print(f"\n🗑️  Unloading {model_name} to free VRAM...")
    del model
    if model_name != "qwen3" and whisper is not None:
        del whisper
        whisper = None
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n🔎 Loading Whisper {args.whisper_model} + ECAPA-TDNN for evaluation...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device=device,
                           compute_type="float16" if device == "cuda" else "int8")
    verifier = load_speaker_model(device=device)

    import jiwer
    all_results = []

    for lang in langs:
        if lang not in generated_files:
            continue

        text_file = os.path.join(args.text_dir, LANG_TEXT_MAP.get(lang, f"{lang}.txt"))
        with open(text_file, "r", encoding="utf-8") as f:
            text_lines = [l.strip() for l in f if l.strip()]

        if lang in ("zh", "ar", "ja", "ko"):
            wer_transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        else:
            wer_transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.RemovePunctuation()])

        out_dir = os.path.join(args.output_dir, model_name, lang)
        total = len(generated_files[lang])
        print(f"\n📊 Evaluating {lang.upper()}: {total} samples")

        with tqdm(total=total, desc=f"{model_name}/{lang} [EVAL]") as pbar:
            for (voice_id, idx), (syn_path, text, temp_ref_path) in generated_files[lang].items():
                # ASR
                try:
                    if os.path.exists(syn_path):
                        segs, _ = whisper.transcribe(syn_path, language=lang, beam_size=5, vad_filter=True)
                        hyp = " ".join(s.text for s in segs).strip()
                    else:
                        hyp = ""
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
                    "Similarity": sim, "InferenceS": 0.0,
                })
                pbar.update(1)

                if idx % 20 == 0:
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
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
