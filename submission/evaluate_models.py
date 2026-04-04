#!/usr/bin/env python3
"""
🚀 IWSLT 2026 — Multi-Model Dev Evaluation & Comparison
=========================================================
Runs all candidate models on dev data and produces a comparison table.
Computes the 3 official IWSLT metrics:
  1. Content Consistency (WER via ASR)
  2. Speaker Similarity (cosine sim of ECAPA-TDNN embeddings)
  3. Speech Quality (UTMOS neural MOS)

Usage:
    python evaluate_models.py --models qwen cosyvoice xtts \
                              --languages fr \
                              --output-dir ./eval_comparison
"""

import os
import sys
import gc
import json
import csv
import argparse
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ModelComparison")

torch.set_float32_matmul_precision("high")

LANG_CONFIG = {
    "ar": {"name": "Arabic",  "whisper_lang": "ar"},
    "zh": {"name": "Chinese", "whisper_lang": "zh"},
    "fr": {"name": "French",  "whisper_lang": "fr"},
}


# ═══════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════

def load_audio_16k(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = librosa.load(path, sr=16000)
    return wav, 16000


def load_audio_tensor_16k(path: str, device: str = "cpu") -> torch.Tensor:
    signal, fs = torchaudio.load(path)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.to(device)


def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval") -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path


def load_speaker_model(device: str = "cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def extract_embedding(wav_path: str, verifier, device: str = "cuda"):
    try:
        signal = load_audio_tensor_16k(wav_path, device=device)
        embedding = verifier.encode_batch(signal)
        return embedding.squeeze().cpu()
    except Exception:
        return None


def compute_similarity(emb1, emb2) -> Optional[float]:
    if emb1 is None or emb2 is None:
        return None
    return float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())


def load_utmos(device: str = "cuda"):
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        predictor = predictor.to(device).eval()
        log.info("UTMOS loaded ✓")
        return predictor
    except Exception as e:
        log.warning(f"UTMOS failed: {e}")
        return None


def compute_utmos(wav_path: str, predictor, device: str = "cuda") -> Optional[float]:
    if predictor is None:
        return None
    try:
        wav = load_audio_tensor_16k(wav_path, device=device)
        with torch.no_grad():
            score = predictor(wav, sr=16000)
        return float(score.item())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# MODEL GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_qwen(text: str, ref_path: str, lang: str, model, **kwargs) -> str:
    """Generate with Qwen3-TTS, return path to generated WAV."""
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="qwen_")
    os.close(fd)

    lang_map = {"ar": "Arabic", "zh": "Chinese", "fr": "French"}
    ref_text = kwargs.get("ref_text", "")

    if hasattr(model, 'generate_voice_clone'):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang_map.get(lang, "English"),
            ref_audio=ref_path,
            ref_text=ref_text,
        )
        sf.write(out_path, wavs[0], sr)
    else:
        wavs = model.generate(
            text=text,
            ref_audio_path=ref_path,
            ref_text=ref_text,
            language=lang_map.get(lang, "English"),
        )
        if isinstance(wavs, torch.Tensor):
            wav_np = wavs.cpu().numpy().squeeze()
            sf.write(out_path, wav_np, 24000)

    return out_path


def generate_cosyvoice(text: str, ref_path: str, lang: str, model, **kwargs) -> str:
    """Generate with CosyVoice3."""
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="cosyvoice_")
    os.close(fd)

    if hasattr(model, 'inference_cross_lingual'):
        for result in model.inference_cross_lingual(
            tts_text=text, prompt_speech_16k=ref_path
        ):
            wav = result['tts_speech']
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            sf.write(out_path, wav.squeeze(), 22050)
            break
    elif hasattr(model, 'inference_zero_shot'):
        for result in model.inference_zero_shot(
            tts_text=text, prompt_text="", prompt_speech_16k=ref_path
        ):
            wav = result['tts_speech']
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            sf.write(out_path, wav.squeeze(), 22050)
            break

    return out_path


def generate_xtts(text: str, ref_path: str, lang: str, model, **kwargs) -> str:
    """Generate with XTTS-v2."""
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="xtts_")
    os.close(fd)

    model.tts_to_file(
        text=text,
        speaker_wav=ref_path,
        language=lang,
        file_path=out_path,
    )
    return out_path


def generate_chatterbox(text: str, ref_path: str, lang: str, model, **kwargs) -> str:
    """Generate with Chatterbox (base or LoRA)."""
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="cbox_")
    os.close(fd)

    with torch.inference_mode():
        wav = model.generate(text, audio_prompt_path=ref_path, language_id=lang)
    wav_out = wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu()
    torchaudio.save(out_path, wav_out, model.sr)
    return out_path


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_on_dataset(
    generate_fn,
    model,
    dataset_name: str,
    lang: str,
    max_samples: int = 50,
    device: str = "cuda",
    whisper_model_name: str = "large-v3",
) -> Dict:
    """Evaluate a single model on a dataset split.

    Returns dict with per-sample and aggregated metrics.
    """
    from datasets import load_dataset
    import jiwer

    log.info(f"  Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="test")

    total = len(ds)
    if max_samples and max_samples < total:
        step = max(1, total // max_samples)
        indices = list(range(0, total, step))[:max_samples]
        ds = ds.select(indices)
    total = len(ds)
    log.info(f"  Evaluating {total} samples")

    # Phase 1: Generate
    samples = []
    for i in tqdm(range(total), desc="Generating"):
        row = ds[i]
        text = (row.get("trg_fr_text") or row.get("text_fr") or "").strip()
        ref_text = (row.get("ref_en_text") or row.get("text_en") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get("audio_en")

        if not text or not ref_data:
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"], "ref_"
        )

        # Ground truth for speaker similarity
        gt_data = row.get("trg_fr_voice") or row.get("cloned_audio_fr")
        gt_path = None
        if gt_data:
            gt_path = save_temp_wav(
                np.asarray(gt_data["array"], dtype=np.float32),
                gt_data["sampling_rate"], "gt_"
            )

        try:
            syn_path = generate_fn(
                text=text, ref_path=ref_path, lang=lang, model=model,
                ref_text=ref_text,
            )
            samples.append({
                "idx": i,
                "syn_path": syn_path,
                "ref_path": ref_path,
                "gt_path": gt_path,
                "text": text,
            })
        except Exception as e:
            log.warning(f"    Sample {i} failed: {e}")
            os.remove(ref_path)
            if gt_path:
                os.remove(gt_path)

    if not samples:
        return {"error": "No samples generated"}

    # Free model GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2: ASR
    log.info(f"  Transcribing {len(samples)} samples...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(whisper_model_name, device=device, compute_type="float16")

    whisper_lang = LANG_CONFIG[lang]["whisper_lang"]
    transcripts = []
    for s in tqdm(samples, desc="ASR"):
        try:
            segments, _ = whisper.transcribe(
                s["syn_path"], language=whisper_lang, beam_size=5
            )
            text = " ".join(seg.text.strip() for seg in segments)
        except Exception:
            text = ""
        transcripts.append(text)

    del whisper
    torch.cuda.empty_cache()

    # Phase 3: Metrics
    log.info("  Computing metrics...")
    verifier = load_speaker_model(device=device)
    utmos_predictor = load_utmos(device=device)

    wer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    results = []
    for s, tx in zip(samples, transcripts):
        # WER
        try:
            wer = float(jiwer.wer(
                s["text"], tx,
                truth_transform=wer_transforms,
                hypothesis_transform=wer_transforms,
            )) if tx.strip() else 1.0
        except Exception:
            wer = None

        # Speaker similarity (synth vs ref audio — cross-lingual)
        ref_emb = extract_embedding(s["ref_path"], verifier, device)
        syn_emb = extract_embedding(s["syn_path"], verifier, device)
        sim = compute_similarity(ref_emb, syn_emb)

        # UTMOS
        utmos = compute_utmos(s["syn_path"], utmos_predictor, device)

        results.append({
            "idx": s["idx"],
            "WER": wer,
            "SpeakerSim": sim,
            "UTMOS": utmos,
            "transcript": tx,
            "reference": s["text"],
        })

        # Cleanup temp files
        for key in ["syn_path", "ref_path", "gt_path"]:
            if s.get(key) and os.path.exists(s[key]):
                os.remove(s[key])

    del verifier, utmos_predictor
    torch.cuda.empty_cache()

    # Aggregate
    def safe_mean(vals):
        valid = [v for v in vals if v is not None]
        return float(np.mean(valid)) if valid else None

    summary = {
        "WER": safe_mean([r["WER"] for r in results]),
        "SpeakerSim": safe_mean([r["SpeakerSim"] for r in results]),
        "UTMOS": safe_mean([r["UTMOS"] for r in results]),
        "num_evaluated": len(results),
    }

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="IWSLT 2026 — Multi-Model Evaluation Comparison"
    )
    parser.add_argument(
        "--models", nargs="+", default=["qwen"],
        choices=["qwen", "cosyvoice", "xtts", "chatterbox"],
    )
    parser.add_argument("--languages", nargs="+", default=["fr"],
                        choices=["ar", "zh", "fr"])
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-data")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--output-dir", default="./eval_comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    MODEL_LOADERS = {
        "qwen": lambda: _load_qwen(),
        "cosyvoice": lambda: _load_cosyvoice(),
        "xtts": lambda: _load_xtts(),
        "chatterbox": lambda: _load_chatterbox(),
    }

    GENERATE_FNS = {
        "qwen": generate_qwen,
        "cosyvoice": generate_cosyvoice,
        "xtts": generate_xtts,
        "chatterbox": generate_chatterbox,
    }

    all_results = {}

    for model_name in args.models:
        for lang in args.languages:
            log.info(f"\n{'='*60}")
            log.info(f"  Evaluating: {model_name.upper()} on {lang.upper()}")
            log.info(f"{'='*60}")

            try:
                model = MODEL_LOADERS[model_name]()
                result = evaluate_on_dataset(
                    generate_fn=GENERATE_FNS[model_name],
                    model=model,
                    dataset_name=args.dataset,
                    lang=lang,
                    max_samples=args.max_samples,
                    device=args.device,
                    whisper_model_name=args.whisper_model,
                )
                key = f"{model_name}_{lang}"
                all_results[key] = result

                # Save individual result
                with open(os.path.join(args.output_dir, f"{key}.json"), "w") as f:
                    json.dump(result, f, indent=2, default=str)

            except Exception as e:
                log.error(f"  Failed: {e}")
                all_results[f"{model_name}_{lang}"] = {"error": str(e)}

            torch.cuda.empty_cache()
            gc.collect()

    # ── Print comparison table ──
    print(f"\n{'='*72}")
    print(f"  IWSLT 2026 VOICE CLONING — MODEL COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Model':<15} {'Lang':<5} {'WER ↓':<10} {'SpkSim ↑':<10} {'UTMOS ↑':<10}")
    print(f"  {'-'*55}")

    for key, res in all_results.items():
        model_name, lang = key.rsplit("_", 1)
        if "error" in res:
            print(f"  {model_name:<15} {lang:<5} {'ERROR':^30}")
        else:
            s = res["summary"]
            wer = f"{s['WER']:.4f}" if s["WER"] is not None else "N/A"
            sim = f"{s['SpeakerSim']:.4f}" if s["SpeakerSim"] is not None else "N/A"
            utmos = f"{s['UTMOS']:.4f}" if s["UTMOS"] is not None else "N/A"
            print(f"  {model_name:<15} {lang:<5} {wer:<10} {sim:<10} {utmos:<10}")

    print(f"{'='*72}")

    # Save comparison
    comp_path = os.path.join(args.output_dir, "comparison.json")
    comp_data = {}
    for key, res in all_results.items():
        if "summary" in res:
            comp_data[key] = res["summary"]
    with open(comp_path, "w") as f:
        json.dump(comp_data, f, indent=2)
    log.info(f"Comparison saved to {comp_path}")


# ═══════════════════════════════════════════════════════════════
# MODEL LOADERS (lazy — only import when needed)
# ═══════════════════════════════════════════════════════════════

def _load_qwen():
    from qwen_tts import Qwen3TTSModel
    # Try flash_attention_2 first, fall back to sdpa
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        log.info("flash-attn not installed, using SDPA attention")
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

def _load_cosyvoice():
    from cosyvoice.cli.cosyvoice import CosyVoice
    return CosyVoice("CosyVoice3-0.5B")

def _load_xtts():
    from TTS.api import TTS
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def _load_chatterbox():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    model.t3.eval()
    return model


if __name__ == "__main__":
    main()
