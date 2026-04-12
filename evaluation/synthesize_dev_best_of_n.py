#!/usr/bin/env python3
"""
Best-of-N Synthesis Pipeline for OmniVoice Fine-tuning
=======================================================
1) Synthesize dev set with top 3 models per language
2) Score each output (CER + speaker similarity)
3) Select best model per sentence
4) Build JSONL training manifest for OmniVoice fine-tuning

Top 3 models per language (by combined score from eval):
  FR: OmniVoice, Qwen3, Chatterbox
  AR: OmniVoice, Chatterbox, VoxCPM2  (Qwen3 no AR support)
  ZH: OmniVoice, VoxCPM2, Qwen3       (Chatterbox no ZH support)

Usage:
  python evaluation/synthesize_dev_best_of_n.py --lang fr
  python evaluation/synthesize_dev_best_of_n.py --lang ar
  python evaluation/synthesize_dev_best_of_n.py --lang zh
"""

import os
import sys
import gc
import csv
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

import logging
# Silence torchcodec spam
logging.getLogger("torchcodec").setLevel(logging.CRITICAL)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

sys.path.insert(0, os.path.dirname(__file__))
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings("ignore")

# ===================================================================
# Model configurations per language (top 3)
# ===================================================================
MODELS_PER_LANG = {
    "fr": ["omnivoice", "qwen3", "chatterbox"],
    "ar": ["omnivoice", "chatterbox", "voxcpm"],  # Qwen3 doesn't support Arabic
    "zh": ["omnivoice", "voxcpm", "qwen3"],
}

# ===================================================================
# Helper functions
# ===================================================================

def save_wav(audio_array, sr, path):
    """Fast WAV save using soundfile (avoids torchaudio overhead)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.asarray(audio_array, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(path, wav, sr)


def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )


def extract_speaker_embedding(wav_path, model, device="cuda"):
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0).to(device)
        # Handle ultra-short audio outputs that crash ECAPA-TDNN padding
        if wav.shape[-1] < 1600:
            return None
        emb = model.encode_batch(wav.unsqueeze(0))
        return emb.squeeze(0).squeeze(0).detach()
    except Exception:
        return None


def compute_cer(ref_text, hyp_text, lang):
    import jiwer
    if lang in ("zh", "ar", "ja", "ko"):
        transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    else:
        transforms = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(), jiwer.RemovePunctuation()
        ])
    try:
        ref_clean = transforms(ref_text)
        hyp_clean = transforms(hyp_text)
        if not ref_clean.strip():
            return 1.0
        return float(jiwer.cer(ref_clean, hyp_clean))
    except Exception:
        return 1.0


# ===================================================================
# Model loaders and generators
# ===================================================================

def generate_omnivoice(model, text, ref_path):
    with torch.inference_mode():
        audio = model.generate(text=text, ref_audio=ref_path)
    return audio[0].cpu().numpy().squeeze(), 24000


def generate_voxcpm(model, text, ref_path):
    with torch.inference_mode():
        wav = model.generate(text=text, reference_wav_path=ref_path)
    sr = model.tts_model.sample_rate
    return np.asarray(wav, dtype=np.float32), sr


def generate_qwen3(model, text, ref_path, ref_text, lang):
    LANG_MAP = {"fr": "French", "zh": "Chinese", "de": "German"}
    lang_name = LANG_MAP.get(lang, "French")
    
    if hasattr(model, 'generate_voice_clone'):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang_name,
            ref_audio=ref_path,
            ref_text=ref_text,
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        return np.asarray(wav, dtype=np.float32).squeeze(), int(sr)
    else:
        try:
            res = model.generate(
                text=text,
                ref_audio=ref_path,
                ref_text=ref_text,
                language=lang_name,
            )
        except Exception:
            res = model.generate(text=text, ref_audio=ref_path, language=lang_name)
            
        if isinstance(res, tuple):
            wav = res[0]
            sr = res[1] if len(res) > 1 else 24000
        else:
            wav, sr = res, 24000
            
        if isinstance(wav, list):
            wav = wav[0]
            
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
            
        return np.asarray(wav, dtype=np.float32).squeeze(), int(sr)


def generate_chatterbox(model, text, ref_path):
    with torch.inference_mode():
        wav = model.generate(text=text, audio_prompt_path=ref_path)
    return wav.cpu().numpy().squeeze(), 24000


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Best-of-N Synthesis for OmniVoice Fine-tuning")
    parser.add_argument("--lang", required=True, choices=["fr", "ar", "zh"])
    parser.add_argument("--dataset", default="ymoslem/acl-6060")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output-dir", default="./dev_synth")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lang = args.lang
    models_to_run = MODELS_PER_LANG[lang]

    print("=" * 64)
    print(f"  BEST-OF-N SYNTHESIS ({lang})")
    print(f"  Models: {', '.join(models_to_run)}")
    print("=" * 64)

    # ---------------------------------------------------------------
    # Phase 1: Extract dataset
    # ---------------------------------------------------------------
    print(f"\n📥 Phase 1: Extracting dev set")
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total = len(ds)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[:args.max_samples]
        ds = ds.select(indices)
    total = len(ds)
    print(f"   Samples: {total}")

    ref_dir = os.path.join(args.output_dir, "ref_audio")
    os.makedirs(ref_dir, exist_ok=True)

    # Pre-collect all rows first, then write WAVs in parallel
    raw_rows = []
    for i in range(total):
        row = ds[i]
        text_target = (row.get(f"text_{lang}") or "").strip()
        ref_data = row.get("audio")
        if not ref_data or not text_target:
            continue
        ref_path = os.path.join(ref_dir, f"ref_{i:05d}.wav")
        raw_rows.append((i, text_target, ref_data, ref_path))

    # Parallel WAV extraction (I/O bound, so threads work great)
    def _extract_one(args_tuple):
        i, text_target, ref_data, ref_path = args_tuple
        if not os.path.exists(ref_path):
            wav = np.asarray(ref_data["array"], dtype=np.float32)
            sf.write(ref_path, wav, ref_data["sampling_rate"])
        return {"idx": i, "text_target": text_target,
                "ref_path": ref_path, "ref_sr": ref_data["sampling_rate"]}

    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for entry in tqdm(pool.map(_extract_one, raw_rows),
                          total=len(raw_rows), desc="Extracting"):
            manifest.append(entry)

    del ds, raw_rows; gc.collect()
    print(f"   Extracted {len(manifest)} samples.")

    # ---------------------------------------------------------------
    # Phase 2: Synthesize with each model
    # ---------------------------------------------------------------
    all_outputs = {}  # model_name -> {idx: wav_path}

    for model_name in models_to_run:
        print(f"\n🎙  Phase 2: Generating with {model_name}")
        out_dir = os.path.join(args.output_dir, model_name, lang)
        os.makedirs(out_dir, exist_ok=True)
        all_outputs[model_name] = {}

        try:
            if model_name == "omnivoice":
                # Monkey-patch load_audio to use torchaudio instead of torchcodec
                import omnivoice.utils.audio as _omni_audio
                def _load_audio_torchaudio(audio_path: str, sampling_rate: int):
                    waveform, sr = torchaudio.load(audio_path, backend="soundfile")
                    if sr != sampling_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    return waveform  # shape: (1, T)
                _omni_audio.load_audio = _load_audio_torchaudio

                # Also patch it in the model module so the already-imported reference updates
                import omnivoice.models.omnivoice as _omni_model
                _omni_model.load_audio = _load_audio_torchaudio

                from omnivoice import OmniVoice
                model = OmniVoice.from_pretrained(
                    "k2-fsa/OmniVoice", device_map=f"{device}:0", dtype=torch.float32)

                for entry in tqdm(manifest, desc=f"  {model_name}"):
                    syn_path = os.path.join(out_dir, f"synth_{entry['idx']:05d}.wav")
                    if os.path.exists(syn_path):
                        all_outputs[model_name][entry["idx"]] = syn_path
                        continue
                    try:
                        wav, sr = generate_omnivoice(model, entry["text_target"], entry["ref_path"])
                        save_wav(wav, sr, syn_path)
                        all_outputs[model_name][entry["idx"]] = syn_path
                    except Exception as e:
                        print(f"   ⚠ {model_name} sample {entry['idx']}: {e}")

                del model; gc.collect(); torch.cuda.empty_cache()

            elif model_name == "voxcpm":
                from voxcpm import VoxCPM
                model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

                for entry in tqdm(manifest, desc=f"  {model_name}"):
                    syn_path = os.path.join(out_dir, f"synth_{entry['idx']:05d}.wav")
                    if os.path.exists(syn_path):
                        all_outputs[model_name][entry["idx"]] = syn_path
                        continue
                    try:
                        wav, sr = generate_voxcpm(model, entry["text_target"], entry["ref_path"])
                        sf.write(syn_path, wav, sr)
                        all_outputs[model_name][entry["idx"]] = syn_path
                    except Exception as e:
                        print(f"   ⚠ {model_name} sample {entry['idx']}: {e}")

                del model; gc.collect(); torch.cuda.empty_cache()

            elif model_name == "qwen3":
                try:
                    # Monkey-patch check_model_inputs to support both
                    # @check_model_inputs and @check_model_inputs() syntax
                    import transformers.modeling_utils as _tmu
                    if hasattr(_tmu, 'check_model_inputs'):
                        _orig_cmi = _tmu.check_model_inputs
                        def _patched_cmi(*args, **kwargs):
                            if len(args) == 1 and callable(args[0]) and not kwargs:
                                return _orig_cmi(args[0])
                            def wrapper(func):
                                return _orig_cmi(func)
                            return wrapper
                        _tmu.check_model_inputs = _patched_cmi

                    # Monkey-patch Qwen Configs to avoid pad_token_id missing error in transformers 5.5
                    try:
                        import qwen_tts.core.models.configuration_qwen3_tts as qcfg
                        qcfg.Qwen3TTSConfig.pad_token_id = None
                        qcfg.Qwen3TTSTalkerConfig.pad_token_id = None
                    except Exception:
                        pass
                        
                    # Monkey-patch ROPE_INIT_FUNCTIONS missing 'default' in transformers 5.5
                    try:
                        import transformers.modeling_rope_utils as _rope_utils
                        if "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:
                            _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _rope_utils.ROPE_INIT_FUNCTIONS.get("linear", list(_rope_utils.ROPE_INIT_FUNCTIONS.values())[0])
                    except Exception:
                        pass

                    from qwen_tts import Qwen3TTSModel
                    try:
                        import flash_attn
                        attn_impl = "flash_attention_2"
                    except ImportError:
                        attn_impl = "sdpa"
                    model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        device_map=device,
                        dtype=torch.float32,
                        attn_implementation=attn_impl,
                    )
                except Exception as qwen_err:
                    print(f"   qwen_tts library failed ({qwen_err}), trying transformers...")
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        device_map="auto",
                        torch_dtype=torch.float32,
                        trust_remote_code=True)

                for entry in tqdm(manifest, desc=f"  {model_name}"):
                    syn_path = os.path.join(out_dir, f"synth_{entry['idx']:05d}.wav")
                    if os.path.exists(syn_path):
                        all_outputs[model_name][entry["idx"]] = syn_path
                        continue
                    try:
                        wav, sr = generate_qwen3(model, entry["text_target"],
                                                 entry["ref_path"],
                                                 entry.get("ref_en_text", ""),
                                                 lang)
                        save_wav(wav, sr, syn_path)
                        all_outputs[model_name][entry["idx"]] = syn_path
                    except Exception as e:
                        print(f"   ⚠ {model_name} sample {entry['idx']}: {e}")

                del model; gc.collect(); torch.cuda.empty_cache()

            elif model_name == "chatterbox":
                from chatterbox.tts import ChatterboxTTS
                model = ChatterboxTTS.from_pretrained(device=device)

                for entry in tqdm(manifest, desc=f"  {model_name}"):
                    syn_path = os.path.join(out_dir, f"synth_{entry['idx']:05d}.wav")
                    if os.path.exists(syn_path):
                        all_outputs[model_name][entry["idx"]] = syn_path
                        continue
                    try:
                        wav, sr = generate_chatterbox(model, entry["text_target"], entry["ref_path"])
                        save_wav(wav, sr, syn_path)
                        all_outputs[model_name][entry["idx"]] = syn_path
                    except Exception as e:
                        print(f"   ⚠ {model_name} sample {entry['idx']}: {e}")

                del model; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
            continue

    # ---------------------------------------------------------------
    # Phase 3: Score all outputs
    # ---------------------------------------------------------------
    print(f"\n📊 Phase 3: Scoring outputs")

    # Load ASR — use beam_size=1 for 3x faster scoring (quality diff is negligible)
    from faster_whisper import WhisperModel as FasterWhisperModel
    whisper = FasterWhisperModel(args.whisper_model, device=device,
                                compute_type="float16" if device == "cuda" else "int8")

    # Load speaker verifier
    verifier = load_speaker_model(device=device)

    # Pre-compute ALL reference embeddings in one pass (avoids redundant recomputation)
    print("   Pre-computing reference speaker embeddings...")
    ref_emb_cache = {}
    for entry in tqdm(manifest, desc="  Ref embeddings", leave=False):
        ref_emb_cache[entry["idx"]] = extract_speaker_embedding(
            entry["ref_path"], verifier, device)

    scores = []  # list of dicts
    for entry in tqdm(manifest, desc="Scoring"):
        idx = entry["idx"]
        ref_emb = ref_emb_cache.get(idx)

        row_scores = {
            "idx": idx,
            "lang": lang,
            "text": entry["text_target"],
            "ref_path": entry["ref_path"],
        }

        best_score = -1.0
        best_model = None
        best_wav = None

        for model_name in models_to_run:
            wav_path = all_outputs.get(model_name, {}).get(idx)
            if not wav_path or not os.path.exists(wav_path):
                row_scores[f"{model_name}_cer"] = None
                row_scores[f"{model_name}_sim"] = None
                row_scores[f"{model_name}_combined"] = None
                continue

            # Transcribe — beam_size=1 is ~3x faster, negligible quality loss for scoring
            try:
                segments, _ = whisper.transcribe(wav_path, language=lang,
                                                  beam_size=1, vad_filter=True)
                hyp = " ".join(seg.text for seg in segments).strip()
            except Exception:
                hyp = ""

            # CER
            cer = compute_cer(entry["text_target"], hyp, lang) if hyp else 1.0

            # Speaker similarity
            syn_emb = extract_speaker_embedding(wav_path, verifier, device)
            if syn_emb is not None and ref_emb is not None:
                sim = float(F.cosine_similarity(
                    syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
            else:
                sim = 0.0

            # Combined score: 50% content accuracy + 50% voice similarity
            combined = 0.5 * (1.0 - min(cer, 1.0)) + 0.5 * max(sim, 0.0)

            row_scores[f"{model_name}_cer"] = cer
            row_scores[f"{model_name}_sim"] = sim
            row_scores[f"{model_name}_combined"] = combined
            row_scores[f"{model_name}_wav"] = wav_path

            if combined > best_score:
                best_score = combined
                best_model = model_name
                best_wav = wav_path

        row_scores["best_model"] = best_model
        row_scores["best_score"] = best_score
        row_scores["best_wav"] = best_wav
        scores.append(row_scores)

    del whisper, verifier, ref_emb_cache; gc.collect(); torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Phase 4: Save scores and build JSONL manifest
    # ---------------------------------------------------------------
    print(f"\n📝 Phase 4: Building training manifest")

    # Save scores CSV
    scores_path = os.path.join(args.output_dir, f"scores_{lang}.csv")
    fieldnames = list(scores[0].keys()) if scores else []
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)
    print(f"   Scores saved to {scores_path}")

    # Build JSONL for OmniVoice fine-tuning
    jsonl_path = os.path.join(args.output_dir, f"train_{lang}.jsonl")
    count = 0
    model_counts = {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in scores:
            if s["best_wav"] and s["best_model"]:
                entry = {
                    "id": f"{lang}_{s['idx']:05d}",
                    "audio_path": os.path.abspath(s["best_wav"]),
                    "text": s["text"],
                    "language_id": lang,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                model_counts[s["best_model"]] = model_counts.get(s["best_model"], 0) + 1

    print(f"   Training JSONL saved to {jsonl_path} ({count} samples)")

    # ---------------------------------------------------------------
    # Final Summary Table
    # ---------------------------------------------------------------
    # Compute per-model stats
    model_stats = {}
    for model_name in models_to_run:
        cers = [s.get(f"{model_name}_cer") for s in scores
                if s.get(f"{model_name}_cer") is not None]
        sims = [s.get(f"{model_name}_sim") for s in scores
                if s.get(f"{model_name}_sim") is not None]
        combos = [s.get(f"{model_name}_combined") for s in scores
                  if s.get(f"{model_name}_combined") is not None]
        wins = model_counts.get(model_name, 0)
        model_stats[model_name] = {
            "wins": wins,
            "pct": 100 * wins / count if count > 0 else 0,
            "avg_cer": np.mean(cers) if cers else np.nan,
            "avg_sim": np.mean(sims) if sims else np.nan,
            "avg_combined": np.mean(combos) if combos else np.nan,
            "generated": len(cers),
        }

    valid_scores = [s["best_score"] for s in scores if s["best_score"] > 0]

    print("\n")
    print("=" * 72)
    print(f"  BEST-OF-N RESULTS — {lang.upper()} ({count} samples)")
    print("=" * 72)
    print()
    print("  Scoring formula:")
    print("    Combined = 0.5 × (1 - CER) + 0.5 × Speaker_Similarity")
    print("    → Higher = better (max 1.0)")
    print()
    print("-" * 72)
    print(f"  {'Model':<14} {'Rows Won':>10} {'% of Data':>10} {'Avg CER':>10} "
          f"{'Avg Sim':>10} {'Avg Score':>10}")
    print("-" * 72)

    for model_name in sorted(model_stats, key=lambda m: -model_stats[m]["wins"]):
        st = model_stats[model_name]
        print(f"  {model_name:<14} {st['wins']:>10} {st['pct']:>9.1f}% "
              f"{st['avg_cer']:>10.4f} {st['avg_sim']:>10.4f} "
              f"{st['avg_combined']:>10.4f}")

    print("-" * 72)
    if valid_scores:
        print(f"  {'BEST-OF-N':<14} {count:>10} {'100.0':>9}% "
              f"{'—':>10} {'—':>10} {np.mean(valid_scores):>10.4f}")
    print("=" * 72)
    print()

    # Also save the table as JSON
    summary = {
        "lang": lang,
        "total_samples": count,
        "scoring_formula": "0.5 * (1 - CER) + 0.5 * Speaker_Similarity",
        "avg_best_of_n_score": float(np.mean(valid_scores)) if valid_scores else None,
        "model_stats": {m: {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in st.items()}
                        for m, st in model_stats.items()},
    }
    summary_path = os.path.join(args.output_dir, f"summary_{lang}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    print(f"\n✅ Done for {lang}!")


if __name__ == "__main__":
    main()
