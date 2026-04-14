#!/usr/bin/env python3
"""
Cross-Lingual Voice Cloning Evaluation Pipeline — XTTS v2
=============================================================
Zero-shot XTTS finetuning evaluation on ymoslem/acl-6060.
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

from TTS.api import TTS

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")

# ===================================================================
# Helper functions
# ===================================================================

def load_speaker_model(device="cuda"):
    try:
        from speechbrain.pretrained import SpeakerRecognition
    except ImportError:
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
        # Force mono: average channels if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)  # shape: [1, time]
        emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).detach()
    except Exception as e:
        print(f"  [sim warning] {wav_path}: {e}")
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
    parser = argparse.ArgumentParser(description="Cross-Lingual Voice Cloning Evaluation — XTTS")
    parser.add_argument("--dataset", default="ymoslem/acl-6060")
    parser.add_argument("--model-dir", required=True, help="Path to XTTS finetuned exp directory (e.g., exp/xtts_finetuned/...)")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr", help="Language code for ASR")
    parser.add_argument("--output-dir", default="./eval_results_xtts_ft")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("📥 Loading evaluation dataset...")
    # Map whisper lang code to XTTS lang code
    xtts_lang_map = {"fr": "fr", "ar": "ar", "zh": "zh-cn", "en": "en"}
    xtts_lang = xtts_lang_map.get(args.whisper_lang, args.whisper_lang)
    
    ds = load_dataset(args.dataset, split=args.split)
    
    if args.max_samples:
        ds = ds.select(range(min(len(ds), args.max_samples)))

    print(f"   Samples: {len(ds)}")
    
    print("🔧 Phase 2: Loading XTTS model...")
    import glob
    import shutil
    
    # XTTS natively hardcodes loading "model.pth" from the checkpoint directory.
    # The Trainer outputs "best_model_XYZ.pth", so we must create a symlink or copy
    # named exactly "model.pth" in the same directory for the API to find it.
    target_model_file = os.path.join(args.model_dir, "model.pth")
    
    if not os.path.exists(target_model_file):
        candidates = glob.glob(os.path.join(args.model_dir, "best_model_*.pth"))
        if candidates:
            best_model_path = sorted(candidates, key=os.path.getmtime)[-1]
            print(f"   Copying {os.path.basename(best_model_path)} -> model.pth")
            shutil.copy(best_model_path, target_model_file)
        else:
            candidates = glob.glob(os.path.join(args.model_dir, "checkpoint_*.pth"))
            if candidates:
                best_model_path = sorted(candidates, key=os.path.getmtime)[-1]
                print(f"   Copying {os.path.basename(best_model_path)} -> model.pth")
                shutil.copy(best_model_path, target_model_file)
            
    config_path = os.path.join(args.model_dir, "config.json")
    vocab_path = os.path.join(args.model_dir, "vocab.json")
    
    # The Trainer does NOT automatically copy the tokenizer vocabulary (vocab.json).
    # If it's missing, XTTS silently builds a NoneType tokenizer and crashes on .encode().
    if not os.path.exists(vocab_path):
        from TTS.utils.manage import ModelManager
        try:
            print("   Restoring missing vocab.json from base XTTS model...")
            manager = ModelManager()
            # download_model returns a tuple like (model_path, config_path, item)
            base_model_path = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0]
            base_vocab_path = os.path.join(base_model_path, "vocab.json")
            if os.path.exists(base_vocab_path):
                shutil.copy(base_vocab_path, vocab_path)
                print("   Successfully recovered vocab.json!")
        except Exception as e:
            print(f"   Failed to recover vocab.json: {e}")
    
    # 🔧 To avoid PEFT state_dict corruption, we load the pristine official base model first, 
    # and then attach our LoRA adapter on top of it.
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    
    # 🧬 Inject LoRA Adapter if it exists
    lora_path = os.path.join(args.model_dir, "lora_adapter")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(os.path.dirname(args.model_dir), "lora_adapter")
        
    if os.path.exists(lora_path):
        print(f"🧬 Injecting LoRA adapter from: {lora_path}")
        from peft import PeftModel
        # The TTS wrapper holds the Xtts model in synthesizer.tts_model
        tts.synthesizer.tts_model.gpt = PeftModel.from_pretrained(tts.synthesizer.tts_model.gpt, lora_path)
        print("   ✅ LoRA adapter successfully attached to the GPT decoder!")

    print("🧠 Loading Faster-Whisper for ASR...")
    from faster_whisper import WhisperModel
    import jiwer
    
    whisper = WhisperModel(args.whisper_model, device=device, compute_type="default")

    print("🗣️  Loading SpeechBrain ECAPA-TDNN for Speaker Similarity...")
    spk_model = load_speaker_model(device=device)

    print(f"\n📊 Evaluating XTTS on {args.whisper_lang}...")
    metrics = {"wer": [], "cer": [], "sim": [], "inf_time": [], "rtf": []}
    per_sample_log = []
    
    ref_dir = os.path.join(args.output_dir, "references")
    syn_dir = os.path.join(args.output_dir, "synthesized")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(syn_dir, exist_ok=True)

    for i, row in enumerate(tqdm(ds)):
        try:
            # Prepare Reference Target
            text_target = (row.get(f"trg_{args.whisper_lang}_text") or row.get(f"text_{args.whisper_lang}") or "").strip()
            ref_data = (row.get("ref_en_voice") or row.get(f"ref_{args.whisper_lang}_voice") or row.get("audio_en") or row.get("audio"))
            
            if not ref_data or not text_target:
                continue
                
            audio_data = ref_data["array"]
            sr = ref_data["sampling_rate"]

            ref_wav_path = os.path.join(ref_dir, f"ref_{i}.wav")
            wav_t = torch.from_numpy(audio_data).float()
            if wav_t.dim() == 1:
                wav_t = wav_t.unsqueeze(0)
            torchaudio.save(ref_wav_path, wav_t, sr)
            
            # Predict
            start_t = time.time()
            syn_wav_path = os.path.join(syn_dir, f"syn_{i}.wav")
            
            # The heart of the wrapper
            tts.tts_to_file(
                text=text_target,
                speaker_wav=ref_wav_path,
                language=xtts_lang,
                file_path=syn_wav_path
            )
            inf_t = time.time() - start_t
            
            # Metric: RTF
            audio_info = torchaudio.info(syn_wav_path)
            audio_dur = audio_info.num_frames / audio_info.sample_rate
            rtf = inf_t / audio_dur if audio_dur > 0 else np.nan
            
            # Metric: Similarity
            ref_emb = extract_speaker_embedding(ref_wav_path, spk_model, device)
            syn_emb = extract_speaker_embedding(syn_wav_path, spk_model, device)
            sim_score = np.nan
            if ref_emb is not None and syn_emb is not None:
                sim_score = F.cosine_similarity(ref_emb.unsqueeze(0), syn_emb.unsqueeze(0)).item()
            
            # Metric: ASR
            segments, _ = whisper.transcribe(
                syn_wav_path,
                beam_size=args.whisper_beam,
                language=args.whisper_lang,
                without_timestamps=True
            )
            transcription = " ".join([seg.text.strip() for seg in segments])
            
            try:
                wer = jiwer.wer(text_target.lower(), transcription.lower())
                cer = jiwer.cer(text_target.lower(), transcription.lower())
            except Exception:
                wer = np.nan
                cer = np.nan
            
            metrics["wer"].append(wer)
            metrics["cer"].append(cer)
            metrics["sim"].append(sim_score)
            metrics["inf_time"].append(inf_t)
            metrics["rtf"].append(rtf)
            
            per_sample_log.append({
                "id": i,
                "target": text_target,
                "pred": transcription,
                "wer": wer,
                "cer": cer,
                "sim": sim_score,
                "inf_time": inf_t,
                "rtf": rtf
            })
            
        except Exception as e:
            print(f"Error on row {i}: {e}")

    # Aggregating
    summary = {
        "wer": safe_mean(metrics["wer"]),
        "cer": safe_mean(metrics["cer"]),
        "sim": safe_mean(metrics["sim"]),
        "inf_time": safe_mean(metrics["inf_time"]),
        "rtf": safe_mean(metrics["rtf"]),
    }
    
    print("\n==============================================================")
    print("  XTTS EVALUATION COMPLETE")
    print(f"  Target: {args.whisper_lang} | Samples: {safe_count(metrics['wer'])}")
    print("==============================================================")
    print(f"  WER        : {summary['wer']:.4f}")
    print(f"  CER        : {summary['cer']:.4f}")
    print(f"  Similarity : {summary['sim']:.4f}")
    print(f"  InferenceS : {summary['inf_time']:.4f}")
    print(f"  RTF        : {summary['rtf']:.4f}")
    print("==============================================================\n")

    csv_path = os.path.join(args.output_dir, "eval_per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "target", "pred", "wer", "cer", "sim", "inf_time", "rtf"])
        w.writeheader()
        w.writerows(per_sample_log)
        
    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()
