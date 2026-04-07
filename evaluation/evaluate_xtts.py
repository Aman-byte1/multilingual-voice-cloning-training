#!/usr/bin/env python3
"""
🚀 COQUI XTTS-v2 EVALUATION PIPELINE
End-to-end Voice Cloning evaluation using XTTS-v2 Zero-Shot
"""

import os
import argparse
import warnings
import tempfile
import json
import numpy as np
import librosa
import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset

# Optimizations
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore")

def compute_pesq_score(ref_wav_path: str, synth_wav_path: str):
    try:
        from pesq import pesq as pesq_fn
        ref, _ = librosa.load(ref_wav_path, sr=16000)
        syn, _ = librosa.load(synth_wav_path, sr=16000)
        min_len = min(len(ref), len(syn))
        if min_len < 8000: return None
        return float(pesq_fn(16000, ref[:min_len], syn[:min_len], "wb"))
    except: return None

def compute_mcd_score(ref_wav_path: str, synth_wav_path: str):
    try:
        from pymcd.mcd import Calculate_MCD
        mcd_calc = Calculate_MCD(MCD_mode="plain")
        return float(mcd_calc.calculate_mcd(ref_wav_path, synth_wav_path))
    except: return None

def load_speaker_model():
    from speechbrain.inference.speaker import SpeakerRecognition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval") -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1: wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path

def main():
    parser = argparse.ArgumentParser(description="Coqui XTTS-v2 Evaluation")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-data")
    parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="./eval_results_xtts")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--whisper-model", default="large-v3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📥 Loading dataset split from {args.dataset} ...")
    ds_test = load_dataset(args.dataset, split="test", cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        indices = list(range(0, total, total // args.max_samples))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)

    print(f"\n🎙 PHASE 1/3: Generating {total} audio samples (XTTS-v2) ...")
    samples_data = []
    skipped = 0
    tts = None
    
    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        
        text_fr = (row.get("trg_fr_text") or row.get("text_fr") or "").strip()
        text_en = (row.get("ref_en_text") or row.get("text_en") or "").strip()
        
        # Audio extraction
        ref_data = row.get("ref_en_voice") or row.get("audio_en") or row.get("ref_fr_voice")
        gt_data = row.get("trg_fr_voice") or row.get("cloned_audio_fr")
        
        if not ref_data or not text_fr: 
            skipped += 1
            continue
            
        synth_wav_path = os.path.join(args.output_dir, f"xtts_{i:05d}.wav")
        
        # Check if already generated
        if os.path.exists(synth_wav_path):
            gt_wav_path = save_temp_wav(np.asarray(gt_data["array"], dtype=np.float32), gt_data["sampling_rate"], "gt_") if gt_data else None
            ref_wav_path = save_temp_wav(np.asarray(ref_data["array"], dtype=np.float32), ref_data["sampling_rate"], "ref_")

            samples_data.append({
                "idx": i, "synth_path": synth_wav_path, "gt_path": gt_wav_path, "ref_path": ref_wav_path,
                "text_fr": text_fr, "text_en": text_en, "speaker_id": row.get("speaker_id", "unknown")
            })
            continue

        # Support both 'amanuelbyte' and 'ymoslem' schemas
        text_fr = (row.get("trg_fr_text") or row.get("text_fr") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get("ref_fr_voice") or row.get("audio")
        gt_data  = row.get("trg_fr_voice")

        if not ref_data or not text_fr:
            skipped += 1
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"], "ref_")

        gt_path = None
        if gt_data is not None:
            gt_path = save_temp_wav(
                np.asarray(gt_data["array"],  dtype=np.float32),
                gt_data["sampling_rate"],  "gt_") if gt_data else None

        # Load model lazily
        if tts is None:
            print(f"🚀 Initializing Coqui XTTS-v2 ...")
            from TTS.api import TTS
            tts = TTS(args.model_name, gpu=True)
            
        try:
            # XTTS-v2 generates at 24kHz natively
            tts.tts_to_file(
                text=text_fr,
                speaker_wav=ref_path,
                language="fr",
                file_path=synth_wav_path
            )
            
            samples_data.append({
                "idx": i, "synth_path": synth_wav_path, "gt_path": gt_path, "ref_path": ref_path,
                "text_fr": text_fr, "text_en": text_en, "speaker_id": row.get("speaker_id", "unknown")
            })
        except Exception as e:
            skipped += 1
            tqdm.write(f"   ⚠ Sample {i} failed: {str(e)}")
            if gt_path is not None and os.path.exists(gt_path):
                os.remove(gt_path)
            if os.path.exists(ref_path):
                os.remove(ref_path)
        
    # Phase 2: ASR
    print(f"\n🎙 PHASE 2/3: Transcribing with faster-whisper {args.whisper_model} ...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
    
    transcripts = []
    for s in tqdm(samples_data, desc="Transcribing"):
        segments, _ = whisper.transcribe(s["synth_path"], language="fr", beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        transcripts.append(text)
        
    del whisper
    torch.cuda.empty_cache()

    print(f"\n📊 PHASE 3/3: Calculating acoustic metrics (Similarity, PESQ, MCD) ...")
    verifier = load_speaker_model()
    import jiwer
    import sacrebleu
    
    results = []
    for s, tx in tqdm(zip(samples_data, transcripts), total=len(samples_data), desc="Finalizing"):
        if s.get("gt_path"):
            p = compute_pesq_score(s["gt_path"], s["synth_path"])
            m = compute_mcd_score(s["gt_path"], s["synth_path"])
            os.remove(s["gt_path"])
        else:
            p, m = None, None
            
        if s.get("ref_path"):
            sim = float(verifier.verify_files(s["ref_path"], s["synth_path"])[0].item())
            os.remove(s["ref_path"])
        else:
            sim = None
            
        ref_clean = jiwer.RemovePunctuation()(s["text_fr"].lower())
        hyp_clean = jiwer.RemovePunctuation()(tx.lower())
        w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean else 0.0
        chr_val = float(sacrebleu.sentence_chrf(tx, [s["text_fr"]]).score)
        
        results.append({
            "idx": s["idx"], "speaker": s["speaker_id"], "WER": w, "chrF": chr_val, 
            "PESQ": p or 0, "MCD": m or 0, "Similarity": sim
        })

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    metric_keys = ["WER", "chrF", "PESQ", "MCD", "Similarity"]
    stats = {k: float(np.mean([r[k] for r in results if r[k] is not None])) for k in metric_keys}
    
    with open(summary_path, "w") as f: json.dump({"stats": stats, "results": results}, f, indent=2)

    print("\n" + "="*50)
    print("  XTTS-v2 ZERO-SHOT EVALUATION COMPLETE")
    print("="*50)
    for k, v in stats.items(): print(f"  {k:12}: {v:.4f}")
    print("="*50)
    print(f"✅ Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
