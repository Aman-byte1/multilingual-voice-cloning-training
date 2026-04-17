import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import jiwer
import numpy as np
import json
import csv

# Metrics tools
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import SpeakerRecognition

def extract_speaker_embedding(path, model, device="cuda"):
    try:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        with torch.no_grad():
            emb = model.encode_batch(waveform.to(device)).squeeze(0).squeeze(0)
        return emb
    except:
        return None

def main():
    lang = "zh"
    out_dir = f"temp_submission/{lang}"
    text_file = "blind_test/text/chinese.txt"
    report_file = "eval_results_zh.json"
    
    if not os.path.exists(out_dir):
        print(f"Error: {out_dir} not found.")
        return

    # 1. Load Ground Truth Text
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [l.strip() for l in f if l.strip()]
    
    # 2. Map all 1344 generated files
    all_files = os.listdir(out_dir)
    gen_files = [f for f in all_files if f.startswith(f"{lang}_") and f.endswith(".wav") and not f.startswith("_")]
    
    print(f"🔍 Found {len(gen_files)} files to evaluate for Chinese.")
    
    # Group by speaker for per-speaker breakdown
    speakers = {}
    for f in gen_files:
        parts = f.split("_")
        ref_name = "_".join(parts[2:]).replace(".wav", "")
        speakers.setdefault(ref_name, []).append(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Load Metrics Models
    print("🚀 Loading Large-v3 Whisper & ECAPA-TDNN...")
    whisper = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "int8")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )
    
    # Pre-cache speaker embeddings
    ref_embs = {}
    for ref_name in speakers.keys():
        ref_path = os.path.join(out_dir, f"_extracted_reference_{ref_name}.wav")
        if not os.path.exists(ref_path):
            ref_path = f"blind_test/audio/zh/{ref_name}.wav"
        ref_embs[ref_name] = extract_speaker_embedding(ref_path, verifier, device)

    results = []
    transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    
    print(f"📈 Processing Samples (this will take ~15-20 minutes)...")
    for f in tqdm(gen_files):
        path = os.path.join(out_dir, f)
        parts = f.split("_")
        idx = int(parts[1]) - 1
        ref_name = "_".join(parts[2:]).replace(".wav", "")
        
        target_text = text_lines[idx]
        ref_emb = ref_embs.get(ref_name)
        
        # Sim
        syn_emb = extract_speaker_embedding(path, verifier, device)
        sim = float(F.cosine_similarity(syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()) if (syn_emb is not None and ref_emb is not None) else 0.0
        
        # CER
        try:
            segments, _ = whisper.transcribe(path, language=lang)
            tx = "".join([s.text for s in segments])
            c_val = jiwer.cer(transforms(target_text), transforms(tx))
        except:
            c_val = 1.0
            
        results.append({
            "file": f,
            "speaker": ref_name,
            "sim": sim,
            "cer": c_val
        })
        
    # 4. Aggregation
    overall_sim = np.mean([r["sim"] for r in results])
    overall_cer = np.mean([r["cer"] for r in results])
    
    speaker_stats = {}
    for ref_name in speakers.keys():
        s_res = [r for r in results if r["speaker"] == ref_name]
        speaker_stats[ref_name] = {
            "avg_sim": np.mean([r["sim"] for r in s_res]),
            "avg_cer": np.mean([r["cer"] for r in s_res])
        }

    summary = {
        "overall": {"cer": overall_cer, "sim": overall_sim},
        "by_speaker": speaker_stats,
        "baseline": {"cer": 0.1314, "sim": 0.6773}
    }
    
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=4)

    # 5. Final Output
    print("\n" + "="*60)
    print("🏆 FINAL CHINESE SUBMISSION REPORT")
    print("="*60)
    print(f"{'Metric':<15} | {'Submission':<15} | {'Baseline':<15} | {'Results'}")
    print("-" * 60)
    print(f"{'CER (Lower ↓)':<15} | {overall_cer:<15.4f} | {'0.1314':<15} | {'WIN' if overall_cer < 0.1314 else 'FAIL'}")
    print(f"{'Sim (Higher ↑)':<15} | {overall_sim:<15.4f} | {'0.6773':<15} | {'WIN' if overall_sim > 0.6773 else 'FAIL'}")
    print("="*60)
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
