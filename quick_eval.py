import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import jiwer
import numpy as np
import json

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
    except Exception as e:
        print(f"Error extracting embedding for {path}: {e}")
        return None

def main():
    lang = "zh"
    out_dir = f"temp_submission/{lang}"
    text_file = "blind_test/text/chinese.txt"
    
    if not os.path.exists(out_dir):
        print(f"Error: {out_dir} not found.")
        return

    # 1. Load Ground Truth Text
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [l.strip() for l in f if l.strip()]
    
    # 2. Pick a speaker to evaluate (look for generated wavs)
    all_files = os.listdir(out_dir)
    # Pattern: zh_001_REFNAME.wav
    gen_files = [f for f in all_files if f.startswith(f"{lang}_") and f.endswith(".wav") and not f.startswith("_")]
    
    if not gen_files:
        print("No generated files found yet. Wait for a speaker to finish.")
        return
        
    # Group by ref_name
    speakers = {}
    for f in gen_files:
        parts = f.split("_")
        if len(parts) < 3: continue
        # Format lang_lineid_refname.wav -> parts[2] is start of refname
        ref_name = "_".join(parts[2:]).replace(".wav", "")
        speakers.setdefault(ref_name, []).append(f)
        
    # Sort by count and pick the first one with significant files
    best_ref = max(speakers.keys(), key=lambda k: len(speakers[k]))
    eval_files = sorted(speakers[best_ref], key=lambda x: int(x.split("_")[1]))
    
    print(f"\n📊 Quick Evaluation for Speaker: {best_ref}")
    print(f"   Found {len(eval_files)} / {len(text_lines)} generated lines.")
    
    # Limit to first 20 for speed if needed
    eval_files = eval_files[:20] 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Load Models
    print("🚀 Loading evaluation models (Whisper Large-v3 & ECAPA-TDNN)...")
    whisper = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "int8")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )
    
    # 4. Reference Embedding
    ref_path = os.path.join(out_dir, f"_extracted_reference_{best_ref}.wav")
    if not os.path.exists(ref_path):
        # Fallback to looking in blind_test
        ref_path = f"blind_test/audio/{lang}/{best_ref}.wav"
        
    ref_emb = extract_speaker_embedding(ref_path, verifier, device)
    
    results = []
    
    transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    
    print(f"📈 Computing Metrics...")
    for f in tqdm(eval_files):
        path = os.path.join(out_dir, f)
        idx = int(f.split("_")[1]) - 1 # 0-indexed
        target_text = text_lines[idx]
        
        # Similarity
        syn_emb = extract_speaker_embedding(path, verifier, device)
        sim = float(F.cosine_similarity(syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()) if (syn_emb is not None and ref_emb is not None) else 0.0
        
        # ASR
        try:
            segments, _ = whisper.transcribe(path, language=lang)
            transcription = "".join([s.text for s in segments])
            c_val = jiwer.cer(transforms(target_text), transforms(transcription))
        except Exception as e:
            c_val = 1.0
            
        results.append({"sim": sim, "cer": c_val})
        
    # 5. Summary
    avg_sim = np.mean([r["sim"] for r in results])
    avg_cer = np.mean([r["cer"] for r in results])
    
    # Baseline from summary_zh.json: CER 0.1314, Sim 0.6773
    baseline_cer = 0.1314
    baseline_sim = 0.6773
    
    print("\n" + "="*60)
    print(f"🏆 PRE-SUBMISSION RESULTS (Sample of {len(results)} lines)")
    print("="*60)
    print(f"{'Metric':<15} | {'Step 400 LoRA':<15} | {'Baseline Omni':<15} | {'Improvement':<12}")
    print("-" * 60)
    print(f"{'CER (Lower ↓)':<15} | {avg_cer:<15.4f} | {baseline_cer:<15.4f} | {((baseline_cer-avg_cer)/baseline_cer)*100:+.1f}%")
    print(f"{'Sim (Higher ↑)':<15} | {avg_sim:<15.4f} | {baseline_sim:<15.4f} | {((avg_sim-baseline_sim)/baseline_sim)*100:+.1f}%")
    print("="*60)
    
    if avg_cer < baseline_cer and avg_sim > baseline_sim:
        print("\n✅ STATUS: CLEAR UPGRADE. New models exceed all baseline metrics.")
    else:
        print("\n⚠️ STATUS: MIXED. Manual quality check recommended.")

if __name__ == "__main__":
    main()
