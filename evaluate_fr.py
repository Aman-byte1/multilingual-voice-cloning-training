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
    lang = "fr"
    out_dir = f"temp_submission/{lang}"
    text_file = "blind_test/text/french.txt"
    report_file = "eval_results_fr_sample.json"
    
    if not os.path.exists(out_dir):
        print(f"Error: {out_dir} not found.")
        return

    # 1. Load Ground Truth Text (First 25 lines)
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [l.strip() for l in f if l.strip()][:25]
    
    # 2. Map generated files for ALL 12 speakers (first 25 lines each)
    all_files = os.listdir(out_dir)
    speakers = list(set(["_".join(f.split("_")[2:]).replace(".wav", "") 
                         for f in all_files if f.startswith(f"{lang}_") and f.endswith(".wav") and not f.startswith("_")]))
    speakers.sort()

    eval_samples = []
    for ref_name in speakers:
        for i in range(1, 26): # First 25 segments
            fname = f"{lang}_{i:03d}_{ref_name}.wav"
            if os.path.exists(os.path.join(out_dir, fname)):
                eval_samples.append({"file": fname, "speaker": ref_name, "idx": i-1})

    print(f"🔍 Sampling {len(eval_samples)} files (25 segments across {len(speakers)} voices) for French A/B check.")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. GENERATE BASE OMNIVOICE FOR COMPARISON 
    print("\n🚀 Loading BASE OmniVoice for A/B Benchmark...")
    from omnivoice import OmniVoice
    base_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    base_model.to(device).eval()
    
    base_results_dir = "eval_base_fr_samples"
    os.makedirs(base_results_dir, exist_ok=True)
    
    base_files_map = {}
    print(f"🎙️ Generating Base Model counterparts...")
    for ref_name in tqdm(speakers, desc="Speakers"):
        ref_path = os.path.join(out_dir, f"_extracted_reference_{ref_name}.wav")
        if not os.path.exists(ref_path): ref_path = f"blind_test/audio/fr/{ref_name}.wav"
        
        waveform, sr_in = torchaudio.load(ref_path)
        ref_tuple = (waveform, sr_in)
        
        for i in range(25):
            path = os.path.join(base_results_dir, f"base_{i}_{ref_name}.wav")
            base_files_map[(ref_name, i)] = path
            if os.path.exists(path): continue
            
            with torch.no_grad():
                res = base_model.generate(text=text_lines[i], ref_audio=ref_tuple, temperature=0.8, top_p=0.9)
                if isinstance(res, tuple): audio_data, _ = res
                else: audio_data = res
                
                if isinstance(audio_data, (list, tuple)):
                    audio_tensor = torch.from_numpy(np.array(audio_data))
                elif not isinstance(audio_data, torch.Tensor):
                    audio_tensor = torch.from_numpy(audio_data)
                else:
                    audio_tensor = audio_data
                
                if audio_tensor.ndim == 1: audio_tensor = audio_tensor.unsqueeze(0)
                torchaudio.save(path, audio_tensor.cpu(), 24000)
    
    del base_model
    torch.cuda.empty_cache()

    # 4. RUN EVALUATION
    print("\n🔎 Loading Evaluation Models...")
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition
    whisper = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "int8")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )
    
    def get_metrics(file_list, is_base=False):
        results = []
        transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        
        ref_embs = {}
        for ref_name in speakers:
            ref_path = os.path.join(out_dir, f"_extracted_reference_{ref_name}.wav")
            if not os.path.exists(ref_path): ref_path = f"blind_test/audio/fr/{ref_name}.wav"
            ref_embs[ref_name] = extract_speaker_embedding(ref_path, verifier, device)

        for s in tqdm(file_list, desc="Evaluating"):
            path = s["path"]
            ref_emb = ref_embs.get(s["speaker"])
            emb = extract_speaker_embedding(path, verifier, device)
            sim = float(F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()) if (emb is not None and ref_emb is not None) else 0.0
            try:
                segments, _ = whisper.transcribe(path, language=lang)
                tx = "".join([s.text for s in segments])
                c_val = jiwer.cer(transforms(text_lines[s["idx"]]), transforms(tx))
            except: c_val = 1.0
            results.append({"sim": sim, "cer": c_val})
        return np.mean([r["sim"] for r in results]), np.mean([r["cer"] for r in results])

    lora_samples = [{"path": os.path.join(out_dir, s["file"]), "speaker": s["speaker"], "idx": s["idx"]} for s in eval_samples]
    base_samples = [{"path": base_files_map[(s["speaker"], s["idx"])], "speaker": s["speaker"], "idx": s["idx"]} for s in eval_samples]

    lora_sim, lora_cer = get_metrics(lora_samples)
    base_sim, base_cer = get_metrics(base_samples)

    print("\n" + "="*70)
    print(f"🏆 REPRESENTATIVE A/B TEST: FRENCH (FR) - {len(eval_samples)} samples")
    print("="*70)
    print(f"{'Metric':<15} | {'Base Omni':<15} | {'Step 200 LoRA':<15} | {'Improvement'}")
    print("-" * 70)
    print(f"{'CER (Lower ↓)':<15} | {base_cer:<15.4f} | {lora_cer:<15.4f} | {((base_cer-lora_cer)/base_cer)*100:+.1f}%")
    print(f"{'Sim (Higher ↑)':<15} | {base_sim:<15.4f} | {lora_sim:<15.4f} | {((lora_sim-base_sim)/base_sim)*100:+.1f}%")
    print("="*70)
    
    if lora_cer < base_cer and lora_sim > base_sim:
        print("\n✅ STATUS: CLEAR UPGRADE. Step 200 FR is numerically superior in all metrics.")
    else:
        print("\n⚠️ STATUS: MIXED. Results show quality trade-offs for French.")

if __name__ == "__main__":
    main()
