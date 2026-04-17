import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import jiwer
import numpy as np
import json
from omnivoice import OmniVoice

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
        return None

def main():
    lang = "zh"
    ref_name = "2023.acl-long.12" # The speaker we just evaluated
    text_file = "blind_test/text/chinese.txt"
    ab_dir = "ab_test_outputs"
    os.makedirs(ab_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Ground Truth Text (first 20 lines)
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [l.strip() for l in f if l.strip()][:20]
    
    # 2. Reference Audio
    ref_path = f"temp_submission/{lang}/_extracted_reference_{ref_name}.wav"
    waveform, sr = torchaudio.load(ref_path)
    ref_tuple = (waveform, sr)

    # 3. GENERATE BASE OMNIVOICE SAMPLES
    print("\n🚀 Loading BASE OmniVoice (No LoRA)...")
    base_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    base_model.to(device).eval()
    
    base_wavs = []
    print(f"🎙️ Generating 20 benchmark samples with BASE model...")
    for idx, text in enumerate(tqdm(text_lines)):
        path = os.path.join(ab_dir, f"base_{idx:03d}.wav")
        with torch.no_grad():
            # Using same params as submission script
            res = base_model.generate(text=text, ref_audio=ref_tuple, temperature=0.8, top_p=0.9)
            if isinstance(res, tuple): audio_data, sr_out = res
            else: audio_data, sr_out = res, 24000
            
            # Robust tensor conversion
            if isinstance(audio_data, (list, tuple)):
                import numpy as np
                audio_tensor = torch.from_numpy(np.array(audio_data))
            elif not isinstance(audio_data, torch.Tensor):
                audio_tensor = torch.from_numpy(audio_data)
            else:
                audio_tensor = audio_data
                
            if audio_tensor.ndim == 1: audio_tensor = audio_tensor.unsqueeze(0)
            torchaudio.save(path, audio_tensor.cpu(), 24000)
            base_wavs.append(path)
            
    del base_model
    torch.cuda.empty_cache()

    # 4. EVALUATION
    print("\n🔎 Loading Evaluation Models...")
    whisper = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "int8")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )
    ref_emb = extract_speaker_embedding(ref_path, verifier, device)

    def run_eval(file_list, label):
        stats = []
        transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        print(f"📊 Evaluating {label}...")
        for i, path in enumerate(tqdm(file_list)):
            # Sim
            emb = extract_speaker_embedding(path, verifier, device)
            sim = float(F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()) if (emb is not None) else 0.0
            # CER
            try:
                segments, _ = whisper.transcribe(path, language=lang)
                tx = "".join([s.text for s in segments])
                c_val = jiwer.cer(transforms(text_lines[i]), transforms(tx))
            except: c_val = 1.0
            stats.append({"sim": sim, "cer": c_val})
        return np.mean([s["sim"] for s in stats]), np.mean([s["cer"] for s in stats])

    # Eval Base
    base_sim, base_cer = run_eval(base_wavs, "BASE OMNIVOICE")

    # Eval Step 400 (Local files)
    step400_files = [os.path.join(f"temp_submission/{lang}", f"{lang}_{i+1:03d}_{ref_name}.wav") for i in range(20)]
    step400_sim, step400_cer = run_eval(step400_files, "STEP 400 LoRA")

    # 5. FINAL COMPARISON
    print("\n" + "="*70)
    print("🏆 A/B TEST: BASE OMNIVOICE vs. STEP 400 LoRA (Sub-sample)")
    print("="*70)
    print(f"{'Metric':<15} | {'Base Model':<15} | {'Step 400 LoRA':<15} | {'Winner'}")
    print("-" * 70)
    # CER (Lower is better)
    cer_winner = "LoRA" if step400_cer < base_cer else "Base"
    print(f"{'CER (Lower ↓)':<15} | {base_cer:<15.4f} | {step400_cer:<15.4f} | {cer_winner}")
    # Similarity (Higher is better)
    sim_winner = "LoRA" if step400_sim > base_sim else "Base"
    print(f"{'Sim (Higher ↑)':<15} | {base_sim:<15.4f} | {step400_sim:<15.4f} | {sim_winner}")
    print("="*70)
    
    if step400_cer > 0.2 and base_cer > 0.2:
        print("\n⚠️ NOTE: Both models have high CER. The Chinese text may be too complex for zero-shot.")
    elif step400_cer > base_cer * 1.5:
        print("\n❌ WARNING: LoRA is significantly hurting intelligibility. Suggest scaling down LoRA.")

if __name__ == "__main__":
    main()
