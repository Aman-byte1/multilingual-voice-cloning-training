"""
Standalone script to re-compute missing Speaker Similarity scores 
from an already generated eval_per_sample.csv without re-synthesizing audio.
"""

import os
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )

def extract_speaker_embedding(wav_path, model, device="cuda"):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    # Ensure audio is properly formatted as Mono for SpeechBrain ECAPA-TDNN [1, time]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    wav = wav.to(device)
    # encode_batch expects shape [batch, time] -> already [1, time]
    emb = model.encode_batch(wav)
    return emb.squeeze(0).squeeze(0).detach()

def main():
    csv_path = "./eval_results_xtts_ft/eval_per_sample.csv"
    ref_dir = "./eval_results_xtts_ft/references"
    syn_dir = "./eval_results_xtts_ft/synthesized"
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print("🧠 Resurrecting missing Speaker Similarity scores...")
    df = pd.read_csv(csv_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk_model = load_speaker_model(device=device)
    
    sim_scores = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        # Matches the naming convention from evaluate_xtts.py
        ref_wav = os.path.join(ref_dir, f"ref_{row['id']}.wav")
        syn_wav = os.path.join(syn_dir, f"syn_{row['id']}.wav")
        
        try:
            ref_emb = extract_speaker_embedding(ref_wav, spk_model, device)
            syn_emb = extract_speaker_embedding(syn_wav, spk_model, device)
            
            sim = F.cosine_similarity(ref_emb.unsqueeze(0), syn_emb.unsqueeze(0)).item()
            sim_scores.append(sim)
        except Exception as e:
            print(f"Error on {row['id']}: {e}")
            sim_scores.append(np.nan)
            
    df["sim"] = sim_scores
    
    # Save patched CSV
    out_csv = "./eval_results_xtts_ft/eval_per_sample_patched.csv"
    df.to_csv(out_csv, index=False)
    
    final_sim = df["sim"].mean()
    print("==============================================================")
    print(f"  Final Aggregated Similarity : {final_sim:.4f}")
    print(f"  Fixed CSV exported to       : {out_csv}")
    print("==============================================================")

if __name__ == "__main__":
    main()
