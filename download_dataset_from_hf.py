#!/usr/bin/env python3
"""
Download Best-of-N dataset from HuggingFace to local `.jsonl` formats
for OmniVoice Tokenization.
"""

import os
import json
import soundfile as sf
from datasets import load_dataset

def main():
    repo_id = os.environ.get("HF_REPO_ID", "amanuelbyte/omnivoice-best-of-n-training")
    
    # 1. Load dataset from Hub
    print(f"📥 Loading dataset '{repo_id}' from HuggingFace...")
    ds = load_dataset(repo_id, split="train")

    # 2. Setup output directories
    output_dir = "./data/finetune/wavs"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = "./data/finetune/merged_all.jsonl"

    print(f"🔊 Extracting audio and building manifest at {jsonl_path}...")
    
    manifest_entries = []
    
    for i, row in enumerate(ds):
        # The best_audio column contains the raw signal and sampling_rate
        audio_info = row["best_audio"]
        audio_array = audio_info["array"]
        sample_rate = audio_info["sampling_rate"]
        
        # Save audio to disk
        out_wav = os.path.join(output_dir, f"sample_{i:05d}.wav")
        sf.write(out_wav, audio_array, sample_rate)
        
        # OmniVoice `extract_audio_tokens.py` expects:
        # {"id": "...", "audio_path": "...", "text": "..."}
        manifest_entries.append({
            "id": f"sample_{i:05d}",
            "audio_path": os.path.abspath(out_wav),
            "text": row["text"]
        })

    # 3. Write purely to merged_all.jsonl 
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Successfully downloaded {len(manifest_entries)} samples!")
    print(f"   You can now bypass Step 1 in finetune_omnivoice.sh.")

if __name__ == "__main__":
    main()
