#!/usr/bin/env python3
"""
Prepare Best-of-N XTTS Dataset
Downloads the Best-of-N dataset from Hugging Face, resamples audio to 22,050 Hz,
and writes LJSpeech-formatted metadata.csv files for French, Arabic, and Chinese.
"""

import os
import argparse
import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="amanuelbyte/omnivoice-best-of-n-training")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"📥 Loading dataset from Hugging Face: {args.repo_id}")
    ds = load_dataset(args.repo_id, split="train")

    # Group by language
    langs = ["fr", "ar", "zh"]
    metadata_lines = {l: [] for l in langs}

    print(f"🔊 Processing {len(ds)} samples and resampling to 22.05 kHz...")
    for row in tqdm(ds):
        lang = row["language"]
        if lang not in langs:
            continue
            
        audio_dict = row["best_audio"]
        audio_array = audio_dict["array"]
        orig_sr = audio_dict["sampling_rate"]
        
        # Resample to exactly 22050 Hz for XTTS
        if orig_sr != 22050:
            audio_array = librosa.resample(y=audio_array, orig_sr=orig_sr, target_sr=22050)
            
        audio_id = row["id"]
        wav_path = os.path.join(wav_dir, f"{audio_id}.wav")
        # Save as 16-bit PCM
        sf.write(wav_path, audio_array, 22050, subtype='PCM_16')
        
        text = row["text"].replace("|", " ") # Remove any pipes from text
        
        # LJSpeech format: ID|text|normalized_text
        # Note: We must NOT include .wav in the ID column for LJSpeech formatter
        metadata_lines[lang].append(f"{audio_id}|{text}|{text}\n")

    # Write separate metadata.csv for each language so TTS Trainer can route them
    for lang in langs:
        meta_path = os.path.join(args.output_dir, f"metadata_{lang}.csv")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.writelines(metadata_lines[lang])
            
        print(f"✅ Saved {lang} metadata to {meta_path} ({len(metadata_lines[lang])} samples)")
        
    print(f"🎉 Dataset preparation complete! All audio saved in {wav_dir}")

if __name__ == "__main__":
    main()
