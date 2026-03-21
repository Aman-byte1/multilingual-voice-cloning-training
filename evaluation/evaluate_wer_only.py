#!/usr/bin/env python3
"""
Fast, batched Word Error Rate (WER) standalone evaluation.

This script scans the `./eval_results` folder for already generated 
audio files (`synth_{speaker}_{id}.wav`), grabs the full ground-truth 
French text from the Hugging Face dataset cache, and runs a heavily 
batched Whisper ASR transcription to calculate the WER in seconds 
by fully maximizing GPU usage.

Prerequisites:
    pip install transformers jiwer datasets torchaudio

Usage:
    python evaluation/evaluate_wer_only.py --batch-size 32
"""

import os
import glob
import argparse
import csv
from typing import List

import torch
import jiwer
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Fast batched WER Evaluation")
    parser.add_argument("--eval-dir", default="./eval_results", 
                        help="Directory containing generated synth_*.wav files")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-expandedtry", 
                        help="HuggingFace dataset ID for ground-truth texts")
    parser.add_argument("--cache-dir", default="./data_cache", 
                        help="HF dataset cache directory")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="ASR batch size. Higher = faster GPU saturation (32-64 is great for A40)")
    parser.add_argument("--model", default="openai/whisper-medium", 
                        help="ASR model to use for transcription")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Initializing fast ASR pipeline on {device.upper()} ...")
    print(f"   Model: {args.model} | Batch Size: {args.batch_size}")
    
    try:
        asr_pipe = pipeline(
            "automatic-speech-recognition", 
            model=args.model, 
            device=device,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"❌ Failed to load ASR pipeline: {e}")
        return

    print(f"\n📥 Loading test dataset to retrieve full ground-truth texts ...")
    try:
        ds_test = load_dataset(
            args.dataset, 
            data_files={"test": "data/test-*.parquet"},
            split="test", 
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # 1. Find all generated audio files
    wav_files = glob.glob(os.path.join(args.eval_dir, "synth_*.wav"))
    if not wav_files:
        print(f"❌ No generated audio files found in {args.eval_dir}")
        print("   Make sure the main evaluation script generated some samples first!")
        return
    
    print(f"🔍 Found {len(wav_files)} synthesized audio files. Pairing with ground-truth texts ...")
    
    # 2. Match audio files with their original text in the dataset using the ID in the filename
    eval_items = []
    for path in wav_files:
        basename = os.path.basename(path)
        # Format: synth_{speaker_id}_{index:05d}.wav
        parts = basename.replace(".wav", "").split("_")
        try:
            idx = int(parts[-1])
            row = ds_test[idx]
            text_fr = (row.get("trg_fr_text") or "").strip()
            
            if text_fr:
                eval_items.append({"path": path, "text": text_fr, "idx": idx})
        except Exception as e:
            # Safely ignore files that don't match the numbering convention
            pass

    if not eval_items:
        print("❌ No valid texts recovered. Ensure the dataset matches the indices.")
        return

    # 3. Transcribe audio files in hyper-fast batches
    print(f"\n🎙 Transcribing {len(eval_items)} audio files in batches of {args.batch_size} ...")
    paths = [item["path"] for item in eval_items]
    transcriptions = []
    
    # We pass the paths directly to the pipeline. Because batch_size was set in pipeline init,
    # it automatically collates and GPU-batches the data behind the scenes.
    for out in tqdm(asr_pipe(paths, generate_kwargs={"language": "french"}), total=len(paths), desc="Batch processing"):
        transcriptions.append(out["text"])

    # 4. Calculate WER
    print("\n📊 Calculating Word Error Rate (WER) ...")
    wer_scores = []
    # Standardize string formatting for fair WER computing
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    results = []
    for item, hyp in zip(eval_items, transcriptions):
        ref = item["text"]
        ref_clean = transform(ref)
        hyp_clean = transform(hyp)
        
        if not ref_clean:
            continue
            
        w = float(jiwer.wer(ref_clean, hyp_clean))
        wer_scores.append(w)
        
        results.append({
            "index": item["idx"],
            "reference_text": ref,
            "transcribed_text": hyp,
            "WER": f"{w:.4f}"
        })

    # 5. Output Summary
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
    
    print("\n" + "="*60)
    print("  FAST WER EVALUATION SUMMARY")
    print("="*60)
    print(f"  Samples Evaluated: {len(wer_scores)}")
    print(f"  ASR Model:         {args.model}")
    print(f"  Average WER:       {avg_wer:.4f} ({(avg_wer*100):.2f}%)")
    print("="*60)

    # Save to a dedicated CSV file
    out_csv = os.path.join(args.eval_dir, "wer_only_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
        
    print(f"\n📄 Saved transcriptions and per-sample WER to: {out_csv}")
    print("✅ Done!")

if __name__ == "__main__":
    main()
