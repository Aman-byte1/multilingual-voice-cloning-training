#!/usr/bin/env python3
"""
Fast, batched Content Accuracy Evaluation:
Computes WER, chrF++, and COMET translations scores for generated audio.

Prerequisites:
    pip install transformers jiwer datasets torchaudio sacrebleu unbabel-comet

Usage:
    python evaluation/evaluate_content_accuracy.py
"""

import os
import glob
import argparse
import csv

import torch
import jiwer
import sacrebleu
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from comet import download_model, load_from_checkpoint

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Fast Batched Translation Accuracy Metrics")
    parser.add_argument("--eval-dir", default="./eval_results", 
                        help="Directory containing generated synth_*.wav files")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-expandedtry")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--batch-size", type=int, default=32, help="ASR batch size")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Must match evaluate_metrics subsampling limit")
    parser.add_argument("--asr-model", default="openai/whisper-medium")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n🚀 Initializing fast ASR pipeline on {device.upper()} ...")
    try:
        asr_pipe = pipeline("automatic-speech-recognition", model=args.asr_model, 
                            device=device, batch_size=args.batch_size, chunk_length_s=30)
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")
        return

    print("🚀 Initializing COMET Translation model ...")
    try:
        comet_path = download_model(args.comet_model)
        comet = load_from_checkpoint(comet_path)
        comet.eval()
        if device == "cuda": comet.cuda()
    except Exception as e:
        print(f"❌ Failed to load COMET: {e}")
        return

    print(f"\n📥 Loading dataset to retrieve full ground-truth texts ...")
    ds_test = load_dataset(args.dataset, data_files={"test": "data/test-*.parquet"}, split="test", cache_dir=args.cache_dir)

    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        indices = list(range(0, total, total // args.max_samples))[:args.max_samples]
        ds_test = ds_test.select(indices)

    wav_files = glob.glob(os.path.join(args.eval_dir, "synth_*.wav"))
    if not wav_files:
        print(f"❌ No audio files in {args.eval_dir}")
        return
    
    eval_items = []
    for path in wav_files:
        basename = os.path.basename(path)
        try:
            idx = int(basename.replace(".wav", "").split("_")[-1])
            row = ds_test[idx]
            ref_fr = (row.get("trg_fr_text") or "").strip()
            src_en = (row.get("ref_en_text") or "").strip() # Needed for COMET
            if ref_fr and src_en:
                eval_items.append({"path": path, "ref_fr": ref_fr, "src_en": src_en, "idx": idx})
        except: pass

    if not eval_items:
        print("❌ No matching text samples recovered.")
        return

    # 1. Transcribe audio files in batches
    print(f"\n🎙 Transcribing {len(eval_items)} audio files in batches of {args.batch_size} ...")
    import librosa
    def audio_generator():
        for item in eval_items:
            audio, _ = librosa.load(item["path"], sr=16000)
            yield audio

    transcriptions = []
    for out in tqdm(asr_pipe(audio_generator(), generate_kwargs={"language": "french"}), total=len(eval_items), desc="Whisper ASR"):
        transcriptions.append(out["text"])

    # 2. Compute WER, chrF++
    print("\n📊 Computing Intelligibility (WER) and Semantic (chrF++) ...")
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    
    wer_scores, chrf_scores = [], []
    for item, hyp in zip(eval_items, transcriptions):
        ref = item["ref_fr"]
        rc, hc = transform(ref), transform(hyp)
        w = float(jiwer.wer(rc, hc)) if rc else 0.0
        wer_scores.append(w)
        
        c = sacrebleu.sentence_chrf(hyp, [ref]).score
        chrf_scores.append(c)

    # 3. Compute COMET
    print("🌍 Computing Translation Quality (COMET) in batched parallel inference ...")
    data = [{"src": item["src_en"], "mt": hyp, "ref": item["ref_fr"]} for item, hyp in zip(eval_items, transcriptions)]
    comet_out = comet.predict(data, batch_size=args.batch_size, gpus=1 if device=="cuda" else 0)
    comet_scores = comet_out.scores

    # Build CSV output
    results = []
    for i, (item, hyp, w, c_plus, cmt) in enumerate(zip(eval_items, transcriptions, wer_scores, chrf_scores, comet_scores)):
        results.append({
            "index": item["idx"],
            "source_en": item["src_en"][:50],
            "reference_fr": item["ref_fr"][:50],
            "transcription": hyp[:50],
            "WER": f"{w:.4f}",
            "chrF++": f"{c_plus:.4f}",
            "COMET": f"{cmt:.4f}"
        })

    # Output Summary
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
    avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
    avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0.0
    
    print("\n" + "="*60)
    print("  CONTENT ACCURACY SUMMARY (Translation Quality)")
    print("="*60)
    print(f"  ASR Model:         {args.asr_model}")
    print(f"  COMET Model:       {args.comet_model}")
    print(f"  Samples:           {len(eval_items)}")
    print("-" * 60)
    print(f"  Average WER:       {avg_wer:.4f} (Intelligibility)")
    print(f"  Average chrF++:    {avg_chrf:.4f} (String similarity)")
    print(f"  Average COMET:     {avg_comet:.4f} (Semantic Quality)")
    print("="*60)

    import csv
    out_csv = os.path.join(args.eval_dir, "content_accuracy_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    
    print(f"✅ Fast content eval done -> {out_csv}")

if __name__ == "__main__":
    main()
