#!/usr/bin/env python3
"""
Aggregate parallel evaluation results.
Merges partition-specific CSVs and JSONs into a single eval_summary.json and eval_per_sample.csv.

Usage: python evaluation/aggregate_results.py --output-dir ./eval_results/zh_base
"""

import os
import csv
import json
import glob
import argparse
import numpy as np


def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def safe_std(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.std(v)) if v else float('nan')

def safe_count(vals):
    return len([x for x in vals if x is not None and not np.isnan(x)])


def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    # Map 'cuda' to 'cuda:0' to prevent SpeechBrain device string parsing warnings/errors
    sb_device = "cuda:0" if device == "cuda" else device
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": sb_device}
    )

def extract_speaker_embedding_from_wav(wav_path, model, device="cuda"):
    import torchaudio
    import torch
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).detach()
    except Exception as e:
        print(f"   ⚠ Failed to extract embedding from wav {wav_path}: {e}")
        return None

def extract_speaker_embedding_from_array(audio_array, sr, model, device="cuda"):
    import torch
    import torchaudio
    try:
        wav = torch.from_numpy(audio_array).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).detach()
    except Exception as e:
        print(f"   ⚠ Failed to extract embedding from array: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Aggregate parallel eval results")
    parser.add_argument("--output-dir", required=True, help="Directory with partition results")
    parser.add_argument("--dataset", default="ymoslem/acl-6060", help="Hugging Face dataset name")
    parser.add_argument("--split", default="eval", help="Dataset split")
    parser.add_argument("--cache-dir", default="./data_cache", help="Dataset cache dir")
    args = parser.parse_args()

    # Find all partition CSVs
    pattern = os.path.join(args.output_dir, "eval_per_sample_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No partition CSVs found matching {pattern}")
        print("Looking for single eval_per_sample.csv...")
        single = os.path.join(args.output_dir, "eval_per_sample.csv")
        if os.path.exists(single):
            print(f"Found {single} — no aggregation needed.")
        return

    print(f"Found {len(csv_files)} partition files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Merge all rows
    all_results = []
    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for k in ["WER", "CER", "Similarity", "InferenceS", "AudioDurS", "RTF"]:
                    try:
                        row[k] = float(row[k]) if row[k] and row[k] != "None" else None
                    except (ValueError, TypeError):
                        row[k] = None
                try:
                    row["idx"] = int(row["idx"])
                except (ValueError, TypeError):
                    pass
                all_results.append(row)

    # Sort by idx
    all_results.sort(key=lambda r: r.get("idx", 0))
    print(f"\nTotal samples: {len(all_results)}")

    # Check if we have missing/nan similarity scores
    missing_sim_indices = [r["idx"] for r in all_results if r["Similarity"] is None or np.isnan(r["Similarity"])]
    
    if missing_sim_indices:
        print(f"\n🔍 Found {len(missing_sim_indices)} samples with missing/nan Speaker Similarity.")
        print("🧠 Re-calculating Speaker Similarity using Hugging Face references...")
        
        import torch
        import torch.nn.functional as F
        from datasets import load_dataset
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Loading SpeechBrain verifier on {device}...")
        spk_model = load_speaker_model(device=device)
        
        print("   Loading Hugging Face dataset...")
        ds_test = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
        
        # Build map from idx to result dictionary
        result_map = {r["idx"]: r for r in all_results}
        
        # We need to map target language to correctly extract references if needed
        # Defaults to 'fr' / 'zh' based on eval split structure
        target_lang = "zh"
        
        for idx in tqdm(missing_sim_indices, desc="Re-calculating Similarity"):
            row_data = ds_test[idx]
            ref_data = row_data.get("ref_en_voice") or row_data.get(f"ref_{target_lang}_voice") or row_data.get("audio_en") or row_data.get("audio")
            
            syn_path = os.path.join(args.output_dir, f"synth_{idx:05d}.wav")
            
            if ref_data and os.path.exists(syn_path):
                # Extract embeddings
                ref_emb = extract_speaker_embedding_from_array(
                    np.asarray(ref_data["array"], dtype=np.float32),
                    ref_data["sampling_rate"],
                    spk_model,
                    device
                )
                syn_emb = extract_speaker_embedding_from_wav(
                    syn_path,
                    spk_model,
                    device
                )
                
                if ref_emb is not None and syn_emb is not None:
                    sim = float(F.cosine_similarity(ref_emb.unsqueeze(0), syn_emb.unsqueeze(0)).item())
                    result_map[idx]["Similarity"] = sim
                else:
                    result_map[idx]["Similarity"] = None
            else:
                result_map[idx]["Similarity"] = None

    # Compute aggregated metrics
    metric_keys = ["WER", "CER", "Similarity", "InferenceS", "RTF"]
    overall = {}
    for k in metric_keys:
        vals = [r[k] for r in all_results]
        overall[k] = {
            "mean": safe_mean(vals),
            "std": safe_std(vals),
            "valid": safe_count(vals)
        }

    # Print summary
    print("\n" + "=" * 62)
    print("  AGGREGATED EVALUATION RESULTS")
    print(f"  Samples: {len(all_results)}")
    print("=" * 62)
    print(f"  {'Metric':<16} {'Mean':>9} {'± Std':>9}  {'Valid':>6}")
    print("-" * 62)
    for k in metric_keys:
        m, s, v = overall[k]["mean"], overall[k]["std"], overall[k]["valid"]
        std_str = f"±{s:.4f}" if not np.isnan(s) else ""
        print(f"  {k:<16} {m:>9.4f} {std_str:>9}  {v:>3}/{len(all_results)}")
    print("=" * 62)

    # Save merged summary
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\n✅ Saved: {summary_path}")

    # Save merged CSV
    merged_csv = os.path.join(args.output_dir, "eval_per_sample.csv")
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "WER", "CER", "Similarity", "InferenceS", "AudioDurS", "RTF", "reference", "transcript"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✅ Saved: {merged_csv}")

    # Cleanup temp_ref if it exists
    temp_ref = os.path.join(args.output_dir, "temp_ref")
    if os.path.exists(temp_ref):
        import shutil
        shutil.rmtree(temp_ref)
        print(f"🧹 Cleaned up: {temp_ref}")


if __name__ == "__main__":
    main()
