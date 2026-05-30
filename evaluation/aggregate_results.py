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


def main():
    parser = argparse.ArgumentParser(description="Aggregate parallel eval results")
    parser.add_argument("--output-dir", required=True, help="Directory with partition results")
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
