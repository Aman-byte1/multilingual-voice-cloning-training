#!/usr/bin/env python3
"""
Evaluation script to compute synthesized speech metrics.

Metrics computed:
1. Perceptual Evaluation of Speech Quality (PESQ) [14]
   - Measures distortions and changes in quality compared to reference audio.
2. Mel Cepstral Distortion (MCD) [16]
   - Measures distance between mel-cepstral coefficients of original and synthesized speech, quantifying tonality similarity.
3. Speaker Similarity
   - Cosine similarity of speaker embeddings to quantify voice match using SpeechBrain ECAPA-TDNN.

Prerequisites:
  pip install pesq pymcd speechbrain librosa
"""

import os
import argparse
import warnings

import numpy as np
import librosa
import torch
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Metric Libraries ---
try:
    from pesq import pesq, NoUtterancesError
except ImportError:
    pesq = None
    print("WARNING: 'pesq' library is not installed. To evaluate PESQ, run: pip install pesq")

try:
    from pymcd.mcd import Calculate_MCD
except ImportError:
    Calculate_MCD = None
    print("WARNING: 'pymcd' library is not installed. To evaluate MCD, run: pip install pymcd")

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    try:
        # Fallback for older speechbrain versions
        from speechbrain.pretrained import SpeakerRecognition
    except ImportError:
        SpeakerRecognition = None
        print("WARNING: 'speechbrain' is not installed. To evaluate Speaker Similarity, run: pip install speechbrain")


class VoiceEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # MCD Toolbox
        self.mcd_toolbox = None
        if Calculate_MCD is not None:
            self.mcd_toolbox = Calculate_MCD(MCD_mode="plain")
            
        # Speaker Recognition Model
        self.verification_model = None
        if SpeakerRecognition is not None:
            print("Loading SpeechBrain ECAPA-TDNN for Speaker Similarity computation...")
            self.verification_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )

    def evaluate_pair(self, ref_audio_path: str, synth_audio_path: str) -> dict:
        """Evaluates a single pair of reference and synthesized audio files."""
        metrics = {
            "PESQ": np.nan,
            "MCD": np.nan,
            "Speaker_Similarity": np.nan
        }

        # 1. PESQ (Perceptual Evaluation of Speech Quality)
        if pesq is not None:
            try:
                # PESQ 'wb' requires identical 16kHz audio sampling rate
                ref_wav, _ = librosa.load(ref_audio_path, sr=16000)
                synth_wav, _ = librosa.load(synth_audio_path, sr=16000)
                
                # Trim to minimum length so arrays match. 
                # PESQ compares aligned signals.
                min_len = min(len(ref_wav), len(synth_wav))
                ref_trim = ref_wav[:min_len]
                synth_trim = synth_wav[:min_len]

                if min_len > 16000 * 0.5: # More than 0.5 seconds required
                    score = pesq(16000, ref_trim, synth_trim, 'wb')
                    metrics["PESQ"] = float(score)
            except NoUtterancesError:
                pass # Silent sections causing errors
            except Exception as e:
                print(f"PESQ computing error for {os.path.basename(synth_audio_path)}: {e}")

        # 2. MCD (Mel Cepstral Distortion)
        if self.mcd_toolbox is not None:
            try:
                mcd_value = self.mcd_toolbox.calculate_mcd(ref_audio_path, synth_audio_path)
                metrics["MCD"] = float(mcd_value)
            except Exception as e:
                print(f"MCD computing error for {os.path.basename(synth_audio_path)}: {e}")

        # 3. Speaker Similarity (Cosine distance of ECAPA-TDNN embeddings)
        if self.verification_model is not None:
            try:
                score, prediction = self.verification_model.verify_files(ref_audio_path, synth_audio_path)
                metrics["Speaker_Similarity"] = float(score.item())
            except Exception as e:
                print(f"Speaker Similarity error for {os.path.basename(synth_audio_path)}: {e}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics: MCD, PESQ, Speaker Similarity for VC.")
    parser.add_argument("--ref", type=str, help="Path to reference original audio file")
    parser.add_argument("--synth", type=str, help="Path to synthesized audio file")
    parser.add_argument("--ref-dir", type=str, help="Path to reference audio directory")
    parser.add_argument("--synth-dir", type=str, help="Path to synthesized audio directory")
    args = parser.parse_args()

    if (not args.ref or not args.synth) and (not args.ref_dir or not args.synth_dir):
        print("Please provide exactly one pair: either --ref and --synth, OR --ref-dir and --synth-dir")
        return

    evaluator = VoiceEvaluator()

    # Mode 1: Single Pair Evaluation
    if args.ref and args.synth:
        print(f"\nEvaluating single pair:\nReference: {args.ref}\nSynthesized: {args.synth}")
        res = evaluator.evaluate_pair(args.ref, args.synth)
        print("\n--- RESULTS ---")
        for k, v in res.items():
            print(f"{k}: {v:.4f}")
            
    # Mode 2: Directory Batch Evaluation
    elif args.ref_dir and args.synth_dir:
        print(f"\nEvaluating directory:\nReference Dir: {args.ref_dir}\nSynthesized Dir: {args.synth_dir}")
        synth_files = [f for f in os.listdir(args.synth_dir) if f.endswith('.wav')]
        
        all_metrics = {"PESQ": [], "MCD": [], "Speaker_Similarity": []}
        
        valid_files_processed = 0
        for f in tqdm(synth_files, desc="Batch evaluation"):
            synth_path = os.path.join(args.synth_dir, f)
            
            # Assume reference and synthesis filename matches.
            # If generated as 'sample_0.wav' then we assume a 'sample_0.wav' exists in ref dir.
            ref_path = os.path.join(args.ref_dir, f)
            
            if not os.path.exists(ref_path):
                continue
                
            valid_files_processed += 1
            res = evaluator.evaluate_pair(ref_path, synth_path)
            
            for k, v in res.items():
                if not np.isnan(v):
                    all_metrics[k].append(v)
                    
        print(f"\n--- AVERAGE RESULTS ACROSS {valid_files_processed} FILES ---")
        for k, vals in all_metrics.items():
            avg = np.mean(vals) if vals else float('nan')
            count_valid = len(vals)
            print(f"{k}: {avg:.4f} (Calculated on {count_valid}/{valid_files_processed} valid audio instances)")

if __name__ == "__main__":
    main()
