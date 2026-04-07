#!/usr/bin/env python3
"""
Cross-Lingual Voice Cloning Evaluation Pipeline
================================================
Chatterbox base vs. LoRA fine-tuned model evaluation.
Supports dynamic language targets (Arabic, Chinese, French, etc.) 
and streamlined metrics (Time, WER, CER, Similarity).

ASR: faster-whisper large-v3
"""

import os
import sys
import math
import csv
import json
import time
import argparse
import warnings
import tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")


# ===================================================================
# LoRA helpers (must match training code)
# ===================================================================

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        self.original_layer = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = original.in_features, original.out_features
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.lora_dropout = (nn.Dropout(dropout) if dropout > 0
                             else nn.Identity())
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original_layer(x)
        lora = (self.lora_dropout(x) @ self.lora_A.T
                @ self.lora_B.T * self.scaling)
        return base + lora


# ===================================================================
# Helper functions
# ===================================================================

def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval") -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path

def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )

def extract_speaker_embedding(wav_path, model, device="cuda"):
    """Extract speaker embedding, resampling to 16kHz (ECAPA-TDNN requirement)."""
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0).to(device)  # mono
        emb = model.encode_batch(wav.unsqueeze(0))
        return emb.squeeze(0).squeeze(0).detach()
    except Exception:
        return None

def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.mean(v) if v else np.nan

def safe_std(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.std(v) if v else np.nan

def safe_count(vals):
    return len([x for x in vals if x is not None and not np.isnan(x)])


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Lingual Voice Cloning Evaluation")
    parser.add_argument("--dataset",
                        default="ymoslem/acl-6060")
    parser.add_argument("--repo-id",
                        default="amanuelbyte/chatterbox-fr-lora")
    parser.add_argument("--lora-file",
                        default="best_lora_adapter.pt")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr",
                        help="Language code (fr, ar, zh, etc.)")
    parser.add_argument("--cfg-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    mode = "BASE" if args.skip_lora else "LoRA"
    target_lang = args.whisper_lang.strip().lower()

    print("=" * 64)
    print(f"  CROSS-LINGUAL EVALUATION — {mode} ({target_lang})")
    print("=" * 64)

    # PHASE 1: Load dataset
    print(f"\n📥 Phase 1: Loading dataset from {args.dataset}")
    ds_test = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)
    print(f"   Samples: {total}")

    # PHASE 2: Load Chatterbox
    print(f"\n🔧 Phase 2: Loading Chatterbox model")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if not args.skip_lora:
        try:
            lora_path = hf_hub_download(repo_id=args.repo_id, filename=args.lora_file)
            payload = torch.load(lora_path, map_location=device, weights_only=True)
            lora_cfg = payload.get("config", {})
            targets = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
            rank = lora_cfg.get("rank", 8)
            alpha = lora_cfg.get("alpha", 16.0)
            dropout = lora_cfg.get("dropout", 0.1)
            
            for name, module in model.t3.named_modules():
                if isinstance(module, nn.Linear) and any(k in name for k in targets):
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.t3.get_submodule(parent_name)
                    setattr(parent, child_name, LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout))
            
            lora_sd = payload.get("lora_state_dict", payload)
            model.t3.load_state_dict(lora_sd, strict=False)
            print(f"   LoRA injected from {args.repo_id} ✓")
        except Exception as e:
            print(f"   ⚠ Failed to load LoRA: {e}. Falling back to BASE.")

    model.t3.eval()
    model_sr = model.sr

    # PHASE 3: Generate
    print(f"\n🎙  Phase 3: Generating {total} samples")
    samples = []
    skipped = 0
    inference_times = []
    audio_durations = []

    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        text_target = (row.get(f"trg_{target_lang}_text") or row.get(f"text_{target_lang}") or "").strip()
        ref_data = row.get("ref_en_voice") or row.get(f"ref_{target_lang}_voice") or row.get("audio_en") or row.get("audio")

        if not ref_data or not text_target:
            skipped += 1
            continue

        ref_path = save_temp_wav(np.asarray(ref_data["array"], dtype=np.float32), ref_data["sampling_rate"], "ref_")

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                wav = model.generate(
                    text_target,
                    audio_prompt_path=ref_path,
                    language_id=target_lang,
                    cfg_weight=args.cfg_weight
                )
            t1 = time.perf_counter()
        except Exception:
            if ref_path and os.path.exists(ref_path): os.remove(ref_path)
            skipped += 1
            continue

        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        wav_out = wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu()
        torchaudio.save(syn_path, wav_out, model_sr)
        
        elapsed = t1 - t0
        audio_dur = wav_out.numel() / model_sr
        inference_times.append(elapsed)
        audio_durations.append(audio_dur)

        samples.append({
            "idx": i,
            "syn_path": syn_path,
            "ref_path": ref_path,
            "text_target": text_target,
            "speaker_id": row.get("speaker_id", "unknown"),
            "inference_s": elapsed,
            "audio_dur_s": audio_dur,
            "rtf": elapsed / audio_dur if audio_dur > 0 else 0
        })

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")
    del model
    torch.cuda.empty_cache()

    # PHASE 4: ASR
    print(f"\n🗣  Phase 4: Transcribing ({target_lang})")
    from faster_whisper import WhisperModel as FasterWhisperModel
    whisper = FasterWhisperModel(args.whisper_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    
    transcripts = []
    for s in tqdm(samples, desc="Transcribing"):
        try:
            segments, _ = whisper.transcribe(s["syn_path"], language=target_lang, beam_size=args.whisper_beam, vad_filter=True)
            transcripts.append(" ".join(seg.text for seg in segments).strip())
        except Exception:
            transcripts.append("")

    del whisper
    torch.cuda.empty_cache()

    # PHASE 5: Metrics
    print(f"\n📊 Phase 5: Computing Metrics")
    verifier = load_speaker_model(device=device)
    import jiwer
    wer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.RemovePunctuation(), jiwer.ReduceToListOfListOfWords(),
    ])

    results = []
    for s, tx in tqdm(zip(samples, transcripts), total=len(samples), desc="Metrics"):
        # Speaker Similarity
        syn_emb = extract_speaker_embedding(s["syn_path"], verifier, device)
        ref_emb = extract_speaker_embedding(s["ref_path"], verifier, device)
        sim = float(F.cosine_similarity(syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()) if (syn_emb is not None and ref_emb is not None) else None

        # WER / CER
        try:
            if tx.strip():
                ref_clean = wer_transforms(s["text_target"])
                hyp_clean = wer_transforms(tx)
                if isinstance(ref_clean, list) and len(ref_clean) > 0 and isinstance(ref_clean[0], list): ref_clean = " ".join(ref_clean[0])
                if isinstance(hyp_clean, list) and len(hyp_clean) > 0 and isinstance(hyp_clean[0], list): hyp_clean = " ".join(hyp_clean[0])
                elif isinstance(ref_clean, list): ref_clean = " ".join(ref_clean)
                if isinstance(hyp_clean, list): hyp_clean = " ".join(hyp_clean)
                
                w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
                c = float(jiwer.cer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
            else:
                w = c = 1.0
        except Exception:
            w = c = None

        if os.path.exists(s["ref_path"]): os.remove(s["ref_path"])

        results.append({
            "idx": s["idx"], "WER": w, "CER": c, "Similarity": sim,
            "InferenceS": s["inference_s"], "AudioDurS": s["audio_dur_s"], "RTF": s["rtf"]
        })

    # Summary
    metric_keys = ["WER", "CER", "Similarity", "InferenceS", "RTF"]
    overall = {k: {"mean": safe_mean([r[k] for r in results]), "std": safe_std([r[k] for r in results]), "valid": safe_count([r[k] for r in results])} for k in metric_keys}

    print("\n" + "=" * 62)
    print("  EVALUATION COMPLETE")
    print(f"  Target: {target_lang} | Samples: {len(results)}")
    print("=" * 62)
    print(f"  {'Metric':<16} {'Mean':>9} {'± Std':>9}  {'Valid':>6}")
    print("-" * 62)
    for k in metric_keys:
        m, s, v = overall[k]["mean"], overall[k]["std"], overall[k]["valid"]
        print(f"  {k:<16} {m:>9.4f} {f'±{s:.4f}' if not np.isnan(s) else '':>9}  {v:>3}/{len(results)}")
    print("=" * 62)

    # Save
    with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

if __name__ == "__main__":
    main()
