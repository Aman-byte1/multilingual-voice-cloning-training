#!/usr/bin/env python3
"""
🚀 HIGH-PERFORMANCE RTX 4090 OPTIMIZED EVALUATION
End-to-end evaluation of the Chatterbox FR LoRA fine-tuned model.

Pipeline (Optimized for Ada/Ampere GPUs):
  1. Serial AR Generation (Phase 1) - Accelerated via TF32
  2. Batched faster-whisper ASR (Phase 2) - Large-v3, CTranslate2
  3. Optional COMET (Phase 3) - Neural quality scoring
  4. Serial Metrics (Phase 4) - PESQ/MCD/Similarity
"""

import os
import sys
import math
import csv
import json
import argparse
import warnings
import tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
from typing import List, Optional, Dict
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Optimization: Enable TensorFloat32 for 4090/A-series speedups
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LoRA helpers (must match training code exactly)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = original.in_features, original.out_features
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original_layer(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_pesq_score(ref_wav_path: str, synth_wav_path: str) -> Optional[float]:
    try:
        from pesq import pesq as pesq_fn
        ref, _ = librosa.load(ref_wav_path, sr=16000)
        syn, _ = librosa.load(synth_wav_path, sr=16000)
        min_len = min(len(ref), len(syn))
        if min_len < 8000: return None
        return float(pesq_fn(16000, ref[:min_len], syn[:min_len], "wb"))
    except: return None

def compute_mcd_score(ref_wav_path: str, synth_wav_path: str) -> Optional[float]:
    try:
        from pymcd.mcd import Calculate_MCD
        mcd_calc = Calculate_MCD(MCD_mode="plain")
        return float(mcd_calc.calculate_mcd(ref_wav_path, synth_wav_path))
    except: return None

def load_speaker_model():
    from speechbrain.inference.speaker import SpeakerRecognition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

def load_comet_model(device="cuda"):
    try:
        from comet import download_model, load_from_checkpoint
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        model.eval()
        if device == "cuda": model.cuda()
        return model
    except ImportError:
        print("   ⚠ COMET not installed — skipping translation quality scoring")
        return None

def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval") -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1: wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path

# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Accelerated Chatterbox Evaluation (RTX 4090 Optimized)")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-cleaned")
    parser.add_argument("--repo-id", default="amanuelbyte/chatterbox-fr-lora-v2")
    parser.add_argument("--lora-file", default="best_lora_adapter.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ASR/COMET")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Loading data
    print(f"📥 Loading TEST split from {args.dataset} ...")
    ds_test = load_dataset(args.dataset, split="test", cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        indices = list(range(0, total, total // args.max_samples))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)

    # 2. Loading model
    if not args.skip_lora:
        if os.path.isfile(args.lora_file):
            lora_path = args.lora_file
        else:
            lora_path = hf_hub_download(repo_id=args.repo_id, filename=args.lora_file)
    else: lora_path = None

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if not args.skip_lora and lora_path:
        payload = torch.load(lora_path, map_location=device, weights_only=True)
        lora_cfg = payload.get("config", {})
        targets = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        rank = lora_cfg.get("rank", 8)
        alpha = lora_cfg.get("alpha", 16.0)
        dropout = lora_cfg.get("dropout", 0.1)
        for name, module in model.t3.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in targets):
                parent_name, child_name = ".".join(name.split(".")[:-1]), name.split(".")[-1]
                parent = model.t3.get_submodule(parent_name)
                setattr(parent, child_name, LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout))
        lora_sd = payload.get("lora_state_dict", payload)
        current_sd = model.t3.state_dict()
        loaded = 0
        for key, value in lora_sd.items():
            if key in current_sd:
                current_sd[key] = value.to(device)
                loaded += 1
        model.t3.load_state_dict(current_sd, strict=False)
        print(f"   LoRA injected, {loaded} tensors loaded ✓\n")
    model.t3.eval()

    # 3. Phase 1: FAST Serial Generation
    print(f"🎙 PHASE 1/4: Generating {total} audio samples (Serial AR) ...")
    samples_data = []
    skipped = 0
    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        
        # Support multiple dataset schemas
        text_fr = (row.get("trg_fr_text") or row.get("text_fr") or "").strip()
        text_en = (row.get("ref_en_text") or row.get("text_en") or "").strip()
        
        ref_data = row.get("ref_en_voice") or row.get("audio_en") or row.get("ref_fr_voice")
        gt_data = row.get("trg_fr_voice") or row.get("cloned_audio_fr")
        
        if not ref_data or not text_fr: 
            skipped += 1
            continue
        
        ref_wav_path = save_temp_wav(np.asarray(ref_data["array"], dtype=np.float32), ref_data["sampling_rate"], "ref_")
        
        # Ground truth for metrics (Similarity/PESQ/MCD)
        if gt_data:
            gt_wav_path = save_temp_wav(np.asarray(gt_data["array"], dtype=np.float32), gt_data["sampling_rate"], "gt_")
        else:
            gt_wav_path = None
        
        try:
            with torch.inference_mode():
                wav = model.generate(text_fr, audio_prompt_path=ref_wav_path, language_id="fr")
        except (RuntimeError, Exception) as e:
            # CUDA assertion errors from s3gen on edge-case inputs — skip sample
            skipped += 1
            tqdm.write(f"   ⚠ Sample {i} failed ({type(e).__name__}), skipping ({skipped} total)")
            os.remove(ref_wav_path)
            os.remove(gt_wav_path)
            # Reset CUDA state after device-side assertion
            torch.cuda.synchronize()
            continue
        
        os.remove(ref_wav_path)
        
        synth_wav_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        torchaudio.save(synth_wav_path, wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu(), model.sr)
        
        samples_data.append({
            "idx": i, "synth_path": synth_wav_path, "gt_path": gt_wav_path,
            "text_fr": text_fr, "text_en": text_en, "speaker_id": row.get("speaker_id", "unknown")
        })
    
    print(f"   Generated {len(samples_data)}/{total} samples ({skipped} skipped)")

    # 4. Phase 2: faster-whisper ASR (Large-v3, CTranslate2-accelerated)
    print(f"\n🎙 PHASE 2/4: Transcribing with faster-whisper {args.whisper_model} ...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
    
    transcripts = []
    for s in tqdm(samples_data, desc="Transcribing"):
        segments, _ = whisper.transcribe(s["synth_path"], language="fr", beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        transcripts.append(text)
    
    # Free whisper GPU memory before next phase
    del whisper
    torch.cuda.empty_cache()

    # 5. Phase 3: COMET (optional — skip if not installed)
    print(f"\n🌍 PHASE 3/4: Scoring translation quality (COMET) ...")
    comet_model = load_comet_model(device=device)
    if comet_model is not None:
        comet_data = [{"src": s["text_en"], "mt": tx, "ref": s["text_fr"]} for s, tx in zip(samples_data, transcripts)]
        comet_scores = comet_model.predict(comet_data, batch_size=args.batch_size, gpus=1 if "cuda" in device else 0).scores
        del comet_model
        torch.cuda.empty_cache()
    else:
        comet_scores = [0.0] * len(samples_data)

    # 6. Phase 4: FINAL METRICS (Acoustic Axes)
    print(f"\n📊 PHASE 4/4: Calculating acoustic metrics (Similarity, PESQ, MCD) ...")
    verifier = load_speaker_model()
    import jiwer
    import sacrebleu
    
    results = []
    for s, tx, comet_val in tqdm(zip(samples_data, transcripts, comet_scores), total=len(samples_data), desc="Finalizing"):
        if s.get("gt_path"):
            p = compute_pesq_score(s["gt_path"], s["synth_path"])
            m = compute_mcd_score(s["gt_path"], s["synth_path"])
            sim = float(verifier.verify_files(s["gt_path"], s["synth_path"])[0].item())
            os.remove(s["gt_path"])
        else:
            p, m, sim = None, None, None
            
        # WER/chrF
        ref_clean = jiwer.RemovePunctuation()(s["text_fr"].lower())
        hyp_clean = jiwer.RemovePunctuation()(tx.lower())
        w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean else 0.0
        chr_val = float(sacrebleu.sentence_chrf(tx, [s["text_fr"]]).score)
        
        results.append({
            "idx": s["idx"], "speaker": s["speaker_id"], "WER": w, "chrF": chr_val, 
            "COMET": comet_val, "PESQ": p or 0, "MCD": m or 0, "Similarity": sim
        })

    # Summary Generation
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    metric_keys = ["WER", "chrF", "PESQ", "MCD", "Similarity"]
    if comet_model is not None:
        metric_keys.append("COMET")
    stats = {k: float(np.mean([r[k] for r in results])) for k in metric_keys}
    with open(summary_path, "w") as f: json.dump({"stats": stats, "results": results}, f, indent=2)

    print("\n" + "="*50)
    print("  FAST EVALUATION COMPLETE (RTX 4090)")
    print("="*50)
    for k, v in stats.items(): print(f"  {k:12}: {v:.4f}")
    print("="*50)
    print(f"✅ Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
