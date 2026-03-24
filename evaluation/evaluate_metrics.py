#!/usr/bin/env python3
"""
🚀 HIGH-PERFORMANCE RTX 4090 OPTIMIZED EVALUATION (v2: Corpus Metrics)
End-to-end evaluation of the Chatterbox FR LoRA fine-tuned model.

Methodology Refresh:
  1. ASR Engine: Upgrade to Whisper-Large-v3 (Accuracy > Speed)
  2. Metrics: Implement standard Corpus-Level WER and chrF++
  3. Precision: Pure FP32 + TF32 acceleration
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
    from comet import download_model, load_from_checkpoint
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model.eval()
    if device == "cuda": model.cuda()
    return model

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
    parser = argparse.ArgumentParser(description="Accelerated Chatterbox Evaluation (Corpus Metrics v2)")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-expandedtry")
    parser.add_argument("--repo-id", default="amanuelbyte/chatterbox-fr-lora")
    parser.add_argument("--lora-file", default="best_lora_adapter.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="Lowered BS for Large-v3 VRAM")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Loading data
    print(f"📥 Loading TEST split from {args.dataset} ...")
    ds_test = load_dataset(args.dataset, data_files={"test": "data/test-*.parquet"}, split="test", cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        indices = list(range(0, total, total // args.max_samples))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)

    # 2. Loading model
    if not args.skip_lora:
        lora_path = hf_hub_download(repo_id=args.repo_id, filename=args.lora_file)
    else: lora_path = None

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if not args.skip_lora and lora_path:
        checkpoint = torch.load(lora_path, map_location=device)
        layers = []
        for name, module in model.t3.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                parent_name, child_name = ".".join(name.split(".")[:-1]), name.split(".")[-1]
                parent = model.t3.get_submodule(parent_name)
                lora_mod = LoRALayer(module, rank=16, alpha=32)
                setattr(parent, child_name, lora_mod)
                layers.append(name)
        model.load_state_dict(checkpoint, strict=False)
        print(f"   LoRA injected ✓\n")
    model.t3.eval()

    # 3. Phase 1: Generation
    print(f"🎙 PHASE 1/4: Generating {total} audio samples (Serial AR with TF32) ...")
    samples_data = [] # Stores paths and metadata
    for i in tqdm(range(total), desc="Generating"):
        row = ds_test[i]
        text_fr = (row.get("trg_fr_text") or "").strip()
        text_en = (row.get("ref_en_text") or "").strip()
        
        # Audio Prompts
        ref_data = row.get("ref_en_voice") or row.get("ref_fr_voice")
        if not ref_data or not text_fr: continue
        
        ref_wav_path = save_temp_wav(np.asarray(ref_data["array"], dtype=np.float32), ref_data["sampling_rate"], "ref_")
        
        # Ground Truth
        gt_data = row.get("trg_fr_voice")
        gt_wav_path = save_temp_wav(np.asarray(gt_data["array"], dtype=np.float32), gt_data["sampling_rate"], "gt_")
        
        with torch.inference_mode():
            wav = model.generate(text_fr, audio_prompt_path=ref_wav_path, language_id="fr")
        
        os.remove(ref_wav_path) # Clean prompt immediately
        
        synth_wav_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        torchaudio.save(synth_wav_path, wav.cpu() if wav.dim() > 1 else wav.unsqueeze(0).cpu(), model.sr)
        
        samples_data.append({
            "idx": i, "synth_path": synth_wav_path, "gt_path": gt_wav_path,
            "text_fr": text_fr, "text_en": text_en, "speaker_id": row.get("speaker_id", "unknown")
        })

    # 4. Phase 2: BATCHED ASR (Whisper Large-v3)
    print(f"\n🎙 PHASE 2/4: Transcribing audio in batches of {args.batch_size} (WHISPER LARGE-V3) ...")
    from transformers import pipeline
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device, batch_size=args.batch_size, chunk_length_s=30)
    
    def audio_gen():
        for s in samples_data:
            audio, _ = librosa.load(s["synth_path"], sr=16000)
            yield audio

    transcripts = []
    for out in tqdm(asr_pipe(audio_gen(), generate_kwargs={"language": "french"}), total=len(samples_data), desc="Transcribing"):
        transcripts.append(out["text"])

    # 5. Phase 3: BATCHED COMET
    print(f"\n🌍 PHASE 3/4: Scoring translation quality in batches ...")
    comet_model = load_comet_model(device=device)
    comet_data = [{"src": s["text_en"], "mt": tx, "ref": s["text_fr"]} for s, tx in zip(samples_data, transcripts)]
    comet_scores = comet_model.predict(comet_data, batch_size=args.batch_size, gpus=1 if "cuda" in device else 0).scores

    # 6. Phase 4: FINAL METRICS
    print(f"\n📊 PHASE 4/4: Calculating benchmark results (Corpus-Level) ...")
    verifier = load_speaker_model()
    import jiwer
    import sacrebleu
    
    # 1. String Matchers (Corpus-level)
    references = [s["text_fr"] for s in samples_data]
    
    # Clean refs/hyps for WER
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    refs_clean = [transform(r) for r in references]
    hyps_clean = [transform(t) for t in transcripts]
    
    corpus_wer = float(jiwer.wer(refs_clean, hyps_clean))
    corpus_chrf = float(sacrebleu.corpus_chrf(transcripts, [references]).score)
    mean_comet = float(np.mean(comet_scores))

    # 2. Acoustic Matchers (Per-sample mean)
    pesq_list, mcd_list, sim_list = [], [], []
    results = []
    for i, (s, tx, comet_val) in enumerate(tqdm(zip(samples_data, transcripts, comet_scores), total=len(samples_data), desc="Acoustic Alignment")):
        p = compute_pesq_score(s["gt_path"], s["synth_path"])
        m = compute_mcd_score(s["gt_path"], s["synth_path"])
        sim = float(verifier.verify_files(s["gt_path"], s["synth_path"])[0].item())
        
        if p: pesq_list.append(p)
        if m: mcd_list.append(m)
        sim_list.append(sim)
        
        os.remove(s["gt_path"]) # Clean GT temp wav
        
        results.append({
            "idx": s["idx"], "speaker": s["speaker_id"], 
            "WER_sent": float(jiwer.wer(refs_clean[i], hyps_clean[i])) if refs_clean[i] else 0.0,
            "COMET_score": comet_val, "PESQ_wb": p or 0, "MCD_plain": m or 0, "Similarity_cos": sim
        })

    # Summary Generation
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    final_stats = {
        "corpus_wer": corpus_wer,
        "corpus_chrf": corpus_chrf,
        "mean_comet": mean_comet,
        "mean_pesq": float(np.mean(pesq_list)) if pesq_list else 0.0,
        "mean_mcd": float(np.mean(mcd_list)) if mcd_list else 0.0,
        "mean_similarity": float(np.mean(sim_list))
    }
    with open(summary_path, "w") as f: json.dump({"stats": final_stats, "results": results}, f, indent=2)

    print("\n" + "="*60)
    print("  DEFINITIVE BENCHMARK RESULTS (Corpus-Scale)")
    print("="*60)
    print(f"  ASR Engine:         Whisper Large-v3")
    print("-" * 60)
    print(f"  Corpus WER:         {corpus_wer:.4f}  (Lower is better, 0.0-1.0 range)")
    print(f"  Corpus chrF++:      {corpus_chrf:.4f} (Higher is better, 0-100 range)")
    print(f"  Mean COMET:         {mean_comet:.4f}  (Higher is better, neural semantic)")
    print("-" * 60)
    print(f"  Mean Similarity:    {final_stats['mean_similarity']:.4f} (0.0-1.0 Cosine)")
    print(f"  Mean PESQ:          {final_stats['mean_pesq']:.4f} (1.0-4.5 Wideband)")
    print(f"  Mean MCD:           {final_stats['mean_mcd']:.4f} (Lower is better)")
    print("="*60)
    print(f"✅ Definitive report done -> {args.output_dir}")

if __name__ == "__main__":
    main()
