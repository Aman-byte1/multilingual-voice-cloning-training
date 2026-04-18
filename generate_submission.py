#!/usr/bin/env python3
"""
IWSLT 2026 Submission Generator - Team: afrinlp
Fixed for direct model loading and robust file discovery.
"""
import os
import gc
import sys
import types
import torch
import torchaudio
import argparse
import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial

# ── OmniVoice compatibility patch ──────────────────────────────
# Must happen BEFORE importing omnivoice

# Step 1: Stub the missing module so import doesn't crash
def _install_flex_stub():
    mod_name = "torch.nn.attention.flex_attention"
    if mod_name in sys.modules:
        return
    try:
        import torch.nn.attention as attn_mod
        stub = types.ModuleType(mod_name)
        stub.create_block_mask = lambda *a, **kw: None
        sys.modules[mod_name] = stub
        setattr(attn_mod, "flex_attention", stub)
    except Exception:
        pass

# Step 2: Patch source file WITHOUT importing omnivoice (find via site-packages)
def _patch_omnivoice_source():
    import site
    search_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for sp in search_dirs:
        candidates = glob.glob(os.path.join(sp, 'omnivoice', '**', 'omnivoice_llm.py'), recursive=True)
        for p in candidates:
            try:
                with open(p, 'r') as f: content = f.read()
                if 'flex_attention' in content:
                    with open(p, 'w') as f: f.write(content.replace('flex_attention', 'eager'))
                    print(f'  ✅ Patched OmniVoice → eager attention: {p}')
            except Exception:
                pass

_install_flex_stub()
_patch_omnivoice_source()

# Now safe to import
from omnivoice import OmniVoice
from peft import PeftModel

# Final Winning Checkpoints
BEST_MODELS = {
    "zh": "amanuelbyte/omnivoice-lora-zh-400",
    "ar": "amanuelbyte/omnivoice-lora-ar-400",
    "fr": "amanuelbyte/omnivoice-lora-fr-200",
}

# Fallback names if -400 doesn't exist for all
FALLBACK_MODELS = {
    "ar": "amanuelbyte/omnivoice-lora-ar",
    "fr": "amanuelbyte/omnivoice-lora-fr",
}

MAX_REF_DURATION = 15.0
MAX_CHARS_PER_CHUNK = 200

def get_best_reference(ref_path, duration=10.0):
    """Extract clean speech segment at ORIGINAL sample rate (let OmniVoice resample)."""
    waveform, sr = torchaudio.load(str(ref_path))
    
    # 1. Force Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. VAD Thresholding to skip non-speech
    window_size = int(sr * 0.05) # 50ms
    stride = window_size // 2
    windows = waveform.unfold(-1, window_size, stride)
    energy = torch.sum(windows**2, dim=-1).squeeze(0)
    
    # Find the first frame that exceeds 5% of max energy (clear speech start)
    threshold = torch.max(energy) * 0.05
    active_frames = (energy > threshold).nonzero()
    
    target_samples = int(duration * sr)
    start_idx = active_frames[0].item() * stride if len(active_frames) > 0 else 0
        
    # Take EXACTLY `duration` seconds of audio starting from speech onset
    end_idx = start_idx + target_samples
    best_chunk = waveform[:, start_idx:end_idx]
    
    # Return at ORIGINAL sample rate — OmniVoice handles resampling internally
    return (best_chunk, sr)

def split_text_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK):
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk: chunks.append(current_chunk)
            if len(sentence) > max_chars:
                words = re.split(r'[,，、\s]+', sentence)
                sub_chunk = ""
                for word in words:
                    if len(sub_chunk) + len(word) + 1 <= max_chars:
                        sub_chunk = (sub_chunk + " " + word).strip()
                    else:
                        if sub_chunk: chunks.append(sub_chunk)
                        sub_chunk = word
                if sub_chunk: chunks.append(sub_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    return chunks

def generate_submission(lang, model_name, text_file, ref_dir, out_root, device="cuda", token=None, 
                        ref_duration=10.0, limit_speakers=0, limit_lines=0, scaling_mode="rs"):
    print(f"\n{'='*60}")
    print(f"  🚀 Generating submission for {lang.upper()}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    print(f"  Loading champion model {model_name}...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch
    
    try:
        # 1. Load the Base Model
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", token=token)
        
        # 2. Download LoRA Weights
        print(f"  📥 Downloading weights from {model_name}...")
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", token=token)
        sd = load_file(weights_path)
        
        # 3. Manual Merge Logic
        print("  🧩 Merging LoRA weights manually...")
        merged_sd = {}
        processed_bases = set()
        
        # LoRA parameters
        import math
        alpha, r = 64, 32
        if scaling_mode == "rs":
            scaling = alpha / math.sqrt(r)  # RSLoRA: ~11.31
            print(f"  ⚖️  Using RSLoRA scaling: {scaling:.4f}")
        else:
            scaling = alpha / r             # Standard LoRA: 2.0
            print(f"  ⚖️  Using Standard scaling: {scaling:.4f}")
        
        # First, find all keys and identify base vs lora
        for k in sd.keys():
            if ".base_layer.weight" in k:
                base_key = k.replace("llm.base_model.model.", "llm.")
                clean_key = base_key.replace(".base_layer", "")
                
                # Check for matching LoRA components
                la_key = k.replace(".base_layer.weight", ".lora_A.default.weight")
                lb_key = k.replace(".base_layer.weight", ".lora_B.default.weight")
                
                if la_key in sd and lb_key in sd:
                    # Perform math: Base + (B @ A) * scaling
                    A = sd[la_key].to(torch.float32)
                    B = sd[lb_key].to(torch.float32)
                    base = sd[k].to(torch.float32)
                    
                    merged = base + (B @ A) * scaling
                    merged_sd[clean_key] = merged.to(sd[k].dtype)
                    processed_bases.add(k)
                    processed_bases.add(la_key)
                    processed_bases.add(lb_key)
            
        # Carry over non-lora weights (norm, embed, etc.)
        for k in sd.keys():
            if k not in processed_bases:
                clean_key = k.replace("llm.base_model.model.", "llm.")
                merged_sd[clean_key] = sd[k]
                
        # 4. Load into model
        model.load_state_dict(merged_sd, strict=False)
        print("  ✅ Smart Merge successful.")
        
    except Exception as e:
        print(f"  ❌ Smart Merge failed: {e}")
        if lang in FALLBACK_MODELS and model_name != FALLBACK_MODELS[lang]:
            print(f"  🔄 Retrying with fallback repository: {FALLBACK_MODELS[lang]}")
            return generate_submission(lang, FALLBACK_MODELS[lang], text_file, ref_dir, out_root, device, token)
        return




    model.to(device)
    model.eval()

    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f if line.strip()]
    if limit_lines > 0:
        text_lines = text_lines[:limit_lines]
    
    ref_audios = sorted(list(Path(ref_dir).glob("*.wav")))
    if limit_speakers > 0:
        ref_audios = ref_audios[:limit_speakers]

    pad_width = max(3, len(str(len(text_lines))))
    out_dir = Path(out_root) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for ref_path in tqdm(ref_audios, desc=f"Ref Audios ({lang})", leave=True):
        ref_name = ref_path.stem
        try:
            # OmniVoice recommends 3-10s of clean speech (longer may degrade quality)
            clean_ref_tuple = get_best_reference(ref_path, duration=ref_duration)
            
            # Save the extracted reference so the user can listen and verify
            ref_snippet_path = out_dir / f"_extracted_reference_{ref_name}.wav"
            torchaudio.save(str(ref_snippet_path), clean_ref_tuple[0], clean_ref_tuple[1])
        except: continue
        
        # Nested progress bar for lines
        for idx, text in enumerate(tqdm(text_lines, desc=f"  Lines ({ref_name[:10]}...)", leave=False)):
            line_id = f"{idx + 1:0{pad_width}d}"
            out_path = out_dir / f"{lang}_{line_id}_{ref_name}.wav"
            if out_path.exists(): continue
            
            try:
                chunks = split_text_into_chunks(text)
                audios = []
                for ct in chunks:
                    with torch.no_grad():
                        # Pass the safely normalized 20s reference
                        res = model.generate(text=ct, ref_audio=clean_ref_tuple, temperature=0.8, top_p=0.9)
                        if isinstance(res, tuple): audio_data, sr = res
                        else: audio_data, sr = res, 24000
                        
                        if isinstance(audio_data, (list, tuple)):
                            import numpy as np
                            audio_tensor = torch.from_numpy(np.array(audio_data))
                        elif not isinstance(audio_data, torch.Tensor):
                            audio_tensor = torch.from_numpy(audio_data)
                        else:
                            audio_tensor = audio_data
                            
                        if audio_tensor.ndim == 1: 
                            audio_tensor = audio_tensor.unsqueeze(0)
                        audios.append(audio_tensor.cpu())
                if audios:
                    # Clean concatenation without problematic cross-fades
                    torchaudio.save(str(out_path), torch.cat(audios, dim=-1), sr)
            except Exception as e: print(f" Error {out_path.name}: {e}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-dir", default="./blind_test/text")
    parser.add_argument("--audio-dir", default="./blind_test/audio")
    parser.add_argument("--output-dir", default="./submission_outputs")
    parser.add_argument("--ref-duration", type=float, default=10.0,
                        help="Reference audio duration in seconds (OmniVoice recommends 3-10s)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--limit-speakers", type=int, default=0, help="Limit number of speakers to process")
    parser.add_argument("--limit-lines", type=int, default=0, help="Limit number of lines per speaker to generate")
    parser.add_argument("--scaling-mode", type=str, default="rs", choices=["rs", "standard"], 
                        help="LoRA scaling mode: 'rs' (alpha/sqrt(r)) or 'standard' (alpha/r)")
    args = parser.parse_args()

    langs = ["zh", "ar", "fr"] if args.lang == "all" else [args.lang]
    for l in langs:
        full_name = {'zh': 'chinese', 'ar': 'arabic', 'fr': 'french'}[l]
        t_cands = [Path(args.text_dir)/f"{l}.txt", Path(args.text_dir)/f"{full_name}.txt"]
        r_cands = [Path(args.audio_dir)/l, Path(args.audio_dir)/full_name, Path(args.audio_dir)]
        
        tf = next((c for c in t_cands if c.exists()), None)
        rd = next((c for c in r_cands if c.exists() and any(c.glob("*.wav"))), None)
        
        if not tf:
            print(f"  ❌ Skipped {l}: Text file not found in {args.text_dir} (Checked: {[str(c) for c in t_cands]})")
            continue
        if not rd:
            print(f"  ❌ Skipped {l}: Reference audio directory not found or empty in {args.audio_dir}")
            continue

        print(f"  ✅ Using text: {tf}")
        print(f"  ✅ Using audio: {rd}")
        
        generate_submission(l, BEST_MODELS[l], tf, rd, args.output_dir, args.device, args.token, 
                           ref_duration=args.ref_duration, 
                           limit_speakers=args.limit_speakers,
                           limit_lines=args.limit_lines,
                           scaling_mode=args.scaling_mode)

if __name__ == "__main__":
    main()
