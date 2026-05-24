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

def _install_flex_stub():
    mod_name = "torch.nn.attention.flex_attention"
    import sys
    import types
    import torch
    
    # 1. Try to import the existing module or create a stub if it doesn't exist
    if mod_name in sys.modules and sys.modules[mod_name] is not None:
        flex_mod = sys.modules[mod_name]
    else:
        try:
            import torch.nn.attention.flex_attention as flex_mod
        except ImportError:
            flex_mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = flex_mod
            try:
                import torch.nn.attention as attn_mod
                setattr(attn_mod, "flex_attention", flex_mod)
            except Exception:
                pass

    # 2. Guarantee that critical objects needed by transformers/omnivoice exist
    if not hasattr(flex_mod, "_DEFAULT_SPARSE_BLOCK_SIZE"):
        setattr(flex_mod, "_DEFAULT_SPARSE_BLOCK_SIZE", 128)
        
    if not hasattr(flex_mod, "create_block_mask"):
        def create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None,
                              _compile=False, device=None, **kw):
            seq_len = int(Q_LEN or KV_LEN or 1)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float32)
            mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(mask.dtype).min)
            return mask
        setattr(flex_mod, "create_block_mask", create_block_mask)

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
    "zh": "amanuelbyte/omnivoice_finetuned_zh",
    "ar": "amanuelbyte/omnivoice_finetuned_ar",
    "fr": "amanuelbyte/omnivoice_finetuned_fr",
}

# Fallback names if -400 doesn't exist for all
FALLBACK_MODELS = {
    "zh": "amanuelbyte/omnivoice-lora-zh-400",
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

def generate_submission(lang, model_name, text_file, ref_dir, out_root, device="cuda", token=None, ref_duration=10.0):
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
        
        # 2. Download LoRA/Merged Weights with robust fallback
        print(f"  📥 Downloading weights from {model_name}...")
        sd = None
        for filename in ["model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin"]:
            try:
                print(f"    Trying to download {filename}...")
                weights_path = hf_hub_download(repo_id=model_name, filename=filename, token=token)
                if filename.endswith(".safetensors"):
                    sd = load_file(weights_path)
                else:
                    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
                print(f"    ✅ Successfully downloaded and loaded {filename}")
                break
            except Exception as e:
                print(f"    ✕ {filename} not found or failed to load: {e}")
        
        if sd is None:
            raise FileNotFoundError(f"Could not find any valid weight file (model.safetensors, pytorch_model.bin, adapter_model.safetensors, adapter_model.bin) in {model_name}")

        # 3. Manual Merge Logic
        print("  🧩 Merging/Loading weights...")
        merged_sd = {}
        processed_bases = set()
        
        # LoRA parameters
        scaling = 64 / 32 # alpha / r
        
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
            
        # Carry over non-lora weights (norm, embed, etc. or already merged weights)
        for k in sd.keys():
            if k not in processed_bases:
                clean_key = k.replace("llm.base_model.model.", "llm.")
                merged_sd[clean_key] = sd[k]
                
        # 4. Load into model
        model.load_state_dict(merged_sd, strict=False)
        print("  ✅ Weight loading / Smart Merge successful.")
        
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
    
    ref_audios = sorted(list(Path(ref_dir).glob("*.wav")))
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
    parser.add_argument("--output-dir", default="./temp_submission")
    parser.add_argument("--ref-duration", type=float, default=10.0,
                        help="Reference audio duration in seconds (OmniVoice recommends 3-10s)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--parallel", action="store_true",
                        help="Run generation in parallel across available GPUs (if --lang all)")
    parser.add_argument("--gpus", default="0,1,2",
                        help="Comma-separated list of GPU IDs to distribute processes over (default: 0,1,2)")
    args = parser.parse_args()

    langs = ["zh", "ar", "fr"] if args.lang == "all" else [args.lang]

    if args.lang == "all" and args.parallel and len(langs) > 1:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpu_list:
            gpu_list = ["0"]
        
        import subprocess
        processes = []
        log_files = []
        
        print(f"🚀 Starting parallel generation for {langs} on GPUs {gpu_list}...")
        
        for idx, lang in enumerate(langs):
            gpu_id = gpu_list[idx % len(gpu_list)]
            
            # Restrict visibility inside the child process using CUDA_VISIBLE_DEVICES
            child_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--lang", lang,
                "--text-dir", args.text_dir,
                "--audio-dir", args.audio_dir,
                "--output-dir", args.output_dir,
                "--ref-duration", str(args.ref_duration),
                "--device", "cuda"
            ]
            if args.token:
                child_cmd.extend(["--token", args.token])
            
            log_path = f"logs/generate_{lang}.log"
            os.makedirs("logs", exist_ok=True)
            log_file = open(log_path, "w", encoding="utf-8")
            log_files.append((lang, log_path, log_file))
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            print(f"  ➜ [{lang.upper()}] starting on GPU {gpu_id} (log: {log_path})")
            p = subprocess.Popen(child_cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((lang, p))
            
        print("\n⏳ Waiting for generation to complete...")
        for lang, p in processes:
            p.wait()
            
        for _, _, f in log_files:
            f.close()
            
        print("\n✅ Parallel generation processes finished!")
    else:
        for l in langs:
            full_name = {'zh': 'chinese', 'ar': 'arabic', 'fr': 'french'}[l]
            t_cands = [Path(args.text_dir)/f"{l}.txt", Path(args.text_dir)/f"{full_name}.txt"]
            r_cands = [Path(args.audio_dir)/l, Path(args.audio_dir)/full_name, Path(args.audio_dir)]
            
            tf = next((c for c in t_cands if c.exists()), None)
            rd = next((c for c in r_cands if c.exists() and any(c.glob("*.wav"))), None)
            if tf and rd: generate_submission(l, BEST_MODELS[l], tf, rd, args.output_dir, args.device, args.token, ref_duration=args.ref_duration)

if __name__ == "__main__":
    main()
