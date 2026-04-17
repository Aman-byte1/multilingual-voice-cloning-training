#!/usr/bin/env python3
"""
IWSLT 2026 Submission Generator - Team: afrinlp
Fixed for direct model loading and robust file discovery.
"""
import os
import gc
import torch
import torchaudio
import argparse
from tqdm import tqdm
from pathlib import Path
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

def trim_reference_audio(ref_path, max_duration=MAX_REF_DURATION):
    waveform, sr = torchaudio.load(str(ref_path))
    max_samples = int(max_duration * sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]
    return waveform, sr

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

def generate_submission(lang, model_name, text_file, ref_dir, out_root, device="cuda", token=None):
    print(f"\n{'='*60}")
    print(f"  🚀 Generating submission for {lang.upper()}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    print(f"  Loading champion model {model_name}...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    try:
        # 1. Load the Base Model
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", token=token)
        
        # 2. Download and Load LoRA Weights manually
        print(f"  📥 Downloading weights from {model_name}...")
        weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", token=token)
        state_dict = load_file(weights_path)
        
        # 3. Clean the state dict prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip standard PEFT prefixes
            new_key = k.replace("llm.base_model.model.", "llm.")
            new_key = new_key.replace(".base_layer", "")
            new_key = new_key.replace(".lora_A.default", "") # Load as merged weights
            new_key = new_key.replace(".lora_B.default", "")
            new_state_dict[new_key] = v
            
        # 4. Load into model
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("  ✅ Smart Load successful.")
        
    except Exception as e:
        print(f"  ❌ Smart Load failed: {e}")
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

    for ref_path in tqdm(ref_audios, desc=f"Ref Audios ({lang})"):
        ref_name = ref_path.stem
        try:
            tw, tsr = trim_reference_audio(ref_path)
            tmp_ref = out_dir / f"_temp_ref_{ref_name}.wav"
            torchaudio.save(str(tmp_ref), tw, tsr)
        except: continue
        
        for idx, text in enumerate(text_lines):
            line_id = f"{idx + 1:0{pad_width}d}"
            out_path = out_dir / f"{lang}_{line_id}_{ref_name}.wav"
            if out_path.exists(): continue
            
            try:
                chunks = split_text_into_chunks(text)
                audios = []
                for ct in chunks:
                    with torch.no_grad():
                        res = model.generate(ct, str(tmp_ref))
                        if isinstance(res, tuple): audio_tensor, sr = res
                        else: audio_tensor, sr = res, 16000 # default
                        if audio_tensor.ndim == 1: audio_tensor = audio_tensor.unsqueeze(0)
                        audios.append(audio_tensor.cpu())
                if audios:
                    torchaudio.save(str(out_path), torch.cat(audios, dim=-1), sr)
            except Exception as e: print(f" Error {out_path.name}: {e}")
        if tmp_ref.exists(): tmp_ref.unlink()
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
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    langs = ["zh", "ar", "fr"] if args.lang == "all" else [args.lang]
    for l in langs:
        full_name = {'zh': 'chinese', 'ar': 'arabic', 'fr': 'french'}[l]
        t_cands = [Path(args.text_dir)/f"{l}.txt", Path(args.text_dir)/f"{full_name}.txt"]
        r_cands = [Path(args.audio_dir)/l, Path(args.audio_dir)/full_name, Path(args.audio_dir)]
        
        tf = next((c for c in t_cands if c.exists()), None)
        rd = next((c for c in r_cands if c.exists() and any(c.glob("*.wav"))), None)
        if tf and rd: generate_submission(l, BEST_MODELS[l], tf, rd, args.output_dir, args.device, args.token)

if __name__ == "__main__":
    main()
