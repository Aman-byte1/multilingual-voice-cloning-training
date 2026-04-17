#!/usr/bin/env python3
import os
import torch
import torchaudio
import argparse
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel
from omnivoice import OmniVoice

# Final Winning Checkpoints
BEST_MODELS = {
    "zh": "amanuelbyte/omnivoice-lora-zh-400",
    "ar": "amanuelbyte/omnivoice-lora-ar-400",
    "fr": "amanuelbyte/omnivoice-lora-fr-200"
}

def generate_submission(lang, model_name, text_file, ref_dir, out_root, device="cuda"):
    print(f"\n🚀 Generating submission for {lang.upper()} using {model_name}")
    
    # 1. Load Model
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    model.llm = PeftModel.from_pretrained(model.llm, model_name)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f if line.strip()]
    
    ref_audios = sorted(list(Path(ref_dir).glob("*.wav")))
    
    out_dir = Path(out_root) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Generate
    # Rule: For each reference audio, include all source lines
    for ref_path in tqdm(ref_audios, desc=f"Ref Audios ({lang})"):
        ref_name = ref_path.stem
        
        for idx, text in enumerate(text_lines):
            line_id = f"{idx + 1:03d}" # zero-padded 1-based
            out_filename = f"{lang}_{line_id}_{ref_name}.wav"
            out_path = out_dir / out_filename
            
            if out_path.exists():
                continue
                
            try:
                # Generate audio
                # OmniVoice prompt: (text, ref_audio_path)
                with torch.no_grad():
                    output_audio = model.generate(text, ref_path)
                    
                # Save
                # OmniVoice generate returns tensor + sample_rate
                audio_tensor, sr = output_audio
                if torch.is_tensor(audio_tensor):
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    torchaudio.save(str(out_path), audio_tensor.cpu(), sr)
            except Exception as e:
                print(f"  ❌ Error generating {out_filename}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["ar", "zh", "fr", "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-dir", default="./blind_test/text")
    parser.add_argument("--audio-dir", default="./blind_test/audio")
    parser.add_argument("--output-dir", default="./submission_outputs")
    args = parser.parse_args()

    langs_to_run = ["zh", "ar", "fr"] if args.lang == "all" else [args.lang]

    for lang in langs_to_run:
        text_file = Path(args.text_dir) / f"{lang}.txt" # Assuming text file naming
        ref_dir = Path(args.audio_dir) / lang
        
        if not text_file.exists():
            print(f"⚠ Text file {text_file} not found. Skipping {lang}.")
            continue
        if not ref_dir.exists():
            print(f"⚠ Reference dir {ref_dir} not found. Skipping {lang}.")
            continue
            
        generate_submission(lang, BEST_MODELS[lang], text_file, ref_dir, args.output_dir, args.device)

if __name__ == "__main__":
    main()
