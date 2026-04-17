#!/usr/bin/env python3
"""
IWSLT 2026 Submission Generator - Team: afrinlp
Handles long reference audio and long target text gracefully.
"""
import os
import gc
import torch
import torchaudio
import argparse
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel

# Final Winning Checkpoints
BEST_MODELS = {
    "zh": "amanuelbyte/omnivoice-lora-zh-400",
    "ar": "amanuelbyte/omnivoice-lora-ar-400",
    "fr": "amanuelbyte/omnivoice-lora-fr-200",
}

# Maximum reference audio duration in seconds
MAX_REF_DURATION = 15.0

# Maximum characters per chunk for long text generation
MAX_CHARS_PER_CHUNK = 200


def trim_reference_audio(ref_path, max_duration=MAX_REF_DURATION):
    """Load reference audio and trim to max_duration seconds."""
    waveform, sr = torchaudio.load(str(ref_path))
    max_samples = int(max_duration * sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]
    return waveform, sr


def split_text_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK):
    """Split long text into sentence-level chunks.
    
    Tries to split on sentence boundaries (. ! ? 。) first,
    then falls back to splitting on commas or spaces.
    """
    if len(text) <= max_chars:
        return [text]

    import re
    # Try sentence-level splits first
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single sentence is too long, split on commas/spaces
            if len(sentence) > max_chars:
                words = re.split(r'[,，、\s]+', sentence)
                sub_chunk = ""
                for word in words:
                    if len(sub_chunk) + len(word) + 1 <= max_chars:
                        sub_chunk = (sub_chunk + " " + word).strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = word
                if sub_chunk:
                    chunks.append(sub_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def generate_submission(lang, model_name, text_file, ref_dir, out_root, device="cuda"):
    print(f"\n{'='*60}")
    print(f"  🚀 Generating submission for {lang.upper()}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    # 1. Load Model
    from omnivoice import OmniVoice
    print("  Loading base OmniVoice...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    print(f"  Merging LoRA adapter: {model_name}")
    model.llm = PeftModel.from_pretrained(model.llm, model_name)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f if line.strip()]
    
    num_lines = len(text_lines)
    ref_audios = sorted(list(Path(ref_dir).glob("*.wav")))
    num_refs = len(ref_audios)
    total = num_lines * num_refs
    
    print(f"  📝 Text lines: {num_lines}")
    print(f"  🎤 Reference audios: {num_refs}")
    print(f"  📊 Total WAVs to generate: {total}")
    
    # Determine zero-padding width
    pad_width = max(3, len(str(num_lines)))

    out_dir = Path(out_root) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files to support resumption
    existing = len(list(out_dir.glob("*.wav")))
    if existing > 0:
        print(f"  ♻️  Found {existing} existing files (will skip)")

    # 3. Generate
    generated = 0
    skipped = 0
    errors = 0
    
    pbar = tqdm(total=total, desc=f"Generating {lang}")
    
    for ref_path in ref_audios:
        ref_name = ref_path.stem
        
        # Pre-load and trim reference audio once per ref
        try:
            trimmed_ref_wav, trimmed_sr = trim_reference_audio(ref_path)
            # Save trimmed version to a temp file for OmniVoice
            trimmed_ref_path = out_dir / f"_temp_ref_{ref_name}.wav"
            torchaudio.save(str(trimmed_ref_path), trimmed_ref_wav, trimmed_sr)
        except Exception as e:
            print(f"\n  ❌ Could not load ref audio {ref_name}: {e}")
            pbar.update(num_lines)
            errors += num_lines
            continue
        
        for idx, text in enumerate(text_lines):
            line_id = f"{idx + 1:0{pad_width}d}"  # zero-padded 1-based
            out_filename = f"{lang}_{line_id}_{ref_name}.wav"
            out_path = out_dir / out_filename
            
            pbar.update(1)
            
            if out_path.exists():
                skipped += 1
                continue
            
            try:
                # Split long text into chunks
                chunks = split_text_into_chunks(text)
                
                all_audio_chunks = []
                for chunk_text in chunks:
                    with torch.no_grad():
                        output_audio = model.generate(chunk_text, str(trimmed_ref_path))
                    
                    audio_tensor, sr = output_audio
                    if torch.is_tensor(audio_tensor):
                        if audio_tensor.ndim == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        all_audio_chunks.append(audio_tensor.cpu())
                
                # Concatenate chunks
                if all_audio_chunks:
                    final_audio = torch.cat(all_audio_chunks, dim=-1)
                    torchaudio.save(str(out_path), final_audio, sr)
                    generated += 1
                    
            except Exception as e:
                print(f"\n  ❌ Error: {out_filename}: {e}")
                errors += 1
            
            # Periodically clear CUDA cache
            if (generated + skipped) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Clean up temp ref file
        if trimmed_ref_path.exists():
            trimmed_ref_path.unlink()
    
    pbar.close()
    
    print(f"\n  ✅ {lang.upper()} DONE: Generated={generated}, Skipped={skipped}, Errors={errors}")
    
    # Free model memory before next language
    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="IWSLT 2026 Submission Generator")
    parser.add_argument("--lang", choices=["ar", "zh", "fr", "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-dir", default="./blind_test/text")
    parser.add_argument("--audio-dir", default="./blind_test/audio")
    parser.add_argument("--output-dir", default="./submission_outputs")
    args = parser.parse_args()

    langs_to_run = ["zh", "ar", "fr"] if args.lang == "all" else [args.lang]

    for lang in langs_to_run:
        # Try common naming patterns for text and ref dirs
        text_candidates = [
            Path(args.text_dir) / f"{lang}.txt",
            Path(args.text_dir) / f"{lang}",
            Path(args.text_dir) / f"{'arabic' if lang == 'ar' else 'chinese' if lang == 'zh' else 'french'}.txt",
        ]
        ref_candidates = [
            Path(args.audio_dir) / lang,
            Path(args.audio_dir),
        ]
        
        text_file = None
        for candidate in text_candidates:
            if candidate.exists():
                text_file = candidate
                break
        
        ref_dir = None
        for candidate in ref_candidates:
            if candidate.exists() and any(candidate.glob("*.wav")):
                ref_dir = candidate
                break
        
        if text_file is None:
            print(f"⚠ Text file not found for {lang}. Tried: {[str(c) for c in text_candidates]}")
            print(f"  Please check the contents of {args.text_dir}")
            continue
        if ref_dir is None:
            print(f"⚠ Reference audio dir not found for {lang}. Tried: {[str(c) for c in ref_candidates]}")
            print(f"  Please check the contents of {args.audio_dir}")
            continue
        
        print(f"\n  📁 Text file: {text_file}")
        print(f"  📁 Ref dir:   {ref_dir}")
        
        generate_submission(lang, BEST_MODELS[lang], text_file, ref_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
