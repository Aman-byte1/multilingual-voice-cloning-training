#!/usr/bin/env python3
"""
🚀 IWSLT 2026 — Qwen3-TTS Cross-Lingual Voice Cloning Generator
=================================================================
Generates submissions for all 3 target languages (Arabic, Chinese, French)
using Qwen3-TTS-12Hz-1.7B zero-shot voice cloning.

Usage:
    # All languages
    python generate_qwen.py --ref-dir ./blind_test/ref_audio \
                            --text-dir ./blind_test \
                            --output-dir ./outputs/qwen

    # Single language
    python generate_qwen.py --languages fr \
                            --ref-dir ./blind_test/ref_audio \
                            --text-dir ./blind_test \
                            --output-dir ./outputs/qwen
"""

import os
import sys
import gc
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("QwenGenerator")

# ─── LANGUAGE CONFIG ───────────────────────────────────────────
LANG_CONFIG = {
    "ar": {
        "name": "Arabic",
        "text_file": "arabic.txt",
        "qwen_lang": "Arabic",
        "whisper_lang": "ar",
    },
    "zh": {
        "name": "Chinese",
        "text_file": "chinese.txt",
        "qwen_lang": "Chinese",
        "whisper_lang": "zh",
    },
    "fr": {
        "name": "French",
        "text_file": "french.txt",
        "qwen_lang": "French",
        "whisper_lang": "fr",
    },
}


def load_target_texts(text_path: str) -> List[str]:
    """Load target text file, one line per utterance."""
    with open(text_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # Filter empty lines but preserve line numbering
    return lines


def get_ref_audio_files(ref_dir: str) -> List[str]:
    """Get sorted list of reference audio files."""
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = sorted([
        f for f in os.listdir(ref_dir)
        if Path(f).suffix.lower() in exts
    ])
    return files


def load_qwen_model(model_name: str, device: str = "cuda"):
    """Load Qwen3-TTS model."""
    log.info(f"Loading Qwen3-TTS: {model_name}")
    try:
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        log.info("Qwen3-TTS loaded ✓")
        return model
    except ImportError:
        log.warning("qwen_tts package not found. Trying transformers-based loading...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        log.info("Qwen3-TTS loaded via transformers ✓")
        return model


def generate_for_language(
    model,
    lang_code: str,
    text_path: str,
    ref_dir: str,
    output_dir: str,
    ref_text_path: Optional[str] = None,
):
    """Generate all utterances for a single language."""
    cfg = LANG_CONFIG[lang_code]
    log.info(f"\n{'='*60}")
    log.info(f"  Generating {cfg['name']} ({lang_code})")
    log.info(f"{'='*60}")

    # Load texts
    target_lines = load_target_texts(text_path)
    log.info(f"  Target texts: {len(target_lines)} lines")

    # Load reference audio list
    ref_files = get_ref_audio_files(ref_dir)
    log.info(f"  Reference audios: {len(ref_files)} files")

    # Load reference transcripts if available (for Qwen ref_text param)
    ref_texts = {}
    if ref_text_path and os.path.exists(ref_text_path):
        with open(ref_text_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    ref_texts[parts[0]] = parts[1]

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    skipped = 0
    errors = []

    for line_idx, text in enumerate(tqdm(target_lines, desc=f"Gen {lang_code}")):
        # Line number is 1-based, zero-padded to 3 digits
        line_num = f"{line_idx + 1:03d}"

        if not text.strip():
            skipped += 1
            continue

        # Determine which reference audio to use
        # IWSLT format: each line maps to a reference audio file
        # The reference audio files should be provided per-line or as a pool
        if line_idx < len(ref_files):
            ref_file = ref_files[line_idx]
        else:
            # If fewer ref files than lines, cycle through them
            ref_file = ref_files[line_idx % len(ref_files)]

        ref_path = os.path.join(ref_dir, ref_file)
        ref_stem = Path(ref_file).stem  # filename without extension

        # Output filename: {lang}_{lineNumber}_{originalName}.wav
        out_name = f"{lang_code}_{line_num}_{ref_stem}.wav"
        out_path = os.path.join(output_dir, out_name)

        # Skip if already generated
        if os.path.exists(out_path):
            generated += 1
            continue

        try:
            # Get reference text if available
            ref_text = ref_texts.get(ref_stem, "")

            # Generate with Qwen3-TTS
            if hasattr(model, 'generate_voice_clone'):
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=cfg["qwen_lang"],
                    ref_audio=ref_path,
                    ref_text=ref_text if ref_text else None,
                )
                sf.write(out_path, wavs[0], sr)
            else:
                # Fallback: use model.generate or tts method
                wavs = model.generate(
                    text=text,
                    ref_audio_path=ref_path,
                    language=cfg["qwen_lang"],
                )
                if isinstance(wavs, torch.Tensor):
                    wav_np = wavs.cpu().numpy()
                    if wav_np.ndim > 1:
                        wav_np = wav_np.squeeze()
                    sf.write(out_path, wav_np, 24000)
                else:
                    sf.write(out_path, wavs[0], 24000)

            generated += 1

        except Exception as e:
            skipped += 1
            errors.append({"line": line_idx, "error": str(e)})
            tqdm.write(f"  ⚠ Line {line_num} failed: {e}")

    log.info(f"  {cfg['name']}: {generated} generated, {skipped} skipped")
    if errors:
        err_path = os.path.join(output_dir, f"errors_{lang_code}.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        log.info(f"  Errors saved to {err_path}")

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="IWSLT 2026 — Qwen3-TTS Voice Cloning Generator"
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model name or path",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["ar", "zh", "fr"],
        choices=["ar", "zh", "fr"],
        help="Target languages to generate",
    )
    parser.add_argument(
        "--ref-dir",
        required=True,
        help="Directory containing reference audio files",
    )
    parser.add_argument(
        "--text-dir",
        required=True,
        help="Directory containing target text files (arabic.txt, chinese.txt, french.txt)",
    )
    parser.add_argument(
        "--ref-text-path",
        default=None,
        help="Optional: file with reference audio transcripts (format: filename|text)",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/qwen",
        help="Output directory for generated WAVs",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use",
    )
    args = parser.parse_args()

    # Load model once
    model = load_qwen_model(args.model_name, args.device)

    total_gen = 0
    total_skip = 0

    for lang in args.languages:
        cfg = LANG_CONFIG[lang]
        text_path = os.path.join(args.text_dir, cfg["text_file"])

        if not os.path.exists(text_path):
            log.warning(f"Text file not found: {text_path}, skipping {lang}")
            continue

        lang_output = os.path.join(args.output_dir, lang)
        gen, skip = generate_for_language(
            model=model,
            lang_code=lang,
            text_path=text_path,
            ref_dir=args.ref_dir,
            output_dir=lang_output,
            ref_text_path=args.ref_text_path,
        )
        total_gen += gen
        total_skip += skip

        # Free some GPU memory between languages
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()

    log.info(f"\n{'='*60}")
    log.info(f"  QWEN3-TTS GENERATION COMPLETE")
    log.info(f"  Generated: {total_gen} | Skipped: {total_skip}")
    log.info(f"  Output: {args.output_dir}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
