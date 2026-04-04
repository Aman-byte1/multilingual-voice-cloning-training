#!/usr/bin/env python3
"""
🚀 IWSLT 2026 — CosyVoice3 Cross-Lingual Voice Cloning Generator
==================================================================
Generates submissions using CosyVoice3 (Alibaba FunAudioLLM).
Best-in-class cross-lingual speaker identity preservation.

Setup:
    git clone https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice && pip install -e .

Usage:
    python generate_cosyvoice.py --ref-dir ./blind_test/ref_audio \
                                  --text-dir ./blind_test \
                                  --output-dir ./outputs/cosyvoice
"""

import os
import sys
import gc
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

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
log = logging.getLogger("CosyVoiceGen")

LANG_CONFIG = {
    "ar": {"name": "Arabic",  "text_file": "arabic.txt",  "cosyvoice_lang": "ar"},
    "zh": {"name": "Chinese", "text_file": "chinese.txt", "cosyvoice_lang": "zh"},
    "fr": {"name": "French",  "text_file": "french.txt",  "cosyvoice_lang": "fr"},
}


def load_target_texts(text_path: str) -> List[str]:
    with open(text_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_ref_audio_files(ref_dir: str) -> List[str]:
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    return sorted([f for f in os.listdir(ref_dir) if Path(f).suffix.lower() in exts])


def load_cosyvoice_model(model_name: str = "CosyVoice3-0.5B"):
    """Load CosyVoice3 model."""
    log.info(f"Loading CosyVoice3: {model_name}")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice
        # Try loading from pretrained
        model = CosyVoice(model_name)
        log.info("CosyVoice3 loaded ✓")
        return model
    except ImportError:
        log.error(
            "CosyVoice not installed. Install from:\n"
            "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
            "  cd CosyVoice && pip install -e ."
        )
        sys.exit(1)


def generate_for_language(
    model,
    lang_code: str,
    text_path: str,
    ref_dir: str,
    output_dir: str,
):
    cfg = LANG_CONFIG[lang_code]
    log.info(f"\n{'='*60}")
    log.info(f"  Generating {cfg['name']} ({lang_code}) with CosyVoice3")
    log.info(f"{'='*60}")

    target_lines = load_target_texts(text_path)
    ref_files = get_ref_audio_files(ref_dir)
    log.info(f"  Target texts: {len(target_lines)} | Ref audios: {len(ref_files)}")

    os.makedirs(output_dir, exist_ok=True)
    generated, skipped = 0, 0
    errors = []

    for line_idx, text in enumerate(tqdm(target_lines, desc=f"Gen {lang_code}")):
        line_num = f"{line_idx + 1:03d}"
        if not text.strip():
            skipped += 1
            continue

        ref_file = ref_files[line_idx] if line_idx < len(ref_files) else ref_files[line_idx % len(ref_files)]
        ref_path = os.path.join(ref_dir, ref_file)
        ref_stem = Path(ref_file).stem

        out_name = f"{lang_code}_{line_num}_{ref_stem}.wav"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            generated += 1
            continue

        try:
            # CosyVoice3 cross-lingual synthesis
            # The API may vary — adapt based on actual CosyVoice3 version
            if hasattr(model, 'inference_cross_lingual'):
                # Standard CosyVoice3 API
                output_speech = model.inference_cross_lingual(
                    tts_text=text,
                    prompt_speech_16k=ref_path,
                )
                # CosyVoice returns a generator of dict with 'tts_speech' key
                for result in output_speech:
                    wav = result['tts_speech']
                    if isinstance(wav, torch.Tensor):
                        wav = wav.cpu().numpy()
                    if wav.ndim > 1:
                        wav = wav.squeeze()
                    sf.write(out_path, wav, 22050)
                    break  # Take only first result
            elif hasattr(model, 'inference_zero_shot'):
                # Alternative API
                output_speech = model.inference_zero_shot(
                    tts_text=text,
                    prompt_text="",  # No prompt text in cross-lingual mode
                    prompt_speech_16k=ref_path,
                )
                for result in output_speech:
                    wav = result['tts_speech']
                    if isinstance(wav, torch.Tensor):
                        wav = wav.cpu().numpy()
                    if wav.ndim > 1:
                        wav = wav.squeeze()
                    sf.write(out_path, wav, 22050)
                    break
            else:
                raise AttributeError("No compatible CosyVoice inference method found")

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

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="IWSLT 2026 — CosyVoice3 Voice Cloning Generator"
    )
    parser.add_argument("--model-name", default="CosyVoice3-0.5B")
    parser.add_argument("--languages", nargs="+", default=["ar", "zh", "fr"],
                        choices=["ar", "zh", "fr"])
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--output-dir", default="./outputs/cosyvoice")
    args = parser.parse_args()

    model = load_cosyvoice_model(args.model_name)

    total_gen, total_skip = 0, 0
    for lang in args.languages:
        cfg = LANG_CONFIG[lang]
        text_path = os.path.join(args.text_dir, cfg["text_file"])
        if not os.path.exists(text_path):
            log.warning(f"Text file not found: {text_path}, skipping {lang}")
            continue

        gen, skip = generate_for_language(
            model=model,
            lang_code=lang,
            text_path=text_path,
            ref_dir=args.ref_dir,
            output_dir=os.path.join(args.output_dir, lang),
        )
        total_gen += gen
        total_skip += skip
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()

    log.info(f"\n{'='*60}")
    log.info(f"  COSYVOICE3 GENERATION COMPLETE")
    log.info(f"  Generated: {total_gen} | Skipped: {total_skip}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
