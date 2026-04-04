#!/usr/bin/env python3
"""
📦 IWSLT 2026 — Submission Packager & Validator
================================================
Packages generated WAV files into correctly-named zip archives
and validates against the official naming format.

Submission format:
    {team}_{language}.zip
    ├── {lang}_{lineNumber}_{originalName}.wav
    ├── {lang}_{lineNumber}_{originalName}.wav
    └── ...

Usage:
    python package_submission.py --team tcd \
                                  --model-dir ./outputs/qwen \
                                  --languages fr ar zh \
                                  --source-dir ./blind_test \
                                  --ref-dir ./blind_test/ref_audio \
                                  --output-dir ./final
"""

import os
import sys
import re
import argparse
import zipfile
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Packager")

LANG_CONFIG = {
    "ar": {"text_file": "arabic.txt", "name": "Arabic"},
    "zh": {"text_file": "chinese.txt", "name": "Chinese"},
    "fr": {"text_file": "french.txt", "name": "French"},
}


def validate_wav_naming(wav_files: List[str], language: str,
                        source_lines: int, ref_stems: List[str]) -> Dict:
    """
    Validate WAV file naming against IWSLT format.
    Expected: {language}_{lineNumber}_{originalName}.wav
    """
    errors = []
    warnings = []
    valid = 0

    pattern = re.compile(rf"^{language}_(\d+)_(.+)\.wav$")

    expected_files = set()
    for i in range(1, source_lines + 1):
        line_num = f"{i:03d}"
        for stem in ref_stems:
            expected_files.add(f"{language}_{line_num}_{stem}.wav")

    for wav in wav_files:
        fname = os.path.basename(wav)
        match = pattern.match(fname)

        if not match:
            errors.append(f"Invalid filename format: {fname}")
            continue

        line_num = int(match.group(1))
        orig_name = match.group(2)

        if line_num < 1 or line_num > source_lines:
            errors.append(f"Line number out of range: {fname} (max: {source_lines})")
        elif orig_name not in ref_stems:
            warnings.append(f"Unknown ref name: {orig_name} in {fname}")
        else:
            valid += 1

    # Check for missing files
    found_files = set(os.path.basename(w) for w in wav_files)
    missing = expected_files - found_files
    if missing and len(missing) < 20:
        for m in sorted(missing):
            warnings.append(f"Missing expected file: {m}")
    elif missing:
        warnings.append(f"{len(missing)} expected files missing")

    return {
        "valid": valid,
        "total": len(wav_files),
        "errors": errors,
        "warnings": warnings,
        "pass": len(errors) == 0,
    }


def package_language(
    team: str,
    language: str,
    wav_dir: str,
    output_dir: str,
    source_text_path: str = None,
    ref_dir: str = None,
) -> str:
    """Package WAVs for one language into a zip file."""
    log.info(f"\n{'─'*50}")
    log.info(f"  Packaging: {LANG_CONFIG[language]['name']} ({language})")
    log.info(f"  WAV dir: {wav_dir}")

    # Collect WAV files
    wav_files = sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ])
    log.info(f"  Found {len(wav_files)} WAV files")

    if not wav_files:
        log.error(f"  No WAV files found in {wav_dir}")
        return None

    # Validate if source text and ref dir available
    if source_text_path and os.path.exists(source_text_path):
        with open(source_text_path, "r", encoding="utf-8") as f:
            source_lines = len([l for l in f.readlines() if l.strip()])

        ref_stems = []
        if ref_dir and os.path.exists(ref_dir):
            ref_stems = [Path(f).stem for f in sorted(os.listdir(ref_dir))
                         if Path(f).suffix.lower() in {".wav", ".mp3", ".flac"}]

        log.info(f"  Source text lines: {source_lines}")
        log.info(f"  Reference stems: {len(ref_stems)}")

        result = validate_wav_naming(wav_files, language, source_lines, ref_stems)

        log.info(f"  Validation: {result['valid']}/{result['total']} valid")
        if result["errors"]:
            log.error(f"  ❌ {len(result['errors'])} ERRORS:")
            for e in result["errors"][:10]:
                log.error(f"     {e}")
        if result["warnings"]:
            log.warning(f"  ⚠ {len(result['warnings'])} warnings")
            for w in result["warnings"][:5]:
                log.warning(f"     {w}")

        if not result["pass"]:
            log.error(f"  Validation FAILED — fix errors before submitting!")

    # Create zip
    os.makedirs(output_dir, exist_ok=True)
    zip_name = f"{team}_{language}.zip"
    zip_path = os.path.join(output_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav_path in tqdm(wav_files, desc=f"Zipping {language}"):
            zf.write(wav_path, os.path.basename(wav_path))

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    log.info(f"  ✅ Created: {zip_path} ({zip_size_mb:.1f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="IWSLT 2026 — Submission Packager"
    )
    parser.add_argument("--team", required=True, help="Team name for zip filename")
    parser.add_argument("--model-dir", required=True,
                        help="Dir with per-language subdirs (e.g., outputs/qwen/fr/)")
    parser.add_argument("--languages", nargs="+", default=["ar", "zh", "fr"],
                        choices=["ar", "zh", "fr"])
    parser.add_argument("--source-dir", default=None,
                        help="Dir with source text files (arabic.txt, etc.)")
    parser.add_argument("--ref-dir", default=None,
                        help="Dir with reference audio files")
    parser.add_argument("--output-dir", default="./final")
    args = parser.parse_args()

    log.info(f"Team: {args.team}")
    log.info(f"Languages: {', '.join(args.languages)}")

    zip_paths = []
    for lang in args.languages:
        wav_dir = os.path.join(args.model_dir, lang)
        if not os.path.exists(wav_dir):
            log.warning(f"WAV dir not found: {wav_dir}, skipping {lang}")
            continue

        source_text = None
        if args.source_dir:
            source_text = os.path.join(args.source_dir, LANG_CONFIG[lang]["text_file"])

        zip_path = package_language(
            team=args.team,
            language=lang,
            wav_dir=wav_dir,
            output_dir=args.output_dir,
            source_text_path=source_text,
            ref_dir=args.ref_dir,
        )
        if zip_path:
            zip_paths.append(zip_path)

    log.info(f"\n{'='*60}")
    log.info(f"  PACKAGING COMPLETE")
    log.info(f"  Created {len(zip_paths)} submission file(s):")
    for zp in zip_paths:
        log.info(f"    📦 {zp}")
    log.info(f"{'='*60}")
    log.info(f"\n  Next: Run official verify_submission_naming.py on each zip!")


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

if __name__ == "__main__":
    main()
