#!/usr/bin/env python3
"""Validate IWSLT voice-cloning submission WAV filename conventions."""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FILENAME_REGEX = re.compile(
    r"^(?P<lang>[A-Za-z]{2,3})_(?P<line>\d+)_(?P<original>.+)\.wav$"
)

@dataclass
class ValidationIssue:
    path: Path
    message: str

@dataclass
class ParsedFile:
    path: Path
    language: str
    line_id: int
    line_str: str
    original_name: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission filenames for IWSLT voice cloning.")
    parser.add_argument("submission_dir", nargs="?", default="output", help="Directory containing submission WAV files.")
    parser.add_argument("--language", type=str, required=True, help="Required language code (ar, fr, zh).")
    parser.add_argument("--line-width", type=int, default=3, help="Expected digit width for line number.")
    parser.add_argument("--source-file", type=str, required=True, help="Source text file.")
    parser.add_argument("--reference-dir", type=str, required=True, help="Reference WAV files directory.")
    parser.add_argument("--strict-width", action="store_true", help="Require exact line-width.")
    return parser.parse_args()

def count_nonempty_lines(source_file: Path) -> int:
    with source_file.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def collect_audio_files(submission_dir: Path) -> List[Path]:
    return sorted([p for p in submission_dir.rglob("*") if p.is_file()])

def collect_reference_names(reference_dir: Path) -> List[str]:
    reference_wavs = sorted([p for p in reference_dir.rglob("*.wav") if p.is_file()])
    return [p.stem for p in reference_wavs]

def validate_filename(path: Path, line_width: int, strict_width: bool) -> Tuple[Optional[ParsedFile], List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    if path.suffix.lower() != ".wav":
        issues.append(ValidationIssue(path=path, message="File extension is not .wav"))
        return None, issues
    match = FILENAME_REGEX.match(path.name)
    if not match:
        issues.append(ValidationIssue(path=path, message="Filename mismatch"))
        return None, issues
    language = match.group("lang").lower()
    line_str = match.group("line")
    original_name = match.group("original")
    if strict_width and len(line_str) != line_width:
        issues.append(ValidationIssue(path=path, message=f"Line width mismatch: {len(line_str)}"))
    try:
        line_id = int(line_str)
    except ValueError:
        return None, [ValidationIssue(path=path, message="Invalid line ID")]
    parsed = ParsedFile(path=path, language=language, line_id=line_id, line_str=line_str, original_name=original_name)
    return parsed, issues

def validate_per_reference_coverage(parsed_files: List[ParsedFile], expected_lines: int, language: str, reference_names: List[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    by_original: Dict[str, List[ParsedFile]] = {}
    for item in parsed_files:
        by_original.setdefault(item.original_name, []).append(item)
    for original_name in sorted(reference_names):
        items = by_original.get(original_name, [])
        if not items:
            issues.append(ValidationIssue(path=Path("."), message=f"Reference '{original_name}' missing all lines"))
            continue
        line_ids = [x.line_id for x in items]
        unique_line_ids = set(line_ids)
        missing = [i for i in range(1, expected_lines + 1) if i not in unique_line_ids]
        for miss in missing[:5]:
            issues.append(ValidationIssue(path=items[0].path, message=f"Reference '{original_name}' missing line {miss}"))
    return issues

def main() -> int:
    args = parse_args()
    submission_dir = Path(args.submission_dir)
    audio_files = collect_audio_files(submission_dir)
    if not audio_files: return 2
    parsed_files = []; issues = []
    for path in audio_files:
        parsed, file_issues = validate_filename(path, args.line_width, args.strict_width)
        issues.extend(file_issues)
        if parsed: parsed_files.append(parsed)
    ref_names = collect_reference_names(Path(args.reference_dir))
    exp_lines = count_nonempty_lines(Path(args.source_file))
    issues.extend(validate_per_reference_coverage(parsed_files, exp_lines, args.language, ref_names))
    if issues:
        for i in issues[:20]: print(f"ERROR: {i.path}: {i.message}")
        return 1
    print("PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
