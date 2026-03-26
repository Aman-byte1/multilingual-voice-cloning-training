#!/usr/bin/env python3
"""
Convert the merged Chatterbox model from .pt → .safetensors format.

After LoRA fine-tuning, the merged T3 weights are saved as t3_mtl23ls_v2.pt.
ChatterboxMultilingualTTS.from_local() requires .safetensors format.

Usage:
    python fix_merged_model.py
    python fix_merged_model.py --model-dir ./chatterbox_fr_finetuned/merged_model
"""

import os
import sys
import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_pretrained_cache() -> str:
    """Locate the Chatterbox pretrained model cache directory."""
    # Common HuggingFace cache locations
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")

    candidates = []
    if os.path.isdir(hub_dir):
        for d in os.listdir(hub_dir):
            if "chatterbox" in d.lower() and "multilingual" in d.lower():
                full = os.path.join(hub_dir, d)
                candidates.append(full)
            elif "resemble" in d.lower() and "chatterbox" in d.lower():
                full = os.path.join(hub_dir, d)
                candidates.append(full)

    # Also check torch hub cache
    torch_home = os.environ.get("TORCH_HOME",
                    os.path.join(os.path.expanduser("~"), ".cache", "torch"))
    torch_hub = os.path.join(torch_home, "hub")
    if os.path.isdir(torch_hub):
        for d in os.listdir(torch_hub):
            if "chatterbox" in d.lower():
                candidates.append(os.path.join(torch_hub, d))

    for c in candidates:
        # Walk to find the actual model files
        for root, dirs, files in os.walk(c):
            if "ve.pt" in files or "s3gen.pt" in files or "conds.pt" in files:
                return root

    return ""


def convert_merged_model(model_dir: str):
    """Convert .pt to .safetensors and copy required companion files."""
    pt_path = os.path.join(model_dir, "t3_mtl23ls_v2.pt")
    st_path = os.path.join(model_dir, "t3_mtl23ls_v2.safetensors")

    if not os.path.exists(pt_path):
        logger.error(f"Cannot find {pt_path}")
        logger.error("Run training first, or check the --model-dir path.")
        sys.exit(1)

    # Load state dict
    logger.info(f"Loading {pt_path} …")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Convert to safetensors
    try:
        from safetensors.torch import save_file
        logger.info(f"Converting to safetensors → {st_path}")
        save_file(state_dict, st_path)
        logger.info("Conversion complete ✓")
    except ImportError:
        logger.error("safetensors package not installed. Run: pip install safetensors")
        sys.exit(1)

    # Copy companion files from pretrained cache if they're missing
    required_files = [
        "ve.pt",
        "s3gen.pt",
        "conds.pt",
        "grapheme_mtl_merged_expanded_v1.json",
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        logger.info(f"Missing companion files: {missing}")
        cache_dir = find_pretrained_cache()

        if cache_dir:
            logger.info(f"Found pretrained cache at: {cache_dir}")
            import shutil
            for f in missing:
                src = os.path.join(cache_dir, f)
                dst = os.path.join(model_dir, f)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    logger.info(f"  Copied {f}")
                else:
                    logger.warning(f"  {f} not found in cache")
        else:
            logger.warning(
                "Could not find pretrained cache. You may need to manually copy:\n"
                + "\n".join(f"  - {f}" for f in missing)
                + "\nfrom the Chatterbox pretrained model directory."
            )

    # Final check
    all_present = all(
        os.path.exists(os.path.join(model_dir, f))
        for f in ["t3_mtl23ls_v2.safetensors"] + required_files
    )

    if all_present:
        logger.info(f"\n✓ Merged model ready at: {model_dir}")
        logger.info("Load with:")
        logger.info(f"  model = ChatterboxMultilingualTTS.from_local('{model_dir}', device='cuda')")
    else:
        logger.warning("Some files are still missing. Check the output above.")

    # List final directory contents
    logger.info(f"\nContents of {model_dir}:")
    for f in sorted(os.listdir(model_dir)):
        size = os.path.getsize(os.path.join(model_dir, f))
        logger.info(f"  {f:50s}  {size / 1e6:8.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert merged Chatterbox model .pt → .safetensors")
    parser.add_argument(
        "--model-dir", type=str,
        default="./chatterbox_fr_finetuned/merged_model",
        help="Directory containing the merged t3_mtl23ls_v2.pt file",
    )
    args = parser.parse_args()
    convert_merged_model(args.model_dir)


if __name__ == "__main__":
    main()
