#!/usr/bin/env python3
"""
Repair merged OmniVoice checkpoints:
  1. Add missing `llm.` prefix to bare LLM weight keys
  2. Copy missing OmniVoice-specific weights (audio_embeddings, audio_heads,
     codebook_layer_offsets) from the base model
  3. Save the repaired checkpoint back to pytorch_model.bin
"""

import os
import sys
import torch
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

OMNIVOICE_SPECIFIC_KEYS = {
    "audio_embeddings.weight",
    "audio_heads.weight",
    "codebook_layer_offsets",
}

# Keys that belong to the LLM and need `llm.` prefix
LLM_KEY_PATTERNS = [
    "layers.",
    "embed_tokens.",
    "norm.",
    "lm_head.",
]


def needs_llm_prefix(key: str) -> bool:
    """Check if a key is a bare LLM key that needs `llm.` prefix."""
    if key.startswith("llm."):
        return False
    return any(key.startswith(p) for p in LLM_KEY_PATTERNS)


def fix_checkpoint(merged_dir: str, base_state_dict: dict):
    merged_path = Path(merged_dir)
    
    # Find the checkpoint file
    ckpt_file = None
    for name in ["pytorch_model.bin", "model.safetensors"]:
        candidate = merged_path / name
        if candidate.exists():
            ckpt_file = candidate
            break
    
    if ckpt_file is None:
        print(f"  ❌ No checkpoint found in {merged_dir}")
        return False

    print(f"  📂 Loading {ckpt_file.name}...")
    
    if ckpt_file.name.endswith(".safetensors"):
        from safetensors.torch import load_file, save_file
        state_dict = load_file(str(ckpt_file))
        use_safetensors = True
    else:
        state_dict = torch.load(str(ckpt_file), map_location="cpu", weights_only=True)
        use_safetensors = False

    # ── Step 1: Check if keys need fixing ────────────────────────
    bare_llm_keys = [k for k in state_dict if needs_llm_prefix(k)]
    has_llm_prefix = any(k.startswith("llm.") for k in state_dict)
    missing_omnivoice = [k for k in OMNIVOICE_SPECIFIC_KEYS if k not in state_dict]

    if not bare_llm_keys and not missing_omnivoice:
        print(f"  ✅ Checkpoint already looks correct! No changes needed.")
        return True

    print(f"  🔧 Found {len(bare_llm_keys)} bare LLM keys to prefix")
    print(f"  🔧 Found {len(missing_omnivoice)} missing OmniVoice keys to restore")

    # ── Step 2: Rename keys ──────────────────────────────────────
    new_state_dict = {}
    for k, v in state_dict.items():
        if needs_llm_prefix(k):
            new_key = f"llm.{k}"
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # ── Step 3: Add missing OmniVoice-specific weights ───────────
    for key in missing_omnivoice:
        if key in base_state_dict:
            new_state_dict[key] = base_state_dict[key]
            print(f"    ✚ Restored {key} from base model")
        else:
            print(f"    ⚠ Could not find {key} in base model")

    # ── Step 4: Backup and save ──────────────────────────────────
    backup = ckpt_file.with_suffix(ckpt_file.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(str(ckpt_file), str(backup))
        print(f"  💾 Backed up original → {backup.name}")

    if use_safetensors:
        save_file(new_state_dict, str(ckpt_file))
    else:
        torch.save(new_state_dict, str(ckpt_file))

    # Summary
    llm_keys = [k for k in new_state_dict if k.startswith("llm.")]
    omni_keys = [k for k in new_state_dict if k in OMNIVOICE_SPECIFIC_KEYS]
    print(f"  ✅ Saved repaired checkpoint: {len(llm_keys)} llm keys, {len(omni_keys)} omnivoice keys")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix merged OmniVoice checkpoint weight keys")
    parser.add_argument("--lang", type=str, default="all",
                        help="Language to fix: fr, ar, zh, or all")
    parser.add_argument("--exp-dir", type=str, default="./exp",
                        help="Experiment directory")
    args = parser.parse_args()

    langs = ["fr", "ar", "zh"] if args.lang == "all" else [args.lang]

    # Load base model state dict once
    print("📥 Loading base OmniVoice model state dict...")
    base_path = Path(snapshot_download("k2-fsa/OmniVoice"))
    
    # Load from safetensors or pytorch_model.bin
    base_ckpt = base_path / "model.safetensors"
    if base_ckpt.exists():
        from safetensors.torch import load_file
        base_state_dict = load_file(str(base_ckpt))
    else:
        base_ckpt = base_path / "pytorch_model.bin"
        base_state_dict = torch.load(str(base_ckpt), map_location="cpu", weights_only=True)
    
    print(f"  ✅ Base model has {len(base_state_dict)} keys\n")

    for lang in langs:
        merged_dir = os.path.join(args.exp_dir, f"omnivoice_finetuned_{lang}", "merged_model")
        print(f"\n{'='*60}")
        print(f"  🌍 Fixing {lang.upper()} — {merged_dir}")
        print(f"{'='*60}")
        
        if not os.path.exists(merged_dir):
            print(f"  ❌ Directory not found: {merged_dir}")
            continue

        fix_checkpoint(merged_dir, base_state_dict)

    print(f"\n🎉 Done! You can now re-run evaluation.")


if __name__ == "__main__":
    main()
