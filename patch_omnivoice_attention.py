#!/usr/bin/env python3
"""
Patch OmniVoice to work with 'eager' attention on GPUs where flex_attention
fails (e.g., GPUs with <128KB shared memory per SM like A40, T4).

Three changes:
  1. builder.py:     flex_attention → eager
  2. omnivoice.py:   BlockMask → standard 4D causal mask
  3. builder.py:     enable gradient checkpointing (reduces activation memory)

Run on the remote server:
    python patch_omnivoice_attention.py [--omnivoice-dir ./OmniVoice]
"""

import argparse
import os
import re
import sys


def patch_builder(omnivoice_dir: str) -> bool:
    """Patch builder.py: eager attention + gradient checkpointing."""
    path = os.path.join(omnivoice_dir, "omnivoice", "training", "builder.py")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found, skipping")
        return False

    with open(path, "r") as f:
        content = f.read()

    changed = False

    # 1. flex_attention → eager
    if "flex_attention" in content:
        content = content.replace(
            'attn_implementation="flex_attention"',
            'attn_implementation="eager"',
        )
        print(f"  ✅ flex_attention → eager")
        changed = True

    # 2. Enable gradient checkpointing
    if "gradient_checkpointing_enable" not in content:
        content = content.replace(
            "return model, tokenizer",
            (
                "# Enable gradient checkpointing to reduce activation memory\n"
                "    # (critical for eager attention which stores O(n^2) attn weights per layer)\n"
                "    model.llm.gradient_checkpointing_enable()\n"
                "\n"
                "    return model, tokenizer"
            ),
        )
        print(f"  ✅ Enabled gradient checkpointing")
        changed = True

    if changed:
        with open(path, "w") as f:
            f.write(content)
    else:
        print(f"  ✓ builder.py already patched")

    return True


def patch_model(omnivoice_dir: str) -> bool:
    """Patch omnivoice.py: BlockMask → standard 4D causal mask."""
    path = os.path.join(omnivoice_dir, "omnivoice", "models", "omnivoice.py")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found, skipping")
        return False

    with open(path, "r") as f:
        content = f.read()

    if "# PATCHED: standard causal mask" in content:
        print(f"  ✓ omnivoice.py already patched")
        return True

    old = """        if attention_mask is None and document_ids is not None:
            attention_mask = create_block_mask(
                _get_packed_mask(
                    document_ids[0].to(inputs_embeds.device),
                ),
                B=None,
                H=None,
                Q_LEN=input_ids.size(-1),
                KV_LEN=input_ids.size(-1),
                _compile=True,
                device=inputs_embeds.device,
            )"""

    new = """        if attention_mask is None and document_ids is not None:
            # PATCHED: standard causal mask (replaces BlockMask for eager attn)
            _seq_len = input_ids.size(-1)
            _doc_ids = document_ids[0].to(inputs_embeds.device)
            _same_doc = _doc_ids.unsqueeze(0) == _doc_ids.unsqueeze(1)
            _causal = torch.tril(torch.ones(
                _seq_len, _seq_len,
                device=inputs_embeds.device, dtype=torch.bool,
            ))
            _valid = _same_doc & _causal
            _mask_dtype = inputs_embeds.dtype
            attention_mask = torch.zeros(
                _seq_len, _seq_len,
                device=inputs_embeds.device, dtype=_mask_dtype,
            )
            attention_mask.masked_fill_(~_valid, torch.finfo(_mask_dtype).min)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)"""

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print(f"  ✅ BlockMask → standard 4D causal mask")
        return True
    else:
        print(f"  ⚠ Could not find BlockMask pattern — may already be patched")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Patch OmniVoice for eager attention compatibility"
    )
    parser.add_argument(
        "--omnivoice-dir",
        default="./OmniVoice",
        help="Path to OmniVoice repo (default: ./OmniVoice)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Patching OmniVoice for eager attention")
    print("=" * 60)

    ok1 = patch_builder(args.omnivoice_dir)
    ok2 = patch_model(args.omnivoice_dir)

    if ok1 and ok2:
        print("\n✅ All patches applied successfully!")
    else:
        print("\n⚠ Some patches failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
