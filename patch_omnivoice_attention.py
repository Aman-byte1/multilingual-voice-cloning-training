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
    """Patch builder.py: eager/sdpa attention + gradient checkpointing."""
    path = os.path.join(omnivoice_dir, "omnivoice", "training", "builder.py")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found, skipping")
        return False

    with open(path, "r") as f:
        content = f.read()

    changed = False

    # Clean up any corruption from previous bad patches (e.g. `config: config.attn...`)
    import re
    cleaned_content = re.sub(r'(config:\s*TrainingConfig,?)(\s*config\.attn_implementation = "sdpa"\s*)+', r'\1', content)
    cleaned_content = re.sub(r'config:\s*config:\s*config\.attn_implementation = "sdpa"', 'config: TrainingConfig,', cleaned_content)
    if cleaned_content != content:
        content = cleaned_content
        print("  🧹 Cleaned up corrupted builder.py signatures")
        changed = True

    # 1. Force config.attn_implementation = "sdpa" inside build_model_and_tokenizer
    target1 = 'logger.info("Initializing Model & Tokenizer...")'
    if 'config.attn_implementation = "sdpa"' not in content.split('def build_model_and_tokenizer')[1].split('def build_dataloaders')[0]:
        if target1 in content:
            content = content.replace(target1, 'config.attn_implementation = "sdpa"\n    ' + target1)
            print("  ✅ Forced attn_implementation = 'sdpa' in build_model_and_tokenizer")
            changed = True

    # 2. Force config.attn_implementation = "sdpa" inside build_dataloaders
    target2 = 'logger.info("Initializing Data Readers...")'
    if 'config.attn_implementation = "sdpa"' not in content.split('def build_dataloaders')[-1]:
        if target2 in content:
            content = content.replace(target2, 'config.attn_implementation = "sdpa"\n    ' + target2)
            print("  ✅ Forced attn_implementation = 'sdpa' in build_dataloaders")
            changed = True

    # 3. Enable gradient checkpointing
    if "gradient_checkpointing_enable" not in content:
        content = content.replace(
            "return model, tokenizer",
            (
                "# Enable gradient checkpointing to reduce activation memory\n"
                "    # (critical for eager/sdpa attention which stores O(n^2) attn weights per layer)\n"
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
    """Patch omnivoice.py: replace flex_attention forward path with dense causal mask.

    The upstream OmniVoice code already wraps the flex_attention import in
    try/except and sets ``_flex_attention_available``.  When flex_attention is
    NOT available the ``forward()`` method raises ``RuntimeError``.  This patch
    replaces that RuntimeError path with a working dense causal mask so training
    works with ``attn_implementation="eager"`` (or ``"sdpa"``).
    """
    path = os.path.join(omnivoice_dir, "omnivoice", "models", "omnivoice.py")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found, skipping")
        return False

    with open(path, "r") as f:
        content = f.read()

    changed = False

    # ── Patch 1: Replace the forward() flex_attention block ────────────
    # The upstream forward() has this pattern:
    #   if attention_mask is None and document_ids is not None:
    #       if not _flex_attention_available:
    #           raise RuntimeError(...)
    #       attention_mask = create_block_mask(...)
    #
    # We replace the ENTIRE if-block with a dense causal mask that always works.

    old_forward_block = """\
        if attention_mask is None and document_ids is not None:
            if not _flex_attention_available:
                raise RuntimeError(
                    "flex_attention is not available in the current environment. "
                    "If you do not need flex_attention, set "
                    '"attn_implementation": "sdpa" in your training config.'
                )
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

    new_forward_block = """\
        if attention_mask is None and document_ids is not None:
            # PATCHED: dense causal mask (replaces flex_attention / BlockMask)
            _seq_len = input_ids.size(-1)
            _doc_ids = document_ids[0].to(inputs_embeds.device)
            _same_doc = _doc_ids.unsqueeze(0) == _doc_ids.unsqueeze(1)
            _valid_tok = _doc_ids >= 0
            _causal = torch.tril(torch.ones(
                _seq_len, _seq_len,
                device=inputs_embeds.device, dtype=torch.bool,
            ))
            _valid = _same_doc & _valid_tok.unsqueeze(0) & _valid_tok.unsqueeze(1) & _causal
            _mask_dtype = inputs_embeds.dtype
            attention_mask = torch.zeros(
                _seq_len, _seq_len,
                device=inputs_embeds.device, dtype=_mask_dtype,
            )
            attention_mask.masked_fill_(~_valid, torch.finfo(_mask_dtype).min)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)"""

    if "# PATCHED: dense causal mask" in content:
        print("  ✓ forward() causal mask already patched")
    elif old_forward_block in content:
        content = content.replace(old_forward_block, new_forward_block)
        print("  ✅ forward() → dense causal mask (replaces flex_attention)")
        changed = True
    else:
        print("  ⚠ Could not find forward() flex_attention block — may already be patched")

    if changed:
        with open(path, "w") as f:
            f.write(content)

    return True



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

    # Clean / Restore the submodule first to get rid of any corruption
    if os.path.exists(os.path.join(args.omnivoice_dir, ".git")):
        print("  🧹 Restoring OmniVoice repository to clean state...")
        import subprocess
        subprocess.run(["git", "-C", args.omnivoice_dir, "restore", "."])
        subprocess.run(["git", "-C", args.omnivoice_dir, "clean", "-fd"])

    ok1 = patch_builder(args.omnivoice_dir)
    ok2 = patch_model(args.omnivoice_dir)

    if ok1 and ok2:
        print("\n✅ All patches applied successfully!")
    else:
        print("\n⚠ Some patches failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
