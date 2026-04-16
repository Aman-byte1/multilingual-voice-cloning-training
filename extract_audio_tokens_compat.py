#!/usr/bin/env python3
"""Compatibility launcher for OmniVoice audio token extraction.

Some torch builds (including many A40 container images) do not ship
`torch.nn.attention.flex_attention`. OmniVoice imports that module at import
-time, so we install a minimal shim before running the extractor module.
"""

import importlib
import sys
import types

import torch


def ensure_flex_attention_stub() -> None:
    """Install a minimal flex_attention shim for older torch builds."""
    module_name = "torch.nn.attention.flex_attention"
    if module_name in sys.modules:
        return

    try:
        import torch.nn.attention as attention_module

        stub = types.ModuleType(module_name)

        def create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=None,
            KV_LEN=None,
            _compile=False,
            device=None,
            **kwargs,
        ):
            seq_len = int(Q_LEN or KV_LEN or 1)
            causal = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            mask = torch.zeros(
                (1, 1, seq_len, seq_len), device=device, dtype=torch.float32
            )
            mask.masked_fill_(
                ~causal.unsqueeze(0).unsqueeze(0),
                torch.finfo(mask.dtype).min,
            )
            return mask

        class BlockMask:
            pass

        def dummy_flex_attention(*args, **kwargs):
            pass

        stub.create_block_mask = create_block_mask
        stub._DEFAULT_SPARSE_BLOCK_SIZE = 128
        stub.BlockMask = BlockMask
        stub.flex_attention = dummy_flex_attention
        sys.modules[module_name] = stub
        setattr(attention_module, "flex_attention", stub)
    except Exception:
        # If this fails, the original OmniVoice import error will surface,
        # which keeps failure behavior explicit for debugging.
        return


if __name__ == "__main__":
    ensure_flex_attention_stub()
    sys.argv = ["omnivoice.scripts.extract_audio_tokens", *sys.argv[1:]]
    module = importlib.import_module("omnivoice.scripts.extract_audio_tokens")
    module.main()
