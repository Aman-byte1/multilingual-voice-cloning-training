#!/usr/bin/env python3
"""Compatibility launcher for OmniVoice audio token extraction."""
import importlib
import sys
import types
import torch
from functools import partial

def ensure_flex_attention_stub() -> None:
    module_name = "torch.nn.attention.flex_attention"
    if module_name in sys.modules:
        return
    try:
        import torch.nn.attention as attention_module
        stub = types.ModuleType(module_name)

        stub._DEFAULT_SPARSE_BLOCK_SIZE = 128

        class BlockMask:
            pass

        class AuxRequest:
            pass

        def create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None,
                              _compile=False, device=None, **kwargs):
            seq_len = int(Q_LEN or KV_LEN or 1)
            if isinstance(mask_mod, partial) and mask_mod.args:
                document_ids = mask_mod.args[0]
                if torch.is_tensor(document_ids):
                    seq_len = int(document_ids.numel())
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float32)
            mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            return mask

        def flex_attention(query, key, value, *args, **kwargs):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )

        stub.BlockMask = BlockMask
        stub.AuxRequest = AuxRequest
        stub.create_block_mask = create_block_mask
        stub.flex_attention = flex_attention

        sys.modules[module_name] = stub
        setattr(attention_module, "flex_attention", stub)
    except Exception:
        return

if __name__ == "__main__":
    ensure_flex_attention_stub()
    sys.argv = ["omnivoice.scripts.extract_audio_tokens", *sys.argv[1:]]
    module = importlib.import_module("omnivoice.scripts.extract_audio_tokens")
    module.main()
