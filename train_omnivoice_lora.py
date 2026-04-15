#!/usr/bin/env python3
"""
Custom LoRA Fine-Tuning Script for OmniVoice
--------------------------------------------
This script wraps the native OmniVoice architecture and its underlying 
Qwen3 LLM backbone in PEFT LoRA adapters. It protects generalized knowledge 
while mapping specific Best-of-N speakers into the acoustic space.
"""

import os
import sys
import argparse
from functools import partial
import torch

from peft import LoraConfig, get_peft_model, TaskType


def patch_omnivoice_block_mask() -> bool:
    """Replace OmniVoice's BlockMask builder with a dense causal mask.

    Qwen3 eager attention expects a standard additive attention mask, but
    OmniVoice still builds a torch BlockMask for packed document_ids. This
    compatibility shim preserves the packed same-document causal structure
    while returning a dense tensor that eager attention can consume.
    """

    try:
        import omnivoice.models.omnivoice as omnivoice_module
    except Exception as exc:
        print(f"⚠ Could not import OmniVoice module for mask patch: {exc}", flush=True)
        return False

    original_create_block_mask = omnivoice_module.create_block_mask

    def _dense_create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=None,
        KV_LEN=None,
        _compile=False,
        device=None,
        **kwargs,
    ):
        if isinstance(mask_mod, partial) and mask_mod.args:
            document_ids = mask_mod.args[0]
            if torch.is_tensor(document_ids):
                doc_ids = document_ids.to(device=device)
                q_len = int(Q_LEN if Q_LEN is not None else doc_ids.numel())
                kv_len = int(KV_LEN if KV_LEN is not None else doc_ids.numel())
                doc_ids = doc_ids[: min(q_len, kv_len)]

                same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                valid_tokens = doc_ids >= 0
                causal = torch.tril(
                    torch.ones(
                        same_doc.shape,
                        device=doc_ids.device,
                        dtype=torch.bool,
                    )
                )
                valid = same_doc & valid_tokens.unsqueeze(0) & valid_tokens.unsqueeze(1) & causal

                mask = torch.zeros(
                    (1, 1, same_doc.size(0), same_doc.size(1)),
                    device=doc_ids.device,
                    dtype=torch.float32,
                )
                mask.masked_fill_(~valid.unsqueeze(0).unsqueeze(0), torch.finfo(mask.dtype).min)
                return mask

        return original_create_block_mask(
            mask_mod,
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            _compile=_compile,
            device=device,
            **kwargs,
        )

    omnivoice_module.create_block_mask = _dense_create_block_mask
    return True

def main():
    parser = argparse.ArgumentParser(description="OmniVoice LoRA Training Entry Point")
    parser.add_argument("--train_config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints")
    parser.add_argument("--data_config", type=str, required=True, help="Path to data config JSON")
    
    # Custom LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha for LoRA adapters")
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Use conservative memory settings (smaller packed batch + higher grad accumulation)",
    )
    
    args = parser.parse_args()

    patch_omnivoice_block_mask()

    from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
    from omnivoice.training.config import TrainingConfig
    from omnivoice.training.trainer import OmniTrainer

    # 1. Load Configuration
    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    if args.low_vram:
        # Eager attention has quadratic memory growth in sequence length.
        # Reduce packed length and compensate with more accumulation.
        config.batch_tokens = min(config.batch_tokens, 1024)
        config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, 32)
        config.num_workers = 0
        print(
            "🧯 Low-VRAM mode: "
            f"batch_tokens={config.batch_tokens}, "
            f"grad_accum={config.gradient_accumulation_steps}, "
            f"num_workers={config.num_workers}",
            flush=True,
        )

    # 2. Build Components
    print("🚀 Initializing native OmniVoice architecture...", flush=True)
    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    # 3. Inject PEFT LoRA Adapters into the Qwen3 backbone!
    print(f"🧬 Applying LoRA adapters (Rank: {args.lora_rank}) to OmniVoice Qwen3 backbone...", flush=True)
    
    # Target all main attention/math projection layers
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Wrap the language model component (Qwen3) of OmniVoice
    # Note: OmniVoice is defined as OmniVoice(config, llm=AutoModel.from_pretrained(...))
    model.llm = get_peft_model(model.llm, lora_config)

    # Reduce activation memory during training.
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = False
    
    # Optional debug print for trainable parameter count
    model.llm.print_trainable_parameters()

    # 4. Initialize Trainer and Start
    print("🔥 Starting LoRA Fine-Tuning loop...", flush=True)
    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    main()
