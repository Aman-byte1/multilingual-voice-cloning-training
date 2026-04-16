#!/usr/bin/env python3
"""
OmniVoice LoRA fine-tuning for IWSLT 2026 ACL-6060 voice cloning
Per-language adapters for ar / fr / zh
Optimized for 3x A40 48GB with DDP and bf16
"""
import os
import sys
import json
import argparse
import types
import logging
from pathlib import Path
from functools import partial

import torch
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('training.log')]
)
logger = logging.getLogger(__name__)

def ensure_flex_attention_stub() -> bool:
    module_name = "torch.nn.attention.flex_attention"
    if module_name in sys.modules:
        return True
    try:
        import torch.nn.attention as attn
        stub = types.ModuleType(module_name)
        def create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None, _compile=False, device=None, **kwargs):
            seq_len = int(Q_LEN or KV_LEN or 1)
            if isinstance(mask_mod, partial) and mask_mod.args:
                document_ids = mask_mod.args[0]
                if torch.is_tensor(document_ids):
                    seq_len = int(document_ids.numel())
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float32)
            mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            return mask
        stub.create_block_mask = create_block_mask
        sys.modules[module_name] = stub
        setattr(attn, "flex_attention", stub)
        return True
    except Exception as exc:
        logger.warning(f"Could not install flex_attention stub: {exc}")
        return False

def patch_omnivoice_block_mask() -> bool:
    try:
        import omnivoice.models.omnivoice as omni_mod
    except Exception as exc:
        logger.error(f"Could not import OmniVoice module: {exc}")
        return False
    original = omni_mod.create_block_mask
    def _dense(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None, _compile=False, device=None, **kwargs):
        if isinstance(mask_mod, partial) and mask_mod.args:
            document_ids = mask_mod.args[0]
            if torch.is_tensor(document_ids):
                doc_ids = document_ids.to(device=device)
                q_len = int(Q_LEN if Q_LEN is not None else doc_ids.numel())
                kv_len = int(KV_LEN if KV_LEN is not None else doc_ids.numel())
                n = min(q_len, kv_len, doc_ids.numel())
                doc_ids = doc_ids[:n]
                same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                valid_tokens = doc_ids >= 0
                causal = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
                valid = same_doc & valid_tokens.unsqueeze(0) & valid_tokens.unsqueeze(1) & causal
                mask = torch.zeros((1, 1, n, n), device=device, dtype=torch.float32)
                mask.masked_fill_(~valid.unsqueeze(0).unsqueeze(0), -1e4)
                return mask
        return original(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, _compile=_compile, device=device, **kwargs)
    omni_mod.create_block_mask = _dense
    logger.info("Patched OmniVoice block mask")
    return True

def verify_configs(train_path: str, data_path: str):
    for p, name in [(train_path, "train"), (data_path, "data")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} config not found: {p}")
        with open(p, 'r') as f:
            json.load(f)
    logger.info("Configs validated")

def get_lora_config(args, model) -> LoraConfig:
    base_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.target_audio_modules:
        base_targets.extend(["audio_proj", "text_to_audio_proj"])
    available = {n.split('.')[-1] for n, _ in model.llm.named_modules()}
    targets = [t for t in base_targets if t in available]
    if not targets:
        raise RuntimeError("No LoRA target modules matched model naming")
    cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=args.use_rslora,
        init_lora_weights="gaussian"
    )
    logger.info(f"LoRA targets: {targets}")
    return cfg

def optimize_for_vram(config, level: str):
    presets = {
        "low": {"batch_tokens": 2048, "gradient_accumulation_steps": 32, "num_workers": 2},
        "medium": {"batch_tokens": 4096, "gradient_accumulation_steps": 16, "num_workers": 4},
        "high": {"batch_tokens": 8192, "gradient_accumulation_steps": 8, "num_workers": 4},
    }
    s = presets.get(level, presets["medium"])
    config.batch_tokens = s["batch_tokens"]
    config.gradient_accumulation_steps = s["gradient_accumulation_steps"]
    config.num_workers = s["num_workers"]
    logger.info(f"VRAM {level}: tokens={config.batch_tokens} accum={config.gradient_accumulation_steps}")
    return config

def main():
    parser = argparse.ArgumentParser(description="OmniVoice LoRA fine-tuning")
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_rslora", action="store_true")
    parser.add_argument("--target_audio_modules", action="store_true", default=True)
    parser.add_argument("--vram_level", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    verify_configs(args.train_config, args.data_config)
    ensure_flex_attention_stub()
    patch_omnivoice_block_mask()

    from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
    from omnivoice.training.config import TrainingConfig
    from omnivoice.training.trainer import OmniTrainer

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.steps is not None:
        config.steps = args.steps
    config = optimize_for_vram(config, args.vram_level)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Building model and tokenizer")
    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    lora_cfg = get_lora_config(args, model)
    model.llm = get_peft_model(model.llm, lora_cfg)

    if args.freeze_embeddings:
        for n, p in model.llm.named_parameters():
            if "embed" in n.lower() and "audio" not in n.lower():
                p.requires_grad = False

    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = False

    model.llm.print_trainable_parameters()

    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    if args.resume_from:
        trainer.resume_from_checkpoint(args.resume_from)

    trainer.train()

    final_path = os.path.join(args.output_dir, "final_lora")
    model.llm.save_pretrained(final_path)
    logger.info(f"Saved LoRA to {final_path}")

    if hasattr(model.llm, "merge_and_unload"):
        merged_path = os.path.join(args.output_dir, "merged_model")
        os.makedirs(merged_path, exist_ok=True)
        merged = model.llm.merge_and_unload()
        merged.save_pretrained(merged_path, safe_serialization=True)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(merged_path)
        cfg_path = os.path.join(merged_path, "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, "r"))
            cfg["architectures"] = ["OmniVoiceForCausalLM"]
            cfg["model_type"] = "omnivoice"
            json.dump(cfg, open(cfg_path, "w"), indent=2)
        logger.info(f"Saved merged model to {merged_path}")

if __name__ == "__main__":
    main()