#!/usr/bin/env python3
"""
Custom LoRA Fine-Tuning Script for OmniVoice Voice Cloning
----------------------------------------------------------
Optimized for Chinese voice cloning with proper error handling.
"""

import os
import sys
import argparse
import types
from functools import partial
import torch
import logging
from pathlib import Path

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def ensure_torch_float8_compat() -> bool:
    """Deprecated: kept for compatibility with older imports."""
    return True


def ensure_flex_attention_stub() -> bool:
    """Register a minimal flex_attention module for older torch builds.

    OmniVoice imports `torch.nn.attention.flex_attention` during module
    import. If the installed torch build does not ship that module, we inject
    a compatibility stub so OmniVoice can load and the later patch can replace
    it with a dense causal mask implementation.
    """

    module_name = "torch.nn.attention.flex_attention"
    if module_name in sys.modules:
        return True

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
            if isinstance(mask_mod, partial) and mask_mod.args:
                document_ids = mask_mod.args[0]
                if torch.is_tensor(document_ids):
                    seq_len = int(document_ids.numel())

            causal = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            mask = torch.zeros(
                (1, 1, seq_len, seq_len), device=device, dtype=torch.float32
            )
            mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(mask.dtype).min)
            return mask

        stub.create_block_mask = create_block_mask
        sys.modules[module_name] = stub
        setattr(attention_module, "flex_attention", stub)
        return True
    except Exception as exc:
        logger.warning(f"Could not install flex_attention stub: {exc}")
        return False


def patch_omnivoice_block_mask() -> bool:
    """Replace OmniVoice's BlockMask builder with a dense causal mask."""
    try:
        import omnivoice.models.omnivoice as omnivoice_module
    except Exception as exc:
        logger.error(f"Could not import OmniVoice module: {exc}")
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
                    torch.ones(same_doc.shape, device=doc_ids.device, dtype=torch.bool)
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
            mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, _compile=_compile, device=device, **kwargs
        )

    omnivoice_module.create_block_mask = _dense_create_block_mask
    logger.info("✓ Successfully patched OmniVoice block mask")
    return True



_FLOAT8_ALIASES = [
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",   # the one that actually crashed
]
for _f8_name in _FLOAT8_ALIASES:
    if not hasattr(torch, _f8_name):
        setattr(torch, _f8_name, torch.float32)

def verify_configs(train_config_path: str, data_config_path: str):
    """Verify that config files exist and are valid JSON."""
    import json
    
    for config_path, name in [(train_config_path, "train"), (data_config_path, "data")]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{name} config not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✓ Loaded {name} config: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {name} config: {e}")


def get_lora_config(args) -> LoraConfig:
    """Create optimized LoRA configuration for voice cloning."""
    
    # Validate rank/alpha ratio
    if args.lora_alpha < args.lora_rank:
        logger.warning(
            f"⚠ lora_alpha ({args.lora_alpha}) < lora_rank ({args.lora_rank}). "
            f"Recommended: alpha = 2 * rank"
        )
    
    # Target modules - comprehensive coverage
    target_modules = [
        # Self-attention projections (critical for timbre)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Feed-forward network (important for prosody)
        "gate_proj", "up_proj", "down_proj",
    ]
    
    # Add audio-specific modules if they exist
    if args.target_audio_modules:
        target_modules.extend(["audio_proj", "text_to_audio_proj"])
    
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        use_rslora=args.use_rslora,
        init_lora_weights="gaussian"
    )
    
    logger.info(f"LoRA Config: rank={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    logger.info(f"Target modules: {config.target_modules}")
    
    return config


def optimize_for_vram(config, vram_level: str):
    """Apply VRAM-specific optimizations."""
    
    optimizations = {
        "low": {
            "batch_tokens": 2048,
            "gradient_accumulation_steps": 32,
            "num_workers": 2,
        },
        "medium": {
            "batch_tokens": 4096,
            "gradient_accumulation_steps": 16,
            "num_workers": 4,
        },
        "high": {
            "batch_tokens": 8192,
            "gradient_accumulation_steps": 8,
            "num_workers": 4,
        }
    }
    
    settings = optimizations.get(vram_level, optimizations["medium"])
    
    for key, value in settings.items():
        if key == "gradient_accumulation_steps":
            setattr(config, key, max(getattr(config, key, 1), value))
        else:
            setattr(config, key, min(getattr(config, key, float('inf')), value))
    
    logger.info(
        f"VRAM Optimization ({vram_level}): "
        f"batch_tokens={config.batch_tokens}, "
        f"grad_accum={config.gradient_accumulation_steps}, "
        f"workers={config.num_workers}"
    )
    
    return config


def print_training_summary(model, config, args):
    """Print comprehensive training setup summary."""
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    # Model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"\n📊 Model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # LoRA info
    logger.info(f"\n🧬 LoRA:")
    logger.info(f"  Rank: {args.lora_rank}")
    logger.info(f"  Alpha: {args.lora_alpha}")
    logger.info(f"  Dropout: {args.lora_dropout}")
    logger.info(f"  RSLoRA: {args.use_rslora}")
    
    # Training info
    logger.info(f"\n🔥 Training:")
    logger.info(f"  Steps: {config.steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch tokens: {config.batch_tokens}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Mixed precision: {config.mixed_precision}")
    
    # Paths
    logger.info(f"\n📁 Paths:")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Train config: {args.train_config}")
    logger.info(f"  Data config: {args.data_config}")
    
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice LoRA Fine-Tuning for Voice Cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--train_config", type=str, required=True,
                       help="Path to training config JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save checkpoints and logs")
    parser.add_argument("--data_config", type=str, required=True,
                       help="Path to data config JSON")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=32,
                       help="Rank for LoRA adapters (16-64 for voice cloning)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="Alpha scaling (recommended: 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="Dropout rate for LoRA layers")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate override (defaults to train_config value)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Training steps override (defaults to train_config value)")
    parser.add_argument("--use_rslora", action="store_true",
                       help="Use Rank-Stabilized LoRA for better training stability")
    parser.add_argument("--target_audio_modules", action="store_true",
                       help="Also apply LoRA to audio projection modules")
    
    # Memory optimization
    parser.add_argument("--vram_level", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="VRAM optimization preset")
    parser.add_argument("--low-vram", action="store_true",
                       help="Alias for --vram_level low")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Enable 8-bit quantization (experimental)")
    
    # Training behavior
    parser.add_argument("--freeze_embeddings", action="store_true",
                       help="Freeze text embeddings (focus on audio generation)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Handle legacy --low-vram flag
    if args.low_vram:
        args.vram_level = "low"
    
    # Validate alpha/rank ratio
    if args.lora_alpha == 64 and args.lora_rank != 32:
        logger.warning(f"Auto-adjusting lora_alpha to {2 * args.lora_rank} (2 * rank)")
        args.lora_alpha = 2 * args.lora_rank
    
    try:
        # Verify configs exist and are valid
        verify_configs(args.train_config, args.data_config)

        # Make OmniVoice importable even on torch builds that lack flex_attention.
        ensure_flex_attention_stub()
        
        # Apply mask patch
        if not patch_omnivoice_block_mask():
            logger.warning("⚠ Mask patch failed, continuing anyway...")
        
        # Import OmniVoice components
        from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
        from omnivoice.training.config import TrainingConfig
        from omnivoice.training.trainer import OmniTrainer
        
        # Load configuration
        logger.info("📋 Loading training configuration...")
        config = TrainingConfig.from_json(args.train_config)
        config.output_dir = args.output_dir
        config.data_config = args.data_config
        
        # Apply explicit CLI overrides when provided.
        if args.lr is not None:
            logger.info(f"Overriding learning rate: {config.learning_rate} -> {args.lr}")
            config.learning_rate = args.lr

        if args.steps is not None:
            logger.info(f"Overriding steps: {config.steps} -> {args.steps}")
            config.steps = args.steps

        # Apply VRAM optimizations
        config = optimize_for_vram(config, args.vram_level)
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build model and tokenizer
        logger.info("🚀 Building OmniVoice model...")
        model, tokenizer = build_model_and_tokenizer(config)
        
        # Apply 8-bit quantization if requested
        if args.use_8bit:
            logger.info("Applying 8-bit quantization...")
            model.llm = prepare_model_for_kbit_training(model.llm)
        
        # Build dataloaders
        logger.info("📦 Building dataloaders...")
        train_loader, eval_loader = build_dataloaders(config, tokenizer)
        
        # Log dataset info. Iterable datasets in streaming mode may not define __len__.
        try:
            logger.info(f"  Train batches: {len(train_loader)}")
        except TypeError:
            logger.info("  Train batches: streaming dataset (length unavailable)")

        if eval_loader is not None:
            try:
                logger.info(f"  Eval batches: {len(eval_loader)}")
            except TypeError:
                logger.info("  Eval batches: streaming dataset (length unavailable)")
        
        # Apply LoRA
        logger.info("🧬 Applying LoRA adapters...")
        lora_config = get_lora_config(args)
        model.llm = model.llm.to(torch.float32)
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm = model.llm.to(torch.float32)
        
        # Freeze embeddings if requested
        if args.freeze_embeddings:
            logger.info("❄️  Freezing text embeddings...")
            for name, param in model.llm.named_parameters():
                if "embed" in name.lower() and "audio" not in name.lower():
                    param.requires_grad = False
        
        # Enable gradient checkpointing
        if hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable()
            logger.info("✓ Enabled gradient checkpointing")
        
        if hasattr(model.llm, "config"):
            model.llm.config.use_cache = False
        
        # Print trainable parameters
        model.llm.print_trainable_parameters()
        
        # Print comprehensive summary
        print_training_summary(model, config, args)
        
        # Initialize trainer
        logger.info("🔥 Initializing trainer...")
        trainer = OmniTrainer(
            model=model,
            config=config,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Starting training loop...\n")
        trainer.train()
        
        # Save final model
        final_path = os.path.join(args.output_dir, "final_lora")
        model.llm.save_pretrained(final_path)
        logger.info(f"\n💾 Saved final LoRA adapters to: {final_path}")
        
        # Save merged model (optional)
        if hasattr(model.llm, "merge_and_unload"):
            merged_path = os.path.join(args.output_dir, "merged_model")
            logger.info(f"💾 Saving merged model to: {merged_path}")
            merged_model = model.llm.merge_and_unload()
            merged_model.save_pretrained(merged_path)
        
        logger.info("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()