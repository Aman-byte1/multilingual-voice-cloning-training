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
import torch

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer

from peft import LoraConfig, get_peft_model, TaskType

def main():
    parser = argparse.ArgumentParser(description="OmniVoice LoRA Training Entry Point")
    parser.add_argument("--train_config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints")
    parser.add_argument("--data_config", type=str, required=True, help="Path to data config JSON")
    
    # Custom LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha for LoRA adapters")
    
    args = parser.parse_args()

    # 1. Load Configuration
    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

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
        # We save the specific OmniVoice heads fully to prevent shape mismatches / corruption
        modules_to_save=["lm_head", "audio_head"] 
    )

    # Wrap the language model component (Qwen3) of OmniVoice
    # Note: OmniVoice is defined as OmniVoice(config, llm=AutoModel.from_pretrained(...))
    model.llm = get_peft_model(model.llm, lora_config)
    
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
