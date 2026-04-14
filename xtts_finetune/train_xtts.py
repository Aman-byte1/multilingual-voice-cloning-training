#!/usr/bin/env python3
"""
XTTS v2 Fine-Tuning Script
Initializes the Coqui Trainer API to fine-tune the XTTS model on our 
curated Best-of-N dataset (French, Arabic, Chinese).
"""

import os
import argparse
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Path to the prepared xtts_dataset directory")
    parser.add_argument("--output-path", required=True, help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep small for XTTS!)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 1. Download Base XTTS v2 Model
    print("📥 Downloading Base XTTS v2 model (if not already cached)...")
    manager = ModelManager()
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_path, config_path, model_item = manager.download_model(model_name)

    # 2. Load and override config
    if config_path is None:
        config_path = os.path.join(model_path, "config.json")
    config = XttsConfig()
    config.load_json(config_path)
    
    # Overrides for fine-tuning
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    config.print_step = 25
    config.print_eval = False
    
    # Disable mixed precision — XTTS v2 is known to get NaN losses with fp16
    config.mixed_precision = False
    
    # Optimizer settings for fine-tuning
    config.optimizer = "AdamW"
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 0.01}
    config.lr = 1e-5
    
    # Note: Chinese language tag in XTTS is "zh-cn"
    lang_map = {"fr": "fr", "ar": "ar", "zh": "zh-cn"}
    dataset_configs = []
    
    for lang_code, xtts_lang in lang_map.items():
        meta_file = os.path.join(args.dataset_path, f"metadata_{lang_code}.csv")
        if os.path.exists(meta_file):
            dataset_configs.append(
                BaseDatasetConfig(
                    formatter="ljspeech", 
                    meta_file_train=f"metadata_{lang_code}.csv",
                    path=args.dataset_path, 
                    language=xtts_lang
                )
            )
            print(f"📖 Registered dataset for {xtts_lang}")

    config.datasets = dataset_configs
    
    # 3. Initialize model and load pre-trained weights
    print("🔧 Initializing XTTS model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=False)
    
    # 4. Load dataset
    print("🔍 Loading dataset samples...")
    train_samples, eval_samples = load_tts_samples(dataset_configs, eval_split=True, eval_split_max_size=256, eval_split_size=0.1)

    print(f"Total Train samples: {len(train_samples)}")
    print(f"Total Eval samples: {len(eval_samples)}")

    # 5. Initialize Trainer
    trainer_args = TrainerArgs(
        restore_path=None,  # We load the weights manually above
        skip_train_epoch=False,
        start_with_eval=True,
    )

    trainer = Trainer(
        trainer_args,
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # 6. Start Training
    print("🚀 Starting XTTS fine-tuning...")
    trainer.fit()

if __name__ == "__main__":
    main()
