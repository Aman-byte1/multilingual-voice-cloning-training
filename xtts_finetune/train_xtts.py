#!/usr/bin/env python3
"""
XTTS v2 Fine-Tuning Script
Initializes the Coqui Trainer API to fine-tune the XTTS model on our 
curated Best-of-N dataset (French, Arabic, Chinese).
"""

import os
import argparse
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
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
    print("📥 Downloading Base XTTS v2 model files (vocab, model, dvae, mel_stats)...")
    CHECKPOINTS_OUT_PATH = os.path.join(args.output_path, "base_model_files")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(XTTS_CHECKPOINT):
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK, TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], 
            CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # Note: Chinese language tag in XTTS is "zh-cn"
    lang_map = {"fr": "fr", "ar": "ar", "zh": "zh-cn"}
    dataset_configs = []
    
    for lang_code, xtts_lang in lang_map.items():
        meta_file = os.path.join(args.dataset_path, f"metadata_{lang_code}.csv")
        if os.path.exists(meta_file):
            dataset_configs.append(
                BaseDatasetConfig(
                    formatter="ljspeech", 
                    meta_file_train=meta_file,
                    path=args.dataset_path, 
                    language=xtts_lang
                )
            )
            print(f"📖 Registered dataset for {xtts_lang}")

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=args.output_path,
        model_args=model_args,
        run_name="xtts_finetuned",
        project_name="xtts_finetune",
        run_description="GPT XTTS fine-tuning on multilingual dataset",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=25,
        plot_step=100,
        log_model_step=1000,
        save_step=2000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {"text": "Bonjour, ceci est un test de clonage vocal en français.", "speaker_wav": None, "language": "fr"}
        ],
        datasets=dataset_configs
    )

    # 3. Initialize model and load pre-trained weights
    print("🔧 Initializing GPTTrainer...")
    model = GPTTrainer.init_from_config(config)
    
    # 4. Load dataset
    print("🔍 Loading dataset samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_configs, 
        eval_split=True, 
        eval_split_max_size=config.eval_split_max_size, 
        eval_split_size=config.eval_split_size
    )

    print(f"Total Train samples: {len(train_samples)}")
    print(f"Total Eval samples: {len(eval_samples)}")

    # 5. Initialize Trainer
    trainer_args = TrainerArgs(
        restore_path=None,  # Checkpoint is restored via xtts_checkpoint in GPTArgs
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=8,  # Helps simulate larger batch size
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
