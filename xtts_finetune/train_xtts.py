#!/usr/bin/env python3
"""
XTTS v2 LoRA Fine-Tuning Script
================================
Uses PEFT LoRA adapters on the GPT decoder only, following the approach from
gokhaneraslan/XTTS_V2-finetuning. This prevents catastrophic forgetting by
freezing the entire base model and only training small low-rank adapter matrices
on the GPT attention/feedforward layers.

Reference: https://github.com/gokhaneraslan/XTTS_V2-finetuning
"""

import os
import sys
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
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (trainer counts from 0, so 2 = 1 real epoch)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (lower = less params, higher = more expressive)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 1. Download Base XTTS v2 Model
    print("📥 Downloading Base XTTS v2 model files...")
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

    # 2. Register multilingual datasets
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

    # 3. Model args (identical to official Coqui recipe)
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,   # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    # Find a speaker wav for test sentences
    sample_speaker_wav = None
    wavs_dir = os.path.join(args.dataset_path, "wavs")
    if os.path.exists(wavs_dir):
        wav_files = [f for f in os.listdir(wavs_dir) if f.endswith(".wav")]
        if wav_files:
            sample_speaker_wav = os.path.join(wavs_dir, wav_files[0])

    # 4. Training config — following official recipe + gokhaneraslan's LoRA config
    config = GPTTrainerConfig(
        output_path=args.output_path,
        model_args=model_args,
        run_name="xtts_lora_finetuned",
        project_name="xtts_lora_finetune",
        run_description="LoRA XTTS fine-tuning on multilingual Best-of-N dataset",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=8,
        eval_split_max_size=256,
        epochs=args.epochs,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
        datasets=dataset_configs
    )

    # 5. Initialize base model
    print("🔧 Initializing GPTTrainer base model...")
    model = GPTTrainer.init_from_config(config)

    # 6. Apply LoRA adapters to the GPT decoder ONLY
    # This is the critical fix: instead of full-param finetuning (which caused
    # catastrophic forgetting and doubled WER from 0.073 -> 0.164), we inject
    # tiny trainable matrices into the GPT attention layers.
    # Reference: gokhaneraslan/XTTS_V2-finetuning/lora_train.py
    print("🧬 Applying LoRA adapters to GPT decoder...")
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # We add 'qkv' and 'proj_out' heavily used inside the conditioning_encoder!
        target_modules=["c_attn", "c_proj", "c_fc", "qkv", "proj_out"],  
    )

    # Apply LoRA ONLY to the GPT component, leaving DVAE/HiFi-GAN/conditioning frozen
    model.xtts.gpt = get_peft_model(model.xtts.gpt, peft_config)

    trainable_params, total_params = model.xtts.gpt.get_nb_trainable_parameters()
    print(f"🔥 Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"🧊 Total:     {total_params:,}")
    
    # 7. Load dataset
    print("🔍 Loading dataset samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_configs, 
        eval_split=True, 
        eval_split_max_size=config.eval_split_max_size, 
        eval_split_size=config.eval_split_size
    )

    print(f"Total Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    # 8. Initialize Trainer (following official recipe: grad_accum=84)
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=16,  # 1392 steps / 16 = ~87 weight updates per epoch
        ),
        config,
        output_path=args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # 9. Train!
    print("🚀 Starting LoRA XTTS fine-tuning...")
    trainer.fit()

    # 10. Save LoRA adapter separately for easy loading
    print("💾 Saving LoRA adapter...")
    lora_output_dir = os.path.join(args.output_path, "lora_adapter")
    os.makedirs(lora_output_dir, exist_ok=True)
    model.xtts.gpt.save_pretrained(lora_output_dir)
    config.save_json(os.path.join(lora_output_dir, "original_xtts_config.json"))
    print(f"✅ LoRA adapter saved to {lora_output_dir}")

if __name__ == "__main__":
    main()
