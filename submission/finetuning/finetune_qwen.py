#!/usr/bin/env python3
"""
🔧 IWSLT 2026 — Qwen3-TTS Fine-Tuning Pipeline
=================================================
End-to-end fine-tuning using the official Qwen3-TTS SFT workflow,
adapted for the IWSLT cross-lingual voice cloning competition.

Pipeline:
  1. Download & filter data from HuggingFace dataset
  2. Export to WAV files + JSONL format required by Qwen3-TTS
  3. Extract audio codes via prepare_data (Qwen3-TTS tokenizer)
  4. Run SFT training with recommended hyperparams (lr=2e-6, lora_scale=0.3)
  5. Save checkpoints

Usage:
    python finetune_qwen.py \
        --dataset amanuelbyte/acl-voice-cloning-fr-data \
        --output-dir /workspace/qwen_ft \
        --num-epochs 5 \
        --batch-size 2 \
        --lr 2e-6

Based on: https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("QwenFT")

TARGET_SR = 24000  # Qwen3-TTS requires 24kHz
MIN_DUR = 1.0
MAX_DUR = 20.0


def export_dataset_to_wavs(
    dataset_name: str,
    output_dir: str,
    max_samples: int = None,
    split: str = "train",
) -> str:
    """
    Download HuggingFace dataset and export to WAV + JSONL for Qwen3-TTS.

    Returns path to the raw JSONL file.
    """
    from datasets import load_dataset

    log.info(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    log.info(f"Processing {len(ds)} samples")

    wav_dir = os.path.join(output_dir, "wavs")
    ref_dir = os.path.join(output_dir, "refs")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "train_raw.jsonl")
    entries = []
    skipped = 0

    # Pick a single good reference audio (first valid one) for consistency
    # Official recommendation: use same ref_audio for all samples
    best_ref_path = None

    for i in tqdm(range(len(ds)), desc="Exporting WAVs"):
        row = ds[i]

        # Get target audio (French cloned voice)
        tgt_data = row.get("trg_fr_voice") or row.get("cloned_audio_fr")
        text_fr = (row.get("trg_fr_text") or row.get("text_fr") or "").strip()

        # Get reference audio (English original voice)
        ref_data = row.get("ref_en_voice") or row.get("audio_en")

        if not tgt_data or not text_fr or not ref_data:
            skipped += 1
            continue

        # Process target audio
        tgt_arr = np.asarray(tgt_data["array"], dtype=np.float32)
        tgt_sr = tgt_data["sampling_rate"]

        # Resample to 24kHz (required by Qwen3-TTS)
        if tgt_sr != TARGET_SR:
            tgt_arr = librosa.resample(tgt_arr, orig_sr=tgt_sr, target_sr=TARGET_SR)

        # Duration filter
        dur = len(tgt_arr) / TARGET_SR
        if not (MIN_DUR <= dur <= MAX_DUR):
            skipped += 1
            continue

        # Save target WAV
        tgt_path = os.path.join(wav_dir, f"utt_{i:05d}.wav")
        sf.write(tgt_path, tgt_arr, TARGET_SR)

        # Save reference WAV (resample to 24kHz)
        ref_arr = np.asarray(ref_data["array"], dtype=np.float32)
        ref_sr = ref_data["sampling_rate"]
        if ref_sr != TARGET_SR:
            ref_arr = librosa.resample(ref_arr, orig_sr=ref_sr, target_sr=TARGET_SR)

        ref_path = os.path.join(ref_dir, f"ref_{i:05d}.wav")
        sf.write(ref_path, ref_arr, TARGET_SR)

        # Use first valid ref as the shared reference
        if best_ref_path is None:
            best_ref_path = ref_path

        entries.append({
            "audio": tgt_path,
            "text": text_fr,
            "ref_audio": best_ref_path,  # Same ref for all (official recommendation)
        })

    log.info(f"Exported {len(entries)} samples, skipped {skipped}")
    log.info(f"Reference audio: {best_ref_path}")

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"Raw JSONL: {jsonl_path}")
    return jsonl_path


def prepare_audio_codes(
    input_jsonl: str,
    output_jsonl: str,
    device: str = "cuda:0",
    tokenizer_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    batch_size: int = 32,
):
    """
    Extract audio codes using the Qwen3-TTS tokenizer.
    This is the official prepare_data.py logic.
    """
    from qwen_tts import Qwen3TTSTokenizer

    log.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_path,
        device_map=device,
    )

    total_lines = open(input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]
    log.info(f"Processing {len(total_lines)} entries for audio code extraction")

    final_lines = []
    batch_lines = []
    batch_audios = []

    for line in tqdm(total_lines, desc="Extracting codes"):
        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= batch_size:
            enc_res = tokenizer.encode(batch_audios)
            for code, bl in zip(enc_res.audio_codes, batch_lines):
                bl['audio_codes'] = code.cpu().tolist()
                final_lines.append(bl)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer.encode(batch_audios)
        for code, bl in zip(enc_res.audio_codes, batch_lines):
            bl['audio_codes'] = code.cpu().tolist()
            final_lines.append(bl)
        batch_lines.clear()
        batch_audios.clear()

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    log.info(f"Codes extracted: {output_jsonl} ({len(final_lines)} entries)")
    return output_jsonl


def run_sft(
    train_jsonl: str,
    init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    output_model_path: str = "output",
    batch_size: int = 2,
    lr: float = 2e-6,
    num_epochs: int = 5,
    speaker_name: str = "iwslt_speaker",
    gradient_accumulation_steps: int = 4,
):
    """
    Run SFT training — adapted from official sft_12hz.py with
    community-recommended hyperparams for the 1.7B model.
    """
    from accelerate import Accelerator
    from dataset import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    log.info(f"Loading model: {init_model_path}")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(init_model_path)

    # Load dataset
    train_data = open(train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    log.info(f"Training on {len(train_data)} samples")

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    target_speaker_embedding = None
    model.train()

    log.info(f"Starting training: {num_epochs} epochs, lr={lr}, bs={batch_size}")
    log.info(f"Gradient accumulation: {gradient_accumulation_steps}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_steps += 1

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(num_steps, 1)
        accelerator.print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if accelerator.is_main_process:
            output_dir = os.path.join(output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(init_model_path, output_dir, dirs_exist_ok=True)

            # Update config for custom voice
            input_config_file = os.path.join(init_model_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")

            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {speaker_name: 3000}
            talker_config["spk_is_dialect"] = {speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Save model weights
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.detach().to("cpu")
                for k, v in unwrapped_model.state_dict().items()
            }

            # Drop speaker encoder weights (not needed for inference)
            keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
            for k in keys_to_drop:
                del state_dict[k]

            # Embed target speaker
            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = \
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            accelerator.print(f"Checkpoint saved: {output_dir}")

    log.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="IWSLT 2026 — Qwen3-TTS Fine-Tuning")
    parser.add_argument("--dataset", default="amanuelbyte/acl-voice-cloning-fr-data",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (None = all)")
    parser.add_argument("--output-dir", default="/workspace/qwen_ft",
                        help="Output directory for checkpoints")
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base model path")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6,
                        help="Learning rate (2e-6 recommended for 1.7B)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--speaker-name", default="iwslt_speaker",
                        help="Speaker name for custom voice config")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip data export (use existing train_raw.jsonl)")
    parser.add_argument("--skip-codes", action="store_true",
                        help="Skip audio code extraction (use existing train_with_codes.jsonl)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_jsonl = os.path.join(args.output_dir, "train_raw.jsonl")
    codes_jsonl = os.path.join(args.output_dir, "train_with_codes.jsonl")

    # Step 1: Export dataset to WAVs + JSONL
    if not args.skip_export:
        log.info("=" * 60)
        log.info("  STEP 1: Exporting dataset to WAV files")
        log.info("=" * 60)
        raw_jsonl = export_dataset_to_wavs(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            split=args.split,
        )
    else:
        log.info("Skipping export (using existing train_raw.jsonl)")

    # Step 2: Extract audio codes
    if not args.skip_codes:
        log.info("=" * 60)
        log.info("  STEP 2: Extracting audio codes")
        log.info("=" * 60)
        codes_jsonl = prepare_audio_codes(
            input_jsonl=raw_jsonl,
            output_jsonl=codes_jsonl,
        )
    else:
        log.info("Skipping code extraction (using existing train_with_codes.jsonl)")

    # Step 3: Fine-tune
    log.info("=" * 60)
    log.info("  STEP 3: Fine-tuning Qwen3-TTS")
    log.info(f"  Model: {args.model_path}")
    log.info(f"  Epochs: {args.num_epochs} | LR: {args.lr} | BS: {args.batch_size}")
    log.info("=" * 60)

    # Change to finetuning dir so dataset.py import works
    ft_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ft_dir)

    run_sft(
        train_jsonl=codes_jsonl,
        init_model_path=args.model_path,
        output_model_path=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        speaker_name=args.speaker_name,
        gradient_accumulation_steps=args.grad_accum,
    )

    log.info("=" * 60)
    log.info("  FINE-TUNING COMPLETE")
    log.info(f"  Checkpoints: {args.output_dir}/checkpoint-epoch-*")
    log.info("")
    log.info("  Inference test:")
    log.info(f'    python -c "')
    log.info(f'    import torch, soundfile as sf')
    log.info(f'    from qwen_tts import Qwen3TTSModel')
    log.info(f'    tts = Qwen3TTSModel.from_pretrained(')
    log.info(f'        \\"{args.output_dir}/checkpoint-epoch-{args.num_epochs-1}\\",')
    log.info(f'        device_map=\\"cuda:0\\", dtype=torch.bfloat16,')
    log.info(f'        attn_implementation=\\"flash_attention_2\\")')
    log.info(f'    wavs, sr = tts.generate_custom_voice(')
    log.info(f'        text=\\"Bonjour, comment allez-vous?\\",')
    log.info(f'        speaker=\\"{args.speaker_name}\\")')
    log.info(f'    sf.write(\\"test_ft.wav\\", wavs[0], sr)')
    log.info(f'    "')
    log.info("=" * 60)


if __name__ == "__main__":
    main()
