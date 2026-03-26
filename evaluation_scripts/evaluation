#!/usr/bin/env python3
"""
Test the fine-tuned Chatterbox Multilingual model on French voice cloning.

Usage:
    # Test with merged model
    python test_finetuned.py --ref-audio reference_speaker.wav

    # Test with LoRA adapter
    python test_finetuned.py --ref-audio reference_speaker.wav --use-lora

    # Custom text
    python test_finetuned.py --ref-audio ref.wav --text "Bonjour le monde"
"""

import os
import argparse
import logging
import time

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Default test sentences in French
TEST_SENTENCES = [
    "Bonjour, comment allez-vous aujourd'hui ? Je suis ravi de vous rencontrer.",
    "L'intelligence artificielle transforme notre façon de communiquer et de travailler ensemble.",
    "La recherche en traitement automatique du langage naturel progresse chaque jour.",
    "Il fait un temps magnifique pour se promener dans le parc.",
    "Nous devons prendre en considération tous les aspects de cette problématique.",
    "La musique classique a une capacité unique à émouvoir les auditeurs.",
    "Les nouvelles technologies nous permettent de rester connectés malgré la distance.",
    "Je vous souhaite une excellente journée et beaucoup de succès dans vos projets.",
]


def test_merged_model(model_dir: str, ref_audio: str, texts: list,
                       output_dir: str, exaggeration: float, cfg_weight: float):
    """Test with the merged (base + LoRA) model."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading merged model from {model_dir} …")
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    logger.info("Model loaded ✓")

    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(texts):
        logger.info(f"\n[{i+1}/{len(texts)}] \"{text[:60]}…\"")
        t0 = time.time()

        wav = model.generate(
            text,
            audio_prompt_path=ref_audio,
            language_id="fr",
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            out_path = os.path.join(output_dir, f"test_merged_{i:02d}.wav")
            torchaudio.save(out_path, wav.cpu(), model.sr)
            dur = wav.shape[-1] / model.sr
            logger.info(f"  → {out_path}  ({dur:.1f}s, {time.time()-t0:.1f}s gen time)")

    logger.info(f"\nAll outputs saved to {output_dir}")


def test_lora_model(model_dir: str, ref_audio: str, texts: list,
                     output_dir: str, exaggeration: float, cfg_weight: float):
    """Test by loading base model + LoRA adapter."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from finetune_chatterbox_fr import (
        inject_lora, load_lora_state, TrainingConfig
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainingConfig()

    logger.info("Loading base Chatterbox Multilingual model …")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Inject & load LoRA
    lora_path = os.path.join(model_dir, "checkpoints", "best_lora_adapter.pt")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(model_dir, "checkpoints", "final_lora_adapter.pt")

    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from {lora_path} …")
        layers = inject_lora(
            model.t3, cfg.target_modules,
            cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
        )
        load_lora_state(layers, lora_path, device)
        logger.info("LoRA loaded ✓")
    else:
        logger.warning(f"No LoRA adapter found. Using base model only.")

    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(texts):
        logger.info(f"\n[{i+1}/{len(texts)}] \"{text[:60]}…\"")
        t0 = time.time()

        wav = model.generate(
            text,
            audio_prompt_path=ref_audio,
            language_id="fr",
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            out_path = os.path.join(output_dir, f"test_lora_{i:02d}.wav")
            torchaudio.save(out_path, wav.cpu(), model.sr)
            dur = wav.shape[-1] / model.sr
            logger.info(f"  → {out_path}  ({dur:.1f}s, {time.time()-t0:.1f}s gen time)")

    logger.info(f"\nAll outputs saved to {output_dir}")


def test_baseline(ref_audio: str, texts: list, output_dir: str,
                   exaggeration: float, cfg_weight: float):
    """Generate baseline (no fine-tuning) for comparison."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading base Chatterbox Multilingual model (no fine-tuning) …")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(texts):
        logger.info(f"\n[{i+1}/{len(texts)}] \"{text[:60]}…\"")
        t0 = time.time()

        wav = model.generate(
            text,
            audio_prompt_path=ref_audio,
            language_id="fr",
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            out_path = os.path.join(output_dir, f"test_baseline_{i:02d}.wav")
            torchaudio.save(out_path, wav.cpu(), model.sr)
            dur = wav.shape[-1] / model.sr
            logger.info(f"  → {out_path}  ({dur:.1f}s, {time.time()-t0:.1f}s gen time)")


def main():
    p = argparse.ArgumentParser(description="Test fine-tuned Chatterbox French model")
    p.add_argument("--model-dir", default="./chatterbox_fr_finetuned",
                   help="Fine-tuned model output directory")
    p.add_argument("--ref-audio", required=True,
                   help="Reference speaker audio file (.wav)")
    p.add_argument("--text", nargs="+", default=None,
                   help="Custom text(s) to synthesize")
    p.add_argument("--output-dir", default="./test_outputs",
                   help="Where to save generated audio")
    p.add_argument("--use-lora", action="store_true",
                   help="Load base + LoRA instead of merged model")
    p.add_argument("--baseline", action="store_true",
                   help="Also generate baseline (unfinetuned) outputs")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    args = p.parse_args()

    texts = args.text if args.text else TEST_SENTENCES

    if args.baseline:
        logger.info("=== Baseline (no fine-tuning) ===")
        test_baseline(args.ref_audio, texts,
                      os.path.join(args.output_dir, "baseline"),
                      args.exaggeration, args.cfg_weight)

    if args.use_lora:
        logger.info("=== LoRA Fine-tuned ===")
        test_lora_model(args.model_dir, args.ref_audio, texts,
                        os.path.join(args.output_dir, "lora"),
                        args.exaggeration, args.cfg_weight)
    else:
        merged = os.path.join(args.model_dir, "merged_model")
        if os.path.isdir(merged):
            logger.info("=== Merged Fine-tuned ===")
            test_merged_model(merged, args.ref_audio, texts,
                              os.path.join(args.output_dir, "merged"),
                              args.exaggeration, args.cfg_weight)
        else:
            logger.info("Merged model not found, falling back to LoRA …")
            test_lora_model(args.model_dir, args.ref_audio, texts,
                            os.path.join(args.output_dir, "lora"),
                            args.exaggeration, args.cfg_weight)

    logger.info("\nDone ✓")


if __name__ == "__main__":
    main()
