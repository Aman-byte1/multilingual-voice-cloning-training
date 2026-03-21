#!/usr/bin/env python3
"""
Inference script to run the fine-tuned Chatterbox Multilingual TTS model with 
a LoRA adapter downloaded directly from Hugging Face.

Example Usage:
    python evaluation/inference.py --text "Bonjour, comment ça va?" --ref-audio path/to/english_voice.wav --output my_clone.wav
"""

import os
import math
import argparse
from typing import List

import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download

# Base Multilingual TTS model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer wrapping nn.Linear."""
    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f, out_f = original.in_features, original.out_features
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original_layer(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora


def inject_lora(model: nn.Module, targets: List[str], rank: int,
                alpha: float, dropout: float = 0.0) -> List[LoRALayer]:
    """Inject LoRA layers into matching nn.Linear modules."""
    layers = []
    for name, module in list(model.named_modules()):
        for target in targets:
            if target in name and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                lora = LoRALayer(module, rank, alpha, dropout)
                setattr(parent, parts[-1], lora)
                layers.append(lora)
                break
    return layers


def load_lora_state(layers: List[LoRALayer], path: str, device: str = "cuda"):
    """Load LoRA weights into injected layers."""
    state = torch.load(path, map_location=device, weights_only=True)
    for i, layer in enumerate(layers):
        if f"layer_{i}_A" in state:
            layer.lora_A.data = state[f"layer_{i}_A"].to(device)
            layer.lora_B.data = state[f"layer_{i}_B"].to(device)


def main():
    parser = argparse.ArgumentParser(description="Inference with Chatterbox FR LoRA model")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize (in French)")
    parser.add_argument("--ref-audio", type=str, required=True, help="Path to reference audio file (e.g., english source)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--repo-id", type=str, default="amanuelbyte/chatterbox-fr-lora", help="Hugging Face repository ID")
    parser.add_argument("--lora-file", type=str, default="best_lora_adapter.pt", help="Which LoRA weight file to get from the repo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Download LoRA adapter weights from HF hub
    print(f"Downloading LoRA weights '{args.lora_file}' from HF repo: {args.repo_id}...")
    lora_path = hf_hub_download(repo_id=args.repo_id, filename=args.lora_file)
    print(f"Downloaded LoRA to: {lora_path}")

    # 2. Load the base ChatterboxMultilingualTTS model
    print("Loading base ChatterboxMultilingualTTS model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # 3. Inject LoRA layers
    print("Injecting LoRA layers into T3 Transformer...")
    targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layers = inject_lora(model.t3, targets=targets, rank=32, alpha=64.0, dropout=0.0)
    print(f"Injected LoRA into {len(layers)} linear layers (rank=32, alpha=64.0)")

    # 4. Load LoRA adapter weights
    print(f"Loading weights from {lora_path} ...")
    load_lora_state(layers, lora_path, device=device)
    model.eval()

    # 5. Synthesize speech
    print(f"Synthesizing text: '{args.text}'")
    print(f"Using reference audio: '{args.ref_audio}'")
    
    with torch.no_grad():
        wav = model.generate(
            args.text, 
            audio_prompt_path=args.ref_audio,
            language_id="fr"
        )
    
    # 6. Save back to WAV
    if isinstance(wav, torch.Tensor):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        torchaudio.save(args.output, wav.cpu(), model.sr)
        print(f"✅ Saved generated audio to: {args.output}")
    else:
        print("Model did not return a valid tensor.")

if __name__ == "__main__":
    main()
