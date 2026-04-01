import os
import argparse
from huggingface_hub import HfApi, ModelCard, ModelCardData

def main():
    parser = argparse.ArgumentParser(description="Upload Chatterbox LoRA model to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="amanuelbyte/chatterbox-fr-lora-v2", help="Target HuggingFace Repo ID")
    parser.add_argument("--model-dir", type=str, default="./chatterbox_fr_finetuned", help="Directory containing checkpoints")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="HF Token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Please provide a HuggingFace token via --token or the HF_TOKEN environment variable.")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, exist_ok=True, repo_type="model")

    # Generate Model Card
    readme_content = f"""
---
language:
- fr
- en
library_name: transformers
tags:
- voice-cloning
- text-to-speech
- lora
- chatterbox
- cross-lingual
pipeline_tag: text-to-speech
---

# Chatterbox French Cross-Lingual Voice Cloning LoRA (Rank 32)
This repository contains a highly optimized LoRA adapter for the Chatterbox Multilingual TTS model, fine-tuned specifically for high-fidelity English-to-French cross-lingual voice cloning.

## Model Description
* **Base Architecture**: Chatterbox Multilingual TTS
* **Adapter Type**: LoRA (Low-Rank Adaptation)
* **Optimization Strategy**: Sparse Dataset Formulation (2 references per target)
* **LoRA Rank**: 32 | **Alpha**: 64
* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Performance Metrics (RTX 4090, 100 Samples)
By switching to a low-redundancy "Sparse" dataset and training with Rank 32, this model achieves near-zero-shot levels of speaker preservation while significantly improving French phonetic accuracy.

* **Speaker Similarity**: 0.8452 (Out of 1.0)
* **WER (Word Error Rate)**: 0.0758
* **chrF (Format/Prosody)**: 93.57
* **PESQ**: 1.22
* **MCD (Mel-Cepstral Distortion)**: 13.29

## How to use for Inference

You must have the Chatterbox framework installed to use this adapter.

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# 1. Define LoRA mapping helper
class LoRALayer(nn.Module):
    def __init__(self, original, rank=32, alpha=64.0, dropout=0.05):
        super().__init__()
        self.original_layer = original
        self.scaling = alpha / rank
        in_f, out_f = original.in_features, original.out_features
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling)

# 2. Load Base Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# 3. Download and Inject LoRA
lora_path = hf_hub_download(repo_id="{args.repo_id}", filename="best_lora_adapter.pt")
payload = torch.load(lora_path, map_location=device, weights_only=True)
lora_sd = payload.get("lora_state_dict", payload)

targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
for name, module in model.t3.named_modules():
    if isinstance(module, nn.Linear) and any(x in name for x in targets):
        parent_name, child_name = ".".join(name.split(".")[:-1]), name.split(".")[-1]
        parent = model.t3.get_submodule(parent_name)
        setattr(parent, child_name, LoRALayer(module))

# Load the weights into the injected layers
current_sd = model.t3.state_dict()
for key, value in lora_sd.items():
    if key in current_sd:
        current_sd[key] = value.to(device)
model.t3.load_state_dict(current_sd, strict=False)
model.t3.eval()

# 4. Generate Audio
import torchaudio
text = "Bonjour, bienvenue à cette démonstration de clonage vocal cross-lingue!"
ref_audio_path = "path/to/english_voice.wav"

with torch.inference_mode():
    wav = model.generate(text, audio_prompt_path=ref_audio_path, language_id="fr")
    
torchaudio.save("output_french.wav", wav.cpu().unsqueeze(0), 16000)
```
"""

    # Write Model Card
    readme_path = os.path.join(args.model_dir, "checkpoints", "README.md")
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Generated README.md at {readme_path}")

    # Upload the folder
    print(f"Uploading files from {args.model_dir}/checkpoints to {args.repo_id}...")
    api.upload_folder(
        folder_path=os.path.join(args.model_dir, "checkpoints"),
        repo_id=args.repo_id,
        repo_type="model",
    )
    
    print(f"✅ Success! Your model and README are live at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
