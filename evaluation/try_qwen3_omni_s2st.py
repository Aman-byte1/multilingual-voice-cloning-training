#!/usr/bin/env python3
"""
Quick experiment: Qwen3-Omni as end-to-end Speech-to-Speech Translator
======================================================================
Feed English audio → Get French/Arabic/Chinese audio out.
No pre-translated text needed. The model translates internally.

Requirements:
  pip install git+https://github.com/huggingface/transformers
  pip install accelerate qwen-omni-utils soundfile
  pip install flash-attn --no-build-isolation  (optional but recommended)

GPU: A100 80GB recommended (model is ~60GB in BF16)
"""

import os
import gc
import torch
import soundfile as sf
import torchaudio
import numpy as np
from datasets import load_dataset
from huggingface_hub import login

# ── Config ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATASET = "ymoslem/acl-6060"
SPLIT = "eval"
NUM_SAMPLES = 5  # just a quick test
TARGET_LANGS = ["French", "Arabic", "Chinese"]
OUTPUT_DIR = "./s2st_qwen3_omni_test"
# ────────────────────────────────────────────────────────────────────

login(token=HF_TOKEN)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Extract a few English reference audios ──────────────────
print("📥 Loading dataset...")
ds = load_dataset(DATASET, split=SPLIT)

ref_samples = []
for i in range(min(NUM_SAMPLES, len(ds))):
    row = ds[i]
    ref_data = row.get("ref_en_voice") or row.get("audio_en") or row.get("audio")
    if ref_data is None:
        continue

    ref_path = os.path.join(OUTPUT_DIR, f"ref_en_{i:03d}.wav")
    wav = torch.from_numpy(np.asarray(ref_data["array"], dtype=np.float32))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(ref_path, wav, ref_data["sampling_rate"])

    ref_samples.append({
        "idx": i,
        "ref_path": ref_path,
        "en_text": row.get("ref_en_text", ""),
        "trg_fr": row.get("trg_fr_text", ""),
        "trg_ar": row.get("trg_ar_text", ""),
        "trg_zh": row.get("trg_zh_text", ""),
    })

del ds
gc.collect()
print(f"   Extracted {len(ref_samples)} samples")

# ── Step 2: Load Qwen3-Omni ────────────────────────────────────────
print("🔧 Loading Qwen3-Omni (this takes a while)...")
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
print("   Model loaded ✓")

# ── Step 3: Translate and Generate Speech ───────────────────────────
for lang in TARGET_LANGS:
    print(f"\n🌍 Translating to {lang}...")

    for s in ref_samples:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": s["ref_path"]},
                    {"type": "text", "text": (
                        f"Listen to this English speech carefully. "
                        f"Translate what you hear into {lang} and speak the translation aloud. "
                        f"Only output the translated speech, nothing else."
                    )},
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=False
        )
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False
        )
        inputs = inputs.to(model.device).to(model.dtype)

        try:
            text_ids, audio_out = model.generate(
                **inputs,
                speaker="Chelsie",
                use_audio_in_video=False,
                thinker_return_dict_in_generate=True,
            )

            # Decode the text response
            gen_text = processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            lang_code = {"French": "fr", "Arabic": "ar", "Chinese": "zh"}[lang]
            out_path = os.path.join(OUTPUT_DIR, f"s2st_{lang_code}_{s['idx']:03d}.wav")

            if audio_out is not None:
                sf.write(out_path, audio_out.reshape(-1).detach().cpu().numpy(), 24000)
                print(f"   ✅ Sample {s['idx']}: saved → {out_path}")
                print(f"      Model said: {gen_text[:100]}...")
                # Show ground truth for comparison
                gt_key = f"trg_{lang_code}"
                print(f"      Ground truth: {s.get(gt_key, 'N/A')[:100]}...")
            else:
                print(f"   ⚠ Sample {s['idx']}: no audio generated")
                print(f"      Model said: {gen_text[:200]}")

        except Exception as e:
            print(f"   ❌ Sample {s['idx']} failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

print(f"\n🎉 Done! Check {OUTPUT_DIR}/ for results.")
print("Compare the generated audio against the ground-truth translations above.")
