#!/usr/bin/env python3
"""
S2ST Experiment v2: Can models generate from ref_text ONLY?
============================================================
Give ONLY: ref_audio + ref_text (English transcript)
Do NOT give: target text
See if the model can generate speech from just the reference.
"""

import os
import gc
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET = "ymoslem/acl-6060"
NUM_SAMPLES = 3
OUTPUT_DIR = "./s2st_experiment_v2"

login(token=HF_TOKEN)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Extract test samples ───────────────────────────────────────────
print("📥 Extracting test samples...")
ds = load_dataset(DATASET, split="eval")

samples = []
for i in range(min(NUM_SAMPLES, len(ds))):
    row = ds[i]
    ref_data = row.get("audio")
    en_text = row.get("text_en", "").strip()
    fr_text = row.get("text_fr", "").strip()

    if not ref_data or not en_text:
        continue

    ref_path = os.path.join(OUTPUT_DIR, f"ref_en_{i:03d}.wav")
    wav = torch.from_numpy(np.asarray(ref_data["array"], dtype=np.float32))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(ref_path, wav, ref_data["sampling_rate"])
    samples.append({
        "idx": i,
        "ref_path": ref_path,
        "en_text": en_text,
        "fr_text": fr_text,
    })

del ds; gc.collect()
print(f"   Got {len(samples)} samples\n")
for s in samples:
    print(f"   Sample {s['idx']}:")
    print(f"     EN ref_text: {s['en_text'][:100]}")
    print(f"     FR trg_text: {s['fr_text'][:100]} (NOT used — this is just for comparison)")
print()


# ════════════════════════════════════════════════════════════════════
# TEST 1: VoxCPM2 — ref_audio + prompt_text ONLY, no target text
# ════════════════════════════════════════════════════════════════════
def test_voxcpm():
    print("=" * 60)
    print("  TEST: VoxCPM2 — ref_text only, NO target text")
    print("=" * 60)
    try:
        from voxcpm import VoxCPM
        import soundfile as sf
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    except Exception as e:
        print(f"  ❌ Could not load: {e}")
        return

    sr = model.tts_model.sample_rate

    for s in samples:
        print(f"\n  Sample {s['idx']}:")

        # Attempt 1: prompt_text=en_text, text="" (empty)
        print("  → Attempt 1: prompt_text=EN, text='' (empty)")
        try:
            wav = model.generate(
                text="",
                reference_wav_path=s["ref_path"],
                prompt_text=s["en_text"]
            )
            out = os.path.join(OUTPUT_DIR, f"voxcpm_reftext_empty_{s['idx']:03d}.wav")
            sf.write(out, wav, sr)
            print(f"    ✅ Generated: {out}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

        # Attempt 2: prompt_text=en_text, text=en_text (same text as ref)
        print("  → Attempt 2: prompt_text=EN, text=EN (repeat same text)")
        try:
            wav = model.generate(
                text=s["en_text"],
                reference_wav_path=s["ref_path"],
                prompt_text=s["en_text"]
            )
            out = os.path.join(OUTPUT_DIR, f"voxcpm_reftext_repeat_{s['idx']:03d}.wav")
            sf.write(out, wav, sr)
            print(f"    ✅ Generated: {out}")
            print(f"       → Should sound like the original speaker repeating themselves")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

        # Attempt 3: prompt_text=en_text, text=" " (single space)
        print("  → Attempt 3: prompt_text=EN, text=' ' (space)")
        try:
            wav = model.generate(
                text=" ",
                reference_wav_path=s["ref_path"],
                prompt_text=s["en_text"]
            )
            out = os.path.join(OUTPUT_DIR, f"voxcpm_reftext_space_{s['idx']:03d}.wav")
            sf.write(out, wav, sr)
            print(f"    ✅ Generated: {out}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print()


# ════════════════════════════════════════════════════════════════════
# TEST 2: OmniVoice — ref_audio + ref_text ONLY, no target text
# ════════════════════════════════════════════════════════════════════
def test_omnivoice():
    print("=" * 60)
    print("  TEST: OmniVoice — ref_text only, NO target text")
    print("=" * 60)
    try:
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice",
                                          device_map="cuda:0", dtype=torch.float16)
    except Exception as e:
        print(f"  ❌ Could not load: {e}")
        return

    for s in samples:
        print(f"\n  Sample {s['idx']}:")

        # Attempt 1: ref_text=en_text, text="" (empty)
        print("  → Attempt 1: ref_text=EN, text='' (empty)")
        try:
            audio = model.generate(
                text="",
                ref_audio=s["ref_path"],
                ref_text=s["en_text"]
            )
            out = os.path.join(OUTPUT_DIR, f"omnivoice_reftext_empty_{s['idx']:03d}.wav")
            torchaudio.save(out, audio[0].cpu(), 24000)
            print(f"    ✅ Generated: {out}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

        # Attempt 2: ref_text=en_text, text=en_text (repeat)
        print("  → Attempt 2: ref_text=EN, text=EN (repeat)")
        try:
            audio = model.generate(
                text=s["en_text"],
                ref_audio=s["ref_path"],
                ref_text=s["en_text"]
            )
            out = os.path.join(OUTPUT_DIR, f"omnivoice_reftext_repeat_{s['idx']:03d}.wav")
            torchaudio.save(out, audio[0].cpu(), 24000)
            print(f"    ✅ Generated: {out}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print()


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🧪 S2ST EXPERIMENT v2: ref_text only, NO target text\n")
    print("Goal: see if models generate anything from just ref_audio + ref_text\n")

    test_voxcpm()
    test_omnivoice()

    print("🎉 Done! Check outputs in:", OUTPUT_DIR)
