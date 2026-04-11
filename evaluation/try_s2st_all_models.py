#!/usr/bin/env python3
"""
S2ST Experiment: Can voice cloning models translate?
=====================================================
Tests VoxCPM2 by feeding English ref audio + English text
but requesting output in French. Does it translate or just
read English with a French accent?

Run on A40 (48GB).
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
OUTPUT_DIR = "./s2st_experiment"

login(token=HF_TOKEN)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Extract test samples ───────────────────────────────────────────
print("📥 Extracting test samples...")
ds = load_dataset(DATASET, split="eval")
print(f"   Dataset columns: {ds.column_names}")

samples = []
for i in range(min(NUM_SAMPLES, len(ds))):
    row = ds[i]

    # Try multiple possible column names
    en_text = ""
    for col in ["ref_en_text", "text_en", "en_text", "source_text"]:
        val = row.get(col, "")
        if val and val.strip():
            en_text = val.strip()
            break

    fr_text = ""
    for col in ["trg_fr_text", "text_fr", "fr_text", "target_text"]:
        val = row.get(col, "")
        if val and val.strip():
            fr_text = val.strip()
            break

    ref_data = None
    for col in ["ref_en_voice", "audio_en", "audio", "ref_voice"]:
        val = row.get(col)
        if val is not None and "array" in val:
            ref_data = val
            break

    if not ref_data:
        print(f"   ⚠ Sample {i}: no audio found, skipping")
        continue

    if not en_text and not fr_text:
        # Last resort: print first row keys and values to debug
        print(f"   ⚠ Sample {i}: no text found. Row keys: {list(row.keys())}")
        # Try to grab ANY text columns
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 10:
                print(f"      {k}: {v[:80]}...")
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
    print(f"     EN: {s['en_text'][:100]}")
    print(f"     FR: {s['fr_text'][:100]}")
print()


# ════════════════════════════════════════════════════════════════════
# TEST: VoxCPM2
# ════════════════════════════════════════════════════════════════════
def test_voxcpm():
    print("=" * 60)
    print("  TEST: VoxCPM2")
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
        # Test A: Normal — FR text + ref audio (baseline)
        if s["fr_text"]:
            try:
                wav = model.generate(text=s["fr_text"], reference_wav_path=s["ref_path"])
                out = os.path.join(OUTPUT_DIR, f"voxcpm_normal_fr_{s['idx']:03d}.wav")
                sf.write(out, wav, sr)
                print(f"  ✅ Normal (FR text): {out}")
                print(f"     Text given: {s['fr_text'][:80]}")
            except Exception as e:
                print(f"  ❌ Normal failed: {e}")

        # Test B: EN text + ref audio — does it just read English?
        if s["en_text"]:
            try:
                wav = model.generate(text=s["en_text"], reference_wav_path=s["ref_path"])
                out = os.path.join(OUTPUT_DIR, f"voxcpm_en_text_{s['idx']:03d}.wav")
                sf.write(out, wav, sr)
                print(f"  ✅ EN text (auto-detect): {out}")
                print(f"     Text given: {s['en_text'][:80]}")
                print(f"     → VoxCPM2 auto-detects language. Should speak English.")
            except Exception as e:
                print(f"  ❌ EN text failed: {e}")

        print()

    del model; gc.collect(); torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════
# TEST: OmniVoice
# ════════════════════════════════════════════════════════════════════
def test_omnivoice():
    print("=" * 60)
    print("  TEST: OmniVoice")
    print("=" * 60)
    try:
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice",
                                          device_map="cuda:0", dtype=torch.float16)
    except Exception as e:
        print(f"  ❌ Could not load: {e}")
        return

    for s in samples:
        if s["fr_text"]:
            try:
                audio = model.generate(text=s["fr_text"], ref_audio=s["ref_path"])
                out = os.path.join(OUTPUT_DIR, f"omnivoice_normal_fr_{s['idx']:03d}.wav")
                torchaudio.save(out, audio[0].cpu(), 24000)
                print(f"  ✅ Normal (FR text): {out}")
            except Exception as e:
                print(f"  ❌ Normal failed: {e}")

        if s["en_text"]:
            try:
                audio = model.generate(text=s["en_text"], ref_audio=s["ref_path"])
                out = os.path.join(OUTPUT_DIR, f"omnivoice_en_text_{s['idx']:03d}.wav")
                torchaudio.save(out, audio[0].cpu(), 24000)
                print(f"  ✅ EN text (no translate): {out}")
            except Exception as e:
                print(f"  ❌ EN text failed: {e}")

        print()

    del model; gc.collect(); torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════
# RUN TESTS
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🧪 S2ST EXPERIMENT: Can voice cloning models translate?\n")
    print("Each model gets the SAME English reference audio.")
    print("We test: (A) normal FR text, (B) EN text to see if it translates.\n")

    # Run VoxCPM2 first (it works), then OmniVoice
    test_voxcpm()
    test_omnivoice()

    print("🎉 Experiment complete!")
    print(f"   Listen to outputs in: {OUTPUT_DIR}/")
    print("   Compare *_normal_fr_* (baseline) vs *_en_text_* (translation test)")
