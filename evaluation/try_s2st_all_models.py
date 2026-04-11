#!/usr/bin/env python3
"""
S2ST Experiment: Can voice cloning models translate?
=====================================================
Tests each model by feeding English ref audio + English text
but requesting output in French. Does it translate or just
read English with a French accent?

Models tested:
  1. OmniVoice  (pip install omnivoice)
  2. XTTS-v2    (pip install coqui-tts)
  3. VoxCPM2    (pip install voxcpm)
  4. Qwen3-TTS  (pip install qwen3-tts)

Run on A40 (48GB). Each model tested independently.
"""

import os
import gc
import time
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
samples = []
for i in range(min(NUM_SAMPLES, len(ds))):
    row = ds[i]
    ref_data = row.get("ref_en_voice") or row.get("audio_en") or row.get("audio")
    if not ref_data:
        continue
    ref_path = os.path.join(OUTPUT_DIR, f"ref_en_{i:03d}.wav")
    wav = torch.from_numpy(np.asarray(ref_data["array"], dtype=np.float32))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(ref_path, wav, ref_data["sampling_rate"])
    samples.append({
        "idx": i,
        "ref_path": ref_path,
        "en_text": row.get("ref_en_text", ""),
        "fr_text": row.get("trg_fr_text", ""),
    })
del ds; gc.collect()
print(f"   Got {len(samples)} samples\n")

for s in samples:
    print(f"   Sample {s['idx']}:")
    print(f"     EN: {s['en_text'][:80]}...")
    print(f"     FR: {s['fr_text'][:80]}...")
print()


# ════════════════════════════════════════════════════════════════════
# TEST 1: OmniVoice — give EN audio + FR text (normal cloning)
#         vs EN audio + EN text with no language flag
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
        # Test A: Normal — give French text (this is what eval does)
        try:
            audio = model.generate(text=s["fr_text"], ref_audio=s["ref_path"])
            out = os.path.join(OUTPUT_DIR, f"omnivoice_normal_fr_{s['idx']:03d}.wav")
            torchaudio.save(out, audio[0].cpu(), 24000)
            print(f"  ✅ Normal (FR text): {out}")
        except Exception as e:
            print(f"  ❌ Normal failed: {e}")

        # Test B: Give ENGLISH text — does it auto-translate?
        try:
            audio = model.generate(text=s["en_text"], ref_audio=s["ref_path"])
            out = os.path.join(OUTPUT_DIR, f"omnivoice_en_text_{s['idx']:03d}.wav")
            torchaudio.save(out, audio[0].cpu(), 24000)
            print(f"  ✅ EN text (no translate): {out}")
            print(f"     → Listen to this: does it speak English or try French?")
        except Exception as e:
            print(f"  ❌ EN text failed: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print()


# ════════════════════════════════════════════════════════════════════
# TEST 2: XTTS-v2 — give EN text but set language="fr"
# ════════════════════════════════════════════════════════════════════
def test_xtts():
    print("=" * 60)
    print("  TEST: XTTS-v2")
    print("=" * 60)
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "isin_mps_friendly"):
            _pu.isin_mps_friendly = torch.isin
        from TTS.api import TTS
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    except Exception as e:
        print(f"  ❌ Could not load: {e}")
        return

    for s in samples:
        # Test A: Normal — FR text + language="fr"
        try:
            out = os.path.join(OUTPUT_DIR, f"xtts_normal_fr_{s['idx']:03d}.wav")
            model.tts_to_file(text=s["fr_text"], language="fr",
                              speaker_wav=s["ref_path"], file_path=out)
            print(f"  ✅ Normal (FR text, lang=fr): {out}")
        except Exception as e:
            print(f"  ❌ Normal failed: {e}")

        # Test B: EN text but language="fr" — what happens?
        try:
            out = os.path.join(OUTPUT_DIR, f"xtts_en_text_fr_lang_{s['idx']:03d}.wav")
            model.tts_to_file(text=s["en_text"], language="fr",
                              speaker_wav=s["ref_path"], file_path=out)
            print(f"  ✅ EN text + lang=fr: {out}")
            print(f"     → Does it speak English with French accent? Or garbage?")
        except Exception as e:
            print(f"  ❌ EN text + lang=fr failed: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print()


# ════════════════════════════════════════════════════════════════════
# TEST 3: VoxCPM2
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
        # Test A: Normal — FR text + ref audio
        try:
            wav = model.generate(text=s["fr_text"], reference_wav_path=s["ref_path"])
            out = os.path.join(OUTPUT_DIR, f"voxcpm_normal_fr_{s['idx']:03d}.wav")
            sf.write(out, wav, sr)
            print(f"  ✅ Normal (FR text): {out}")
        except Exception as e:
            print(f"  ❌ Normal failed: {e}")

        # Test B: EN text + ref audio — auto-detect language?
        try:
            wav = model.generate(text=s["en_text"], reference_wav_path=s["ref_path"])
            out = os.path.join(OUTPUT_DIR, f"voxcpm_en_text_{s['idx']:03d}.wav")
            sf.write(out, wav, sr)
            print(f"  ✅ EN text (auto-detect): {out}")
            print(f"     → VoxCPM2 auto-detects language. Should speak English.")
        except Exception as e:
            print(f"  ❌ EN text failed: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print()


# ════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🧪 S2ST EXPERIMENT: Can voice cloning models translate?\n")
    print("Each model gets the SAME English reference audio.")
    print("We test: (A) normal FR text, (B) EN text to see if it translates.\n")

    # Run whichever models are installed
    test_omnivoice()
    test_xtts()
    test_voxcpm()

    print("🎉 Experiment complete!")
    print(f"   Listen to outputs in: {OUTPUT_DIR}/")
    print("   Compare *_normal_fr_* (baseline) vs *_en_text_* (translation test)")
