#!/usr/bin/env python3
"""
IWSLT 2026 — A/B Evaluation (any language)
Base OmniVoice vs. LoRA on Blind Test
Metrics: WER, CER, Speaker Similarity
Sample: 25 segments × 12 speakers = 300 pairs
Usage:
    python3 evaluate_lang.py --lang zh
    python3 evaluate_lang.py --lang ar
    python3 evaluate_lang.py --lang fr
"""
import os, sys, gc, json, types, argparse
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import jiwer
import numpy as np
from functools import partial

# ── flex_attention stub (for older PyTorch) ─────────────────────
def _install_flex_stub():
    mod_name = "torch.nn.attention.flex_attention"
    if mod_name in sys.modules:
        return
    try:
        import torch.nn.attention as attn_mod
        stub = types.ModuleType(mod_name)
        def create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None,
                              _compile=False, device=None, **kw):
            seq_len = int(Q_LEN or KV_LEN or 1)
            if isinstance(mask_mod, partial) and mask_mod.args:
                d = mask_mod.args[0]
                if torch.is_tensor(d):
                    seq_len = int(d.numel())
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float32)
            mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(mask.dtype).min)
            return mask
        stub.create_block_mask = create_block_mask
        sys.modules[mod_name] = stub
        setattr(attn_mod, "flex_attention", stub)
    except Exception:
        pass

_install_flex_stub()

# ── config per language ─────────────────────────────────────────
LANG_CONFIG = {
    "zh": {"text_file": "blind_test/text/chinese.txt", "lora": "Step-400"},
    "ar": {"text_file": "blind_test/text/arabic.txt",  "lora": "Step-400"},
    "fr": {"text_file": "blind_test/text/french.txt",  "lora": "Step-200"},
}

# ── helpers ─────────────────────────────────────────────────────
def extract_speaker_embedding(path, verifier, device="cuda"):
    try:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            emb = verifier.encode_batch(wav.to(device)).squeeze(0).squeeze(0)
        return emb
    except Exception:
        return None

def safe_tensor(audio_data):
    if isinstance(audio_data, (list, tuple)):
        t = torch.from_numpy(np.array(audio_data))
    elif not isinstance(audio_data, torch.Tensor):
        t = torch.from_numpy(audio_data)
    else:
        t = audio_data
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t.cpu().float()

# ── main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["zh", "ar", "fr"])
    parser.add_argument("--n-segments", type=int, default=25)
    args = parser.parse_args()

    LANG       = args.lang
    N_SEG      = args.n_segments
    cfg        = LANG_CONFIG[LANG]
    OUT_DIR    = f"temp_submission/{LANG}"
    TEXT_FILE  = cfg["text_file"]
    LORA_LABEL = cfg["lora"]
    REPORT     = f"eval_results_{LANG}_ab.json"

    if not os.path.exists(OUT_DIR):
        print(f"❌ {OUT_DIR} not found. Run generate_submission.py --lang {LANG} first.")
        return

    # 1. Ground-truth text
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text_lines = [l.strip() for l in f if l.strip()][:N_SEG]

    # 2. Discover speakers
    all_files = os.listdir(OUT_DIR)
    speakers = sorted(set(
        "_".join(fn.split("_")[2:]).replace(".wav", "")
        for fn in all_files
        if fn.startswith(f"{LANG}_") and fn.endswith(".wav") and not fn.startswith("_")
    ))

    eval_samples = []
    for spk in speakers:
        for i in range(1, N_SEG + 1):
            fname = f"{LANG}_{i:03d}_{spk}.wav"
            if os.path.exists(os.path.join(OUT_DIR, fname)):
                eval_samples.append({"file": fname, "speaker": spk, "idx": i - 1})

    print(f"🔍  {LANG.upper()}: {len(eval_samples)} samples ({N_SEG} segs × {len(speakers)} voices)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Generate BASE OmniVoice counterparts
    print(f"\n🚀 Loading BASE OmniVoice…")
    from omnivoice import OmniVoice
    base_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    base_model.to(device).eval()

    base_dir = f"eval_base_{LANG}_samples"
    os.makedirs(base_dir, exist_ok=True)
    base_map = {}

    print("🎙️  Generating base-model audio…")
    for spk in tqdm(speakers, desc="Speakers"):
        ref_path = os.path.join(OUT_DIR, f"_extracted_reference_{spk}.wav")
        if not os.path.exists(ref_path):
            ref_path = f"blind_test/audio/{spk}.wav"
        wav, sr_in = torchaudio.load(ref_path)
        ref_tuple = (wav, sr_in)

        for i in range(N_SEG):
            out_path = os.path.join(base_dir, f"base_{i}_{spk}.wav")
            base_map[(spk, i)] = out_path
            if os.path.exists(out_path):
                continue
            with torch.no_grad():
                res = base_model.generate(text=text_lines[i], ref_audio=ref_tuple,
                                          temperature=0.8, top_p=0.9)
                audio_data = res[0] if isinstance(res, tuple) else res
                torchaudio.save(out_path, safe_tensor(audio_data), 24000)

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Load eval models
    print("\n🔎 Loading Whisper-large-v3 + ECAPA-TDNN…")
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition

    whisper = WhisperModel("large-v3", device=device,
                           compute_type="float16" if device == "cuda" else "int8")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device},
    )

    # Pre-cache ref embeddings
    ref_embs = {}
    for spk in speakers:
        ref_path = os.path.join(OUT_DIR, f"_extracted_reference_{spk}.wav")
        if not os.path.exists(ref_path):
            ref_path = f"blind_test/audio/{spk}.wav"
        ref_embs[spk] = extract_speaker_embedding(ref_path, verifier, device)

    # 5. Score
    transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

    def score_set(file_list, label):
        results = []
        for s in tqdm(file_list, desc=f"Eval ({label})"):
            path = s["path"]
            ref_emb = ref_embs.get(s["speaker"])

            emb = extract_speaker_embedding(path, verifier, device)
            sim = (float(F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item())
                   if emb is not None and ref_emb is not None else 0.0)
            try:
                segs, _ = whisper.transcribe(path, language=LANG)
                hyp = "".join([seg.text for seg in segs])
                ref_text = text_lines[s["idx"]]
                cer = jiwer.cer(transforms(ref_text), transforms(hyp))
                wer = jiwer.wer(transforms(ref_text), transforms(hyp))
            except Exception:
                cer, wer = 1.0, 1.0
            results.append({"sim": sim, "cer": cer, "wer": wer})

        avg = lambda k: float(np.mean([r[k] for r in results]))
        return avg("cer"), avg("wer"), avg("sim")

    lora_list = [{"path": os.path.join(OUT_DIR, s["file"]), "speaker": s["speaker"], "idx": s["idx"]} for s in eval_samples]
    base_list = [{"path": base_map[(s["speaker"], s["idx"])], "speaker": s["speaker"], "idx": s["idx"]} for s in eval_samples]

    lora_cer, lora_wer, lora_sim = score_set(lora_list, f"LoRA {LORA_LABEL}")
    base_cer, base_wer, base_sim = score_set(base_list, "Base")

    # 6. Report
    def pct(b, n, lower=True):
        if b == 0: return "+0.0%"
        d = ((b - n) / b) * 100 if lower else ((n - b) / b) * 100
        return f"{d:+.1f}%"

    FULL = {"zh": "CHINESE", "ar": "ARABIC", "fr": "FRENCH"}[LANG]
    W = 70
    print("\n" + "=" * W)
    print(f"🏆  {FULL} ({LANG.upper()}) A/B TEST — {len(eval_samples)} samples")
    print("=" * W)
    print(f"{'Metric':<16}| {'Base OmniVoice':<16}| {'LoRA ' + LORA_LABEL:<16}| {'Δ Improvement'}")
    print("-" * W)
    print(f"{'CER  (↓)':<16}| {base_cer:<16.4f}| {lora_cer:<16.4f}| {pct(base_cer, lora_cer)}")
    print(f"{'WER  (↓)':<16}| {base_wer:<16.4f}| {lora_wer:<16.4f}| {pct(base_wer, lora_wer)}")
    print(f"{'SIM  (↑)':<16}| {base_sim:<16.4f}| {lora_sim:<16.4f}| {pct(base_sim, lora_sim, False)}")
    print("=" * W)

    wins = sum([lora_cer < base_cer, lora_wer < base_wer, lora_sim > base_sim])
    if wins == 3:   print("\n✅ CLEAR UPGRADE — LoRA wins all 3 metrics.")
    elif wins >= 2: print("\n✅ UPGRADE — LoRA wins 2/3 metrics.")
    else:           print("\n⚠️  MIXED — trade-offs detected.")

    report = {
        "language": LANG,
        "n_samples": len(eval_samples),
        "base": {"cer": base_cer, "wer": base_wer, "sim": base_sim},
        "lora": {"cer": lora_cer, "wer": lora_wer, "sim": lora_sim},
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Saved → {REPORT}")


if __name__ == "__main__":
    main()
