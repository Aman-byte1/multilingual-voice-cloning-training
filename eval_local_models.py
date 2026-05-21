#!/usr/bin/env python3
"""
eval_local_models.py
--------------------
Evaluate locally fine-tuned OmniVoice LoRA models against the base model.
Uses your dev.jsonl samples (which contain reference audio paths + target text).

Usage:
    python eval_local_models.py --lang fr
    python eval_local_models.py --lang ar
    python eval_local_models.py --lang zh
    python eval_local_models.py --lang all
"""

import os
import sys
import gc
import json
import types
import argparse
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from functools import partial

# ── Install flex_attention stub BEFORE importing omnivoice ──────────────────
def _install_flex_stub():
    mod_name = "torch.nn.attention.flex_attention"
    import sys
    import types
    import torch
    
    # 1. Try to import the existing module or create a stub if it doesn't exist
    if mod_name in sys.modules and sys.modules[mod_name] is not None:
        flex_mod = sys.modules[mod_name]
    else:
        try:
            import torch.nn.attention.flex_attention as flex_mod
        except ImportError:
            flex_mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = flex_mod
            try:
                import torch.nn.attention as attn_mod
                setattr(attn_mod, "flex_attention", flex_mod)
            except Exception:
                pass

    # 2. Guarantee that critical objects needed by transformers/omnivoice exist
    if not hasattr(flex_mod, "_DEFAULT_SPARSE_BLOCK_SIZE"):
        setattr(flex_mod, "_DEFAULT_SPARSE_BLOCK_SIZE", 128)
        
    if not hasattr(flex_mod, "create_block_mask"):
        def create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None,
                              _compile=False, device=None, **kw):
            seq_len = int(Q_LEN or KV_LEN or 1)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float32)
            mask.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(mask.dtype).min)
            return mask
        setattr(flex_mod, "create_block_mask", create_block_mask)

_install_flex_stub()

from omnivoice import OmniVoice


# ── Safe CUDA cleanup ──────────────────────────────────────────────────────

def _safe_cuda_cleanup():
    """Clean up GPU memory, tolerating deferred device-side asserts."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass  # async CUDA error — already printed by driver
    try:
        torch.cuda.empty_cache()
    except Exception:
        # If empty_cache fails, the CUDA context is corrupted.
        # Reset the device so subsequent models can still use the GPU.
        try:
            device_idx = torch.cuda.current_device()
            torch.cuda.device(device_idx)
            torch.cuda.empty_cache()
        except Exception:
            print("  ⚠ CUDA context unrecoverable — scoring will use CPU if needed")
    gc.collect()


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_dev_samples(lang: str, n: int = 30):
    """Load up to `n` samples from data/finetune_{lang}/dev.jsonl."""
    path = Path(f"./data/finetune_{lang}/dev.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Dev manifest not found: {path}")
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def load_ref_audio(audio_path: str, max_secs: float = 10.0):
    """Load reference audio, clip to max_secs, return (waveform, sr)."""
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:                          # stereo → mono
        wav = wav.mean(0, keepdim=True)
    max_samples = int(max_secs * sr)
    if wav.shape[-1] > max_samples:
        wav = wav[:, :max_samples]
    return wav, sr


def safe_tensor(audio_data) -> torch.Tensor:
    """Normalise whatever OmniVoice.generate() returns into a 2-D tensor."""
    if isinstance(audio_data, (list, tuple)):
        t = torch.from_numpy(np.array(audio_data))
    elif not isinstance(audio_data, torch.Tensor):
        t = torch.from_numpy(np.asarray(audio_data))
    else:
        t = audio_data
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t.cpu().float()


def extract_speaker_embedding(path: str, verifier, device: str):
    """Return ECAPA-TDNN speaker embedding for an audio file."""
    try:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        with torch.no_grad():
            emb = verifier.encode_batch(wav.to(device)).squeeze(0).squeeze(0)
        return emb
    except Exception as e:
        print(f"  ⚠ Speaker embedding failed for {path}: {e}")
        return None


def synthesise_set(model, samples, out_dir: Path, prefix: str, device: str, lang: str):
    """
    Generate one .wav per sample and return list of dicts:
      {path, text, ref_path}
    Skips samples that are already synthesised.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, s in enumerate(tqdm(samples, desc=f"  Synth [{prefix}]")):
        out_path = out_dir / f"{prefix}_{lang}_{i:04d}.wav"
        text = s.get("text") or s.get("normalized_text") or s.get("transcript", "")
        ref_audio_path = s.get("audio_path") or s.get("ref_audio") or s.get("audio", "")

        if not text or not ref_audio_path or not Path(ref_audio_path).exists():
            continue

        if not out_path.exists():
            try:
                ref_tuple = load_ref_audio(ref_audio_path)
                with torch.no_grad():
                    res = model.generate(
                        text=text,
                        ref_audio=ref_tuple,
                        temperature=0.8,
                        top_p=0.9,
                    )
                audio_data = res[0] if isinstance(res, tuple) else res
                sr_out = res[1] if isinstance(res, tuple) and len(res) > 1 else 24000
                torchaudio.save(str(out_path), safe_tensor(audio_data), sr_out)
            except Exception as e:
                print(f"  ⚠ Synthesis failed sample {i}: {e}")
                continue

        results.append({"path": str(out_path), "text": text, "ref_path": ref_audio_path})
    return results


def score_results(results, whisper, verifier, lang: str, device: str):
    """Compute average CER, WER, and speaker similarity."""
    import jiwer
    transforms = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

    cers, wers, sims = [], [], []
    for s in tqdm(results, desc="  Score"):
        # Speaker similarity
        ref_emb = extract_speaker_embedding(s["ref_path"], verifier, device)
        syn_emb = extract_speaker_embedding(s["path"],     verifier, device)
        if ref_emb is not None and syn_emb is not None:
            sim = float(F.cosine_similarity(ref_emb.unsqueeze(0), syn_emb.unsqueeze(0)).item())
        else:
            sim = 0.0
        sims.append(sim)

        # ASR → WER / CER
        try:
            segs, _ = whisper.transcribe(s["path"], language=lang)
            hyp = "".join(seg.text for seg in segs).strip()
            ref = transforms(s["text"])
            hyp = transforms(hyp)
            cers.append(jiwer.cer(ref, hyp))
            wers.append(jiwer.wer(ref, hyp))
        except Exception:
            cers.append(1.0)
            wers.append(1.0)

    return (
        float(np.mean(cers)) if cers else 1.0,
        float(np.mean(wers)) if wers else 1.0,
        float(np.mean(sims)) if sims else 0.0,
    )


def fix_merged_model_configs(lang: str):
    import shutil
    merged_dir = Path(f"./exp/omnivoice_finetuned_{lang}/merged_model")
    if not merged_dir.exists():
        return
    
    print(f"  🔧 Checking & repairing configuration files in {merged_dir}...")
    try:
        from huggingface_hub import snapshot_download
        base_path = Path(snapshot_download("k2-fsa/OmniVoice"))
        
        # Files/directories to copy from base model to merged model
        items_to_copy = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "audio_tokenizer"
        ]
        
        for item in items_to_copy:
            src = base_path / item
            dst = merged_dir / item
            if src.exists():
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        print(f"  ✅ Configuration files and tokenizer successfully repaired for {lang.upper()}!")
    except Exception as e:
        print(f"  ⚠ Failed to repair configs for {lang.upper()}: {e}")


def evaluate_language(lang: str, n_samples: int, device: str):
    merged_dir = Path(f"./exp/omnivoice_finetuned_{lang}/merged_model")
    if not merged_dir.exists():
        print(f"❌ Merged model not found at {merged_dir}. Did training finish?")
        return

    # Automatically fix and restore original configurations/tokenizer settings
    fix_merged_model_configs(lang)

    print(f"\n{'='*65}")
    print(f"  🌍 Evaluating language: {lang.upper()}")
    print(f"{'='*65}")

    samples = load_dev_samples(lang, n=n_samples)
    print(f"  📋 Loaded {len(samples)} dev samples")

    out_root = Path(f"./eval_output/{lang}")

    # ── 1. Base model ────────────────────────────────────────────
    print(f"\n  🔵 Loading BASE OmniVoice…")
    base_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    base_model.to(device).eval()

    base_results = synthesise_set(
        base_model, samples, out_root / "base", "base", device, lang
    )

    del base_model
    _safe_cuda_cleanup()

    # ── 2. Fine-tuned model ──────────────────────────────────────
    print(f"\n  🟢 Loading FINE-TUNED OmniVoice from {merged_dir}…")
    ft_model = OmniVoice.from_pretrained(str(merged_dir))
    ft_model.to(device).eval()

    ft_results = synthesise_set(
        ft_model, samples, out_root / "finetuned", "ft", device, lang
    )

    del ft_model
    _safe_cuda_cleanup()

    # ── 3. Load scoring models ───────────────────────────────────
    print(f"\n  📊 Loading Whisper + ECAPA-TDNN for scoring…")
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import SpeakerRecognition

    # Try GPU first; fall back to CPU if CUDA context was corrupted
    score_device = device
    try:
        torch.cuda.synchronize()
    except Exception:
        print("  ⚠ CUDA context corrupted after synthesis — scoring will use CPU")
        score_device = "cpu"

    try:
        whisper = WhisperModel(
            "large-v3", device=score_device,
            compute_type="float16" if score_device == "cuda" else "int8"
        )
    except Exception:
        print("  ⚠ Whisper failed on GPU — falling back to CPU")
        score_device = "cpu"
        whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")

    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": score_device},
    )

    # ── 4. Score ─────────────────────────────────────────────────
    print(f"\n  📈 Scoring BASE…")
    base_cer, base_wer, base_sim = score_results(base_results, whisper, verifier, lang, device)

    print(f"\n  📈 Scoring FINE-TUNED…")
    ft_cer, ft_wer, ft_sim = score_results(ft_results, whisper, verifier, lang, device)

    del whisper, verifier
    _safe_cuda_cleanup()

    # ── 5. Report ────────────────────────────────────────────────
    def pct_change(old, new, lower_is_better=True):
        if old == 0:
            return "N/A"
        if lower_is_better:
            return f"{((old - new) / old) * 100:+.1f}%"
        else:
            return f"{((new - old) / old) * 100:+.1f}%"

    W = 68
    FULL = {"zh": "CHINESE", "ar": "ARABIC", "fr": "FRENCH"}[lang]
    n = len(base_results)

    print(f"\n{'='*W}")
    print(f"🏆  {FULL} ({lang.upper()}) — {n} samples")
    print(f"{'='*W}")
    print(f"{'Metric':<14}| {'Base OmniVoice':<16}| {'Fine-Tuned LoRA':<16}| Δ")
    print(f"{'-'*W}")
    print(f"{'CER  (↓)':<14}| {base_cer:<16.4f}| {ft_cer:<16.4f}| {pct_change(base_cer, ft_cer, True)}")
    print(f"{'WER  (↓)':<14}| {base_wer:<16.4f}| {ft_wer:<16.4f}| {pct_change(base_wer, ft_wer, True)}")
    print(f"{'SIM  (↑)':<14}| {base_sim:<16.4f}| {ft_sim:<16.4f}| {pct_change(base_sim, ft_sim, False)}")
    print(f"{'='*W}")

    wins = sum([ft_cer < base_cer, ft_wer < base_wer, ft_sim > base_sim])
    if wins == 3:
        verdict = "✅ CLEAR UPGRADE — Fine-tuned model wins all 3 metrics!"
    elif wins >= 2:
        verdict = "✅ UPGRADE — Fine-tuned model wins 2/3 metrics."
    else:
        verdict = "⚠️  MIXED — trade-offs detected. Consider more training steps."
    print(f"\n{verdict}\n")

    # Save JSON report
    report = {
        "language": lang,
        "n_samples": n,
        "base":      {"cer": base_cer, "wer": base_wer, "sim": base_sim},
        "finetuned": {"cer": ft_cer,   "wer": ft_wer,   "sim": ft_sim},
    }
    report_path = Path(f"./eval_report_{lang}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  📄 Report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate local OmniVoice LoRA models vs base")
    parser.add_argument("--lang", default="all", choices=["fr", "ar", "zh", "all"],
                        help="Language to evaluate (default: all)")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of dev samples to evaluate per language (default: 30)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parallel", action="store_true",
                        help="Run evaluations in parallel across available GPUs (if --lang all)")
    parser.add_argument("--gpus", default="0,1,2",
                        help="Comma-separated list of GPU IDs to distribute processes over (default: 0,1,2)")
    args = parser.parse_args()

    langs = ["fr", "ar", "zh"] if args.lang == "all" else [args.lang]

    if args.lang == "all" and args.parallel and len(langs) > 1:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpu_list:
            gpu_list = ["0"]
        
        import subprocess
        processes = []
        log_files = []
        
        print(f"🚀 Starting parallel evaluation for {langs} on GPUs {gpu_list}...")
        
        for idx, lang in enumerate(langs):
            gpu_id = gpu_list[idx % len(gpu_list)]
            
            # Restrict visibility inside the child process using CUDA_VISIBLE_DEVICES
            child_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--lang", lang,
                "--n-samples", str(args.n_samples),
                "--device", "cuda"
            ]
            
            log_path = f"logs/eval_{lang}.log"
            os.makedirs("logs", exist_ok=True)
            log_file = open(log_path, "w", encoding="utf-8")
            log_files.append((lang, log_path, log_file))
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            print(f"  ➜ [{lang.upper()}] starting on GPU {gpu_id} (log: {log_path})")
            p = subprocess.Popen(child_cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((lang, p))
            
        print("\n⏳ Waiting for evaluations to complete...")
        for lang, p in processes:
            p.wait()
            
        for _, _, f in log_files:
            f.close()
            
        print("\n✅ Parallel evaluation processes finished!")
        print("📊 Summaries:")
        
        for lang, log_path, _ in log_files:
            report_path = f"eval_report_{lang}.json"
            if os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)
                base = report["base"]
                ft = report["finetuned"]
                
                def pct_change(old, new, lower_is_better=True):
                    if old == 0:
                        return "N/A"
                    if lower_is_better:
                        return f"{((old - new) / old) * 100:+.1f}%"
                    else:
                        return f"{((new - old) / old) * 100:+.1f}%"
                
                W = 68
                FULL = {"zh": "CHINESE", "ar": "ARABIC", "fr": "FRENCH"}[lang]
                n = report["n_samples"]
                print(f"\n{'='*W}")
                print(f"🏆  {FULL} ({lang.upper()}) — {n} samples")
                print(f"{'='*W}")
                print(f"{'Metric':<14}| {'Base OmniVoice':<16}| {'Fine-Tuned LoRA':<16}| Δ")
                print(f"{'-'*W}")
                print(f"{'CER  (↓)':<14}| {base['cer']:<16.4f}| {ft['cer']:<16.4f}| {pct_change(base['cer'], ft['cer'], True)}")
                print(f"{'WER  (↓)':<14}| {base['wer']:<16.4f}| {ft['wer']:<16.4f}| {pct_change(base['wer'], ft['wer'], True)}")
                print(f"{'SIM  (↑)':<14}| {base['sim']:<16.4f}| {ft['sim']:<16.4f}| {pct_change(base['sim'], ft['sim'], False)}")
                print(f"{'='*W}")
            else:
                print(f"\n❌ Could not find report for {lang.upper()}. Check {log_path} for errors.")
                try:
                    with open(log_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()[-15:]
                        print(f"--- Last 15 lines of {log_path} ---")
                        for line in lines:
                            print(f"  {line.rstrip()}")
                except Exception:
                    pass
    else:
        for lang in langs:
            evaluate_language(lang, args.n_samples, args.device)


if __name__ == "__main__":
    main()
