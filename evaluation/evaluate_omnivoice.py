#!/usr/bin/env python3
"""
Cross-Lingual Voice Cloning Evaluation Pipeline — OmniVoice
=============================================================
Zero-shot OmniVoice evaluation on ymoslem/acl-6060.
Supports 600+ languages including Arabic, Chinese, French.
Metrics: WER, CER, Speaker Similarity, Inference Time, RTF.

ASR: faster-whisper large-v3
"""

import os
import sys
import csv
import json
import time
import argparse
import warnings
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

sys.path.insert(0, os.path.dirname(__file__))

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")

# ===================================================================
# Helper functions
# ===================================================================

def save_temp_wav(audio_array: np.ndarray, sr: int, prefix: str = "eval", output_dir: str = None) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{prefix}.wav")
    else:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
        os.close(fd)

    wav_t = torch.from_numpy(audio_array).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    torchaudio.save(path, wav_t, sr)
    return path


def load_speaker_model(device="cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain_spkrec"),
        run_opts={"device": device}
    )


def extract_speaker_embedding(wav_path, model, device="cuda"):
    """Extract speaker embedding, resampling to 16kHz (ECAPA-TDNN requirement)."""
    try:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0).to(device)  # mono
        emb = model.encode_batch(wav.unsqueeze(0))
        return emb.squeeze(0).squeeze(0).detach()
    except Exception:
        return None


def safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.mean(v) if v else np.nan

def safe_std(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.std(v) if v else np.nan

def safe_count(vals):
    return len([x for x in vals if x is not None and not np.isnan(x)])


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Lingual Voice Cloning Evaluation — OmniVoice")
    parser.add_argument("--dataset", default="ymoslem/acl-6060")
    parser.add_argument("--model-name", default="k2-fsa/OmniVoice")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-beam", type=int, default=5)
    parser.add_argument("--whisper-lang", default="fr", help="Language code for ASR")
    parser.add_argument("--output-dir", default="./eval_results_omnivoice")
    parser.add_argument("--cache-dir", default="./data_cache")
    parser.add_argument("--resume", action="store_true", help="Skip if audio exists")
    parser.add_argument("--use-ref-text", action="store_true",
                        help="Pass English ref transcript to model for transcript-assisted cloning")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    target_lang = args.whisper_lang.strip().lower()

    ref_text_mode = "WITH ref_text" if args.use_ref_text else "NO ref_text"
    print("=" * 64)
    print(f"  OmniVoice EVALUATION ({target_lang}) [{ref_text_mode}]")
    print("=" * 64)

    temp_ref_dir = os.path.join(args.output_dir, "temp_ref")
    os.makedirs(temp_ref_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # PHASE 1: Load dataset and pre-extract to disk
    # ---------------------------------------------------------------
    print(f"\n📥  Phase 1: Loading dataset and extracting to disk")
    ds_test = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total = len(ds_test)
    if args.max_samples and args.max_samples < total:
        step = max(1, total // args.max_samples)
        indices = list(range(0, total, step))[:args.max_samples]
        ds_test = ds_test.select(indices)
    total = len(ds_test)
    print(f"   Samples: {total}")

    manifest = []
    for i in tqdm(range(total), desc="Extracting"):
        row = ds_test[i]
        text_target = (row.get(f"trg_{target_lang}_text") or
                       row.get(f"text_{target_lang}") or "").strip()
        ref_data = (row.get("ref_en_voice") or
                    row.get(f"ref_{target_lang}_voice") or
                    row.get("audio_en") or
                    row.get("audio"))
        # Extract English ref transcript for ref_text mode
        ref_en_text = (row.get("ref_en_text") or
                       row.get("text_en") or "").strip()

        if not ref_data or not text_target:
            continue

        ref_path = save_temp_wav(
            np.asarray(ref_data["array"], dtype=np.float32),
            ref_data["sampling_rate"],
            f"ref_{i:05d}",
            output_dir=temp_ref_dir
        )
        manifest.append({
            "idx": i,
            "text_target": text_target,
            "ref_path": ref_path,
            "ref_en_text": ref_en_text,
            "speaker_id": row.get("speaker_id", "unknown"),
        })

    del ds_test
    gc.collect()
    print(f"   Extracted {len(manifest)} samples. Dataset freed from RAM.")

    # ---------------------------------------------------------------
    # PHASE 2: Load OmniVoice
    # ---------------------------------------------------------------
    print(f"\n🔧  Phase 2: Loading OmniVoice model")
    from omnivoice import OmniVoice

    # Load the base model first
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=f"{device}:0",
        dtype=torch.float16
    )
    
    if args.model_name != "k2-fsa/OmniVoice":
        print(f"   Injecting PEFT adapters from {args.model_name}")
        from peft import PeftModel
        # Wrap Qwen3 inside OmniVoice with our trained LoRA
        model.llm = PeftModel.from_pretrained(model.llm, args.model_name)
    
    print("   OmniVoice loaded ✓")

    OMNI_SR = 24000  # OmniVoice outputs at 24kHz

    # ---------------------------------------------------------------
    # PHASE 3: Generate
    # ---------------------------------------------------------------
    print(f"\n🎙  Phase 3: Generating {len(manifest)} samples")
    samples = []
    skipped = 0

    for entry in tqdm(manifest, desc="Generating"):
        i = entry["idx"]
        syn_path = os.path.join(args.output_dir, f"synth_{i:05d}.wav")
        ref_path = entry["ref_path"]
        text_target = entry["text_target"]

        if args.resume and os.path.exists(syn_path):
            try:
                wav_info = torchaudio.info(syn_path)
                audio_dur = wav_info.num_frames / wav_info.sample_rate
                samples.append({
                    "idx": i, "syn_path": syn_path, "ref_path": ref_path,
                    "text_target": text_target, "speaker_id": entry["speaker_id"],
                    "inference_s": 0, "audio_dur_s": audio_dur, "rtf": 0
                })
                continue
            except Exception:
                pass

        try:
            t0 = time.perf_counter()
            gen_kwargs = {
                "text": text_target,
                "ref_audio": ref_path,
            }
            if args.use_ref_text and entry.get("ref_en_text"):
                gen_kwargs["ref_text"] = entry["ref_en_text"]
            audio_out = model.generate(**gen_kwargs)
            t1 = time.perf_counter()

            if isinstance(audio_out, list):
                wav_tensor = audio_out[0]
            else:
                wav_tensor = audio_out

            import numpy as np
            if isinstance(wav_tensor, np.ndarray):
                wav_tensor = torch.from_numpy(wav_tensor)
                
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            torchaudio.save(syn_path, wav_tensor.cpu(), OMNI_SR)

        except Exception as e:
            print(f"   ⚠ Sample {i} generation failed: {e}")
            skipped += 1
            continue

        elapsed = t1 - t0
        audio_dur = wav_tensor.shape[-1] / OMNI_SR

        samples.append({
            "idx": i,
            "syn_path": syn_path,
            "ref_path": ref_path,
            "text_target": text_target,
            "speaker_id": entry["speaker_id"],
            "inference_s": elapsed,
            "audio_dur_s": audio_dur,
            "rtf": elapsed / audio_dur if audio_dur > 0 else 0
        })

        if i % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"   Generated: {len(samples)} | Skipped: {skipped}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # PHASE 4: ASR Transcription
    # ---------------------------------------------------------------
    print(f"\n🗣  Phase 4: Transcribing ({target_lang})")
    from faster_whisper import WhisperModel as FasterWhisperModel
    whisper = FasterWhisperModel(
        args.whisper_model, device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    transcripts = []
    for s in tqdm(samples, desc="Transcribing"):
        try:
            segments, _ = whisper.transcribe(
                s["syn_path"], language=target_lang,
                beam_size=args.whisper_beam, vad_filter=True
            )
            transcripts.append(" ".join(seg.text for seg in segments).strip())
        except Exception:
            transcripts.append("")

    del whisper
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # PHASE 5: Metrics
    # ---------------------------------------------------------------
    print(f"\n📊  Phase 5: Computing Metrics")
    verifier = load_speaker_model(device=device)
    import jiwer

    if target_lang in ("zh", "ar", "ja", "ko"):
        wer_transforms = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()
        ])
    else:
        wer_transforms = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(), jiwer.RemovePunctuation()
        ])

    results = []
    for s, tx in tqdm(zip(samples, transcripts), total=len(samples), desc="Metrics"):
        syn_emb = extract_speaker_embedding(s["syn_path"], verifier, device)
        ref_emb = extract_speaker_embedding(s["ref_path"], verifier, device)
        sim = float(F.cosine_similarity(
            syn_emb.unsqueeze(0), ref_emb.unsqueeze(0)
        ).item()) if (syn_emb is not None and ref_emb is not None) else None

        try:
            if tx.strip():
                ref_clean = wer_transforms(s["text_target"])
                hyp_clean = wer_transforms(tx)
                w = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
                c = float(jiwer.cer(ref_clean, hyp_clean)) if ref_clean.strip() else 1.0
            else:
                w = c = 1.0
        except Exception:
            w = c = None

        # Clean up temp ref
        if os.path.exists(s["ref_path"]):
            os.remove(s["ref_path"])

        results.append({
            "idx": s["idx"], "WER": w, "CER": c, "Similarity": sim,
            "InferenceS": s["inference_s"], "AudioDurS": s["audio_dur_s"],
            "RTF": s["rtf"]
        })

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    metric_keys = ["WER", "CER", "Similarity", "InferenceS", "RTF"]
    overall = {}
    for k in metric_keys:
        overall[k] = {
            "mean": safe_mean([r[k] for r in results]),
            "std": safe_std([r[k] for r in results]),
            "valid": safe_count([r[k] for r in results])
        }

    print("\n" + "=" * 62)
    print("  OmniVoice EVALUATION COMPLETE")
    print(f"  Target: {target_lang} | Samples: {len(results)}")
    print("=" * 62)
    print(f"  {'Metric':<16} {'Mean':>9} {'± Std':>9}  {'Valid':>6}")
    print("-" * 62)
    for k in metric_keys:
        m = overall[k]["mean"]
        s = overall[k]["std"]
        v = overall[k]["valid"]
        print(f"  {k:<16} {m:>9.4f} {f'±{s:.4f}' if not np.isnan(s) else '':>9}  {v:>3}/{len(results)}")
    print("=" * 62)

    with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    csv_path = os.path.join(args.output_dir, "eval_per_sample.csv")
    print(f"\n  Per-sample results saved to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "WER", "CER", "Similarity", "InferenceS", "AudioDurS", "RTF"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Cleanup temp dir
    if os.path.exists(temp_ref_dir):
        import shutil
        shutil.rmtree(temp_ref_dir)


if __name__ == "__main__":
    main()
