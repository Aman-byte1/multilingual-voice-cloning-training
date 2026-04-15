#!/usr/bin/env python3
"""
Launch VoxCPM LoRA fine-tuning using upstream OpenBMB training script.

This script writes a YAML config compatible with:
  scripts/train_voxcpm_finetune.py --config_path <yaml>
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def detect_sample_rates(pretrained_path: str) -> tuple[int, int]:
    cfg_path = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found under pretrained model path: {pretrained_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    audio_vae_cfg = cfg.get("audio_vae_config", {})
    sample_rate = int(audio_vae_cfg.get("sample_rate", 16000))
    out_sample_rate = int(audio_vae_cfg.get("out_sample_rate", 0))
    return sample_rate, out_sample_rate


def ensure_pretrained_path(pretrained_path: str, hf_model_id: str) -> str:
    if pretrained_path:
        p = os.path.abspath(os.path.expanduser(pretrained_path))
        if not os.path.isdir(p):
            raise FileNotFoundError(f"--pretrained-path does not exist: {p}")
        return p

    print(f"Downloading model snapshot for {hf_model_id} ...")
    local_dir = snapshot_download(repo_id=hf_model_id, repo_type="model")
    return local_dir


def ensure_python_dependencies(voxcpm_root: str) -> None:
    # Train script requires these modules; install only when missing.
    required = ["argbind", "tensorboardX", "yaml"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        return

    print(f"Installing missing Python deps: {missing}")
    req_file = os.path.join(voxcpm_root, "requirements.txt")
    if os.path.isfile(req_file):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)

    # Ensure core runtime deps are present even if not pinned in requirements.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "argbind", "tensorboardX", "safetensors"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch VoxCPM LoRA fine-tuning")
    parser.add_argument("--voxcpm-root", default="./third_party/VoxCPM")
    parser.add_argument("--voxcpm-repo", default="https://github.com/OpenBMB/VoxCPM.git")
    parser.add_argument("--no-auto-clone", action="store_true")
    parser.add_argument("--pretrained-path", default="", help="Local VoxCPM model directory")
    parser.add_argument("--hf-model-id", default="openbmb/VoxCPM2")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default="")
    parser.add_argument("--save-path", default="./exp/voxcpm_finetuned_zh")
    parser.add_argument("--tensorboard", default="")
    parser.add_argument("--num-iters", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--valid-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--disable-lm", action="store_true")
    parser.add_argument("--disable-dit", action="store_true")
    parser.add_argument("--enable-proj", action="store_true")
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--distribute", action="store_true")
    args = parser.parse_args()

    voxcpm_root = os.path.abspath(os.path.expanduser(args.voxcpm_root))

    if not os.path.isdir(voxcpm_root):
        if args.no_auto_clone:
            raise FileNotFoundError(
                f"VoxCPM root does not exist: {voxcpm_root}. "
                "Create it manually or remove --no-auto-clone."
            )
        os.makedirs(os.path.dirname(voxcpm_root), exist_ok=True)
        print(f"Cloning VoxCPM into {voxcpm_root} ...")
        subprocess.run(["git", "clone", args.voxcpm_repo, voxcpm_root], check=True)

    train_script = os.path.join(voxcpm_root, "scripts", "train_voxcpm_finetune.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError(
            f"VoxCPM training script not found: {train_script}. "
            f"Clone OpenBMB/VoxCPM into --voxcpm-root first."
        )

    ensure_python_dependencies(voxcpm_root)

    train_manifest = os.path.abspath(os.path.expanduser(args.train_manifest))
    if not os.path.isfile(train_manifest):
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")

    val_manifest = os.path.abspath(os.path.expanduser(args.val_manifest)) if args.val_manifest else ""
    if val_manifest and not os.path.isfile(val_manifest):
        raise FileNotFoundError(f"Val manifest not found: {val_manifest}")

    pretrained_path = ensure_pretrained_path(args.pretrained_path, args.hf_model_id)
    sample_rate, out_sample_rate = detect_sample_rates(pretrained_path)

    save_path = os.path.abspath(os.path.expanduser(args.save_path))
    os.makedirs(save_path, exist_ok=True)

    tb_path = args.tensorboard.strip() or os.path.join(save_path, "logs")

    config = {
        "pretrained_path": pretrained_path,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "sample_rate": sample_rate,
        "out_sample_rate": out_sample_rate,
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "num_workers": int(args.num_workers),
        "num_iters": int(args.num_iters),
        "log_interval": int(args.log_interval),
        "valid_interval": int(args.valid_interval),
        "save_interval": int(args.save_interval),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "max_steps": int(args.max_steps),
        "max_batch_tokens": int(args.max_batch_tokens),
        "save_path": save_path,
        "tensorboard": tb_path,
        "max_grad_norm": float(args.max_grad_norm),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": not args.disable_lm,
            "enable_dit": not args.disable_dit,
            "enable_proj": bool(args.enable_proj),
            "r": int(args.lora_rank),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        "hf_model_id": args.hf_model_id,
        "distribute": bool(args.distribute),
    }

    cfg_path = os.path.join(save_path, "voxcpm_lora_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)

    cmd = [sys.executable, train_script, "--config_path", cfg_path]

    print("Running VoxCPM training with config:")
    print(cfg_path)
    print("Command:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=voxcpm_root, check=True)


if __name__ == "__main__":
    main()
