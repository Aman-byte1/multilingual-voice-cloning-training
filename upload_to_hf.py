import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def upload_models(hf_token: str, username: str, exp_dir: str = "./exp"):
    # Authenticate with the token
    login(token=hf_token)
    api = HfApi()

    # Find the fine-tuned model directories
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"❌ Could not find {exp_dir} directory.")
        return

    langs = ["fr", "ar", "zh"]
    
    for lang in langs:
        model_dir = exp_path / f"omnivoice_finetuned_{lang}"
        if not model_dir.exists():
            print(f"⚠ Skipping {lang} - directory not found: {model_dir}")
            continue

        repo_id = f"{username}/omnivoice_finetuned_{lang}"
        print(f"\n🚀 Creating repository: {repo_id}")
        
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        except Exception as e:
            print(f"⚠ Could not create repo {repo_id}: {e}")
        
        print(f"📦 Uploading {lang} model to {repo_id}...")
        try:
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {lang} fine-tuned OmniVoice model"
            )
            print(f"✅ Successfully uploaded {lang} model to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"❌ Failed to upload {lang} model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned models to Hugging Face")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face User Access Token")
    parser.add_argument("--username", type=str, required=True, help="Hugging Face Username")
    parser.add_argument("--exp-dir", type=str, default="./exp", help="Directory containing the finetuned models")
    
    args = parser.parse_args()
    upload_models(args.token, args.username, args.exp_dir)
