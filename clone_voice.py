import os
import sys
import torch
import torchaudio
import argparse
from omnivoice import OmniVoice

def main():
    parser = argparse.ArgumentParser(description="Clone a voice using OmniVoice")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="cloned_voice.wav", help="Output audio file path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()

    print(f"🚀 Loading OmniVoice model on {args.device}...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    model.to(args.device).eval()

    print(f"🎙️ Loading reference audio: {args.ref}")
    try:
        try:
            wav, sr = torchaudio.load(args.ref)
        except Exception as e:
            print(f"⚠️ Torchaudio failed, trying librosa... ({e})")
            import librosa
            wav_np, sr = librosa.load(args.ref, sr=None)
            wav = torch.from_numpy(wav_np).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error loading reference audio: {e}")
        return

    print(f"📈 Synthesizing text: {args.text}")
    with torch.no_grad():
        res = model.generate(
            text=args.text,
            ref_audio=(wav, sr),
            temperature=0.8,
            top_p=0.9
        )
        
    audio_data = res[0] if isinstance(res, tuple) else res
    
    # Ensure audio data is in the right format for saving
    if isinstance(audio_data, torch.Tensor):
        audio_tensor = audio_data
    else:
        # Convert list or numpy array to tensor
        import numpy as np
        audio_tensor = torch.from_numpy(np.array(audio_data))
        
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    audio_tensor = audio_tensor.cpu().float()

    print(f"💾 Saving to {args.output}...")
    torchaudio.save(args.output, audio_tensor, 24000)
    print("✅ Done!")

if __name__ == "__main__":
    main()
