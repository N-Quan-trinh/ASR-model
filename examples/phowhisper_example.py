#!/usr/bin/env python3
"""
Example usage of PhoWhisper ASR model.
"""

import torch
import torchaudio
from src.models.transformer_asr import PhoWhisperASR


def main():
    """Example of using PhoWhisper for Vietnamese speech recognition."""
    
    print("PhoWhisper ASR Example")
    print("=" * 40)
    
    # Initialize PhoWhisper model
    print("Loading PhoWhisper model...")
    model = PhoWhisperASR(
        model_name="vinai/phobert-base",  # Vietnamese PhoWhisper model
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example audio processing
    print("Model loaded successfully!")
    print("\nUsage example:")
    print("1. Load audio file:")
    print("   audio, sr = torchaudio.load('path/to/audio.wav')")
    print("2. Transcribe:")
    print("   result = model.transcribe(audio, sampling_rate=sr)")
    print("3. Get transcription:")
    print("   text = result['transcription'][0]")
    
    print("\nFor inference with command line:")
    print("python scripts/inference_phowhisper.py --audio path/to/audio.wav")
    
    print("\nAvailable PhoWhisper models:")
    print("- vinai/phobert-base (recommended for Vietnamese)")
    print("- openai/whisper-base (general purpose)")
    print("- openai/whisper-small")
    print("- openai/whisper-medium")
    print("- openai/whisper-large")


if __name__ == "__main__":
    main() 