#!/usr/bin/env python3
"""
Inference script for PhoWhisper ASR model.
"""

import argparse
import torch
import torchaudio
import logging
from pathlib import Path
import json

from src.models.transformer_asr import PhoWhisperASR


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sampling rate
        
    Returns:
        Audio tensor
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze()
        
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='PhoWhisper ASR Inference')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file')
    parser.add_argument('--model', type=str, default='vinai/phowhisper-base',
                       help='PhoWhisper model name')
    parser.add_argument('--device', type=str, default='auto',
                       help="Device to run on: 'auto'|'cpu'|'cuda'")
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (JSON)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if audio file exists
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        return
    
    # Resolve device
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_str = 'cpu'
        else:
            device_str = args.device
    device = torch.device(device_str)
    logger.info(f"Using device: {device_str}")
    
    try:
        # Load PhoWhisper model
        logger.info(f"Loading PhoWhisper model: {args.model}")
        model = PhoWhisperASR(model_name=args.model, device=device_str)
        
        # Load audio
        logger.info(f"Loading audio file: {args.audio}")
        audio = load_audio(args.audio)
        
        # Transcribe
        logger.info("Transcribing audio...")
        result = model.transcribe(audio.unsqueeze(0), sampling_rate=16000)
        
        # Print results
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULTS")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Audio: {args.audio}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Transcription: {result['transcription'][0]}")
        print("="*50)
        
        # Save to file if requested
        if args.output:
            output_data = {
                "audio_file": args.audio,
                "model": args.model,
                "transcription": result['transcription'][0],
                "confidence": result['confidence'],
                "model_type": result['model_type']
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == '__main__':
    main() 