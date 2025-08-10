#!/usr/bin/env python3
"""
Minimal inference script for PhoWhisper ASR.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torchaudio

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer_asr import PhoWhisperASR


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio robustly and return mono waveform at target_sr.
    Tries torchaudio, then librosa, then pydub.
    """
    # Try torchaudio first
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        # Fallback to librosa
        try:
            import numpy as np
            import librosa

            y, sr = librosa.load(path, sr=None, mono=False)
            if y.ndim == 1:
                wav = torch.from_numpy(y).unsqueeze(0)
            else:
                wav = torch.from_numpy(y)
        except Exception:
            # Fallback to pydub (requires ffmpeg)
            from pydub import AudioSegment
            import numpy as np

            seg = AudioSegment.from_file(path)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples()).astype("float32")
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels)).T  # (channels, time)
            else:
                samples = samples[None, :]
            wav = torch.from_numpy(samples)

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr

    # Convert to mono if stereo
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="VinAI/PhoWhisper-medium")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--timestamps", action="store_true", help="Produce timestamped transcript using chunked decoding")
    ap.add_argument("--chunk-sec", type=float, default=30.0, help="Chunk length in seconds for timestamped mode")
    ap.add_argument("--overlap-sec", type=float, default=0.5, help="Overlap between chunks in seconds")
    ap.add_argument("--txt-out", type=str, default=None, help="Path to save formatted TXT output")
    ap.add_argument("--srt-out", type=str, default=None, help="Path to save SRT subtitle file")
    args = ap.parse_args()

    audio = load_audio(args.audio)
    asr = PhoWhisperASR(model_name=args.model, device=args.device)
    if not args.timestamps:
        out = asr.transcribe(audio, sampling_rate=16000)
        text = out["transcription"][0]
        print(text)
        print(f"confidence={out['confidence']:.3f}")
        if args.txt_out:
            Path(args.txt_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.txt_out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
                f.write(f"confidence={out['confidence']:.3f}\n")
        return

    # Timestamped, chunked transcription
    sr = 16000
    total_samples = int(audio.shape[-1])
    chunk_samples = int(args.chunk_sec * sr)
    step_samples = max(1, int(max(args.chunk_sec - args.overlap_sec, 0.01) * sr))
    segments: List[Dict] = []

    start = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        seg_audio = audio[start:end]
        if seg_audio.numel() == 0:
            break
        seg_out = asr.transcribe(seg_audio, sampling_rate=sr)
        seg_text = seg_out["transcription"][0].strip()
        segments.append({
            "start_sec": start / sr,
            "end_sec": end / sr,
            "text": seg_text,
            "confidence": seg_out["confidence"],
        })
        if end == total_samples:
            break
        start += step_samples

    def fmt_hhmmss(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def fmt_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # Print formatted timestamped transcript
    for seg in segments:
        print(f"[{fmt_hhmmss(seg['start_sec'])} - {fmt_hhmmss(seg['end_sec'])}] {seg['text']}")

    # Optional TXT output
    if args.txt_out:
        Path(args.txt_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.txt_out, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{fmt_hhmmss(seg['start_sec'])} - {fmt_hhmmss(seg['end_sec'])}] {seg['text']}\n")

    # Optional SRT output
    if args.srt_out:
        Path(args.srt_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.srt_out, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                f.write(f"{i}\n")
                f.write(f"{fmt_srt_time(seg['start_sec'])} --> {fmt_srt_time(seg['end_sec'])}\n")
                f.write(seg["text"] + "\n\n")


if __name__ == "__main__":
    main()


