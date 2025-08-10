"""
Minimal PhoWhisper ASR wrapper using Hugging Face Whisper.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class PhoWhisperASR(nn.Module):
    """Simple PhoWhisper-based ASR model.

    Usage:
        asr = PhoWhisperASR(model_name="VinAI/PhoWhisper-medium", device="auto")
        out = asr.transcribe(audio_tensor, sampling_rate=16000)
    """

    def __init__(self, model_name: str = "VinAI/PhoWhisper-medium", device: str = "auto") -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.device = self._resolve_device(device)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="vi", task="transcribe")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sampling_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio.

        Args:
            audio: Tensor shape (samples,) or (batch, samples)
            sampling_rate: input SR (Hz)
        Returns: dict with transcription and confidence
        """
        if isinstance(audio, torch.Tensor):
            arr = audio.detach().cpu().numpy()
        else:
            arr = np.asarray(audio)

        if arr.ndim == 1:
            batch_audio = [arr]
        elif arr.ndim == 2:
            batch_audio = [row for row in arr]
        else:
            raise ValueError(f"Unexpected audio shape: {arr.shape}")

        inputs = self.processor(batch_audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)

        generated = self.model.generate(
            inputs,
            forced_decoder_ids=self.forced_decoder_ids,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = generated.sequences
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        confidence = self._estimate_confidence(generated.scores)
        return {"transcription": texts, "confidence": confidence}

    def _estimate_confidence(self, scores: List[torch.Tensor]) -> float:
        if not scores:
            return 0.0
        vals: List[torch.Tensor] = []
        for step in scores:
            probs = step.softmax(dim=-1)
            vals.append(probs.max(dim=-1)[0])
        stacked = torch.stack(vals, dim=0)
        return float(stacked.mean().clamp(0.0, 1.0).item())


