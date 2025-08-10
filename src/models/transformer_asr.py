"""
Transformer-based ASR model with PhoWhisper integration.
"""

import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Optional, List, Dict, Any
import logging
import numpy as np


class TransformerASR(nn.Module):
    """Transformer-based ASR model with PhoWhisper support."""
    
    def __init__(self, model_type: str = "transformer", 
                 phowhisper_model: str = "VinAI/PhoWhisper-medium",
                 device: str = "auto"):
        """
        Initialize Transformer ASR model.
        
        Args:
            model_type: Type of model ("transformer" or "phowhisper")
            phowhisper_model: PhoWhisper model name from Hugging Face
            device: Device to run model on
        """
        super().__init__()
        
        self.model_type = model_type

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available. Falling back to CPU.")
                resolved_device = "cpu"
            elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                self.logger.warning("MPS requested but not available. Falling back to CPU.")
                resolved_device = "cpu"
            else:
                resolved_device = device
        self.device = resolved_device
        self.logger = logging.getLogger(__name__)
        
        if model_type == "phowhisper":
            self._load_phowhisper_model(phowhisper_model)
        else:
            self._load_custom_transformer()
    
    def _load_phowhisper_model(self, model_name: str):
        """Load PhoWhisper model from Hugging Face."""
        try:
            self.logger.info(f"Loading PhoWhisper model: {model_name}")
            
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Set language and task using forced decoder IDs (recommended for Whisper)
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="vi", task="transcribe"
            )
            
            self.logger.info("PhoWhisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading PhoWhisper model: {e}")
            raise
    
    def _load_custom_transformer(self):
        """Load custom transformer model (placeholder)."""
        self.logger.info("Loading custom transformer model")
        # TODO: Implement custom transformer architecture
        pass
    
    def transcribe(self, audio_input: torch.Tensor, 
                   sampling_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio using PhoWhisper.
        
        Args:
            audio_input: Audio tensor of shape (batch_size, samples)
            sampling_rate: Audio sampling rate
            
        Returns:
            Dictionary containing transcription and metadata
        """
        if self.model_type != "phowhisper":
            raise ValueError("Transcription only supported for PhoWhisper models")
        
        try:
            # Prepare input: Whisper expects list of 1D arrays (one per sample)
            if isinstance(audio_input, torch.Tensor):
                audio_np = audio_input.detach().cpu().numpy()
            elif isinstance(audio_input, np.ndarray):
                audio_np = audio_input
            elif isinstance(audio_input, list):
                audio_np = audio_input
            else:
                raise TypeError("audio_input must be Tensor, ndarray, or list")

            # Ensure batch of 1D arrays
            if isinstance(audio_np, list):
                batch_audio = [np.asarray(a).reshape(-1) for a in audio_np]
            else:
                if audio_np.ndim == 1:
                    batch_audio = [audio_np]
                elif audio_np.ndim == 2:
                    # Assume (batch, samples) or (channels, samples) already mono-resolved upstream
                    batch_audio = [row.reshape(-1) for row in audio_np]
                else:
                    raise ValueError(f"Unexpected audio tensor shape: {audio_np.shape}")

            inputs = self.processor(
                batch_audio,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                generated = self.model.generate(
                    inputs,
                    forced_decoder_ids=getattr(self, "forced_decoder_ids", None),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                predicted_ids = generated.sequences
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )
            
            return {
                "transcription": transcription,
                "model_type": self.model_type,
                "confidence": self._calculate_confidence(generated.scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            raise
    
    def _calculate_confidence(self, scores: List[torch.Tensor]) -> float:
        """
        Calculate confidence score for transcription.
        
        Args:
            scores: List of logits tensors per generated timestep
            
        Returns:
            Confidence score between 0 and 1
        """
        if not scores:
            return 0.0

        # scores is a list of length T, each tensor shape: (batch_size, vocab_size)
        with torch.no_grad():
            time_step_confidences: List[torch.Tensor] = []
            for step_logits in scores:
                step_probs = step_logits.softmax(dim=-1)  # (batch, vocab)
                step_max, _ = step_probs.max(dim=-1)      # (batch,)
                time_step_confidences.append(step_max)

            # Stack over time: (T, batch)
            stacked = torch.stack(time_step_confidences, dim=0)
            # Mean over time and batch
            confidence: torch.Tensor = stacked.mean()
            return float(confidence.clamp(0.0, 1.0).item())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for custom transformer (placeholder).
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.model_type == "phowhisper":
            raise NotImplementedError("Use transcribe() method for PhoWhisper models")
        
        # TODO: Implement custom transformer forward pass
        return x
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.model_type == "phowhisper":
            self.model.save_pretrained(path)
            self.processor.save_pretrained(path)
        else:
            torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        if self.model_type == "phowhisper":
            self.model = WhisperForConditionalGeneration.from_pretrained(path)
            self.processor = WhisperProcessor.from_pretrained(path)
            self.model = self.model.to(self.device)
        else:
            self.load_state_dict(torch.load(path))


class PhoWhisperASR(TransformerASR):
    """Convenience class for PhoWhisper ASR."""
    
    def __init__(self, model_name: str = "vinai/phowhisper-base", 
                 device: str = "auto"):
        """
        Initialize PhoWhisper ASR model.
        
        Args:
            model_name: PhoWhisper model name
            device: Device to run on
        """
        super().__init__(model_type="phowhisper", 
                        phowhisper_model=model_name, 
                        device=device) 