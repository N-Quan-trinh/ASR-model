## ASR model — PhoWhisper Inference

Minimal, ready-to-run Vietnamese ASR using the VinAI PhoWhisper model (Hugging Face Whisper). Includes robust audio loading (m4a, wav, mp3 via torchaudio/librosa/pydub), optional timestamped chunked transcription, and SRT export.

### 1) Setup

- Python (3.10–3.12 recommended)

Option A — pip
```bash
pip install -r "requirements.txt"
```

Option B — conda (optional)
```bash
conda env create -f environment.yml
conda activate asr_whisper
```

Notes
- First run will download the PhoWhisper model (several GB). Ensure a stable connection and free disk space.
- On Apple Silicon, `--device auto` will use MPS if available, otherwise CPU.

### 2) Basic transcription

Run directly on your raw audio (m4a, wav, mp3, ...). The script converts to 16 kHz mono on the fly.
```bash
python3 scripts/inference_phowhisper.py \
  --audio "data/raw/your_audio.m4a" \
  --model VinAI/PhoWhisper-medium \
  --device auto
```
Output
- Prints the transcript to stdout
- Prints an overall confidence estimate

Save to a text file
```bash
python3 scripts/inference_phowhisper.py \
  --audio "data/raw/your_audio.m4a" \
  --model VinAI/PhoWhisper-medium \
  --device auto \
  > "data/processed/your_audio_transcription.txt"
```

### 3) Timestamped transcript (chunked) + SRT

Produce a timestamped transcript and optional SRT captions. You can tune chunk length and overlap.
```bash
python3 scripts/inference_phowhisper.py \
  --audio "data/raw/your_audio.m4a" \
  --model VinAI/PhoWhisper-medium \
  --device auto \
  --timestamps \
  --chunk-sec 30 \
  --overlap-sec 0.5 \
  --txt-out "data/processed/your_audio_timestamped.txt" \
  --srt-out "data/processed/your_audio.srt"
```



### 4) Tips & troubleshooting

- Large model download: The first run will download model weights from Hugging Face. This can take several minutes.
- Audio loading:
  - Uses torchaudio by default
  - Falls back to librosa (audioread) and then pydub (requires ffmpeg) if needed
- If you see MPS/CUDA warnings, the script will safely fall back to CPU.
- If you experience slow processing on CPU, consider shorter `--chunk-sec` for a faster perceived progress in timestamped mode.

### 5) Project layout

```
ASR model/
  ├── requirements.txt
  ├── environment.yml                # optional conda env
  ├── scripts/
  │   └── inference_phowhisper.py    # main CLI entry point
  └── src/
      └── models/
          └── transformer_asr.py     # PhoWhisperASR wrapper
```

### 6) Programmatic usage

```python
from src.models.transformer_asr import PhoWhisperASR
import torch

audio = ...  # torch.Tensor of shape (samples,) at 16 kHz
asr = PhoWhisperASR(model_name="VinAI/PhoWhisper-medium", device="auto")
result = asr.transcribe(audio, sampling_rate=16000)
print(result["transcription"][0])
print(result["confidence"])  # float
```




