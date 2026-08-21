#!/usr/bin/env python3
"""Command-line transcription with IndicConformer (batch / terminal use).

CLI entry point for IndicConformer 600M Multilingual ASR.
Supports CTC and RNNT decoding across 22 Indian languages.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def transcribe_audio(
    audio_path: str,
    language: str = "hi",
    decoder: str = "ctc",
) -> str:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)

    audio_data, sr = sf.read(str(audio_file))
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Convert to tensor
    wav = torch.from_numpy(audio_data).float().unsqueeze(0)
    
    if sr != TARGET_SAMPLE_RATE:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        wav = resampler(wav)

    transcription = model(wav, language, decoder)
    return transcription.strip() if isinstance(transcription, str) else str(transcription)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe speech to text using IndicConformer 600M (22 Indian languages)"
    )
    parser.add_argument("audio_path", help="Path to a .wav, .mp3, .m4a, or similar audio file")
    parser.add_argument(
        "--language", default="hi",
        help="ISO-639-1 language code (e.g. 'hi', 'ta', 'bn'). Default: hi",
    )
    parser.add_argument(
        "--decoder", default="ctc", choices=["ctc", "rnnt"],
        help="Decoding strategy: ctc (faster) or rnnt (more accurate)"
    )
    args = parser.parse_args()

    text = transcribe_audio(
        args.audio_path,
        language=args.language,
        decoder=args.decoder,
    )
    print(text)


if __name__ == "__main__":
    main()
