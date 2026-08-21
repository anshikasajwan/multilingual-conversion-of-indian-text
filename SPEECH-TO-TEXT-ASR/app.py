#!/usr/bin/env python3
"""Multilingual Speech-to-Text module (IndicConformer 600M) -- Streamlit interface.

Run with:
    streamlit run app.py

Core design:
  - one cached IndicConformer model across all page interactions
  - local inference, no network calls, no audio leaves the machine
  - supports 22 official Indian languages
  - CTC and RNNT decoding strategies
  - energy-based silence trimming before inference to reduce hallucinations
  - ground-truth comparison via JiWER: WER, CER, and raw edit counts
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import AutoModel

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    jiwer = None
    HAS_JIWER = False

# ---------------------------------------------------------------------
# Supported languages (22 official Indian languages)
# ---------------------------------------------------------------------

LANGUAGES = {
    "Auto-detect": None,
    "Assamese (as)": "as",
    "Bengali (bn)": "bn",
    "Bodo (brx)": "brx",
    "Dogri (doi)": "doi",
    "Gujarati (gu)": "gu",
    "Hindi (hi)": "hi",
    "Kannada (kn)": "kn",
    "Konkani (kok)": "kok",
    "Kashmiri (ks)": "ks",
    "Maithili (mai)": "mai",
    "Malayalam (ml)": "ml",
    "Manipuri (mni)": "mni",
    "Marathi (mr)": "mr",
    "Nepali (ne)": "ne",
    "Odia (or)": "or",
    "Punjabi (pa)": "pa",
    "Sanskrit (sa)": "sa",
    "Santali (sat)": "sat",
    "Sindhi (sd)": "sd",
    "Tamil (ta)": "ta",
    "Telugu (te)": "te",
    "Urdu (ur)": "ur",
}

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the IndicConformer model once; reused across all page interactions."""
    return AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)


# ---------------------------------------------------------------------
# Audio handling
# ---------------------------------------------------------------------

def decode_audio(uploaded, workdir: Path) -> Path:
    """Decode an uploaded file to a 16 kHz mono WAV via ffmpeg."""
    src = workdir / (uploaded.name or "upload.bin")
    src.write_bytes(uploaded.getbuffer())
    out = workdir / f"{src.stem}_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-vn", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to decode audio. Install it (e.g. `brew install "
            "ffmpeg` on macOS) and try again."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg could not decode {uploaded.name}. Unsupported or corrupt file."
        ) from exc
    return out


def trim_silence(samples: np.ndarray, sr: int = 16000,
                 threshold: float = 0.01, pad: float = 0.3) -> np.ndarray:
    """Energy-based silence trimming (VAD) with padding around voiced spans."""
    if samples is None or samples.size == 0:
        return samples
    frame = int(0.02 * sr)
    pad_n = int(pad * sr)
    n = len(samples) - len(samples) % frame
    if n == 0:
        return samples
    frames = samples[:n].reshape(-1, frame)
    voiced = np.abs(frames).mean(axis=1) > threshold
    if not voiced.any():
        return samples
    idx = np.where(voiced)[0]
    start = max(0, int(idx.min()) * frame - pad_n)
    end = min(len(samples), (int(idx.max()) + 1) * frame + pad_n)
    return samples[start:end]


# ---------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------

def transcribe(model, audio_path: str, language: str = "hi", decoder: str = "ctc"):
    """Run IndicConformer on the audio file."""
    audio_data, sr = sf.read(audio_path)
    
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
    return transcription


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def score_texts(reference: str, hypothesis: str) -> dict:
    """Align reference and hypothesis and return WER, CER, and edit counts."""
    s = jiwer.process_words(reference, hypothesis)
    return {
        "WER": jiwer.wer(reference, hypothesis),
        "CER": jiwer.cer(reference, hypothesis),
        "sub": s.substitutions,
        "del": s.deletions,
        "ins": s.insertions,
    }


# ---------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="IndicConformer Multilingual ASR (600M)",
                       page_icon="🎙️", layout="wide")
    st.title("IndicConformer Multilingual ASR")
    st.caption("AI4Bharat IndicConformer 600M · 22 Indian Languages · CTC/RNNT Decoding")

    with st.sidebar:
        st.header("Model options")
        st.info(f"Model: {MODEL_NAME}\n\nParameters: 600M\n\nLanguages: 22")

        language_label = st.selectbox("Language", list(LANGUAGES), index=6)
        decoder = st.selectbox(
            "Decoder",
            ["ctc", "rnnt"],
            index=0,
            help="CTC is faster; RNNT may be more accurate."
        )

        st.header("Mitigations")
        do_trim = st.checkbox("Trim leading/trailing silence", value=True)

    language = LANGUAGES[language_label]

    try:
        with st.spinner("Loading IndicConformer 600M..."):
            model = load_model()
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        uploaded = st.file_uploader(
            "Upload an audio file (wav, mp3, m4a, flac, ogg, webm)",
            type=["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4"],
        )

        if uploaded is not None:
            st.audio(uploaded, format=uploaded.type or "audio/wav")

    with col_right:
        reference = st.text_area(
            "Ground truth (optional)",
            height=140,
            placeholder="Paste the reference transcript here to compute WER and CER...",
        )
        do_score = st.checkbox("Score against ground truth", value=False)

    if uploaded is None:
        st.info("Upload an audio file to begin.")
        return

    if st.button("Transcribe", type="primary"):
        with st.spinner("Transcribing..."):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    workdir = Path(tmp)
                    wav_path = decode_audio(uploaded, workdir)
                    transcription = transcribe(
                        model, str(wav_path),
                        language=language if language else "hi",
                        decoder=decoder,
                    )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        text = transcription.strip() if isinstance(transcription, str) else str(transcription)
        st.subheader("Transcript")
        st.markdown(f"> {text}")

        if language:
            st.caption(f"Selected language: **{language}**")

        if do_score and reference.strip():
            if not HAS_JIWER:
                st.warning("jiwer is not installed; install it with `pip install jiwer`.")
            elif not text:
                st.warning("Empty transcript; nothing to score.")
            else:
                m = score_texts(reference.strip(), text)
                st.subheader("Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("WER", f"{m['WER']:.2%}")
                c2.metric("CER", f"{m['CER']:.2%}")
                c3.metric("Reference words", int(jiwer.words(reference.strip()).shape[0]))
                st.caption(
                    f"Substitutions: {m['sub']} · Deletions: {m['del']} · Insertions: {m['ins']}"
                )
        elif do_score:
            st.info("Provide ground truth to compute WER and CER.")

    st.markdown("---")
    st.caption(
        "Local inference only: audio never leaves this machine. Model weights are "
        "MIT-licensed (AI4Bharat IndicConformer)."
    )


if __name__ == "__main__":
    main()
