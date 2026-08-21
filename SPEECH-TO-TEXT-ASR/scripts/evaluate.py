#!/usr/bin/env python3
"""Evaluation script for IndicConformer 600M Multilingual ASR.

This script evaluates the IndicConformer model on a test dataset and computes:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Real-Time Factor (RTF)
- Per-language and per-condition breakdowns

Usage:
    python evaluate.py --data_dir /path/to/test_data --output results.json

Dataset structure expected:
    test_data/
    ├── hindi/
    │   ├── clean/
    │   │   ├── audio1.flac
    │   │   └── audio1.txt  (ground truth)
    │   ├── noisy/
    │   ├── telephony/
    │   └── code_switch/
    ├── tamil/
    │   └── ...
    └── ...
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def load_audio(audio_path: Path) -> torch.Tensor:
    """Load and preprocess audio to 16kHz mono tensor."""
    audio_data, sr = sf.read(str(audio_path))
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Convert to tensor
    wav = torch.from_numpy(audio_data).float().unsqueeze(0)
    
    if sr != TARGET_SAMPLE_RATE:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        wav = resampler(wav)

    return wav


def score_texts(reference: str, hypothesis: str) -> dict:
    """Compute WER, CER, and edit counts between reference and hypothesis."""
    s = jiwer.process_words(reference, hypothesis)
    return {
        "WER": jiwer.wer(reference, hypothesis),
        "CER": jiwer.cer(reference, hypothesis),
        "substitutions": s.substitutions,
        "deletions": s.deletions,
        "insertions": s.insertions,
    }


def evaluate_dataset(
    model,
    data_dir: Path,
    languages: list[str],
    conditions: list[str],
    decoder: str = "ctc",
) -> dict:
    """Evaluate model on the entire dataset."""
    results = {
        "model": MODEL_NAME,
        "decoder": decoder,
        "per_language": {},
        "overall": {"WER": 0.0, "CER": 0.0, "total_words": 0, "total_chars": 0}
    }

    total_wer_weighted = 0.0
    total_cer_weighted = 0.0
    total_words = 0
    total_chars = 0

    for lang in languages:
        lang_dir = data_dir / lang
        if not lang_dir.exists():
            print(f"Warning: Language directory {lang} not found, skipping.")
            continue

        lang_results = {"conditions": {}, "WER": 0.0, "CER": 0.0}
        lang_wer_weighted = 0.0
        lang_cer_weighted = 0.0
        lang_words = 0
        lang_chars = 0

        for condition in conditions:
            cond_dir = lang_dir / condition
            if not cond_dir.exists():
                continue

            audio_files = list(cond_dir.glob("*.flac")) + list(cond_dir.glob("*.wav"))
            cond_results = []

            for audio_file in sorted(audio_files):
                gt_file = audio_file.with_suffix(".txt")
                if not gt_file.exists():
                    print(f"Warning: No ground truth for {audio_file}")
                    continue

                reference = gt_file.read_text().strip()

                # Transcribe
                start_time = time.time()
                wav = load_audio(audio_file)
                hypothesis = model(wav, lang, decoder)
                hypothesis = hypothesis.strip() if isinstance(hypothesis, str) else str(hypothesis)
                inference_time = time.time() - start_time

                # Get audio duration
                audio_duration = wav.shape[1] / TARGET_SAMPLE_RATE

                # Score
                scores = score_texts(reference, hypothesis)
                scores["file"] = audio_file.name
                scores["inference_time"] = inference_time
                scores["audio_duration"] = audio_duration
                scores["RTF"] = inference_time / audio_duration if audio_duration > 0 else float('inf')

                cond_results.append(scores)

                # Accumulate for weighted average
                ref_words = len(reference.split())
                ref_chars = len(reference)
                lang_wer_weighted += scores["WER"] * ref_words
                lang_cer_weighted += scores["CER"] * ref_chars
                lang_words += ref_words
                lang_chars += ref_chars

            if cond_results:
                avg_wer = sum(r["WER"] for r in cond_results) / len(cond_results)
                avg_cer = sum(r["CER"] for r in cond_results) / len(cond_results)
                avg_rtf = sum(r["RTF"] for r in cond_results) / len(cond_results)
                lang_results["conditions"][condition] = {
                    "WER": avg_wer,
                    "CER": avg_cer,
                    "RTF": avg_rtf,
                    "num_files": len(cond_results),
                }

        if lang_words > 0:
            lang_results["WER"] = lang_wer_weighted / lang_words
            lang_results["CER"] = lang_cer_weighted / lang_chars if lang_chars > 0 else 0.0
            lang_results["total_words"] = lang_words
            lang_results["total_chars"] = lang_chars

        results["per_language"][lang] = lang_results

        total_wer_weighted += lang_wer_weighted
        total_cer_weighted += lang_cer_weighted
        total_words += lang_words
        total_chars += lang_chars

    if total_words > 0:
        results["overall"]["WER"] = total_wer_weighted / total_words
        results["overall"]["CER"] = total_cer_weighted / total_chars if total_chars > 0 else 0.0
        results["overall"]["total_words"] = total_words
        results["overall"]["total_chars"] = total_chars

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate IndicConformer 600M ASR model")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to test dataset")
    parser.add_argument("--output", type=Path, default="results.json", help="Output JSON file")
    parser.add_argument("--decoder", default="ctc", choices=["ctc", "rnnt"], help="Decoder type")
    parser.add_argument("--languages", nargs="+",
                       default=["hi", "ta", "bn", "te", "mr"],
                       help="Language codes to evaluate")
    parser.add_argument("--conditions", nargs="+",
                       default=["clean", "noisy", "telephony", "code_switch"],
                       help="Conditions to evaluate")
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)

    print(f"Evaluating on languages: {args.languages}")
    print(f"Conditions: {args.conditions}")
    print(f"Decoder: {args.decoder}")

    results = evaluate_dataset(model, args.data_dir, args.languages, args.conditions, args.decoder)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall WER: {results['overall']['WER']:.2%}")
    print(f"Overall CER: {results['overall']['CER']:.2%}")
    print(f"Total words evaluated: {results['overall']['total_words']}")
    print("\nPer-language results:")
    for lang, lang_res in results["per_language"].items():
        print(f"  {lang}: WER={lang_res['WER']:.2%}, CER={lang_res['CER']:.2%}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
