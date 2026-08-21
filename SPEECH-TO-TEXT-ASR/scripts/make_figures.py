#!/usr/bin/env python3
"""Generate figures for IndicConformer evaluation paper.

This script creates publication-quality figures for visualizing ASR evaluation results.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_wer_by_language(results: dict, output_path: Path):
    """Plot WER comparison across languages."""
    languages = list(results["per_language"].keys())
    wer_values = [results["per_language"][lang]["WER"] * 100 for lang in languages]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(languages, wer_values, color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("WER (%)", fontsize=12)
    ax.set_title("Word Error Rate by Language - IndicConformer 600M", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, wer_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_wer_by_condition(results: dict, output_path: Path):
    """Plot WER breakdown by difficulty condition."""
    languages = list(results["per_language"].keys())
    conditions = ["clean", "noisy", "telephony", "code_switch"]
    condition_labels = ["Clean", "Noisy", "Telephony", "Code-Switch"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(languages))
    width = 0.2

    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for i, (condition, label) in enumerate(zip(conditions, condition_labels)):
        values = []
        for lang in languages:
            cond_data = results["per_language"][lang].get("conditions", {}).get(condition, {})
            values.append(cond_data.get("WER", 0) * 100)

        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i],
                     edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("WER (%)", fontsize=12)
    ax.set_title("WER by Condition Across Languages", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(languages, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rtf_analysis(results: dict, output_path: Path):
    """Plot Real-Time Factor analysis."""
    languages = list(results["per_language"].keys())
    rtf_values = []

    for lang in languages:
        cond_rtfs = []
        for cond_data in results["per_language"][lang].get("conditions", {}).values():
            if "RTF" in cond_data:
                cond_rtfs.append(cond_data["RTF"])
        rtf_values.append(np.mean(cond_rtfs) if cond_rtfs else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(languages, rtf_values, color='coral', edgecolor='black', linewidth=0.5)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold (RTF=1)')
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Real-Time Factor", fontsize=12)
    ax.set_title("Real-Time Factor by Language", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, rtf_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(results: dict, output_path: Path):
    """Plot a heatmap of WER across languages and conditions."""
    languages = list(results["per_language"].keys())
    conditions = ["clean", "noisy", "telephony", "code_switch"]
    condition_labels = ["Clean", "Noisy", "Telephony", "Code-Switch"]

    matrix = np.zeros((len(languages), len(conditions)))
    for i, lang in enumerate(languages):
        for j, cond in enumerate(conditions):
            cond_data = results["per_language"][lang].get("conditions", {}).get(cond, {})
            matrix[i, j] = cond_data.get("WER", 0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(condition_labels)))
    ax.set_yticks(np.arange(len(languages)))
    ax.set_xticklabels(condition_labels)
    ax.set_yticklabels(languages)

    for i in range(len(languages)):
        for j in range(len(conditions)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                         ha="center", va="center", color="black", fontsize=9)

    ax.set_title("WER (%) Heatmap: Language × Condition", fontsize=14)
    plt.colorbar(im, ax=ax, label="WER (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results", type=Path, default="results.json", help="Results JSON file")
    parser.add_argument("--output_dir", type=Path, default="figures", help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    plot_wer_by_language(results, args.output_dir / "wer_by_language.png")
    plot_wer_by_condition(results, args.output_dir / "wer_by_condition.png")
    plot_rtf_analysis(results, args.output_dir / "rtf_analysis.png")
    plot_confusion_matrix(results, args.output_dir / "confusion_matrix.png")

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
