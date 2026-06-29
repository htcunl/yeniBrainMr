#!/usr/bin/env python
"""
70/10/20 baseline ve 80/10/10 locked test sonuçlarını
tek görselde karşılaştırmalı bar chart olarak çizer.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_metrics(results_path: Path) -> dict[str, float]:
    text = results_path.read_text(encoding="utf-8")
    patterns = {
        "Dice": r"Dice:\s+([0-9.]+)",
        "IoU": r"IoU:\s+([0-9.]+)",
        "Sensitivity": r"Sensitivity:\s+([0-9.]+)",
        "Specificity": r"Specificity:\s+([0-9.]+)",
        "Precision": r"Precision:\s+([0-9.]+)",
        "F1": r"F1:\s+([0-9.]+)",
        "HD95": r"HD95:\s+([0-9.]+)",
        "ASD": r"ASD:\s+([0-9.]+)",
        "MCC": r"MCC:\s+([0-9.]+)",
    }
    out: dict[str, float] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"{results_path} içinde '{key}' bulunamadı.")
        out[key] = float(m.group(1))
    return out


def annotate_bars(ax, bars, fmt: str = "{:.4f}", y_pad: float = 0.01):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + y_pad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def main():
    root = Path(__file__).resolve().parent.parent
    baseline_file = root / "test_results" / "test_results.txt"
    locked_file = root / "test_results_801010_locked" / "test_results.txt"
    out_file = root / "test_results_801010_locked" / "metrics_bar_comparison_701020_vs_801010.png"

    baseline = parse_metrics(baseline_file)
    locked = parse_metrics(locked_file)

    main_metrics = ["Dice", "IoU", "Sensitivity", "Specificity", "Precision", "F1", "MCC"]
    dist_metrics = ["HD95", "ASD"]

    x_main = np.arange(len(main_metrics))
    x_dist = np.arange(len(dist_metrics))
    width = 0.36

    baseline_main = [baseline[m] for m in main_metrics]
    locked_main = [locked[m] for m in main_metrics]
    baseline_dist = [baseline[m] for m in dist_metrics]
    locked_dist = [locked[m] for m in dist_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [3.4, 1.2]})
    fig.suptitle("70/10/20 vs 80/10/10 Locked Test - Metrik Karşılaştırması", fontsize=14, fontweight="bold")

    # Sol panel: [0,1] aralığındaki metrikler
    ax1 = axes[0]
    b1 = ax1.bar(x_main - width / 2, baseline_main, width, label="70/10/20 Baseline", color="#3498db", alpha=0.9)
    b2 = ax1.bar(x_main + width / 2, locked_main, width, label="80/10/10 Locked Test", color="#e67e22", alpha=0.9)
    ax1.set_xticks(x_main)
    ax1.set_xticklabels(main_metrics, rotation=20)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Skor")
    ax1.set_title("Örtüşme/Sınıflandırma Metrikleri")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="lower left")
    annotate_bars(ax1, b1, fmt="{:.4f}", y_pad=0.01)
    annotate_bars(ax1, b2, fmt="{:.4f}", y_pad=0.01)

    # Sağ panel: piksel cinsinden mesafe metrikleri
    ax2 = axes[1]
    b3 = ax2.bar(x_dist - width / 2, baseline_dist, width, label="70/10/20", color="#2ecc71", alpha=0.9)
    b4 = ax2.bar(x_dist + width / 2, locked_dist, width, label="80/10/10", color="#9b59b6", alpha=0.9)
    ax2.set_xticks(x_dist)
    ax2.set_xticklabels(dist_metrics)
    ax2.set_ylabel("Piksel")
    ax2.set_title("Sınır Mesafe Metrikleri")
    ax2.grid(axis="y", alpha=0.3)
    annotate_bars(ax2, b3, fmt="{:.2f}", y_pad=max(max(baseline_dist), max(locked_dist)) * 0.02)
    annotate_bars(ax2, b4, fmt="{:.2f}", y_pad=max(max(baseline_dist), max(locked_dist)) * 0.02)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Oluşturuldu: {out_file}")


if __name__ == "__main__":
    main()
