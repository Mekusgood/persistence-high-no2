#!/usr/bin/env python3
"""
plot_figure_s1.py
Figure S1: threshold sensitivity of run-structure metrics, 70th–90th percentiles.

Values are the corrected Table 8 (land-only series). All rows use linear-interpolation
percentiles of the 72 monthly land-only city means (Option B), so the 75th-percentile row
matches the threshold reported in the Methods (18 high months in each city). The figure is generated from the table values below so the
two cannot diverge.

Outputs: figure_S1_threshold_sensitivity.pdf / .png / _caption.txt
Usage  : python plot_figure_s1.py --outdir "/content/drive/MyDrive/NO2_persistence_exports/scripts/figures"
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CITY_ORDER = ["Lagos", "Kinshasa", "Dar es Salaam", "Johannesburg"]
COLORS = {"Lagos": "#0072B2", "Kinshasa": "#E69F00",
          "Dar es Salaam": "#009E73", "Johannesburg": "#CC79A7"}   # Okabe-Ito
MARKERS = {"Lagos": "o", "Kinshasa": "s", "Dar es Salaam": "^", "Johannesburg": "D"}
PERCENTILES = [70, 75, 80, 90]
PRIMARY = 75

# city, percentile, threshold (1e15 molec cm-2), high months, runs, mean run, longest run,
# peak season, peak fraction   -- from corrected Table 8
TABLE8 = [
    ("Lagos", 70, 4.22, 22, 8, 2.8, 5, "DJF", 0.73),
    ("Kinshasa", 70, 2.41, 22, 8, 2.8, 5, "JJA", 0.64),
    ("Dar es Salaam", 70, 1.02, 22, 11, 2.0, 3, "JJA", 0.36),
    ("Johannesburg", 70, 7.43, 22, 12, 1.8, 3, "JJA", 0.50),
    ("Lagos", 75, 4.31, 18, 7, 2.6, 4, "DJF", 0.72),
    ("Kinshasa", 75, 2.60, 18, 10, 1.8, 3, "JJA", 0.61),
    ("Dar es Salaam", 75, 1.04, 18, 12, 1.5, 2, "JJA", 0.39),
    ("Johannesburg", 75, 7.75, 18, 9, 2.0, 3, "JJA", 0.56),
    ("Lagos", 80, 4.43, 15, 8, 1.9, 4, "DJF", 0.80),
    ("Kinshasa", 80, 2.63, 15, 7, 2.1, 3, "JJA", 0.67),
    ("Dar es Salaam", 80, 1.06, 15, 10, 1.5, 2, "JJA", 0.40),
    ("Johannesburg", 80, 8.16, 15, 8, 1.9, 3, "JJA", 0.67),
    ("Lagos", 90, 4.95, 8, 6, 1.3, 2, "DJF", 0.88),
    ("Kinshasa", 90, 2.84, 8, 5, 1.6, 3, "JJA", 0.75),
    ("Dar es Salaam", 90, 1.10, 8, 7, 1.1, 2, "JJA", 0.63),
    ("Johannesburg", 90, 10.57, 8, 5, 1.6, 2, "JJA", 0.75),
]
COLS = ["city", "percentile", "threshold", "high_months", "runs", "mean_run",
        "longest_run", "peak_season", "peak_fraction"]

CAPTION = (
    "Figure S1. Threshold sensitivity of run-structure metrics for Lagos, Kinshasa, Dar es Salaam "
    "and Johannesburg across the 70th, 75th, 80th and 90th percentile city-specific thresholds, "
    "computed from the monthly land-only city means. Panels show the longest run of consecutive "
    "high-NO2 months, the mean run length, and the peak-season fraction (share of high-NO2 months "
    "falling in the peak season). The vertical dotted line marks the primary 75th-percentile "
    "threshold (18 high months in each city). Other thresholds are linear-interpolation "
    "percentiles of the same series. "
    "This figure is descriptive."
)


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def check_table(df):
    """Guard: the primary row must carry the Earth Engine high-month counts."""
    primary = df[df.percentile == PRIMARY].set_index("city")["high_months"]
    expected = {"Lagos": 18, "Kinshasa": 18, "Dar es Salaam": 18, "Johannesburg": 18}
    for city, n in expected.items():
        assert primary[city] == n, f"{city}: expected {n} high months at P{PRIMARY}, got {primary[city]}"
    assert len(df) == 16, "Table 8 must contain 16 rows (4 cities x 4 percentiles)"


def plot(df, outdir: Path):
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    panels = [
        ("longest_run", "Consecutive months", "Longest run", None),
        ("mean_run", "Months", "Mean run length", None),
        ("peak_fraction", "Fraction of high-NO$_2$ months", "Peak-season fraction", (0.3, 1.0)),
    ]
    handles = []
    for ax, (col, ylabel, title, ylim) in zip(axes, panels):
        for city in CITY_ORDER:
            g = df[df.city == city].sort_values("percentile")
            h, = ax.plot(g.percentile, g[col], marker=MARKERS[city], color=COLORS[city],
                         lw=1.3, ms=4.5, label=city)
            if ax is axes[0]:
                handles.append(h)
        ax.axvline(PRIMARY, ls=":", color="#555555", lw=0.9, zorder=0)
        ax.set_xticks(PERCENTILES)
        ax.set_xlabel("Threshold percentile", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e0e0e0", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_yticks(np.arange(1, 7))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    stem = outdir / "figure_S1_threshold_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    (outdir / "figure_S1_threshold_sensitivity_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    print(f"Saved {stem.with_suffix('.pdf')} and .png; caption written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    df = pd.DataFrame(TABLE8, columns=COLS)
    check_table(df)
    plot(df, args.outdir)


if __name__ == "__main__":
    main()