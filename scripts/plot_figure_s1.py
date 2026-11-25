#!/usr/bin/env python3
"""
plot_figure_s1.py
Figure S1: threshold sensitivity of run-structure metrics, 70th–90th percentiles.

Table 8 values are computed from the canonical input (option_B_landonly_input.csv) using
linear-interpolation percentiles of the 72 monthly land-only city means (Option B), so the
75th-percentile row matches the Methods (18 high months in each city).

Outputs: figure_S1_threshold_sensitivity.pdf / .png / _caption.txt
Usage  : python plot_figure_s1.py --input option_B_landonly_input.csv --outdir figures
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

COLS = ["city", "percentile", "threshold", "high_months", "runs", "mean_run",
        "longest_run", "peak_season", "peak_fraction"]

MOL_TO_1E15 = 6.02214076e19 / 1e15   # mol m-2 -> 1e15 molec cm-2


def _half_up(x, nd):
    """Round half up (not Python's half-even) so 0.625 becomes 0.63."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def _runs(flags):
    padded = np.concatenate(([0], flags, [0]))
    d = np.diff(padded)
    lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    if lengths.size == 0:
        return 0, 0.0, 0
    return int(lengths.size), float(lengths.mean()), int(lengths.max())


def compute_table(csv_path: Path) -> pd.DataFrame:
    """Run-structure metrics for each city and percentile, from the land-only monthly series."""
    df = pd.read_csv(csv_path)
    rows = []
    for pct in PERCENTILES:
        for city in CITY_ORDER:
            g = df[df.city == city].sort_values("month_start")
            v = g["no2_tropospheric_column"].to_numpy(dtype=float)
            thr = np.percentile(v, pct)
            if pct == PRIMARY:
                csv_thr = float(g["city_no2_p75"].iloc[0])
                assert np.isclose(thr, csv_thr, rtol=1e-6), f"{city}: P{pct} differs from city_no2_p75"
            flags = (v >= thr).astype(int)
            n_runs, mean_run, longest = _runs(flags)
            counts = pd.Series(g["season_4"].to_numpy()[flags == 1]).value_counts()
            n_high = int(flags.sum())
            rows.append((city, pct, _half_up(thr * MOL_TO_1E15, 2), n_high, n_runs,
                         _half_up(mean_run, 1), longest, counts.idxmax(),
                         _half_up(counts.max() / n_high, 2)))
    return pd.DataFrame(rows, columns=COLS)


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
    ap.add_argument("--input", type=Path, required=True,
                    help="option_B_landonly_input.csv (canonical input)")
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    df = compute_table(args.input)
    check_table(df)
    print(df.to_string(index=False))
    plot(df, args.outdir)


if __name__ == "__main__":
    main()