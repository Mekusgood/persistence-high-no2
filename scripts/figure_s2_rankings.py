#!/usr/bin/env python3
"""
Figure S2: city rankings by mean tropospheric NO2 burden and by monthly-scale
run-structure metrics (number of runs, mean run length, longest run), using the
Option B land-only series (18 high-NO2 months per city).

Input : option_B_landonly_input.csv
Output: figure_S2_rankings.pdf and figure_S2_rankings.png
Usage : python figure_s2_rankings.py --csv option_B_landonly_input.csv --out-dir plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MOL_M2_TO_1E15 = 6.02214e23 / 1e4 / 1e15     # mol m-2 -> 1e15 molecules cm-2
CITIES = ["Johannesburg", "Lagos", "Kinshasa", "Dar es Salaam"]
STYLE = {"Lagos": ("#1f77b4", "o"), "Johannesburg": ("#e377c2", "D"),
         "Kinshasa": ("#ff7f0e", "s"), "Dar es Salaam": ("#2ca02c", "^")}
EXPECTED = {  # Table 2, raw series
    "runs": {"Lagos": 7, "Johannesburg": 9, "Kinshasa": 10, "Dar es Salaam": 12},
    "mean_run": {"Lagos": 2.6, "Johannesburg": 2.0, "Kinshasa": 1.8, "Dar es Salaam": 1.5},
    "longest": {"Lagos": 4, "Johannesburg": 3, "Kinshasa": 3, "Dar es Salaam": 2},
}


def run_stats(flags):
    runs, n = [], 0
    for v in flags:
        if v:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return len(runs), float(np.mean(runs)), int(max(runs))


def rank_values(values, higher_is_better):
    """Return {city: rank} with ties sharing the lowest rank number."""
    order = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    ranks, prev, pos = {}, None, 0
    for i, (c, v) in enumerate(order, start=1):
        if v != prev:
            pos = i; prev = v
        ranks[c] = pos
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.csv).sort_values(["city", "month_start"])
    assert d.groupby("city").high_no2_flag.sum().eq(18).all(), "expected 18 high months per city"

    burden = {c: float(d[d.city == c].no2_tropospheric_column.mean() * MOL_M2_TO_1E15) for c in CITIES}
    stats = {c: run_stats(d[d.city == c].high_no2_flag.astype(int).values) for c in CITIES}
    runs = {c: stats[c][0] for c in CITIES}
    mean_run = {c: stats[c][1] for c in CITIES}
    longest = {c: stats[c][2] for c in CITIES}

    for key, got in [("runs", runs), ("mean_run", mean_run), ("longest", longest)]:
        for c in CITIES:
            exp = EXPECTED[key][c]
            ok = abs(got[c] - exp) < 0.051 if key == "mean_run" else got[c] == exp
            assert ok, f"{key} {c}: got {got[c]}, Table 2 says {exp}"
    print("Check passed: run statistics match Table 2.")

    r_burden = rank_values(burden, True)
    r_runs = rank_values(runs, False)
    r_mean = rank_values(mean_run, True)
    r_long = rank_values(longest, True)
    for c in CITIES:
        print(f"{c:15s} burden {burden[c]:.3f} (rank {r_burden[c]})  runs {runs[c]} (rank {r_runs[c]})  "
              f"mean run {mean_run[c]:.1f} (rank {r_mean[c]})  longest {longest[c]} (rank {r_long[c]})")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    panels = [
        (r_runs, runs, "{:d} runs", "Number of runs rank"),
        (r_mean, mean_run, "{:.1f} months", "Mean run length rank"),
        (r_long, longest, "{:d} months", "Longest run rank"),
    ]
    xl, xr = 0.0, 1.0
    for ax, (rank_r, vals, fmt, rtitle) in zip(axes, panels):
        for c in CITIES:
            col, mk = STYLE[c]
            y_left = r_burden[c]
            y_right = rank_r[c]
            ax.plot([xl, xr], [-y_left, -y_right], color=col, lw=1.8, zorder=2)
            ax.scatter([xl], [-y_left], color=col, marker=mk, s=46, zorder=3)
            ax.scatter([xr], [-y_right], color=col, marker=mk, s=46, zorder=3)
            ax.text(xl - 0.05, -y_left, f"{y_left}{['st','nd','rd','th'][y_left-1]}  {c}",
                    ha="right", va="center", color=col, fontsize=9, fontweight="bold")
            ax.text(xr + 0.05, -y_right, fmt.format(vals[c]) + ("  (tied)" if sum(1 for k in rank_r if rank_r[k] == rank_r[c]) > 1 else ""),
                    ha="left", va="center", color=col, fontsize=9, fontweight="bold")
        ax.set_xlim(-0.75, 1.45)
        ax.set_ylim(-4.5, -0.5)
        ax.axis("off")
        ax.text(xl, -4.85, "Mean NO$_2$ burden rank", ha="center", va="top", fontsize=8.5)
        ax.text(xr, -4.85, rtitle, ha="center", va="top", fontsize=8.5)
    fig.tight_layout()

    stem = args.out_dir / "figure_S2_rankings"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
