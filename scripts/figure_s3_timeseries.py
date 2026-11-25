#!/usr/bin/env python3
"""
Supplementary time series: monthly land-only city-mean tropospheric NO2 (30 km buffer),
with the city-wide 75th-percentile threshold and high months marked.

Units: all NO2 values are shown in 10^15 molecules cm^-2, the same unit used for the
maps (UNIT_SCALE_NO2 = 1e15). The input table is in mol m^-2, converted with
1 mol m^-2 = 6.022e19 molecules cm^-2 (then divided by 1e15).

Input : option_B_landonly_input.csv (288 city-months; Option B thresholds, 18 high months per city)
        Required columns: city, month_start, no2_tropospheric_column [mol m-2],
        city_no2_p75 [mol m-2], high_no2_flag
Output: figure_s3_timeseries.pdf and figure_s3_timeseries.png

Usage (Colab):
  !python figure_s3_timeseries_corrected.py \
     --csv "/content/drive/MyDrive/NO2_persistence_exports/scripts/csv_files/option_B_landonly_input.csv" \
     --out-dir "./plots"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

MOL_M2_TO_MOLEC_CM2 = 6.022e19   # 1 mol m^-2 = 6.022e19 molecules cm^-2
UNIT_SCALE_NO2 = 1e15            # plot unit: 10^15 molecules cm^-2
UNIT_LABEL = r"NO$_2$ (10$^{15}$ molecules cm$^{-2}$)"
CITY_ORDER = ["Lagos", "Kinshasa", "Dar es Salaam", "Johannesburg"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def to_plot_units(mol_m2):
    return mol_m2 * MOL_M2_TO_MOLEC_CM2 / UNIT_SCALE_NO2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.csv)
    d["month_start"] = pd.to_datetime(d["month_start"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)

    for ax, city, lab in zip(axes.flat, CITY_ORDER, PANEL_LABELS):
        s = d[d.city == city].sort_values("month_start")
        x = s["month_start"]
        y = to_plot_units(s["no2_tropospheric_column"].values)
        thr = float(to_plot_units(s["city_no2_p75"].iloc[0]))
        hi = s["high_no2_flag"].astype(bool).values

        ax.plot(x, y, color="#4a4a4a", lw=1.0, zorder=2)
        ax.scatter(x[hi], y[hi], s=16, color="#2171b5", zorder=3,
                   label=f"High month (n = {hi.sum()})")
        ax.scatter(x[~hi], y[~hi], s=10, facecolor="white", edgecolor="#7f7f7f",
                   lw=0.7, zorder=3, label="Below threshold")
        ax.axhline(thr, color="#c0392b", ls="--", lw=1.0, zorder=1)

        # P75 label: placed just outside the right-hand end of the dashed line.
        # x in axes coordinates (1.01 = a short gap past the axes edge), y in data coordinates.
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.01, thr, f"P75 = {thr:.2f}", transform=trans,
                color="#c0392b", ha="left", va="center", fontsize=8, clip_on=False)

        ax.set_title(f"{lab} {city}", loc="left", fontsize=10)
        ax.set_ylabel(UNIT_LABEL)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Legend at the bottom of the first (top-left) subplot, inside its lower edge.
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower center", ncol=2)

    fig.tight_layout(rect=[0, 0.0, 0.97, 1.0])

    stem = args.out_dir / "figure_s3_timeseries"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
