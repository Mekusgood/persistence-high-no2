"""
Last modified: November 18, 2026
plot_monthly_timeseries.py
==========================
Monthly tropospheric NO2 time series for four African cities (2019-2024), land-only.

Input  : option_B_landonly_input.csv
         Columns used: month_start, city, no2_tropospheric_column (mol m-2, land-only),
         city_no2_p75 (mol m-2, land-only city-wide P75 from GEE).
         The all-cell column is never read.

Thresholds
----------
The dashed red line is the city-wide 75th percentile (P75) read from the city_no2_p75
column of the input CSV, so it matches the persistence flags used in the manuscript.
The script also recomputes P75 from the 72 monthly values and prints the difference as a check.

Outputs
-------
  <figures>/Fig2_monthly_no2_timeseries.pdf
  <figures>/Fig2_monthly_no2_timeseries.tiff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# =============================================================================
# Settings
# =============================================================================

CITY_ORDER = ["Lagos", "Kinshasa", "Dar es Salaam", "Johannesburg"]

# Unit conversion: mol m-2 -> 10^15 molecules cm-2
AVOGADRO    = 6.02214076e23
M2_TO_CM2   = 1.0e4
SCALE       = 1e15

# Monthly line + markers
LINE_COLOR   = "#3a4fd1"
LINE_LW      = 1.4
MARKER_STYLE = "o"
MARKER_SIZE  = 4.5

# 75th-percentile threshold: thin dashed red
THRESH_COLOR = "#e32b2b"
THRESH_LW    = 1.0
THRESH_STYLE = "--"

# Figure layout
FIGSIZE      = (7.5, 9.0)   # inches: full page width, four stacked panels
FIG_DPI      = 600
HSPACE       = 0.20
LEFT_MARGIN  = 0.09
RIGHT_MARGIN = 0.76  # room on the right for legends, which sit outside the data area
TOP_MARGIN   = 0.94
BOT_MARGIN   = 0.07

# X-axis: data run Jan 2019 to Dec 2024; axis ends at 2025-01-01
X_START_PAD_DAYS = 15
X_END           = pd.Timestamp("2025-01-01")
YEAR_TICKS      = list(range(2019, 2026))

# Typography
FONT_STACK    = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
SUPTITLE_SIZE = 11.5
TITLE_SIZE    = 10.0
LABEL_SIZE    = 9.0
TICK_SIZE     = 8.5
LEGEND_SIZE   = 8.0

# Output
SAVE_PDF    = True
SAVE_TIFF   = True
OUTPUT_STEM = "Fig2_monthly_no2_timeseries"



# <<< Style >>>

def set_style() -> None:
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     FONT_STACK,
        "font.size":           10,
        "axes.linewidth":      0.9,
        "axes.edgecolor":      "#333333",
        "axes.facecolor":      "white",
        "figure.facecolor":    "white",
        "savefig.facecolor":   "white",
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.major.size":    4.0,
        "ytick.major.size":    4.0,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
    })


# =============================================================================
# Data loading
# =============================================================================

def mol_m2_to_1e15(values):
    """mol m-2 -> 10^15 molecules cm-2."""
    return np.asarray(values, dtype=float) * AVOGADRO / M2_TO_CM2 / SCALE


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "no2_tropospheric_column" not in df.columns:
        raise KeyError("CSV must contain 'no2_tropospheric_column' (mol m-2, land-only).")
    if "allcell" in " ".join(df.columns).lower():
        print("  Note: all-cell columns are present in the file but are not used.")

    df["month_start"] = pd.to_datetime(df["month_start"])
    df["no2_pmolec"] = mol_m2_to_1e15(df["no2_tropospheric_column"])

    if "city_no2_p75" in df.columns:
        df["p75_pmolec"] = mol_m2_to_1e15(df["city_no2_p75"])

    df["city"] = pd.Categorical(df["city"], categories=CITY_ORDER, ordered=True)
    return df.sort_values(["city", "month_start"]).reset_index(drop=True)


# =============================================================================
# Thresholds
# =============================================================================
def compute_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """
    City-specific P75 (10^15 molecules cm^-2) from the city_no2_p75 column, so the
    plotted line matches the persistence flags. The value recomputed from the 72
    monthly values is printed as a check.
    """
    thresholds: dict[str, float] = {}

    print("\n  75th-percentile NO2 thresholds (10^15 molecules cm^-2):")
    print(f"  {'City':<16} {'From CSV':>10} {'Recomputed':>12} {'Diff':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*10}")

    for city in CITY_ORDER:
        sub = df[df["city"].astype(str) == city]
        vals = sub["no2_pmolec"].dropna().values
        if vals.size == 0:
            continue
        recomputed = float(np.nanpercentile(vals, 75))

        if "p75_pmolec" in sub.columns and sub["p75_pmolec"].notna().any():
            csv_vals = sub["p75_pmolec"].dropna()
            if csv_vals.nunique() > 1:
                raise ValueError(f"city_no2_p75 varies within {city}")
            t = float(csv_vals.iloc[0])
        else:
            t = recomputed
            print(f"  WARNING: city_no2_p75 not found for {city}; using recomputed value.")

        thresholds[city] = t
        print(f"  {city:<16} {t:>10.4f} {recomputed:>12.4f} {t - recomputed:>+10.4f}")

    print()
    return thresholds

# =============================================================================
# Plot
# =============================================================================

def plot_monthly_timeseries(df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    set_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    cities     = [c for c in CITY_ORDER if c in df["city"].astype(str).unique()]
    thresholds = compute_thresholds(df)

    fig, axes = plt.subplots(nrows=len(cities), ncols=1, figsize=FIGSIZE, sharex=True)
    if len(cities) == 1:
        axes = [axes]

    fig.subplots_adjust(
        hspace=HSPACE, left=LEFT_MARGIN, right=RIGHT_MARGIN,
        top=TOP_MARGIN, bottom=BOT_MARGIN,
    )

    for idx, (ax, city) in enumerate(zip(axes, cities)):
        city_df = (
            df[df["city"].astype(str) == city]
            .sort_values("month_start")
            .reset_index(drop=True)
        )
        dates = city_df["month_start"]
        no2   = city_df["no2_pmolec"]

        # 1. Monthly line + markers
        ax.plot(
            dates, no2,
            color=LINE_COLOR, linewidth=LINE_LW,
            marker=MARKER_STYLE, markersize=MARKER_SIZE,
            markerfacecolor=LINE_COLOR, markeredgecolor=LINE_COLOR,
            markeredgewidth=0.3, zorder=3, label="Monthly mean",
        )

        # 2. City-specific P75 threshold, two decimals to match the manuscript
        thresh = thresholds[city]
        ax.axhline(
            thresh, color=THRESH_COLOR, linewidth=THRESH_LW,
            linestyle=THRESH_STYLE, zorder=2, label=f"P75 ({thresh:.2f})",  
            #linestyle=THRESH_STYLE, zorder=2, label=f"P75 ({recomputed_val:.2f})",
        )

        # 3. Horizontal grey grid lines only
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, linestyle="-", zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # 4. Panel title
        ax.set_title(f"({chr(97 + idx)}) {city}", fontsize=TITLE_SIZE,
                     fontweight="bold", loc="left", pad=4)

        # 5. Y-axis label
        ax.set_ylabel(r"NO$_2$ VCD (10$^{15}$ molecules cm$^{-2}$)", fontsize=LABEL_SIZE)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.tick_params(labelsize=TICK_SIZE)

        # 6. Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 7. Legend
        handles, labels = ax.get_legend_handles_labels()
        # Legend sits outside the axes, to the right of each panel, so it never covers data
        ax.legend(
            handles=handles, labels=labels, fontsize=LEGEND_SIZE,
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
            frameon=False, handlelength=2.0, ncol=1,
        )

    # X-axis: data start Jan 2019, axis ends at 2025-01-01, annual ticks 2019-2025
    x_min = df["month_start"].min() - pd.DateOffset(days=X_START_PAD_DAYS)
    for ax in axes:
        ax.set_xlim(x_min, X_END)
        ax.margins(x=0)

    year_ticks = [pd.Timestamp(f"{y}-01-01") for y in YEAR_TICKS]
    axes[-1].set_xticks(year_ticks)
    axes[-1].set_xticklabels([str(t.year) for t in year_ticks], fontsize=TICK_SIZE)
    axes[-1].set_xlabel("Year", fontsize=LABEL_SIZE)


    # Save
    saved: list[Path] = []
    stem = figures_dir / OUTPUT_STEM

    if SAVE_PDF:
        p = stem.with_suffix(".pdf")
        fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.15)
        print(f"  Saved PDF : {p}")
        saved.append(p)
    if SAVE_TIFF:
        p = stem.with_suffix(".tiff")
        fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.15, pil_kwargs={"compression": "tiff_lzw"})
        print(f"  Saved TIFF : {p}")
        saved.append(p)

    plt.close(fig)
    return saved

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly NO2 time series, land-only, four cities, 2019-2024."
    )
    parser.add_argument("--input", type=Path, required=True)
                       # help="option_B_landonly_input.csv")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures") / "plots")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = load_data(args.input)
    print(f"  {len(df)} rows | cities: {df['city'].astype(str).unique().tolist()} | "
          f"range: {df['month_start'].min().date()} to {df['month_start'].max().date()}")

    paths = plot_monthly_timeseries(df, args.figures_dir)
    print("Saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
