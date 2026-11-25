"""
sensitivity_analysis.py  (revised) v3.0

PURPOSE
    Sensitivity analyses
      A. Pooled four-city 75th percentile (absolute column cut).
      B. Detrended series: city P75 of residuals after linear trend removal.
      C. Rolling local threshold: centred 36-month P75.
      D. 2020-excluded: city P75 recomputed on 60 months (2019, 2021-2024).
    Exchangeable null only. Each test has its own seeded generator.
    Per-city linear trends are also reported.

INPUT
    option_B_landonly_input.csv
        Required columns: city, month_start, no2_tropospheric_column (mol m-2),
                          city_no2_p75 (mol m-2, CSV threshold)

OUTPUT
    v3_m2_trend_summary.csv
    v3_m2_sensitivity_flags.csv   (includes value_molec_cm2)
    v3_m2_sensitivity_tests.csv   (includes threshold_basis, n_months)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_CSV = Path("option_B_landonly_input.csv")
OUT_TRENDS = Path("v3_m2_trend_summary.csv")
OUT_FLAGS = Path("v3_m2_sensitivity_flags.csv")
OUT_TESTS = Path("v3_m2_sensitivity_tests.csv")

PCT = 75
ROLL_WINDOW = 36
ROLL_MIN = 12
N_PERM = 10_000
SEED = 20261016
MOL_TO_MOLEC = 6.022e19

# ---------------------------------------------------------------------------
# 2. LOAD AND VALIDATE INPUT
# ---------------------------------------------------------------------------
def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"city", "month_start", "no2_tropospheric_column", "city_no2_p75"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    df["month_start"] = pd.to_datetime(df["month_start"])
    df = df.sort_values(["city", "month_start"]).reset_index(drop=True)
    if (df.groupby("city").size() != 72).any():
        raise ValueError("Expected 72 months per city.")
    return df

# ---------------------------------------------------------------------------
# 3. RUN STATISTICS
# ---------------------------------------------------------------------------
def run_stats(flags: np.ndarray) -> np.ndarray:
    """Return [runs, mean run length, longest run]."""
    flags = np.asarray(flags, dtype=int)
    padded = np.concatenate(([0], flags, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    lengths = ends - starts
    if lengths.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.array([lengths.size, lengths.mean(), lengths.max()], dtype=float)

# ---------------------------------------------------------------------------
# 4. EXCHANGEABLE NULL WITH A SEEDED GENERATOR PER TEST
# ---------------------------------------------------------------------------
def exchangeable_pvalues(flags: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided p-values from a generator seeded for this test only."""
    rng = np.random.default_rng(seed)
    k = int(flags.sum())
    base = np.zeros(flags.size, dtype=int)
    base[:k] = 1
    null = np.empty((N_PERM, 3))
    for i in range(N_PERM):
        null[i] = run_stats(rng.permutation(base))
    obs = run_stats(flags)
    p = np.array([
        (null[:, 0] <= obs[0]).mean(),   # runs: fewer than null
        (null[:, 1] >= obs[1]).mean(),   # mean run: greater than null
        (null[:, 2] >= obs[2]).mean(),   # longest run: greater than null
    ])
    return obs, p

# ---------------------------------------------------------------------------
# 5. TRENDS (Section B)
# ---------------------------------------------------------------------------
def city_trend(y: np.ndarray) -> dict:
    t = np.arange(y.size) / 12.0
    fit = stats.linregress(t, y)
    return {
        "intercept": fit.intercept,
        "slope_mol_m2_per_yr": fit.slope,
        "slope_pct_of_mean_per_yr": 100.0 * fit.slope / y.mean(),
        "r": fit.rvalue,
        "p_trend": fit.pvalue,
    }

# ---------------------------------------------------------------------------
# 6. FLAG CONSTRUCTION
# ---------------------------------------------------------------------------
def flags_detrended(y, intercept, slope):
    t = np.arange(y.size) / 12.0
    resid = y - (intercept + slope * t)
    return (resid >= np.percentile(resid, PCT)).astype(int)

def flags_rolling(y):
    thr = (pd.Series(y)
           .rolling(ROLL_WINDOW, center=True, min_periods=ROLL_MIN)
           .quantile(PCT / 100.0).to_numpy())
    return (y >= thr).astype(int), thr

def flags_2020_excluded(y, years):
    """Recompute P75 on the 60 non-2020 months; flag all 72 months against it."""
    keep = years != 2020
    thr = np.percentile(y[keep], PCT)
    return (y >= thr).astype(int), thr, keep

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_input(INPUT_CSV)

    # Pooled absolute cut: numpy P75 of all 288 values (one stated exception).
    pooled_thr = np.percentile(df["no2_tropospheric_column"].to_numpy(), PCT)
    print(f"Pooled P75 (numpy, 288 values): {pooled_thr * MOL_TO_MOLEC:.4e} molec cm-2")

    trend_rows, flag_frames, test_rows = [], [], []

    for idx, (city, g) in enumerate(df.groupby("city", sort=False)):
        y = g["no2_tropospheric_column"].to_numpy(dtype=float)
        years = g["month_start"].dt.year.to_numpy()
        months = g["month_start"].dt.strftime("%Y-%m-%d").to_numpy()
        csv_thr = float(g["city_no2_p75"].iloc[0])   # CSV threshold (mol m-2)

        tr = city_trend(y)
        trend_rows.append({"city": city,
                           **{k: v for k, v in tr.items() if k != "intercept"}})

        flag_ref = (y >= csv_thr).astype(int)
        flag_A = (y >= pooled_thr).astype(int)
        flag_B = flags_detrended(y, tr["intercept"], tr["slope_mol_m2_per_yr"])
        flag_C, thr_C = flags_rolling(y)
        flag_D, thr_D, keep = flags_2020_excluded(y, years)

        scenarios = {
            "Reference_CSV_P75": (flag_ref, csv_thr * MOL_TO_MOLEC, 72,
                                  "CSV city P75 (column)"),
            "A_pooled_P75": (flag_A, pooled_thr * MOL_TO_MOLEC, 72,
                             "pooled P75 over 288 values (numpy)"),
            "B_detrended_P75": (flag_B, np.nan, 72,
                                "detrended residual P75 (not a column value)"),
            "C_rolling36_P75": (flag_C, np.nanmedian(thr_C) * MOL_TO_MOLEC, 72,
                                "median of centred 36-month rolling P75"),
            "D_excl2020_P75": (flag_D, thr_D * MOL_TO_MOLEC, 60,
                               "city P75 over 60 non-2020 months"),
        }

        for s_idx, (name, (flags, thr_molec, n_months, basis)) in enumerate(scenarios.items()):
            # Seed = SEED + city index * 10 + scenario index: stable and unique.
            seed = SEED + idx * 10 + s_idx
            obs, p = exchangeable_pvalues(flags, seed)
            test_rows.append({
                "scenario": name,
                "city": city,
                "n_months": n_months,
                "n_high": int(flags.sum()),
                "threshold_molec_cm2": thr_molec,
                "threshold_basis": basis,
                "runs": obs[0],
                "mean_run": obs[1],
                "longest_run": obs[2],
                "p_runs": p[0],
                "p_mean_run": p[1],
                "p_longest_run": p[2],
                "seed": seed,
            })

        flag_frames.append(pd.DataFrame({
            "city": city,
            "month_start": months,
            "value_mol_m2": y,
            "value_molec_cm2": y * MOL_TO_MOLEC,
            "flag_ref": flag_ref,
            "flag_A_pooled": flag_A,
            "flag_B_detrended": flag_B,
            "flag_C_rolling": flag_C,
            "flag_D_excl2020": flag_D,
        }))
        print(f"{city}: done")

    pd.DataFrame(trend_rows).to_csv(OUT_TRENDS, index=False)
    pd.concat(flag_frames, ignore_index=True).to_csv(OUT_FLAGS, index=False)
    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT_TESTS, index=False)

    print("\nTests:")
    print(tests.round(4).to_string(index=False))
    print(f"\nSaved {OUT_TRENDS.resolve()}, {OUT_FLAGS.resolve()}, {OUT_TESTS.resolve()}")


if __name__ == "__main__":
    main()