"""
Added 16 October 2026
deseasonalized_series.py

PURPOSE
    Convert each city's monthly tropospheric NO2 column series into
    anomalies relative to its calendar-month climatology, define high-NO2
    flags on those anomalies using the city's own 75th percentile, and
    write the flags to a CSV for the run-length tests.

INPUT
    option_B_landonly_input.csv
        288 rows = 4 cities x 72 months (2019-01 to 2024-12).
        Required columns:
            city                       : city name
            month_start                : first day of month (YYYY-MM-DD)
            no2_tropospheric_column    : monthly city-mean NO2 column
            high_no2_flag              : raw-threshold flag (used only for comparison)

OUTPUT
    deseasonalized_flags.csv
        One row per city-month. Columns:
            city, month_start, cal_month,
            no2_column, climatology, anomaly,
            anomaly_p75_threshold, anomaly_high_flag,
            raw_high_flag

    deseasonalized_summary.csv
        One row per city. Columns:
            city, n_months, n_climatology_years, n_high_anomaly,
            n_high_raw, n_agree, anomaly_p75_threshold
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_CSV = Path("option_B_landonly_input.csv")
OUTPUT_FLAGS = Path("deseasonalized_flags.csv")
OUTPUT_SUMMARY = Path("deseasonalized_summary.csv")

PERCENTILE = 75          # same percentile as the raw-threshold analysis
EXPECTED_MONTHS = 72     # 6 years x 12 months per city
EXPECTED_CITIES = 4


# ---------------------------------------------------------------------------
# 2. LOAD AND VALIDATE INPUT
# ---------------------------------------------------------------------------
def load_input(path: Path) -> pd.DataFrame:
    """Read the monthly table and check structure before any calculation."""
    df = pd.read_csv(path)

    required = {"city", "month_start", "no2_tropospheric_column", "high_no2_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    df["month_start"] = pd.to_datetime(df["month_start"])

    counts = df.groupby("city").size()
    if (counts != EXPECTED_MONTHS).any():
        raise ValueError(f"Expected {EXPECTED_MONTHS} months per city, found:\n{counts}")
    if df["city"].nunique() != EXPECTED_CITIES:
        raise ValueError(f"Expected {EXPECTED_CITIES} cities, found {df['city'].nunique()}")
    if df["no2_tropospheric_column"].isna().any():
        raise ValueError("Missing NO2 values found; resolve before deseasonalizing.")

    return df


# ---------------------------------------------------------------------------
# 3. CALENDAR-MONTH CLIMATOLOGY AND ANOMALIES (per city)
# ---------------------------------------------------------------------------
def add_anomalies(df_city: pd.DataFrame) -> pd.DataFrame:
    """Subtract each calendar month's mean (over all years) from that month's values."""
    out = df_city.sort_values("month_start").copy()
    out["cal_month"] = out["month_start"].dt.month

    # Climatology: mean for each calendar month over the 6 years.
    # Each value is based on 6 observations.
    out["climatology"] = out.groupby("cal_month")["no2_tropospheric_column"].transform("mean")
    out["anomaly"] = out["no2_tropospheric_column"] - out["climatology"]
    return out


# ---------------------------------------------------------------------------
# 4. ANOMALY HIGH-MONTH FLAGS (per city)
# ---------------------------------------------------------------------------
def add_anomaly_flags(df_city: pd.DataFrame) -> pd.DataFrame:
    """Flag months whose anomaly is at or above the city's own 75th percentile."""
    out = df_city.copy()
    threshold = np.percentile(out["anomaly"].to_numpy(), PERCENTILE)
    out["anomaly_p75_threshold"] = threshold
    out["anomaly_high_flag"] = (out["anomaly"] >= threshold).astype(int)
    return out


# ---------------------------------------------------------------------------
# 5. CHECKS (per city)
# ---------------------------------------------------------------------------
def check_city(out: pd.DataFrame, city: str) -> dict:
    """Confirm flag counts and compare with the raw-threshold flags."""
    n_high_anom = int(out["anomaly_high_flag"].sum())
    n_high_raw = int(out["high_no2_flag"].sum())
    n_agree = int((out["anomaly_high_flag"] == out["high_no2_flag"]).sum())

    # Expect about 18 high months for a 75th-percentile threshold on 72 values.
    # Ties at the threshold can change the count by one or two.
    if abs(n_high_anom - 18) > 2:
        print(f"WARNING {city}: anomaly high count {n_high_anom} is far from 18")

    print(f"{city}: anomaly high = {n_high_anom}, raw high = {n_high_raw}, agree = {n_agree}/72")

    return {
        "city": city,
        "n_months": len(out),
        "n_climatology_years": out["month_start"].dt.year.nunique(),
        "n_high_anomaly": n_high_anom,
        "n_high_raw": n_high_raw,
        "n_agree": n_agree,
        "anomaly_p75_threshold": out["anomaly_p75_threshold"].iloc[0],
    }


# ---------------------------------------------------------------------------
# 6. MAIN: PROCESS ALL CITIES AND WRITE OUTPUT
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_input(INPUT_CSV)

    flag_frames = []
    summary_rows = []

    for city, df_city in df.groupby("city", sort=False):
        print(f"\nProcessing {city} ...")
        out = add_anomalies(df_city)
        out = add_anomaly_flags(out)
        summary_rows.append(check_city(out, city))

        out["raw_high_flag"] = out["high_no2_flag"]
        out = out.rename(columns={"no2_tropospheric_column": "no2_column"})
        flag_frames.append(out[[
            "city", "month_start", "cal_month",
            "no2_column", "climatology", "anomaly",
            "anomaly_p75_threshold", "anomaly_high_flag",
            "raw_high_flag",
        ]])

    flags = pd.concat(flag_frames, ignore_index=True)
    flags["month_start"] = flags["month_start"].dt.strftime("%Y-%m-%d")
    flags.to_csv(OUTPUT_FLAGS, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\nSaved {OUTPUT_FLAGS.resolve()}")
    print(f"Saved {OUTPUT_SUMMARY.resolve()}")
    print("\nSummary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()