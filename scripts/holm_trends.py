"""
holm_trends.py (version 2)

PURPOSE
    Apply Holm's step-down correction across the four per-city trend tests.

INPUT
    v3_m2_trend_summary.csv  (columns: city, p_trend)

OUTPUT
    v3_m2_trend_holm.csv     (columns: city, p_trend, p_holm)
"""

from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

INPUT_CSV = Path("v3_m2_trend_summary.csv")
OUTPUT_CSV = Path("v3_m2_trend_holm.csv")

df = pd.read_csv(INPUT_CSV)
reject, p_holm, _, _ = multipletests(df["p_trend"], alpha=0.05, method="holm")
df["p_holm"] = p_holm
df[["city", "p_trend", "p_holm"]].to_csv(OUTPUT_CSV, index=False)
print(df[["city", "p_trend", "p_holm"]].round(4).to_string(index=False))
print(f"\nSaved {OUTPUT_CSV.resolve()}")