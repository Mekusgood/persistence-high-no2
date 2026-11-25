#!/usr/bin/env python3
"""
fit_logistic_optionB.py

Binomial logistic regression of high-NO2 month occurrence (Option B flags:
72 high months, 18 per city) for the manuscript's Table 5 / Table S1 and the
logistic row of Table 6 (AUC, EPV).

Models
  Base      : y ~ wind + precip + BLH + city + season   (Lagos, DJF reference)
  Ventilation: y ~ precip + VI + city + season          (Lagos, DJF reference)

Outputs
  - odds ratios, 95% CIs, Wald p-values for both models
  - AIC for both models
  - likelihood-ratio test for season (3 df) in both models
  - in-sample AUC (base model)
  - leave-one-year-out cross-validated AUC (base model), pooled out-of-fold predictions
  - EPV for both models

Input : option_B_landonly_input.csv (288 rows, city-month)
Usage : python fit_logistic_optionB.py --input option_B_landonly_input.csv
Requires: pandas, numpy, statsmodels, scikit-learn, scipy
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

BASE = "y ~ wind + precip + blh + C(city, Treatment('Lagos')) + C(season_4, Treatment('DJF'))"
VENT = "y ~ precip + vi + C(city, Treatment('Lagos')) + C(season_4, Treatment('DJF'))"
BASE_NOSEASON = "y ~ wind + precip + blh + C(city, Treatment('Lagos'))"
VENT_NOSEASON = "y ~ precip + vi + C(city, Treatment('Lagos'))"


def load(path):
    df = pd.read_csv(path).rename(columns={
        "wind_speed_ms": "wind",
        "precipitation_mm": "precip",
        "boundary_layer_height_m": "blh",
        "ventilation_index_m2s": "vi",
    })
    df["y"] = df["high_no2_flag"].astype(int)
    df["year"] = df["year"].astype(int)
    assert len(df) == 288, f"expected 288 city-months, got {len(df)}"
    assert df["y"].sum() == 72, f"expected 72 high months, got {df['y'].sum()}"
    return df


def odds_table(model):
    ci = model.conf_int()
    return pd.DataFrame({
        "coef": model.params,
        "OR": np.exp(model.params),
        "CI_low": np.exp(ci[0]),
        "CI_high": np.exp(ci[1]),
        "p": model.pvalues,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    df = load(args.input)

    base = smf.logit(BASE, data=df).fit(disp=0)
    vent = smf.logit(VENT, data=df).fit(disp=0)
    base_ns = smf.logit(BASE_NOSEASON, data=df).fit(disp=0)
    vent_ns = smf.logit(VENT_NOSEASON, data=df).fit(disp=0)

    pd.set_option("display.width", 120)
    print("=== Base model ===")
    print(odds_table(base).round(4).to_string())
    print("AIC =", round(base.aic, 2))
    print("\n=== Ventilation model ===")
    print(odds_table(vent).round(4).to_string())
    print("AIC =", round(vent.aic, 2))

    print("\n=== Season likelihood-ratio tests (3 df) ===")
    for name, full, red in [("base", base, base_ns), ("ventilation", vent, vent_ns)]:
        lr = 2 * (full.llf - red.llf)
        print(f"{name}: LR = {lr:.3f}, p = {chi2.sf(lr, 3):.3f}")

    print("\n=== Discrimination (base model) ===")
    auc_in = roc_auc_score(df["y"], base.predict(df))
    oof = np.zeros(len(df))
    for yr in sorted(df["year"].unique()):
        train = df[df["year"] != yr]
        test = df[df["year"] == yr]
        fold = smf.logit(BASE, data=train).fit(disp=0)
        oof[test.index] = fold.predict(test)
    auc_cv = roc_auc_score(df["y"], oof)
    print(f"In-sample AUC = {auc_in:.3f}")
    print(f"Leave-one-year-out AUC (pooled out-of-fold) = {auc_cv:.3f}")
    print(f"Difference = {auc_in - auc_cv:.3f}")

    print("\n=== EPV ===")
    print(f"Base (9 predictors): {72/9:.1f}")
    print(f"Ventilation (8 predictors): {72/8:.1f}")


if __name__ == "__main__":
    main()
