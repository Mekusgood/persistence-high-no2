"""
Last modified: Nov 16, 2026
Final run-length tests for NO2 persistence (land-only GEE series, Option B thresholds).

Monthly series: exchangeable null, season-preserving null (permute flags within season).
Anomaly series: exchangeable null.
Bootstrap: six non-overlapping annual blocks, resampled with replacement (matches Methods).
p-values: permutation p = (k + 1) / (N + 1); reported as "<0.0001" when k = 0.
Holm: step-down over 12 tests per null and series (4 cities x 3 statistics).
"""
from pathlib import Path
import numpy as np
import pandas as pd

MONTHLY = Path("option_B_landonly_input.csv")
ANOMALY = Path("deseasonalized_flags.csv")
OUT = Path("final_run_tests.csv")

N_PERM = 10_000
N_BOOT = 10_000
SEED = 20261016
STATS = ["runs", "mean_run", "longest_run"]
CITIES = ["Lagos", "Kinshasa", "Dar es Salaam", "Johannesburg"]
rng = np.random.default_rng(SEED)


def run_stats(flags):
    flags = np.asarray(flags, dtype=int)
    d = np.diff(np.concatenate(([0], flags, [0])))
    L = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    if L.size == 0:
        return np.zeros(3)
    return np.array([L.size, L.mean(), L.max()], dtype=float)


def exchangeable_null(flags):
    base = np.zeros(flags.size, dtype=int)
    base[: int(flags.sum())] = 1
    return np.array([run_stats(rng.permutation(base)) for _ in range(N_PERM)])


def season_null(flags, seasons):
    uniq = np.unique(seasons)
    out = np.empty((N_PERM, 3))
    for i in range(N_PERM):
        new = flags.copy()
        for s in uniq:
            idx = np.flatnonzero(seasons == s)
            new[idx] = rng.permutation(flags[idx])
        out[i] = run_stats(new)
    return out


def annual_block_ci(flags):
    blocks = [flags[i:i + 12] for i in range(0, 72, 12)]  # six non-overlapping years
    boots = np.array([run_stats(np.concatenate([blocks[k] for k in rng.integers(0, 6, 6)]))
                      for _ in range(N_BOOT)])
    return np.percentile(boots, [2.5, 97.5], axis=0)


def p_plus1(null_col, obs, stat):
    if stat == "runs":
        k = int((null_col <= obs).sum())
    else:
        k = int((null_col >= obs).sum())
    return (k + 1) / (N_PERM + 1), k


def holm(pvals):
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def analyse(series_name, df, use_season):
    rows = []
    for city in CITIES:
        g = df[df["city"] == city].sort_values("month_start").reset_index(drop=True)
        assert len(g) == 72, city
        flags = g["flag"].to_numpy(dtype=int)
        obs = run_stats(flags)
        ex = exchangeable_null(flags)
        se = season_null(flags, g["season"].to_numpy()) if use_season else None
        ci = annual_block_ci(flags)
        for j, stat in enumerate(STATS):
            pe, ke = p_plus1(ex[:, j], obs[j], stat)
            row = dict(series=series_name, city=city, statistic=stat,
                       n_high=int(flags.sum()), observed=obs[j],
                       ci95_low=ci[0, j], ci95_high=ci[1, j],
                       p_exchangeable=pe, k_exchangeable=ke)
            if use_season:
                ps, ks = p_plus1(se[:, j], obs[j], stat)
                row.update(p_season_preserving=ps, k_season=ks)
            rows.append(row)
    return rows


def main():
    m = pd.read_csv(MONTHLY)
    m = m.rename(columns={"high_no2_flag": "flag", "season_4": "season"})
    a = pd.read_csv(ANOMALY).rename(columns={"anomaly_high_flag": "flag"})
    a["season"] = None

    rows = analyse("monthly", m, use_season=True) + analyse("anomaly", a, use_season=False)
    res = pd.DataFrame(rows)

    # Holm families: 12 tests per null and series
    res["p_exchangeable_holm"] = np.nan
    res.loc[res.series == "monthly", "p_exchangeable_holm"] = holm(res.loc[res.series == "monthly", "p_exchangeable"])
    res.loc[res.series == "anomaly", "p_exchangeable_holm"] = holm(res.loc[res.series == "anomaly", "p_exchangeable"])
    res["p_season_holm"] = np.nan
    mm = res.series == "monthly"
    res.loc[mm, "p_season_holm"] = holm(res.loc[mm, "p_season_preserving"])

    res.to_csv(OUT, index=False)
    show = res.copy()
    for c in ["p_exchangeable", "p_exchangeable_holm", "p_season_preserving", "p_season_holm"]:
        if c in show:
            show[c] = show[c].map(lambda x: "<0.0001" if pd.notna(x) and x < 0.0001 else (f"{x:.4f}" if pd.notna(x) else ""))
    print(show.drop(columns=["k_exchangeable"] + (["k_season"] if "k_season" in show else [])).round(3).to_string(index=False))
    print(f"\nSaved {OUT.resolve()}")


if __name__ == "__main__":
    main()
