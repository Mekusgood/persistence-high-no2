#!/usr/bin/env python3
"""
Last modified: 11/25/2026
M3 diagnostics for HAQAST Sentinel-5P TROPOMI NO2 Level 3 GLOBAL monthly files (v2.4, QA75).

Source facts (README, Jan 2025):
  - Variables: Tropospheric_NO2 (molec cm-2, float32), Number_obs (count, float32).
  - Undefined value for both variables: -999.0 (Table 3.3). Not NaN in the raw data.
  - Grid: 0.1 deg, Latitude 1350 (-60.0 to 75.0), Longitude 3590 (-179.5 to 179.5).
  - Number_obs = number of valid pixels used for the average (qa_value >= 0.75).
  - Algorithm eras (Sec 1.1): RPRO to 25 Jul 2022; OFFL 26 Jul 2022 to 6 Sep 2024;
    L2 operational algorithm 2.7.1 from 7 Sep 2024 (flagged, not excluded).

Outputs (in --out-dir):
  m3_city_coverage.csv  one row per file x city x buffer radius
  m3_file_summary.csv   one row per file: metadata + whole-grid QC counts
  m3_file_attrs.json    global attributes and variable attributes per file

Units: 1 mol m-2 = 6.022e19 molec cm-2, so mol m-2 = molec cm-2 / 6.022e19.

Usage:
  python m3_tropomi_diagnostics.py --files A.nc4 B.nc4 --out-dir m3_out
  python m3_tropomi_diagnostics.py --data-dir /path/to/nc4s --out-dir m3_out
Optional land mask: pip install global-land-mask (otherwise land_fraction is left blank).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FILL_VALUE = -999.0
FILL_TEST = -998.0            # anything at or below this is treated as undefined
MOLEC_CM2_PER_MOL_M2 = 6.022e19
R_EARTH_KM = 6371.0088

CITIES = {                    # (lat, lon), as used in the GEE scripts
    "Lagos": (6.5244, 3.3792),
    "Kinshasa": (-4.4419, 15.2663),
    "Dar es Salaam": (-6.7924, 39.2083),
    "Johannesburg": (-26.2041, 28.0473),
}
RADII_KM = [20, 30, 40]
NOBS_THRESHOLDS = [5, 10, 15]  # for the global share of cells with few valid pixels

GLOBAL_ATTRS = [
    "ShortName", "LongName", "VersionID", "ProcessingLevel", "DataSetQuality",
    "source", "institution", "history", "comment", "ProductionDateTime",
    "RangeBeginningDate", "RangeEndingDate", "GranuleID", "Conventions",
]

FILENAME_RE = re.compile(r"Monthly_(\d{2})(\d{4})_V(\d[._]\d)_(\d{8})\.nc4$")  # accepts V2.4 or V2_4

try:
    from global_land_mask import globe
    HAVE_LAND_MASK = True
except ImportError:
    HAVE_LAND_MASK = False


def s5p_era(month_start: pd.Timestamp) -> str:
    """Algorithm era of the Level 2 input, from README Sec 1.1."""
    if month_start < pd.Timestamp("2022-07-26"):
        return "RPRO (v2.4-2.6)"
    if month_start <= pd.Timestamp("2024-09-06"):
        return "OFFL (v2.4-2.6)"
    return "OFFL (v2.7.1 from 2024-09-07)"


def parse_filename(path: Path):
    m = FILENAME_RE.search(path.name)
    if not m:
        return pd.NaT, None, None
    month, year = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1), m.group(3), m.group(4)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def load_grid_arrays(ds: xr.Dataset):
    """Return NO2 and Number_obs as (Latitude, Longitude) numpy arrays, raw (no decoding)."""
    out = {}
    for name in ["Tropospheric_NO2", "Number_obs"]:
        da = ds[name]
        if set(da.dims) != {"Latitude", "Longitude"}:
            raise ValueError(f"{name} has unexpected dims {da.dims}")
        out[name] = da.transpose("Latitude", "Longitude").values.astype(np.float64)
    return out["Tropospheric_NO2"], out["Number_obs"]


def masks(no2_raw, nobs_raw):
    """Boolean masks used for every summary. Raw values, so -999 is still visible."""
    no2_fill = no2_raw <= FILL_TEST
    no2_nan = ~np.isfinite(no2_raw)
    no2_ok = ~no2_fill & ~no2_nan
    no2_neg = no2_ok & (no2_raw < 0)

    nobs_fill = nobs_raw <= FILL_TEST
    nobs_nan = ~np.isfinite(nobs_raw)
    nobs_ok = ~nobs_fill & ~nobs_nan
    nobs_zero = nobs_ok & (nobs_raw <= 0)

    usable = no2_ok & nobs_ok & (nobs_raw > 0)  # NO2 present AND at least one valid pixel
    return dict(no2_fill=no2_fill, no2_nan=no2_nan, no2_ok=no2_ok, no2_neg=no2_neg,
                nobs_fill=nobs_fill, nobs_nan=nobs_nan, nobs_zero=nobs_zero, usable=usable)


def city_metrics(lat, lon, no2_raw, nobs_raw, m, clat, clon, radius_km, land_flag):
    """Metrics for one city buffer. Cell centres inside the buffer are selected."""
    pad = 40 / 111.0 + 0.3  # window big enough for the largest radius
    ilat = np.where(np.abs(lat - clat) <= pad)[0]
    ilon = np.where(np.abs(lon - clon) <= pad / np.cos(np.radians(clat)))[0]
    LAT, LON = np.meshgrid(lat[ilat], lon[ilon], indexing="ij")
    dist = haversine_km(LAT, LON, clat, clon)

    def sub(arr):
        return arr[np.ix_(ilat, ilon)]

    sel = dist <= radius_km
    n_cells = int(sel.sum())

    usable = sub(m["usable"]) & sel
    no2_ok = sub(m["no2_ok"]) & sel
    nobs_vals = sub(nobs_raw)[usable]
    no2_vals = sub(no2_raw)[usable]

    row = dict(
        radius_km=radius_km,
        cells_in_buffer=n_cells,
        cells_usable=int(usable.sum()),
        coverage_frac=usable.sum() / n_cells if n_cells else np.nan,
        cells_fill_no2=int((sub(m["no2_fill"]) & sel).sum()),
        cells_nan_no2=int((sub(m["no2_nan"]) & sel).sum()),
        cells_negative_no2=int((sub(m["no2_neg"]) & sel).sum()),
        cells_nobs_zero_or_fill=int(((sub(m["nobs_zero"]) | sub(m["nobs_fill"])) & sel).sum()),
        nobs_min=float(nobs_vals.min()) if nobs_vals.size else np.nan,
        nobs_median=float(np.median(nobs_vals)) if nobs_vals.size else np.nan,
        nobs_max=float(nobs_vals.max()) if nobs_vals.size else np.nan,
        no2_mean_molec_cm2=float(no2_vals.mean()) if no2_vals.size else np.nan,
        no2_median_molec_cm2=float(np.median(no2_vals)) if no2_vals.size else np.nan,
    )
    row["no2_mean_mol_m2"] = row["no2_mean_molec_cm2"] / MOLEC_CM2_PER_MOL_M2
    row["no2_median_mol_m2"] = row["no2_median_molec_cm2"] / MOLEC_CM2_PER_MOL_M2
    row["cells_ok_no2_only"] = int(no2_ok.sum())

    if land_flag is not None:
        land = land_flag[np.ix_(ilat, ilon)]
        row["land_fraction"] = float(land[sel].mean()) if n_cells else np.nan
    else:
        row["land_fraction"] = np.nan
    return row


def file_qc(lat, lon, no2_raw, nobs_raw, m):
    """Whole-grid QC counts for one file."""
    total = no2_raw.size
    usable = m["usable"]
    nobs_ok = nobs_raw[~m["nobs_fill"] & ~m["nobs_nan"]]
    qc = dict(
        grid_cells=int(total),
        pct_no2_fill=100 * m["no2_fill"].sum() / total,
        pct_no2_nan=100 * m["no2_nan"].sum() / total,
        pct_no2_negative=100 * m["no2_neg"].sum() / total,
        pct_usable=100 * usable.sum() / total,
        nobs_median_usable=float(np.median(nobs_raw[usable])) if usable.any() else np.nan,
        nobs_p05_usable=float(np.percentile(nobs_raw[usable], 5)) if usable.any() else np.nan,
        nobs_p95_usable=float(np.percentile(nobs_raw[usable], 95)) if usable.any() else np.nan,
        nobs_max=float(nobs_ok.max()) if nobs_ok.size else np.nan,
    )
    for t in NOBS_THRESHOLDS:
        qc[f"pct_usable_nobs_lt_{t}"] = (
            100 * (usable & (nobs_raw < t)).sum() / max(usable.sum(), 1)
        )
    return qc


def attrs_dump(ds: xr.Dataset):
    return {
        "global": {k: str(ds.attrs.get(k, "")) for k in GLOBAL_ATTRS},
        "variables": {
            v: {k: str(ds[v].attrs.get(k, "")) for k in
                ["long_name", "units", "_FillValue", "missing_value", "valid_range"]}
            for v in ["Tropospheric_NO2", "Number_obs"]
        },
        "encoding": {
            v: {k: str(ds[v].encoding.get(k, "")) for k in ["dtype", "_FillValue", "scale_factor"]}
            for v in ["Tropospheric_NO2", "Number_obs"]
        },
    }


def process_file(path: Path, radii, land_flag):
    month_start, version, created = parse_filename(path)
    # mask_and_scale=False keeps -999 visible so the QC counts are honest.
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        lat = ds["Latitude"].values.astype(np.float64)
        lon = ds["Longitude"].values.astype(np.float64)
        no2_raw, nobs_raw = load_grid_arrays(ds)
        attrs = attrs_dump(ds)
        rng_begin = ds.attrs.get("RangeBeginningDate", "")

    m = masks(no2_raw, nobs_raw)
    qc = file_qc(lat, lon, no2_raw, nobs_raw, m)

    city_rows = []
    for city, (clat, clon) in CITIES.items():
        for r in radii:
            row = city_metrics(lat, lon, no2_raw, nobs_raw, m, clat, clon, r, land_flag)
            row.update(city=city, file=path.name, month_start=month_start.date() if pd.notna(month_start) else None)
            city_rows.append(row)

    summary = dict(
        file=path.name,
        month_start=month_start.date() if pd.notna(month_start) else None,
        filename_version=version,
        filename_created=created,
        RangeBeginningDate=rng_begin,
        ProductionDateTime=attrs["global"]["ProductionDateTime"],
        VersionID=attrs["global"]["VersionID"],
        ProcessingLevel=attrs["global"]["ProcessingLevel"],
        s5p_era=s5p_era(month_start) if pd.notna(month_start) else None,
        NO2_units=attrs["variables"]["Tropospheric_NO2"]["units"],
        NO2_fill_attr=attrs["variables"]["Tropospheric_NO2"]["_FillValue"],
        **qc,
    )
    return city_rows, summary, attrs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="+", type=Path)
    src.add_argument("--data-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("m3_out"))
    ap.add_argument("--radii", type=int, nargs="+", default=RADII_KM)
    args = ap.parse_args()

    files = args.files if args.files else sorted(args.data_dir.glob("HAQ_TROPOMI_NO2_GLOBAL_QA75_L3_Monthly_*.nc4"))
    if not files:
        raise SystemExit("No input files found.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    land_flag = None
    if HAVE_LAND_MASK:
        # Build a land flag on the 0.1 deg grid once; reused for every file.
        with xr.open_dataset(files[0], mask_and_scale=False) as ds0:
            lat0 = ds0["Latitude"].values
            lon0 = ds0["Longitude"].values
        LAT, LON = np.meshgrid(lat0, lon0, indexing="ij")
        land_flag = globe.is_land(LAT, LON)
    else:
        print("global_land_mask not installed: land_fraction will be NaN.")

    all_city, all_summary, all_attrs = [], [], {}
    for f in files:
        print(f"Processing {f.name}")
        city_rows, summary, attrs = process_file(f, args.radii, land_flag)
        all_city.extend(city_rows)
        all_summary.append(summary)
        all_attrs[f.name] = attrs

    city_df = pd.DataFrame(all_city)
    city_df = city_df[["file", "month_start", "city", "radius_km", "cells_in_buffer", "cells_usable",
                       "coverage_frac", "cells_fill_no2", "cells_nan_no2", "cells_negative_no2",
                       "cells_nobs_zero_or_fill", "nobs_min", "nobs_median", "nobs_max",
                       "no2_mean_molec_cm2", "no2_median_molec_cm2", "no2_mean_mol_m2",
                       "no2_median_mol_m2", "land_fraction", "cells_ok_no2_only"]]
    summary_df = pd.DataFrame(all_summary)

    city_df.to_csv(args.out_dir / "m3_city_coverage.csv", index=False)
    summary_df.to_csv(args.out_dir / "m3_file_summary.csv", index=False)
    (args.out_dir / "m3_file_attrs.json").write_text(json.dumps(all_attrs, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n== City coverage (30 km and sensitivity radii) ==")
    print(city_df[["file", "city", "radius_km", "cells_in_buffer", "cells_usable", "coverage_frac",
                   "cells_negative_no2", "nobs_median", "no2_mean_mol_m2"]].to_string(index=False))
    print("\n== File QC ==")
    print(summary_df[["file", "VersionID", "s5p_era", "ProductionDateTime", "pct_no2_fill",
                      "pct_no2_negative", "pct_usable", "nobs_median_usable"]].to_string(index=False))
    print(f"\nWrote {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
