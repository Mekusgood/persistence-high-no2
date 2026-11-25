# Persistence of high-NO₂ months in four sub-Saharan African cities

Analysis code and data for the manuscript on monthly-scale persistence of high tropospheric NO₂ in Lagos, Kinshasa, Dar es Salaam and Johannesburg, using TROPOMI (Sentinel-5P) data for 2019–2024.

## Contents

```
data/
  option_B_landonly_input.csv     Canonical input: 288 city-months (4 cities × 72 months), land-only
results/
  final_run_tests.csv             Run statistics (runs, mean run, longest run); exchangeable and season-preserving nulls; Holm values
scripts/
  gee/persistence_landonly_OFFL_Oct18_26.js   Google Earth Engine script: TROPOMI monthly land-only series, city P75 thresholds, GeoTIFF export
  fit_logistic_optionB.py         Logistic regression (base and ventilation models), AIC, odds ratios, leave-one-year-out AUC
  final_run_tests.py              Run tests: exchangeable and season-preserving nulls, block-bootstrap CIs, Holm adjustment
  deseasonalized_series.py        Calendar-month climatology; anomaly-based high-month flags (deseasonalised check)
  sensitivity_analysis.py         Stationarity scenarios A–D (pooled, detrended, rolling, 2020 excluded)
  holm_trends.py                  Holm adjustment of the four per-city trend p-values
  tropomi_diagnostics.py          HAQAST L3 coverage diagnostics for the 30 km city buffers (coverage check only)
  fig6_landonly.py                Figure 6: city maps of NO₂ persistence frequency
  plot_monthly_timeseries.py      Figure 2: monthly NO₂ time series with city-specific P75 threshold (PDF and TIFF)
  plot_figure_s1.py               Figure S1: threshold sensitivity; table computed from the canonical input
  figure_s2_rankings.py           Figure S2: bump charts of city rankings; checks against Table 2
  figure_s3_timeseries.py         Figure S3: monthly time series for the supplement
```

## Canonical input

`data/option_B_landonly_input.csv` is the only input file. Every script that needs the monthly series reads this file. Scripts that take no command-line input read it from the current folder, so copy it there before running them.

| Column | Description |
|---|---|
| `city` | City name |
| `month_start` | First day of the month |
| `no2_tropospheric_column` | Monthly land-only city mean, tropospheric NO₂ column (mol m⁻²) |
| `city_no2_p75` | City-specific 75th percentile of the 72 monthly values (mol m⁻²), linear interpolation |
| `high_no2_flag` | 1 if the month is at or above the city's P75 (18 months per city) |
| `season_4` | DJF, MAM, JJA or SON |
| `wind_speed_ms` | Wind speed (m s⁻¹) |
| `precipitation_mm` | Monthly precipitation (mm) |
| `boundary_layer_height_m` | Boundary-layer height (m) |
| `ventilation_index_m2s` | Ventilation index (m² s⁻¹) |
| `year` | Calendar year |

Unit conversion: 1 mol m⁻² = 6.02214 × 10⁴ × 10¹⁵ molecules cm⁻².

## Data sources

- **TROPOMI NO₂:** Sentinel-5P tropospheric NO₂ (`COPERNICUS/S5P/OFFL/L3_NO2`), accessed through Google Earth Engine.
- **Land mask:** JRC Global Surface Water occurrence; pixels with occurrence of 50% or more are excluded.
- **City outlines (Figure 6):** FAO. 2015. Global Administrative Unit Layers (GAUL), 2015 version. https://data.apps.fao.org. Accessed through Google Earth Engine as `FAO/GAUL/2015/level2`. Outlines are not included in this repository because of their size; export them from Earth Engine to a local folder and pass that folder to `fig6_landonly.py` with `--outline-dir`.
- **Coverage diagnostics:** HAQAST TROPOMI NO₂ Level-3 product v2.4 (DOI 10.5067/MKJG22GUOD34), used only by `tropomi_diagnostics.py`.

## Pipeline and run order

1. **Extract the series.** Run `scripts/gee/persistence_landonly_OFFL_Oct18_26.js` in the Earth Engine Code Editor. It exports the four `*_NO2_persistence_frequency_landonly.tif` files and the `*_monthly_city_mean_landonly.csv` files. Combine the monthly CSVs and the covariates into `data/option_B_landonly_input.csv`. *(The covariate extraction and assembly script is not yet in this repository.)*
2. **Coverage check (optional).** `python scripts/tropomi_diagnostics.py --data-dir <HAQAST files> --out-dir diagnostics`
3. **Logistic regression.** `python scripts/fit_logistic_optionB.py --input data/option_B_landonly_input.csv`
4. **Deseasonalised check.** In the folder containing the canonical CSV: `python scripts/deseasonalized_series.py` (writes `deseasonalized_flags.csv` and `deseasonalized_summary.csv`).
5. **Run tests.** In the same folder: `python scripts/final_run_tests.py` (reads the canonical CSV and `deseasonalized_flags.csv`; writes `final_run_tests.csv`).
6. **Stationarity scenarios.** In the same folder: `python scripts/sensitivity_analysis.py`, then `python scripts/holm_trends.py`.
7. **Figures.**
   - `python scripts/plot_monthly_timeseries.py --input data/option_B_landonly_input.csv --figures-dir figures/plots`
   - `python scripts/fig6_landonly.py --data-dir <GEE GeoTIFF folder> --outline-dir <GAUL outline folder> --out-dir figures/fig6`
   - `python scripts/plot_figure_s1.py --input data/option_B_landonly_input.csv --outdir figures/s1`
   - `python scripts/figure_s2_rankings.py --csv data/option_B_landonly_input.csv --out-dir figures/s2`
   - `python scripts/figure_s3_timeseries.py --csv data/option_B_landonly_input.csv --out-dir figures/s3`

Install dependencies first with `pip install -r requirements.txt`.

`plot_monthly_timeseries.py` reads the threshold from `city_no2_p75` and prints a recomputed P75 for comparison; the two should agree to four decimal places.

## Notes

- **Random seeds.** Permutation and bootstrap tests use seed 20261016 (run tests) and per-city seeds derived from it (sensitivity analysis).
- **Figure S1** computes its table from the canonical input and checks that the 75th-percentile row has 18 high months in each city.
- **Not yet in this repository:** the covariate extraction and input assembly script, and the Holm script for the 12-test scenario table (`S2_holm_v3.csv`).

## Citation

Cite the manuscript once published, and the archived repository DOI (to be added).

## Licence

- **Code** (everything under `scripts/`): MIT License. See `LICENSE`.
- **Data and text** (`data/`, `results/`, README): Creative Commons Attribution 4.0 (CC-BY-4.0).
