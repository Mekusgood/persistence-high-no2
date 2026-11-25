# Persistence of high-NO₂ months in four sub-Saharan African cities

Analysis code for the manuscript on monthly-scale persistence of high tropospheric NO₂ in Lagos, Kinshasa, Dar es Salaam and Johannesburg, using TROPOMI (Sentinel-5P) data for 2019–2024.

## Contents

```
data/
  option_B_landonly_input.csv   288 city-months (4 cities × 72 months), land-only
results/
  final_run_tests.csv           Run statistics, exchangeable and season-preserving nulls, Holm values
scripts/
  fit_logistic_optionB.py       Logistic regression (base and ventilation models), AIC, odds ratios, leave-one-year-out AUC
  plot_monthly_timeseries.py    Figure 2: monthly NO₂ time series with city-specific P75 threshold (PDF and TIFF)
  figure_s2_rankings.py         Figure S2: bump charts of city rankings; checks against Table 2
  figure_s3_timeseries.py       Figure S3: monthly time series for the supplement
  plot_figure_s1.py             Figure S1: threshold sensitivity (see note below)
```

## Data

`option_B_landonly_input.csv` columns:

| Column | Description |
|---|---|
| `city` | City name |
| `month_start` | First day of the month |
| `no2_tropospheric_column` | Monthly land-only city mean, tropospheric NO₂ column (mol m⁻²) |
| `city_no2_p75` | City-specific 75th percentile of the 72 monthly values (mol m⁻²) |
| `high_no2_flag` | 1 if the month is above the city's P75 |
| `season_4` | DJF, MAM, JJA or SON |
| `wind_speed_ms` | Wind speed (m s⁻¹) |
| `precipitation_mm` | Monthly precipitation (mm) |
| `boundary_layer_height_m` | Boundary-layer height (m) |
| `ventilation_index_m2s` | Ventilation index (m² s⁻¹) |
| `year` | Calendar year |

Unit conversion used in the plotting scripts: 1 mol m⁻² = 6.02214 × 10⁴ × 10¹⁵ molecules cm⁻².

## Running the scripts

Install dependencies, then run each script from the repository root:

```bash
pip install -r requirements.txt

python scripts/fit_logistic_optionB.py --input data/option_B_landonly_input.csv
python scripts/plot_monthly_timeseries.py --input data/option_B_landonly_input.csv --figures-dir figures/plots
python scripts/figure_s2_rankings.py --csv data/option_B_landonly_input.csv --out-dir figures/s2
python scripts/figure_s3_timeseries.py --csv data/option_B_landonly_input.csv --out-dir figures/s3
python scripts/plot_figure_s1.py --outdir figures/s1
```

`plot_monthly_timeseries.py` reads the threshold from `city_no2_p75` and prints a recomputed P75 for comparison; the two should agree to four decimal places.

## Notes

- **Figure S1 values are hard-coded.** `plot_figure_s1.py` draws the threshold-sensitivity figure from per-city values typed into the script (lines 40–43), not from the input CSV. Those values were checked against the Option B analysis output but are not recomputed here.
- **Not included in this folder:** the Google Earth Engine script that extracted the monthly series and P75 thresholds; the stationarity-scenario and trend scripts behind Table S2 and Table S3; and the script that produced `final_run_tests.csv`. These should be added before the repository is cited in the Data Availability statement.
- Logistic regression outputs reproduce the values in the manuscript: base AIC 224.46, ventilation AIC 229.48, cross-validated AUC 0.856.

## Citation

Cite the manuscript once published, and the archived repository DOI (to be added).

## Licence

To be added by the author.
