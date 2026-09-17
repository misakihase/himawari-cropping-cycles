# himawari-cropping-cycles
# Code for Hase et al.

This repository contains the main custom code used in the study by Hase et al. submitted to *Science Advances*.

The scripts cover the principal analyses used to generate the main results of the study, including daily Himawari EVI2 compositing, temporal smoothing, crop phenology and cropping-intensity detection, drought analysis, GPP estimation, GPP sensitivity analysis, and site-level MODIS comparison.

Figure-formatting and plotting scripts are not included because they do not affect the analytical results.

## Overview of the workflow

The main analysis workflow is:

1. Himawari daily observations → daily EVI2 composite
2. Daily EVI2 → LOWESS-smoothed EVI2 time series
3. Smoothed EVI2 → heading, planting, and harvesting dates
4. Phenological dates → annual crop cycles and cropping intensity
5. MSWEP precipitation → regional SPI-3
6. EVI2 → FPAR → GPP
7. Counterfactual GPP experiments → Shapley attribution of GPP variability
8. MODIS site-level EVI2 → phenology comparison with Himawari

## Scripts

### `01_EVI2_composite.py`

Generates daily Himawari EVI2 composites from sub-daily observations.

For each pixel:

- invalid observations are removed using cloud, aerosol, reflectance, and physical-range filters;
- valid observations are ranked by NDVI;
- EVI2 values corresponding to the upper 10% of valid NDVI observations are averaged.

Input daily arrays are assumed to have shape:

```text
(36, 3200, 3500)
```

The output is one daily EVI2 composite with shape:

```text
(3200, 3500)
```

The script processes 2016–2024.

---

### `02_Loess_EVI2.py`

Applies LOWESS smoothing to daily EVI2 time series over cropland pixels.

Main parameter:

```python
LOWESS_SPAN = 0.095
```

Missing values are linearly interpolated before smoothing.

The output is an annual smoothed EVI2 time series for each pixel.

---

### `03_Extract_cropping_cycle.py`

Detects crop phenological events from two consecutive years of smoothed EVI2 observations.

The detected events are:

```text
1 = heading date
2 = planting date
3 = harvesting date
```

Two years are analyzed together so that crop cycles close to calendar-year boundaries can be identified with temporal context.

The planting and harvesting search intervals depend on the number of heading dates detected in the two-year period:

| Heading dates in two years | Planting | Harvesting |
|---|---:|---:|---:|
| 1–2 | ≥50 days before | ≥25 days after |
| 3–4 | ≥45 days before | ≥20 days after |
| 5–6 | ≥40 days before | ≥15 days after |

These groups approximately correspond to single-, double-, and triple-cropping systems.

---

### `04_Extract_DOY.py`

Extracts annual crop-cycle dates from the two-year phenology-event arrays.

For each heading date, the script identifies:

- the nearest preceding planting date;
- the nearest subsequent harvesting date.

Crop cycles are assigned to a calendar year according to the planting date.

The script outputs:

```text
Ddate_<year>_<year+1>.flt
Tdate_<year>_<year+1>.flt
Hdate_<year>_<year+1>.flt
Num_<year>_<year+1>.flt
```

`Num` represents annual cropping intensity, i.e., the number of detected crop cycles.

---

### `05_spi3_calculation.py`

Calculates regional 3-month Standardized Precipitation Index (SPI-3) from monthly MSWEP precipitation.

Study period:

```text
1981–2024
```

Reference period:

```text
1991–2020
```

SPI-3 is calculated by:

1. computing 3-month accumulated precipitation;
2. fitting a gamma distribution separately for each calendar month during the reference period;
3. converting cumulative probabilities to the standard normal distribution.

SPI-3 is calculated separately for:

- Chao Phraya
- Irrawaddy
- Mekong
- Java

---

### `06_GPP_calculation.py`

Calculates daily and annual cropland FPAR and GPP from Himawari-derived EVI2.

FPAR is calculated using pixel-wise minimum–maximum scaling of EVI2 over 2016–2023:

```text
FPAR = (EVI2 - EVI2_min) / (EVI2_max - EVI2_min)
```

FPAR is constrained to the range 0–1.

GPP is calculated using a MOD17-type light-use-efficiency model:

```text
GPP = PAR × FPAR × epsilon
```

where:

```text
epsilon = epsilon_max × f(Tmin) × f(VPD)
```

Main cropland parameters:

```python
EPSILON_MAX = 0.001044  # kg C MJ-1
TMIN_MIN = -8.0         # deg C
TMIN_MAX = 12.02        # deg C
VPD_MIN = 650.0         # Pa
VPD_MAX = 4300.0        # Pa
PAR_FRACTION = 0.45
```

The analysis period is 2016–2023.

---

### `07_annual_FPAR_climate.py`

Calculates annual regional means of:

- FPAR
- PAR
- minimum temperature
- VPD

for the four study regions.

For this regional sensitivity analysis, FPAR is calculated using fixed EVI2 bounds based on the 2nd and 98th percentiles of all valid EVI2 observations within each region over 2016–2023.

This prevents year-specific EVI2 scaling from introducing artificial interannual variability.

---

### `08_GPP_counterfactual.py`

Performs the counterfactual GPP experiments used to attribute interannual GPP variability.

The four drivers are:

```text
PAR
FPAR
Tmin
VPD
```

For each pixel and day of year, a 2016–2023 climatology is calculated for each driver.

For every target year, GPP is then calculated for all:

```text
2^4 = 16
```

combinations in which each driver is either:

- the target-year value, or
- the 2016–2023 day-of-year climatological value.

The bitmask is:

```text
PAR  = 1
FPAR = 2
Tmin = 4
VPD  = 8
```

Therefore:

```text
mask 0  = all drivers use climatological values
mask 15 = all drivers use target-year values
```

The resulting annual counterfactual GPP values are saved at the pixel level for subsequent Shapley analysis.

---

### `09_Shapley_bootstrap.py`

Calculates the Shapley contribution of PAR, FPAR, Tmin, and VPD to annual GPP variability from the 16 counterfactual GPP experiments.

The Shapley values satisfy:

```text
ΔGPP =
Shapley_PAR +
Shapley_FPAR +
Shapley_Tmin +
Shapley_VPD
```

where:

```text
ΔGPP =
GPP(target-year drivers) -
GPP(all-climatological drivers)
```

Relative driver contributions are calculated from squared Shapley values:

```text
share_i = phi_i^2 / sum(phi^2)
```

Pixel bootstrap sampling is used to estimate uncertainty.

Default settings:

```python
N_BOOTSTRAP = 2000
BOOTSTRAP_SAMPLE_SIZE = 10000
RANDOM_SEED = 0
```

---

### `10_MODIS_site_analysis.py`

Performs the site-level MODIS analysis used for comparison with Himawari.

MODIS Terra and Aqua surface reflectance data were processed in Google Earth Engine for the two representative sites. The exported daily EVI2 time series are used as the input to this script.

Expected CSV columns:

```text
year
doy
evi2
```

The script:

1. reconstructs the daily MODIS EVI2 time series;
2. interpolates missing observations;
3. applies LOWESS smoothing;
4. applies the same crop-phenology detection procedure used for Himawari;
5. outputs smoothed EVI2 and detected phenological dates.

Only site-level MODIS analysis was performed because the MODIS comparison in the manuscript is restricted to the representative sites.

## Input data

The scripts require the datasets described in the Data Availability section of the manuscript.

Major input datasets include:

- Himawari-8/9 AHI surface reflectance products;
- MODIS Terra and Aqua surface reflectance products (MOD09GA and MYD09GA);
- ERA5 meteorological data;
- MSWEP precipitation;
- land-cover data;
- regional masks used for the four study regions.

Large satellite and climate datasets are not included in this repository.

Derived cropping-intensity products generated in the study are archived separately as described in the manuscript Data Availability statement.

The MODIS daily site-level EVI2 time series were exported from Google Earth Engine and can be provided with the code repository because they are small derived input files.

## File paths

The scripts contain placeholder input/output directories such as:

```python
Path("/path/to/...")
```

These paths should be modified for the user's local environment.

All scripts assume that input arrays have already been aligned to the common study grid where applicable.

## Python environment

The main Python dependencies are:

```text
numpy
pandas
scipy
statsmodels
tifffile
joblib
tqdm
```

The exact environment used in the analysis can be documented separately if required.

## Reproducibility notes

The repository contains the custom analytical code associated with the main conclusions of the study.

Routine file-format conversion, data-download operations, and figure-formatting scripts are not included unless they directly affect the reported analytical results.

Google Earth Engine was used to prepare the MODIS site-level time series. No standalone local script was used for the GEE data-export step.

## Suggested execution order

A typical Himawari workflow is:

```text
90percentile_NDVI_clean_v2.py
        ↓
Loess_EVI2_clean.py
        ↓
Regional_Phenology_clean_v2.py
        ↓
ExtractDOY_clean.py
```

The climate and carbon analyses are:

```text
spi3_calculation_clean.py

GPP_calculation_clean.py
        ↓
07_annual_FPAR_climate.py
        ↓
08_GPP_counterfactual.py
        ↓
09_Shapley_bootstrap.py
```

The MODIS site-level comparison is run independently:

```text
10_MODIS_site_analysis.py
```

## Contact

For questions regarding the code or data associated with this study, please contact the corresponding author listed in the manuscript.
