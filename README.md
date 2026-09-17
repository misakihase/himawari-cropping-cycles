# Code for Hase et al.

This repository contains the main custom code used in the study by Hase et al., submitted to *Science Advances*.

The scripts cover the principal analyses used to generate the main results of the study: daily Himawari EVI2 compositing, temporal smoothing, crop phenology and cropping-intensity detection, drought analysis, GPP estimation, counterfactual GPP experiments and Shapley attribution, and the MODIS comparisons used to assess the effect of polar-orbiting sampling.

Figure-formatting and plotting scripts are not included because they do not affect the analytical results.

## Overview of the workflow

The Himawari analysis chain is:

1. Himawari sub-daily observations → daily EVI2 composite (`01`)
2. Daily EVI2 → LOWESS-smoothed EVI2 time series (`02`)
3. Smoothed EVI2 → heading, planting, and harvesting dates (`03`)
4. Phenological dates → annual crop cycles and cropping intensity (`04`)
5. MSWEP precipitation → regional SPI-3 (`05`)
6. EVI2 → FPAR → GPP (`06`)
7. Counterfactual GPP experiments (`08`) → Shapley attribution of GPP variability (`09`), with annual regional driver means from (`07`)

Two MODIS comparisons are run independently of that chain and of each other:

8. MODIS site-level EVI2 → phenology comparison with Himawari (`10`)
9. MCD15A2H 8-day FPAR export (`00`) → reprojection and smoothing (`11`) → region-wide MODIS GPP (`12`)

## Scripts

### `00_MCD15_GEE.js`

Google Earth Engine script that exports MCD15A2H 8-day FPAR over the study domain.

One GeoTIFF is exported per 8-day compositing period, named `MCD15_FPAR_<YYYYMMDD>.tif`, where `<YYYYMMDD>` is the start date of the period. This is the naming expected by `11_MCD15_FPAR_preprocess.py`.

Quality screening retains only retrievals from the main radiative-transfer algorithm and discards contaminated pixels:

- `FparLai_QC` bits 5–7 (SCF_QC) must be 0 or 1;
- `FparExtra_QC` flags for snow/ice, cirrus, internal cloud, and cloud shadow must all be clear.

Main settings:

```javascript
var YEAR = 2016;                   // edited once per year
var PRODUCT = 'MCD15A2H';          // 'MCD15A2H' (Terra+Aqua) or 'MOD15A2H' (Terra)
var REGION = ee.Geometry.Rectangle([92.0, -9.0, 127.0, 23.0]);
var SCALE = 500;                   // native FPAR resolution, m
var FPAR_SCALE_FACTOR = 0.01;      // stored integer -> fraction
```

The script is run once per year by editing `YEAR` and starting the tasks from the Earth Engine Tasks tab.

---

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

On a 365-day series this corresponds to a smoothing window of approximately 35 days. Missing values are linearly interpolated before smoothing.

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
|---|---:|---:|
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

- Irrawaddy
- Chao Phraya
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
TMIN_MIN = -8.0         # deg C; f(Tmin) = 0 at or below
TMIN_MAX = 12.02        # deg C; f(Tmin) = 1 at or above
VPD_MIN = 650.0         # Pa;    f(VPD)  = 1 at or below
VPD_MAX = 4300.0        # Pa;    f(VPD)  = 0 at or above
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

For this regional sensitivity analysis only, FPAR is calculated using fixed EVI2 bounds based on the 2nd and 98th percentiles of all valid EVI2 observations within each region over 2016–2023. This prevents year-specific EVI2 scaling from introducing artificial interannual variability. The main GPP estimate in `06_GPP_calculation.py` uses pixel-wise minimum–maximum scaling instead.

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

The Shapley values are exactly additive:

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

Because the Shapley operator is linear in the 16 counterfactual means, the Shapley value of a pixel mean equals the pixel mean of the Shapley values, and the calculation is carried out as a single matrix–vector product.

The reported measure of each driver's contribution to interannual variability is the standard deviation of its Shapley value across 2016–2023. A squared-contribution share is also written to the output for reference:

```text
share_i = phi_i^2 / sum(phi^2)
```

but the standard deviation is preferred because it retains physical units and is stable across years, whereas the share can change sharply between years without any corresponding change in the underlying contribution.

Uncertainty is estimated by a paired pixel bootstrap: one pixel resample per replicate is applied to all years and all 16 counterfactuals simultaneously, so the covariance among them is preserved.

Usage and defaults:

```text
python 09_Shapley_bootstrap.py --name ChaoPhraya

--n_boot    2000   bootstrap replicates
--target_n  0      pixels drawn per replicate; 0 means the full sample size
--seed      0
--ci_low    2.5
--ci_high   97.5
```

Outputs:

```text
<name>_shapley_by_year.csv       phi and share per year, with intervals
<name>_shapley_interannual.csv   SD of phi across years, with intervals
```

---

### `10_MODIS_site_analysis.py`

Performs the site-level MODIS analysis used for the phenology comparison with Himawari (Fig. 2a,b).

Daily MODIS Terra and Aqua surface reflectances were processed in Google Earth Engine for the two representative sites, and the exported daily EVI2 time series are used as the input to this script.

Expected CSV columns:

```text
year
doy
evi2
```

The script:

1. reconstructs the daily MODIS EVI2 time series for 2016–2023;
2. interpolates missing observations;
3. applies LOWESS smoothing to two-year windows;
4. applies the same crop-phenology detection procedure used for Himawari;
5. outputs smoothed EVI2 and detected phenological dates.

This script covers the site-level phenology comparison only. The region-wide MODIS comparison is handled by `11` and `12`.

---

### `11_MCD15_FPAR_preprocess.py`

Reprojects the MCD15A2H 8-day FPAR exported by `00_MCD15_GEE.js` onto the AHI analysis grid and applies LOESS smoothing.

Reprojection uses nearest-neighbour resampling, so that the 500 m product is mapped onto the 0.01° analysis grid without introducing values that were not retrieved.

Main settings:

```python
NY, NX = 3200, 3500      # analysis grid
RES = 0.01               # degrees
RESAMPLING = Resampling.nearest
FILL_SINGLE_MISSING_SLOT = True   # interpolate a slot whose neighbours both exist
MIN_FINITE_SLOTS = 3              # pixels with fewer valid slots are left unsmoothed
```

Usage:

```text
python 11_MCD15_FPAR_preprocess.py 2018
python 11_MCD15_FPAR_preprocess.py 2018 --frac_days 35 --n_jobs 16
```

The default LOESS span of 35 days matches the effective span used for the AHI series in `02_Loess_EVI2.py`, so that the two records are smoothed comparably.

Outputs include the reprojected FPAR, the gap-filling flags, the smoothed 8-day FPAR, and a daily FPAR series. The daily series is produced only so that the light-use-efficiency model can be run at a daily time step; within each 8-day period the FPAR value is constant, matching the temporal resolution of the source product.

---

### `12_MCD15_GPP_calculation.py`

Calculates daily and annual cropland GPP from the MODIS 8-day FPAR produced by `11_MCD15_FPAR_preprocess.py`.

This is the MODIS counterpart of `06_GPP_calculation.py`. The light-use-efficiency model, its parameters, and the daily meteorological forcing are identical; only the FPAR source differs.

```python
MIN_VALID_DAYS = 330     # below this, the annual total is set to NaN
```

Pixels whose valid-day count falls below this threshold are set to NaN in the annual output, so that partially sampled years are not reported as annual totals.

Usage:

```text
python 12_MCD15_GPP_calculation.py 2018
```

## Input data

The scripts require the datasets described in the Data Availability section of the manuscript.

Major input datasets include:

- Himawari-8/9 AHI surface reflectance products;
- MODIS Terra and Aqua surface reflectance products (MOD09GA and MYD09GA);
- MODIS combined Terra+Aqua FPAR (MCD15A2H, Collection 6.1);
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

These paths should be modified for the user's local environment. In `11_MCD15_FPAR_preprocess.py` and `12_MCD15_GPP_calculation.py` they can also be set through environment variables (`LANDCOVER_PATH`, `OUTDIR`, `SRAD_DIR`, `TMIN_DIR`, `VPD_DIR`).

All scripts assume that input arrays have already been aligned to the common study grid where applicable.

## Python environment

The main Python dependencies are:

```text
numpy
pandas
scipy
statsmodels
rasterio
tifffile
joblib
tqdm
```

These are listed in `requirements.txt`. `00_MCD15_GEE.js` runs in the Google Earth Engine code editor and requires no local Python environment.

## Reproducibility notes

The repository contains the custom analytical code associated with the main conclusions of the study.

Routine file-format conversion, data-download operations, and figure-formatting scripts are not included unless they directly affect the reported analytical results.

Google Earth Engine was used for all MODIS data preparation. The MCD15A2H export is scripted here as `00_MCD15_GEE.js`; the site-level MOD09GA/MYD09GA EVI2 export was performed interactively in the Earth Engine code editor and its output CSV files are provided as input to `10_MODIS_site_analysis.py`.

## Execution order

Scripts are numbered in the order in which they are run. Within the Himawari chain, `01` through `04` are sequential, and `06` through `09` follow, with `05` and `07` independent of one another:

```text
01_EVI2_composite.py
        ↓
02_Loess_EVI2.py
        ↓
03_Extract_cropping_cycle.py
        ↓
04_Extract_DOY.py
```

```text
05_spi3_calculation.py     (independent)

06_GPP_calculation.py
        ↓
07_annual_FPAR_climate.py
        ↓
08_GPP_counterfactual.py
        ↓
09_Shapley_bootstrap.py
```

The two MODIS comparisons run independently of the Himawari chain:

```text
10_MODIS_site_analysis.py   (independent)

00_MCD15_GEE.js
        ↓
11_MCD15_FPAR_preprocess.py
        ↓
12_MCD15_GPP_calculation.py
```

## Contact

For questions regarding the code or data associated with this study, please contact the corresponding author listed in the manuscript.
