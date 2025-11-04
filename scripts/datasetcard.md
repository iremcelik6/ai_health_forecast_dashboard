
---

# 📄 **datasetcard.md**
```markdown
# 📊 Dataset Card — AI Health Forecast Dashboard

This document describes all datasets used in the **AI Health Forecast Dashboard** project.  
The datasets were collected from publicly available government sources between **2000 and 2023**.

---

## 🗂️ Dataset Overview

| File | Source | Period | Description |
|------|---------|--------|-------------|
| pollution_2000_2023.csv | [EPA Air Quality System](https://www.epa.gov/aqs) | 2000–2023 | Contains annual averages of CO, NO₂, SO₂, and O₃ pollutants (mean, AQI, and max values). |
| respiratory_disease_mortality_rate_usa.csv | [CDC WONDER](https://wonder.cdc.gov/) | 2000–2023 | Annual respiratory mortality rates per 100k for each U.S. state. |
| cardiovascular_disease_death_rate_usa.csv | [CDC WONDER](https://wonder.cdc.gov/) | 2000–2023 | Annual cardiovascular disease mortality rates per 100k. |
| annual_aqi_by_county_2010.csv | [NOAA](https://www.noaa.gov/) | 2010 | County-level air quality index (AQI) data for calibration. |

---

## 🧹 Data Processing
- Missing values filled using state-level mean imputation  
- Normalized pollutant concentrations using `MinMaxScaler`  
- Aggregated to **state-year** level (e.g., California 2010–2023)  
- Merged into a single dataset: `data/processed/state_timeseries.csv`

---

## 🧩 Data Schema
| Column | Type | Description |
|--------|------|-------------|
| state_fips | string | FIPS state code |
| state_abbr | string | Two-letter state abbreviation |
| state | string | Full state name |
| year | int | Observation year |
| co_mean, no2_mean, so2_mean, o3_mean | float | Average pollutant concentration |
| resp_rate | float | Respiratory mortality rate per 100k |
| cardio_rate | float | Cardiovascular mortality rate per 100k |

---

## ⚠️ Data Exclusion Notice
The original raw data files (each ~100 MB) are **not uploaded to GitHub** due to file size limitations.  
They can be accessed publicly through the original data sources listed above.

---

## 📁 Processed Data (used in training)
- `data/processed/state_timeseries.csv` → Cleaned dataset used in LSTM training  
- `outputs/predictions_state.csv` → Forecasted mortality rates (2024–2028)  
- `outputs/what_if_results.csv` → What-If scenario (+10% pollutant) outcomes  

---

