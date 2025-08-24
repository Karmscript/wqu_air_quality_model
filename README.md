
# Dar-es-Salaam PM2.5 Time Series Analysis

## Project Overview

This project analyzes **air quality data (PM2.5 levels)** from Dar es Salaam, Tanzania, using **time series analysis** and **AutoRegressive modeling**. The goal is to clean, visualize, and predict PM2.5 levels to provide insights for environmental monitoring and predictive decision-making.

---

## Dataset

* **Source:** MongoDB database (`air-quality`)
* **Collection:** `dar-es-salaam`
* **Contents:** PM2.5 readings from multiple sensor sites with timestamps

> Note: For privacy and size, raw data is not included. Sample data can be generated or queried from the original MongoDB collection.

---

## Tools & Libraries

* **Python:** Pandas, NumPy
* **Visualization:** matplotlib, Plotly
* **Time Series Analysis:** statsmodels (AutoReg)
* **Database:** MongoDB (pymongo)

---

## Project Steps

### 1. Data Extraction

* Connect to MongoDB and access the Dar es Salaam collection.
* Determine all sensor sites and identify the site with the most readings.

### 2. Data Wrangling

* Localize timestamps to Africa/Dar\_es\_Salaam timezone.
* Remove PM2.5 outliers above 100.
* Resample to hourly mean PM2.5 readings.
* Impute missing values using forward-fill.

### 3. Data Visualization

* Time series plot of PM2.5 levels.
* 7-day rolling average plot.
* ACF and PACF plots to understand autocorrelations.

### 4. Model Training

* Split data: 90% training, 10% testing.
* Establish baseline MAE.
* Train AutoRegressive (AR) models for lags 1–30 to determine the best hyperparameter.
* Train `best_model` using the optimal lag.
* Calculate residuals and evaluate performance.

### 5. Walk-Forward Validation

* Perform walk-forward validation on the test set.
* Combine predictions and actual values into a DataFrame.
* Plot interactive comparison using Plotly.

---

## Key Outputs

* Cleaned & wrangled PM2.5 Series (`y`)
* Time series & rolling average plots
* ACF & PACF plots
* Baseline MAE & MAE per lag series
* Trained AutoRegressive model (`best_model`)
* Walk-forward validation predictions (`y_pred_wfv`)
* Interactive Plotly visualization

---

## Insights

* Certain hours/days exhibit higher PM2.5 levels, indicating potential patterns in pollution.
* AR model can reasonably predict short-term PM2.5 trends.
* Pipeline is adaptable for predictive monitoring and alerts.

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/Karmscript/wqu_air_quality_model

2. Install required packages

```bash
pip install -r requirements.txt
```

3. Run scripts 

---

## License

This project is for educational purposes.

