# USD/VND Exchange Rate Forecasting

A time-series forecasting project that predicts the USD/VND exchange rate using a hybrid deep learning and statistical approach — combining **LSTM** (short-term evaluation) with **ARIMA-GARCH** (long-horizon forecasting).

---

## Overview

Exchange rate forecasting is a challenging task due to the non-linear, non-stationary nature of financial time series. This project addresses that challenge by combining the pattern-recognition strengths of LSTM neural networks with the statistically grounded long-horizon capabilities of ARIMA-GARCH, avoiding the flat-line convergence issue that plagues pure recursive deep learning forecasts.

**Key results:**
- LSTM test set RMSE: ~114 VND | MAPE: ~0.41%
- 199-day forecast (16 June – 31 December 2026) with realistic day-over-day fluctuations
- Output CSV formatted identically to the source data for seamless downstream use

---

## Methodology

### Why a hybrid approach?

Pure LSTM recursive forecasting (feeding each predicted value back as input) suffers from **error accumulation** over long horizons: after ~50–100 steps, the model's input window becomes nearly constant, causing predictions to converge to a fixed value. This project solves that by using two complementary models:

| Component | Role | Horizon |
|-----------|------|---------|
| LSTM | One-step-ahead evaluation on held-out test set | Short-term (days) |
| ARIMA(5,1,0) | Trend / drift estimation | Long-term (months) |
| GARCH(1,1) | Conditional volatility modeling → realistic noise | Long-term (months) |

### Forecasting pipeline

```
Historical prices (643 days)
        │
        ├──► LSTM (window=30) ──► Test set evaluation (Biểu đồ 2)
        │
        ├──► ARIMA(5,1,0) ──────► Trend forecast (199 days)
        │
        └──► GARCH(1,1) ─────────► Conditional volatility
                                          │
                              ARIMA trend + damped GARCH noise
                                          │
                                  Final forecast (Biểu đồ 3 + CSV)
```

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Investing.com (USD/VND daily) |
| Period | 01 January 2024 – 15 June 2026 |
| Observations | 643 trading days |
| Price range | 24,260 – 26,432 VND |
| Features | Open, High, Low, Close, Volume, % Change |

---

## Project Structure

```
.
├── USD_VND_LSTM_ARIMA_GARCH_Forecast.ipynb   # Main notebook (Google Colab)
├── Dữ_liệu_Lịch_sử_USD_VND.csv              # Input: historical USD/VND data
├── forecast_USD_VND_16_06_2026_to_31_12_2026.csv  # Output: forecast results
├── chart1_lich_su_ty_gia.png                 # Chart 1: Historical price trend
├── chart2_test_vs_predict.png                # Chart 2: LSTM test vs actual
├── chart3_du_bao_2026.png                    # Chart 3: Future forecast
└── README.md
```

---

## Results

### Chart 1 — Historical USD/VND Rate (Jan 2024 – Jun 2026)
Visualizes the full historical price series before model training. Notable events visible in the data include the sharp depreciation of VND in mid-2024 and the subsequent partial recovery.

### Chart 2 — LSTM: Test Set vs. Actual (the key evaluation chart)
One-step-ahead predictions on the held-out 15% test set.

| Metric | Value |
|--------|-------|
| RMSE | ~114 VND |
| MAE | ~90 VND |
| MAPE | ~0.41% |

### Chart 3 — Forecast: 16 June – 31 December 2026
Long-horizon forecast combining ARIMA trend with GARCH-derived stochastic noise. Unlike a pure LSTM recursive forecast, this produces realistic daily fluctuations throughout the entire 199-day horizon.

---

## How to Run

### Prerequisites

```bash
pip install tensorflow scikit-learn statsmodels arch pandas matplotlib
```

### Google Colab (recommended)

1. Open `USD_VND_LSTM_ARIMA_GARCH_Forecast.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run the first cell and upload `Dữ_liệu_Lịch_sử_USD_VND.csv` when prompted
3. Run all cells top to bottom
4. Download the forecast CSV from the last cell

### Local

```bash
jupyter notebook USD_VND_LSTM_ARIMA_GARCH_Forecast.ipynb
```

---

## Output Format

The forecast CSV is structured identically to the input file for seamless integration:

```
"Ngày","Lần cuối","Mở","Cao","Thấp","KL","% Thay đổi"
"16/06/2026","26,292.6","26,290.0","26,305.7","26,289.0","","0.01%"
"17/06/2026","26,291.3","26,292.6","26,308.3","26,251.3","","-0.00%"
...
```

- Encoding: UTF-8 with BOM
- All fields quoted
- Numbers use comma thousand separator (`26,292.6`)
- Percentage change without leading `+` sign (matching source format)

---

## Model Details

### LSTM Architecture

```
Input (30, 1)
    → LSTM(64, return_sequences=True) + Dropout(0.2)
    → LSTM(32) + Dropout(0.2)
    → Dense(16, relu)
    → Dense(1)
```

- Optimizer: Adam | Loss: MSE
- Early stopping: patience=10, restore best weights
- Train/Test split: 85% / 15% (chronological, no shuffling)

### ARIMA

Order `(5, 1, 0)` selected based on the integrated nature of the price series (I(1)) and ACF/PACF analysis of the differenced series. The differencing order d=1 ensures stationarity.

### GARCH

`GARCH(1,1)` fitted on log-returns (×100) with normal innovations. Forecast conditional variance is used to parameterize day-specific noise, producing **heteroskedastic** (time-varying volatility) innovations — a well-established property of exchange rate series.

---

## Limitations

- The model does not incorporate macroeconomic covariates (Fed interest rates, SBV interventions, foreign reserves, trade balance), which are significant drivers of USD/VND.
- Forecast reliability degrades beyond ~30 days; the 199-day horizon should be interpreted as a **trend scenario**, not a point prediction.
- GARCH noise is sampled from a fixed random seed for reproducibility; re-running with a different seed produces a different (equally valid) trajectory.
- The dataset covers only 2024–2026 — a period of elevated global FX volatility — which may not generalize to calmer regimes.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![statsmodels](https://img.shields.io/badge/statsmodels-ARIMA-green)
![arch](https://img.shields.io/badge/arch-GARCH-red)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-yellow)

- **Deep Learning:** TensorFlow / Keras (LSTM)
- **Statistical Models:** statsmodels (ARIMA), arch (GARCH)
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib
- **Environment:** Google Colab / Jupyter Notebook
