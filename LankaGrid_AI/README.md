# LankaGrid AI — Sri Lankan National Grid Generation Forecasting

**Purpose:** One-day-ahead forecast of total daily electricity generation (GWh) using Sri Lankan observations. Daily generation is a proxy for energy supplied, **not identical to end-use demand**.

## Data
This project uses the aggregated daily dataset `data/electricity.csv`, derived from `data/merged-generation-profile-2017-2026.csv`. It contains one row per consecutive date and two columns:

- `date`: the observation date
- `total_gwh`: sum of the available numeric generation-source columns in GWh
Rows where every numeric generation-source column is unavailable are omitted; they are not imputed or set to zero.

Do not use or add simulated data. If a source CSV contains individual generating plants or energy in MWh, aggregate plants by date and convert MWh to GWh by dividing by 1,000 before training. The training script rejects duplicate dates, missing calendar dates, nonpositive values, and datasets with fewer than 150 observations.

Official references: https://www.ceb.lk/index.php?aam_media=43085 and https://www.cbsl.gov.lk/en/statistics/economic-indicators/weekly-indicators

## Run on Windows

From Command Prompt or PowerShell, open the project folder:

```bat
cd "C:\Users\Vinod Madhawa\Desktop\ML Assignment 1\LankaGrid_AI"
```

### First-time setup

Create and activate the virtual environment, install dependencies, and generate the model outputs:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --csv data/electricity.csv --date-col date --target-col total_gwh --output outputs
```

PowerShell users can activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, enable it only for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Start the dashboard

After setup and training, run:

```bat
.venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>.

To stop the dashboard, press `Ctrl+C` in the terminal. The dashboard loads saved metrics and predictions from `outputs/`; it does not retrain models through the UI.

If port `8501` is already in use, start the dashboard on another port and open that address:

```bat
.venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8502
```

Open <http://127.0.0.1:8502>.

### Run again without retraining

If `outputs/` already contains the generated artifacts, activate the existing environment and start the dashboard directly:

```bat
cd "C:\Users\Vinod Madhawa\Desktop\ML Assignment 1\LankaGrid_AI"
.venv\Scripts\activate.bat
.venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

## Dashboard

The Streamlit dashboard provides:

- Executive overview with live KPI cards and generation trends
- Date, year and month filtering in the electricity data explorer
- Ensemble method explanations and saved model configurations
- Interactive held-out metrics and ensemble-family comparisons
- Historical backtesting, residual analysis and error patterns
- One-day-ahead predictions from saved model artifacts
- Permutation importance and methodology limitations
- AdaBoost and XGBoost boosting comparisons
- XGBoost + Gradient Boosting hybrid forecast
- Dark and light themes
- Responsive layouts for desktop, tablet, and mobile screens

The Next-Day Forecast Studio uses the latest observed date plus one day. It provides no arbitrary-date or multi-day recursive forecast. The dashboard displays no uncertainty interval because the saved artifacts do not contain statistically validated prediction intervals.

## Method
Chronological 80/20 train/test split. Features: 1, 2, 3, 7, 14, 28 day lagged generation, shifted rolling statistics, calendar features. Models: Decision Tree baseline, Bagging, Random Forest, AdaBoost, Gradient Boosting, Histogram Gradient Boosting, XGBoost, Stacking, fixed 50/50 Random Forest + Gradient Boosting hybrid, and fixed 50/50 XGBoost + Gradient Boosting hybrid. Metrics: MAE, RMSE, R². No test data is used for fitting the component models. The hybrids are predefined demonstrations rather than test-optimized weights.

**Research limitation:** scikit-learn's standard StackingRegressor uses ordinary cross-validation internally; this is not fully time-series-aware. For a more rigorous paper, implement expanding-window out-of-fold stacking, a validation set for hyperparameter tuning, and rolling-origin evaluation. The held-out test set is chronological.
