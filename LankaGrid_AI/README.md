# LankaGrid AI — Sri Lankan National Grid Generation Forecasting

**Purpose:** One-day-ahead forecast of total daily electricity generation (GWh) using Sri Lankan observations. Daily generation is a proxy for energy supplied, **not identical to end-use demand**.

## Data
This project uses `data/merged-generation-profile-2017-2026.csv`. Training aggregates the numeric generation-source columns by date and derives total GWh for each day. Rows where every numeric source is unavailable are omitted; they are not imputed or set to zero.

Do not use or add simulated data. The source CSV is aggregated across numeric generation-source columns by date. Missing source values are excluded only where every numeric source is unavailable; duplicate dates, nonpositive totals, and datasets with fewer than 150 valid observations are rejected.

Official references: https://www.ceb.lk/index.php?aam_media=43085 and https://www.cbsl.gov.lk/en/statistics/economic-indicators/weekly-indicators

## Run the Dashboard from GitHub

### 1. Download the project

Open Command Prompt or PowerShell and clone the repository:

```powershell
git clone https://github.com/vinodmadhawa/LankaGrid-AI.git
cd LankaGrid-AI
cd LankaGrid_AI
```

Verify that `app.py`, `train.py`, `requirements.txt`, `data/`, and `outputs/` are in the current folder.

### 2. Create and activate the Python environment

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Command Prompt users can activate the environment with:

```bat
.venv\Scripts\activate.bat
```

### 3. Train the models

Run this once or whenever the source data or training code is updated:

```powershell
python train.py --csv "data/merged-generation-profile-2017-2026.csv" --date-col "Date (GMT+5:30)" --target-col total_gwh --output outputs
```

This creates the saved model and prediction files in `outputs/`.

### 4. Start the dashboard

Use the virtual-environment executable to avoid environment conflicts:

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>.

To stop the dashboard, press `Ctrl+C` in the terminal.

If port `8501` is already in use:

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8502
```

Open <http://127.0.0.1:8502>.

### 5. Run the dashboard without retraining

After the repository has already been cloned and dependencies have been installed, the dashboard can use the saved files in `outputs/`:

```powershell
cd "path\to\LankaGrid-AI\LankaGrid_AI"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

The dashboard loads saved metrics and predictions; it does not retrain models through the browser interface.

## Dashboard

The Streamlit dashboard provides:

- Executive overview with live KPI cards and generation trends
- Date, year and month filtering in the electricity data explorer
- Ensemble method explanations and saved model configurations
- Interactive held-out metrics and ensemble-family comparisons
- Historical backtesting, residual analysis and error patterns
- One-day-ahead predictions from saved model artifacts
- Permutation importance and methodology limitations
- Bagging, boosting, stacking, Extra Trees, AdaBoost and XGBoost comparisons
- XGBoost + Gradient Boosting hybrid forecast
- Dark and light themes
- Responsive layouts for desktop, tablet, and mobile screens

The Next-Day Forecast Studio uses the latest observed date plus one day. It provides no arbitrary-date or multi-day recursive forecast. The dashboard displays no uncertainty interval because the saved artifacts do not contain statistically validated prediction intervals.

## Method
Chronological 80/20 train/test split. Features: 1, 2, 3, 7, 14, 28 day lagged generation, shifted rolling statistics, calendar features. Models: Decision Tree baseline, Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Histogram Gradient Boosting, XGBoost, Stacking, fixed 50/50 Random Forest + Gradient Boosting hybrid, and fixed 50/50 XGBoost + Gradient Boosting hybrid. Metrics: MAE, RMSE, R². No test data is used for fitting the component models. The hybrids are predefined demonstrations rather than test-optimized weights.

## Assignment submission

- Submit the GitHub repository link and hosted dashboard URL if available.
- Submit the Python notebook and/or reproducible Python scripts.
- Include a maximum three-minute screen-recorded explanation covering: the Sri Lankan context, source data and aggregation, feature engineering, model comparison, the best-performing model, and limitations.
- Explain each result in your own words; do not present generated code as your own without disclosure.
- The repository should run with the commands above and include saved artifacts under `outputs/`.

**Research limitation:** scikit-learn's standard StackingRegressor uses ordinary cross-validation internally; this is not fully time-series-aware. For a more rigorous paper, implement expanding-window out-of-fold stacking, a validation set for hyperparameter tuning, and rolling-origin evaluation. The held-out test set is chronological.
