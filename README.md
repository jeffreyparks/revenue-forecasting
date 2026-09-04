# Revenue Forecasting Tools

Forecasting models and tools for paid media channels

## Overview

Regression-based demand forecasting using media spend and promotional calendars as predictors. Models are trained on historical actuals and used to generate forward-looking predictions based on planned spend scenarios.

## Tech Stack

| Category | Tools |
|----------|-------|
| Package Management | `uv` |
| Modeling | `statsmodels`, `scikit-learn` |
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Apps | `streamlit` |

**Note:** `statsmodels` is preferred for inference (coefficients, diagnostics, p-values). Models are wrapped as sklearn estimators for pipeline compatibility.

## Setup

- clone repo, navigate to project folder
- from command line: `uv venv && uv sync --native-tls`

## Project Structure

```
├── functions/
│   ├── config.py          # Global configuration
│   ├── lr_models.py       # Linear model estimators and pipelines
│   └── transform.py       # Data transformation utilities
├── notebooks/
│   ├── prep_promos.ipynb  # Promotional calendar → dummy variables
│   ├── prep_training.ipynb # Raw data → training datasets
│   ├── model_train_predict.ipynb
│   └── backtest.ipynb     # Rolling backtest validation
├── app/
│   ├── analyze_models.py  # Model diagnostics viewer (Streamlit)
│   └── backtest_app.py    # Backtest results analysis (Streamlit)
├── data/
│   ├── meta/              # Fiscal calendar, promo calendar
│   ├── raw/               # Source actuals (Redshift, Seawalls)
│   ├── train/             # Processed training data
│   ├── predict/           # Spend scenarios for prediction
│   ├── output/            # Model predictions
│   ├── qa/                # QA exports for manual review
│   └── backtest/          # Backtest results
└── sandbox/               # Experiments and drafts
```

## Workflow

1. **Metadata** — Update promo/fiscal calendars, run `prep_promos.ipynb`
2. **Training Data** — Update raw actuals, run `prep_training.ipynb`
3. **Prediction Scenarios** — Update spend plans in `data/predict/`
4. **Model Training** — Run model notebooks, review diagnostics
5. **Prediction** — Generate forecasts, export to `data/output/`
6. **Backtesting** — Run `backtest.ipynb` for rolling validation

## Apps

**Model Diagnostics Viewer**
```bash
uv run streamlit run app/analyze_models.py
```
Interactive inspection of fitted models: coefficients, fit statistics, residual plots, Cook's distance.

**Backtest Analysis**
```bash
uv run streamlit run app/backtest_app.py
```
Review backtest scores and prediction accuracy across rolling time windows.

## Contributing

- Keep `main` branch production-ready
- Create feature branches for development (e.g., `feature/adaptive-k-selection`)
- Submit pull requests for review before merging
- **Data Policy:** Do not commit actual data files to the repository. Only sample datasets (files ending in `_sample.csv` or `*_sample.*`) should be committed.

