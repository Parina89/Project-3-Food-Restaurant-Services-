# Restaurant Sales Forecasting Pipeline

End-to-end ML pipeline to predict daily restaurant sales.

## Quickstart

```bash
pip install -r requirements.txt
python pipeline.py
```

## Pipeline Stages

| Week | Stage | Key Outputs |
|------|-------|-------------|
| 1 | EDA & Data Ingestion | Daily time series, decomposition |
| 2 | Feature Engineering | Lag features, rolling stats, encoding |
| 3 | Model Training | Baseline, LR, RF, XGBoost |
| 4 | Evaluation & Reporting | MAE/RMSE/MAPE, feature importance, chart |

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Random Forest | 1591.8 | 2124.2 | 21.5% |
| XGBoost | 1699.8 | 2289.3 | 23.1% |
| Naive Baseline | 2241.3 | 2970.1 | 28.2% |
| Linear Regression | 3599.8 | 12370.2 | 51.9% |

**Best model**: Random Forest (MAPE: 21.5%)

**Top demand driver**: Day-of-Week features (79% of model weight)

**Revenue impact**: ~Rs.397,945/day average forecast error
