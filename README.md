````md
# ForecastIQ — AI Restaurant Demand Forecasting & Inventory Optimization

ForecastIQ is an AI-powered restaurant demand forecasting system designed to predict future sales and optimize inventory planning using machine learning and time-series forecasting techniques.

The system analyzes historical restaurant sales data, identifies temporal patterns such as trends and seasonality, and generates future demand forecasts to help reduce food waste, improve operational efficiency, and support data-driven decision-making.

---

# Problem Statement

Restaurants often face major operational challenges due to inaccurate demand estimation.

Traditional inventory planning methods rely on:
- manual estimation
- historical averages
- spreadsheets
- intuition-based decision making

This leads to:
- overstocking and food waste
- understocking and lost sales
- inefficient staffing
- poor supply chain planning

ForecastIQ solves this problem by using machine learning to predict future restaurant demand and generate inventory recommendations automatically.

---

# Features

## AI Demand Forecasting
- Predicts restaurant sales for the next:
  - 7 days
  - 14 days
  - 30 days

## Time-Series Feature Engineering
Implements advanced forecasting features such as:
- lag features
- rolling means
- rolling standard deviation
- cyclical time encoding
- weekend detection
- season encoding

## Inventory Optimization
Automatically converts predicted sales into ingredient requirements.

Example:
- burger buns
- chicken quantity
- onion quantity

## Revenue Forecasting
Estimates future revenue based on predicted demand.

## Interactive Dashboard
Built using Streamlit with:
- KPI cards
- demand forecast tables
- interactive charts
- business insights

## AI Business Insights
Provides operational suggestions such as:
- peak demand periods
- weekend demand spikes
- inventory planning recommendations

---

# Tech Stack

| Component | Technology |
|---|---|
| Frontend Dashboard | Streamlit |
| Machine Learning | XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Serialization | Joblib |
| Language | Python |

---

# Machine Learning Approach

The project uses:
# XGBoost Regressor

for time-series forecasting.

The model is trained using engineered temporal features including:
- lag_1
- lag_7
- lag_14
- rolling_mean_7
- rolling_std_7
- cyclical month/day encoding
- seasonal indicators

---

# Time-Series Components Used

The forecasting pipeline captures:
- Trend
- Seasonality
- Cyclical Patterns
- Irregular Variations

---

# Evaluation Metrics

The model performance is evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)

---

# Project Workflow

```text
Historical Sales Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
XGBoost Forecasting Model
        ↓
Future Sales Prediction
        ↓
Inventory Optimization
        ↓
Business Insights Dashboard
````

---

# Folder Structure

```text
ForecastIQ/
│
├── app.py
├── forecast_utils.py
├── train_model.py
├── model.pkl
├── restaurant_dataset.csv
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd ForecastIQ
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Model Training

```bash
python train_model.py
```

This generates:

```text
model.pkl
```

---

# Run Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

# Dataset Format

The dataset should contain:

| date       | quantity_sold |
| ---------- | ------------- |
| 2025-01-01 | 120           |
| 2025-01-02 | 135           |

---

# Outputs Generated

The system provides:

* Future daily sales prediction
* Revenue forecasting
* Inventory recommendations
* Forecast visualization
* AI operational insights

---

# Example Output

| Date       | Predicted Sales | Revenue    |
| ---------- | --------------- | ---------- |
| 02-01-2025 | 6449            | ₹16,12,164 |
| 03-01-2025 | 6342            | ₹15,85,456 |

---

# Inventory Forecast Example

| Ingredient  | Required Quantity |
| ----------- | ----------------- |
| Burger Buns | 6449              |
| Chicken     | 773 kg            |
| Onion       | 128 kg            |

---

# Why XGBoost?

XGBoost was selected because:

* high forecasting accuracy
* handles nonlinear relationships
* strong performance on structured data
* robust against overfitting
* efficient training and inference

---

# Future Enhancements

Possible improvements:

* Weather API integration
* Holiday/event forecasting
* Multi-item demand forecasting
* Real-time inventory tracking
* LSTM/Deep Learning forecasting
* Multi-restaurant support
* Cloud deployment

---

# Business Impact

ForecastIQ helps restaurants:

* reduce food waste
* optimize inventory planning
* improve revenue forecasting
* reduce operational costs
* improve staffing decisions

---


# License

This project is developed for educational and research purposes.

```
```
