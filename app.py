
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from forecast_utils import generate_future_features

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ForecastIQ",
    layout="wide"
)

# ---------------------------------------------------
# STYLING
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #F6F2EC;
    }


    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #374151;
    }


    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("ForecastIQ — Restaurant Demand Intelligence")

st.markdown(
    "AI-powered restaurant sales forecasting and inventory optimization"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 14, 30]
)

average_price = st.sidebar.number_input(
    "Average Price Per Item (₹)",
    value=250
)

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------

if uploaded_file:

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # ---------------------------------------------------
    # DATE CONVERSION
    # ---------------------------------------------------

    df['date'] = pd.to_datetime(df['date'])

    # ---------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------

    model = joblib.load("model.pkl")

    # ---------------------------------------------------
    # GENERATE FUTURE FEATURES
    # ---------------------------------------------------

    future_df = generate_future_features(
        df,
        forecast_days
    )

    # ---------------------------------------------------
    # FEATURE COLUMNS
    # ---------------------------------------------------

    feature_columns = [
        'rolling_mean_7',
        'rolling_std_7',
        'rolling_mean_14',
        'rolling_std_14',
        'rolling_mean_30',
        'lag_1',
        'lag_7',
        'lag_14',
        'lag_30',
        'day_of_week',
        'day_of_month',
        'week_of_year',
        'month',
        'quarter',
        'is_weekend',
        'season_encoded',
        'day_Fri',
        'day_Mon',
        'day_Sat',
        'day_Sun',
        'day_Thu',
        'day_Tue',
        'day_Wed',
        'month_sin',
        'month_cos',
        'dow_sin',
        'dow_cos'
    ]

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------

    predictions = model.predict(
        future_df[feature_columns]
    )

    future_df['predicted_sales'] = predictions

    # ---------------------------------------------------
    # REVENUE ESTIMATION
    # ---------------------------------------------------

    future_df['estimated_revenue'] = (
        future_df['predicted_sales'] * average_price
    )

    # ---------------------------------------------------
    # KPI METRICS
    # ---------------------------------------------------

    total_sales = int(
        future_df['predicted_sales'].sum()
    )

    total_revenue = int(
        future_df['estimated_revenue'].sum()
    )

    avg_daily_sales = int(
        future_df['predicted_sales'].mean()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predicted Sales",
        total_sales
    )

    col2.metric(
        "Estimated Revenue",
        f"₹{total_revenue:,}"
    )

    col3.metric(
        "Average Daily Sales",
        avg_daily_sales
    )

    # ---------------------------------------------------
    # DAILY FORECAST TABLE
    # ---------------------------------------------------

    st.subheader(
        f"Daily Predicted Sales (Next {forecast_days} Days)"
    )

    display_df = future_df[[
        'date',
        'predicted_sales',
        'estimated_revenue'
    ]].copy()

    display_df['date'] = pd.to_datetime(
        display_df['date']
    ).dt.strftime('%d-%m-%Y')

    display_df['predicted_sales'] = (
        display_df['predicted_sales']
        .round(0)
        .astype(int)
    )

    display_df['estimated_revenue'] = (
        display_df['estimated_revenue']
        .round(0)
        .astype(int)
    )

    display_df.columns = [
        'Date',
        'Predicted Sales',
        'Estimated Revenue (₹)'
    ]

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # ---------------------------------------------------
    # DAILY SALES CARDS
    # ---------------------------------------------------

    st.subheader("Forecast Summary")

    cols = st.columns(len(display_df))

    for i, row in display_df.iterrows():

        cols[i].metric(
            row['Date'],
            f"{row['Predicted Sales']} orders"
        )

    # ---------------------------------------------------
    # FORECAST CHART
    # ---------------------------------------------------

    st.subheader("Historical vs Forecasted Sales")

    historical_chart = df[[
        'date',
        'quantity_sold'
    ]].copy()

    historical_chart.columns = [
        'date',
        'sales'
    ]

    historical_chart['type'] = 'Historical'

    future_chart = future_df[[
        'date',
        'predicted_sales'
    ]].copy()

    future_chart.columns = [
        'date',
        'sales'
    ]

    future_chart['type'] = 'Forecast'

    combined = pd.concat([
        historical_chart,
        future_chart
    ])

    fig = px.line(
        combined,
        x='date',
        y='sales',
        color='type',
        title='Restaurant Sales Forecast'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ---------------------------------------------------
    # DAILY INVENTORY RECOMMENDATIONS
    # ---------------------------------------------------

    st.subheader(
        f"Daily Inventory Recommendations (Next {forecast_days} Days)"
    )

    inventory_forecast = future_df[[
        'date',
        'predicted_sales'
    ]].copy()

    inventory_forecast['Burger Buns'] = (
        inventory_forecast['predicted_sales']
        .round(0)
        .astype(int)
    )

    inventory_forecast['Chicken (kg)'] = (
        inventory_forecast['predicted_sales'] * 0.12
    ).round(2)

    inventory_forecast['Onion (kg)'] = (
        inventory_forecast['predicted_sales'] * 0.02
    ).round(2)

    inventory_forecast['date'] = pd.to_datetime(
        inventory_forecast['date']
    ).dt.strftime('%d-%m-%Y')

    inventory_forecast.columns = [
        'Date',
        'Predicted Orders',
        'Burger Buns',
        'Chicken (kg)',
        'Onion (kg)'
    ]

    st.dataframe(
        inventory_forecast,
        use_container_width=True
    )


    # ---------------------------------------------------
    # AI BUSINESS INSIGHTS
    # ---------------------------------------------------

    st.subheader("AI Business Insights")

    peak_day = future_df.loc[
        future_df['predicted_sales'].idxmax(),
        'date'
    ]

    peak_sales = int(
        future_df['predicted_sales'].max()
    )

    st.success(
        f"Highest expected demand is on "
        f"{peak_day.date()} "
        f"with approximately "
        f"{peak_sales} orders."
    )

    weekend_sales = future_df[
        future_df['is_weekend'] == 1
    ]['predicted_sales'].mean()

    weekday_sales = future_df[
        future_df['is_weekend'] == 0
    ]['predicted_sales'].mean()

    if weekend_sales > weekday_sales:

        st.info(
            "Weekend demand is expected to be "
            "higher than weekdays. "
            "Consider increasing staff "
            "and inventory preparation."
        )

else:

    st.info(
        "Upload a restaurant sales dataset "
        "to begin forecasting."
    )
