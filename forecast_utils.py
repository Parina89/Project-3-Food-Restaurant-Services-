import pandas as pd
import numpy as np


def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3


def generate_future_features(df, days=7):

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    history = list(df['quantity_sold'])

    future_rows = []

    last_date = df['date'].max()

    for i in range(days):

        future_date = last_date + pd.Timedelta(days=i + 1)

        day_of_week = future_date.dayofweek
        day_of_month = future_date.day
        week_of_year = future_date.isocalendar().week
        month = future_date.month
        quarter = future_date.quarter

        is_weekend = 1 if day_of_week >= 5 else 0

        season_encoded = get_season(month)

        lag_1 = history[-1]
        lag_7 = history[-7]
        lag_14 = history[-14]
        lag_30 = history[-30]

        rolling_mean_7 = np.mean(history[-7:])
        rolling_std_7 = np.std(history[-7:])

        rolling_mean_14 = np.mean(history[-14:])
        rolling_std_14 = np.std(history[-14:])

        rolling_mean_30 = np.mean(history[-30:])

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        row = {
            'rolling_mean_7': rolling_mean_7,
            'rolling_std_7': rolling_std_7,
            'rolling_mean_14': rolling_mean_14,
            'rolling_std_14': rolling_std_14,
            'rolling_mean_30': rolling_mean_30,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'lag_14': lag_14,
            'lag_30': lag_30,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'week_of_year': week_of_year,
            'month': month,
            'date': future_date,
            'quarter': quarter,
            'is_weekend': is_weekend,
            'season_encoded': season_encoded,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
        }

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for d in days:
            row[f'day_{d}'] = 0

        row[f'day_{days[day_of_week]}'] = 1

        future_rows.append(row)

        history.append(rolling_mean_7)

    return pd.DataFrame(future_rows)
