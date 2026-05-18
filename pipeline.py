"""Restaurant Sales Forecasting Pipeline — run: python pipeline.py"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor; USE_XGB = True
except ImportError:
    USE_XGB = False

# 1. Load
print("[1/5] Loading...")
df = pd.read_csv("restaurant_dataset.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
daily = (df.groupby("date")["quantity_sold"].sum()
           .reindex(pd.date_range(df["date"].min(), df["date"].max(), freq="D"))
           .fillna(0)
           .rename("daily_quantity_sold"))
daily.index.name = "date"

# 2. Features
print("[2/5] Engineering features...")
feat = daily.to_frame()
feat["day_of_week"] = feat.index.dayofweek
feat["month"]       = feat.index.month
feat["is_weekend"]  = (feat.index.dayofweek >= 5).astype(int)
feat["month_sin"]   = np.sin(2 * np.pi * feat.index.month / 12)
feat["month_cos"]   = np.cos(2 * np.pi * feat.index.month / 12)
feat["dow_sin"]     = np.sin(2 * np.pi * feat.index.dayofweek / 7)
feat["dow_cos"]     = np.cos(2 * np.pi * feat.index.dayofweek / 7)
for lag in [1, 7, 14, 30]:
    feat[f"lag_{lag}"] = feat["daily_quantity_sold"].shift(lag)
for w in [7, 14, 30]:
    feat[f"rolling_mean_{w}"] = feat["daily_quantity_sold"].shift(1).rolling(w).mean()
    feat[f"rolling_std_{w}"]  = feat["daily_quantity_sold"].shift(1).rolling(w).std()
feat.dropna(inplace=True)

# 3. Split
print("[3/5] Splitting...")
TARGET = "daily_quantity_sold"
FEATS  = [c for c in feat.columns if c != TARGET]
X, y   = feat[FEATS], feat[TARGET]
s = int(len(feat) * 0.80)
X_train, X_test = X.iloc[:s], X.iloc[s:]
y_train, y_test = y.iloc[:s], y.iloc[s:]

# 4. Train
print("[4/5] Training...")
model = (XGBRegressor(n_estimators=300, learning_rate=0.05,
                      max_depth=6, random_state=42, verbosity=0)
         if USE_XGB else
         RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Evaluate
print("[5/5] Evaluating...")
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test.values - y_pred) / np.clip(y_test.values,1,None))) * 100
print(f"  MAE {mae:.2f} | RMSE {rmse:.2f} | MAPE {mape:.2f}% | R2 {r2:.4f}")

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14,7),
    gridspec_kw={"height_ratios":[3,1]}, sharex=True)
ax1.plot(y_test.index, y_test.values, label="Actual", color="steelblue", lw=1.8)
ax1.plot(y_test.index, y_pred, label="Forecast", color="tomato", lw=1.8, ls="--")
ax1.set_title(f"Daily Sales Forecast | MAE {mae:.1f} | MAPE {mape:.1f}%",
              fontweight="bold")
ax1.set_ylabel("Quantity Sold"); ax1.legend(); ax1.grid(alpha=0.3)
resid = y_test.values - y_pred
ax2.bar(y_test.index, resid,
        color=np.where(resid>=0,"steelblue","tomato"), width=1)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Residual"); ax2.set_xlabel("Date")
plt.tight_layout()
plt.savefig("forecast_chart.png", dpi=150, bbox_inches="tight")
print("Saved: forecast_chart.png")