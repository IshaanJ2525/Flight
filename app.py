# stock_forecast_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from datetime import date, timedelta

# -------------------------------
# APP TITLE
# -------------------------------
st.set_page_config(page_title="Stock Price Forecaster", layout="wide")
st.title("📈 Stock Price Forecast & Decision Helper")

# -------------------------------
# USER INPUTS
# -------------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()

forecast_period = st.selectbox(
    "Select Forecast Period",
    ["1 Day", "1 Week", "1 Month", "Custom Date Range"]
)

if forecast_period == "Custom Date Range":
    start_date = st.date_input("Start Date", date.today())
    end_date = st.date_input("End Date", date.today() + timedelta(days=30))
else:
    start_date = date.today()
    if forecast_period == "1 Day":
        end_date = date.today() + timedelta(days=1)
    elif forecast_period == "1 Week":
        end_date = date.today() + timedelta(weeks=1)
    elif forecast_period == "1 Month":
        end_date = date.today() + timedelta(days=30)

# -------------------------------
# GET DATA
# -------------------------------
st.subheader("Fetching Historical Data...")
df = yf.download(ticker, period="5y")
df.reset_index(inplace=True)

# Prepare data for Prophet
data = df[["Date", "Close"]]
data.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# -------------------------------
# TRAIN MODEL
# -------------------------------
st.subheader("Training Forecast Model...")
model = Prophet(daily_seasonality=True)
model.fit(data)

# Make future dataframe
future = model.make_future_dataframe(periods=(end_date - date.today()).days)
forecast = model.predict(future)

# -------------------------------
# PLOT FORECAST
# -------------------------------
st.subheader("Forecast Plot")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# -------------------------------
# CALCULATE METRICS
# -------------------------------
# Use last 60 days for evaluation
train = data[:-60]
test = data[-60:]

model_eval = Prophet(daily_seasonality=True)
model_eval.fit(train)
future_eval = model_eval.make_future_dataframe(periods=60)
forecast_eval = model_eval.predict(future_eval)

pred = forecast_eval["yhat"][-60:]
true = test["y"].values

mae = mean_absolute_error(true, pred)
rmse = np.sqrt(mean_squared_error(true, pred))
mape = np.mean(np.abs((true - pred) / true)) * 100
r2 = r2_score(true, pred)

st.subheader("Forecast Accuracy Metrics")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")
st.write(f"**R² Score:** {r2:.2f}")

# -------------------------------
# DECISION MAKER
# -------------------------------
latest_price = df["Close"].iloc[-1]
future_price = forecast[forecast["ds"] == pd.to_datetime(end_date)]["yhat"].values

if len(future_price) > 0:
    change_percent = ((future_price[0] - latest_price) / latest_price) * 100
    st.subheader("Decision Suggestion")
    if change_percent > 5:
        st.success(f"Predicted ↑ {change_percent:.2f}% → Suggestion: **BUY**")
    elif change_percent < -5:
        st.error(f"Predicted ↓ {change_percent:.2f}% → Suggestion: **SELL**")
    else:
        st.info(f"Predicted change {change_percent:.2f}% → Suggestion: **HOLD**")
else:
    st.warning("No forecast available for that date.")

# -------------------------------
# HISTORICAL PRICE PLOT
# -------------------------------
st.subheader("Historical Price Data")
st.line_chart(df.set_index("Date")["Close"])
