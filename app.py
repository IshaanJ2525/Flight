import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title="Stock Forecast", layout="wide")
st.title("📈 Stock Forecast App (with Indian Stock Support)")

# Example Indian stock tickers for dropdown
indian_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Bajaj Finance": "BAJFINANCE.NS"
}

# Search bar
search_query = st.text_input("Search for a stock (type name or symbol)")

# Filter dropdown results
matching_stocks = {name: symbol for name, symbol in indian_stocks.items()
                   if search_query.lower() in name.lower() or search_query.lower() in symbol.lower()}

if matching_stocks:
    selected_name = st.selectbox("Select Stock", list(matching_stocks.keys()))
    ticker = matching_stocks[selected_name]
else:
    ticker = search_query.strip() if search_query else None

# Forecast period selection
period_map = {"Day": 1, "Week": 7, "Month": 30, "3 Months": 90, "6 Months": 180, "Year": 365}
period_choice = st.selectbox("Forecast Period", list(period_map.keys()))
period_days = period_map[period_choice]

if ticker:
    st.write(f"Fetching data for **{ticker}**...")

    # Download stock data
    df = yf.download(ticker, period="5y")
    if df.empty:
        st.error("No data found for this stock. Try another symbol.")
    else:
        df.reset_index(inplace=True)
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(subset=['y'], inplace=True)

        if df.empty:
            st.error("No valid closing price data found.")
        else:
            # Plot raw data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Stock Price"))
            fig.layout.update(title_text="Historical Stock Prices", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

            # Forecast with Prophet
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=period_days)
            forecast = model.predict(future)

            # Show forecast data
            st.subheader("Forecast Data")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Plot forecast
            st.subheader("Forecast Chart")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1, use_container_width=True)

            # Forecast components
            st.subheader("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
