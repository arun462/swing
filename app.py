import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("📈 Swing Trade Finder (Manual Indicators)")

# User input for stock symbol
symbol = st.text_input("Enter NSE stock symbol (e.g., BANKINDIA.NS):", value="BANKINDIA.NS")

# Manual EMA calculation
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Manual RSI calculation
def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# When symbol is entered, fetch and process data
if symbol:
    data = yf.download(symbol, period="6mo", interval="1d")
    
    if not data.empty:
        data['EMA20'] = ema(data['Close'], 20)
        data['EMA50'] = ema(data['Close'], 50)
        data['RSI'] = rsi(data['Close'])

        # Swing trade signal: EMA20 crosses above EMA50 + RSI > 40
        buy_signals = (
            (data['EMA20'] > data['EMA50']) &
            (data['EMA20'].shift(1) <= data['EMA50'].shift(1)) &
            (data['RSI'] > 40)
        )
        data['BuySignal'] = buy_signals

        st.subheader("Detected Swing Trades:")
        st.dataframe(data[data['BuySignal']])

        # Optional chart
        st.subheader("Closing Price & Signals")
        st.line_chart(data[['Close', 'EMA20', 'EMA50']])
    else:
        st.warning("No data found for the symbol. Please check your input.")
