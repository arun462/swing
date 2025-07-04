import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta

st.title("Swing Trade Finder")

symbol = st.text_input("Enter NSE stock symbol (e.g., BANKINDIA.NS):", value="BANKINDIA.NS")

if symbol:
    data = yf.download(symbol, period="6mo", interval="1d")
    data['EMA20'] = ta.ema(data['Close'], length=20)
    data['EMA50'] = ta.ema(data['Close'], length=50)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    buy_signals = (
        (data['EMA20'] > data['EMA50']) &
        (data['EMA20'].shift(1) <= data['EMA50'].shift(1)) &
        (data['RSI'] > 40)
    )
    
    data['BuySignal'] = buy_signals
    st.subheader("Detected Swing Trades:")
    st.dataframe(data[data['BuySignal']])
