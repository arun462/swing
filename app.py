import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    filename="swing_trade_analyzer.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

st.set_page_config(page_title="NSE Swing Trade Analyzer", layout="wide")
st.title("🔍 NSE Swing Trade Analyzer — Live from NSE")

st.sidebar.header("Market Cap Groups")
analyze_large = st.sidebar.checkbox("NIFTY50 (Large Cap)", value=True)
analyze_mid = st.sidebar.checkbox("NIFTY Midcap150", value=False)
analyze_small = st.sidebar.checkbox("NIFTY Smallcap250", value=False)

sample_size = st.sidebar.slider("Number of symbols to analyze per group", 1, 50, 10)

groups = {}

try:
    if analyze_large:
        st.sidebar.write("Fetching NIFTY50 constituents...")
        df_large = pd.read_html("https://www.niftyindices.com/IndexConstituent/ind_nifty50")[0]
        symbols_large = df_large['Symbol'].str.strip().apply(lambda s: f"{s}.NS").tolist()
        groups["NIFTY50 (Large Cap)"] = symbols_large
        st.sidebar.success(f"Fetched {len(symbols_large)} large caps.")

    if analyze_mid:
        st.sidebar.write("Fetching NIFTY Midcap150 constituents...")
        df_mid = pd.read_html("https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150")[0]
        symbols_mid = df_mid['Symbol'].str.strip().apply(lambda s: f"{s}.NS").tolist()
        groups["NIFTY Midcap150"] = symbols_mid
        st.sidebar.success(f"Fetched {len(symbols_mid)} mid caps.")

    if analyze_small:
        st.sidebar.write("Fetching NIFTY Smallcap250 constituents...")
        df_small = pd.read_html("https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250")[0]
        symbols_small = df_small['Symbol'].str.strip().apply(lambda s: f"{s}.NS").tolist()
        groups["NIFTY Smallcap250"] = symbols_small
        st.sidebar.success(f"Fetched {len(symbols_small)} small caps.")

except Exception as e:
    st.error(f"⚠️ Error fetching data from NSE: {e}")
    st.stop()

if not groups:
    st.warning("Please select at least one group in the sidebar.")
    st.stop()

@st.cache_data(show_spinner=False, ttl=86400)  # cache downloads for 1 day
def download_stock_data(symbol):
    return yf.download(symbol, period="3mo", interval="1d", progress=False)

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze_symbol(symbol):
    st.subheader(f"🔹 {symbol}")
    data = download_stock_data(symbol)
    if data.empty:
        st.warning(f"No data for {symbol}. Skipping.")
        return None

    data['EMA20'] = ema(data['Close'], 20)
    data['EMA50'] = ema(data['Close'], 50)
    data['RSI'] = rsi(data['Close'])
    data = data.dropna()

    buy_signals = (
        (data['EMA20'] > data['EMA50']) &
        (data['EMA20'].shift(1) <= data['EMA50'].shift(1)) &
        (data['RSI'] > 40)
    )
    data['BuySignal'] = buy_signals

    swing_trades = []
    for buy_idx in data.index[data['BuySignal']]:
        entry_date = buy_idx
        entry_price = float(data.loc[entry_date, 'Close'])

        future = data.loc[entry_date:].head(20)
        exit_idx, exit_price = None, None

        for tup in future.itertuples():
            if (tup.RSI > 70) or (tup.EMA20 < tup.EMA50):
                exit_idx = tup.Index
                exit_price = float(tup.Close)
                break

        if exit_idx is None:
            exit_idx = future.index[-1]
            exit_price = float(future.iloc[-1]['Close'])

        hold_days = (exit_idx - entry_date).days
        pnl = exit_price - entry_price

        swing_trades.append({
            'Entry Date': entry_date.strftime('%Y-%m-%d'),
            'Entry Price': round(entry_price, 2),
            'Exit Date': exit_idx.strftime('%Y-%m-%d'),
            'Exit Price': round(exit_price, 2),
            'Holding Days': hold_days,
            'PnL': round(pnl, 2)
        })

    if swing_trades:
        trades_df = pd.DataFrame(swing_trades)
        st.dataframe(trades_df)
    else:
        st.info(f"No swing trades detected for {symbol}.")

    return symbol

max_workers = 5  # reduced workers for stability & speed

for group_name, symbols in groups.items():
    st.header(f"📊 Analysis for {group_name}")

    # Randomly sample symbols to analyze
    sample_symbols = random.sample(symbols, min(sample_size, len(symbols)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_symbol, symbol): symbol
            for symbol in sample_symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {e}")
