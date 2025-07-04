import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("📈 Dynamic Swing Trade Analyzer (Manual Indicators)")

symbol = st.text_input("Enter NSE stock symbol (e.g., BANKINDIA.NS):", value="BANKINDIA.NS")

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

if symbol:
    data = yf.download(symbol, period="6mo", interval="1d")
    
    if not data.empty:
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
            entry_price = data.loc[entry_date, 'Close']
            future = data.loc[entry_date:].head(20)  # up to 20 days lookahead
            exit_idx, exit_price = None, None

         for tup in future.itertuples():
    if None in (getattr(tup, "RSI", None), getattr(tup, "EMA20", None), getattr(tup, "EMA50", None)):
        continue  # skip rows missing indicators
    if (tup.RSI > 70) or (tup.EMA20 < tup.EMA50):
        exit_idx = tup.Index
        exit_price = tup.Close
        break

            if exit_idx is None:
                exit_idx = future.index[-1]
                exit_price = future.iloc[-1]['Close']

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
            st.subheader("📋 Detected Swing Trades")
            trades_df = pd.DataFrame(swing_trades)
            st.dataframe(trades_df)

            avg_hold = trades_df['Holding Days'].mean()
            avg_pnl = trades_df['PnL'].mean()
            st.info(f"Average hold period: {avg_hold:.1f} days | Average PnL: ₹{avg_pnl:.2f}")
        else:
            st.warning("No swing trades detected in the selected period.")

        cols_to_plot = [col for col in ['Close', 'EMA20', 'EMA50'] if col in data.columns]
        if cols_to_plot:
            data_clean = data[cols_to_plot].dropna()
            if not data_clean.empty:
                st.subheader("📊 Closing Price & EMAs")
                st.line_chart(data_clean)
            else:
                st.warning("Not enough data to plot after cleaning.")
        else:
            st.warning("No columns available for plotting.")
    else:
        st.warning("No data found for this symbol. Check your input.")
