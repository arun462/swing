import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.title("📈 Multi-Stock Swing Trade Analyzer (Manual Indicators)")

symbols_input = st.text_area(
    "Enter NSE stock symbols (comma-separated or one per line, e.g., BANKINDIA.NS, RELIANCE.NS):",
    value="BANKINDIA.NS, RELIANCE.NS"
)

symbols = [s.strip().upper() for s in symbols_input.replace("\n", ",").split(",") if s.strip()]

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

for symbol in symbols:
    st.header(f"📊 Analysis for {symbol}")
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
            try:
                entry_price = float(entry_price)
            except (TypeError, ValueError):
                continue  # skip invalid entry

            future = data.loc[entry_date:].head(20)  # up to 20 trading days
            exit_idx, exit_price = None, None

            for tup in future.itertuples():
                if None in (getattr(tup, "RSI", None), getattr(tup, "EMA20", None), getattr(tup, "EMA50", None)):
                    continue  # skip rows missing indicators
                if (tup.RSI > 70) or (tup.EMA20 < tup.EMA50):
                    exit_idx = tup.Index
                    try:
                        exit_price = float(tup.Close)
                    except (TypeError, ValueError):
                        exit_price = None
                    break

            if exit_idx is None:
                exit_idx = future.index[-1]
                try:
                    exit_price = float(future.iloc[-1]['Close'])
                except (TypeError, ValueError):
                    exit_price = None

            if pd.isna(entry_price) or pd.isna(exit_price):
                continue

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
            st.subheader(f"🗂 Detected Swing Trades for {symbol}")
            trades_df = pd.DataFrame(swing_trades)
            st.dataframe(trades_df)

            avg_hold = trades_df['Holding Days'].mean()
            avg_pnl = trades_df['PnL'].mean()
            st.info(f"Average hold period: {avg_hold:.1f} days | Average PnL: ₹{avg_pnl:.2f}")
        else:
            st.warning(f"No swing trades detected for {symbol} in the selected period.")

        # ✅ Current entry signal & potential future setup
        latest = data.iloc[-1]
        if len(data) >= 2:
            prev = data.iloc[-2]
            current_entry = (
                (latest['EMA20'] > latest['EMA50']) and
                (prev['EMA20'] <= prev['EMA50']) and
                (latest['RSI'] > 40)
            )

            if current_entry:
                st.success(f"✅ Current swing entry signal detected on {latest.name.strftime('%Y-%m-%d')} for {symbol}!")
            else:
                st.info(f"❌ No swing entry signal on latest candle ({latest.name.strftime('%Y-%m-%d')}) for {symbol}.")

            ema_gap = latest['EMA20'] - latest['EMA50']
            prev_ema_gap = prev['EMA20'] - prev['EMA50']
            if (ema_gap > prev_ema_gap) and (latest['RSI'] > prev['RSI']) and (latest['RSI'] > 35):
                st.warning(f"⚠️ EMAs converging with rising RSI → possible entry forming soon for {symbol} (gap: {ema_gap:.2f})")
        else:
            st.warning(f"Not enough data for entry signal check for {symbol}.")

        cols_to_plot = [col for col in ['Close', 'EMA20', 'EMA50'] if col in data.columns]
        if cols_to_plot:
            data_clean = data[cols_to_plot].dropna()
            if not data_clean.empty:
                st.subheader(f"📈 Price & EMAs for {symbol}")
                st.line_chart(data_clean)
            else:
                st.warning(f"Not enough data to plot after cleaning for {symbol}.")
        else:
            st.warning(f"No columns available for plotting for {symbol}.")
    else:
        st.error(f"No data found for {symbol}. Check the symbol spelling or if it’s valid on NSE.")
