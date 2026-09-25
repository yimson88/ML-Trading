
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

import ml_engine

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    from plyer import notification
except Exception:
    notification = None

st.set_page_config(
    page_title="SMC + ML Market Structure Trading System",
    page_icon="📊",
    layout="wide"
)

PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "pip_size": 0.0001, "note": "Liquid major pair; good for structure and liquidity analysis."},
    "GBPUSD": {"ticker": "GBPUSD=X", "pip_size": 0.0001, "note": "Strong movement major pair; good for momentum and liquidity sweeps."},
    "USDJPY": {"ticker": "JPY=X", "pip_size": 0.01, "note": "Liquid major pair; often respects higher timeframe structure."},
    "AUDUSD": {"ticker": "AUDUSD=X", "pip_size": 0.0001, "note": "Useful for risk sentiment and trend phases."},
    "NZDUSD": {"ticker": "NZDUSD=X", "pip_size": 0.0001, "note": "Good movement during Asia and USD sessions."},
    "USDCAD": {"ticker": "CAD=X", "pip_size": 0.0001, "note": "Useful for USD and oil-related market themes."},
    "USDCHF": {"ticker": "CHF=X", "pip_size": 0.0001, "note": "Safe-haven behavior; useful in risk-off conditions."},
    "EURJPY": {"ticker": "EURJPY=X", "pip_size": 0.01, "note": "Volatile cross pair with good trend potential."},
    "GBPJPY": {"ticker": "GBPJPY=X", "pip_size": 0.01, "note": "Very volatile; reduce risk size."},
    "EURAUD": {"ticker": "EURAUD=X", "pip_size": 0.0001, "note": "Can produce long directional moves."},
    "XAUUSD": {"ticker": "GC=F", "pip_size": 0.01, "note": "Gold futures proxy. Highly volatile; use smaller risk."},
    "BTCUSD": {"ticker": "BTC-USD", "pip_size": 1.0, "note": "Bitcoin trades 24/7. Highly volatile; use smaller risk."},
}

# -----------------------------
# DATA
# -----------------------------

def remove_abnormal_prices(df, max_deviation=0.35):
    if df is None or df.empty:
        return df
    df = df.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]
    df = df[(df["High"] >= df["Low"]) & (df["High"] >= df["Open"]) & (df["High"] >= df["Close"]) &
            (df["Low"] <= df["Open"]) & (df["Low"] <= df["Close"])]
    if len(df) < 30:
        return df.reset_index(drop=True)
    typical = df[price_cols].median(axis=1)
    rolling_med = typical.rolling(window=96, min_periods=20).median()
    ratio = typical / rolling_med
    keep = rolling_med.isna() | ratio.between(1 - max_deviation, 1 + max_deviation)
    return df.loc[keep].reset_index(drop=True)

def clean_yfinance_data(data):
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.reset_index()
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
    return remove_abnormal_prices(df, max_deviation=0.35)

def _load_csv_override(pair_name, timeframe):
    """
    Optional: if the user drops a broker/MT5 export CSV at
    data/history/{PAIR}_{TIMEFRAME}.csv (columns Date,Open,High,Low,Close[,Volume]),
    it is used INSTEAD of yfinance for that timeframe. This lets the ML layer
    train on far more history than yfinance's intraday limits allow
    (yfinance caps 15m data at ~60 days and 1h data at ~730 days).
    """
    path = os.path.join("data", "history", f"{pair_name}_{timeframe}.csv")
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_csv(path)
        raw.columns = [c.strip().capitalize() for c in raw.columns]
        if "Date" not in raw.columns:
            raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)
        if "Volume" not in raw.columns:
            raw["Volume"] = 0
        raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
        return remove_abnormal_prices(raw, max_deviation=0.35)
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_data(ticker, daily_start, pair_name=None):
    daily_override = _load_csv_override(pair_name, "D1") if pair_name else None
    h1_override = _load_csv_override(pair_name, "H1") if pair_name else None
    m15_override = _load_csv_override(pair_name, "M15") if pair_name else None

    if daily_override is not None and not daily_override.empty:
        daily = daily_override
    else:
        daily_raw = yf.download(ticker, start=daily_start, interval="1d", auto_adjust=False, progress=False)
        daily = clean_yfinance_data(daily_raw)

    if h1_override is not None and not h1_override.empty:
        h1 = h1_override
    else:
        h1_raw = yf.download(ticker, period="730d", interval="1h", auto_adjust=False, progress=False)
        h1 = clean_yfinance_data(h1_raw)

    if m15_override is not None and not m15_override.empty:
        m15 = m15_override
    else:
        m15_raw = yf.download(ticker, period="60d", interval="15m", auto_adjust=False, progress=False)
        m15 = clean_yfinance_data(m15_raw)

    return daily, h1, m15


def add_cameroon_trading_window(df, start_hour=6, end_hour=22, enforce_window=True):
    """
    Adds Cameroon local time and trading-window filtering.
    Market data timestamps from yfinance are treated as UTC, then converted to Africa/Douala.
    Active window default: 06:00 to 22:00 Cameroon time.
    """
    df = df.copy()

    local_time = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert("Africa/Douala")

    df["Cameroon_Time"] = local_time.dt.strftime("%Y-%m-%d %H:%M")
    hour_value = local_time.dt.hour + (local_time.dt.minute / 60)

    if start_hour < end_hour:
        df["Trading_Window"] = (hour_value >= start_hour) & (hour_value < end_hour)
    else:
        # Supports overnight windows if ever needed.
        df["Trading_Window"] = (hour_value >= start_hour) | (hour_value < end_hour)

    if enforce_window:
        outside_window = ~df["Trading_Window"]

        if "Entry_Signal" in df.columns:
            active_signal = df["Entry_Signal"].isin(["BUY SMC", "SELL SMC"])
            clear_rows = outside_window & active_signal

            df.loc[clear_rows, "SMC_Reason"] = "Signal ignored because it occurred outside the Cameroon trading window."
            df.loc[clear_rows, "Setup_Quality"] = "Outside trading window"
            df.loc[clear_rows, "Entry_Signal"] = "Wait"
            df.loc[clear_rows, ["Entry", "Suggested_SL", "Suggested_TP"]] = np.nan

    return df

# -----------------------------
# INDICATORS + SMC TOOLS
# -----------------------------

def add_atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(period).mean()
    return df

def add_ema(df):
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df

def add_swings(df, swing_len=3):
    df = df.copy()
    df["Swing_High"] = np.nan
    df["Swing_Low"] = np.nan
    if len(df) < swing_len * 2 + 1:
        return df
    for i in range(swing_len, len(df) - swing_len):
        if df["High"].iloc[i] == df["High"].iloc[i-swing_len:i+swing_len+1].max():
            df.loc[df.index[i], "Swing_High"] = df["High"].iloc[i]
        if df["Low"].iloc[i] == df["Low"].iloc[i-swing_len:i+swing_len+1].min():
            df.loc[df.index[i], "Swing_Low"] = df["Low"].iloc[i]
    df["Last_Swing_High"] = df["Swing_High"].ffill()
    df["Last_Swing_Low"] = df["Swing_Low"].ffill()
    return df

def add_structure(df):
    df = df.copy()
    prev_high = df["Last_Swing_High"].shift(1)
    prev_low = df["Last_Swing_Low"].shift(1)
    df["BOS_Bullish"] = (df["Close"] > prev_high) & prev_high.notna()
    df["BOS_Bearish"] = (df["Close"] < prev_low) & prev_low.notna()
    df["Structure"] = "Neutral"
    state = "Neutral"
    for i in range(len(df)):
        if bool(df["BOS_Bullish"].iloc[i]):
            state = "Bullish"
        elif bool(df["BOS_Bearish"].iloc[i]):
            state = "Bearish"
        df.loc[df.index[i], "Structure"] = state
    df["CHOCH_Bullish"] = (df["Structure"].shift(1) == "Bearish") & (df["Structure"] == "Bullish")
    df["CHOCH_Bearish"] = (df["Structure"].shift(1) == "Bullish") & (df["Structure"] == "Bearish")
    return df

def add_liquidity(df):
    df = df.copy()
    prev_high = df["Last_Swing_High"].shift(1)
    prev_low = df["Last_Swing_Low"].shift(1)
    df["Buy_Side_Liquidity_Sweep"] = (df["High"] > prev_high) & (df["Close"] < prev_high) & prev_high.notna()
    df["Sell_Side_Liquidity_Sweep"] = (df["Low"] < prev_low) & (df["Close"] > prev_low) & prev_low.notna()

    # Equal highs/lows as liquidity pools
    tolerance = df["ATR_14"] * 0.15
    df["Equal_Highs_Liquidity"] = (df["High"] - prev_high).abs() <= tolerance
    df["Equal_Lows_Liquidity"] = (df["Low"] - prev_low).abs() <= tolerance
    return df

def add_fvg(df):
    df = df.copy()
    df["Bullish_FVG"] = df["Low"] > df["High"].shift(2)
    df["Bearish_FVG"] = df["High"] < df["Low"].shift(2)
    df["FVG_Low"] = np.nan
    df["FVG_High"] = np.nan
    df.loc[df["Bullish_FVG"], "FVG_Low"] = df["High"].shift(2)
    df.loc[df["Bullish_FVG"], "FVG_High"] = df["Low"]
    df.loc[df["Bearish_FVG"], "FVG_Low"] = df["High"]
    df.loc[df["Bearish_FVG"], "FVG_High"] = df["Low"].shift(2)
    df["Last_Bullish_FVG_Low"] = df["FVG_Low"].where(df["Bullish_FVG"]).ffill()
    df["Last_Bullish_FVG_High"] = df["FVG_High"].where(df["Bullish_FVG"]).ffill()
    df["Last_Bearish_FVG_Low"] = df["FVG_Low"].where(df["Bearish_FVG"]).ffill()
    df["Last_Bearish_FVG_High"] = df["FVG_High"].where(df["Bearish_FVG"]).ffill()
    return df

def add_supply_demand(df, buffer_mult=0.5):
    df = df.copy()
    df["Demand_Zone_Low"] = np.nan
    df["Demand_Zone_High"] = np.nan
    df["Supply_Zone_Low"] = np.nan
    df["Supply_Zone_High"] = np.nan
    for i in range(5, len(df)):
        atr = df["ATR_14"].iloc[i]
        if pd.isna(atr):
            continue
        if bool(df["BOS_Bullish"].iloc[i]) or bool(df["CHOCH_Bullish"].iloc[i]):
            candles = df.iloc[max(0, i-10):i]
            bearish = candles[candles["Close"] < candles["Open"]]
            if not bearish.empty:
                c = bearish.iloc[-1]
                low = min(c["Low"], c["Open"], c["Close"])
                high = max(c["Open"], c["Close"]) + atr * buffer_mult
                df.loc[df.index[i], "Demand_Zone_Low"] = low
                df.loc[df.index[i], "Demand_Zone_High"] = high
        if bool(df["BOS_Bearish"].iloc[i]) or bool(df["CHOCH_Bearish"].iloc[i]):
            candles = df.iloc[max(0, i-10):i]
            bullish = candles[candles["Close"] > candles["Open"]]
            if not bullish.empty:
                c = bullish.iloc[-1]
                high = max(c["High"], c["Open"], c["Close"])
                low = min(c["Open"], c["Close"]) - atr * buffer_mult
                df.loc[df.index[i], "Supply_Zone_Low"] = low
                df.loc[df.index[i], "Supply_Zone_High"] = high
    df["Last_Demand_Low"] = df["Demand_Zone_Low"].ffill()
    df["Last_Demand_High"] = df["Demand_Zone_High"].ffill()
    df["Last_Supply_Low"] = df["Supply_Zone_Low"].ffill()
    df["Last_Supply_High"] = df["Supply_Zone_High"].ffill()
    return df

def add_premium_discount(df):
    df = df.copy()
    df["Equilibrium"] = (df["Last_Swing_High"] + df["Last_Swing_Low"]) / 2
    df["Premium_Discount"] = "Neutral"
    df.loc[df["Close"] < df["Equilibrium"], "Premium_Discount"] = "Discount"
    df.loc[df["Close"] > df["Equilibrium"], "Premium_Discount"] = "Premium"
    return df

def add_smc(df, swing_len):
    df = add_atr(df.copy(), 14)
    df = add_ema(df)
    df = add_swings(df, swing_len)
    df = add_structure(df)
    df = add_liquidity(df)
    df = add_fvg(df)
    df = add_supply_demand(df)
    df = add_premium_discount(df)
    return df

def build_smc_system(daily, h1, m15, risk_reward, atr_mult, swing_len, strict_mode, session_start, session_end, enforce_session):
    daily = add_smc(daily, swing_len)
    h1 = add_smc(h1, swing_len)
    m15 = add_smc(m15, swing_len)

    daily["Daily_Bias"] = daily["Structure"].shift(1).fillna("Neutral")
    h1["H1_Structure"] = h1["Structure"].shift(1).fillna("Neutral")

    daily_state = daily[["Date", "Daily_Bias"]].dropna().sort_values("Date")
    h1_state = h1[["Date", "H1_Structure"]].dropna().sort_values("Date")
    m15 = m15.sort_values("Date").reset_index(drop=True)

    m15 = pd.merge_asof(m15, daily_state, on="Date", direction="backward") if not daily_state.empty else m15.assign(Daily_Bias="Neutral")
    m15 = pd.merge_asof(m15, h1_state, on="Date", direction="backward") if not h1_state.empty else m15.assign(H1_Structure="Neutral")
    m15["Daily_Bias"] = m15["Daily_Bias"].fillna("Neutral")
    m15["H1_Structure"] = m15["H1_Structure"].fillna("Neutral")

    m15["In_Demand_Zone"] = m15["Last_Demand_Low"].notna() & (m15["Low"] <= m15["Last_Demand_High"]) & (m15["Close"] >= m15["Last_Demand_Low"])
    m15["In_Supply_Zone"] = m15["Last_Supply_High"].notna() & (m15["High"] >= m15["Last_Supply_Low"]) & (m15["Close"] <= m15["Last_Supply_High"])
    m15["In_Bullish_FVG"] = m15["Last_Bullish_FVG_Low"].notna() & (m15["Low"] <= m15["Last_Bullish_FVG_High"]) & (m15["Close"] >= m15["Last_Bullish_FVG_Low"])
    m15["In_Bearish_FVG"] = m15["Last_Bearish_FVG_High"].notna() & (m15["High"] >= m15["Last_Bearish_FVG_Low"]) & (m15["Close"] <= m15["Last_Bearish_FVG_High"])

    m15["Entry_Signal"] = "Wait"
    m15["Setup_Quality"] = "No setup"
    m15["Entry"] = np.nan
    m15["Suggested_SL"] = np.nan
    m15["Suggested_TP"] = np.nan
    m15["SMC_Reason"] = "No valid SMC confluence"

    bullish_zone = m15["In_Demand_Zone"] | m15["In_Bullish_FVG"] | (m15["Premium_Discount"] == "Discount")
    bearish_zone = m15["In_Supply_Zone"] | m15["In_Bearish_FVG"] | (m15["Premium_Discount"] == "Premium")
    bullish_trigger = m15["Sell_Side_Liquidity_Sweep"] | m15["CHOCH_Bullish"] | m15["BOS_Bullish"]
    bearish_trigger = m15["Buy_Side_Liquidity_Sweep"] | m15["CHOCH_Bearish"] | m15["BOS_Bearish"]

    if strict_mode:
        bullish = (m15["Daily_Bias"] == "Bullish") & (m15["H1_Structure"] == "Bullish") & bullish_trigger & bullish_zone & (m15["Close"] > m15["Open"])
        bearish = (m15["Daily_Bias"] == "Bearish") & (m15["H1_Structure"] == "Bearish") & bearish_trigger & bearish_zone & (m15["Close"] < m15["Open"])
    else:
        bullish = (m15["H1_Structure"] == "Bullish") & bullish_trigger & bullish_zone & (m15["Close"] > m15["Open"])
        bearish = (m15["H1_Structure"] == "Bearish") & bearish_trigger & bearish_zone & (m15["Close"] < m15["Open"])

    m15.loc[bullish, "Entry_Signal"] = "BUY SMC"
    m15.loc[bearish, "Entry_Signal"] = "SELL SMC"
    m15.loc[bullish | bearish, "Entry"] = m15["Close"]

    buy_zone_sl = m15[["Last_Demand_Low", "Last_Bullish_FVG_Low", "Last_Swing_Low"]].min(axis=1)
    sell_zone_sl = m15[["Last_Supply_High", "Last_Bearish_FVG_High", "Last_Swing_High"]].max(axis=1)
    buy_atr_sl = m15["Close"] - (atr_mult * m15["ATR_14"])
    sell_atr_sl = m15["Close"] + (atr_mult * m15["ATR_14"])

    m15.loc[bullish, "Suggested_SL"] = np.minimum(buy_zone_sl, buy_atr_sl)
    m15.loc[bearish, "Suggested_SL"] = np.maximum(sell_zone_sl, sell_atr_sl)

    buy_risk = m15["Close"] - m15["Suggested_SL"]
    sell_risk = m15["Suggested_SL"] - m15["Close"]
    m15.loc[bullish, "Suggested_TP"] = m15["Close"] + buy_risk * risk_reward
    m15.loc[bearish, "Suggested_TP"] = m15["Close"] - sell_risk * risk_reward

    m15.loc[bullish, "Setup_Quality"] = "A+ Buy SMC"
    m15.loc[bearish, "Setup_Quality"] = "A+ Sell SMC"
    m15.loc[bullish, "SMC_Reason"] = "Bullish HTF structure + 15m SMC trigger near demand/FVG/discount."
    m15.loc[bearish, "SMC_Reason"] = "Bearish HTF structure + 15m SMC trigger near supply/FVG/premium."

    invalid_buy = (m15["Entry_Signal"] == "BUY SMC") & ((m15["Suggested_SL"] >= m15["Entry"]) | (m15["Suggested_TP"] <= m15["Entry"]))
    invalid_sell = (m15["Entry_Signal"] == "SELL SMC") & ((m15["Suggested_SL"] <= m15["Entry"]) | (m15["Suggested_TP"] >= m15["Entry"]))
    invalid = invalid_buy | invalid_sell
    m15.loc[invalid, ["Entry_Signal", "Setup_Quality", "SMC_Reason"]] = ["Wait", "Invalid risk", "SL/TP rejected."]
    m15.loc[invalid, ["Entry", "Suggested_SL", "Suggested_TP"]] = np.nan

    # Apply Cameroon trading window after signals are created.
    m15 = add_cameroon_trading_window(
        m15,
        start_hour=session_start,
        end_hour=session_end,
        enforce_window=enforce_session
    )

    return daily, h1, m15

# -----------------------------
# BACKTEST
# -----------------------------

def calculate_max_drawdown(equity_values):
    if not equity_values:
        return 0.0
    equity = pd.Series(equity_values)
    dd = (equity - equity.cummax()) / equity.cummax()
    return float(dd.min() * 100)

def backtest_smc(df, pip_size, initial_balance, risk_percent, max_hold_bars):
    trades = []
    balance = float(initial_balance)
    equity_values = [balance]
    d = df.reset_index(drop=True).copy()
    i = 0
    while i < len(d) - 2:
        row = d.iloc[i]

        # HARD SESSION FILTER:
        # The backtest must not count any setup unless the SIGNAL candle is inside
        # the selected Cameroon trading window.
        if "Trading_Window" in d.columns and not bool(row.get("Trading_Window", False)):
            i += 1
            continue

        signal = row.get("Entry_Signal", "Wait")
        if signal not in ["BUY SMC", "SELL SMC"]:
            i += 1
            continue

        entry_idx = i + 1
        entry_price = float(d.iloc[entry_idx]["Open"])
        sl = row.get("Suggested_SL", np.nan)
        tp = row.get("Suggested_TP", np.nan)
        if pd.isna(sl) or pd.isna(tp):
            i += 1
            continue

        direction = "BUY" if signal == "BUY SMC" else "SELL"
        if direction == "BUY" and not (sl < entry_price < tp):
            i += 1
            continue
        if direction == "SELL" and not (tp < entry_price < sl):
            i += 1
            continue

        exit_idx = min(entry_idx + max_hold_bars, len(d) - 1)
        exit_price = None
        exit_reason = "TIME EXIT"

        for j in range(entry_idx, min(entry_idx + max_hold_bars + 1, len(d))):
            high = float(d.iloc[j]["High"])
            low = float(d.iloc[j]["Low"])
            if direction == "BUY":
                hit_sl = low <= sl
                hit_tp = high >= tp
                if hit_sl:
                    exit_price, exit_reason, exit_idx = sl, "SL", j
                    break
                if hit_tp:
                    exit_price, exit_reason, exit_idx = tp, "TP", j
                    break
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp
                if hit_sl:
                    exit_price, exit_reason, exit_idx = sl, "SL", j
                    break
                if hit_tp:
                    exit_price, exit_reason, exit_idx = tp, "TP", j
                    break

        if exit_price is None:
            exit_price = float(d.iloc[exit_idx]["Close"])

        if direction == "BUY":
            pips = (exit_price - entry_price) / pip_size
            risk_pips = (entry_price - sl) / pip_size
        else:
            pips = (entry_price - exit_price) / pip_size
            risk_pips = (sl - entry_price) / pip_size
        if risk_pips <= 0:
            i += 1
            continue
        r_mult = pips / risk_pips
        pnl = balance * (risk_percent / 100) * r_mult
        balance += pnl
        equity_values.append(balance)
        trades.append({
            "Signal Date": row["Date"],
            "Signal Cameroon Time": row.get("Cameroon_Time", "N/A"),
            "Entry Date": d.iloc[entry_idx]["Date"],
            "Entry Cameroon Time": d.iloc[entry_idx].get("Cameroon_Time", "N/A"),
            "Exit Date": d.iloc[exit_idx]["Date"],
            "Exit Cameroon Time": d.iloc[exit_idx].get("Cameroon_Time", "N/A"),
            "Direction": direction,
            "Entry": round(entry_price, 5),
            "SL": round(sl, 5),
            "TP": round(tp, 5),
            "Exit": round(exit_price, 5),
            "Exit Reason": exit_reason,
            "Pips/Points": round(pips, 1),
            "R Multiple": round(r_mult, 2),
            "Balance": round(balance, 2),
            "Reason": row.get("SMC_Reason", "")
        })
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"Total Trades": 0, "Win Rate": 0, "Net R": 0, "Average R": 0, "Profit Factor": 0, "Max Drawdown %": 0, "Final Balance": initial_balance}, pd.DataFrame({"Trade": [0], "Balance": [initial_balance]})

    wins = trades_df[trades_df["R Multiple"] > 0]
    losses = trades_df[trades_df["R Multiple"] < 0]
    gross_profit = wins["R Multiple"].sum()
    gross_loss = abs(losses["R Multiple"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    metrics = {
        "Total Trades": int(len(trades_df)),
        "Win Rate": round(len(wins) / len(trades_df) * 100, 2),
        "Net R": round(float(trades_df["R Multiple"].sum()), 2),
        "Average R": round(float(trades_df["R Multiple"].mean()), 2),
        "Profit Factor": round(float(pf), 2) if np.isfinite(pf) else "∞",
        "Max Drawdown %": round(calculate_max_drawdown(equity_values), 2),
        "Final Balance": round(float(trades_df.iloc[-1]["Balance"]), 2),
    }
    equity_df = pd.DataFrame({"Trade": range(len(equity_values)), "Balance": equity_values})
    return trades_df, metrics, equity_df

# -----------------------------
# CHARTS & ALERTS
# -----------------------------

def smc_chart(df, title, symbol, overlays=None, show_signals=False, show_zones=True):
    overlays = overlays or []
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candles"))
    for col in overlays:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))
    if "Swing_High" in df.columns:
        s = df[df["Swing_High"].notna()]
        if not s.empty:
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Swing_High"], mode="markers", marker=dict(size=7, symbol="triangle-down"), name="Swing High"))
    if "Swing_Low" in df.columns:
        s = df[df["Swing_Low"].notna()]
        if not s.empty:
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Swing_Low"], mode="markers", marker=dict(size=7, symbol="triangle-up"), name="Swing Low"))
    if show_zones and len(df) > 0:
        last = df.iloc[-1]
        x0 = df["Date"].iloc[max(0, len(df) - 120)]
        x1 = df["Date"].iloc[-1]
        if pd.notna(last.get("Last_Demand_Low", np.nan)) and pd.notna(last.get("Last_Demand_High", np.nan)):
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=last["Last_Demand_Low"], y1=last["Last_Demand_High"], fillcolor="rgba(0,180,120,0.18)", line=dict(width=1), layer="below")
            fig.add_annotation(x=x1, y=last["Last_Demand_High"], text="Demand", showarrow=False)
        if pd.notna(last.get("Last_Supply_Low", np.nan)) and pd.notna(last.get("Last_Supply_High", np.nan)):
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=last["Last_Supply_Low"], y1=last["Last_Supply_High"], fillcolor="rgba(220,60,60,0.16)", line=dict(width=1), layer="below")
            fig.add_annotation(x=x1, y=last["Last_Supply_Low"], text="Supply", showarrow=False)
    if show_signals and "Entry_Signal" in df.columns:
        buys = df[df["Entry_Signal"] == "BUY SMC"]
        sells = df[df["Entry_Signal"] == "SELL SMC"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Close"], mode="markers", marker=dict(size=12, symbol="triangle-up"), name="BUY SMC"))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Close"], mode="markers", marker=dict(size=12, symbol="triangle-down"), name="SELL SMC"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=symbol, height=620, xaxis_rangeslider_visible=False)
    return fig

def equity_chart(equity_df, title="SMC Backtest Equity Curve"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df["Trade"], y=equity_df["Balance"], mode="lines+markers", name="Equity Curve"))
    fig.update_layout(title=title, xaxis_title="Trade Number", yaxis_title="Balance", height=420)
    return fig

def browser_beep():
    components.html("""<script>
    try { const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = ctx.createOscillator(); const gain = ctx.createGain();
    oscillator.connect(gain); gain.connect(ctx.destination);
    oscillator.type = 'sine'; oscillator.frequency.value = 880; gain.gain.value = 0.08;
    oscillator.start(); setTimeout(() => oscillator.stop(), 450); } catch(e) {}
    </script>""", height=0)

def send_alert(symbol, row, desktop_alert, sound_alert):
    signal = row.get("Entry_Signal", "Wait")
    if signal not in ["BUY SMC", "SELL SMC"]:
        return
    key = f"{symbol}-{row.get('Date')}-{signal}"
    if "last_alert_key" not in st.session_state:
        st.session_state["last_alert_key"] = None
    if st.session_state["last_alert_key"] == key:
        return
    st.session_state["last_alert_key"] = key
    msg = f"{symbol} {signal} | Entry: {float(row['Entry']):.5f} | SL: {float(row['Suggested_SL']):.5f} | TP: {float(row['Suggested_TP']):.5f}"
    st.toast(msg, icon="🚨")
    st.success(f"🚨 VALID SMC SETUP ALERT: {msg}")
    if sound_alert:
        browser_beep()
    if desktop_alert and notification is not None:
        try:
            notification.notify(title=f"{symbol} SMC Setup Alert", message=msg, timeout=10)
        except Exception:
            pass

def format_zone(row, low_col, high_col):
    if pd.isna(row.get(low_col, np.nan)) or pd.isna(row.get(high_col, np.nan)):
        return "N/A"
    return f"{float(row[low_col]):.5f} - {float(row[high_col]):.5f}"

# -----------------------------
# APP
# -----------------------------

st.title("📊 Smart Money Concepts + ML Trading System")
st.caption("Daily = bias | 1H = market structure | 15m = SMC entry trigger | XGBoost scores every setup's win-probability | Educational strategy testing only")

with st.sidebar:
    st.header("System Controls")
    selected = st.selectbox("Choose market", list(PAIRS.keys()), index=0)
    ticker = PAIRS[selected]["ticker"]
    pip_size = PAIRS[selected]["pip_size"]
    st.info(PAIRS[selected]["note"])

    daily_start = st.date_input("Daily data start date", value=pd.to_datetime("2020-01-01"))

    st.subheader("SMC Settings")
    swing_len = st.selectbox("Swing sensitivity", [2, 3, 4, 5], index=1)
    risk_reward = st.selectbox("Risk-to-reward target", [1.5, 2.0, 2.5, 3.0, 4.0], index=2)
    atr_mult = st.selectbox("ATR safety buffer", [0.5, 1.0, 1.5, 2.0], index=1)
    strict_mode = st.checkbox("Strict mode: require Daily + 1H alignment", value=True)

    st.divider()
    st.subheader("Cameroon Trading Window")
    enforce_session = st.checkbox("Only allow signals during my watch time", value=True)
    session_start = st.selectbox("Start watching from", list(range(0, 24)), index=6, format_func=lambda x: f"{x:02d}:00 Cameroon time")
    session_end = st.selectbox("Stop watching at", list(range(1, 25)), index=21, format_func=lambda x: f"{x if x < 24 else 0:02d}:00 Cameroon time")
    st.caption("Default plan: watch for signals from 06:00 to 22:00 Cameroon time.")

    st.divider()
    st.subheader("Alerts")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_minutes = st.selectbox("Refresh every", [1, 3, 5, 10, 15], index=2)
    desktop_alert = st.checkbox("Desktop notification", value=True)
    sound_alert = st.checkbox("Sound alert", value=True)

    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=refresh_minutes * 60 * 1000, key="smc_auto_refresh")
        else:
            st.warning("Install streamlit-autorefresh to use auto-refresh.")

    refresh = st.button("Refresh Analysis")

    st.divider()
    st.subheader("Backtest Settings")
    run_backtest = st.checkbox("Run backtest", value=True)
    initial_balance = st.number_input("Initial balance", min_value=100.0, value=10000.0, step=100.0)
    risk_percent = st.selectbox("Risk per trade (%)", [0.25, 0.5, 1.0, 2.0], index=1)
    max_hold = st.selectbox("Max hold time on 15m candles", [8, 16, 32, 48, 96], index=2)

    st.divider()
    st.subheader("🤖 ML Confidence Layer (XGBoost)")
    st.caption(
        "Scores every rule-based SMC setup with a win-probability learned from this "
        "market's own signal history. Filters use ADX (trend strength) and Bollinger "
        "Band Width (volatility/squeeze avoidance) on top of the ML score."
    )
    ml_conf_threshold = st.slider("Minimum ML win-probability to confirm a setup", 0.50, 0.95, 0.58, 0.01)
    ml_min_adx = st.slider("ADX filter: minimum trend strength", 0, 40, 18, 1)
    ml_min_bbw_pct = st.slider("BB Width filter: minimum volatility percentile", 0.0, 0.90, 0.15, 0.05)
    ml_horizon = st.selectbox("Label horizon for training (15m candles)", [16, 24, 32, 48, 64], index=2)
    ml_n_splits = st.selectbox("Walk-forward folds", [3, 4, 5, 6, 8], index=2)
    train_now = st.button("🔁 Train / Update ML Model for this market")

if refresh:
    st.cache_data.clear()

try:
    daily, h1, m15 = load_data(ticker, str(daily_start), pair_name=selected)
    if daily.empty or h1.empty or m15.empty:
        st.error("Not enough data was returned. Try another market or refresh.")
        st.stop()

    daily, h1, m15 = build_smc_system(daily, h1, m15, risk_reward, atr_mult, swing_len, strict_mode, session_start, session_end, enforce_session)

    valid_daily = daily.dropna(subset=["Close", "ATR_14", "Structure"])
    valid_h1 = h1.dropna(subset=["Close", "ATR_14", "Structure"])
    valid_m15_check = m15.dropna(subset=["Close", "ATR_14", "Entry_Signal"])
    if valid_daily.empty or valid_h1.empty or valid_m15_check.empty:
        st.error("Indicators are not ready yet. Dataset is too small.")
        st.stop()

    # ---------------------------------------------------------------
    # ML CONFIDENCE LAYER
    # Build multi-timeframe features (Daily -> H1 -> M15, ADX, Bollinger
    # Band Width, RSI, MACD, ATR, EMA slopes, SMC structure/liquidity/zones),
    # label every historical SMC signal by replaying its own ATR/zone-based
    # SL and RR-based TP, load (or train) an XGBoost win-probability model,
    # and score every row in the dataset.
    # ---------------------------------------------------------------
    m15_features, m15_labeled, signal_rows = ml_engine.build_ml_dataset(daily, h1, m15, max_hold_bars=ml_horizon)
    bundle = ml_engine.load_model_bundle(selected)

    if train_now:
        with st.spinner(f"Training XGBoost model for {selected} on {len(signal_rows)} historical SMC signals (walk-forward validation)..."):
            if len(signal_rows) < 20 or signal_rows["Label_Win"].nunique() < 2:
                st.sidebar.error(
                    f"Not enough resolved historical signals yet to train a reliable model "
                    f"({len(signal_rows)} found). Widen the daily start date, lower strict "
                    f"mode, or let more signals accumulate, then try again."
                )
            else:
                oos_df, fold_metrics, wf_summary = ml_engine.run_walk_forward(
                    signal_rows, n_splits=ml_n_splits
                )
                final_model, medians = ml_engine.train_final_model(signal_rows)
                if final_model is None:
                    st.sidebar.error("Training failed: historical signals only contain one outcome class (all wins or all losses).")
                else:
                    fi = ml_engine.get_feature_importance(final_model)
                    meta = {
                        "n_signals": int(len(signal_rows)),
                        "training_win_rate": round(float(signal_rows["Label_Win"].mean()), 4),
                        "walk_forward_summary": wf_summary,
                        "fold_metrics": fold_metrics,
                        "label_horizon_bars": int(ml_horizon),
                        "risk_reward": float(risk_reward),
                        "atr_mult": float(atr_mult),
                        "strict_mode": bool(strict_mode),
                    }
                    ml_engine.save_model_bundle(selected, final_model, ml_engine.FEATURE_COLUMNS, medians, meta)
                    os.makedirs(ml_engine.REPORTS_DIR, exist_ok=True)
                    fi.to_csv(os.path.join(ml_engine.REPORTS_DIR, f"{selected}_feature_importance.csv"), index=False)
                    if not oos_df.empty:
                        oos_df.to_csv(os.path.join(ml_engine.REPORTS_DIR, f"{selected}_walk_forward_oos.csv"), index=False)
                    st.sidebar.success(f"Model trained on {len(signal_rows)} historical signals (training win rate {meta['training_win_rate']:.1%}).")
                    bundle = ml_engine.load_model_bundle(selected)

    m15 = ml_engine.score_signals_with_model(
        m15_features, bundle, ml_conf_threshold,
        min_adx=ml_min_adx, min_bbw_percentile=ml_min_bbw_pct
    )

    valid_m15 = m15.dropna(subset=["Close", "ATR_14", "Entry_Signal"])
    if valid_m15.empty:
        st.error("Indicators are not ready yet. Dataset is too small.")
        st.stop()

    latest_m15 = valid_m15.iloc[-1].to_dict()
    send_alert(selected, latest_m15, desktop_alert, sound_alert)

    if latest_m15.get("Entry_Signal") in ["BUY SMC", "SELL SMC"]:
        if bundle is None:
            st.info("A rule-based SMC setup is present, but no ML model has been trained yet for this market. Use the sidebar button to train one.")
        elif latest_m15.get("ML_Taken"):
            st.success(f"🤖 ML-CONFIRMED SETUP — win-probability {latest_m15.get('ML_Proba', 0):.1%} ≥ threshold.")
        else:
            st.warning(f"Rule-based SMC setup present but the ML layer filtered it out: {latest_m15.get('ML_Filter_Reason', 'n/a')} (probability {latest_m15.get('ML_Proba', 0):.1%}).")

    st.subheader(f"{selected} SMC + ML Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Daily Bias", latest_m15.get("Daily_Bias", "Neutral"))
    c2.metric("1H Structure", latest_m15.get("H1_Structure", "Neutral"))
    c3.metric("15m Structure", latest_m15.get("Structure", "Neutral"))
    c4.metric("15m Signal", latest_m15.get("Entry_Signal", "Wait"))
    c5.metric("Cameroon Time", latest_m15.get("Cameroon_Time", "N/A"))
    c6.metric("R:R", f"{risk_reward}:1")

    c7, c8, c9, c10 = st.columns(4)
    ml_proba_val = latest_m15.get("ML_Proba", np.nan)
    c7.metric("ML Win Probability", "N/A" if pd.isna(ml_proba_val) else f"{ml_proba_val:.1%}")
    c8.metric("ML Verdict", "Confirmed" if latest_m15.get("ML_Taken") else ("No model" if bundle is None else "Filtered"))
    c9.metric("ADX (15m)", "N/A" if pd.isna(latest_m15.get("ADX_14", np.nan)) else f"{latest_m15['ADX_14']:.1f}")
    c10.metric("Historical Signals Trained On", bundle["meta"].get("n_signals", "N/A") if bundle else "No model")

    if latest_m15["Entry_Signal"] in ["BUY SMC", "SELL SMC"]:
        st.success("Valid SMC setup currently present.")
    else:
        st.info("No valid SMC setup now. Waiting for liquidity + structure + zone confluence.")

    st.subheader("Current SMC Trade Plan")
    plan = pd.DataFrame({
        "Item": [
            "Market", "Daily Bias", "1H Structure", "15m Structure", "Signal",
            "Cameroon Time", "Inside Trading Window",
            "Entry", "Stop-Loss", "Take-Profit", "Premium/Discount",
            "Demand Zone", "Supply Zone", "Bullish FVG", "Bearish FVG",
            "Sell-side Sweep", "Buy-side Sweep", "BOS Bullish", "BOS Bearish",
            "CHOCH Bullish", "CHOCH Bearish", "Reason",
            "ML Win Probability", "ML Verdict", "ADX (15m)", "BB Width Percentile (15m)"
        ],
        "Value": [
            selected,
            latest_m15.get("Daily_Bias", "N/A"),
            latest_m15.get("H1_Structure", "N/A"),
            latest_m15.get("Structure", "N/A"),
            latest_m15.get("Entry_Signal", "Wait"),
            latest_m15.get("Cameroon_Time", "N/A"),
            bool(latest_m15.get("Trading_Window", False)),
            "N/A" if pd.isna(latest_m15.get("Entry", np.nan)) else round(float(latest_m15["Entry"]), 5),
            "N/A" if pd.isna(latest_m15.get("Suggested_SL", np.nan)) else round(float(latest_m15["Suggested_SL"]), 5),
            "N/A" if pd.isna(latest_m15.get("Suggested_TP", np.nan)) else round(float(latest_m15["Suggested_TP"]), 5),
            latest_m15.get("Premium_Discount", "N/A"),
            format_zone(latest_m15, "Last_Demand_Low", "Last_Demand_High"),
            format_zone(latest_m15, "Last_Supply_Low", "Last_Supply_High"),
            format_zone(latest_m15, "Last_Bullish_FVG_Low", "Last_Bullish_FVG_High"),
            format_zone(latest_m15, "Last_Bearish_FVG_Low", "Last_Bearish_FVG_High"),
            bool(latest_m15.get("Sell_Side_Liquidity_Sweep", False)),
            bool(latest_m15.get("Buy_Side_Liquidity_Sweep", False)),
            bool(latest_m15.get("BOS_Bullish", False)),
            bool(latest_m15.get("BOS_Bearish", False)),
            bool(latest_m15.get("CHOCH_Bullish", False)),
            bool(latest_m15.get("CHOCH_Bearish", False)),
            latest_m15.get("SMC_Reason", "N/A"),
            "N/A" if pd.isna(latest_m15.get("ML_Proba", np.nan)) else f"{latest_m15['ML_Proba']:.1%}",
            "Confirmed" if latest_m15.get("ML_Taken") else latest_m15.get("ML_Filter_Reason", "N/A"),
            "N/A" if pd.isna(latest_m15.get("ADX_14", np.nan)) else round(float(latest_m15["ADX_14"]), 2),
            "N/A" if pd.isna(latest_m15.get("BBW_Percentile", np.nan)) else f"{latest_m15['BBW_Percentile']:.0%}",
        ]
    })
    st.dataframe(plan, use_container_width=True)

    setups = m15[m15["Entry_Signal"].isin(["BUY SMC", "SELL SMC"])].dropna(subset=["Entry", "Suggested_SL", "Suggested_TP"]).tail(20)
    st.subheader("Latest Valid SMC Setups (with ML scoring)")
    if setups.empty:
        st.warning("No valid SMC setup found in the available 15m dataset.")
    else:
        setup_cols = [
            "Date", "Cameroon_Time", "Trading_Window",
            "Daily_Bias", "H1_Structure", "Structure", "Premium_Discount",
            "Entry_Signal", "Entry", "Suggested_SL", "Suggested_TP",
            "ML_Proba", "ML_Taken", "ML_Filter_Reason",
            "Sell_Side_Liquidity_Sweep", "Buy_Side_Liquidity_Sweep",
            "BOS_Bullish", "BOS_Bearish", "CHOCH_Bullish", "CHOCH_Bearish",
            "SMC_Reason"
        ]
        setup_cols = [c for c in setup_cols if c in setups.columns]
        st.dataframe(setups[setup_cols], use_container_width=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["SMC Charts", "Backtest", "Zones & Liquidity", "Data Export", "Strategy Rules", "🤖 ML Insights"]
    )

    with tab1:
        st.subheader("Daily Bias Chart")
        st.plotly_chart(smc_chart(daily, f"{selected} Daily Bias with Market Structure", selected, ["EMA_20", "EMA_50", "EMA_200"], False, True), use_container_width=True)
        st.subheader("1H Market Structure Chart")
        st.plotly_chart(smc_chart(h1, f"{selected} 1H Structure with Supply/Demand", selected, ["EMA_20", "EMA_50", "EMA_200"], False, True), use_container_width=True)
        st.subheader("15m Entry Chart")
        st.plotly_chart(smc_chart(m15, f"{selected} 15m SMC Entry Triggers", selected, ["EMA_20", "EMA_50", "EMA_200"], True, True), use_container_width=True)

    with tab2:
        st.subheader("SMC Strategy Backtest (rule-based, no ML filter)")
        if run_backtest:
            trades, metrics, equity = backtest_smc(m15, pip_size, initial_balance, risk_percent, max_hold)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Trades", metrics["Total Trades"])
            m2.metric("Win Rate", f"{metrics['Win Rate']}%")
            m3.metric("Net R", metrics["Net R"])
            m4.metric("Final Balance", metrics["Final Balance"])
            m5, m6, m7 = st.columns(3)
            m5.metric("Average R", metrics["Average R"])
            m6.metric("Profit Factor", metrics["Profit Factor"])
            m7.metric("Max Drawdown", f"{metrics['Max Drawdown %']}%")
            st.plotly_chart(equity_chart(equity), use_container_width=True)
            if trades.empty:
                st.warning("No trades found with the current strict SMC rules.")
            else:
                st.dataframe(trades.tail(100), use_container_width=True)
                st.download_button("Download Backtest CSV", trades.to_csv(index=False).encode("utf-8"), f"{selected}_SMC_backtest.csv", "text/csv")
            st.warning("Backtest is educational only. It does not guarantee future profit.")
        else:
            st.info("Backtest is disabled.")

    with tab3:
        st.subheader("Latest Zones & Liquidity")
        zone = pd.DataFrame({
            "Concept": [
                "Last Demand Zone", "Last Supply Zone", "Last Bullish FVG", "Last Bearish FVG",
                "Last Swing High", "Last Swing Low", "Equilibrium", "Premium/Discount"
            ],
            "Value": [
                format_zone(latest_m15, "Last_Demand_Low", "Last_Demand_High"),
                format_zone(latest_m15, "Last_Supply_Low", "Last_Supply_High"),
                format_zone(latest_m15, "Last_Bullish_FVG_Low", "Last_Bullish_FVG_High"),
                format_zone(latest_m15, "Last_Bearish_FVG_Low", "Last_Bearish_FVG_High"),
                "N/A" if pd.isna(latest_m15.get("Last_Swing_High", np.nan)) else round(float(latest_m15["Last_Swing_High"]), 5),
                "N/A" if pd.isna(latest_m15.get("Last_Swing_Low", np.nan)) else round(float(latest_m15["Last_Swing_Low"]), 5),
                "N/A" if pd.isna(latest_m15.get("Equilibrium", np.nan)) else round(float(latest_m15["Equilibrium"]), 5),
                latest_m15.get("Premium_Discount", "N/A")
            ]
        })
        st.dataframe(zone, use_container_width=True)

    with tab4:
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download Daily SMC Data", daily.to_csv(index=False).encode("utf-8"), f"{selected}_daily_smc.csv", "text/csv")
        c2.download_button("Download 1H SMC Data", h1.to_csv(index=False).encode("utf-8"), f"{selected}_1h_smc.csv", "text/csv")
        c3.download_button("Download 15m SMC + ML Data", m15.to_csv(index=False).encode("utf-8"), f"{selected}_15m_smc_ml.csv", "text/csv")

    with tab5:
        st.markdown("""
        ### Buy Plan
        1. Daily bias bullish.
        2. 1H structure bullish.
        3. 15m shows sell-side liquidity sweep, BOS, or CHOCH.
        4. Price reacts from demand zone, bullish FVG, or discount.
        5. Bullish candle confirms.
        6. SL below demand/FVG/swing low with ATR safety.
        7. TP uses selected risk-to-reward.
        8. ML layer confirms win-probability ≥ threshold, ADX ≥ minimum trend strength, and Bollinger Band Width above the volatility floor.

        ### Sell Plan
        1. Daily bias bearish.
        2. 1H structure bearish.
        3. 15m shows buy-side liquidity sweep, BOS, or CHOCH.
        4. Price reacts from supply zone, bearish FVG, or premium.
        5. Bearish candle confirms.
        6. SL above supply/FVG/swing high with ATR safety.
        7. TP uses selected risk-to-reward.
        8. ML layer confirms win-probability ≥ threshold, ADX ≥ minimum trend strength, and Bollinger Band Width above the volatility floor.

        ### Risk Rules
        - Avoid trading during major news spikes.
        - Use small risk on XAUUSD, BTCUSD, GBPJPY.
        - Backtest before using live.
        - Retrain the ML model periodically as new signals accumulate.
        - This is not financial advice.
        """)

    with tab6:
        st.subheader("Model Status")
        if bundle is None:
            st.warning(
                f"No trained ML model yet for {selected}. There are currently "
                f"{len(signal_rows)} resolved historical SMC signals available "
                f"({int(signal_rows['Label_Win'].sum()) if not signal_rows.empty else 0} wins). "
                f"Click **Train / Update ML Model for this market** in the sidebar once you "
                f"have at least ~20-30 resolved signals."
            )
        else:
            meta = bundle.get("meta", {})
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Trained on signals", meta.get("n_signals", "N/A"))
            mc2.metric("Training win rate", f"{meta.get('training_win_rate', 0):.1%}" if meta.get("training_win_rate") is not None else "N/A")
            mc3.metric("Model saved", bundle.get("saved_at", "N/A")[:19].replace("T", " "))

            wf = meta.get("walk_forward_summary", {})
            if wf.get("status") == "ok":
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("Walk-forward folds", wf.get("n_folds", "N/A"))
                wc2.metric("Mean OOS accuracy", f"{wf.get('mean_accuracy', 0):.1%}" if wf.get("mean_accuracy") is not None else "N/A")
                wc3.metric("Mean OOS ROC AUC", f"{wf.get('mean_roc_auc', 0):.3f}" if wf.get("mean_roc_auc") is not None else "N/A")
                fold_df = pd.DataFrame(meta.get("fold_metrics", []))
                if not fold_df.empty:
                    st.markdown("**Walk-forward fold-by-fold out-of-sample results**")
                    st.dataframe(fold_df, use_container_width=True)
            else:
                st.info(wf.get("message", "Walk-forward validation was not run (insufficient historical signals)."))

            st.divider()
            st.subheader("Feature Importance")
            fi = ml_engine.get_feature_importance(bundle["model"], bundle["feature_columns"])
            if not fi.empty:
                top_fi = fi.head(20)
                fig_fi = px.bar(top_fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                                 title="Top 20 Features Driving the ML Win-Probability Score")
                fig_fi.update_layout(height=560)
                st.plotly_chart(fig_fi, use_container_width=True)
                st.dataframe(fi, use_container_width=True)
                st.download_button(
                    "Download Feature Importance CSV",
                    fi.to_csv(index=False).encode("utf-8"),
                    f"{selected}_feature_importance.csv",
                    "text/csv",
                )

            st.divider()
            st.subheader("ML-Filtered Backtest (only ML-confirmed setups)")
            if run_backtest:
                m15_ml_filtered = ml_engine.apply_ml_filter_to_signals(m15)
                ml_trades, ml_metrics, ml_equity = backtest_smc(m15_ml_filtered, pip_size, initial_balance, risk_percent, max_hold)
                fm1, fm2, fm3, fm4 = st.columns(4)
                fm1.metric("Total Trades", ml_metrics["Total Trades"])
                fm2.metric("Win Rate", f"{ml_metrics['Win Rate']}%")
                fm3.metric("Net R", ml_metrics["Net R"])
                fm4.metric("Final Balance", ml_metrics["Final Balance"])
                fm5, fm6, fm7 = st.columns(3)
                fm5.metric("Average R", ml_metrics["Average R"])
                fm6.metric("Profit Factor", ml_metrics["Profit Factor"])
                fm7.metric("Max Drawdown", f"{ml_metrics['Max Drawdown %']}%")
                st.plotly_chart(equity_chart(ml_equity, title="ML-Filtered Backtest Equity Curve"), use_container_width=True)
                if ml_trades.empty:
                    st.warning("No trades passed the ML confidence + ADX + BB Width filters over this history.")
                else:
                    st.dataframe(ml_trades.tail(100), use_container_width=True)
                    st.download_button(
                        "Download ML-Filtered Backtest CSV",
                        ml_trades.to_csv(index=False).encode("utf-8"),
                        f"{selected}_ML_filtered_backtest.csv",
                        "text/csv",
                    )
                st.caption(
                    "Compare these metrics to the rule-based Backtest tab: the ML filter trades "
                    "fewer, hopefully higher-quality setups. Both backtests are educational only "
                    "and do not guarantee future profit."
                )
            else:
                st.info("Enable 'Run backtest' in the sidebar to see the ML-filtered comparison.")

    st.warning("This system is for education and strategy testing only. It is not financial advice and does not guarantee profit.")

except Exception as e:
    st.error("The app could not load the analysis.")
    st.exception(e)
