"""
train_all_models.py
====================
Offline / batch trainer for the ML confidence layer. Use this instead of the
in-app "Train / Update ML Model" button when you want to (re)train models for
several markets at once, e.g. from a nightly cron job or scheduled task.

For each market it:
  1. Loads Daily / H1 / M15 data (yfinance, or data/history/{PAIR}_{TF}.csv
     overrides if present).
  2. Runs the same SMC engine as app.py (build_smc_system) to generate the
     rule-based BUY SMC / SELL SMC signals.
  3. Builds the multi-timeframe ML feature set and labels every historical
     signal by replaying its own ATR/zone-based SL and RR-based TP.
  4. Runs expanding-window walk-forward validation and prints/saves the
     out-of-sample metrics.
  5. Trains a final XGBoost model on all available signals and persists it
     to models/{PAIR}_xgb_model.joblib.
  6. Saves a feature-importance CSV and a walk-forward OOS CSV to reports/.

Usage
-----
    python train_all_models.py                      # train every pair in PAIRS
    python train_all_models.py --pair EURUSD         # train a single pair
    python train_all_models.py --pair EURUSD GBPUSD  # train a subset
    python train_all_models.py --risk-reward 3.0 --atr-mult 1.0 --swing-len 3 \\
                                --strict-mode --max-hold 32 --label-horizon 32 \\
                                --n-splits 5 --daily-start 2020-01-01
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import ml_engine

# Re-use the exact same SMC engine as app.py so training and live scoring
# are always built on identical logic. We import the functions directly
# from app.py without triggering Streamlit (app.py's top-level Streamlit
# calls are guarded so this works, see NOTE below).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "pip_size": 0.0001},
    "GBPUSD": {"ticker": "GBPUSD=X", "pip_size": 0.0001},
    "USDJPY": {"ticker": "JPY=X", "pip_size": 0.01},
    "AUDUSD": {"ticker": "AUDUSD=X", "pip_size": 0.0001},
    "NZDUSD": {"ticker": "NZDUSD=X", "pip_size": 0.0001},
    "USDCAD": {"ticker": "CAD=X", "pip_size": 0.0001},
    "USDCHF": {"ticker": "CHF=X", "pip_size": 0.0001},
    "EURJPY": {"ticker": "EURJPY=X", "pip_size": 0.01},
    "GBPJPY": {"ticker": "GBPJPY=X", "pip_size": 0.01},
    "EURAUD": {"ticker": "EURAUD=X", "pip_size": 0.0001},
    "XAUUSD": {"ticker": "GC=F", "pip_size": 0.01},
    "BTCUSD": {"ticker": "BTC-USD", "pip_size": 1.0},
}


# ---------------------------------------------------------------------------
# The functions below are duplicated (not imported) from app.py's data +
# SMC engine so this script has zero Streamlit dependency and can run
# headless in a cron job / CI pipeline. Keep this file's SMC logic
# byte-for-byte identical to app.py if you ever change the strategy rules.
# ---------------------------------------------------------------------------

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


def load_csv_override(pair_name, timeframe):
    path = os.path.join("data", "history", f"{pair_name}_{timeframe}.csv")
    if not os.path.exists(path):
        return None
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


def load_data(ticker, daily_start, pair_name):
    d_override = load_csv_override(pair_name, "D1")
    h_override = load_csv_override(pair_name, "H1")
    m_override = load_csv_override(pair_name, "M15")

    daily = d_override if d_override is not None and not d_override.empty else clean_yfinance_data(
        yf.download(ticker, start=daily_start, interval="1d", auto_adjust=False, progress=False))
    h1 = h_override if h_override is not None and not h_override.empty else clean_yfinance_data(
        yf.download(ticker, period="730d", interval="1h", auto_adjust=False, progress=False))
    m15 = m_override if m_override is not None and not m_override.empty else clean_yfinance_data(
        yf.download(ticker, period="60d", interval="15m", auto_adjust=False, progress=False))
    return daily, h1, m15


def add_cameroon_trading_window(df, start_hour=6, end_hour=22, enforce_window=True):
    df = df.copy()
    local_time = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert("Africa/Douala")
    df["Cameroon_Time"] = local_time.dt.strftime("%Y-%m-%d %H:%M")
    hour_value = local_time.dt.hour + (local_time.dt.minute / 60)
    if start_hour < end_hour:
        df["Trading_Window"] = (hour_value >= start_hour) & (hour_value < end_hour)
    else:
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
        if df["High"].iloc[i] == df["High"].iloc[i - swing_len:i + swing_len + 1].max():
            df.loc[df.index[i], "Swing_High"] = df["High"].iloc[i]
        if df["Low"].iloc[i] == df["Low"].iloc[i - swing_len:i + swing_len + 1].min():
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
            candles = df.iloc[max(0, i - 10):i]
            bearish = candles[candles["Close"] < candles["Open"]]
            if not bearish.empty:
                c = bearish.iloc[-1]
                low = min(c["Low"], c["Open"], c["Close"])
                high = max(c["Open"], c["Close"]) + atr * buffer_mult
                df.loc[df.index[i], "Demand_Zone_Low"] = low
                df.loc[df.index[i], "Demand_Zone_High"] = high
        if bool(df["BOS_Bearish"].iloc[i]) or bool(df["CHOCH_Bearish"].iloc[i]):
            candles = df.iloc[max(0, i - 10):i]
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


def build_smc_system(daily, h1, m15, risk_reward, atr_mult, swing_len, strict_mode,
                      session_start, session_end, enforce_session):
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

    m15 = add_cameroon_trading_window(m15, start_hour=session_start, end_hour=session_end, enforce_window=enforce_session)
    return daily, h1, m15


def train_one_pair(pair, args):
    print(f"\n{'=' * 70}\n{pair}\n{'=' * 70}")
    ticker = PAIRS[pair]["ticker"]
    try:
        daily, h1, m15 = load_data(ticker, args.daily_start, pair)
    except Exception as e:
        print(f"  [SKIP] Could not download data for {pair}: {e}")
        return

    if daily.empty or h1.empty or m15.empty:
        print(f"  [SKIP] Not enough data returned for {pair}.")
        return

    daily, h1, m15 = build_smc_system(
        daily, h1, m15,
        risk_reward=args.risk_reward, atr_mult=args.atr_mult, swing_len=args.swing_len,
        strict_mode=args.strict_mode, session_start=args.session_start, session_end=args.session_end,
        enforce_session=args.enforce_session,
    )

    m15_features, m15_labeled, signal_rows = ml_engine.build_ml_dataset(daily, h1, m15, max_hold_bars=args.label_horizon)
    n_signals = len(signal_rows)
    print(f"  Historical SMC signals resolved: {n_signals}")
    if n_signals == 0:
        print(f"  [SKIP] No resolved signals for {pair} with the current settings.")
        return
    print(f"  Training win rate: {signal_rows['Label_Win'].mean():.1%}")

    if n_signals < 20 or signal_rows["Label_Win"].nunique() < 2:
        print(f"  [SKIP] Not enough signals / class variety to train a model for {pair} "
              f"(need >= 20 signals with both wins and losses).")
        return

    oos_df, fold_metrics, wf_summary = ml_engine.run_walk_forward(signal_rows, n_splits=args.n_splits)
    print(f"  Walk-forward summary: {wf_summary}")
    for f in fold_metrics:
        print(f"    fold {f['fold']}: train={f['train_size']} test={f['test_size']} "
              f"acc={f['accuracy']} auc={f['roc_auc']} test_win_rate={f['test_win_rate']}")

    final_model, medians = ml_engine.train_final_model(signal_rows)
    if final_model is None:
        print(f"  [SKIP] Final model training failed for {pair} (single-class labels).")
        return

    fi = ml_engine.get_feature_importance(final_model)
    meta = {
        "n_signals": int(n_signals),
        "training_win_rate": round(float(signal_rows["Label_Win"].mean()), 4),
        "walk_forward_summary": wf_summary,
        "fold_metrics": fold_metrics,
        "label_horizon_bars": int(args.label_horizon),
        "risk_reward": float(args.risk_reward),
        "atr_mult": float(args.atr_mult),
        "strict_mode": bool(args.strict_mode),
    }
    path = ml_engine.save_model_bundle(pair, final_model, ml_engine.FEATURE_COLUMNS, medians, meta)
    print(f"  Model saved -> {path}")

    os.makedirs(ml_engine.REPORTS_DIR, exist_ok=True)
    fi_path = os.path.join(ml_engine.REPORTS_DIR, f"{pair}_feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    print(f"  Feature importance saved -> {fi_path}")
    print("  Top 10 features:")
    print(fi.head(10).to_string(index=False))

    if not oos_df.empty:
        oos_path = os.path.join(ml_engine.REPORTS_DIR, f"{pair}_walk_forward_oos.csv")
        oos_df.to_csv(oos_path, index=False)
        print(f"  Walk-forward OOS predictions saved -> {oos_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch walk-forward train the SMC ML confidence models.")
    parser.add_argument("--pair", nargs="*", default=None,
                         help="One or more pairs to train (default: all pairs in PAIRS).")
    parser.add_argument("--daily-start", default="2020-01-01")
    parser.add_argument("--risk-reward", type=float, default=2.5)
    parser.add_argument("--atr-mult", type=float, default=1.0)
    parser.add_argument("--swing-len", type=int, default=3)
    parser.add_argument("--strict-mode", action="store_true", default=True)
    parser.add_argument("--no-strict-mode", dest="strict_mode", action="store_false")
    parser.add_argument("--session-start", type=int, default=6)
    parser.add_argument("--session-end", type=int, default=22)
    parser.add_argument("--enforce-session", action="store_true", default=True)
    parser.add_argument("--no-enforce-session", dest="enforce_session", action="store_false")
    parser.add_argument("--max-hold", type=int, default=32, help="(kept for symmetry with the app; label horizon governs training)")
    parser.add_argument("--label-horizon", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    pairs = args.pair if args.pair else list(PAIRS.keys())
    unknown = [p for p in pairs if p not in PAIRS]
    if unknown:
        print(f"Unknown pair(s): {unknown}. Available: {list(PAIRS.keys())}")
        sys.exit(1)

    for pair in pairs:
        train_one_pair(pair, args)

    print(f"\nDone. Models in '{ml_engine.MODELS_DIR}/', reports in '{ml_engine.REPORTS_DIR}/'.")


if __name__ == "__main__":
    main()
