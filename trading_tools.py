"""
trading_tools.py
=================
Pure-Python position sizing + trading journal logic for the SMC + ML system.
No Streamlit or network dependency here on purpose, so it stays easy to
test/reuse (e.g. from a future CLI or notebook) exactly like ml_engine.py.

Two independent pieces:
  1. Lot size / position sizing math (calculate_lot_size, compute_pip_value_per_lot).
  2. A CSV-backed trading journal (load/save/add/close trade, running balance,
     summary stats) that persists across app restarts in data/trading_journal.csv,
     with a small JSON sidecar (data/journal_settings.json) for the starting
     balance.
"""

import os
import json

import numpy as np
import pandas as pd

DATA_DIR = "data"
JOURNAL_PATH = os.path.join(DATA_DIR, "trading_journal.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "journal_settings.json")

JOURNAL_COLUMNS = [
    "Trade ID", "Date Logged", "Market", "Direction",
    "Entry", "SL", "TP", "Lot Size", "Risk Amount", "Risk Percent",
    "ML Probability", "ML Verdict",
    "Status", "Outcome", "Exit Price", "Realized PL", "R Multiple",
    "Notes", "Date Closed",
]

DEFAULT_STARTING_BALANCE = 10000.0

# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

# 1 standard lot = this many units of the BASE currency (forex convention).
# Gold/BTC contract sizes vary by broker - these are common cTrader/CFD
# defaults; always confirm against your own broker's contract specifications
# and override in the UI if different.
CONTRACT_SIZE = {
    "EURUSD": 100000, "GBPUSD": 100000, "USDJPY": 100000, "AUDUSD": 100000,
    "NZDUSD": 100000, "USDCAD": 100000, "USDCHF": 100000, "EURJPY": 100000,
    "GBPJPY": 100000, "EURAUD": 100000,
    "XAUUSD": 100,   # 100 oz per 1.0 lot (typical CFD convention)
    "BTCUSD": 1,     # 1 BTC per 1.0 lot (typical cTrader crypto CFD convention)
}

# Pairs where the QUOTE currency is already USD - pip value per lot is a fixed
# USD amount, independent of the current price.
QUOTE_IS_USD = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD", "BTCUSD"}

# Pairs where USD is the BASE currency - pip value must be divided by the
# pair's own current price to convert into USD.
USD_IS_BASE = {"USDJPY", "USDCAD", "USDCHF"}

# Cross pairs (neither leg is directly USD-quoted) need an auxiliary USD
# conversion rate. Maps pair -> (yfinance ticker for the aux rate, mode).
# mode "divide": aux_rate is quote-currency-per-USD (e.g. USDJPY) -> divide.
# mode "multiply": aux_rate is USD-per-that-currency (e.g. AUDUSD) -> multiply.
CROSS_AUX = {
    "EURJPY": ("JPY=X", "divide"),
    "GBPJPY": ("JPY=X", "divide"),
    "EURAUD": ("AUDUSD=X", "multiply"),
}


def compute_pip_value_per_lot(pair_name, pip_size, current_price=None, aux_rate=None,
                               contract_size=None):
    """
    Returns the USD value of a 1-pip move for a 1.0 lot position in `pair_name`,
    or None if a required conversion rate is missing.
    """
    contract_size = contract_size if contract_size is not None else CONTRACT_SIZE.get(pair_name, 100000)
    raw_value = pip_size * contract_size  # value in the pair's own quote currency

    if pair_name in QUOTE_IS_USD:
        return raw_value

    if pair_name in USD_IS_BASE:
        if not current_price or current_price <= 0:
            return None
        return raw_value / current_price

    if pair_name in CROSS_AUX:
        if not aux_rate or aux_rate <= 0:
            return None
        _, mode = CROSS_AUX[pair_name]
        return raw_value * aux_rate if mode == "multiply" else raw_value / aux_rate

    # Unknown pair: fall back to the raw (quote-currency) value.
    return raw_value


def calculate_lot_size(balance, risk_percent, entry, sl, pip_size, pip_value_per_lot,
                        lot_step=0.01, min_lot=0.01, max_lot=100.0):
    """
    Core position-sizing formula:
        risk_amount   = balance * risk_percent / 100
        stop_pips     = |entry - sl| / pip_size
        lot_size      = risk_amount / (stop_pips * pip_value_per_lot)
    Rounded DOWN to the nearest lot_step so you never risk more than intended.
    Returns a dict with every intermediate value plus a `warning` (or None).
    """
    result = {
        "risk_amount": None, "stop_pips": None, "raw_lot": None,
        "lot_size": None, "warning": None,
    }
    try:
        balance = float(balance)
        risk_percent = float(risk_percent)
        entry = float(entry)
        sl = float(sl)
        pip_size = float(pip_size)
        pip_value_per_lot = float(pip_value_per_lot)
    except (TypeError, ValueError):
        result["warning"] = "Missing or invalid inputs."
        return result

    if pip_size <= 0 or pip_value_per_lot <= 0:
        result["warning"] = "Pip size and pip value must be greater than zero."
        return result

    stop_distance = abs(entry - sl)
    if stop_distance <= 0:
        result["warning"] = "Entry and Stop-Loss cannot be the same price."
        return result

    stop_pips = stop_distance / pip_size
    risk_amount = balance * (risk_percent / 100.0)
    raw_lot = risk_amount / (stop_pips * pip_value_per_lot)
    # Floor to the nearest lot_step, but guard against binary floating-point
    # noise (e.g. 0.49999999999999956 instead of an exact 0.5) incorrectly
    # flooring down a whole extra step - a real risk with money math in
    # binary floats. A tiny epsilon fixes this without ever rounding UP a
    # genuinely smaller value (real differences here are always >> 1e-9).
    epsilon = 1e-9
    lot_size = np.floor(raw_lot / lot_step + epsilon) * lot_step
    lot_size = round(float(lot_size), 2)

    result.update({
        "risk_amount": risk_amount, "stop_pips": stop_pips,
        "raw_lot": raw_lot, "lot_size": lot_size,
    })

    if lot_size < min_lot:
        result["warning"] = (
            f"Calculated lot size ({lot_size}) is below a typical broker minimum "
            f"({min_lot}). Your risk amount may be too small for this stop distance."
        )
    elif lot_size > max_lot:
        result["warning"] = (
            f"Calculated lot size ({lot_size}) is unusually large - double-check "
            f"your inputs before trading it."
        )
    return result


def potential_pl(lot_size, pip_value_per_lot, price_distance, pip_size):
    """USD P/L for a given lot size and price distance (e.g. Entry to TP)."""
    try:
        pips = abs(float(price_distance)) / float(pip_size)
        return float(lot_size) * float(pip_value_per_lot) * pips
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Trading journal (CSV-backed, persists across restarts)
# ---------------------------------------------------------------------------

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_journal():
    if not os.path.exists(JOURNAL_PATH):
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    try:
        df = pd.read_csv(JOURNAL_PATH)
    except Exception:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[JOURNAL_COLUMNS]


def save_journal(df):
    _ensure_data_dir()
    df.to_csv(JOURNAL_PATH, index=False)


def load_starting_balance():
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_STARTING_BALANCE
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        return float(data.get("starting_balance", DEFAULT_STARTING_BALANCE))
    except Exception:
        return DEFAULT_STARTING_BALANCE


def save_starting_balance(value):
    _ensure_data_dir()
    with open(SETTINGS_PATH, "w") as f:
        json.dump({"starting_balance": float(value)}, f)


def next_trade_id(df):
    if df.empty:
        return 1
    ids = pd.to_numeric(df["Trade ID"], errors="coerce")
    return int(ids.max()) + 1 if ids.notna().any() else 1


def add_trade(df, market, direction, entry, sl, tp, lot_size, risk_amount,
              risk_percent, ml_probability=None, ml_verdict=None, notes=""):
    _ensure_data_dir()
    new_id = next_trade_id(df)
    row = {col: np.nan for col in JOURNAL_COLUMNS}
    row.update({
        "Trade ID": new_id,
        "Date Logged": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Market": market,
        "Direction": direction,
        "Entry": entry,
        "SL": sl,
        "TP": tp,
        "Lot Size": lot_size,
        "Risk Amount": risk_amount,
        "Risk Percent": risk_percent,
        "ML Probability": ml_probability,
        "ML Verdict": ml_verdict,
        "Status": "Pending",
        "Notes": notes,
    })
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_journal(new_df)
    return new_df, new_id


def close_trade(df, trade_id, outcome, realized_pl, exit_price=None, r_multiple=None, notes=None):
    idx = df.index[pd.to_numeric(df["Trade ID"], errors="coerce") == trade_id]
    if len(idx) == 0:
        return df, False
    i = idx[0]
    df.loc[i, "Status"] = "Closed"
    df.loc[i, "Outcome"] = outcome
    df.loc[i, "Realized PL"] = realized_pl
    if exit_price is not None:
        df.loc[i, "Exit Price"] = exit_price
    if r_multiple is not None:
        df.loc[i, "R Multiple"] = r_multiple
    if notes:
        existing = df.loc[i, "Notes"]
        df.loc[i, "Notes"] = f"{existing} | {notes}" if isinstance(existing, str) and existing else notes
    df.loc[i, "Date Closed"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    save_journal(df)
    return df, True


def delete_trade(df, trade_id):
    idx = df.index[pd.to_numeric(df["Trade ID"], errors="coerce") == trade_id]
    if len(idx) == 0:
        return df, False
    df = df.drop(index=idx).reset_index(drop=True)
    save_journal(df)
    return df, True


def compute_current_balance(df, starting_balance):
    if df.empty:
        return float(starting_balance)
    closed = df[df["Status"] == "Closed"]
    realized = pd.to_numeric(closed["Realized PL"], errors="coerce").fillna(0).sum()
    return float(starting_balance) + float(realized)


def compute_journal_stats(df):
    total = len(df)
    pending = int((df["Status"] == "Pending").sum()) if total else 0
    closed = df[df["Status"] == "Closed"].copy() if total else pd.DataFrame(columns=JOURNAL_COLUMNS)
    if closed.empty:
        return {
            "Total Trades": total, "Closed Trades": 0, "Pending Trades": pending,
            "Win Rate": 0.0, "Total P/L": 0.0, "Average P/L": 0.0,
        }
    pl = pd.to_numeric(closed["Realized PL"], errors="coerce").fillna(0)
    wins = int((pl > 0).sum())
    return {
        "Total Trades": total,
        "Closed Trades": int(len(closed)),
        "Pending Trades": pending,
        "Win Rate": round(float(wins / len(closed) * 100), 2),
        "Total P/L": round(float(pl.sum()), 2),
        "Average P/L": round(float(pl.mean()), 2),
    }


def running_balance_series(df, starting_balance):
    """Returns a chronological (by Date Closed) DataFrame of closed trades
    with a cumulative Running Balance column, for an equity-curve chart."""
    closed = df[df["Status"] == "Closed"].copy()
    if closed.empty:
        return pd.DataFrame(columns=["Date Closed", "Realized PL", "Running Balance"])
    closed["Realized PL"] = pd.to_numeric(closed["Realized PL"], errors="coerce").fillna(0)
    closed = closed.sort_values("Date Closed")
    closed["Running Balance"] = float(starting_balance) + closed["Realized PL"].cumsum()
    return closed[["Date Closed", "Market", "Direction", "Outcome", "Realized PL", "Running Balance"]]
