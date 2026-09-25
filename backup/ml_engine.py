"""
ml_engine.py
============
ML confidence layer for the Smart Money Concepts (SMC) trading system.

This module does NOT replace the rule-based SMC engine in app.py. It wraps it:
every time the rule-based engine fires a "BUY SMC" / "SELL SMC" signal, this
module scores that specific setup with an XGBoost classifier trained on the
market's own signal history, using multi-timeframe technical + SMC-structure
features (ADX, Bollinger Band Width, RSI, MACD, ATR, EMA slopes, structure,
liquidity sweeps, FVGs, supply/demand zones, premium/discount, Daily bias,
H1 structure). Only setups the model considers high-probability winners are
allowed through as "ML-Confirmed" signals.

Responsibilities:
  1. Classic indicators: RSI, MACD, ADX (+DI), Bollinger Bands / Band Width.
  2. Multi-timeframe feature engineering (Daily -> H1 -> M15 merge, causal/shifted).
  3. Label generation: for every historical SMC signal, replay the SAME
     ATR/zone-based SL and RR-based TP the strategy would have used, and
     record whether it actually won (Label_Win) within a max holding period.
  4. Walk-forward (expanding window) training + out-of-sample evaluation.
  5. Final model training on all available signals, for live deployment.
  6. Feature importance reporting.
  7. Model persistence (save/load) per market, via joblib.
  8. Live probability scoring for the latest candle.
  9. An ML-filtered scoring pass across full history, reusable by the
     existing backtest_smc() balance simulation in app.py (see
     apply_ml_filter_to_signals()).
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import log_loss, roc_auc_score

MODELS_DIR = "models"
REPORTS_DIR = "reports"

# ---------------------------------------------------------------------------
# Canonical feature list. Training and inference MUST use this exact list
# (same names, same order) so the persisted model always sees a consistent
# schema.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # Signal context
    "Direction",
    # M15 classic indicators
    "RSI_14", "MACD_Hist_Pct", "ADX_14", "PLUS_DI_14", "MINUS_DI_14",
    "BB_Width_20", "BB_PctB_20", "ATR_Pct",
    "EMA20_Slope_Pct", "Close_EMA20_Pct", "EMA20_EMA50_Pct", "EMA50_EMA200_Pct",
    "Candle_Body_ATR", "Return_Vol_20",
    # M15 SMC structure / liquidity / zones
    "Structure_Code", "Premium_Discount_Code",
    "BOS_Bullish", "BOS_Bearish", "CHOCH_Bullish", "CHOCH_Bearish",
    "Buy_Side_Liquidity_Sweep", "Sell_Side_Liquidity_Sweep",
    "Equal_Highs_Liquidity", "Equal_Lows_Liquidity",
    "In_Demand_Zone", "In_Supply_Zone", "In_Bullish_FVG", "In_Bearish_FVG",
    # H1 context (values are from the last CLOSED H1 candle only)
    "H1_RSI_14", "H1_ADX_14", "H1_BB_Width_20", "H1_MACD_Hist_Pct", "H1_ATR_Pct",
    "H1_Structure_Code",
    # Daily context (values are from the last CLOSED daily candle only)
    "D1_RSI_14", "D1_ADX_14", "D1_BB_Width_20", "D1_ATR_Pct",
    "Daily_Bias_Code",
]

STRUCTURE_CODE = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
PD_CODE = {"Premium": 1, "Discount": -1, "Neutral": 0}

BOOL_FEATURE_COLUMNS = [
    "BOS_Bullish", "BOS_Bearish", "CHOCH_Bullish", "CHOCH_Bearish",
    "Buy_Side_Liquidity_Sweep", "Sell_Side_Liquidity_Sweep",
    "Equal_Highs_Liquidity", "Equal_Lows_Liquidity",
    "In_Demand_Zone", "In_Supply_Zone", "In_Bullish_FVG", "In_Bearish_FVG",
]


# ---------------------------------------------------------------------------
# 1. CLASSIC INDICATORS
# ---------------------------------------------------------------------------

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df[f"RSI_{period}"] = rsi.fillna(50.0)
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd - macd_signal
    return df


def add_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    df[f"ADX_{period}"] = adx.fillna(0.0)
    df[f"PLUS_DI_{period}"] = plus_di.fillna(0.0)
    df[f"MINUS_DI_{period}"] = minus_di.fillna(0.0)
    return df


def add_bollinger(df, period=20, num_std=2.0):
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    df[f"BB_Mid_{period}"] = mid
    df[f"BB_Upper_{period}"] = upper
    df[f"BB_Lower_{period}"] = lower
    df[f"BB_Width_{period}"] = (upper - lower) / mid.replace(0, np.nan)
    df[f"BB_PctB_{period}"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    return df


def add_classic_indicators(df):
    """Adds RSI, MACD, ADX(+DI) and Bollinger Band Width/%B. Returns a copy."""
    df = df.copy()
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_adx(df, 14)
    df = add_bollinger(df, 20, 2.0)
    return df


def encode_state(series, mapping):
    return series.map(mapping).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# 2. MULTI-TIMEFRAME FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def _prep_htf_features(df, prefix):
    """
    Builds a small, prefixed, ONE-BAR-SHIFTED feature frame from a higher
    timeframe (H1 or Daily) so that only the last fully CLOSED higher
    timeframe candle is ever visible to the M15 model (no lookahead),
    mirroring the Daily_Bias / H1_Structure shift(1) convention already
    used by build_smc_system() in app.py.
    """
    d = df.copy()
    macd_hist_pct = (d["MACD_Hist"] / d["Close"]).replace([np.inf, -np.inf], np.nan)
    atr_pct = (d["ATR_14"] / d["Close"]).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame({
        "Date": d["Date"],
        f"{prefix}RSI_14": d["RSI_14"],
        f"{prefix}ADX_14": d["ADX_14"],
        f"{prefix}BB_Width_20": d["BB_Width_20"],
        f"{prefix}MACD_Hist_Pct": macd_hist_pct,
        f"{prefix}ATR_Pct": atr_pct,
    })
    feature_cols = [c for c in out.columns if c != "Date"]
    out[feature_cols] = out[feature_cols].shift(1)
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def merge_multi_timeframe_features(m15, h1, daily):
    """
    m15, h1, daily must already have add_classic_indicators() applied, and
    m15 must already be the output of build_smc_system() (so it carries
    Daily_Bias, H1_Structure, Entry_Signal, Structure, Premium_Discount,
    ATR_14, EMA_20/50/200, and all SMC boolean/zone columns).
    Returns m15 with all engineered + merged ML feature columns added.
    """
    m15 = m15.sort_values("Date").reset_index(drop=True).copy()

    h1_feat = _prep_htf_features(h1, "H1_")
    d1_feat = _prep_htf_features(daily, "D1_")

    if not h1_feat.empty:
        m15 = pd.merge_asof(m15, h1_feat, on="Date", direction="backward")
    if not d1_feat.empty:
        m15 = pd.merge_asof(m15, d1_feat, on="Date", direction="backward")

    # ---- M15-level engineered features ----
    m15["MACD_Hist_Pct"] = (m15["MACD_Hist"] / m15["Close"]).replace([np.inf, -np.inf], np.nan)
    m15["ATR_Pct"] = (m15["ATR_14"] / m15["Close"]).replace([np.inf, -np.inf], np.nan)
    m15["EMA20_Slope_Pct"] = m15["EMA_20"].pct_change(3)
    m15["Close_EMA20_Pct"] = (m15["Close"] / m15["EMA_20"] - 1.0)
    m15["EMA20_EMA50_Pct"] = (m15["EMA_20"] / m15["EMA_50"] - 1.0)
    m15["EMA50_EMA200_Pct"] = (m15["EMA_50"] / m15["EMA_200"] - 1.0)
    m15["Candle_Body_ATR"] = ((m15["Close"] - m15["Open"]) / m15["ATR_14"]).replace([np.inf, -np.inf], np.nan)
    m15["Return_Vol_20"] = m15["Close"].pct_change().rolling(20).std()

    m15["Structure_Code"] = encode_state(m15["Structure"], STRUCTURE_CODE)
    m15["Premium_Discount_Code"] = encode_state(m15["Premium_Discount"], PD_CODE)
    m15["H1_Structure_Code"] = encode_state(m15.get("H1_Structure", pd.Series("Neutral", index=m15.index)), STRUCTURE_CODE)
    m15["Daily_Bias_Code"] = encode_state(m15.get("Daily_Bias", pd.Series("Neutral", index=m15.index)), STRUCTURE_CODE)

    for c in BOOL_FEATURE_COLUMNS:
        if c in m15.columns:
            m15[c] = m15[c].fillna(False).astype(int)
        else:
            m15[c] = 0

    m15["Direction"] = 0
    m15.loc[m15["Entry_Signal"] == "BUY SMC", "Direction"] = 1
    m15.loc[m15["Entry_Signal"] == "SELL SMC", "Direction"] = -1

    return m15


def prepare_feature_matrix(df, feature_columns=None):
    """Reindexes df to the canonical feature columns (missing -> NaN, inf -> NaN)."""
    cols = feature_columns or FEATURE_COLUMNS
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


# ---------------------------------------------------------------------------
# 3. LABELING: replay each historical SMC signal's OWN SL/TP
# ---------------------------------------------------------------------------

def simulate_exit(d, entry_idx, direction, sl, tp, max_hold_bars):
    """Walks candle-by-candle from entry_idx looking for the SL/TP hit."""
    exit_idx = min(entry_idx + max_hold_bars, len(d) - 1)
    exit_price = None
    exit_reason = "TIME EXIT"
    for j in range(entry_idx, min(entry_idx + max_hold_bars + 1, len(d))):
        high = float(d.iloc[j]["High"])
        low = float(d.iloc[j]["Low"])
        if direction == "BUY":
            hit_sl, hit_tp = low <= sl, high >= tp
        else:
            hit_sl, hit_tp = high >= sl, low <= tp
        if hit_sl:
            exit_price, exit_reason, exit_idx = sl, "SL", j
            break
        if hit_tp:
            exit_price, exit_reason, exit_idx = tp, "TP", j
            break
    if exit_price is None:
        exit_price = float(d.iloc[exit_idx]["Close"])
    return exit_price, exit_reason, exit_idx


def label_smc_signals(m15_features, max_hold_bars=32):
    """
    For every historical BUY SMC / SELL SMC row, replays the SAME
    Suggested_SL / Suggested_TP the rule-based strategy generated and
    records whether it actually won. Adds Label_Win (0/1) and Label_R
    (realized R-multiple) columns; both are NaN for rows with no signal
    or where the trade could not be resolved (e.g. too close to the end
    of the dataset).
    """
    d = m15_features.sort_values("Date").reset_index(drop=True).copy()
    d["Label_Win"] = np.nan
    d["Label_R"] = np.nan

    signal_positions = d.index[d["Entry_Signal"].isin(["BUY SMC", "SELL SMC"])].tolist()
    for i in signal_positions:
        entry_idx = i + 1
        if entry_idx >= len(d) - 1:
            continue
        row = d.iloc[i]
        sl, tp = row.get("Suggested_SL", np.nan), row.get("Suggested_TP", np.nan)
        if pd.isna(sl) or pd.isna(tp):
            continue
        direction = "BUY" if row["Entry_Signal"] == "BUY SMC" else "SELL"
        entry_price = float(d.iloc[entry_idx]["Open"])
        exit_price, _, _ = simulate_exit(d, entry_idx, direction, float(sl), float(tp), max_hold_bars)

        if direction == "BUY":
            pips = exit_price - entry_price
            risk = entry_price - float(sl)
        else:
            pips = entry_price - exit_price
            risk = float(sl) - entry_price
        if risk <= 0:
            continue
        r_mult = pips / risk
        d.loc[i, "Label_R"] = r_mult
        d.loc[i, "Label_Win"] = 1.0 if r_mult > 0 else 0.0

    return d


def get_signal_rows(m15_labeled):
    """Returns only the labeled historical signal rows, chronologically sorted."""
    d = m15_labeled.dropna(subset=["Label_Win"]).copy()
    return d.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. WALK-FORWARD TRAINING
# ---------------------------------------------------------------------------

def default_xgb_params():
    return dict(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def train_xgb_model(X, y, params=None):
    from xgboost import XGBClassifier
    p = default_xgb_params()
    if params:
        p.update(params)
    model = XGBClassifier(**p)
    model.fit(X, y)
    return model


def time_series_walk_forward_splits(n, n_splits=5, min_train_size=15, min_test_size=5):
    """Expanding-window walk-forward split over n chronologically ordered samples."""
    if n < (min_train_size + min_test_size):
        return []
    max_splits = max(1, (n - min_train_size) // min_test_size)
    n_splits = max(1, min(n_splits, max_splits))
    fold_size = max(min_test_size, (n - min_train_size) // n_splits)

    splits = []
    train_end = min_train_size
    while train_end < n:
        test_end = min(train_end + fold_size, n)
        if test_end - train_end < 1:
            break
        splits.append((list(range(0, train_end)), list(range(train_end, test_end))))
        train_end = test_end
    return splits


def run_walk_forward(signal_df, feature_columns=None, n_splits=5, xgb_params=None):
    """
    Expanding-window walk-forward validation over historical SMC signals.
    Returns (oos_df, fold_metrics, summary).
    """
    cols = feature_columns or FEATURE_COLUMNS
    n = len(signal_df)
    splits = time_series_walk_forward_splits(n, n_splits=n_splits)

    if not splits:
        return pd.DataFrame(), [], {
            "status": "insufficient_data",
            "n_signals": n,
            "message": f"Only {n} historical signals available; need more history before walk-forward "
                       f"validation is meaningful. The model can still be trained on all available data.",
        }

    X_full_raw = prepare_feature_matrix(signal_df, cols)
    y_full = signal_df["Label_Win"].astype(int).reset_index(drop=True)

    oos_records = []
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y_full.iloc[train_idx]
        if y_train.nunique() < 2:
            continue

        X_train_raw = X_full_raw.iloc[train_idx]
        train_medians = X_train_raw.median(numeric_only=True).fillna(0.0)
        X_train = X_train_raw.fillna(train_medians).fillna(0.0)

        X_test_raw = X_full_raw.iloc[test_idx]
        X_test = X_test_raw.fillna(train_medians).fillna(0.0)
        y_test = y_full.iloc[test_idx]

        model = train_xgb_model(X_train, y_train, params=xgb_params)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = float((pred == y_test.values).mean())
        try:
            ll = float(log_loss(y_test, proba, labels=[0, 1]))
        except Exception:
            ll = None
        try:
            auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None
        except Exception:
            auc = None

        fold_metrics.append({
            "fold": fold_i,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4) if ll is not None else None,
            "roc_auc": round(auc, 4) if auc is not None else None,
            "test_win_rate": round(float(y_test.mean()), 4),
        })

        for pos, prob in zip(test_idx, proba):
            row = signal_df.iloc[pos]
            oos_records.append({
                "Date": row["Date"],
                "Direction": row.get("Entry_Signal"),
                "y_true": int(y_full.iloc[pos]),
                "proba_win": float(prob),
            })

    oos_df = pd.DataFrame(oos_records).sort_values("Date").reset_index(drop=True) if oos_records else pd.DataFrame()
    summary = {
        "status": "ok" if fold_metrics else "insufficient_class_variation",
        "n_signals": n,
        "n_folds": len(fold_metrics),
        "mean_accuracy": round(float(np.mean([f["accuracy"] for f in fold_metrics])), 4) if fold_metrics else None,
        "mean_roc_auc": round(float(np.mean([f["roc_auc"] for f in fold_metrics if f["roc_auc"] is not None])), 4)
        if any(f["roc_auc"] is not None for f in fold_metrics) else None,
    }
    return oos_df, fold_metrics, summary


def train_final_model(signal_df, feature_columns=None, params=None):
    """Trains on ALL available labeled signals, for live deployment."""
    cols = feature_columns or FEATURE_COLUMNS
    y = signal_df["Label_Win"].astype(int).reset_index(drop=True)
    if y.nunique() < 2:
        return None, None

    X_raw = prepare_feature_matrix(signal_df, cols)
    medians = X_raw.median(numeric_only=True).fillna(0.0).to_dict()
    X = X_raw.fillna(medians).fillna(0.0)

    model = train_xgb_model(X, y, params=params)
    return model, medians


# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_columns=None):
    cols = feature_columns or FEATURE_COLUMNS
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(cols):
        return pd.DataFrame(columns=["Feature", "Importance"])
    fi = pd.DataFrame({"Feature": cols, "Importance": importances})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. MODEL PERSISTENCE
# ---------------------------------------------------------------------------

def _safe_name(pair):
    return "".join(ch for ch in pair if ch.isalnum() or ch in ("-", "_"))


def model_path(pair):
    return os.path.join(MODELS_DIR, f"{_safe_name(pair)}_xgb_model.joblib")


def save_model_bundle(pair, model, feature_columns, medians, meta):
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": list(feature_columns),
        "feature_medians": dict(medians),
        "meta": dict(meta),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = model_path(pair)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(pair):
    path = model_path(pair)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 7. LIVE PREDICTION
# ---------------------------------------------------------------------------

def predict_signal_probability(bundle, feature_row):
    """feature_row: dict-like (e.g. df.iloc[-1].to_dict()). Returns float or None."""
    if bundle is None:
        return None
    cols = bundle["feature_columns"]
    medians = bundle.get("feature_medians", {})
    values = {}
    for c in cols:
        val = feature_row.get(c, np.nan)
        try:
            val = float(val)
            if np.isnan(val):
                raise ValueError
        except (TypeError, ValueError):
            val = float(medians.get(c, 0.0))
        values[c] = val
    X = pd.DataFrame([values], columns=cols)
    proba = bundle["model"].predict_proba(X)[:, 1][0]
    return float(proba)


# ---------------------------------------------------------------------------
# 8. ML-FILTERED SCORING ACROSS FULL HISTORY (for backtest reuse + tables)
# ---------------------------------------------------------------------------

def score_signals_with_model(m15_features, bundle, prob_threshold, min_adx=0.0,
                              min_bbw_percentile=0.0, bbw_lookback=100):
    """
    Scores every row with the persisted model (if any) and marks which
    historical SMC signals would have been taken under the ML confidence
    layer + ADX/BBW filters. Adds: ML_Proba, BBW_Percentile, ML_Taken,
    ML_Filter_Reason. Does not require Label_Win (works on unlabeled data too).
    """
    d = m15_features.sort_values("Date").reset_index(drop=True).copy()
    is_signal = d["Entry_Signal"].isin(["BUY SMC", "SELL SMC"])

    if bundle is None:
        d["ML_Proba"] = np.nan
        d["BBW_Percentile"] = d["BB_Width_20"].rolling(bbw_lookback, min_periods=10).rank(pct=True)
        d["ML_Taken"] = False
        d["ML_Filter_Reason"] = np.where(is_signal, "No trained model yet", "No SMC signal")
        return d

    cols = bundle["feature_columns"]
    medians = bundle.get("feature_medians", {})
    X_raw = prepare_feature_matrix(d, cols)
    X = X_raw.fillna(medians).fillna(0.0)
    d["ML_Proba"] = bundle["model"].predict_proba(X)[:, 1]
    d["BBW_Percentile"] = d["BB_Width_20"].rolling(bbw_lookback, min_periods=10).rank(pct=True)

    passes_conf = d["ML_Proba"] >= prob_threshold
    passes_adx = d["ADX_14"] >= min_adx
    passes_bbw = d["BBW_Percentile"].fillna(1.0) >= min_bbw_percentile

    d["ML_Taken"] = is_signal & passes_conf & passes_adx & passes_bbw
    d["ML_Filter_Reason"] = "No SMC signal"
    d.loc[is_signal, "ML_Filter_Reason"] = "OK - ML confirmed"
    d.loc[is_signal & ~passes_conf, "ML_Filter_Reason"] = "Below ML confidence threshold"
    d.loc[is_signal & passes_conf & ~passes_adx, "ML_Filter_Reason"] = "ADX below minimum trend strength"
    d.loc[is_signal & passes_conf & passes_adx & ~passes_bbw, "ML_Filter_Reason"] = "Bollinger Band Width below volatility filter"
    return d


def apply_ml_filter_to_signals(m15_scored):
    """
    Returns a copy where any SMC signal NOT confirmed by the ML layer is
    reset to "Wait", so it can be fed straight into the existing
    backtest_smc() balance simulation to produce an "ML-filtered" backtest
    directly comparable to the raw rule-based one.
    """
    d = m15_scored.copy()
    drop_mask = d["Entry_Signal"].isin(["BUY SMC", "SELL SMC"]) & (~d["ML_Taken"])
    d.loc[drop_mask, "Entry_Signal"] = "Wait"
    d.loc[drop_mask, ["Entry", "Suggested_SL", "Suggested_TP"]] = np.nan
    return d


# ---------------------------------------------------------------------------
# 9. FULL PIPELINE HELPER (used by both app.py and train_all_models.py)
# ---------------------------------------------------------------------------

def build_ml_dataset(daily, h1, m15, max_hold_bars=32):
    """
    daily, h1, m15 = the already-SMC-processed frames returned by
    build_smc_system() in app.py. Returns (m15_features, m15_labeled, signal_rows).
    """
    daily_ml = add_classic_indicators(daily)
    h1_ml = add_classic_indicators(h1)
    m15_ml = add_classic_indicators(m15)
    m15_features = merge_multi_timeframe_features(m15_ml, h1_ml, daily_ml)
    m15_labeled = label_smc_signals(m15_features, max_hold_bars=max_hold_bars)
    signal_rows = get_signal_rows(m15_labeled)
    return m15_features, m15_labeled, signal_rows
