"""
signal_engine.py
================
Headless server-side signal worker for VPS deployments.

This script runs the same SMC -> ML scoring -> Telegram flow as the Streamlit
dashboard, but it does not need a browser session. Run it under systemd, tmux,
screen, or cron so signals continue while your laptop and phone are off.
"""

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import ml_engine
from train_all_models import PAIRS, build_smc_system, load_data


DEFAULT_PAIR = "XAUUSD"
DEFAULT_DAILY_START = "2020-01-01"
DEFAULT_SWING_LEN = 2
DEFAULT_RISK_REWARD = 2.5
DEFAULT_ATR_MULT = 1.5
DEFAULT_STRICT_MODE = True
DEFAULT_H1_DAYS = 180
DEFAULT_ENFORCE_SESSION = False
DEFAULT_SESSION_START = 6
DEFAULT_SESSION_END = 22
DEFAULT_REFRESH_MINUTES = 5
DEFAULT_ML_CONF_THRESHOLD = 0.50
DEFAULT_ML_MIN_ADX = 16
DEFAULT_ML_MIN_BBW_PCT = 0.15
DEFAULT_ML_HORIZON = 48
DEFAULT_STATE_PATH = Path("data") / "signal_engine_state.json"


def _load_streamlit_secret_file():
    path = Path(".streamlit") / "secrets.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return {}


_SECRETS = _load_streamlit_secret_file()


def get_secret(name, default=""):
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    value = _SECRETS.get(name, default)
    return "" if value is None else str(value)


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {"sent_keys": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"sent_keys": {}}
        data.setdefault("sent_keys", {})
        return data
    except Exception as exc:
        print(f"[WARN] Could not read state file {path}: {exc}")
        return {"sent_keys": {}}


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)
    tmp_path.replace(path)


def send_telegram_alert(message, token, chat_ids):
    results = {}
    token = str(token).strip() if token else ""
    if not token:
        print("[WARN] Telegram alert skipped: TELEGRAM_TOKEN is not configured.")
        return results

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    seen = set()
    for chat_id in chat_ids:
        chat_id = str(chat_id).strip() if chat_id else ""
        if not chat_id or chat_id in seen:
            continue
        seen.add(chat_id)
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            ok = bool(result.get("ok"))
            results[chat_id] = ok
            print(f"[INFO] Telegram {'sent' if ok else 'failed'} for {chat_id}: {result}")
        except Exception as exc:
            results[chat_id] = False
            print(f"[ERROR] Telegram connection error for {chat_id}: {exc}")
    return results


def _fmt_price(value, decimals=5):
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def build_telegram_signal_message(symbol, row, risk_reward, bundle):
    direction = row.get("Entry_Signal", "Wait")
    is_buy = direction == "BUY SMC"
    side = "BUY" if is_buy else "SELL"

    proba = row.get("ML_Proba", np.nan)
    has_proba = not (proba is None or (isinstance(proba, float) and np.isnan(proba)))
    ml_taken = bool(row.get("ML_Taken", False))
    has_model = bundle is not None

    if not has_model:
        header_tag = "SMC SETUP - NO ML MODEL YET"
        verdict_block = "No ML model trained yet for this market."
    elif ml_taken:
        header_tag = "ML-VALIDATED SIGNAL"
        verdict_block = f"*ML VALIDATED*\nWin Probability: *{proba:.1%}*" if has_proba else "*ML VALIDATED*"
    else:
        header_tag = "SMC SETUP - FILTERED BY ML"
        proba_txt = f"{proba:.1%}" if has_proba else "N/A"
        verdict_block = f"*Filtered by ML* (win probability {proba_txt})\nReason: {row.get('ML_Filter_Reason', 'n/a')}"

    bbw_pct = row.get("BBW_Percentile", np.nan)
    bbw_pct_str = "N/A" if (bbw_pct is None or (isinstance(bbw_pct, float) and np.isnan(bbw_pct))) else f"{bbw_pct:.0%}"

    lines = [
        f"*{side} SIGNAL*",
        f"_{header_tag}_",
        "--------------------",
        f"*{symbol}* | SMC + ML XGBoost",
        "",
        "*Bias Cascade*",
        f"Daily `{row.get('Daily_Bias', 'N/A')}` | 1H `{row.get('H1_Structure', 'N/A')}` | 15m `{row.get('Structure', 'N/A')}`",
        "",
        "*Trade Plan*",
        f"Entry: `{_fmt_price(row.get('Entry'))}`",
        f"SL: `{_fmt_price(row.get('Suggested_SL'))}`",
        f"TP: `{_fmt_price(row.get('Suggested_TP'))}`",
        f"R:R: `{risk_reward}:1`",
        "",
        "*ML Confidence Layer*",
        verdict_block,
        f"ADX (15m): `{_fmt_price(row.get('ADX_14'), 1)}` | BB Width %ile: `{bbw_pct_str}`",
        "",
        "*SMC Confluence*",
        f"{row.get('SMC_Reason', 'N/A')}",
        "",
        f"{row.get('Cameroon_Time', 'N/A')} Cameroon Time",
        "Educational signal only - not financial advice.",
    ]
    return "\n".join(lines)


def analyze_pair(pair, args):
    cfg = PAIRS[pair]
    daily, h1, m15 = load_data(cfg["ticker"], args.daily_start, pair, h1_days=args.h1_days)
    if daily.empty or h1.empty or m15.empty:
        raise RuntimeError("Not enough data returned.")

    daily, h1, m15 = build_smc_system(
        daily,
        h1,
        m15,
        risk_reward=args.risk_reward,
        atr_mult=args.atr_mult,
        swing_len=args.swing_len,
        strict_mode=args.strict_mode,
        session_start=args.session_start,
        session_end=args.session_end,
        enforce_session=args.enforce_session,
    )

    m15_features, _m15_labeled, _signal_rows = ml_engine.build_ml_dataset(
        daily, h1, m15, max_hold_bars=args.ml_horizon
    )
    bundle = ml_engine.load_model_bundle(pair)
    scored = ml_engine.score_signals_with_model(
        m15_features,
        bundle,
        args.ml_conf_threshold,
        min_adx=args.ml_min_adx,
        min_bbw_percentile=args.ml_min_bbw_pct,
    )
    valid = scored.dropna(subset=["Close", "ATR_14", "Entry_Signal"])
    if valid.empty:
        raise RuntimeError("Indicators are not ready yet.")
    return valid.iloc[-1].to_dict(), bundle


def run_once(args, state):
    token = get_secret("TELEGRAM_TOKEN")
    chat_ids = [get_secret("TELEGRAM_CHAT_ID"), get_secret("TELEGRAM_CHANNEL_ID")]
    any_sent = False

    for pair in args.pair:
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Checking {pair}...")
        try:
            latest, bundle = analyze_pair(pair, args)
        except Exception as exc:
            print(f"[ERROR] {pair}: {exc}")
            continue

        signal = latest.get("Entry_Signal", "Wait")
        print(
            f"[INFO] {pair}: candle={latest.get('Date')} signal={signal} "
            f"ml_taken={bool(latest.get('ML_Taken'))} reason={latest.get('ML_Filter_Reason', 'n/a')}"
        )
        if signal not in ["BUY SMC", "SELL SMC"]:
            continue

        is_ml_confirmed = bool(latest.get("ML_Taken"))
        if args.only_ml_confirmed and not is_ml_confirmed:
            continue

        telegram_key = f"{pair}-{latest.get('Date')}-{signal}-{is_ml_confirmed}"
        if state["sent_keys"].get(pair) == telegram_key:
            print(f"[INFO] {pair}: signal already sent ({telegram_key}).")
            continue

        message = build_telegram_signal_message(pair, latest, args.risk_reward, bundle)
        results = send_telegram_alert(message, token, chat_ids)
        if any(results.values()):
            state["sent_keys"][pair] = telegram_key
            state["last_sent_at"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            any_sent = True
        elif results:
            print(f"[ERROR] {pair}: Telegram send failed; state was not advanced.")

    return any_sent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SMC + ML Telegram signal engine without Streamlit.")
    parser.add_argument("--pair", nargs="+", default=os.environ.get("SIGNAL_PAIRS", DEFAULT_PAIR).replace(",", " ").split())
    parser.add_argument("--daily-start", default=os.environ.get("SIGNAL_DAILY_START", DEFAULT_DAILY_START))
    parser.add_argument("--risk-reward", type=float, default=env_float("SIGNAL_RISK_REWARD", DEFAULT_RISK_REWARD))
    parser.add_argument("--atr-mult", type=float, default=env_float("SIGNAL_ATR_MULT", DEFAULT_ATR_MULT))
    parser.add_argument("--swing-len", type=int, default=env_int("SIGNAL_SWING_LEN", DEFAULT_SWING_LEN))
    parser.add_argument("--strict-mode", dest="strict_mode", action="store_true", default=env_bool("SIGNAL_STRICT_MODE", DEFAULT_STRICT_MODE))
    parser.add_argument("--no-strict-mode", dest="strict_mode", action="store_false")
    parser.add_argument("--h1-days", type=int, default=env_int("SIGNAL_H1_DAYS", DEFAULT_H1_DAYS))
    parser.add_argument("--session-start", type=int, default=env_int("SIGNAL_SESSION_START", DEFAULT_SESSION_START))
    parser.add_argument("--session-end", type=int, default=env_int("SIGNAL_SESSION_END", DEFAULT_SESSION_END))
    parser.add_argument("--enforce-session", dest="enforce_session", action="store_true", default=env_bool("SIGNAL_ENFORCE_SESSION", DEFAULT_ENFORCE_SESSION))
    parser.add_argument("--no-enforce-session", dest="enforce_session", action="store_false")
    parser.add_argument("--ml-conf-threshold", type=float, default=env_float("SIGNAL_ML_CONF_THRESHOLD", DEFAULT_ML_CONF_THRESHOLD))
    parser.add_argument("--ml-min-adx", type=float, default=env_float("SIGNAL_ML_MIN_ADX", DEFAULT_ML_MIN_ADX))
    parser.add_argument("--ml-min-bbw-pct", type=float, default=env_float("SIGNAL_ML_MIN_BBW_PCT", DEFAULT_ML_MIN_BBW_PCT))
    parser.add_argument("--ml-horizon", type=int, default=env_int("SIGNAL_ML_HORIZON", DEFAULT_ML_HORIZON))
    parser.add_argument("--only-ml-confirmed", action="store_true", default=env_bool("SIGNAL_ONLY_ML_CONFIRMED", False))
    parser.add_argument("--state-path", default=os.environ.get("SIGNAL_STATE_PATH", str(DEFAULT_STATE_PATH)))
    parser.add_argument("--loop", action="store_true", default=env_bool("SIGNAL_LOOP", False))
    parser.add_argument("--interval-minutes", type=float, default=env_float("SIGNAL_INTERVAL_MINUTES", DEFAULT_REFRESH_MINUTES))
    args = parser.parse_args()

    unknown = [p for p in args.pair if p not in PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {unknown}. Available: {list(PAIRS.keys())}")
    if args.interval_minutes <= 0:
        raise SystemExit("--interval-minutes must be greater than 0")
    return args


def main():
    args = parse_args()
    state = load_state(args.state_path)
    print(f"[INFO] Signal engine starting. pairs={args.pair} loop={args.loop} interval={args.interval_minutes}m")

    while True:
        run_once(args, state)
        save_state(args.state_path, state)
        if not args.loop:
            break
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
