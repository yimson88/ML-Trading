import os
import sys
import time
import types
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


CACHE_TTL_SECONDS = int(os.getenv("ML_SUMMARY_CACHE_SECONDS", "300"))
DEFAULT_MARKET = os.getenv("ML_SUMMARY_MARKET", "EURUSD")

app = FastAPI(title="Yimson ML Summary API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ML_SUMMARY_CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_cache = {"at": 0.0, "value": None, "market": None}
_defs = None


def install_streamlit_stub():
    if "streamlit" in sys.modules:
        return

    streamlit = types.ModuleType("streamlit")
    streamlit.secrets = {}
    streamlit.session_state = {}

    def cache_data(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    def noop(*_args, **_kwargs):
        return None

    streamlit.cache_data = cache_data
    streamlit.set_page_config = noop
    streamlit.stop = lambda: (_ for _ in ()).throw(SystemExit)
    components = types.ModuleType("streamlit.components")
    components_v1 = types.ModuleType("streamlit.components.v1")
    components.v1 = components_v1
    streamlit.components = components
    sys.modules["streamlit"] = streamlit
    sys.modules["streamlit.components"] = components
    sys.modules["streamlit.components.v1"] = components_v1


def load_app_definitions():
    global _defs
    if _defs is not None:
        return _defs

    install_streamlit_stub()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    source = (root / "app.py").read_text(encoding="utf-8")
    definitions_only = source.split("# -----------------------------\n# APP\n# -----------------------------", 1)[0]
    namespace = {"__file__": str(root / "app.py"), "__name__": "yimson_ml_defs"}
    exec(compile(definitions_only, str(root / "app.py"), "exec"), namespace)
    _defs = namespace
    return namespace


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_text(value, fallback="N/A"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    return str(value)


def build_summary(market=DEFAULT_MARKET):
    ns = load_app_definitions()
    import ml_engine

    pair = ns["PAIRS"][market]
    defaults = {
        "market": market,
        "dailyStart": os.getenv("ML_DAILY_START", "2020-01-01"),
        "swingSensitivity": int(os.getenv("ML_SWING_LEN", "2")),
        "riskReward": float(os.getenv("ML_RISK_REWARD", "2.5")),
        "atrSafetyBuffer": float(os.getenv("ML_ATR_MULT", "1.5")),
        "h1HistoryDays": int(os.getenv("ML_H1_DAYS", "180")),
        "sessionStart": int(os.getenv("ML_SESSION_START", "6")),
        "sessionEnd": int(os.getenv("ML_SESSION_END", "22")),
        "refreshMinutes": int(os.getenv("ML_REFRESH_MINUTES", "5")),
        "initialBalance": float(os.getenv("ML_INITIAL_BALANCE", "10000")),
        "riskPercent": float(os.getenv("ML_RISK_PERCENT", "1")),
        "maxHoldBars": int(os.getenv("ML_MAX_HOLD", "48")),
        "probabilityThreshold": float(os.getenv("ML_PROB_THRESHOLD", "0.50")),
        "probabilityRange": [0.5, 0.95],
        "minAdx": float(os.getenv("ML_MIN_ADX", "16")),
        "adxRange": [0, 40],
        "minBbWidth": float(os.getenv("ML_MIN_BBW", "0.15")),
        "bbWidthRange": [0, 0.9],
        "labelHorizon": int(os.getenv("ML_LABEL_HORIZON", "48")),
        "walkForwardFolds": int(os.getenv("ML_WALK_FORWARD_FOLDS", "6")),
    }

    daily, h1, m15 = ns["load_data"](
        pair["ticker"],
        defaults["dailyStart"],
        pair_name=market,
        h1_days=defaults["h1HistoryDays"],
    )
    if daily.empty or h1.empty or m15.empty:
        raise RuntimeError("Not enough market data returned.")

    daily, h1, m15 = ns["build_smc_system"](
        daily,
        h1,
        m15,
        defaults["riskReward"],
        defaults["atrSafetyBuffer"],
        defaults["swingSensitivity"],
        True,
        defaults["sessionStart"],
        defaults["sessionEnd"],
        False,
    )
    m15_features, _m15_labeled, signal_rows = ns["build_ml_dataset_cached"](
        daily,
        h1,
        m15,
        defaults["labelHorizon"],
    )
    bundle = ml_engine.load_model_bundle(market)
    scored = ml_engine.score_signals_with_model(
        m15_features,
        bundle,
        defaults["probabilityThreshold"],
        min_adx=defaults["minAdx"],
        min_bbw_percentile=defaults["minBbWidth"],
    )
    valid = scored.dropna(subset=["Close", "ATR_14", "Entry_Signal"])
    if valid.empty:
        raise RuntimeError("Indicators are not ready yet.")

    latest = valid.iloc[-1].to_dict()
    proba = safe_float(latest.get("ML_Proba"))
    adx = safe_float(latest.get("ADX_14"))
    signal = safe_text(latest.get("Entry_Signal"), "Wait")
    verdict = "Confirmed" if bool(latest.get("ML_Taken")) else ("No model" if bundle is None else "Filtered")
    trained = int(bundle["meta"].get("n_signals", len(signal_rows))) if bundle else "No model"
    message = (
        f"Valid SMC setup detected: {signal}."
        if signal in {"BUY SMC", "SELL SMC"}
        else "No valid SMC setup now. Waiting for liquidity + structure + zone confluence."
    )

    return {
        "ok": True,
        "source": "ml.yimson.tech",
        "market": market,
        "dailyBias": safe_text(latest.get("Daily_Bias"), "Neutral"),
        "h1Structure": safe_text(latest.get("H1_Structure"), "Neutral"),
        "m15Structure": safe_text(latest.get("Structure"), "Neutral"),
        "m15Signal": signal,
        "cameroonTime": safe_text(latest.get("Cameroon_Time"), "N/A"),
        "riskReward": defaults["riskReward"],
        "mlWinProbability": proba,
        "mlWinProbabilityText": "N/A" if proba is None else f"{proba:.1%}",
        "mlVerdict": verdict,
        "mlTaken": bool(latest.get("ML_Taken")),
        "filterReason": safe_text(latest.get("ML_Filter_Reason"), "N/A"),
        "adx15m": adx,
        "adx15mText": "N/A" if adx is None else f"{adx:.1f}",
        "bbwPercentile": safe_float(latest.get("BBW_Percentile")),
        "historicalSignalsTrainedOn": trained,
        "trainedSignalRows": int(len(signal_rows)),
        "entry": safe_float(latest.get("Entry")),
        "suggestedSl": safe_float(latest.get("Suggested_SL")),
        "suggestedTp": safe_float(latest.get("Suggested_TP")),
        "message": message,
        "defaults": defaults,
        "asOf": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def cached_summary(market=DEFAULT_MARKET, force=False):
    now = time.time()
    if (
        not force
        and _cache["value"] is not None
        and _cache["market"] == market
        and now - _cache["at"] < CACHE_TTL_SECONDS
    ):
        return _cache["value"]
    value = build_summary(market)
    _cache.update({"at": now, "value": value, "market": market})
    return value


@app.get("/api/ml-summary")
def ml_summary(market: str = DEFAULT_MARKET, refresh: bool = False):
    return cached_summary(market=market, force=refresh)


@app.get("/health")
def health():
    return {"ok": True}
