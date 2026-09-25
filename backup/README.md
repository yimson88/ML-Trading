# SMC + ML Trading System (XGBoost Confidence Layer)

This is your Smart Money Concepts (SMC) Streamlit strategy, upgraded with a
complete machine-learning confidence layer. **The original rule-based SMC
engine is untouched** — structure (BOS/CHOCH), liquidity sweeps, fair value
gaps, supply/demand zones, premium/discount, and the Daily → 1H → 15m bias
cascade all work exactly as before. On top of that, an XGBoost model now
scores **every single SMC setup with a win-probability**, learned from that
market's own signal history, and only lets high-probability setups through
as "ML-Confirmed."

## What was added

| Requirement | Where it lives |
|---|---|
| ML-powered version | `ml_engine.py` wraps the SMC engine with an XGBoost confidence layer |
| XGBoost model | `ml_engine.train_xgb_model` / `train_final_model` |
| ADX filter | `add_adx()` (feature) + sidebar "ADX filter: minimum trend strength" (hard filter) |
| Bollinger Band Width filter | `add_bollinger()` (feature) + sidebar "BB Width filter: minimum volatility percentile" (hard filter) |
| Multi-timeframe feature engineering | `merge_multi_timeframe_features()` — Daily → H1 → M15, causal (shifted, no lookahead) |
| ATR dynamic SL/TP | Unchanged from the original strategy — zone/ATR-based SL, RR-based TP, recomputed every candle |
| Probability-based entries | `predict_signal_probability()` / `score_signals_with_model()` gate entries by `P(win) ≥ threshold` |
| Walk-forward training | `run_walk_forward()` — expanding-window, out-of-sample validated |
| Feature importance reporting | `get_feature_importance()` — CSV + bar chart in the "🤖 ML Insights" tab |
| Backtesting framework | `apply_ml_filter_to_signals()` feeds the existing `backtest_smc()` engine for an ML-filtered comparison alongside the original rule-based backtest |
| Model persistence (save/load) | `save_model_bundle()` / `load_model_bundle()` — one `.joblib` file per market in `models/` |
| Live signal generation | `score_signals_with_model()` scores the latest candle every time the app refreshes |

## How the ML layer actually works

1. **It does not replace the SMC rules.** A setup still has to satisfy Daily
   bias + 1H structure + 15m trigger (liquidity sweep / BOS / CHOCH) + zone
   confluence (demand/supply/FVG/premium-discount) exactly as before.
2. **Every historical setup is labeled honestly.** For each past `BUY SMC` /
   `SELL SMC` signal, the model replays the *exact same* `Suggested_SL` /
   `Suggested_TP` the strategy generated at the time, walks forward candle by
   candle, and records whether that specific trade would have won
   (`Label_Win = 1`) or lost (`Label_Win = 0`) within your chosen holding
   period. This is the same mechanic as the backtest engine — no separate,
   artificial labeling scheme.
3. **Features** (40 total) come from three timeframes:
   - **15m**: RSI, MACD histogram, ADX/+DI/-DI, Bollinger Band Width & %B,
     ATR%, EMA slope/ratios, candle body size, rolling volatility, plus every
     SMC feature already in the app (structure, premium/discount, BOS/CHOCH,
     liquidity sweeps, equal highs/lows, demand/supply/FVG zone membership).
   - **1H**: RSI, ADX, Bollinger Band Width, MACD%, ATR%, structure — always
     from the **last fully closed** 1H candle (no lookahead).
   - **Daily**: RSI, ADX, Bollinger Band Width, ATR%, bias — always from the
     **last fully closed** daily candle.
4. **Walk-forward validation**: signals are split chronologically into
   expanding folds — train on the past, test on the (unseen) future, roll
   forward. This is the honest way to estimate how the model would have
   performed live, rather than an in-sample fit that flatters itself.
5. **Final model**: once you're happy with the walk-forward numbers, a model
   is trained on *all* available signals and persisted for live use.
6. **Live scoring**: every time the app refreshes, the latest candle's
   features are fed into the saved model. If the SMC rules fire a signal, the
   app shows the ML win-probability and whether it passed the probability +
   ADX + Bollinger Band Width filters ("ML-Confirmed" vs "Filtered").

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

1. Pick a market in the sidebar.
2. Leave the SMC settings as they are (or tune them) — this is unchanged
   from your original strategy.
3. Scroll to **🤖 ML Confidence Layer (XGBoost)** in the sidebar:
   - **Minimum ML win-probability** — how confident the model must be
     before a setup counts as "ML-Confirmed" (default 0.58).
   - **ADX filter** — minimum trend strength on the 15m chart (default 18;
     raise this to avoid choppy/ranging conditions).
   - **BB Width filter** — minimum volatility percentile vs. the trailing
     100 candles (default 0.15; raise this to skip low-volatility squeezes).
   - **Label horizon** — how many 15m candles ahead a historical trade is
     allowed to resolve when building training labels.
   - **Walk-forward folds** — how many expanding-window folds to validate on.
4. Click **🔁 Train / Update ML Model for this market**. You need at least
   ~20-30 resolved historical signals for training to run (the app tells you
   how many are currently available). The first time you use a market, you
   may need to widen the "Daily data start date" and/or turn off "Strict
   mode" temporarily to accumulate enough signals, since **yfinance only
   provides ~60 days of 15-minute history** (see "Data limitations" below).
5. Open the **🤖 ML Insights** tab to see:
   - Model status (signals trained on, training win rate, when it was saved).
   - Walk-forward out-of-sample accuracy / ROC AUC, fold by fold.
   - Feature importance (table + chart) — which signals the model actually
     leans on.
   - An **ML-filtered backtest**, directly comparable to the rule-based
     backtest in the **Backtest** tab, so you can see whether the ML layer
     improves win rate / profit factor / drawdown versus trading every raw
     SMC signal.

Everything else — charts, zones, CSV export, strategy rules, Cameroon
trading window, Telegram-free alerts (toast/beep/desktop notification) —
behaves exactly as it did before.

## Batch / offline training (all markets at once)

Useful for a nightly cron job so models are always fresh without opening the
app:

```bash
python train_all_models.py                        # train every pair
python train_all_models.py --pair EURUSD GBPUSD    # just these two
python train_all_models.py --risk-reward 3.0 --atr-mult 1.0 --swing-len 3 \
                            --label-horizon 32 --n-splits 5 \
                            --daily-start 2020-01-01
```

This prints walk-forward metrics and feature importances to the console and
writes:
- `models/{PAIR}_xgb_model.joblib` — the persisted model bundle.
- `reports/{PAIR}_feature_importance.csv`
- `reports/{PAIR}_walk_forward_oos.csv`

The Streamlit app automatically picks up any model saved this way the next
time you load that market — no restart required beyond a normal refresh.

## Data limitations (important) and the CSV override

`yfinance` intraday limits:
- 15-minute candles: last **~60 days** only.
- 1-hour candles: last **~730 days**.
- Daily candles: effectively unlimited.

This caps how many historical SMC signals are available to train on early
on. Two ways to work around it:

1. **Just wait.** Every day the app runs, one more day of 15m history rolls
   in, and the ML layer's dataset grows. Retrain periodically (weekly is
   reasonable) via the sidebar button or `train_all_models.py`.
2. **Bring your own history.** If you export longer M15/H1/Daily history
   from your broker or MT5 (Date, Open, High, Low, Close[, Volume] columns),
   drop it at:
   ```
   data/history/{PAIR}_M15.csv
   data/history/{PAIR}_H1.csv
   data/history/{PAIR}_D1.csv
   ```
   e.g. `data/history/EURUSD_M15.csv`. If present, these are used **instead
   of** yfinance for that timeframe automatically — no code changes needed —
   which lets you walk-forward train on years of 15-minute data instead of
   60 days.

## Project structure

```
.
├── app.py                  # Streamlit app: SMC engine + ML confidence layer (run this)
├── ml_engine.py             # All ML logic: indicators, features, labeling,
│                            # walk-forward training, persistence, live scoring
├── train_all_models.py      # CLI: batch walk-forward train + persist + report
├── requirements.txt
├── models/                  # Saved XGBoost model bundles (.joblib), one per pair
├── reports/                 # Feature importance + walk-forward OOS CSVs
└── data/history/            # Optional broker/MT5 CSV overrides (see above)
```

## Notes and honest caveats

- With only ~60 days of 15m data, a brand-new market may have too few
  resolved signals to train reliably at first — the app and CLI both tell
  you clearly when this is the case instead of silently training a
  low-quality model.
- Walk-forward accuracy/AUC numbers in the **ML Insights** tab are the
  honest, out-of-sample estimate of the model's skill — trust those over the
  in-sample "training win rate" metric.
- The ML layer can only make the existing SMC signal set *smaller and
  higher-quality* (it filters, it doesn't invent new signals) — if the
  underlying SMC rules never fire on a given market/session, no amount of
  ML will produce a trade.
- This system, including its ML predictions, is for education and strategy
  research only. It is not financial advice and does not guarantee future
  profit. Always forward-test before risking real capital.
