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

## Telegram alerts (especially ML-Validated trades)

Every SMC setup can now be pushed straight to Telegram with a clean,
formatted card — market, bias cascade, entry/SL/TP, R:R, and a prominent ML
verdict block:

- ✅ **ML-VALIDATED SIGNAL** — win-probability shown, sent whenever the ML
  layer confirms a setup.
- ⚠️ **SMC SETUP — FILTERED BY ML** — only sent if you turn off "Only send
  ML-Validated setups" in the sidebar.
- ⏳ **NO ML MODEL YET** — shown if a rule-based setup fires before you've
  trained a model for that market.

### Setup

1. Message **@BotFather** on Telegram, run `/newbot`, and copy the bot token
   it gives you.
2. Get your **Chat ID**: message your new bot once, then visit
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser — your
   numeric chat ID is in the JSON response. For a **channel**, add the bot
   as an admin of the channel and use the channel's `@username` or its
   numeric ID (also visible via `getUpdates` after posting in the channel).
3. Either:
   - Paste the Bot Token / Chat ID / Channel ID into the sidebar's
     **📲 Telegram Alerts** section each session, or
   - (Recommended) create `.streamlit/secrets.toml` in the project root so
     they load automatically:
     ```toml
     TELEGRAM_TOKEN = "123456789:AAExampleTokenFromBotFather"
     TELEGRAM_CHAT_ID = "987654321"
     TELEGRAM_CHANNEL_ID = "@my_signals_channel"
     ```
     Environment variables with the same names also work as a fallback.
4. Check **Enable Telegram alerts**, click **📨 Send Telegram test message**
   to confirm delivery, then leave it running — it will push a message
   automatically whenever a new (deduplicated) signal appears, without
   spamming the same setup twice.

Both your personal chat and a channel can receive alerts at the same time —
just fill in both fields; leave Channel ID blank to only send to your chat.

## Unattended VPS signal engine

The Streamlit dashboard can refresh itself only while a browser session is
connected. For true unattended Telegram delivery on a VPS, run the separate
headless worker:

```bash
python signal_engine.py --loop --interval-minutes 5
```

That worker runs this same production flow without Streamlit:

```text
load_data()
  -> build_smc_system()
  -> ml_engine.build_ml_dataset()
  -> ml_engine.score_signals_with_model()
  -> latest 15m candle
  -> Telegram
```

It stores sent-signal state in `data/signal_engine_state.json`, so restarting
the VPS or service does not resend the latest candle.

Recommended VPS layout:

```text
mlapp.service           # Streamlit dashboard
signal-engine.service   # server-side Telegram signal worker
```

Create a `.env` file in the project root on the VPS:

```bash
TELEGRAM_TOKEN=123456789:AAExampleTokenFromBotFather
TELEGRAM_CHAT_ID=987654321
TELEGRAM_CHANNEL_ID=@my_signals_channel

# Optional worker settings
SIGNAL_PAIRS=XAUUSD
SIGNAL_INTERVAL_MINUTES=5
SIGNAL_ONLY_ML_CONFIRMED=false
SIGNAL_ML_CONF_THRESHOLD=0.50
SIGNAL_ML_MIN_ADX=16
SIGNAL_ML_MIN_BBW_PCT=0.15
SIGNAL_STRICT_MODE=true
SIGNAL_ENFORCE_SESSION=false
SIGNAL_SESSION_START=6
SIGNAL_SESSION_END=22
```

Copy and edit the included systemd templates:

```bash
sudo cp systemd/mlapp.service /etc/systemd/system/mlapp.service
sudo cp systemd/signal-engine.service /etc/systemd/system/signal-engine.service
sudo nano /etc/systemd/system/mlapp.service
sudo nano /etc/systemd/system/signal-engine.service
```

Replace `YOUR_VPS_USER` and `/home/YOUR_VPS_USER/ML-Trading` with your real
Linux username and project path, then enable both services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mlapp.service
sudo systemctl enable --now signal-engine.service
```

Check logs:

```bash
journalctl -u signal-engine.service -f
journalctl -u mlapp.service -f
```

You can also run one manual check before enabling systemd:

```bash
python signal_engine.py --pair XAUUSD
```

Use `--only-ml-confirmed` if you want Telegram to send only ML-confirmed
setups:

```bash
python signal_engine.py --loop --interval-minutes 5 --only-ml-confirmed
```

## Position Size Calculator & Trading Journal

Two new tabs sit at the front of the app, ahead of the charts, so they're one
click away whenever a signal is live:

### 🧮 Lot Calculator
- Shows your **Current Account Balance** — dynamic, computed as
  `Starting Balance + sum of Realized P/L from every trade you've marked
  Closed in the journal`. It is not a static number you set once; it moves
  every time you log an outcome.
- When a live SMC setup exists for the selected market, Entry/SL/TP and
  direction auto-fill from it. Otherwise, fill them in manually for any
  hypothetical trade.
- Handles pip-value conversion correctly across pair types: direct USD-quote
  pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD, XAUUSD, BTCUSD), USD-as-base pairs
  (USDJPY, USDCAD, USDCHF — converted using the pair's own live price), and
  cross pairs (EURJPY, GBPJPY, EURAUD — converted via a fetched USDJPY/AUDUSD
  rate). Contract size defaults are typical CFD conventions (100,000 for FX,
  100 oz for gold, 1 BTC for crypto) — **confirm these against your own
  broker's contract specifications** and override them in "Advanced: Pip
  Value Settings" if they differ, or just type in the exact pip value your
  cTrader ticket shows.
- Outputs the recommended lot size (floored to 0.01 lot steps so you never
  round up into more risk than intended), stop distance in pips, risk
  amount, and potential profit/loss/R:R if a TP is set.
- Shows the 48-candle (12h) close-by deadline for the trade, and a **"Log
  this trade to journal"** button that records it as Pending in one click.

### 📓 Trading Journal
- Log trades (from the calculator or manually), then come back once a trade
  closes and record the outcome (Win/Loss/Breakeven, entered as $ P/L or as
  an R multiple) — the entry updates in place rather than creating a new
  record.
- Pending trades show their **48-candle close-by deadline** and flag in red
  once it's passed, as a reminder to check cTrader.
- Full history table, CSV export, an equity curve of closed trades, and a
  delete option for correcting mistaken entries.
- Everything persists in `data/trading_journal.csv` and
  `data/journal_settings.json` — safe across app restarts, not tied to a
  browser session.

The Backtest tab's "Initial balance" is intentionally left untouched — that
remains a separate, fixed parameter for historical strategy simulation and
does not mix with your live journal balance.

## Enforcing a 48-candle (15m) trade timeout on cTrader

cTrader's manual trading UI has no built-in "auto-close after N candles"
setting for an already-open position — the "Expiry" field you'll see only
applies to a **pending order** before it's filled (Good Till Cancelled /
Good Till Date), not to a live position. There are two practical ways to
enforce your 48-candle (= 12 hour) rule:

**1. Manual (no coding, works today)**
48 × 15m = 12 hours. The Trading Journal above now computes and shows a
"Close By" deadline for every pending trade (Date Logged + 12h) and flags it
once passed — use that as your reminder to open cTrader and close the
position manually if neither SL nor TP has hit.

**2. Automated, via cTrader Automate (cBot / cAlgo, C#)**
If you want it enforced automatically, cTrader's algo-trading feature
(cAlgo, similar in spirit to an MT5 EA but written in C#) can do this. A
minimal cBot:

```csharp
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class MaxHoldTimeExit : Robot
    {
        [Parameter("Max Hold Bars (15m candles)", DefaultValue = 48)]
        public int MaxHoldBars { get; set; }

        [Parameter("Label filter (blank = all positions)", DefaultValue = "")]
        public string LabelFilter { get; set; }

        protected override void OnBar()
        {
            int currentBarIndex = Bars.Count - 1;

            foreach (var position in Positions)
            {
                if (position.SymbolName != SymbolName)
                    continue;
                if (!string.IsNullOrEmpty(LabelFilter) && position.Label != LabelFilter)
                    continue;

                int entryBarIndex = Bars.GetIndexByTime(position.EntryTime);
                int barsHeld = currentBarIndex - entryBarIndex;

                if (barsHeld >= MaxHoldBars)
                {
                    ClosePosition(position);
                    Print($"Closed {position.Label} ({position.TradeType}) after {barsHeld} bars.");
                }
            }
        }
    }
}
```

Attach it to a 15-minute chart for the symbol you're trading; it counts
actual closed 15m bars since each position's entry (robust across weekend
gaps, unlike a plain 12-hour timer) and closes anything that reaches 48.
Test on a demo account first. If you'd rather not write/compile a cBot
yourself, third-party marketplaces (e.g. ClickAlgo) sell a ready-made
"Position Expiry Timer" cBot that does the same thing with no coding.

## Project structure

```
.
├── app.py                  # Streamlit app: SMC engine + ML confidence layer + Lot Calculator + Journal (run this)
├── ml_engine.py             # All ML logic: indicators, features, labeling,
│                            # walk-forward training, persistence, live scoring
├── trading_tools.py         # Lot sizing math + CSV-backed trading journal (no Streamlit dependency)
├── train_all_models.py      # CLI: batch walk-forward train + persist + report
├── requirements.txt
├── .streamlit/secrets.toml  # Optional: your Telegram token/chat/channel IDs (create this yourself, not committed)
├── models/                  # Saved XGBoost model bundles (.joblib), one per pair
├── reports/                 # Feature importance + walk-forward OOS CSVs
└── data/
    ├── history/              # Optional broker/MT5 CSV overrides (see above)
    ├── trading_journal.csv   # Your logged trades (created on first use)
    └── journal_settings.json # Your starting balance (created on first use)
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
