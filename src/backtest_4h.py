import os
import joblib
import psycopg
import pandas as pd
import numpy as np
from train import build_features, FEATURES

# ==========================================
# ACCOUNT & BACKTEST CONFIGURATION
# ==========================================
INITIAL_BALANCE = 1000.00   # $1,000 Starting Balance
LOT_SIZE        = 0.02      # 0.02 Lots
DOLLAR_PER_PT   = 0.02      # $0.02 per 1.00 index point move
LEVERAGE        = 2000      # 1:2000 Leverage
TIMEFRAME       = "4h"
MODEL_PATH      = f"model_{TIMEFRAME}.pkl"

# How many 5-minute bars before a 4H trade expires if TP isn't hit (4 hours = 48 M5 bars)
MAX_HOLD_BARS_M5 = 48       

def load_all_historical_data():
    """
    Loads all 5m bars from data_m5.csv (or pulls from DB if you store raw candles in Postgres).
    """
    if os.path.exists("data_m5.csv"):
        df = pd.read_csv("data_m5.csv")
        print(f"Loaded {len(df):,} raw rows from data_m5.csv")
        return df
    else:
        raise FileNotFoundError("data_m5.csv not found.")

def run_simulation(df_feat, signals, eq_targets, mode="SINGLE_POSITION"):
    """
    Simulates execution on M5 bars using High/Low/Close to check Take Profit (EQ).
    mode = 'SINGLE_POSITION' (wait until open trade closes)
    mode = 'MULTI_POSITION'  (open a new 0.02 lot trade on every 5m trigger)
    """
    balance = INITIAL_BALANCE
    equity_curve = []
    open_positions = []  # List of dicts: {type, entry, tp, sl, open_idx, margin}
    
    wait_count = 0
    executed_count = 0
    closed_trades = []
    max_concurrent_positions = 0

    closes = df_feat["close"].values
    highs  = df_feat["high"].values if "high" in df_feat.columns else closes
    lows   = df_feat["low"].values if "low" in df_feat.columns else closes

    for i in range(len(df_feat)):
        price = float(closes[i])
        high  = float(highs[i])
        low   = float(lows[i])
        sig   = signals[i]
        tp_eq = round(float(eq_targets[i]), 0)

        # 1. Check existing open positions for Take Profit (EQ) or 4-Hour Time Expiration
        still_open = []
        for pos in open_positions:
            bars_held = i - pos["open_idx"]
            closed = False
            exit_price = price
            exit_reason = ""

            if pos["type"] == "BUY":
                if high >= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP_HIT"
                    closed = True
                elif bars_held >= MAX_HOLD_BARS_M5:
                    exit_price = price
                    exit_reason = "TIME_EXPIRY_4H"
                    closed = True
                
                if closed:
                    pts = exit_price - pos["entry"]
                    pnl = pts * DOLLAR_PER_PT
                    balance += pnl
                    closed_trades.append({"pnl": pnl, "pts": pts, "reason": exit_reason, "bars": bars_held})

            elif pos["type"] == "SELL":
                if low <= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP_HIT"
                    closed = True
                elif bars_held >= MAX_HOLD_BARS_M5:
                    exit_price = price
                    exit_reason = "TIME_EXPIRY_4H"
                    closed = True

                if closed:
                    pts = pos["entry"] - exit_price
                    pnl = pts * DOLLAR_PER_PT
                    balance += pnl
                    closed_trades.append({"pnl": pnl, "pts": pts, "reason": exit_reason, "bars": bars_held})

            if not closed:
                still_open.append(pos)

        open_positions = still_open

        # 2. Calculate Floating PnL & Margin for 1:2000 Leverage
        floating_pnl = 0.0
        used_margin = 0.0
        for pos in open_positions:
            pts = (price - pos["entry"]) if pos["type"] == "BUY" else (pos["entry"] - price)
            floating_pnl += pts * DOLLAR_PER_PT
            used_margin += pos["margin"]

        equity = balance + floating_pnl
        free_margin = equity - used_margin

        # 3. Check New Signal Execution
        can_open = False
        if sig in ["BUY", "SELL"]:
            if mode == "SINGLE_POSITION" and len(open_positions) == 0:
                can_open = True
            elif mode == "MULTI_POSITION":
                can_open = True

        # Margin required per 0.02 lot at 1:2000 leverage:
        # Notional Value = Price * $0.02/pt -> Margin = Notional / 2000
        req_margin = (price * DOLLAR_PER_PT) / LEVERAGE

        if can_open and free_margin > req_margin:
            open_positions.append({
                "type": sig,
                "entry": round(price, 0),
                "tp": tp_eq,
                "open_idx": i,
                "margin": req_margin
            })
            executed_count += 1
        else:
            wait_count += 1

        if len(open_positions) > max_concurrent_positions:
            max_concurrent_positions = len(open_positions)

        equity_curve.append(equity)

    # Close any remaining open positions at the end of the backtest
    final_price = float(closes[-1])
    for pos in open_positions:
        pts = (final_price - pos["entry"]) if pos["type"] == "BUY" else (pos["entry"] - final_price)
        pnl = pts * DOLLAR_PER_PT
        balance += pnl
        closed_trades.append({"pnl": pnl, "pts": pts, "reason": "END_OF_DATA", "bars": len(df_feat) - pos["open_idx"]})

    trades_df = pd.DataFrame(closed_trades)
    wins = len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty else 0
    total_closed = len(trades_df)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    # Max Drawdown calculation
    eq_series = pd.Series(equity_curve)
    rolling_max = eq_series.cummax()
    drawdown = eq_series - rolling_max
    max_dd = drawdown.min()

    return {
        "mode": mode,
        "wait_count": wait_count,
        "executed_count": executed_count,
        "total_closed": total_closed,
        "win_rate": win_rate,
        "final_balance": balance,
        "net_profit": balance - INITIAL_BALANCE,
        "max_dd": max_dd,
        "max_concurrent": max_concurrent_positions
    }

def main():
    raw_df = load_all_historical_data()
    
    # Build 4H features on the M5 dataframe (exact match to predict.py)
    df_feat = build_features(raw_df, TIMEFRAME).dropna().reset_index(drop=True)
    total_bars = len(df_feat)
    
    # Calculate Exact Days Covered (23 hours/day = 276 five-minute bars/day)
    trading_days = total_bars / 276.0
    print("\n" + "="*65)
    print(f"📊 DATA COVERAGE SUMMARY (5-Minute Evaluation of 4H Model)")
    print("="*65)
    print(f"Raw Rows in CSV         : {len(raw_df):,}")
    print(f"Rows dropped by warmup  : {len(raw_df) - total_bars:,} (EMA/RSI lookback)")
    print(f"Evaluated 5-Min Bars    : {total_bars:,}")
    print(f"Trading Days Covered    : {trading_days:.2f} Trading Days (@ 23 hrs/day)")
    if "timestamp" in df_feat.columns or "time" in df_feat.columns:
        ts_col = "timestamp" if "timestamp" in df_feat.columns else "time"
        print(f"Date Range Covered      : {df_feat[ts_col].iloc[0]}  -->  {df_feat[ts_col].iloc[-1]}")
    print("="*65)

    # Load 4H Model & Predict All Rows Vectorized (Fast!)
    models = joblib.load(MODEL_PATH)
    X_all = df_feat[FEATURES]
    
    pred_up = models["up"].predict(X_all)
    pred_down = models["down"].predict(X_all)
    prices = df_feat["close"].values

    proj_high = prices + pred_up
    proj_low = prices - pred_down
    total_range = np.maximum(proj_high - proj_low, 10.0)

    midline_eq = (proj_high + proj_low) / 2.0
    half_range = total_range / 2.0

    sell_trigger = midline_eq + (half_range * 0.60)
    buy_trigger  = midline_eq - (half_range * 0.60)

    # Generate Raw Signals (Matches your 1640 WAIT / 814 EXECUTED count)
    signals = np.where(prices <= buy_trigger, "BUY",
              np.where(prices >= sell_trigger, "SELL", "WAIT"))

    raw_wait = int(np.sum(signals == "WAIT"))
    raw_exec = int(np.sum(signals != "WAIT"))
    print(f"\nRAW SIGNAL COUNT (Every 5m bar): {raw_wait} WAIT | {raw_exec} BUY/SELL TRIGGERS")

    # Run Both Position Modes
    res_single = run_simulation(df_feat, signals, midline_eq, mode="SINGLE_POSITION")
    res_multi  = run_simulation(df_feat, signals, midline_eq, mode="MULTI_POSITION")

    print("\n" + "="*65)
    print("📈 BACKTEST RESULTS ($1,000 Start | 1:2000 Lev | 0.02 Lot = $0.02/pt)")
    print("="*65)
    for res in [res_single, res_multi]:
        print(f"\n>>> MODE: {res['mode']} <<<")
        print(f"  Bars Waited (No Entry)    : {res['wait_count']:,}")
        print(f"  Trades Opened (Executed)  : {res['executed_count']:,}")
        print(f"  Max Open Positions at Once: {res['max_concurrent']}")
        print(f"  Win Rate                  : {res['win_rate']:.1f}%")
        print(f"  Final Balance             : ${res['final_balance']:,.2f} (Net: ${res['net_profit']:+,.2f})")
        print(f"  Max Equity Drawdown       : ${res['max_dd']:,.2f}")

if __name__ == "__main__":
    main()
