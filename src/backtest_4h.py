import os
import joblib
import psycopg
import pandas as pd
import numpy as np
from train import build_features, FEATURES

# ==========================================
# ACCOUNT & BACKTEST CONFIGURATION
# ==========================================
DB_URL          = os.environ.get("DATABASE_URL")
TABLE_NAME      = "m5_bars"  # Your 40,148-row Supabase table
INITIAL_BALANCE = 1000.00    # $1,000 Starting Balance
LOT_SIZE        = 0.02       # 0.02 Lots
DOLLAR_PER_PT   = 0.02       # $0.02 per 1.00 index point move
LEVERAGE        = 2000       # 1:2000 Leverage
TIMEFRAME       = "4h"
MODEL_PATH      = f"model_{TIMEFRAME}.pkl"

def parse_mixed_timestamps(series):
    """
    Safely parses all timestamps (including October rows with microseconds)
    so all 40,148+ rows up to October 22+ are preserved.
    """
    try:
        return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")

def load_all_m5_bars_from_supabase():
    """
    Pulls all 40,148+ rows directly from the 'm5_bars' table in Supabase.
    """
    if not DB_URL:
        raise ValueError("DATABASE_URL environment variable is not set!")

    print(f"Connecting to Supabase and loading all rows from '{TABLE_NAME}'...")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT timestamp, open, high, low, close, volume FROM {TABLE_NAME};")
            rows = cur.fetchall()
            col_names = [desc[0].lower() for desc in cur.description]

    df = pd.DataFrame(rows, columns=col_names)
    print(f"Successfully loaded {len(df):,} raw rows from Supabase '{TABLE_NAME}'!")

    df["timestamp"] = parse_mixed_timestamps(df["timestamp"])
    df = (
        df.dropna(subset=["timestamp"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values(by="timestamp")
        .reset_index(drop=True)
    )

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df

def run_simulation(df_feat, signals, eq_targets, mode="SINGLE_POSITION"):
    """
    Simulates execution across all M5 bars:
    - Positions NEVER close by time expiry. They stay open until price hits their EQ Take-Profit,
      even if it takes days or weeks.
    - SINGLE_POSITION: Waits until the open position hits EQ TP before opening a new trade.
    - MULTI_POSITION : Opens a 0.02 lot trade on every 5m bar where BUY/SELL triggers.
    """
    balance = INITIAL_BALANCE
    equity_curve = []
    open_positions = []
    
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

        # 1. Check active positions ONLY for EQ Take-Profit (No Time Expiry!)
        still_open = []
        for pos in open_positions:
            bars_held = i - pos["open_idx"]
            closed = False
            exit_price = price

            if pos["type"] == "BUY":
                if high >= pos["tp"]:
                    exit_price = pos["tp"]
                    closed = True
                
                if closed:
                    pts = exit_price - pos["entry"]
                    pnl = pts * DOLLAR_PER_PT
                    balance += pnl
                    closed_trades.append({
                        "pnl": pnl,
                        "pts": pts,
                        "reason": "TP_HIT_EQ",
                        "bars": bars_held
                    })

            elif pos["type"] == "SELL":
                if low <= pos["tp"]:
                    exit_price = pos["tp"]
                    closed = True

                if closed:
                    pts = pos["entry"] - exit_price
                    pnl = pts * DOLLAR_PER_PT
                    balance += pnl
                    closed_trades.append({
                        "pnl": pnl,
                        "pts": pts,
                        "reason": "TP_HIT_EQ",
                        "bars": bars_held
                    })

            if not closed:
                still_open.append(pos)

        open_positions = still_open

        # 2. Calculate Floating PnL & Margin Used (1:2000 Leverage)
        floating_pnl = 0.0
        used_margin = 0.0
        for pos in open_positions:
            pts = (price - pos["entry"]) if pos["type"] == "BUY" else (pos["entry"] - price)
            floating_pnl += pts * DOLLAR_PER_PT
            used_margin += pos["margin"]

        equity = balance + floating_pnl
        free_margin = equity - used_margin

        # 3. Check New Entry Signal
        can_open = False
        if sig in ["BUY", "SELL"]:
            if mode == "SINGLE_POSITION" and len(open_positions) == 0:
                can_open = True
            elif mode == "MULTI_POSITION":
                can_open = True

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

    # Calculate final floating PnL for any positions still waiting to hit EQ at the end of data
    final_price = float(closes[-1])
    final_floating_pnl = 0.0
    for pos in open_positions:
        pts = (final_price - pos["entry"]) if pos["type"] == "BUY" else (pos["entry"] - final_price)
        final_floating_pnl += pts * DOLLAR_PER_PT

    final_equity = balance + final_floating_pnl

    trades_df = pd.DataFrame(closed_trades)
    wins = len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty else 0
    losses = len(trades_df[trades_df["pnl"] <= 0]) if not trades_df.empty else 0
    total_closed = len(trades_df)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    avg_hold_hours = (trades_df["bars"].mean() * 5.0 / 60.0) if not trades_df.empty else 0.0
    max_hold_days  = (trades_df["bars"].max() * 5.0 / (60.0 * 23.0)) if not trades_df.empty else 0.0

    eq_series = pd.Series(equity_curve)
    rolling_max = eq_series.cummax()
    drawdown = eq_series - rolling_max
    max_dd = drawdown.min()

    return {
        "mode": mode,
        "wait_count": wait_count,
        "executed_count": executed_count,
        "total_closed": total_closed,
        "still_open": len(open_positions),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_hold_hours": avg_hold_hours,
        "max_hold_days": max_hold_days,
        "closed_balance": balance,
        "net_closed_profit": balance - INITIAL_BALANCE,
        "floating_pnl": final_floating_pnl,
        "final_equity": final_equity,
        "max_dd": max_dd,
        "max_concurrent": max_concurrent_positions
    }

def main():
    raw_df = load_all_m5_bars_from_supabase()

    # Build 4H features across all 40,148+ rows from m5_bars
    df_feat = build_features(raw_df, TIMEFRAME).dropna().reset_index(drop=True)
    total_bars = len(df_feat)

    # 23 hours/day = 276 five-minute bars per trading day
    trading_days = total_bars / 276.0
    approx_months = trading_days / 21.6

    start_ts = df_feat["timestamp"].iloc[0]
    end_ts   = df_feat["timestamp"].iloc[-1]

    print("\n" + "="*70)
    print(f"📊 SUPABASE 'm5_bars' FULL BACKTEST COVERAGE (HOLD UNTIL EQ TARGET)")
    print("="*70)
    print(f"Total Clean Rows Loaded : {len(raw_df):,}")
    print(f"Warmup Rows Dropped     : {len(raw_df) - total_bars:,}")
    print(f"Evaluated 5-Min Bars    : {total_bars:,}")
    print(f"Trading Days Covered    : {trading_days:.2f} Trading Days (~{approx_months:.1f} Months @ 23h/day)")
    print(f"Date Range Covered      : {start_ts}  -->  {end_ts}")
    print("="*70)

    # Load 4H Model & Predict Across All Bars
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

    signals = np.where(prices <= buy_trigger, "BUY",
              np.where(prices >= sell_trigger, "SELL", "WAIT"))

    raw_wait = int(np.sum(signals == "WAIT"))
    raw_exec = int(np.sum(signals != "WAIT"))
    print(f"\nRAW SIGNAL COUNT (Every 5m bar): {raw_wait:,} WAIT | {raw_exec:,} BUY/SELL TRIGGERS")

    # Run Both Position Modes (Hold Until EQ Target)
    res_single = run_simulation(df_feat, signals, midline_eq, mode="SINGLE_POSITION")
    res_multi  = run_simulation(df_feat, signals, midline_eq, mode="MULTI_POSITION")

    print("\n" + "="*70)
    print("📈 BACKTEST RESULTS ($1,000 Start | 1:2000 Lev | 0.02 Lot = $0.02/pt)")
    print("   Exit Rule: Positions NEVER expire by time; held until EQ TP is hit")
    print("="*70)
    for res in [res_single, res_multi]:
        print(f"\n>>> MODE: {res['mode']} <<<")
        print(f"  Bars Waited (No Entry)    : {res['wait_count']:,}")
        print(f"  Trades Opened (Executed)  : {res['executed_count']:,}")
        print(f"  Trades Closed @ EQ Target : {res['total_closed']:,} (Wins: {res['wins']:,} | Losses: {res['losses']:,})")
        print(f"  Trades Still Open at End  : {res['still_open']:,} (Floating PnL: ${res['floating_pnl']:+,.2f})")
        print(f"  Avg Hold Time to Hit EQ   : {res['avg_hold_hours']:.1f} Hours (Max Hold: {res['max_hold_days']:.1f} Days)")
        print(f"  Max Open Positions at Once: {res['max_concurrent']}")
        print(f"  Closed Trade Win Rate     : {res['win_rate']:.1f}%")
        print(f"  Realized Closed Balance   : ${res['closed_balance']:,.2f} (Net Closed: ${res['net_closed_profit']:+,.2f})")
        print(f"  Final Account Equity      : ${res['final_equity']:,.2f} (Includes Open Floating Trades)")
        print(f"  Max Equity Drawdown       : ${res['max_dd']:,.2f}")

if __name__ == "__main__":
    main()
