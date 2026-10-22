import os
import joblib
import psycopg
import pandas as pd
from train import build_features, FEATURES

DB_URL = os.environ.get("DATABASE_URL")

# Backtest Parameters
INITIAL_BALANCE = 1000.0
LOT_SIZE = 0.02
LEVERAGE = 2000
HORIZON_BARS = 48  # 48 x 5-min bars = 4 Hours

def run_4h_backtest():
    if not DB_URL:
        print("Error: DATABASE_URL environment variable not set.")
        return

    if not os.path.exists("data_m5.csv") or not os.path.exists("model_4h.pkl"):
        print("Missing data_m5.csv or model_4h.pkl required for simulation.")
        return

    # 1. Load M5 data and build 4H features
    raw_df = pd.read_csv("data_m5.csv")
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)
    
    df_feat = build_features(raw_df, "4h").dropna().reset_index(drop=True)
    models = joblib.load("model_4h.pkl")

    wins = 0
    losses = 0
    skipped_wait = 0
    total_points = 0.0
    win_points = 0.0
    loss_points = 0.0

    # 2. Simulate rolling over historical bars to apply the 60% rule dynamically
    # Start after enough bars for feature history
    for i in range(100, len(df_feat) - HORIZON_BARS):
        hist_slice = df_feat.iloc[:i+1]
        latest_row = hist_slice.iloc[[-1]]
        X_latest = latest_row[FEATURES]

        price = float(latest_row["close"].iloc[0])
        pred_up = float(models["up"].predict(X_latest)[0])
        pred_down = float(models["down"].predict(X_latest)[0])

        proj_high = price + pred_up
        proj_low = price - pred_down
        total_range = max(proj_high - proj_low, 10.0)

        midline_eq = (proj_high + proj_low) / 2.0
        half_range = total_range / 2.0

        # Apply the new 60% Distance Threshold Rules
        sell_trigger = midline_eq + (half_range * 0.60)
        buy_trigger = midline_eq - (half_range * 0.60)

        if price <= buy_trigger:
            signal = "BUY"
        elif price >= sell_trigger:
            signal = "SELL"
        else:
            signal = "WAIT"

        if signal == "WAIT":
            skipped_wait += 1
            continue

        tp_price = midline_eq
        entry_price = price

        # Look at the forward 4H window (48 bars)
        future_bars = df_feat.iloc[i+1 : i+1+HORIZON_BARS]
        if future_bars.empty:
            continue

        max_high = float(future_bars["high"].max())
        min_low = float(future_bars["low"].min())
        end_close = float(future_bars["close"].iloc[-1])

        if signal == "BUY":
            if max_high >= tp_price:
                pts = abs(tp_price - entry_price)
                wins += 1
                win_points += pts
                total_points += pts
            else:
                pts = end_close - entry_price
                if pts >= 0:
                    wins += 1
                    win_points += pts
                else:
                    losses += 1
                    loss_points += abs(pts)
                total_points += pts

        elif signal == "SELL":
            if min_low <= tp_price:
                pts = abs(entry_price - tp_price)
                wins += 1
                win_points += pts
                total_points += pts
            else:
                pts = entry_price - end_close
                if pts >= 0:
                    wins += 1
                    win_points += pts
                else:
                    losses += 1
                    loss_points += abs(pts)
                total_points += pts

    total_closed = wins + losses
    win_rate = (wins / total_closed * 100.0) if total_closed > 0 else 0.0

    latest_price = float(df_feat["close"].iloc[-1])

    # Contract Spec A: 1 Lot = $1/pt -> 0.02 Lot = $0.02/pt
    val_per_pt_std = LOT_SIZE * 1.0
    net_profit_std = total_points * val_per_pt_std
    final_bal_std = INITIAL_BALANCE + net_profit_std
    margin_std = (latest_price * LOT_SIZE * 1.0) / LEVERAGE

    # Contract Spec B: 1 Lot = $10/pt -> 0.02 Lot = $0.20/pt
    val_per_pt_idx = LOT_SIZE * 10.0
    net_profit_idx = total_points * val_per_pt_idx
    final_bal_idx = INITIAL_BALANCE + net_profit_idx
    margin_idx = (latest_price * LOT_SIZE * 10.0) / LEVERAGE

    print("=" * 55)
    print("   US30 4H BACKTEST (WITH 60% THRESHOLD RULE APPLIED)   ")
    print("=" * 55)
    print(f"Skipped (WAIT signals) : {skipped_wait}")
    print(f"Executed Trades        : {total_closed}")
    print(f"Winning Trades (Wins)  : {wins}")
    print(f"Losing Trades (Losses) : {losses}")
    print(f"Win Rate               : {win_rate:.2f}%")
    print("-" * 55)
    print(f"Net Points Gained      : {total_points:+.1f} pts")
    print("-" * 55)
    print(f"Starting Balance       : ${INITIAL_BALANCE:,.2f}")
    print(f"Entry Lot Size         : {LOT_SIZE} Lots (Leverage 1:{LEVERAGE})")
    print("-" * 55)
    print("[Broker Spec A: 1 Lot = $1/pt  -> 0.02 Lot = $0.02/pt]")
    print(f"  Net Gain / Loss      : ${net_profit_std:+,.2f} ({(net_profit_std/INITIAL_BALANCE)*100:+.2f}%)")
    print(f"  Final Account Balance: ${final_bal_std:,.2f}")
    print("-" * 55)
    print("[Broker Spec B: 1 Lot = $10/pt -> 0.02 Lot = $0.20/pt]")
    print(f"  Net Gain / Loss      : ${net_profit_idx:+,.2f} ({(net_profit_idx/INITIAL_BALANCE)*100:+.2f}%)")
    print(f"  Final Account Balance: ${final_bal_idx:,.2f}")
    print("=" * 55)

if __name__ == "__main__":
    run_4h_backtest()
