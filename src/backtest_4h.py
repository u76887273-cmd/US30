import os
import psycopg
import pandas as pd

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

    # 1. Load all 4H signals from the database
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, timestamp, price, signal, entry, take_profit, status
                FROM supreme_grid_log_4h
                ORDER BY timestamp ASC;
            """)
            rows = cur.fetchall()

    if not rows:
        print("No records found in supreme_grid_log_4h.")
        return

    signals_df = pd.DataFrame(
        rows,
        columns=["id", "timestamp", "price", "signal", "entry", "take_profit", "status"]
    )
    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"], utc=True)

    # 2. Load M5 price bars if available to verify exact 4H outcomes and MTM drawdown
    has_bars = os.path.exists("data_m5.csv")
    bars_df = None
    if has_bars:
        bars_df = pd.read_csv("data_m5.csv")
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
        bars_df = bars_df.sort_values("timestamp").reset_index(drop=True)

    wins = 0
    losses = 0
    open_or_pending = 0
    total_points = 0.0
    win_points = 0.0
    loss_points = 0.0

    for _, row in signals_df.iterrows():
        sig = row["signal"]
        if sig not in ("BUY", "SELL") or row["take_profit"] is None:
            continue

        entry_price = float(row["entry"] if row["entry"] is not None else row["price"])
        tp_price = float(row["take_profit"])
        sig_time = row["timestamp"]
        db_status = str(row["status"]).upper()

        # Check forward 4-hour price action in data_m5.csv if covered
        future_bars = pd.DataFrame()
        if bars_df is not None:
            future_bars = bars_df[bars_df["timestamp"] > sig_time].head(HORIZON_BARS)

        if not future_bars.empty:
            max_high = float(future_bars["high"].max())
            min_low = float(future_bars["low"].min())
            end_close = float(future_bars["close"].iloc[-1])

            if sig == "BUY":
                if max_high >= tp_price or db_status in ("HIT_TP", "WIN_TP"):
                    pts = abs(tp_price - entry_price)
                    wins += 1
                    win_points += pts
                    total_points += pts
                else:
                    # Closed at end of 4H horizon
                    pts = end_close - entry_price
                    if pts >= 0:
                        wins += 1
                        win_points += pts
                    else:
                        losses += 1
                        loss_points += abs(pts)
                    total_points += pts

            elif sig == "SELL":
                if min_low <= tp_price or db_status in ("HIT_TP", "WIN_TP"):
                    pts = abs(entry_price - tp_price)
                    wins += 1
                    win_points += pts
                    total_points += pts
                else:
                    # Closed at end of 4H horizon
                    pts = entry_price - end_close
                    if pts >= 0:
                        wins += 1
                        win_points += pts
                    else:
                        losses += 1
                        loss_points += abs(pts)
                    total_points += pts
        else:
            # Fallback to database status if signal is older than data_m5.csv window
            if db_status in ("HIT_TP", "WIN_TP"):
                pts = abs(tp_price - entry_price)
                wins += 1
                win_points += pts
                total_points += pts
            elif db_status in ("LOSS", "STOPPED", "SL"):
                pts = abs(tp_price - entry_price)
                losses += 1
                loss_points += pts
                total_points -= pts
            else:
                # Signal is still ACTIVE and waiting for future bars
                open_or_pending += 1

    total_closed = wins + losses
    win_rate = (wins / total_closed * 100.0) if total_closed > 0 else 0.0

    # Sample price for margin calculation
    latest_price = float(signals_df["price"].iloc[-1])

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
    print("       US30 4-HOUR (4H) DATABASE BACKTEST REPORT       ")
    print("=" * 55)
    print(f"Total 4H Signals in DB : {len(signals_df)}")
    print(f"Completed Trades       : {total_closed} (Still Active: {open_or_pending})")
    print(f"Winning Trades (Wins)  : {wins}")
    print(f"Losing Trades (Losses) : {losses}")
    print(f"Win Rate               : {win_rate:.2f}%")
    print("-" * 55)
    print(f"Gross Points Won       : +{win_points:.1f} pts")
    print(f"Gross Points Lost      : -{loss_points:.1f} pts")
    print(f"Net Points Gained      : {total_points:+.1f} pts")
    print("-" * 55)
    print(f"Starting Balance       : ${INITIAL_BALANCE:,.2f}")
    print(f"Entry Lot Size         : {LOT_SIZE} Lots (Leverage 1:{LEVERAGE})")
    print("-" * 55)
    print("[Broker Spec A: 1 Lot = $1/pt  -> 0.02 Lot = $0.02/pt]")
    print(f"  Margin Used / Trade  : ${margin_std:.2f}")
    print(f"  Net Gain / Loss      : ${net_profit_std:+,.2f} ({(net_profit_std/INITIAL_BALANCE)*100:+.2f}%)")
    print(f"  Final Account Balance: ${final_bal_std:,.2f}")
    print("-" * 55)
    print("[Broker Spec B: 1 Lot = $10/pt -> 0.02 Lot = $0.20/pt]")
    print(f"  Margin Used / Trade  : ${margin_idx:.2f}")
    print(f"  Net Gain / Loss      : ${net_profit_idx:+,.2f} ({(net_profit_idx/INITIAL_BALANCE)*100:+.2f}%)")
    print(f"  Final Account Balance: ${final_bal_idx:,.2f}")
    print("=" * 55)

if __name__ == "__main__":
    run_4h_backtest()
