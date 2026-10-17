import os
import sys
import psycopg
from psycopg.rows import dict_row

# ============================================================
# US30 QUANTUM SUPREME — LIVE GRID VALIDATOR
# Acts as "Cell 7": Upgrades PENDING -> OPEN -> WIN_TP
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is missing.")
    sys.exit(1)

try:
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    conn.autocommit = True
    cur = conn.cursor()
except Exception as e:
    print(f"Database connection failed: {e}")
    sys.exit(1)

print("Fetching active US30 trades and latest market data...")

# 1. Get the single latest 5-minute candle from m5_bars
cur.execute("SELECT high, low, close FROM m5_bars ORDER BY timestamp DESC LIMIT 1;")
latest_candle = cur.fetchone()

if not latest_candle:
    print("No market data found in m5_bars. Exiting.")
    sys.exit(0)

high = float(latest_candle['high'])
low = float(latest_candle['low'])
close = float(latest_candle['close'])

# 2. Get all trades that are PENDING or OPEN
cur.execute("SELECT * FROM supreme_grid_log WHERE status IN ('PENDING', 'OPEN') ORDER BY timestamp ASC;")
active_trades = cur.fetchall()

if not active_trades:
    print("No active grid positions to manage. All clear.")
    sys.exit(0)

updated_count = 0

# 3. State Machine Loop
for trade in active_trades:
    trade_id = trade['id']
    sig = trade['signal']
    status = trade['status']
    if trade['entry'] is None or trade['take_profit'] is None:
        continue
    entry = float(trade['entry'])
    tp = float(trade['take_profit'])
    
    new_status = status

    if sig == "BUY":
        # Check if resting Limit Order was triggered
        if status == "PENDING" and low <= entry:
            new_status = "OPEN"
            print(f"[{trade_id}] BUY Limit Triggered! PENDING -> OPEN")
            
        # Check if floating trade hit Take Profit
        if new_status == "OPEN" and high >= tp:
            new_status = "WIN_TP"
            print(f"[{trade_id}] BUY Target Hit! OPEN -> WIN_TP")

    elif sig == "SELL":
        # Check if resting Limit Order was triggered
        if status == "PENDING" and high >= entry:
            new_status = "OPEN"
            print(f"[{trade_id}] SELL Limit Trigr DC
