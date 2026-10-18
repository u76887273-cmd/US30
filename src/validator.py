import os
import sys
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    sys.exit(1)

conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
cur = conn.cursor()

cur.execute("SELECT high, low, close FROM m5_bars ORDER BY timestamp DESC LIMIT 1;")
candle = cur.fetchone()
if not candle:
    sys.exit(0)

high = float(candle["high"])
low = float(candle["low"])

cur.execute("SELECT * FROM supreme_grid_log WHERE status IN ('PENDING', 'OPEN') ORDER BY timestamp ASC;")
trades = cur.fetchall()

updated = 0
for t in trades:
    if t["entry"] is None or t["take_profit"] is None:
        continue
    tid = t["id"]
    sig = t["signal"]
    status = t["status"]
    entry = float(t["entry"])
    tp = float(t["take_profit"])
    new_status = status

    if sig == "BUY":
        if status == "PENDING" and low <= entry:
            new_status = "OPEN"
        if new_status == "OPEN" and high >= tp:
            new_status = "WIN_TP"
    elif sig == "SELL":
        if status == "PENDING" and high >= entry:
            new_status = "OPEN"
        if new_status == "OPEN" and low <= tp:
            new_status = "WIN_TP"

    if new_status != status:
        cur.execute("UPDATE supreme_grid_log SET status = %s WHERE id = %s", (new_status, tid))
        updated += 1

cur.close()
conn.close()
print(f"US30 Validator complete. Updated: {updated}")
