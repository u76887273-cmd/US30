import os
import json
import psycopg
import pandas as pd

DB_URL = os.environ.get("DATABASE_URL")
TABLES = ["supreme_grid_log", "supreme_grid_log_4h"]

def get_current_high_low():
    high_p, low_p = None, None
    if os.path.exists("data_m5.csv"):
        try:
            df = pd.read_csv("data_m5.csv")
            if not df.empty:
                high_p = float(df["high"].iloc[-1])
                low_p = float(df["low"].iloc[-1])
        except Exception:
            pass

    if os.path.exists("latest_quote.json"):
        try:
            with open("latest_quote.json", "r", encoding="utf-8") as f:
                spot = float(json.load(f).get("current_price") or 0)
            if spot > 1000:
                high_p = max(high_p, spot) if high_p else spot
                low_p = min(low_p, spot) if low_p else spot
        except Exception:
            pass

    return high_p, low_p

def validate_tables():
    if not DB_URL:
        return
    high_p, low_p = get_current_high_low()
    if high_p is None or low_p is None:
        return

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for table in TABLES:
                try:
                    cur.execute(f"""
                        UPDATE {table}
                        SET status = 'TP_HIT'
                        WHERE status = 'ACTIVE'
                          AND (
                              (signal = 'BUY' AND take_profit <= %s)
                              OR
                              (signal = 'SELL' AND take_profit >= %s)
                          );
                    """, (high_p, low_p))
                except Exception as e:
                    conn.rollback()
                    print(f"Validator note on {table}: {e}")
                else:
                    conn.commit()
    print(f"✅ Grid State Validator completed (Checked against High={high_p:.2f}, Low={low_p:.2f}).")

if __name__ == "__main__":
    validate_tables()
