import os
import json
import datetime
import psycopg
import pandas as pd

DB_URL = os.environ.get("DATABASE_URL")
CSV_PATH = "data_m5.csv"

TABLES = {
    "30m": "supreme_grid_log",
    "4h":  "supreme_grid_log_4h"
}

def parse_mixed_timestamps(series):
    try:
        return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")

def get_live_spot():
    if os.path.exists("latest_quote.json"):
        try:
            with open("latest_quote.json", "r", encoding="utf-8") as f:
                q = json.load(f)
            if q.get("status") == "QUOTE_RECEIVED":
                spot = float(q.get("current_price") or 0)
                if spot > 1000:
                    return spot
        except Exception:
            pass
    return None

def load_recent_m5_bars():
    if not os.path.exists(CSV_PATH):
        print(f"⚠️ {CSV_PATH} not found. Skipping validation.")
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return pd.DataFrame()

    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = parse_mixed_timestamps(df["timestamp"])
    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "high", "low"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def validate_table(conn, tf, table_name, df, live_spot):
    if df.empty:
        return

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    latest_bar_ts = df["timestamp"].iloc[-1]

    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {table_name}
            SET status = 'LOGGED'
            WHERE status = 'ACTIVE' AND signal = 'WAIT';
        """)

        cur.execute(f"""
            SELECT id, timestamp, signal, take_profit
            FROM {table_name}
            WHERE status IN ('ACTIVE', 'PENDING', 'OPEN') AND signal IN ('BUY', 'SELL');
        """)
        active_rows = cur.fetchall()

        hit_ids = []
        for row_id, sig_ts, signal, tp in active_rows:
            if tp is None:
                continue
            tp_val = float(tp)
            sig_ts_utc = pd.to_datetime(sig_ts, utc=True)

            # Never validate a brand-new signal generated in the current run (< 2 minutes old)
            if (now_utc - sig_ts_utc).total_seconds() < 120:
                continue

            # Only check M5 bars that opened strictly AFTER the signal was created
            bars_after_signal = df[df["timestamp"] > sig_ts_utc]

            high_candidates = []
            low_candidates = []

            if not bars_after_signal.empty:
                high_candidates.append(float(bars_after_signal["high"].max()))
                low_candidates.append(float(bars_after_signal["low"].min()))

            if live_spot is not None:
                high_candidates.append(float(live_spot))
                low_candidates.append(float(live_spot))

            if not high_candidates or not low_candidates:
                continue

            max_high_since = max(high_candidates)
            min_low_since = min(low_candidates)

            if signal == "BUY" and max_high_since >= tp_val:
                hit_ids.append(row_id)
            elif signal == "SELL" and min_low_since <= tp_val:
                hit_ids.append(row_id)

        if hit_ids:
            cur.execute(
                f"UPDATE {table_name} SET status = 'HIT_TP' WHERE id = ANY(%s)",
                (hit_ids,)
            )
            conn.commit()
            print(f"[{tf.upper()}] ✅ Marked {len(hit_ids)} active US30 trade(s) as HIT_TP (Checked up to {latest_bar_ts}).")
        else:
            conn.commit()
            print(f"[{tf.upper()}] Checked {len(active_rows)} active US30 BUY/SELL signal(s); none hit TP yet.")

if __name__ == "__main__":
    if not DB_URL:
        print("⚠️ DATABASE_URL not set. Skipping validation.")
    else:
        m5_df = load_recent_m5_bars()
        live_spot = get_live_spot()
        if not m5_df.empty:
            with psycopg.connect(DB_URL) as conn:
                for tf, table_name in TABLES.items():
                    validate_table(conn, tf, table_name, m5_df, live_spot)
