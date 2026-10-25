import os
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

def validate_table(conn, tf, table_name, df):
    if df.empty:
        return

    latest_bar = df.iloc[-1]
    latest_ts = latest_bar["timestamp"]
    latest_high = float(latest_bar["high"])
    latest_low = float(latest_bar["low"])

    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {table_name}
            SET status = 'LOGGED'
            WHERE status = 'ACTIVE' AND signal = 'WAIT';
        """)

        cur.execute(f"""
            SELECT id, timestamp, signal, take_profit
            FROM {table_name}
            WHERE status = 'ACTIVE' AND signal IN ('BUY', 'SELL');
        """)
        active_rows = cur.fetchall()

        hit_ids = []
        for row_id, sig_ts, signal, tp in active_rows:
            if tp is None:
                continue
            tp_val = float(tp)
            sig_ts_utc = pd.to_datetime(sig_ts, utc=True)

            bars_since_signal = df[df["timestamp"] >= (sig_ts_utc - pd.Timedelta(minutes=5))]

            if not bars_since_signal.empty:
                max_high_since = float(bars_since_signal["high"].max())
                min_low_since = float(bars_since_signal["low"].min())
            else:
                max_high_since = latest_high
                min_low_since = latest_low

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
            print(f"[{tf.upper()}] ✅ Marked {len(hit_ids)} active trade(s) as HIT_TP (Checked up to {latest_ts}).")
        else:
            conn.commit()
            print(f"[{tf.upper()}] Checked {len(active_rows)} active BUY/SELL signal(s); none hit TP yet.")

if __name__ == "__main__":
    if not DB_URL:
        print("⚠️ DATABASE_URL not set. Skipping validation.")
    else:
        m5_df = load_recent_m5_bars()
        if not m5_df.empty:
            with psycopg.connect(DB_URL) as conn:
                for tf, table_name in TABLES.items():
                    validate_table(conn, tf, table_name, m5_df)
