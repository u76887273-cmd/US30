import os
import psycopg
import pandas as pd

DB_URL = os.environ.get("DATABASE_URL")

TABLES = {
    "30m": ("eurusd_grid_log_30m", "data_30m.csv"),
    "4h":  ("eurusd_grid_log_4h",  "data_4h.csv")
}

def validate_table(conn, tf, table_name, csv_path):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    last_bar = df.iloc[-1]
    current_high = float(last_bar["high"])
    current_low = float(last_bar["low"])

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, signal, take_profit
            FROM {table_name}
            WHERE status = 'ACTIVE'
        """)
        active_rows = cur.fetchall()

        hit_ids = []
        for row_id, signal, tp in active_rows:
            tp_val = float(tp)
            if signal == "BUY" and current_high >= tp_val:
                hit_ids.append(row_id)
            elif signal == "SELL" and current_low <= tp_val:
                hit_ids.append(row_id)

        if hit_ids:
            cur.execute(
                f"UPDATE {table_name} SET status = 'HIT_TP' WHERE id = ANY(%s)",
                (hit_ids,)
            )
            conn.commit()
            print(f"[{tf.upper()}] Marked {len(hit_ids)} trades as HIT_TP.")
        else:
            print(f"[{tf.upper()}] Checked {len(active_rows)} active signals; none hit TP yet.")

if __name__ == "__main__":
    with psycopg.connect(DB_URL) as conn:
        for tf, (table_name, csv_path) in TABLES.items():
            validate_table(conn, tf, table_name, csv_path)
