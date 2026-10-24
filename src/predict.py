import os
import datetime
import joblib
import psycopg
import pandas as pd
from train import build_features, FEATURES

DB_URL = os.environ.get("DATABASE_URL")

TABLES = {
    "30m": "eurusd_grid_log_30m",
    "4h":  "eurusd_grid_log_4h"
}

def init_db_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                price NUMERIC(10, 4) NOT NULL,
                signal VARCHAR(16) NOT NULL,
                reason TEXT,
                entry_zone VARCHAR(64),
                entry NUMERIC(10, 4),
                take_profit NUMERIC(10, 4),
                status VARCHAR(32) DEFAULT 'ACTIVE',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            GRANT SELECT ON {table_name} TO anon, authenticated;
        """)
    conn.commit()

def predict_for_timeframe(conn, raw_df, tf):
    model_path = f"model_{tf}.pkl"
    table_name = TABLES[tf]

    if not os.path.exists(model_path):
        print(f"[{tf.upper()}] Model file {model_path} missing.")
        return

    df_feat = build_features(raw_df, tf).dropna().reset_index(drop=True)
    models = joblib.load(model_path)
    latest_row = df_feat.iloc[[-1]]
    X_latest = latest_row[FEATURES]

    price = float(latest_row["close"].iloc[0])
    rsi = float(latest_row["rsi"].iloc[0])

    pred_up = float(models["up"].predict(X_latest)[0])
    pred_down = float(models["down"].predict(X_latest)[0])

    proj_high = price + pred_up
    proj_low = price - pred_down
    total_range = max(proj_high - proj_low, 0.0004)

    # Equilibrium (Midline of AI Projected Range)
    midline_eq = (proj_high + proj_low) / 2.0

    if price < midline_eq:
        signal = "BUY"
    elif price > midline_eq:
        signal = "SELL"
    else:
        signal = "WAIT"

    # Dynamic 20% Entry Limit Zones
    zone_height = max(total_range * 0.20, 0.0002)
    buy_low = round(proj_low, 4)
    buy_high = round(proj_low + zone_height, 4)
    sell_low = round(proj_high - zone_height, 4)
    sell_high = round(proj_high, 4)

    price_4d = round(price, 4)
    eq_4d = round(midline_eq, 4)

    if signal == "BUY":
        entry_zone = f"B: {buy_low:.4f} - {buy_high:.4f} | S: WAITING"
    elif signal == "SELL":
        entry_zone = f"B: WAITING | S: {sell_low:.4f} - {sell_high:.4f}"
    else:
        entry_zone = f"B: {buy_low:.4f} - {buy_high:.4f} | S: {sell_low:.4f} - {sell_high:.4f}"

    up_ratio = (pred_up / total_range) * 100.0
    down_ratio = (pred_down / total_range) * 100.0
    bias_pct = up_ratio if signal == "BUY" else down_ratio

    reason = f"Bias:{bias_pct:.1f}% | RSI:{rsi:.1f} | EQ:{eq_4d:.4f}"
    run_ts = datetime.datetime.now(datetime.timezone.utc)

    init_db_table(conn, table_name)

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table_name} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (run_ts, price_4d, signal, reason, entry_zone, price_4d, eq_4d))
    conn.commit()
    print(f"[{tf.upper()}] {signal} @ {price_4d:.4f} | Bias: {bias_pct:.1f}% | RSI: {rsi:.1f} | EQ/TP: {eq_4d:.4f}")

if __name__ == "__main__":
    if not os.path.exists("data_m5.csv"):
        print("data_m5.csv missing.")
    else:
        raw_df = pd.read_csv("data_m5.csv")
        with psycopg.connect(DB_URL) as conn:
            for tf in ["30m", "4h"]:
                predict_for_timeframe(conn, raw_df, tf)
