import os
import datetime
import joblib
import psycopg
import pandas as pd
from train import build_features, FEATURES

DB_URL = os.environ.get("DATABASE_URL")

TABLES = {
    "30m": "supreme_grid_log",
    "4h":  "supreme_grid_log_4h"
}

THRESHOLD = 0.55  # 0.55 of half-width from midline

def init_db_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                price NUMERIC(12, 2) NOT NULL,
                signal VARCHAR(16) NOT NULL,
                reason TEXT,
                entry_zone VARCHAR(64),
                entry NUMERIC(12, 2),
                take_profit NUMERIC(12, 2),
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
    total_range = max(proj_high - proj_low, 10.0)
    half_width = total_range / 2.0

    # Equilibrium (Midline of AI Projected Range)
    midline_eq = (proj_high + proj_low) / 2.0

    # Distance from Midline measured as a fraction of Half-Width (0.0 at Midline to 1.0 at extreme)
    down_from_midline_ratio = (midline_eq - price) / half_width
    up_from_midline_ratio = (price - midline_eq) / half_width

    # BUY when price is down >= 0.55 of half the total width below Midline
    if price <= (midline_eq - THRESHOLD * half_width):
        signal = "BUY"
        bias_pct = down_from_midline_ratio * 100.0
    # SELL when price is up >= 0.55 of half the total width above Midline
    elif price >= (midline_eq + THRESHOLD * half_width):
        signal = "SELL"
        bias_pct = up_from_midline_ratio * 100.0
    else:
        signal = "WAIT"
        bias_pct = max(down_from_midline_ratio, up_from_midline_ratio, 0.0) * 100.0

    # Dynamic 20% Entry Limit Zones (Integer rounded for US30 Index points)
    zone_height = max(total_range * 0.20, 5.0)
    buy_low = int(round(proj_low))
    buy_high = int(round(proj_low + zone_height))
    sell_low = int(round(proj_high - zone_height))
    sell_high = int(round(proj_high))

    price_int = int(round(price))
    eq_int = int(round(midline_eq))

    if signal == "BUY":
        entry_zone = f"B: {buy_low} - {buy_high} | S: WAITING"
    elif signal == "SELL":
        entry_zone = f"B: WAITING | S: {sell_low} - {sell_high}"
    else:
        entry_zone = f"B: {buy_low} - {buy_high} | S: {sell_low} - {sell_high}"

    reason = f"Bias:{bias_pct:.1f}% | RSI:{rsi:.1f} | EQ:{eq_int}"
    run_ts = datetime.datetime.now(datetime.timezone.utc)

    init_db_table(conn, table_name)

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table_name} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (run_ts, price_int, signal, reason, entry_zone, price_int, eq_int))
    conn.commit()
    print(f"[{tf.upper()}] {signal} @ {price_int} | Bias: {bias_pct:.1f}% | RSI: {rsi:.1f} | EQ/TP: {eq_int}")

if __name__ == "__main__":
    if not os.path.exists("data_m5.csv"):
        print("data_m5.csv missing.")
    else:
        raw_df = pd.read_csv("data_m5.csv")
        with psycopg.connect(DB_URL) as conn:
            for tf in ["30m", "4h"]:
                predict_for_timeframe(conn, raw_df, tf)
