import os
import datetime
import joblib
import psycopg
import yfinance as yf
import pandas as pd
from train import build_features, FEATURES

DB_URL = os.environ.get("DATABASE_URL")

TABLES = {
    "30m": "supreme_grid_log",
    "4h":  "supreme_grid_log_4h"
}

def get_nasdaq_direction():
    """Fetches live Nasdaq 100 (NQ=F) 15m trend for US30 macro check."""
    for ticker in ["NQ=F", "QQQ"]:
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            if not data.empty and len(data) >= 2:
                close_col = data["Close"]
                last_close = float(close_col.iloc[-1].iloc[0]) if hasattr(close_col.iloc[-1], "iloc") else float(close_col.iloc[-1])
                prev_close = float(close_col.iloc[-2].iloc[0]) if hasattr(close_col.iloc[-2], "iloc") else float(close_col.iloc[-2])
                change_pct = ((last_close - prev_close) / prev_close) * 100
                direction = "UP" if last_close >= prev_close else "DOWN"
                return f"NAS:{direction} ({last_close:.1f} | {change_pct:+.2f}%)"
        except Exception as e:
            print(f"Ticker {ticker} check failed: {e}")
    return "NAS:FLAT"

def init_db_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                price NUMERIC(10, 2) NOT NULL,
                signal VARCHAR(16) NOT NULL,
                reason TEXT,
                entry_zone VARCHAR(64),
                entry NUMERIC(10, 2),
                take_profit NUMERIC(10, 2),
                status VARCHAR(32) DEFAULT 'ACTIVE',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            GRANT SELECT ON {table_name} TO anon, authenticated;
        """)
    conn.commit()

def predict_for_timeframe(conn, raw_df, tf, nas_tag):
    model_path = f"model_{tf}.pkl"
    table_name = TABLES[tf]

    if not os.path.exists(model_path):
        print(f"[{tf.upper()}] Model file {model_path} missing.")
        return

    # Build timeframe-specific features so 30m and 4h each have their own true RSI & EMAs
    df_feat = build_features(raw_df, tf).dropna().reset_index(drop=True)
    models = joblib.load(model_path)
    latest_row = df_feat.iloc[[-1]]
    X_latest = latest_row[FEATURES]

    price = float(latest_row["close"].iloc[0])
    rsi = float(latest_row["rsi"].iloc[0])

    # Predict Upside and Downside range in price units
    pred_up = float(models["up"].predict(X_latest)[0])
    pred_down = float(models["down"].predict(X_latest)[0])

    proj_high = price + pred_up
    proj_low = price - pred_down
    total_range = max(proj_high - proj_low, 10.0)

    # Equilibrium (Midline of AI Projected Range)
    midline_eq = (proj_high + proj_low) / 2.0

    # Pure Midline Rule
    if price < midline_eq:
        signal = "BUY"
    elif price > midline_eq:
        signal = "SELL"
    else:
        signal = "WAIT"

    # Dynamic 20% Entry Limit Zones sliced from Projected High/Low Range
    zone_height = max(total_range * 0.20, 5.0)
    buy_low = round(proj_low, 0)
    buy_high = round(proj_low + zone_height, 0)
    sell_low = round(proj_high - zone_height, 0)
    sell_high = round(proj_high, 0)

    price_0d = round(price, 0)
    eq_0d = round(midline_eq, 0)

    if signal == "BUY":
        entry_zone = f"B: {buy_low:.0f} - {buy_high:.0f} | S: WAITING"
    elif signal == "SELL":
        entry_zone = f"B: WAITING | S: {sell_low:.0f} - {sell_high:.0f}"
    else:
        entry_zone = f"B: {buy_low:.0f} - {buy_high:.0f} | S: {sell_low:.0f} - {sell_high:.0f}"

    up_ratio = (pred_up / total_range) * 100.0
    down_ratio = (pred_down / total_range) * 100.0
    bias_pct = up_ratio if signal == "BUY" else down_ratio

    reason = f"Bias:{bias_pct:.1f}% | RSI:{rsi:.1f} | EQ:{eq_0d:.0f} | {nas_tag}"
    run_ts = datetime.datetime.now(datetime.timezone.utc)

    init_db_table(conn, table_name)

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table_name} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (run_ts, price_0d, signal, reason, entry_zone, price_0d, eq_0d))
    conn.commit()
    print(f"[{tf.upper()}] {signal} @ {price_0d:.0f} | RSI: {rsi:.1f} | EQ/TP: {eq_0d:.0f}")

if __name__ == "__main__":
    if not os.path.exists("data_m5.csv"):
        print("data_m5.csv missing.")
    else:
        nas_status = get_nasdaq_direction()
        print(f"Live Nasdaq Status: {nas_status}")
        raw_df = pd.read_csv("data_m5.csv")
        with psycopg.connect(DB_URL) as conn:
            for tf in ["30m", "4h"]:
                predict_for_timeframe(conn, raw_df, tf, nas_status)
