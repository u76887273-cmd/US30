import os
import json
import datetime
import joblib
import psycopg
import pandas as pd
from train import build_features, train_horizon, FEATURES, HORIZONS

DB_URL = os.environ.get("DATABASE_URL")
RAW_BARS_TABLE = "m5_bars"

TABLES = {
    "30m": "supreme_grid_log",
    "4h":  "supreme_grid_log_4h"
}

def parse_mixed_timestamps(series):
    try:
        return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")

def get_live_spot_quote():
    """Reads latest_quote.json produced by data.py if a fresh cTrader spot quote was received."""
    if os.path.exists("latest_quote.json"):
        try:
            with open("latest_quote.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "QUOTE_RECEIVED":
                price = data.get("current_price")
                if price is not None and float(price) > 1000:
                    return float(price)
        except Exception as e:
            print(f"Note reading latest_quote.json: {e}")
    return None

def sync_recent_bars_and_load_warmup(conn, live_spot):
    local_df = pd.DataFrame()
    if os.path.exists("data_m5.csv"):
        try:
            local_df = pd.read_csv("data_m5.csv")
            local_df.columns = [c.lower() for c in local_df.columns]
            if "timestamp" in local_df.columns and "close" in local_df.columns:
                local_df["timestamp"] = parse_mixed_timestamps(local_df["timestamp"])
                for col in ["open", "high", "low", "close"]:
                    local_df[col] = pd.to_numeric(local_df[col], errors="coerce")
                local_df["volume"] = pd.to_numeric(local_df.get("volume", 0.0), errors="coerce").fillna(0.0)

                local_df = local_df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")

                if live_spot is not None and not local_df.empty:
                    idx = local_df.index[-1]
                    local_df.loc[idx, "close"] = live_spot
                    if live_spot > float(local_df.loc[idx, "high"]):
                        local_df.loc[idx, "high"] = live_spot
                    if live_spot < float(local_df.loc[idx, "low"]):
                        local_df.loc[idx, "low"] = live_spot

                recent_slice = local_df.tail(500)
                with conn.cursor() as cur:
                    # Delete overlapping recent timestamps first so updated closes always overwrite stale rows
                    min_recent_ts = recent_slice["timestamp"].iloc[0].isoformat()
                    cur.execute(f"DELETE FROM {RAW_BARS_TABLE} WHERE timestamp >= %s;", (min_recent_ts,))
                    insert_data = [
                        (
                            row["timestamp"].isoformat(),
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                            float(row["volume"])
                        )
                        for _, row in recent_slice.iterrows()
                    ]
                    cur.executemany(
                        f"""
                        INSERT INTO {RAW_BARS_TABLE} (timestamp, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                        """,
                        insert_data
                    )
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Note on live bar sync: {e}")

    db_df = pd.DataFrame()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {RAW_BARS_TABLE}
                ORDER BY timestamp DESC
                LIMIT 1500;
            """)
            rows = cur.fetchall()
            col_names = [desc[0].lower() for desc in cur.description]

        db_df = pd.DataFrame(rows, columns=col_names)
        if not db_df.empty:
            db_df["timestamp"] = parse_mixed_timestamps(db_df["timestamp"])
            for col in ["open", "high", "low", "close"]:
                db_df[col] = pd.to_numeric(db_df[col], errors="coerce")
            db_df["volume"] = pd.to_numeric(db_df["volume"], errors="coerce").fillna(0.0)
    except Exception as e:
        print(f"Note on DB warmup fetch: {e}")

    # Place local_df second so fresh cTrader bars always take priority with keep="last"
    combined = pd.concat([db_df, local_df], ignore_index=True)
    combined = (
        combined.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values(by="timestamp")
        .tail(1500)
        .reset_index(drop=True)
    )

    if live_spot is not None and not combined.empty:
        idx = combined.index[-1]
        combined.loc[idx, "close"] = live_spot
        if live_spot > float(combined.loc[idx, "high"]):
            combined.loc[idx, "high"] = live_spot
        if live_spot < float(combined.loc[idx, "low"]):
            combined.loc[idx, "low"] = live_spot

    return combined

def init_db_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                price NUMERIC(10, 2) NOT NULL,
                signal VARCHAR(16) NOT NULL,
                reason TEXT,
                entry_zone VARCHAR(128),
                entry NUMERIC(10, 2),
                take_profit NUMERIC(10, 2),
                status VARCHAR(32) DEFAULT 'ACTIVE',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            ALTER TABLE {table_name} ALTER COLUMN entry_zone TYPE VARCHAR(128);
            GRANT SELECT ON {table_name} TO anon, authenticated;
        """)
    conn.commit()

def predict_for_timeframe(conn, raw_df, tf, live_spot):
    model_path = f"model_{tf}.pkl"
    table_name = TABLES[tf]

    if not os.path.exists(model_path):
        print(f"[{tf.upper()}] Model file {model_path} missing. Training now...")
        train_horizon(raw_df, tf, HORIZONS[tf])

    models = joblib.load(model_path)
    expected_features = len(FEATURES)
    actual_features = getattr(models["up"], "n_features_in_", expected_features)
    if actual_features != expected_features:
        print(f"[{tf.upper()}] Cached model has {actual_features} features (expected {expected_features}). Retraining...")
        train_horizon(raw_df, tf, HORIZONS[tf])
        models = joblib.load(model_path)

    df_feat = build_features(raw_df, tf).dropna(subset=FEATURES).reset_index(drop=True)
    if df_feat.empty:
        print(f"[{tf.upper()}] Not enough warmup bars to compute features (rows={len(raw_df)}).")
        return

    latest_row = df_feat.iloc[[-1]]
    X_latest = latest_row[FEATURES]

    # Use live cTrader spot price if available, otherwise latest M5 bar close
    price = float(live_spot) if live_spot is not None else float(latest_row["close"].iloc[0])
    rsi = float(latest_row["rsi"].iloc[0])
    anchor_ema = float(latest_row["ema_fast"].iloc[0])

    pred_up = float(models["up"].predict(X_latest)[0])
    pred_down = float(models["down"].predict(X_latest)[0])

    proj_high = anchor_ema + pred_up
    proj_low = anchor_ema - pred_down
    total_range = max(proj_high - proj_low, 10.0)

    midline_eq = (proj_high + proj_low) / 2.0
    half_range = total_range / 2.0

    # 1. 55% Distance Threshold (0.55) applied to BOTH 30m and 4h
    sell_trigger = midline_eq + (half_range * 0.55)
    buy_trigger = midline_eq - (half_range * 0.55)

    if price <= buy_trigger:
        signal = "BUY"
    elif price >= sell_trigger:
        signal = "SELL"
    else:
        signal = "WAIT"

    # 2. Three Buy Tiers & Three Sell Tiers (0.55, 0.80, 1.00 of half_range from EQ)
    b_055 = round(midline_eq - (half_range * 0.55), 0)
    b_080 = round(midline_eq - (half_range * 0.80), 0)
    b_100 = round(midline_eq - (half_range * 1.00), 0)  # Equals proj_low

    s_055 = round(midline_eq + (half_range * 0.55), 0)
    s_080 = round(midline_eq + (half_range * 0.80), 0)
    s_100 = round(midline_eq + (half_range * 1.00), 0)  # Equals proj_high

    price_0d = round(price, 0)
    eq_0d = round(midline_eq, 0)

    entry_zone = (
        f"B0.55:{b_055:.0f},B0.80:{b_080:.0f},B1.00:{b_100:.0f} | "
        f"S0.55:{s_055:.0f},S0.80:{s_080:.0f},S1.00:{s_100:.0f}"
    )

    up_ratio = (pred_up / total_range) * 100.0
    down_ratio = (pred_down / total_range) * 100.0
    bias_pct = up_ratio if price < midline_eq else down_ratio

    reason = f"Bias:{bias_pct:.1f}% | RSI:{rsi:.1f} | EQ:{eq_0d:.0f}"
    run_ts = datetime.datetime.now(datetime.timezone.utc)

    init_db_table(conn, table_name)

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table_name} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (run_ts, price_0d, signal, reason, entry_zone, price_0d, eq_0d))
    conn.commit()
    print(
        f"[{tf.upper()}] {signal} @ {price_0d:.0f} (Spot: {price:.2f}) | AnchorEMA:{anchor_ema:.0f} | "
        f"Buy(0.55/0.8/1.0): {b_055:.0f}/{b_080:.0f}/{b_100:.0f} | "
        f"Sell(0.55/0.8/1.0): {s_055:.0f}/{s_080:.0f}/{s_100:.0f} | EQ/TP: {eq_0d:.0f}"
    )

if __name__ == "__main__":
    live_spot = get_live_spot_quote()
    with psycopg.connect(DB_URL) as conn:
        raw_df = sync_recent_bars_and_load_warmup(conn, live_spot)
        for tf in ["30m", "4h"]:
            predict_for_timeframe(conn, raw_df, tf, live_spot)
