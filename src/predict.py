import os
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
    """Safely parses all ISO8601 and mixed timestamps without dropping rows."""
    try:
        return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")

def sync_recent_bars_and_load_warmup(conn):
    """
    1. Syncs any new 5m bars fetched by data.py into Supabase 'm5_bars' every 15 mins.
    2. Combines local data_m5.csv and the latest 1,500 bars from 'm5_bars' so the 4H indicators
       always have enough warmup bars even on runs where train.py is skipped.
    """
    local_df = pd.DataFrame()
    if os.path.exists("data_m5.csv"):
        try:
            local_df = pd.read_csv("data_m5.csv")
            local_df.columns = [c.lower() for c in local_df.columns]
            if "timestamp" in local_df.columns and "close" in local_df.columns:
                local_df["timestamp"] = parse_mixed_timestamps(local_df["timestamp"])
                for col in ["open", "high", "low", "close"]:
                    local_df[col] = pd.to_numeric(local_df[col], errors="coerce")
                if "volume" in local_df.columns:
                    local_df["volume"] = pd.to_numeric(local_df["volume"], errors="coerce").fillna(0.0)
                else:
                    local_df["volume"] = 0.0

                local_df = local_df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")

                # Sync up to the newest 500 bars from data_m5.csv into Supabase m5_bars
                recent_slice = local_df.tail(500)
                with conn.cursor() as cur:
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
            print(f"Note on live bar sync: {e}")

    # Pull the latest 1,500 bars from Supabase m5_bars
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

    # Combine DB and local bars so we always have maximum history for 4H warmup
    combined = pd.concat([db_df, local_df], ignore_index=True)
    combined = (
        combined.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values(by="timestamp")
        .tail(1500)
        .reset_index(drop=True)
    )
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
                entry_zone VARCHAR(64),
                entry NUMERIC(10, 2),
                take_profit NUMERIC(10, 2),
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
        print(f"[{tf.upper()}] Model file {model_path} missing. Training now...")
        train_horizon(raw_df, tf, HORIZONS[tf])

    models = joblib.load(model_path)

    # Self-healing check: if cached model was trained on old 6 features, retrain on new 9 features
    expected_features = len(FEATURES)
    actual_features = getattr(models["up"], "n_features_in_", expected_features)
    if actual_features != expected_features:
        print(f"[{tf.upper()}] Cached model has {actual_features} features (expected {expected_features}). Retraining...")
        train_horizon(raw_df, tf, HORIZONS[tf])
        models = joblib.load(model_path)

    # Build timeframe-specific features so 30m and 4h each have their own true Wilder's RSI & EMAs
    df_feat = build_features(raw_df, tf).dropna(subset=FEATURES).reset_index(drop=True)
    if df_feat.empty:
        print(f"[{tf.upper()}] Not enough warmup bars to compute features (rows={len(raw_df)}).")
        return

    latest_row = df_feat.iloc[[-1]]
    X_latest = latest_row[FEATURES]

    price = float(latest_row["close"].iloc[0])
    rsi = float(latest_row["rsi"].iloc[0])
    anchor_ema = float(latest_row["ema_fast"].iloc[0])

    # Predict Upside and Downside stretch relative to anchor_ema
    pred_up = float(models["up"].predict(X_latest)[0])
    pred_down = float(models["down"].predict(X_latest)[0])

    # Anchor projected range to the timeframe's structural EMA (matches train.py target_up/target_down)
    proj_high = anchor_ema + pred_up
    proj_low = anchor_ema - pred_down
    total_range = max(proj_high - proj_low, 10.0)

    # Equilibrium (Midline of AI Projected Range) & Half-Range (Distance from EQ to Limit)
    midline_eq = (proj_high + proj_low) / 2.0
    half_range = total_range / 2.0

    # 60% Distance Thresholds (0.60) applied to BOTH 30m and 4h
    sell_trigger = midline_eq + (half_range * 0.60)
    buy_trigger = midline_eq - (half_range * 0.60)

    if price <= buy_trigger:
        signal = "BUY"
    elif price >= sell_trigger:
        signal = "SELL"
    else:
        signal = "WAIT"

    # Dynamic 20% Entry Limit Zones (20% of the distance from EQ to Upper/Lower Limit)
    zone_height = max(half_range * 0.20, 5.0)
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
    print(f"[{tf.upper()}] {signal} @ {price_0d:.0f} | AnchorEMA:{anchor_ema:.0f} | Buy<={buy_trigger:.0f} Sell>={sell_trigger:.0f} | EQ/TP: {eq_0d:.0f} | {reason}")

if __name__ == "__main__":
    with psycopg.connect(DB_URL) as conn:
        raw_df = sync_recent_bars_and_load_warmup(conn)
        for tf in ["30m", "4h"]:
            predict_for_timeframe(conn, raw_df, tf)
