import os
import sys
import json
import datetime
import urllib.request
import urllib.parse
import joblib
import psycopg
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    build_features,
    train_horizon,
    sync_and_load_all_m5_bars,
    FEATURES,
    HORIZONS,
    ensure_unique_timestamp_constraint,
)

DB_URL = os.environ.get("DATABASE_URL")
RAW_BARS_TABLE = "m5_bars"  # Matches your existing US30 Supabase table

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
    """Reads latest_quote.json produced by data.py if a fresh quote was received."""
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

def sync_recent_bars_and_load_warmup(conn):
    """Syncs pure closed M5 bars into Supabase without overwriting them with mid-bar live spot ticks."""
    ensure_unique_timestamp_constraint(conn)
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

                recent_slice = local_df.tail(250)
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
                        ON CONFLICT (timestamp) DO UPDATE SET
                            high = GREATEST({RAW_BARS_TABLE}.high, EXCLUDED.high),
                            low = LEAST({RAW_BARS_TABLE}.low, EXCLUDED.low),
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume;
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
                entry_zone VARCHAR(128),
                entry NUMERIC(10, 2),
                take_profit NUMERIC(10, 2),
                status VARCHAR(32) DEFAULT 'ACTIVE',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            ALTER TABLE {table_name} ALTER COLUMN entry_zone TYPE VARCHAR(128);
            ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_timestamp_key;
            GRANT SELECT ON {table_name} TO anon, authenticated;
        """)
    conn.commit()

def send_telegram_alert(tf, signal, price_0d, entry_zone, eq_0d, ema_0d, rsi, stretch_pct):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (bot_token and chat_id and signal in ["BUY", "SELL"]):
        return

    icon = "🟢" if signal == "BUY" else "🔴"
    msg = (
        f"🦅 *US30 QUANTUM ({tf.upper()})*\n"
        f"========================\n"
        f"{icon} *SIGNAL: {signal}*\n"
        f"💰 *Live Price:* {price_0d:.0f}\n"
        f"🏁 *Midline:* {eq_0d:.0f} | *EMA:* {ema_0d:.0f}\n"
        f"📍 *Limit Tiers:* {entry_zone}\n"
        f"📊 *Stretch:* {stretch_pct:.1f}% | *RSI:* {rsi:.1f}"
    )
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
        print(f"[{tf.upper()}] Telegram alert sent!")
    except Exception as e:
        print(f"[{tf.upper()}] Telegram alert note: {e}")

def predict_for_timeframe(conn, raw_df, tf, live_spot):
    model_path = f"model_{tf}.pkl"
    table_name = TABLES[tf]

    if not os.path.exists(model_path):
        print(f"[{tf.upper()}] Model file {model_path} missing. Loading full dataset to train...")
        full_df = sync_and_load_all_m5_bars()
        train_horizon(full_df, tf, HORIZONS[tf])

    models = joblib.load(model_path)
    expected_features = len(FEATURES)
    actual_features = getattr(models["up"], "n_features_in_", expected_features)
    if actual_features != expected_features:
        print(f"[{tf.upper()}] Cached model has {actual_features} features (expected {expected_features}). Retraining on full dataset...")
        full_df = sync_and_load_all_m5_bars()
        train_horizon(full_df, tf, HORIZONS[tf])
        models = joblib.load(model_path)

    # Compute indicators on pure closed M5 bars (matching train.py 100%)
    df_feat = build_features(raw_df, tf).dropna(subset=FEATURES).reset_index(drop=True)
    if df_feat.empty:
        print(f"[{tf.upper()}] Not enough warmup bars to compute features (rows={len(raw_df)}).")
        return

    latest_row = df_feat.iloc[[-1]]
    X_latest = latest_row[FEATURES]

    price = float(live_spot) if live_spot is not None else float(latest_row["close"].iloc[0])
    rsi = float(latest_row["rsi"].iloc[0])
    anchor_ema = float(latest_row["ema_fast"].iloc[0])
    atr_val = max(float(latest_row["atr"].iloc[0]), 5.0)

    pred_up = float(models["up"].predict(X_latest)[0])
    pred_down = float(models["down"].predict(X_latest)[0])

    proj_high = anchor_ema + pred_up
    proj_low = anchor_ema - pred_down

    # Ensure structural projected range is valid even if signed excursion is tight
    if proj_high - proj_low < atr_val:
        mid_temp = (proj_high + proj_low) / 2.0
        proj_high = mid_temp + (atr_val / 2.0)
        proj_low = mid_temp - (atr_val / 2.0)

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
    ema_0d = round(anchor_ema, 0)

    entry_zone = (
        f"B0.55:{b_055:.0f},B0.80:{b_080:.0f},B1.00:{b_100:.0f} | "
        f"S0.55:{s_055:.0f},S0.80:{s_080:.0f},S1.00:{s_100:.0f}"
    )

    # True Stretch % from Equilibrium toward outer limit (55.0%+ triggers BUY/SELL)
    stretch_pct = min((abs(price - midline_eq) / max(half_range, 1.0)) * 100.0, 100.0)

    reason = f"Bias:{stretch_pct:.1f}% | RSI:{rsi:.1f} | EQ:{eq_0d:.0f} | EMA:{ema_0d:.0f}"
    run_ts = datetime.datetime.now(datetime.timezone.utc)

    init_db_table(conn, table_name)

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table_name} (timestamp, price, signal, reason, entry_zone, entry, take_profit, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
        """, (run_ts, price_0d, signal, reason, entry_zone, price_0d, eq_0d))
    conn.commit()
    print(
        f"[{tf.upper()}] {signal} @ {price_0d:.0f} (Spot: {price:.2f}) | AnchorEMA:{ema_0d:.0f} | "
        f"Buy(0.55/0.8/1.0): {b_055:.0f}/{b_080:.0f}/{b_100:.0f} | "
        f"Sell(0.55/0.8/1.0): {s_055:.0f}/{s_080:.0f}/{s_100:.0f} | Midline: {eq_0d:.0f} | Stretch: {stretch_pct:.1f}%"
    )

    send_telegram_alert(tf, signal, price_0d, entry_zone, eq_0d, ema_0d, rsi, stretch_pct)

if __name__ == "__main__":
    live_spot = get_live_spot_quote()
    with psycopg.connect(DB_URL) as conn:
        raw_df = sync_recent_bars_and_load_warmup(conn)
        for tf in ["30m", "4h"]:
            predict_for_timeframe(conn, raw_df, tf, live_spot)
