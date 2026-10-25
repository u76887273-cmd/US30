import os
import joblib
import psycopg
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DB_URL = os.environ.get("DATABASE_URL")
TABLE_NAME = "m5_bars"

HORIZONS = {
    "30m": 6,   # 6 x 5-min bars = 30 Minutes
    "4h":  48   # 48 x 5-min bars = 4 Hours
}

# Upgraded 9-Feature Vector (includes Price-to-EMA Stretch & Cyclical Session Time)
FEATURES = [
    "ema_diff",
    "ema_stretch",
    "rsi",
    "atr",
    "ret_short",
    "ret_long",
    "vol_ratio",
    "hour_sin",
    "hour_cos"
]

def parse_mixed_timestamps(series):
    try:
        return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")

def sync_and_load_all_m5_bars():
    if not DB_URL:
        print("⚠️ DATABASE_URL not set. Falling back to local data_m5.csv...")
        if not os.path.exists("data_m5.csv"):
            raise FileNotFoundError("Neither DATABASE_URL nor data_m5.csv is available.")
        df = pd.read_csv("data_m5.csv")
        df["timestamp"] = parse_mixed_timestamps(df["timestamp"])
        return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT timestamp, open, high, low, close, volume FROM {TABLE_NAME};")
            rows = cur.fetchall()
            col_names = [desc[0].lower() for desc in cur.description]

    db_df = pd.DataFrame(rows, columns=col_names)
    raw_db_count = len(db_df)
    db_df["timestamp"] = parse_mixed_timestamps(db_df["timestamp"])
    db_df = db_df.dropna(subset=["timestamp"])

    if os.path.exists("data_m5.csv"):
        try:
            local_df = pd.read_csv("data_m5.csv")
            local_df.columns = [c.lower() for c in local_df.columns]
            if "timestamp" in local_df.columns and "close" in local_df.columns:
                local_df["timestamp"] = parse_mixed_timestamps(local_df["timestamp"])
                local_df = local_df.dropna(subset=["timestamp"])

                existing_ts = set(db_df["timestamp"])
                new_bars = local_df[~local_df["timestamp"].isin(existing_ts)].copy()

                if not new_bars.empty:
                    print(f"🔄 Found {len(new_bars):,} new 5m bars in data_m5.csv! Inserting into Supabase '{TABLE_NAME}'...")
                    with psycopg.connect(DB_URL) as conn:
                        with conn.cursor() as cur:
                            insert_data = [
                                (
                                    row["timestamp"].isoformat(),
                                    float(row["open"]),
                                    float(row["high"]),
                                    float(row["low"]),
                                    float(row["close"]),
                                    float(row.get("volume", 0.0))
                                )
                                for _, row in new_bars.iterrows()
                            ]
                            cur.executemany(
                                f"""
                                INSERT INTO {TABLE_NAME} (timestamp, open, high, low, close, volume)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING;
                                """,
                                insert_data
                            )
                        conn.commit()
                    db_df = pd.concat([db_df, new_bars[col_names]], ignore_index=True)
        except Exception as e:
            print(f"Note on local data_m5.csv sync check: {e}")

    for col in ["open", "high", "low", "close", "volume"]:
        db_df[col] = pd.to_numeric(db_df[col], errors="coerce")

    db_df = (
        db_df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values(by="timestamp")
        .reset_index(drop=True)
    )

    db_df.to_csv("data_m5.csv", index=False)
    print("=" * 70)
    print(f"✅ FULL DATASET READY FOR TRAINING ('{TABLE_NAME}')")
    print(f"   Total Raw Rows Loaded : {raw_db_count:,}")
    print(f"   Clean Unique M5 Bars  : {len(db_df):,}")
    print(f"   Start Timestamp       : {db_df['timestamp'].iloc[0]}")
    print(f"   Latest Timestamp      : {db_df['timestamp'].iloc[-1]}")
    print("=" * 70)

    return db_df

def build_features(df, tf_name="30m"):
    df = df.copy()
    scale = 1 if tf_name == "30m" else 8

    df["ema_fast"] = df["close"].ewm(span=20 * scale, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50 * scale, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / (14 * scale), adjust=False).mean()

    df["ema_stretch"] = (df["close"] - df["ema_fast"]) / (df["atr"] + 1e-5)

    delta = df["close"].diff(1)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / (14 * scale), min_periods=14 * scale, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / (14 * scale), min_periods=14 * scale, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    df["ret_short"] = df["close"].pct_change(3 * scale)
    df["ret_long"] = df["close"].pct_change(12 * scale)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20 * scale).mean() + 1e-5)

    ts_utc = parse_mixed_timestamps(df["timestamp"])
    fractional_hour = ts_utc.dt.hour + (ts_utc.dt.minute / 60.0)
    df["hour_sin"] = np.sin(2.0 * np.pi * fractional_hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * fractional_hour / 24.0)

    return df

def train_horizon(raw_df, tf_name, horizon_bars):
    df = build_features(raw_df, tf_name)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_bars)
    future_high = df["high"].shift(-1).rolling(window=indexer).max()
    future_low = df["low"].shift(-1).rolling(window=indexer).min()

    df["target_up"] = (future_high - df["ema_fast"]).clip(lower=1.0)
    df["target_down"] = (df["ema_fast"] - future_low).clip(lower=1.0)

    df_clean = df.dropna(subset=FEATURES + ["target_up", "target_down"]).reset_index(drop=True)
    X = df_clean[FEATURES]

    model_up = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1
    )
    model_down = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1
    )

    model_up.fit(X, df_clean["target_up"])
    model_down.fit(X, df_clean["target_down"])

    joblib.dump({"up": model_up, "down": model_down}, f"model_{tf_name}.pkl")
    print(
        f"[{tf_name.upper()}] Trained Upside/Downside Regressors on {len(X):,} bars "
        f"({df_clean['timestamp'].iloc[0]} --> {df_clean['timestamp'].iloc[-1]} | "
        f"Latest {tf_name.upper()} RSI: {df_clean['rsi'].iloc[-1]:.1f})"
    )

if __name__ == "__main__":
    raw_df = sync_and_load_all_m5_bars()
    for tf, bars in HORIZONS.items():
        train_horizon(raw_df, tf, bars)
