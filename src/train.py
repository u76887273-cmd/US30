import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HORIZONS = {
    "30m": 6,   # 6 x 5-min bars = 30 Minutes
    "4h":  48   # 48 x 5-min bars = 4 Hours
}

FEATURES = ["ema_diff", "rsi", "atr", "ret_short", "ret_long", "vol_ratio"]

def build_features(df, tf_name="30m"):
    df = df.copy()
    scale = 1 if tf_name == "30m" else 8

    df["ema_fast"] = df["close"].ewm(span=20 * scale, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50 * scale, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

    # Timeframe-specific ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14 * scale).mean()

    # Timeframe-specific RSI
    delta = df["close"].diff(scale)
    gain = delta.clip(lower=0).rolling(14 * scale).mean()
    loss = -delta.clip(upper=0).rolling(14 * scale).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Timeframe-specific Momentum & Volume
    df["ret_short"] = df["close"].pct_change(3 * scale)
    df["ret_long"] = df["close"].pct_change(12 * scale)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20 * scale).mean() + 1e-5)

    return df

def train_horizon(raw_df, tf_name, horizon_bars):
    df = build_features(raw_df, tf_name)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_bars)
    future_high = df["high"].shift(-1).rolling(window=indexer).max()
    future_low = df["low"].shift(-1).rolling(window=indexer).min()

    df["target_up"] = (future_high - df["close"]).clip(lower=0.0001)
    df["target_down"] = (df["close"] - future_low).clip(lower=0.0001)

    df_clean = df.dropna(subset=FEATURES + ["target_up", "target_down"]).reset_index(drop=True)
    X = df_clean[FEATURES]

    model_up = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model_down = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    model_up.fit(X, df_clean["target_up"])
    model_down.fit(X, df_clean["target_down"])

    joblib.dump({"up": model_up, "down": model_down}, f"model_{tf_name}.pkl")
    print(f"[{tf_name.upper()}] Trained Upside/Downside Regressors on {len(X):,} bars (Latest {tf_name.upper()} RSI: {df_clean['rsi'].iloc[-1]:.1f})")

if __name__ == "__main__":
    if not os.path.exists("data_m5.csv"):
        print("data_m5.csv not found.")
    else:
        raw_df = pd.read_csv("data_m5.csv")
        for tf, bars in HORIZONS.items():
            train_horizon(raw_df, tf, bars)
