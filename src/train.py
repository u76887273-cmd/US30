import os
import sys
import subprocess
import joblib
import psycopg
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ============================================================
# US30 QUANTUM SUPREME — AUTOMATED AI MODEL TRAINER (ONCE PER DAY)
# ============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LAST_TRAINED_FILE = f"{MODEL_DIR}/last_trained.txt"
TODAY_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ------------------------------------------------------------
# 0. CHECK IF ALREADY TRAINED TODAY (UTC)
# ------------------------------------------------------------
if os.path.exists(LAST_TRAINED_FILE):
    with open(LAST_TRAINED_FILE, "r") as f:
        last_trained = f.read().strip()
    if last_trained == TODAY_UTC:
        print(f"US30 AI models already trained today ({TODAY_UTC}). Skipping training.")
        sys.exit(0)

print(f"Starting daily US30 AI model training for {TODAY_UTC}...")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is missing.")
    sys.exit(1)

# ------------------------------------------------------------
# 1. FETCH ALL HISTORICAL DATA FROM SUPABASE (m5_bars)
# ------------------------------------------------------------
print("Connecting to US30 Supabase database...")
try:
    conn = psycopg.connect(DATABASE_URL)
    cur = conn.cursor()
    print("Fetching all historical M5 bars from m5_bars...")
    cur.execute("SELECT * FROM m5_bars ORDER BY timestamp ASC;")
    columns = [desc[0] for desc in cur.description]
    data = cur.fetchall()
    cur.close()
    conn.close()
except Exception as e:
    print(f"Database error: {e}")
    sys.exit(1)

if not data or len(data) < 200:
    print(f"Not enough data to train (found {len(data) if data else 0} rows). Need at least 200 rows.")
    sys.exit(0)

bars = pd.DataFrame(data, columns=columns)
bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
bars = bars.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
print(f"Loaded {len(bars):,} historical 5-minute candles.")

# ------------------------------------------------------------
# 2. ENGINEER ALL 31 INPUT FEATURES
# ------------------------------------------------------------
print("Engineering 31 AI features across historical dataset...")

bars["return_1"] = bars["close"].pct_change(1) * 100
bars["return_3"] = bars["close"].pct_change(3) * 100
bars["return_6"] = bars["close"].pct_change(6) * 100

bars["body"] = bars["close"] - bars["open"]
bars["upper_wick"] = bars["high"] - bars[["open", "close"]].max(axis=1)
bars["lower_wick"] = bars[["open", "close"]].min(axis=1) - bars["low"]

bars["ema20"] = bars["close"].ewm(span=20, adjust=False).mean()
bars["ema50"] = bars["close"].ewm(span=50, adjust=False).mean()
bars["ema_gap"] = bars["ema20"] - bars["ema50"]
bars["ema20_slope"] = bars["ema20"].diff()

bars["prev_close"] = bars["close"].shift(1)
bars["tr1"] = bars["high"] - bars["low"]
bars["tr2"] = (bars["high"] - bars["prev_close"]).abs()
bars["tr3"] = (bars["low"] - bars["prev_close"]).abs()
bars["tr"] = bars[["tr1", "tr2", "tr3"]].max(axis=1)
bars["atr14"] = bars["tr"].rolling(14).mean()
bars["atr50"] = bars["tr"].rolling(50).mean()
bars["relative_atr"] = bars["atr14"] / bars["atr50"]

delta = bars["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
bars["rsi14"] = 100 - (100 / (1 + rs))

bars["date"] = bars["timestamp"].dt.date
bars["typical_price"] = (bars["high"] + bars["low"] + bars["close"]) / 3
cum_vp = (bars["typical_price"] * bars["volume"]).groupby(bars["date"]).cumsum()
cum_v = bars["volume"].groupby(bars["date"]).cumsum().replace(0, np.nan)
bars["vwap"] = cum_vp / cum_v
bars["vwap_distance"] = bars["close"] - bars["vwap"]

or_df = bars.groupby("date").head(12).groupby("date").agg({
    "high": "max",
    "low": "min",
    "open": "first"
})
bars["opening_high"] = bars["date"].map(or_df["high"])
bars["opening_low"] = bars["date"].map(or_df["low"])
bars["opening_open"] = bars["date"].map(or_df["open"])

bars["dist_or_high"] = bars["close"] - bars["opening_high"]
bars["dist_or_low"] = bars["close"] - bars["opening_low"]
bars["dist_open_price"] = bars["close"] - bars["opening_open"]
bars["inside_or"] = ((bars["close"] < bars["opening_high"]) & (bars["close"] > bars["opening_low"])).astype(int)

hour = bars["timestamp"].dt.hour
bars["tokyo"] = ((hour >= 0) & (hour < 8)).astype(int)
bars["london"] = ((hour >= 8) & (hour < 16)).astype(int)
bars["newyork"] = ((hour >= 13) & (hour < 21)).astype(int)

# ------------------------------------------------------------
# 3. CREATE 30-MINUTE (6-BAR) FORWARD TARGETS
# ------------------------------------------------------------
print("Calculating 30-minute forward targets (next 6 candles)...")

future_highs = pd.concat([bars["high"].shift(-i) for i in range(1, 7)], axis=1)
future_lows = pd.concat([bars["low"].shift(-i) for i in range(1, 7)], axis=1)

bars["future_high_30m"] = future_highs.max(axis=1, skipna=False)
bars["future_low_30m"] = future_lows.min(axis=1, skipna=False)

bars["target_upside"] = (bars["future_high_30m"] - bars["close"]).clip(lower=0)
bars["target_downside"] = (bars["close"] - bars["future_low_30m"]).clip(lower=0)
bars["target_width"] = bars["future_high_30m"] - bars["future_low_30m"]

FEATURES = [
    "open", "high", "low", "close", "volume",
    "return_1", "return_3", "return_6",
    "body", "upper_wick", "lower_wick",
    "ema20", "ema50", "ema_gap", "ema20_slope",
    "atr14", "atr50", "relative_atr",
    "rsi14",
    "vwap", "vwap_distance",
    "opening_high", "opening_low", "opening_open",
    "dist_or_high", "dist_or_low", "dist_open_price", "inside_or",
    "tokyo", "london", "newyork"
]

train_df = bars.dropna(subset=["atr50", "target_upside", "target_downside", "target_width"]).copy()
X = train_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

y_upside = train_df["target_upside"]
y_downside = train_df["target_downside"]
y_width = train_df["target_width"]

print(f"Training models on {len(X):,} clean historical samples...")

model_params = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "random_state": 42
}

upside_model = HistGradientBoostingRegressor(**model_params)
downside_model = HistGradientBoostingRegressor(**model_params)
width_model = HistGradientBoostingRegressor(**model_params)

upside_model.fit(X, y_upside)
downside_model.fit(X, y_downside)
width_model.fit(X, y_width)

up_mae = mean_absolute_error(y_upside, upside_model.predict(X))
down_mae = mean_absolute_error(y_downside, downside_model.predict(X))
width_mae = mean_absolute_error(y_width, width_model.predict(X))

joblib.dump(FEATURES, f"{MODEL_DIR}/features.joblib")
joblib.dump(upside_model, f"{MODEL_DIR}/upside30.joblib")
joblib.dump(downside_model, f"{MODEL_DIR}/downside30.joblib")
joblib.dump(width_model, f"{MODEL_DIR}/width30.joblib")

with open(LAST_TRAINED_FILE, "w") as f:
    f.write(TODAY_UTC)

print("\n" + "=" * 64)
print("US30 QUANTUM SUPREME — TRAINING COMPLETE")
print("=" * 64)
print(f"Total Bars Trained : {len(X):,}")
print(f"Date Range         : {train_df['timestamp'].iloc[0]} -> {train_df['timestamp'].iloc[-1]}")
print(f"Upside 30m MAE     : {up_mae:.2f} pts")
print(f"Downside 30m MAE   : {down_mae:.2f} pts")
print(f"Width 30m MAE      : {width_mae:.2f} pts")
print("=" * 64)

try:
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
    subprocess.run([
        "git", "add",
        f"{MODEL_DIR}/features.joblib",
        f"{MODEL_DIR}/upside30.joblib",
        f"{MODEL_DIR}/downside30.joblib",
        f"{MODEL_DIR}/width30.joblib",
        LAST_TRAINED_FILE
    ], check=True)
    subprocess.run(["git", "commit", "-m", f"Daily US30 AI model retrain ({TODAY_UTC})"], check=False)
    subprocess.run(["git", "push"], check=False)
    print("Successfully saved updated US30 models to GitHub repository.")
except Exception as e:
    print(f"Git commit/push skipped: {e}")
