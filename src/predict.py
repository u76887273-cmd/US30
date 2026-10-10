import os
import joblib
import pandas as pd
import numpy as np

# Import our custom modules
from src.config import (
    UPSIDE_MODEL_PATH,
    DOWNSIDE_MODEL_PATH,
    RANGE_MODEL_PATH,
    FEATURES_PATH
)
from src.database import load_recent_bars
from src.features import generate_features

def main():
    print("========================================")
    print("      US30 QUANTUM — LIVE PREDICTION    ")
    print("========================================")

    # 1. Check if model files exist
    if not os.path.exists(UPSIDE_MODEL_PATH):
        print(f"!!! Error: Models not found in {os.path.dirname(UPSIDE_MODEL_PATH)}")
        print("Please upload your .joblib files to the models/ folder.")
        return

    # 2. Load Models
    print("Loading models...")
    upside_model = joblib.load(UPSIDE_MODEL_PATH)
    downside_model = joblib.load(DOWNSIDE_MODEL_PATH)
    range_model = joblib.load(RANGE_MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    # 3. Fetch Recent Data (Just enough for EMA50/VWAP)
    print("Fetching recent M5 bars from database...")
    raw_df = load_recent_bars(limit=200)
    
    if raw_df is None or raw_df.empty:
        print("!!! Error: No data found in database. Run data.py first.")
        return

    # 4. Generate Features
    print("Calculating features...")
    features_df = generate_features(raw_df)

    if features_df.empty:
        print("!!! Error: Not enough data to generate features.")
        return

    # 5. Extract Latest Row
    latest_row = features_df.iloc[-1:]
    current_price = float(latest_row["close"].iloc[0])
    current_time = latest_row["timestamp"].iloc[0]

    # Prepare exactly the columns the model expects
    X = latest_row[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 6. Make Predictions
    predicted_upside = max(float(upside_model.predict(X)[0]), 0.0)
    predicted_downside = max(float(downside_model.predict(X)[0]), 0.0)
    predicted_width = max(float(range_model.predict(X)[0]), 1.0)

    # 7. Calculate Hybrid Range
    raw_upper = current_price + predicted_upside
    raw_lower = current_price - predicted_downside
    midpoint = (raw_upper + raw_lower) / 2.0
    
    upper = midpoint + (predicted_width / 2.0)
    lower = midpoint - (predicted_width / 2.0)
    width = upper - lower

    # 8. Calculate Zones
    edge = width * 0.15
    buy_zone_low = lower
    buy_zone_high = lower + edge
    sell_zone_low = upper - edge
    sell_zone_high = upper

    if width > 0:
        position = (current_price - lower) / width
    else:
        position = 0.50
    position = max(0.0, min(1.0, position))

    # 9. Determine Status
    if position <= 0.15:
        status = "BUY"
    elif position >= 0.85:
        status = "SELL"
    else:
        status = "WAIT"

    # 10. Print Output
    print()
    print("========================================")
    print("        US30 QUANTUM — 30M RANGE        ")
    print("========================================")
    print(f"Time (UTC)    : {current_time}")
    print(f"Current Price : {current_price:.0f}")
    print(f"Status        : {status}")
    print(f"Width         : {width:.0f}")
    print()

    if status == "WAIT":
        print(f"Buy  : {buy_zone_low:.0f} → {buy_zone_high:.0f}")
        print(f"Sell : {sell_zone_low:.0f} → {sell_zone_high:.0f}")
        print(f"Buy Target  : {upper:.0f}")
        print(f"Sell Target : {lower:.0f}")
    elif status == "BUY":
        print(f"Buy Zone : {buy_zone_low:.0f} → {buy_zone_high:.0f}")
        print(f"Target   : {upper:.0f}")
    elif status == "SELL":
        print(f"Sell Zone : {sell_zone_low:.0f} → {sell_zone_high:.0f}")
        print(f"Target    : {lower:.0f}")
    print("========================================")

if __name__ == "__main__":
    main()
