import pandas as pd
import numpy as np

def generate_features(df):
    """
    Takes a DataFrame of raw M5 bars (timestamp, open, high, low, close, volume)
    and generates all required features for the US30 Quantum 30-minute range model.
    """
    
    # 1. TIMESTAMP / SORT
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    # 2. RETURNS / MOMENTUM
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["move_1"] = df["close"].diff(1)
    df["move_3"] = df["close"].diff(3)
    df["move_6"] = df["close"].diff(6)

    # 3. CANDLE STRUCTURE
    df["body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_ratio"] = df["body"].abs() / df["candle_range"].replace(0, np.nan)

    # 4. EMA / TREND STRUCTURE
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_gap"] = df["ema20"] - df["ema50"]
    df["ema20_slope"] = df["ema20"] - df["ema20"].shift(5)
    df["ema50_slope"] = df["ema50"] - df["ema50"].shift(5)
    df["price_vs_ema20"] = df["close"] - df["ema20"]
    df["price_vs_ema50"] = df["close"] - df["ema50"]

    # 5. RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 6. ATR / VOLATILITY
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    
    df["atr14"] = true_range.rolling(14).mean()
    df["atr50"] = true_range.rolling(50).mean()
    df["relative_atr"] = df["atr14"] / df["atr50"].replace(0, np.nan)
    df["range_ma6"] = df["candle_range"].rolling(6).mean()
    df["range_ma12"] = df["candle_range"].rolling(12).mean()

    # 7. MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 8. VOLUME
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["volume_roc"] = df["volume"].pct_change(5)

    # 9. DAILY VWAP
    df["trade_date"] = df["timestamp"].dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical_price * df["volume"]).groupby(df["trade_date"]).cumsum()
    cum_volume = df["volume"].groupby(df["trade_date"]).cumsum()
    df["vwap"] = cum_pv / cum_volume.replace(0, np.nan)
    df["vwap_distance"] = df["close"] - df["vwap"]

    # 10. RECENT SWING STRUCTURE
    df["swing_high20"] = df["high"].rolling(20).max()
    df["swing_low20"] = df["low"].rolling(20).min()
    df["dist_high"] = df["swing_high20"] - df["close"]
    df["dist_low"] = df["close"] - df["swing_low20"]

    # 11. 09:30 CASH OPEN FEATURES
    df["opening_open"] = np.nan
    is_0930 = ((df["timestamp"].dt.hour == 9) & (df["timestamp"].dt.minute == 30))
    df.loc[is_0930, "opening_open"] = df.loc[is_0930, "open"]
    df["opening_open"] = df.groupby("trade_date")["opening_open"].transform("first")

    after_0930 = (df["timestamp"].dt.hour > 9) | ((df["timestamp"].dt.hour == 9) & (df["timestamp"].dt.minute >= 30))
    
    or_high_source = df["high"].where(after_0930)
    or_low_source = df["low"].where(after_0930)
    
    df["opening_high"] = or_high_source.groupby(df["trade_date"]).cummax()
    df["opening_low"] = or_low_source.groupby(df["trade_date"]).cummin()
    df["opening_range_width"] = df["opening_high"] - df["opening_low"]
    
    df["dist_or_high"] = df["opening_high"] - df["close"]
    df["dist_or_low"] = df["close"] - df["opening_low"]
    
    or_width = df["opening_range_width"]
    df["or_position"] = (df["close"] - df["opening_low"]) / or_width.replace(0, np.nan)
    
    df["inside_or"] = ((df["close"] <= df["opening_high"]) & (df["close"] >= df["opening_low"])).astype(int)
    df["breakout_up"] = (df["close"] > df["opening_high"]).astype(int)
    df["breakout_down"] = (df["close"] < df["opening_low"]).astype(int)
    df["cash_open"] = is_0930.astype(int)
    df["dist_open_price"] = df["close"] - df["opening_open"]

    # 12. PRE-09:30 HANDLING
    pre_open = ~after_0930
    df.loc[pre_open, "opening_high"] = df.loc[pre_open, "close"]
    df.loc[pre_open, "opening_low"] = df.loc[pre_open, "close"]
    df.loc[pre_open, "opening_range_width"] = 0
    df.loc[pre_open, "dist_or_high"] = 0
    df.loc[pre_open, "dist_or_low"] = 0
    df.loc[pre_open, "or_position"] = 0
    df.loc[pre_open, "inside_or"] = 0
    df.loc[pre_open, "breakout_up"] = 0
    df.loc[pre_open, "breakout_down"] = 0
    df.loc[pre_open, "opening_open"] = df.loc[pre_open, "open"]
    df.loc[pre_open, "dist_open_price"] = 0

    # 13. SESSION FLAGS
    hour = df["timestamp"].dt.hour
    df["london"] = ((hour >= 7) & (hour < 16)).astype(int)
    df["newyork"] = ((hour >= 13) & (hour < 21)).astype(int)

    # 14. TIME-OF-DAY FEATURES
    df["hour_utc"] = df["timestamp"].dt.hour
    df["minute_utc"] = df["timestamp"].dt.minute
    df["minutes_from_midnight"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    # 15. CLEAN AND RETURN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    
    return df
