import os
import psycopg
import pandas as pd

DB_URL = os.environ.get("DATABASE_URL")

def load_bars():
    if DB_URL:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT timestamp, high, low FROM m5_bars ORDER BY timestamp ASC;")
                df = pd.DataFrame(cur.fetchall(), columns=["timestamp", "high", "low"])
    elif os.path.exists("data_m5.csv"):
        df = pd.read_csv("data_m5.csv")[["timestamp", "high", "low"]]
    else:
        raise FileNotFoundError("Provide DATABASE_URL or data_m5.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

def analyze_quiet_hours():
    df = load_bars()

    # 1-Hour (12 bars) and 2-Hour (24 bars) forward rolling windows
    idx_1h = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    idx_2h = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)

    df["high_1h"] = df["high"].rolling(window=idx_1h).max()
    df["low_1h"] = df["low"].rolling(window=idx_1h).min()
    df["range_1h"] = df["high_1h"] - df["low_1h"]

    df["high_2h"] = df["high"].rolling(window=idx_2h).max()
    df["low_2h"] = df["low"].rolling(window=idx_2h).min()
    df["range_2h"] = df["high_2h"] - df["low_2h"]

    # Drop trailing incomplete windows and large time gaps
    df["span_mins"] = (df["timestamp"].shift(-23) - df["timestamp"]).dt.total_seconds() / 60.0
    valid = df.iloc[:-24].copy()
    valid = valid[valid["span_mins"] <= 135].dropna(subset=["range_1h", "range_2h"])

    valid["cairo_hour"] = valid["timestamp"].dt.tz_convert("Africa/Cairo").dt.hour
    
    # Calculate percentages for staying under 100 points
    valid["1h_under_100"] = (valid["range_1h"] <= 100.0).astype(float) * 100.0
    valid["2h_under_100"] = (valid["range_2h"] <= 100.0).astype(float) * 100.0

    summary = (
        valid.groupby("cairo_hour")
        .agg(
            avg_1h=("range_1h", "mean"),
            avg_2h=("range_2h", "mean"),
            pct_1h_under_100=("1h_under_100", "mean"),
            pct_2h_under_100=("2h_under_100", "mean"),
        )
        .reset_index()
        .sort_values(by=["pct_2h_under_100", "avg_2h"], ascending=[False, True])
    )

    print("=" * 76)
    print(f"{'Cairo Time':<12} | {'Avg 1H Pts':<12} | {'Avg 2H Pts':<12} | {'1H <= 100':<11} | {'2H <= 100':<11}")
    print("-" * 76)
    for _, row in summary.iterrows():
        print(
            f"{int(row['cairo_hour']):02d}:00 Cairo | "
            f"{row['avg_1h']:>10.1f} | "
            f"{row['avg_2h']:>10.1f} | "
            f"{row['pct_1h_under_100']:>9.1f}% | "
            f"{row['pct_2h_under_100']:>9.1f}%"
        )
    print("=" * 76)

if __name__ == "__main__":
    analyze_quiet_hours()
