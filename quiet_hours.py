import os
import psycopg
import pandas as pd

DB_URL = os.environ.get("DATABASE_URL")
WINDOW_BARS = 24  # 24 x 5-min bars = 2 Hours

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
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    return (
        df.dropna()
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def analyze_quiet_hours():
    df = load_bars()

    # Forward 2-hour (24-bar) rolling High - Low range
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=WINDOW_BARS)
    df["high_2h"] = df["high"].rolling(window=indexer).max()
    df["low_2h"] = df["low"].rolling(window=indexer).min()
    df["range_2h"] = df["high_2h"] - df["low_2h"]

    # Drop incomplete trailing window and weekend gaps (> 2h15m elapsed across 24 bars)
    df["span_mins"] = (df["timestamp"].shift(-(WINDOW_BARS - 1)) - df["timestamp"]).dt.total_seconds() / 60.0
    valid = df.iloc[:-WINDOW_BARS].copy()
    valid = valid[valid["span_mins"] <= 135].dropna(subset=["range_2h"])

    valid["utc_hour"] = valid["timestamp"].dt.hour
    valid["cairo_hour"] = valid["timestamp"].dt.tz_convert("Africa/Cairo").dt.hour
    valid["under_100"] = (valid["range_2h"] <= 100.0).astype(float) * 100.0
    valid["under_200"] = (valid["range_2h"] <= 200.0).astype(float) * 100.0

    summary = (
        valid.groupby("utc_hour")
        .agg(
            cairo_hour=("cairo_hour", lambda x: x.mode().iloc[0]),
            samples=("range_2h", "count"),
            avg_range=("range_2h", "mean"),
            median_range=("range_2h", "median"),
            pct_under_100=("under_100", "mean"),
            pct_under_200=("under_200", "mean"),
        )
        .reset_index()
        .sort_values(by=["pct_under_100", "avg_range"], ascending=[False, True])
    )

    print("=" * 82)
    print("US30 QUIETEST 2-HOUR WINDOWS (RANKED BY % TIME <= 100 PTS & <= 200 PTS)")
    print("=" * 82)
    print(f"{'UTC Hour':<10} | {'Cairo Hour':<12} | {'Avg 2H Pts':<11} | {'Med 2H Pts':<11} | {'<= 100 Pts':<11} | {'<= 200 Pts':<11}")
    print("-" * 82)
    for _, row in summary.iterrows():
        print(
            f"{int(row['utc_hour']):02d}:00 UTC | "
            f"{int(row['cairo_hour']):02d}:00 Cairo | "
            f"{row['avg_range']:>9.1f}   | "
            f"{row['median_range']:>9.1f}   | "
            f"{row['pct_under_100']:>8.1f}%  | "
            f"{row['pct_under_200']:>8.1f}%"
        )
    print("=" * 82)

if __name__ == "__main__":
    analyze_quiet_hours()
