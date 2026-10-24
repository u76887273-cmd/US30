import psycopg

def on_trendbars(result):
    res = Protobuf.extract(result)
    rows = []

    for b in res.trendbar:
        low = b.low / 100000.0
        open_p = (b.low + b.deltaOpen) / 100000.0
        close_p = (b.low + b.deltaClose) / 100000.0
        high_p = (b.low + b.deltaHigh) / 100000.0
        ts = datetime.datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=datetime.timezone.utc)
        rows.append({
            "timestamp": ts,
            "open": round(open_p, 5),
            "high": round(high_p, 5),
            "low": round(low, 5),
            "close": round(close_p, 5),
            "volume": b.volume
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    
    # Supabase Injection - Accumulate History
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS eurusd_features_m5 (
                        timestamp TIMESTAMPTZ NOT NULL UNIQUE,
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION,
                        volume DOUBLE PRECISION
                    );
                """)
                
                # Upsert new bars
                tuples = [(r["timestamp"].isoformat(), r["open"], r["high"], r["low"], r["close"], r["volume"]) for _, r in df.iterrows()]
                cur.executemany("""
                    INSERT INTO eurusd_features_m5 (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO UPDATE SET
                        high = GREATEST(eurusd_features_m5.high, EXCLUDED.high),
                        low = LEAST(eurusd_features_m5.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume;
                """, tuples)
                
                # Extract full historical dataset for AI Training
                cur.execute("SELECT timestamp, open, high, low, close, volume FROM eurusd_features_m5 ORDER BY timestamp ASC")
                full_history = cur.fetchall()
                df = pd.DataFrame(full_history, columns=["timestamp", "open", "high", "low", "close", "volume"])
            conn.commit()

    # Save final dataset for train.py
    df.to_csv("data_m5.csv", index=False)
    print(f"[M5] Saved {len(df)} historical 5-min bars to data_m5.csv (Latest close: {df['close'].iloc[-1]:.5f})")

    if reactor.running:
        reactor.stop()
