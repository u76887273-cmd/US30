import os
import datetime
import psycopg
import pandas as pd
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.environ.get("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = int(os.environ.get("ACCOUNT_ID") or os.environ.get("CTRADER_ACCOUNT_ID") or 0)
SYMBOL_ID = int(os.environ.get("SYMBOL_ID") or 1)  # EUR/USD Symbol ID = 1

client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)

def on_error(failure):
    print(f"cTrader API Error: {failure}")
    if reactor.running:
        reactor.stop()

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

    # Sync with Supabase persistent historical storage (eurusd_features_m5)
    db_url = os.environ.get("DATABASE_URL")
    if db_url and not df.empty:
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
                tuples = [
                    (r["timestamp"].isoformat(), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]))
                    for _, r in df.iterrows()
                ]
                cur.executemany("""
                    INSERT INTO eurusd_features_m5 (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO UPDATE SET
                        high = GREATEST(eurusd_features_m5.high, EXCLUDED.high),
                        low = LEAST(eurusd_features_m5.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume;
                """, tuples)

                cur.execute("SELECT timestamp, open, high, low, close, volume FROM eurusd_features_m5 ORDER BY timestamp ASC")
                full_history = cur.fetchall()
                df = pd.DataFrame(full_history, columns=["timestamp", "open", "high", "low", "close", "volume"])
            conn.commit()

    # Save to CSVs so train.py, predict.py, and validator.py work seamlessly
    df.to_csv("data_m5.csv", index=False)
    df.to_csv("data_30m.csv", index=False)
    df.to_csv("data_4h.csv", index=False)
    print(f"[M5] Saved {len(df):,} total 5-min bars (Latest close: {df['close'].iloc[-1]:.4f})")

    if reactor.running:
        reactor.stop()

def on_account_auth(result):
    print("cTrader account authenticated. Pulling EUR/USD M5 bars...")
    now = datetime.datetime.now(datetime.timezone.utc)
    from_ts = int((now - datetime.timedelta(days=14)).timestamp() * 1000)
    to_ts = int(now.timestamp() * 1000)

    req = ProtoOAGetTrendbarsReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.symbolId = SYMBOL_ID
    req.period = ProtoOATrendbarPeriod.M5
    req.fromTimestamp = from_ts
    req.toTimestamp = to_ts

    d = client.send(req)
    d.addCallback(on_trendbars)
    d.addErrback(on_error)

def on_app_auth(result):
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.accessToken = ACCESS_TOKEN
    d = client.send(req)
    d.addCallback(on_account_auth)
    d.addErrback(on_error)

def connected(client_instance):
    req = ProtoOAApplicationAuthReq()
    req.clientId = CLIENT_ID
    req.clientSecret = CLIENT_SECRET
    d = client_instance.send(req)
    d.addCallback(on_app_auth)
    d.addErrback(on_error)

if __name__ == "__main__":
    client.setConnectedCallback(connected)
    client.startService()
    reactor.run()
