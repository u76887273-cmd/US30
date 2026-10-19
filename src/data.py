import os
import sys
import psycopg
import pandas as pd
from datetime import datetime, timezone, timedelta
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# ============================================================
# US30 QUANTUM SUPREME — SHARED DB M5 DATA FETCHER (SAFE MODE)
# ============================================================

raw_db_url = os.environ.get("DATABASE_URL", "")
DATABASE_URL = raw_db_url.strip().strip('"').strip("'")

CTRADER_CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID", "").strip()
CTRADER_CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET", "").strip()
CTRADER_ACCESS_TOKEN = os.environ.get("CTRADER_ACCESS_TOKEN", "").strip()
ACCOUNT_ID = int(os.environ.get("ACCOUNT_ID", "25145646"))
SYMBOL_ID = int(os.environ.get("SYMBOL_ID", "1287"))

if not DATABASE_URL.startswith("postgres"):
    print("CRITICAL ERROR WITH DATABASE_URL!")
    sys.exit(1)

conn = psycopg.connect(DATABASE_URL, autocommit=True)
cur = conn.cursor()

# Check if the 30-minute repo already updated m5_bars recently
cur.execute("SELECT COUNT(*), MAX(timestamp) FROM m5_bars;")
row_count, latest_ts = cur.fetchone()
print(f"Shared Supabase DB currently has {row_count} M5 bars (Latest: {latest_ts}).")

# Safely parse latest_ts whether the DB column is TEXT or TIMESTAMPTZ
if latest_ts is not None:
    try:
        latest_dt = pd.to_datetime(latest_ts, utc=True).to_pydatetime()
        now_utc = datetime.now(timezone.utc)
        if (now_utc - latest_dt).total_seconds() < 600 and row_count > 300:
            print("m5_bars is already up-to-date from the 30-minute US30 engine! Proceeding directly to 4H AI.")
            cur.close()
            conn.close()
            sys.exit(0)
    except Exception as e:
        print(f"Timestamp check note: {e}")

if not all([CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN]):
    print("Using existing m5_bars from shared Supabase DB.")
    cur.close()
    conn.close()
    sys.exit(0)

client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)

def stop_reactor():
    try:
        cur.close()
        conn.close()
    except Exception:
        pass
    if reactor.running:
        reactor.stop()

def on_error(failure):
    print(f"cTrader API Error: {failure}")
    stop_reactor()

def on_trendbars_res(result):
    try:
        res = Protobuf.extract(result)
        bars = res.trendbar
        print(f"Received {len(bars)} M5 bars from cTrader.")

        # Fetch last 2000 timestamps as strings & datetimes so we never insert duplicates or need ON CONFLICT
        cur.execute("SELECT timestamp FROM m5_bars ORDER BY timestamp DESC LIMIT 2000;")
        existing_raw = [r[0] for r in cur.fetchall() if r[0] is not None]
        existing_dt = set()
        for val in existing_raw:
            try:
                existing_dt.add(pd.to_datetime(val, utc=True).floor("min"))
            except Exception:
                pass

        inserted = 0
        for b in bars:
            ts = datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=timezone.utc)
            if pd.to_datetime(ts, utc=True).floor("min") in existing_dt:
                continue
            low = b.low / 100000.0
            open_p = (b.low + b.deltaOpen) / 100000.0
            high_p = (b.low + b.deltaHigh) / 100000.0
            close_p = (b.low + b.deltaClose) / 100000.0
            vol = b.volume

            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S+00")
            cur.execute("""
                INSERT INTO m5_bars (timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (ts_str, open_p, high_p, low, close_p, vol))
            inserted += 1

        print(f"Inserted {inserted} new US30 M5 bars into shared Supabase DB.")
    except Exception as e:
        print(f"Database insert note: {e}")
    finally:
        stop_reactor()

def on_account_auth_res(result):
    now = datetime.now(timezone.utc)
    from_ts = int((now - timedelta(days=5)).timestamp() * 1000)
    to_ts = int(now.timestamp() * 1000)

    req = ProtoOAGetTrendbarsReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.period = ProtoOATrendbarPeriod.M5
    req.symbolId = SYMBOL_ID
    req.fromTimestamp = from_ts
    req.toTimestamp = to_ts
    d = client.send(req)
    d.addCallbacks(on_trendbars_res, on_error)

def on_app_auth_res(result):
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.accessToken = CTRADER_ACCESS_TOKEN
    d = client.send(req)
    d.addCallbacks(on_account_auth_res, on_error)

def connected(client):
    req = ProtoOAApplicationAuthReq()
    req.clientId = CTRADER_CLIENT_ID
    req.clientSecret = CTRADER_CLIENT_SECRET
    d = client.send(req)
    d.addCallbacks(on_app_auth_res, on_error)

# 20-second safety timeout so GitHub Actions NEVER hangs
reactor.callLater(20, stop_reactor)

client.setConnectedCallback(connected)
client.startService()
reactor.run()
