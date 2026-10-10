import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from twisted.internet import reactor

from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# Import our custom modules
from src.config import (
    CTRADER_CLIENT_ID, 
    CTRADER_CLIENT_SECRET, 
    CTRADER_ACCESS_TOKEN,
    ACCOUNT_ID,
    SYMBOL_ID
)
from src.database import get_latest_timestamp, save_new_bars

# ============================================================
# 1. SETUP TARGET TIME & MISSING DATA RANGE
# ============================================================

now_utc = datetime.now(timezone.utc)
target_minute = (now_utc.minute // 5) * 5
target_ts = now_utc.replace(minute=target_minute, second=0, microsecond=0)

last_ts = get_latest_timestamp("m5_bars")

if not last_ts:
    print("Database is empty or table missing. Defaulting to 7 days of history.")
    last_ts = target_ts - timedelta(days=7)

FIVE_MIN_MS = 5 * 60 * 1000
CHUNK_DAYS = 7
CHUNK_MS = CHUNK_DAYS * 24 * 60 * 60 * 1000

FROM = int(last_ts.timestamp() * 1000) + FIVE_MIN_MS
TO = int(target_ts.timestamp() * 1000)

if FROM >= TO:
    print("✓ SUPABASE DATABASE IS ALREADY UP TO DATE.")
    # Exit cleanly without starting cTrader
    import sys
    sys.exit(0)

print(f"Download start : {datetime.fromtimestamp(FROM / 1000, tz=timezone.utc)}")
print(f"Download target: {datetime.fromtimestamp(TO / 1000, tz=timezone.utc)}")

# ============================================================
# 2. CTRADER AUTOMATION SETUP
# ============================================================

current_from = FROM
total_inserted = 0
download_error = None

client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)

def stop_client():
    try:
        client.stopService()
    except Exception:
        pass
    if reactor.running:
        reactor.stop()

def send_next_request():
    global current_from
    if current_from >= TO:
        stop_client()
        return

    chunk_to = min(current_from + CHUNK_MS, TO)
    req = ProtoOAGetTrendbarsReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.symbolId = SYMBOL_ID
    req.period = ProtoOATrendbarPeriod.M5
    req.fromTimestamp = current_from
    req.toTimestamp = chunk_to
    client.send(req)

def connected(client):
    req = ProtoOAApplicationAuthReq()
    req.clientId = CTRADER_CLIENT_ID
    req.clientSecret = CTRADER_CLIENT_SECRET
    client.send(req)

def received(client, message):
    global current_from, total_inserted, download_error

    # Application Auth -> Account Auth
    if message.payloadType == ProtoOAApplicationAuthRes().payloadType:
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.accessToken = CTRADER_ACCESS_TOKEN
        client.send(req)
        return

    # Account Auth -> Start Requests
    if message.payloadType == ProtoOAAccountAuthRes().payloadType:
        send_next_request()
        return

    # Error Handling
    if message.payloadType == ProtoOAErrorRes().payloadType:
        res = Protobuf.extract(message)
        download_error = f"{getattr(res, 'errorCode', 'UNKNOWN')}: {getattr(res, 'description', '')}"
        print(f"!!! API ERROR: {download_error}")
        stop_client()
        return

    # Historical Bars Handling
    if message.payloadType == ProtoOAGetTrendbarsRes().payloadType:
        res = Protobuf.extract(message)
        trendbars = res.trendbar
        
        if getattr(res, "hasMore", False):
            print("!!! RESPONSE LIMIT DETECTED !!!")
            stop_client()
            return

        chunk_rows = []
        for b in trendbars:
            low_price = b.low / 100000
            open_price = (b.low + b.deltaOpen) / 100000
            high_price = (b.low + b.deltaHigh) / 100000
            close_price = (b.low + b.deltaClose) / 100000
            timestamp = datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=timezone.utc)
            
            chunk_rows.append([timestamp, open_price, high_price, low_price, close_price, int(b.volume)])

        if chunk_rows:
            df = pd.DataFrame(chunk_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
            
            # Save directly to Supabase via our database.py helper
            save_new_bars(df)
            total_inserted += len(df)

        chunk_to = min(current_from + CHUNK_MS, TO)
        current_from = chunk_to + FIVE_MIN_MS
        
        if current_from > TO:
            current_from = TO
            
        send_next_request()

# ============================================================
# 3. RUN REACTOR
# ============================================================

client.setConnectedCallback(connected)
client.setMessageReceivedCallback(received)

print("Connecting to cTrader...")
try:
    client.startService()
    reactor.run()
except Exception as e:
    print(f"Connection error: {e}")

print(f"✓ AUTOMATED UPDATE COMPLETE. Total new bars: {total_inserted}")
