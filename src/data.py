import os
import sys
import datetime
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
SYMBOL_ID = int(os.environ.get("SYMBOL_ID") or 1287)  # US30 Symbol ID = 1287

client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
timeout_call = None

def stop_reactor_safely():
    global timeout_call
    if timeout_call and timeout_call.active():
        timeout_call.cancel()
    if reactor.running:
        reactor.stop()

def on_timeout():
    print("⚠️ cTrader API request timed out after 45 seconds. Stopping reactor.")
    if reactor.running:
        reactor.stop()

def on_error(failure):
    print(f"❌ cTrader API Error: {failure}")
    stop_reactor_safely()

def get_last_closed_m5_boundary(now_utc):
    """Returns the exact UTC timestamp of the most recent fully closed M5 candle boundary."""
    floored_minute = (now_utc.minute // 5) * 5
    return now_utc.replace(minute=floored_minute, second=0, microsecond=0)

def on_trendbars(result):
    res = Protobuf.extract(result)
    rows = []

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    last_closed_boundary = get_last_closed_m5_boundary(now_utc)

    for b in res.trendbar:
        # Auto-detect cTrader scaling (100000.0 for Open API v2 standard, or 100.0 for legacy/digits)
        scale = 100000.0 if b.low > 100000000 else 100.0

        low = b.low / scale
        open_p = (b.low + b.deltaOpen) / scale
        close_p = (b.low + b.deltaClose) / scale
        high_p = (b.low + b.deltaHigh) / scale
        ts = datetime.datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=datetime.timezone.utc)

        # CRITICAL FIX: Drop any currently forming (unclosed) M5 candle.
        # An M5 bar opening at `ts` only closes at `ts + 5 minutes`.
        if ts + datetime.timedelta(minutes=5) > last_closed_boundary:
            continue

        rows.append({
            "timestamp": ts.isoformat(),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low, 2),
            "close": round(close_p, 2),
            "volume": float(b.volume)
        })

    if not rows:
        print("⚠️ No new closed M5 bars returned from cTrader (market may be closed). Keeping existing data_m5.csv.")
        stop_reactor_safely()
        return

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df.to_csv("data_m5.csv", index=False)
    # Keep legacy copies in sync in case external scripts reference them
    df.to_csv("data_30m.csv", index=False)
    df.to_csv("data_4h.csv", index=False)

    print(
        f"[M5] Saved {len(df):,} closed 5-min bars "
        f"(Latest bar: {df['timestamp'].iloc[-1]} | Close: {df['close'].iloc[-1]:.2f})"
    )
    stop_reactor_safely()

def on_account_auth(result):
    print("cTrader account authenticated. Pulling closed US30 M5 bars...")
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    last_closed_boundary = get_last_closed_m5_boundary(now_utc)

    # Fetch 14 days of closed M5 bars (~2,800 bars) up to the last closed 5-minute boundary
    from_ts = int((last_closed_boundary - datetime.timedelta(days=14)).timestamp() * 1000)
    to_ts = int(last_closed_boundary.timestamp() * 1000)

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
    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
        print("❌ Missing cTrader credentials in environment variables.")
        sys.exit(1)

    # 45-second watchdog prevents hanging GitHub Actions jobs if cTrader socket stalls
    timeout_call = reactor.callLater(45, on_timeout)
    client.setConnectedCallback(connected)
    client.startService()
    reactor.run()
