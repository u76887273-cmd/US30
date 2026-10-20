import os
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

def on_error(failure):
    print(f"cTrader API Error: {failure}")
    if reactor.running:
        reactor.stop()

def on_trendbars(result):
    res = Protobuf.extract(result)
    rows = []

    for b in res.trendbar:
        # Auto-detect cTrader scaling (100000.0 for Open API v2 standard, or 100.0 for legacy/digits)
        scale = 100000.0 if b.low > 100000000 else 100.0

        low = b.low / scale
        open_p = (b.low + b.deltaOpen) / scale
        close_p = (b.low + b.deltaClose) / scale
        high_p = (b.low + b.deltaHigh) / scale
        ts = datetime.datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=datetime.timezone.utc)
        rows.append({
            "timestamp": ts,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low, 2),
            "close": round(close_p, 2),
            "volume": b.volume
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # Save to both data_30m.csv and data_4h.csv so validator.py works without changes
    df.to_csv("data_m5.csv", index=False)
    df.to_csv("data_30m.csv", index=False)
    df.to_csv("data_4h.csv", index=False)
    print(f"[M5] Saved {len(df)} 5-min bars (Latest close: {df['close'].iloc[-1]:.2f})")

    if reactor.running:
        reactor.stop()

def on_account_auth(result):
    print("cTrader account authenticated. Pulling US30 M5 bars...")
    now = datetime.datetime.now(datetime.timezone.utc)
    # Fetch 14 days of M5 bars (~2,800 bars)
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
