import os
import sys
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
ENV_SYMBOL_ID = os.environ.get("SYMBOL_ID")

client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
resolved_symbol_id = int(ENV_SYMBOL_ID) if (ENV_SYMBOL_ID and ENV_SYMBOL_ID.isdigit()) else 1


def save_and_sync_supabase(df):
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
                if not df.empty:
                    tuples = [
                        (
                            r["timestamp"].isoformat(),
                            float(r["open"]),
                            float(r["high"]),
                            float(r["low"]),
                            float(r["close"]),
                            float(r["volume"]),
                        )
                        for _, r in df.iterrows()
                    ]
                    cur.executemany(
                        """
                        INSERT INTO eurusd_features_m5 (timestamp, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp) DO UPDATE SET
                            high = GREATEST(eurusd_features_m5.high, EXCLUDED.high),
                            low = LEAST(eurusd_features_m5.low, EXCLUDED.low),
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume;
                        """,
                        tuples,
                    )

                cur.execute(
                    "SELECT timestamp, open, high, low, close, volume FROM eurusd_features_m5 ORDER BY timestamp ASC"
                )
                full_history = cur.fetchall()
                if full_history:
                    df = pd.DataFrame(
                        full_history,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
            conn.commit()

    if df.empty:
        print("ERROR: No EUR/USD bars returned from cTrader or Supabase.")
        sys.exit(1)

    df.to_csv("data_m5.csv", index=False)
    df.to_csv("data_30m.csv", index=False)
    df.to_csv("data_4h.csv", index=False)
    print(f"[M5] Saved {len(df):,} total 5-min bars (Latest close: {df['close'].iloc[-1]:.4f})")


def on_error(failure):
    print(f"cTrader API Error: {failure}")
    if reactor.running:
        reactor.stop()


def on_trendbars(result):
    res = Protobuf.extract(result)

    # Check if cTrader returned an API error message instead of trendbars
    if hasattr(res, "errorCode") and res.errorCode:
        print(f"cTrader Trendbar Error [{res.errorCode}]: {getattr(res, 'description', '')}")
        save_and_sync_supabase(pd.DataFrame())
        if reactor.running:
            reactor.stop()
        return

    bars = getattr(res, "trendbar", [])
    rows = []

    for b in bars:
        if b.low <= 0:
            continue
        low = b.low / 100000.0
        open_p = (b.low + b.deltaOpen) / 100000.0
        close_p = (b.low + b.deltaClose) / 100000.0
        high_p = (b.low + b.deltaHigh) / 100000.0
        ts = datetime.datetime.fromtimestamp(
            b.utcTimestampInMinutes * 60, tz=datetime.timezone.utc
        )
        rows.append(
            {
                "timestamp": ts,
                "open": round(open_p, 5),
                "high": round(high_p, 5),
                "low": round(low, 5),
                "close": round(close_p, 5),
                "volume": b.volume,
            }
        )

    df = (
        pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
    save_and_sync_supabase(df)

    if reactor.running:
        reactor.stop()


def request_trendbars(symbol_id):
    now = datetime.datetime.now(datetime.timezone.utc)
    # 10 days of M5 bars (~2,000 bars) safely fits inside cTrader's single-call limit
    from_ts = int((now - datetime.timedelta(days=10)).timestamp() * 1000)
    to_ts = int(now.timestamp() * 1000)

    req = ProtoOAGetTrendbarsReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.symbolId = int(symbol_id)
    req.period = ProtoOATrendbarPeriod.M5
    req.fromTimestamp = from_ts
    req.toTimestamp = to_ts

    d = client.send(req)
    d.addCallback(on_trendbars)
    d.addErrback(on_error)


def on_symbols_list(result):
    global resolved_symbol_id
    res = Protobuf.extract(result)

    if hasattr(res, "symbol"):
        # Look for exact EURUSD match first, then any symbol containing EURUSD
        exact_match = None
        partial_match = None
        for s in res.symbol:
            name_clean = s.symbolName.upper().replace("/", "").replace(".", "").strip()
            if name_clean == "EURUSD":
                exact_match = (s.symbolId, s.symbolName)
                break
            elif "EURUSD" in name_clean and partial_match is None:
                partial_match = (s.symbolId, s.symbolName)

        chosen = exact_match or partial_match
        if chosen:
            resolved_symbol_id = chosen[0]
            print(f"Auto-detected EUR/USD Symbol: '{chosen[1]}' -> Symbol ID = {resolved_symbol_id}")
        else:
            print(f"Could not find EURUSD by name; using default Symbol ID = {resolved_symbol_id}")

    request_trendbars(resolved_symbol_id)


def on_account_auth(result):
    res = Protobuf.extract(result)
    if hasattr(res, "errorCode") and res.errorCode:
        print(f"cTrader Account Auth Error [{res.errorCode}]: {getattr(res, 'description', '')}")
        if reactor.running:
            reactor.stop()
        return

    print("cTrader account authenticated. Detecting EUR/USD Symbol ID...")
    req = ProtoOASymbolsListReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    d = client.send(req)
    d.addCallback(on_symbols_list)
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
