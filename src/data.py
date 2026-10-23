"""Fetch cTrader US30 M5 bars and live spot quote with clean connection handling."""

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *


def get_env_str(*keys, default=""):
    for key in keys:
        value = os.environ.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return default


def get_env_int(*keys, default=0):
    for key in keys:
        value = os.environ.get(key)
        if value and str(value).strip():
            try:
                return int(str(value).strip())
            except ValueError:
                pass
    return default


CLIENT_ID = get_env_str("CTRADER_CLIENT_ID", "CLIENT_ID", "CTRADER_ID", "APP_CLIENT_ID")
CLIENT_SECRET = get_env_str("CTRADER_CLIENT_SECRET", "CLIENT_SECRET", "CTRADER_SECRET", "APP_CLIENT_SECRET")
ACCESS_TOKEN = get_env_str("CTRADER_ACCESS_TOKEN", "ACCESS_TOKEN", "CTRADER_TOKEN", "TOKEN")
ACCOUNT_ID = get_env_int("CTRADER_ACCOUNT_ID", "ACCOUNT_ID", "CTID_TRADER_ACCOUNT_ID", default=0)
SYMBOL_ID = get_env_int("CTRADER_SYMBOL_ID", "SYMBOL_ID", default=1287)
EXPECTED_SYMBOL = get_env_str("CTRADER_EXPECTED_SYMBOL", default="US30").upper()
CTRADER_ENV = get_env_str("CTRADER_ENV", "CTRADER_ACCOUNT_ENV", default="live").lower()
DATA_DIR = Path(get_env_str("DATA_DIR", default="."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRICE_SCALE = 100000.0
PRICE_DIGITS = get_env_int("CTRADER_PRICE_DIGITS", default=2)
CTRADER_HOST = EndPoints.PROTOBUF_DEMO_HOST if CTRADER_ENV == "demo" else EndPoints.PROTOBUF_LIVE_HOST

client = Client(CTRADER_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
timeout_call = None
symbol_name = "#US30"
m5_done = False
quote_done = False
shutting_down = False
latest_spot_bid = None


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value):
    return value.astimezone(dt.timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def inject_live_spot_into_m5():
    """Updates the latest M5 candle with the real-time cTrader spot BID."""
    m5_path = DATA_DIR / "data_m5.csv"
    if latest_spot_bid is None or not m5_path.exists():
        return
    try:
        df = pd.read_csv(m5_path)
        if df.empty:
            return
        last_idx = df.index[-1]
        spot_val = round(float(latest_spot_bid), PRICE_DIGITS)
        df.loc[last_idx, "close"] = spot_val
        if spot_val > float(df.loc[last_idx, "high"]):
            df.loc[last_idx, "high"] = spot_val
        if spot_val < float(df.loc[last_idx, "low"]):
            df.loc[last_idx, "low"] = spot_val
        df.to_csv(m5_path, index=False)
        print(f"⚡ Synced live spot BID ({spot_val:.{PRICE_DIGITS}f}) into latest M5 bar close.")
    except Exception as e:
        print(f"Note on syncing spot into M5 CSV: {e}")


def stop_reactor_safely():
    global timeout_call, shutting_down
    shutting_down = True
    if timeout_call is not None and timeout_call.active():
        timeout_call.cancel()
    try:
        client.stopService()
    except Exception:
        pass
    if reactor.running:
        reactor.callLater(0.1, reactor.stop)


def maybe_finish():
    if m5_done and quote_done and not shutting_down:
        inject_live_spot_into_m5()
        print("✅ M5 bars and live cTrader spot price synced cleanly.")
        stop_reactor_safely()


def on_timeout():
    if not shutting_down:
        print("⚠️ WARNING: cTrader fetch timed out after 40 seconds. Proceeding with available data.")
        inject_live_spot_into_m5()
        stop_reactor_safely()


def on_fatal_error(failure):
    if not shutting_down:
        print(f"❌ cTrader API error: {failure}")
        stop_reactor_safely()
    return None


def connected(client_instance):
    request = ProtoOAApplicationAuthReq()
    request.clientId = CLIENT_ID
    request.clientSecret = CLIENT_SECRET
    deferred = client_instance.send(request)
    deferred.addCallback(on_app_auth)
    deferred.addErrback(on_fatal_error)


def on_app_auth(_result):
    request = ProtoOAAccountAuthReq()
    request.ctidTraderAccountId = ACCOUNT_ID
    request.accessToken = ACCESS_TOKEN
    deferred = client.send(request)
    deferred.addCallback(on_account_auth)
    deferred.addErrback(on_fatal_error)


def on_account_auth(_result):
    print("cTrader account authenticated. Requesting live spot & M5 bars...")
    start_market_data_requests()


def trendbar_to_row(bar):
    low_raw = int(bar.low)
    scale = PRICE_SCALE if low_raw > 100000000 else 100.0
    low = low_raw / scale
    open_price = (low_raw + int(bar.deltaOpen)) / scale
    close_price = (low_raw + int(bar.deltaClose)) / scale
    high_price = (low_raw + int(bar.deltaHigh)) / scale
    timestamp = dt.datetime.fromtimestamp(
        int(bar.utcTimestampInMinutes) * 60, tz=dt.timezone.utc
    )
    return {
        "timestamp": timestamp.isoformat(),
        "open": round(open_price, PRICE_DIGITS),
        "high": round(high_price, PRICE_DIGITS),
        "low": round(low, PRICE_DIGITS),
        "close": round(close_price, PRICE_DIGITS),
        "volume": float(bar.volume),
    }


def on_trendbars(result):
    global m5_done
    response = Protobuf.extract(result)
    rows = [trendbar_to_row(bar) for bar in response.trendbar]
    path = DATA_DIR / "data_m5.csv"

    if rows:
        df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        print(
            f"[M5] Saved {len(df):,} real M5 bars. "
            f"Latest bar open: {df['timestamp'].iloc[-1]} | "
            f"bar close: {df['close'].iloc[-1]:.{PRICE_DIGITS}f}"
        )
    else:
        print("⚠️ WARNING: No M5 bars returned by cTrader.")

    m5_done = True
    maybe_finish()
    return result


def start_market_data_requests():
    now = utc_now()
    now_ms = int(now.timestamp() * 1000)

    # 1. Subscribe to live spot events for instant BID/ASK quote
    spot_req = ProtoOASubscribeSpotsReq()
    spot_req.ctidTraderAccountId = ACCOUNT_ID
    spot_req.symbolId.append(SYMBOL_ID)
    spot_req.subscribeToSpotTimestamp = True
    d_spot = client.send(spot_req, responseTimeoutInSeconds=20)
    d_spot.addErrback(on_fatal_error)

    # 2. Request latest 5 days (~1,440 bars) of M5 candles
    bar_req = ProtoOAGetTrendbarsReq()
    bar_req.ctidTraderAccountId = ACCOUNT_ID
    bar_req.symbolId = SYMBOL_ID
    bar_req.period = ProtoOATrendbarPeriod.M5
    bar_req.fromTimestamp = int((now - dt.timedelta(days=5)).timestamp() * 1000)
    bar_req.toTimestamp = now_ms
    bar_req.count = 1800

    d_bars = client.send(bar_req, responseTimeoutInSeconds=20)
    d_bars.addCallback(on_trendbars)
    d_bars.addErrback(on_fatal_error)


def on_message_received(_client, message):
    global quote_done, latest_spot_bid
    if shutting_down:
        return
    try:
        event = Protobuf.extract(message)
    except Exception:
        return

    if not isinstance(event, ProtoOASpotEvent):
        return
    if int(getattr(event, "symbolId", -1)) != SYMBOL_ID:
        return

    bid_raw = int(getattr(event, "bid", 0) or 0)
    ask_raw = int(getattr(event, "ask", 0) or 0)
    close_raw = int(getattr(event, "sessionClose", 0) or 0)

    def scale_val(raw):
        if raw <= 0:
            return None
        s = PRICE_SCALE if raw > 100000000 else 100.0
        return round(raw / s, PRICE_DIGITS)

    bid = scale_val(bid_raw)
    ask = scale_val(ask_raw)
    session_close = scale_val(close_raw)

    if bid is not None:
        current_price, price_basis = bid, "bid"
    elif ask is not None:
        current_price, price_basis = ask, "ask_fallback"
    elif session_close is not None:
        current_price, price_basis = session_close, "session_close_fallback"
    else:
        return

    latest_spot_bid = current_price
    payload = {
        "status": "QUOTE_RECEIVED",
        "symbol": symbol_name,
        "symbol_id": SYMBOL_ID,
        "current_price": current_price,
        "price_basis": price_basis,
        "bid": bid,
        "ask": ask,
        "received_at_utc": iso_utc(utc_now()),
    }
    write_json_atomic(DATA_DIR / "latest_quote.json", payload)

    if not quote_done:
        quote_done = True
        bid_text = f"{bid:.{PRICE_DIGITS}f}" if bid is not None else "n/a"
        ask_text = f"{ask:.{PRICE_DIGITS}f}" if ask is not None else "n/a"
        print(f"[SPOT] {symbol_name} | BID={bid_text} | ASK={ask_text} | Live Price={current_price:.{PRICE_DIGITS}f}")
        maybe_finish()


def disconnected(_client, _reason):
    pass  # Suppress noisy Twisted disconnect traceback on normal exit


if __name__ == "__main__":
    missing = [
        name for name, value in [
            ("CTRADER_CLIENT_ID", CLIENT_ID),
            ("CTRADER_CLIENT_SECRET", CLIENT_SECRET),
            ("CTRADER_ACCESS_TOKEN", ACCESS_TOKEN),
            ("CTRADER_ACCOUNT_ID / ACCOUNT_ID", ACCOUNT_ID),
        ] if not value
    ]
    if missing:
        print("⚠️ Notice: Missing environment variable(s): " + ", ".join(missing))
        sys.exit(0)

    timeout_call = reactor.callLater(40, on_timeout)
    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message_received)
    client.startService()
    reactor.run()
