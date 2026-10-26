"""Fetch cTrader US30 M5 bars and live spot quote with hardcoded credentials."""

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

# ==========================================
# HARDCODED CTRADER CREDENTIALS & CONFIG
# ==========================================
CLIENT_ID = "40052_sFwZLFliw9QV9nZ5jODc1hClaf2IfOn6f3vXYRzgY4webzpn1k"
CLIENT_SECRET = "658cBabOPEpzrjV5zknPetyoEJcizdvg3hQcA8nlZDI29WT2nu"
ACCESS_TOKEN = "k-KgV20T1eC0taKCQFH6wlJPU2jCyv-vl4QryAQXNz4"
ACCOUNT_ID = 40052
ENV_SYMBOL_ID = 1287  # US30 Symbol ID
CTRADER_ENV = "live"
# ==========================================

DATA_DIR = Path(".")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRICE_DIGITS = 2
CTRADER_HOST = EndPoints.PROTOBUF_LIVE_HOST

client = Client(CTRADER_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
timeout_call = None
spot_wait_call = None
symbol_name = "#US30"
resolved_symbol_id = ENV_SYMBOL_ID
m5_done = False
quote_done = False
shutting_down = False
latest_m5_close = None


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value):
    return value.astimezone(dt.timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def stop_reactor_safely():
    global timeout_call, spot_wait_call, shutting_down
    shutting_down = True
    if timeout_call is not None and timeout_call.active():
        timeout_call.cancel()
    if spot_wait_call is not None and spot_wait_call.active():
        spot_wait_call.cancel()
    try:
        client.stopService()
    except Exception:
        pass
    if reactor.running:
        reactor.callLater(0.1, reactor.stop)


def maybe_finish():
    if m5_done and quote_done and not shutting_down:
        print("✅ US30 closed M5 bars and live cTrader spot quote synced cleanly.")
        stop_reactor_safely()


def on_spot_wait_timeout():
    global quote_done
    if not shutting_down and m5_done and not quote_done:
        if latest_m5_close is not None:
            payload = {
                "status": "QUOTE_RECEIVED",
                "symbol": symbol_name,
                "symbol_id": resolved_symbol_id,
                "current_price": latest_m5_close,
                "price_basis": "m5_close_fallback",
                "bid": latest_m5_close,
                "ask": latest_m5_close,
                "received_at_utc": iso_utc(utc_now()),
            }
            write_json_atomic(DATA_DIR / "latest_quote.json", payload)
            print(f"ℹ️ Using latest M5 bar close ({latest_m5_close:.{PRICE_DIGITS}f}) as live price.")
        quote_done = True
        stop_reactor_safely()


def on_timeout():
    if not shutting_down:
        print("⚠️ WARNING: cTrader fetch timed out after 35 seconds. Proceeding with available data.")
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
    print("cTrader account authenticated successfully. Requesting symbol list...")
    req = ProtoOASymbolsListReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    deferred = client.send(req, responseTimeoutInSeconds=15)
    deferred.addCallback(on_symbols_list)
    deferred.addErrback(on_symbols_list_error)


def on_symbols_list_error(failure):
    print(f"⚠️ Could not fetch symbol list ({failure}). Falling back to configured Symbol ID: {ENV_SYMBOL_ID}")
    start_market_data_requests()
    return None


def on_symbols_list(result):
    global resolved_symbol_id
    response = Protobuf.extract(result)
    if isinstance(response, ProtoOAErrorRes):
        print(f"⚠️ Symbol list error response. Falling back to ID {ENV_SYMBOL_ID}")
        start_market_data_requests()
        return result

    found = False
    for sym in getattr(response, "symbol", []):
        name = getattr(sym, "symbolName", "").upper()
        if "US30" in name or "DOW" in name:
            resolved_symbol_id = int(sym.symbolId)
            print(f"🎯 Discovered US30 Symbol Name: '{sym.symbolName}' -> Resolved Symbol ID: {resolved_symbol_id}")
            found = True
            break

    if not found:
        print(f"ℹ️ US30 not found in symbol list by name. Using configured ID: {ENV_SYMBOL_ID}")

    start_market_data_requests()
    return result


def trendbar_to_row(bar):
    low_raw = int(bar.low)
    scale = 100000.0 if low_raw > 100000000 else 100.0
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
    global m5_done, spot_wait_call, latest_m5_close
    try:
        response = Protobuf.extract(result)
    except Exception as e:
        print(f"❌ Failed to extract protobuf message: {e}")
        m5_done = True
        maybe_finish()
        return result

    if isinstance(response, ProtoOAErrorRes):
        err_code = getattr(response, "errorCode", "UNKNOWN")
        err_desc = getattr(response, "description", "No description provided")
        print(f"❌ cTrader Error Response from Server: Code=[{err_code}] Description=[{err_desc}]")
        m5_done = True
        maybe_finish()
        return result

    rows = [trendbar_to_row(bar) for bar in getattr(response, "trendbar", [])]
    path = DATA_DIR / "data_m5.csv"

    if rows:
        df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        latest_m5_close = float(df["close"].iloc[-1])
        print(
            f"[M5] Saved {len(df):,} pure closed M5 bars. "
            f"Latest bar: {df['timestamp'].iloc[-1]} | "
            f"Close: {latest_m5_close:.{PRICE_DIGITS}f}"
        )
    else:
        print("⚠️ WARNING: No M5 bars returned by cTrader for Symbol ID:", resolved_symbol_id)

    m5_done = True

    if not quote_done and not shutting_down:
        spot_req = ProtoOASubscribeSpotsReq()
        spot_req.ctidTraderAccountId = ACCOUNT_ID
        spot_req.symbolId.append(resolved_symbol_id)
        spot_req.subscribeToSpotTimestamp = True
        d_spot = client.send(spot_req, responseTimeoutInSeconds=15)
        d_spot.addErrback(on_fatal_error)
        spot_wait_call = reactor.callLater(5.0, on_spot_wait_timeout)

    maybe_finish()
    return result


def start_market_data_requests():
    now = utc_now()
    now_ms = int(now.timestamp() * 1000)

    lookback_hours = 84 if now.weekday() in (6, 0) else 36

    bar_req = ProtoOAGetTrendbarsReq()
    bar_req.ctidTraderAccountId = ACCOUNT_ID
    bar_req.symbolId = resolved_symbol_id
    bar_req.period = ProtoOATrendbarPeriod.M5
    bar_req.fromTimestamp = int((now - dt.timedelta(hours=lookback_hours)).timestamp() * 1000)
    bar_req.toTimestamp = now_ms

    print(f"Requesting M5 bars for Symbol ID {resolved_symbol_id}...")
    d_bars = client.send(bar_req, responseTimeoutInSeconds=20)
    d_bars.addCallback(on_trendbars)
    d_bars.addErrback(on_fatal_error)


def on_message_received(_client, message):
    global quote_done
    if shutting_down:
        return
    try:
        event = Protobuf.extract(message)
    except Exception:
        return

    if isinstance(event, ProtoOAErrorRes):
        print(f"❌ cTrader Spot Error: {getattr(event, 'description', 'Unknown error')} (Code: {getattr(event, 'errorCode', 'N/A')})")
        return

    if not isinstance(event, ProtoOASpotEvent):
        return
    if int(getattr(event, "symbolId", -1)) != resolved_symbol_id:
        return

    bid_raw = int(getattr(event, "bid", 0) or 0)
    ask_raw = int(getattr(event, "ask", 0) or 0)

    def scale_val(raw):
        if raw <= 0:
            return None
        scale = 100000.0 if raw > 100000000 else 100.0
        return round(raw / scale, PRICE_DIGITS)

    bid = scale_val(bid_raw)
    ask = scale_val(ask_raw)

    if bid is not None and ask is not None:
        current_price, price_basis = round((bid + ask) / 2.0, PRICE_DIGITS), "mid_bid_ask"
    elif bid is not None:
        current_price, price_basis = bid, "bid"
    elif ask is not None:
        current_price, price_basis = ask, "ask"
    else:
        return

    payload = {
        "status": "QUOTE_RECEIVED",
        "symbol": symbol_name,
        "symbol_id": resolved_symbol_id,
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
    pass


if __name__ == "__main__":
    print("DEBUG -> Hardcoded cTrader credentials loaded for Account ID:", ACCOUNT_ID)

    write_json_atomic(
        DATA_DIR / "latest_quote.json",
        {
            "status": "PENDING",
            "symbol": symbol_name,
            "symbol_id": ENV_SYMBOL_ID,
            "current_price": None,
            "received_at_utc": iso_utc(utc_now()),
        },
    )

    timeout_call = reactor.callLater(35, on_timeout)
    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message_received)
    client.startService()
    reactor.run()
