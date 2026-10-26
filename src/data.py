"""Fetch cTrader US30 M5 bars and live spot quote with auto Account ID & Symbol ID resolution."""

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


# Built-in fallbacks from your cTrader Open API Read-Only credentials & US30 Symbol ID (1287)
DEFAULT_CLIENT_ID = "40052_sFwZLFliw9QV9nZ5jODc1hClaf2IfOn6f3vXYRzgY4wEbzpn1k"
DEFAULT_CLIENT_SECRET = "658cBabOPePzrjV5zknPetyoJECizdvg3hQcA8nlZDI29WT2nu"
DEFAULT_ACCESS_TOKEN = "k-KgV20T1eC0taKCQFH6wlJPU2jCyv-vl4QryAQXKz4"
DEFAULT_SYMBOL_ID = 1287

CLIENT_ID = get_env_str("CTRADER_CLIENT_ID", "CLIENT_ID", "CTRADER_ID", "APP_CLIENT_ID", default=DEFAULT_CLIENT_ID)
CLIENT_SECRET = get_env_str("CTRADER_CLIENT_SECRET", "CLIENT_SECRET", "CTRADER_SECRET", "APP_CLIENT_SECRET", default=DEFAULT_CLIENT_SECRET)
ACCESS_TOKEN = get_env_str("CTRADER_ACCESS_TOKEN", "ACCESS_TOKEN", "CTRADER_TOKEN", "TOKEN", default=DEFAULT_ACCESS_TOKEN)
ACCOUNT_ID = get_env_int("ACCOUNT_ID", "CTRADER_ACCOUNT_ID", "CTID_TRADER_ACCOUNT_ID", default=0)
SYMBOL_ID = get_env_int("SYMBOL_ID", "CTRADER_SYMBOL_ID", default=DEFAULT_SYMBOL_ID)
EXPECTED_SYMBOL = get_env_str("CTRADER_EXPECTED_SYMBOL", default="US30").upper()
CTRADER_ENV = get_env_str("CTRADER_ENV", "CTRADER_ACCOUNT_ENV", default="live").lower()
DATA_DIR = Path(get_env_str("DATA_DIR", default="."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRICE_DIGITS = get_env_int("CTRADER_PRICE_DIGITS", default=2)
CTRADER_HOST = EndPoints.PROTOBUF_DEMO_HOST if CTRADER_ENV == "demo" else EndPoints.PROTOBUF_LIVE_HOST

client = Client(CTRADER_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
timeout_call = None
spot_wait_call = None
symbol_name = "#US30"
m5_done = False
quote_done = False
shutting_down = False
retried_symbol_lookup = False
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
                "symbol_id": SYMBOL_ID,
                "current_price": latest_m5_close,
                "price_basis": "m5_close_fallback",
                "bid": latest_m5_close,
                "ask": latest_m5_close,
                "received_at_utc": iso_utc(utc_now()),
            }
            write_json_atomic(DATA_DIR / "latest_quote.json", payload)
            print(f"ℹ️ No new spot tick arrived within 6s. Using latest M5 bar close ({latest_m5_close:.{PRICE_DIGITS}f}).")
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
    if ACCOUNT_ID > 0:
        authenticate_trading_account(ACCOUNT_ID)
    else:
        print("🔍 ACCOUNT_ID not set in env. Auto-discovering trading account from Access Token...")
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = ACCESS_TOKEN
        deferred = client.send(req, responseTimeoutInSeconds=20)
        deferred.addCallback(on_account_list)
        deferred.addErrback(on_fatal_error)


def on_account_list(result):
    global ACCOUNT_ID
    response = Protobuf.extract(result)
    accounts = list(getattr(response, "ctidTraderAccount", []))
    if not accounts:
        print("❌ ERROR: No trading accounts found for the provided Access Token.")
        stop_reactor_safely()
        return

    # Prefer live account when on live host, otherwise pick the first available account
    chosen = None
    want_live = CTRADER_ENV != "demo"
    for acc in accounts:
        if bool(getattr(acc, "isLive", False)) == want_live:
            chosen = acc
            break
    if chosen is None:
        chosen = accounts[0]

    ACCOUNT_ID = int(chosen.ctidTraderAccountId)
    is_live_str = "LIVE" if getattr(chosen, "isLive", False) else "DEMO"
    print(f"✅ Auto-discovered cTrader Account ID: {ACCOUNT_ID} ({is_live_str})")
    authenticate_trading_account(ACCOUNT_ID)


def authenticate_trading_account(acc_id):
    request = ProtoOAAccountAuthReq()
    request.ctidTraderAccountId = int(acc_id)
    request.accessToken = ACCESS_TOKEN
    deferred = client.send(request, responseTimeoutInSeconds=20)
    deferred.addCallback(on_account_auth)
    deferred.addErrback(on_fatal_error)


def on_account_auth(_result):
    print(f"cTrader account {ACCOUNT_ID} authenticated. Requesting live spot & M5 bars for Symbol ID {SYMBOL_ID}...")
    start_market_data_requests()


def lookup_us30_symbol():
    req = ProtoOASymbolsListReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    d = client.send(req, responseTimeoutInSeconds=20)
    d.addCallback(on_symbols_list)
    d.addErrback(on_fatal_error)


def on_symbols_list(result):
    global SYMBOL_ID, symbol_name
    response = Protobuf.extract(result)
    symbols = list(getattr(response, "symbol", []))

    preferred = ["#US30", "US30", "US30.CASH", "WS30", "DJ30", "USA30"]
    chosen = None
    for target in preferred:
        for s in symbols:
            if str(getattr(s, "symbolName", "")).upper().strip() == target:
                chosen = s
                break
        if chosen:
            break

    if not chosen:
        for s in symbols:
            sname = str(getattr(s, "symbolName", "")).upper()
            if any(k in sname for k in ["US30", "WS30", "DJ30", "DOW"]):
                chosen = s
                break

    if not chosen:
        print("❌ Could not find #US30 symbol in cTrader account.")
        stop_reactor_safely()
        return

    SYMBOL_ID = int(chosen.symbolId)
    symbol_name = str(chosen.symbolName)
    print(f"✅ Auto-resolved broker symbol '{symbol_name}' with SYMBOL_ID={SYMBOL_ID}")
    start_market_data_requests()


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
    global m5_done, spot_wait_call, retried_symbol_lookup, latest_m5_close
    response = Protobuf.extract(result)
    rows = [trendbar_to_row(bar) for bar in response.trendbar]
    path = DATA_DIR / "data_m5.csv"

    if not rows and not retried_symbol_lookup:
        retried_symbol_lookup = True
        print(f"⚠️ No M5 bars returned for SYMBOL_ID={SYMBOL_ID}. Auto-resolving #US30 Symbol ID...")
        lookup_us30_symbol()
        return result

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
        print("⚠️ WARNING: No M5 bars returned by cTrader.")

    m5_done = True
    if not quote_done and not shutting_down:
        spot_wait_call = reactor.callLater(6.0, on_spot_wait_timeout)
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
    global quote_done
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
        scale = 100000.0 if raw > 100000000 else 100.0
        return round(raw / scale, PRICE_DIGITS)

    bid = scale_val(bid_raw)
    ask = scale_val(ask_raw)
    session_close = scale_val(close_raw)

    if bid is not None:
        current_price, price_basis = bid, "bid"
    elif ask is not None:
        current_price, price_basis = ask, "ask_fallback"
    elif session_close is not None and latest_m5_close is None:
        current_price, price_basis = session_close, "session_close_fallback"
    else:
        return

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
    pass


if __name__ == "__main__":
    write_json_atomic(
        DATA_DIR / "latest_quote.json",
        {
            "status": "PENDING",
            "symbol": symbol_name,
            "symbol_id": SYMBOL_ID,
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
