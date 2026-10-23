"""Fetch cTrader US30 historical bars and live spot quote, synced for pipeline execution."""

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

BAR_REQUESTS = [
    ("M5", ProtoOATrendbarPeriod.M5, 5, 1800, "data_m5.csv"),
    ("M30", ProtoOATrendbarPeriod.M30, 21, 1200, "data_30m.csv"),
    ("H4", ProtoOATrendbarPeriod.H4, 150, 1000, "data_4h.csv"),
]

PRICE_SCALE = 100000.0
PRICE_DIGITS = get_env_int("CTRADER_PRICE_DIGITS", default=2)
CTRADER_HOST = EndPoints.PROTOBUF_DEMO_HOST if CTRADER_ENV == "demo" else EndPoints.PROTOBUF_LIVE_HOST

client = Client(CTRADER_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
timeout_call = None
symbol_name = ""
bars_done = set()
quote_done = False
latest_spot_bid = None
latest_spot_ts = None
EXPECTED_BAR_LABELS = {item[0] for item in BAR_REQUESTS}


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
    """Ensures data_m5.csv has the exact live spot bid price on its latest bar."""
    m5_path = DATA_DIR / "data_m5.csv"
    if latest_spot_bid is None or not m5_path.exists():
        return
    try:
        df = pd.read_csv(m5_path)
        if df.empty:
            return
        last_idx = df.index[-1]
        df.loc[last_idx, "close"] = round(float(latest_spot_bid), PRICE_DIGITS)
        if latest_spot_bid > float(df.loc[last_idx, "high"]):
            df.loc[last_idx, "high"] = round(float(latest_spot_bid), PRICE_DIGITS)
        if latest_spot_bid < float(df.loc[last_idx, "low"]):
            df.loc[last_idx, "low"] = round(float(latest_spot_bid), PRICE_DIGITS)
        df.to_csv(m5_path, index=False)
        print(f"⚡ Synced live spot BID ({latest_spot_bid:.{PRICE_DIGITS}f}) into latest M5 bar close.")
    except Exception as e:
        print(f"Note on syncing spot into M5 CSV: {e}")


def stop_reactor_safely():
    global timeout_call
    if timeout_call is not None and timeout_call.active():
        timeout_call.cancel()
    if reactor.running:
        reactor.stop()


def maybe_finish():
    if bars_done >= EXPECTED_BAR_LABELS and quote_done:
        inject_live_spot_into_m5()
        print("\n✅ All bar requests completed and live spot price synced.")
        stop_reactor_safely()


def on_timeout():
    missing_bars = sorted(EXPECTED_BAR_LABELS - bars_done)
    print("\n⚠️ WARNING: cTrader fetch timed out after 45 seconds.")
    if missing_bars:
        print("Bar timeframe(s) not completed:", ", ".join(missing_bars))
    inject_live_spot_into_m5()
    stop_reactor_safely()


def on_fatal_error(failure):
    print(f"❌ cTrader API error: {failure}")
    stop_reactor_safely()
    return failure


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
    print("cTrader account authenticated. Verifying symbol ID...")
    request = ProtoOASymbolsListReq()
    request.ctidTraderAccountId = ACCOUNT_ID
    deferred = client.send(request)
    deferred.addCallback(on_symbols_list)
    deferred.addErrback(on_fatal_error)


def on_symbols_list(result):
    global symbol_name
    response = Protobuf.extract(result)
    symbols = getattr(response, "symbol", [])
    matches = [item for item in symbols if int(getattr(item, "symbolId", -1)) == SYMBOL_ID]

    if matches:
        symbol_name = str(getattr(matches[0], "symbolName", "") or "")
        print(f"Symbol mapping: ID {SYMBOL_ID} -> {symbol_name or '#US30'}")
    else:
        print(f"⚠️ Warning: SYMBOL_ID={SYMBOL_ID} not in light symbol list, proceeding with direct fetch.")

    request = ProtoOASymbolByIdReq()
    request.ctidTraderAccountId = ACCOUNT_ID
    request.symbolId.append(SYMBOL_ID)
    deferred = client.send(request)
    deferred.addCallback(on_symbol_details)
    deferred.addErrback(lambda _err: start_market_data_requests())


def on_symbol_details(result):
    global PRICE_DIGITS
    response = Protobuf.extract(result)
    found = [item for item in getattr(response, "symbol", []) if int(getattr(item, "symbolId", -1)) == SYMBOL_ID]
    if found:
        PRICE_DIGITS = int(getattr(found[0], "digits", PRICE_DIGITS))
    print(f"Price precision: {PRICE_DIGITS} decimal places")
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


def on_trendbars(label, filename, result):
    response = Protobuf.extract(result)
    rows = [trendbar_to_row(bar) for bar in response.trendbar]
    path = DATA_DIR / filename

    if rows:
        df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        print(
            f"[{label}] Saved {len(df):,} real {label} bars. "
            f"Latest bar open: {df['timestamp'].iloc[-1]} UTC | "
            f"bar close: {df['close'].iloc[-1]:.{PRICE_DIGITS}f}"
        )
    else:
        print(f"⚠️ WARNING: No {label} bars returned by cTrader.")

    bars_done.add(label)
    maybe_finish()
    return result


def on_bar_error(label, filename, failure):
    print(f"ERROR requesting {label} bars: {failure}")
    bars_done.add(label)
    maybe_finish()
    return None


def start_market_data_requests():
    now = utc_now()
    now_ms = int(now.timestamp() * 1000)

    # 1. Subscribe to live spot events first so the real-time tick arrives promptly
    spot_req = ProtoOASubscribeSpotsReq()
    spot_req.ctidTraderAccountId = ACCOUNT_ID
    spot_req.symbolId.append(SYMBOL_ID)
    spot_req.subscribeToSpotTimestamp = True
    d_spot = client.send(spot_req, responseTimeoutInSeconds=20)
    d_spot.addErrback(on_fatal_error)

    # 2. Request M5, M30, and H4 historical bars
    for label, period, days, count, filename in BAR_REQUESTS:
        request = ProtoOAGetTrendbarsReq()
        request.ctidTraderAccountId = ACCOUNT_ID
        request.symbolId = SYMBOL_ID
        request.period = period
        request.fromTimestamp = int((now - dt.timedelta(days=days)).timestamp() * 1000)
        request.toTimestamp = now_ms
        request.count = count

        deferred = client.send(request, responseTimeoutInSeconds=20)
        deferred.addCallback(
            lambda result, _label=label, _filename=filename: on_trendbars(_label, _filename, result)
        )
        deferred.addErrback(
            lambda failure, _label=label, _filename=filename: on_bar_error(_label, _filename, failure)
        )


def spot_time_to_utc(raw_timestamp):
    if not raw_timestamp:
        return None
    value = int(raw_timestamp)
    seconds = value / 1000.0 if value > 100_000_000_000 else float(value)
    try:
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def on_message_received(_client, message):
    global quote_done, latest_spot_bid, latest_spot_ts
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

    received_at = utc_now()
    spot_at = spot_time_to_utc(getattr(event, "timestamp", 0))
    age_seconds = round((received_at - spot_at).total_seconds(), 1) if spot_at else None

    latest_spot_bid = current_price
    latest_spot_ts = spot_at or received_at

    payload = {
        "status": "QUOTE_RECEIVED" if price_basis in ("bid", "ask_fallback") else "SESSION_CLOSE_FALLBACK",
        "is_stale": age_seconds is not None and age_seconds > 120,
        "symbol": symbol_name or EXPECTED_SYMBOL,
        "symbol_id": SYMBOL_ID,
        "current_price": current_price,
        "price_basis": price_basis,
        "bid": bid,
        "ask": ask,
        "mid": round((bid + ask) / 2.0, PRICE_DIGITS) if bid is not None and ask is not None else None,
        "session_close": session_close,
        "source": "ctrader_open_api_spot_event",
        "received_at_utc": iso_utc(received_at),
        "spot_timestamp_utc": iso_utc(spot_at) if spot_at is not None else None,
        "spot_age_seconds": age_seconds,
    }
    write_json_atomic(DATA_DIR / "latest_quote.json", payload)

    if not quote_done:
        quote_done = True
        bid_text = f"{bid:.{PRICE_DIGITS}f}" if bid is not None else "n/a"
        ask_text = f"{ask:.{PRICE_DIGITS}f}" if ask is not None else "n/a"
        print(
            f"[SPOT] {payload['symbol']} | BID={bid_text} | ASK={ask_text} | "
            f"dashboard current_price={current_price:.{PRICE_DIGITS}f} ({price_basis})"
        )
        maybe_finish()


def disconnected(_client, reason):
    print(f"cTrader disconnected: {reason}")


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

    write_json_atomic(
        DATA_DIR / "latest_quote.json",
        {
            "symbol": EXPECTED_SYMBOL,
            "symbol_id": SYMBOL_ID,
            "current_price": None,
            "status": "PENDING",
            "received_at_utc": iso_utc(utc_now()),
            "is_stale": True,
        },
    )

    timeout_call = reactor.callLater(45, on_timeout)
    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message_received)
    client.startService()
    reactor.run()
