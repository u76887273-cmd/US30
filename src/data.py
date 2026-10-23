"""Fetch cTrader US30 historical bars and a spot quote.

Outputs (in DATA_DIR, default current directory):
  data_m5.csv      True M5 historical bars
  data_30m.csv     True M30 historical bars
  data_4h.csv      True H4 historical bars
  latest_quote.json Latest bid/ask quote (or explicitly-labelled close fallback)

Required environment variables:
  CTRADER_CLIENT_ID
  CTRADER_CLIENT_SECRET
  CTRADER_ACCESS_TOKEN
  CTRADER_ACCOUNT_ID

Optional:
  CTRADER_SYMBOL_ID          defaults to 1287 (VERIFY for your FxPro account)
  CTRADER_EXPECTED_SYMBOL    defaults to US30; matched against the cTrader symbol list
  DATA_DIR                   defaults to current directory
"""

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *


# ---------- Environment/configuration ----------

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

# Conservative date ranges to stay below common historical-response limits.
# The cTrader symbol list is checked before fetching data so a wrong symbol ID
# will not silently become the model's US30 price feed.
BAR_REQUESTS = [
    ("M5", ProtoOATrendbarPeriod.M5, 5, 1800, "data_m5.csv"),
    ("M30", ProtoOATrendbarPeriod.M30, 21, 1200, "data_30m.csv"),
    ("H4", ProtoOATrendbarPeriod.H4, 150, 1000, "data_4h.csv"),
]
PRICE_SCALE = 100000.0  # cTrader Open API relative prices use 1/100,000 units.
PRICE_DIGITS = get_env_int("CTRADER_PRICE_DIGITS", default=2)

if CTRADER_ENV == "demo":
    CTRADER_HOST = EndPoints.PROTOBUF_DEMO_HOST
elif CTRADER_ENV == "live":
    CTRADER_HOST = EndPoints.PROTOBUF_LIVE_HOST
else:
    CTRADER_HOST = EndPoints.PROTOBUF_LIVE_HOST  # Invalid values are rejected in main().

client = Client(CTRADER_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
timeout_call = None
symbol_name = ""
bars_done = set()
quote_done = False
EXPECTED_BAR_LABELS = {item[0] for item in BAR_REQUESTS}


# ---------- Small utilities ----------

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
    global timeout_call
    if timeout_call is not None and timeout_call.active():
        timeout_call.cancel()
    if reactor.running:
        reactor.stop()


def maybe_finish():
    if bars_done >= EXPECTED_BAR_LABELS and quote_done:
        print("\nAll bar requests completed and a spot price was received.")
        stop_reactor_safely()


def on_timeout():
    missing_bars = sorted(EXPECTED_BAR_LABELS - bars_done)
    print("\nWARNING: cTrader fetch timed out after 45 seconds.")
    if missing_bars:
        print("Bar timeframe(s) not completed:", ", ".join(missing_bars))
    if not quote_done:
        print("No spot quote was received; latest_quote.json remains marked PENDING if no valid quote was written.")
    stop_reactor_safely()


def on_fatal_error(failure):
    print(f"cTrader API error: {failure}")
    stop_reactor_safely()
    return failure


# ---------- Authentication and symbol identity ----------

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

    if not matches:
        print(f"ERROR: SYMBOL_ID={SYMBOL_ID} was not found in this account's cTrader symbol list.")
        print("Find the US30 symbol ID from the same FxPro cTrader account and set CTRADER_SYMBOL_ID.")
        stop_reactor_safely()
        return

    symbol_name = str(getattr(matches[0], "symbolName", "") or "")
    print(f"Symbol mapping: ID {SYMBOL_ID} -> {symbol_name or '(name not supplied by API)'}")

    if symbol_name and EXPECTED_SYMBOL not in symbol_name.upper():
        print(
            f"ERROR: ID {SYMBOL_ID} maps to '{symbol_name}', not a symbol containing "
            f"'{EXPECTED_SYMBOL}'. Refusing to save the wrong instrument as US30."
        )
        print("If the broker uses a different valid name, set CTRADER_EXPECTED_SYMBOL accordingly.")
        stop_reactor_safely()
        return

    # Get the broker's configured price precision for this symbol.
    request = ProtoOASymbolByIdReq()
    request.ctidTraderAccountId = ACCOUNT_ID
    request.symbolId.append(SYMBOL_ID)
    deferred = client.send(request)
    deferred.addCallback(on_symbol_details)
    deferred.addErrback(on_fatal_error)


def on_symbol_details(result):
    global PRICE_DIGITS
    response = Protobuf.extract(result)
    found = [item for item in getattr(response, "symbol", []) if int(getattr(item, "symbolId", -1)) == SYMBOL_ID]
    if found:
        PRICE_DIGITS = int(getattr(found[0], "digits", PRICE_DIGITS))
    print(f"Price precision: {PRICE_DIGITS} decimal places")
    start_market_data_requests()


# ---------- Historical bar collection ----------

def trendbar_to_row(bar):
    # cTrader documents the price scale as 1/100,000. Do not infer the scale
    # from the raw value; that heuristic can break for other price levels.
    low_raw = int(bar.low)
    low = low_raw / PRICE_SCALE
    open_price = (low_raw + int(bar.deltaOpen)) / PRICE_SCALE
    close_price = (low_raw + int(bar.deltaClose)) / PRICE_SCALE
    high_price = (low_raw + int(bar.deltaHigh)) / PRICE_SCALE
    timestamp = dt.datetime.fromtimestamp(
        int(bar.utcTimestampInMinutes) * 60, tz=dt.timezone.utc
    )
    return {
        "timestamp": timestamp.isoformat(),
        "open": round(open_price, PRICE_DIGITS),
        "high": round(high_price, PRICE_DIGITS),
        "low": round(low, PRICE_DIGITS),
        "close": round(close_price, PRICE_DIGITS),
        "volume": float(bar.volume),  # cTrader reports bar volume in ticks.
    }


def on_trendbars(label, filename, result):
    response = Protobuf.extract(result)
    rows = [trendbar_to_row(bar) for bar in response.trendbar]
    path = DATA_DIR / filename

    if not rows:
        print(f"WARNING: No {label} bars returned by cTrader; writing an empty CSV with headers.")
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(path, index=False)
    else:
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

    bars_done.add(label)
    maybe_finish()
    return result


def on_bar_error(label, filename, failure):
    print(f"ERROR requesting {label} bars: {failure}")
    # Mark this request as finished so other timeframes and the quote can complete.
    bars_done.add(label)
    maybe_finish()
    return None


def start_market_data_requests():
    now = utc_now()
    now_ms = int(now.timestamp() * 1000)

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
            lambda result, _label=label, _filename=filename: on_trendbars(
                _label, _filename, result
            )
        )
        deferred.addErrback(
            lambda failure, _label=label, _filename=filename: on_bar_error(
                _label, _filename, failure
            )
        )

    # Historical bars do not provide a guaranteed live/current quote.
    # Subscribe separately for spot bid/ask events.
    request = ProtoOASubscribeSpotsReq()
    request.ctidTraderAccountId = ACCOUNT_ID
    request.symbolId.append(SYMBOL_ID)
    request.subscribeToSpotTimestamp = True
    deferred = client.send(request, responseTimeoutInSeconds=20)
    deferred.addErrback(on_fatal_error)


# ---------- Live quote collection ----------

def spot_time_to_utc(raw_timestamp):
    if not raw_timestamp:
        return None
    value = int(raw_timestamp)
    # Open API timestamps are normally milliseconds; tolerate second timestamps too.
    seconds = value / 1000.0 if value > 100_000_000_000 else float(value)
    try:
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def on_message_received(_client, message):
    global quote_done
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

    bid = round(bid_raw / PRICE_SCALE, PRICE_DIGITS) if bid_raw > 0 else None
    ask = round(ask_raw / PRICE_SCALE, PRICE_DIGITS) if ask_raw > 0 else None
    session_close = round(close_raw / PRICE_SCALE, PRICE_DIGITS) if close_raw > 0 else None

    # cTrader's chart commonly follows BID (the orange SELL price in the screenshot).
    # Prefer BID so the app's current-price field can match that chart convention.
    if bid is not None:
        current_price, price_basis = bid, "bid"
    elif ask is not None:
        current_price, price_basis = ask, "ask_fallback"
    elif session_close is not None:
        current_price, price_basis = session_close, "session_close_fallback"
    else:
        print("Spot event received but it contains no bid, ask, or session close. Waiting for another event.")
        return

    received_at = utc_now()
    spot_at = spot_time_to_utc(getattr(event, "timestamp", 0))
    age_seconds = None
    if spot_at is not None:
        age_seconds = round((received_at - spot_at).total_seconds(), 1)

    is_quote = price_basis in ("bid", "ask_fallback")
    payload = {
        "status": "QUOTE_RECEIVED" if is_quote else "SESSION_CLOSE_FALLBACK",
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
        "note": (
            "Bid/ask quote from cTrader spot event. Check spot_age_seconds for freshness."
            if price_basis in ("bid", "ask_fallback")
            else "Session-close fallback, not proof of a live market quote. Check timestamp/freshness."
        ),
    }
    write_json_atomic(DATA_DIR / "latest_quote.json", payload)
    quote_done = True

    bid_text = f"{bid:.{PRICE_DIGITS}f}" if bid is not None else "n/a"
    ask_text = f"{ask:.{PRICE_DIGITS}f}" if ask is not None else "n/a"
    print(
        f"[SPOT] {payload['symbol']} | BID={bid_text} | ASK={ask_text} | "
        f"dashboard current_price={current_price:.{PRICE_DIGITS}f} ({price_basis}) | "
        f"spot time={payload['spot_timestamp_utc'] or 'not supplied'}"
    )
    maybe_finish()


def disconnected(_client, reason):
    print(f"cTrader disconnected: {reason}")


if __name__ == "__main__":
    missing = [
        name
        for name, value in [
            ("CTRADER_CLIENT_ID", CLIENT_ID),
            ("CTRADER_CLIENT_SECRET", CLIENT_SECRET),
            ("CTRADER_ACCESS_TOKEN", ACCESS_TOKEN),
            ("CTRADER_ACCOUNT_ID / ACCOUNT_ID", ACCOUNT_ID),
        ]
        if not value
    ]
    if CTRADER_ENV not in ("live", "demo"):
        print("ERROR: Set CTRADER_ENV to either live or demo.")
        sys.exit(1)
    if missing:
        print("ERROR: Missing environment variable(s): " + ", ".join(missing))
        sys.exit(1)

    print(f"Connecting to cTrader {CTRADER_ENV.upper()} endpoint.")

    # Replace any old quote before connecting, so a failed fetch cannot silently
    # leave yesterday's/last run's quote looking current.
    write_json_atomic(
        DATA_DIR / "latest_quote.json",
        {
            "symbol": EXPECTED_SYMBOL,
            "symbol_id": SYMBOL_ID,
            "current_price": None,
            "status": "PENDING",
            "source": "ctrader_open_api_spot_event",
            "received_at_utc": iso_utc(utc_now()),
            "is_stale": True,
            "note": "A fresh quote has not yet been received in this run.",
        },
    )

    timeout_call = reactor.callLater(45, on_timeout)
    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message_received)
    client.startService()
    reactor.run()
