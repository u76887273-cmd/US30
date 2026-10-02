from datetime import datetime, timezone

from ctrader_open_api import (
    Client,
    Protobuf,
    TcpProtocol,
    EndPoints,
)

from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
)

from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAGetTrendbarsReq,
    ProtoOAGetTrendbarsRes,
)

from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOATrendbarPeriod,
)

from twisted.internet import reactor

from .config import (
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_ACCESS_TOKEN,
    ACCOUNT_ID,
    SYMBOL_ID,
)


# ============================================================
# US30 QUANTUM — cTRADER M5 DATA
# ============================================================

PRICE_SCALE = 100000


# ============================================================
# INTERNAL STATE
# ============================================================

_result = None
_error = None
_stage = "starting"

_request_from = None
_request_to = None


# ============================================================
# ERROR HANDLER
# ============================================================

def on_error(failure):
    global _error

    _error = RuntimeError(
        f"cTrader API error: {failure}"
    )

    if reactor.running:
        reactor.stop()


# ============================================================
# CONNECTED
# ============================================================

def on_connected(client):
    global _stage

    _stage = "application_auth"

    request = ProtoOAApplicationAuthReq()

    request.clientId = CTRADER_CLIENT_ID
    request.clientSecret = CTRADER_CLIENT_SECRET

    deferred = client.send(request)
    deferred.addErrback(on_error)


# ============================================================
# DISCONNECTED
# ============================================================

def on_disconnected(client, reason):
    global _error

    if _result is None and _error is None:
        _error = RuntimeError(
            f"cTrader connection disconnected: {reason}"
        )

    if reactor.running:
        reactor.stop()


# ============================================================
# MESSAGE HANDLER
# ============================================================

def on_message(client, message):
    global _stage
    global _result
    global _error

    try:

        # ----------------------------------------------------
        # APPLICATION AUTHENTICATION SUCCESS
        # ----------------------------------------------------

        if (
            message.payloadType
            == ProtoOAApplicationAuthRes().payloadType
        ):

            _stage = "account_auth"

            request = ProtoOAAccountAuthReq()

            request.ctidTraderAccountId = ACCOUNT_ID
            request.accessToken = CTRADER_ACCESS_TOKEN

            deferred = client.send(request)
            deferred.addErrback(on_error)

            return

        # ----------------------------------------------------
        # ACCOUNT AUTHENTICATION SUCCESS
        # ----------------------------------------------------

        if (
            message.payloadType
            == ProtoOAAccountAuthRes().payloadType
        ):

            _stage = "requesting_bars"

            request = ProtoOAGetTrendbarsReq()

            request.ctidTraderAccountId = ACCOUNT_ID

            request.symbolId = SYMBOL_ID

            request.period = (
                ProtoOATrendbarPeriod.Value("M5")
            )

            request.fromTimestamp = int(
                _request_from
            )

            request.toTimestamp = int(
                _request_to
            )

            deferred = client.send(request)
            deferred.addErrback(on_error)

            return

        # ----------------------------------------------------
        # M5 TREND BAR RESPONSE
        # ----------------------------------------------------

        if (
            message.payloadType
            == ProtoOAGetTrendbarsRes().payloadType
        ):

            response = Protobuf.extract(message)

            bars = []

            for bar in response.trendbar:

                # ------------------------------------------------
                # cTrader trendbar price reconstruction
                # ------------------------------------------------

                low = bar.low / PRICE_SCALE

                open_price = (
                    bar.low + bar.deltaOpen
                ) / PRICE_SCALE

                high = (
                    bar.low + bar.deltaHigh
                ) / PRICE_SCALE

                close = (
                    bar.low + bar.deltaClose
                ) / PRICE_SCALE

                # ------------------------------------------------
                # cTrader timestamp is UTC minutes
                # ------------------------------------------------

                timestamp_ms = (
                    int(bar.utcTimestampInMinutes)
                    * 60
                    * 1000
                )

                timestamp = datetime.fromtimestamp(
                    timestamp_ms / 1000,
                    tz=timezone.utc,
                ).replace(tzinfo=None)

                bars.append(
                    {
                        "timestamp": timestamp,
                        "open": float(open_price),
                        "high": float(high),
                        "low": float(low),
                        "close": float(close),
                        "volume": int(bar.volume),
                    }
                )

            _result = bars
            _stage = "complete"

            if reactor.running:
                reactor.stop()

            return

    except Exception as exc:

        _error = RuntimeError(
            f"Failed to process cTrader response: {exc}"
        )

        if reactor.running:
            reactor.stop()


# ============================================================
# DOWNLOAD M5 BARS
# ============================================================

def get_m5_bars(
    from_timestamp_ms,
    to_timestamp_ms,
):
    """
    Download US30 M5 bars from cTrader.

    Parameters
    ----------
    from_timestamp_ms : int
        Starting Unix timestamp in milliseconds.

    to_timestamp_ms : int
        Ending Unix timestamp in milliseconds.

    Returns
    -------
    list
        List of dictionaries containing:

        timestamp
        open
        high
        low
        close
        volume
    """

    global _result
    global _error
    global _stage
    global _request_from
    global _request_to

    # --------------------------------------------------------
    # Reset state
    # --------------------------------------------------------

    _result = None
    _error = None
    _stage = "starting"

    _request_from = int(
        from_timestamp_ms
    )

    _request_to = int(
        to_timestamp_ms
    )

    # --------------------------------------------------------
    # Create cTrader client
    # --------------------------------------------------------

    client = Client(
        EndPoints.PROTOBUF_LIVE_HOST,
        EndPoints.PROTOBUF_PORT,
        TcpProtocol,
    )

    # --------------------------------------------------------
    # Register callbacks
    # --------------------------------------------------------

    client.setConnectedCallback(
        on_connected
    )

    client.setDisconnectedCallback(
        on_disconnected
    )

    client.setMessageReceivedCallback(
        on_message
    )

    # --------------------------------------------------------
    # Start connection
    # --------------------------------------------------------

    client.startService()

    # --------------------------------------------------------
    # Run Twisted event loop
    # --------------------------------------------------------

    reactor.run()

    # --------------------------------------------------------
    # Stop cTrader client
    # --------------------------------------------------------

    try:
        client.stopService()
    except Exception:
        pass

    # --------------------------------------------------------
    # Handle errors
    # --------------------------------------------------------

    if _error is not None:
        raise _error

    if _result is None:
        raise RuntimeError(
            "cTrader returned no M5 data."
        )

    # --------------------------------------------------------
    # Return downloaded bars
    # --------------------------------------------------------

    return _result
