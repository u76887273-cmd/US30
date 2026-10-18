import os
import sys
import json
import urllib.request
import urllib.parse
import joblib
import psycopg
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# US30 QUANTUM SUPREME — LIVE PREDICTION ENGINE (UNLEASHED)
# ============================================================

# Database connection from GitHub Secrets
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is missing.")
    sys.exit(1)

# Paths for GitHub Actions runner
MODEL_DIR = "models"

# Broker spread adjustment removed (set to 0.0)
FEE = 0.0 

# ------------------------------------------------------------
# DATABASE CONNECTION & SCHEMA + DASHBOARD PERMISSIONS
# ------------------------------------------------------------
try:
    conn = psycopg.connect(DATABASE_URL)
    conn.autocommit = True 
except Exception as e:
    print(f"Database connection failed: {e}")
    sys.exit(1)

cur = conn.cursor()

# Ensure the log table exists AND allow the Live Dashboard (anon) to read it
cur.execute("""
CREATE TABLE IF NOT EXISTS supreme_grid_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ UNIQUE,
    price REAL,
    signal TEXT,
    reason TEXT,
    entry_zone TEXT,
    entry REAL,
    take_profit REAL,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT SELECT ON TABLE public.supreme_grid_log TO anon, authenticated;
ALTER TABLE public.supreme_grid_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow public read access" ON public.supreme_grid_log;
CREATE POLICY "Allow public read access" ON public.supreme_grid_log FOR SELECT USING (true);
NOTIFY pgrst, 'reload schema';
""")

print("Fetching recent US30 market data from m5_bars...")
cur.execute("SELECT * FROM m5_bars ORDER BY timestamp DESC LIMIT 200;")
columns = [desc[0] for desc in cur.description]
data = cur.fetchall()

if not data:
    print("No data found in m5_bars. Exiting.")
    sys.exit(0)

bars = pd.DataFrame(data, columns=columns)
bars["timestamp"] = pd.to_datetime(bars["timestamp"], format="mixed", utc=True)
for col in ["open", "high", "low", "close", "volume"]:
    bars[col] = pd.to_numeric(bars[col], errors="coerce")
bars = bars.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
bars = bars.sort_values("timestamp").reset_index(drop=True)

# ------------------------------------------------------------
# DYNAMIC FEATURE ENGINEERING (31 FEATURES)
# ------------------------------------------------------------
print("Engineering advanced features for AI...")

# Returns
bars['return_1'] = bars['close'].pct_change(1) * 100
bars['return_3'] = bars['close'].pct_change(3) * 100
bars['return_6'] = bars['close'].pct_change(6) * 100

# Candle shapes
bars['body'] = bars['close'] - bars['open']
bars['upper_wick'] = bars['high'] - bars[['open', 'close']].max(axis=1)
bars['lower_wick'] = bars[['open', 'close']].min(axis=1) - bars['low']

# EMAs
bars['ema20'] = bars['close'].ewm(span=20, adjust=False).mean()
bars['ema50'] = bars['close'].ewm(span=50, adjust=False).mean()
bars['ema_gap'] = bars['ema20'] - bars['ema50']
bars['ema20_slope'] = bars['ema20'].diff()

# ATRs
bars['prev_close'] = bars['close'].shift(1)
bars['tr1'] = bars['high'] - bars['low']
bars['tr2'] = (bars['high'] - bars['prev_close']).abs()
bars['tr3'] = (bars['low'] - bars['prev_close']).abs()
bars['tr'] = bars[['tr1', 'tr2', 'tr3']].max(axis=1)
bars['atr14'] = bars['tr'].rolling(14).mean()
bars['atr50'] = bars['tr'].rolling(50).mean()
bars['relative_atr'] = bars['atr14'] / bars['atr50']

# RSI 14
delta = bars["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
bars["rsi14"] = 100 - (100 / (1 + rs))

# VWAP (Daily Reset) - SAFE CUMSUM METHOD
bars['date'] = bars['timestamp'].dt.date
bars['typical_price'] = (bars['high'] + bars['low'] + bars['close']) / 3

cum_vp = (bars['typical_price'] * bars['volume']).groupby(bars['date']).cumsum()
cum_v = bars['volume'].groupby(bars['date']).cumsum().replace(0, np.nan)
bars['vwap'] = cum_vp / cum_v
bars['vwap_distance'] = bars['close'] - bars['vwap']

# Opening Range (OR)
or_df = bars.groupby('date').head(12).groupby('date').agg({
    'high': 'max', 'low': 'min', 'open': 'first'
})
bars['opening_high'] = bars['date'].map(or_df['high'])
bars['opening_low'] = bars['date'].map(or_df['low'])
bars['opening_open'] = bars['date'].map(or_df['open'])

bars['dist_or_high'] = bars['close'] - bars['opening_high']
bars['dist_or_low'] = bars['close'] - bars['opening_low']
bars['dist_open_price'] = bars['close'] - bars['opening_open']
bars['inside_or'] = ((bars['close'] < bars['opening_high']) & (bars['close'] > bars['opening_low'])).astype(int)

# Sessions
hour = bars['timestamp'].dt.hour
bars['tokyo'] = ((hour >= 0) & (hour < 8)).astype(int)
bars['london'] = ((hour >= 8) & (hour < 16)).astype(int)
bars['newyork'] = ((hour >= 13) & (hour < 21)).astype(int)

latest = bars.iloc[-1:]

# ------------------------------------------------------------
# LOAD SUPREME MODELS & PREDICT
# ------------------------------------------------------------
print("Loading US30 Supreme ML Models...")
FEATURES = joblib.load(f"{MODEL_DIR}/features.joblib")
upside_model = joblib.load(f"{MODEL_DIR}/upside30.joblib")

downside_path = f"{MODEL_DIR}/downside30.joblib"
width_path = f"{MODEL_DIR}/width30.joblib" if os.path.exists(f"{MODEL_DIR}/width30.joblib") else f"{MODEL_DIR}/range30.joblib"

downside_model = joblib.load(downside_path)
width_model = joblib.load(width_path)

X = latest[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

price = float(latest.close.iloc[0])
atr   = float(latest.atr14.iloc[0])
rsi   = float(latest.rsi14.iloc[0])
time_stamp = latest.timestamp.iloc[0]

pred_upside   = float(upside_model.predict(X)[0])
pred_downside = float(downside_model.predict(X)[0])
pred_width    = float(width_model.predict(X)[0])

# ------------------------------------------------------------
# PROJECT ALL ZONES (ZONE HEIGHT = 20%, FEE = 0.0)
# ------------------------------------------------------------
proj_high = price + pred_upside
proj_low  = price - pred_downside
midline = (proj_high + proj_low) / 2
zone_height = pred_width * 0.20

buy_zone_bottom = proj_low - FEE
buy_zone_top = (proj_low + zone_height) - FEE
buy_zone_str = f"{buy_zone_bottom:.0f} to {buy_zone_top:.0f}"
buy_limit = buy_zone_top
buy_tp = midline

sell_zone_bottom = proj_high - zone_height
sell_zone_top = proj_high
sell_zone_str = f"{sell_zone_bottom:.0f} to {sell_zone_top:.0f}"
sell_limit = sell_zone_bottom
sell_tp = midline - FEE

# ------------------------------------------------------------
# RAW SIGNAL GENERATION (PURE DIRECTIONAL / 0.6 RULE CANCELED)
# ------------------------------------------------------------
raw_signal = "WAIT"

if pred_upside > pred_downside:
    raw_signal = "BUY"
elif pred_downside > pred_upside:
    raw_signal = "SELL"

# ------------------------------------------------------------
# THE GRID DEFENSE SYSTEM (CANCELED)
# ------------------------------------------------------------
signal = raw_signal
reason = "Autonomous Execution" if raw_signal != "WAIT" else "Equal upside and downside (WAIT)"

# ------------------------------------------------------------
# ASSIGN FINAL VARIABLES FOR DATABASE
# ------------------------------------------------------------
if signal == "BUY":
    chart_zone, chart_entry, chart_tp = buy_zone_str, buy_limit, buy_tp
elif signal == "SELL":
    chart_zone, chart_entry, chart_tp = sell_zone_str, sell_limit, sell_tp
else:
    chart_zone = f"B: {buy_zone_str} | S: {sell_zone_str}"
    chart_entry = "-" 
    chart_tp = midline

# ------------------------------------------------------------
# INSERT INTO LOG (SUPABASE via psycopg3)
# ------------------------------------------------------------
ts = str(time_stamp)
entry_val = None if chart_entry == "-" else float(chart_entry)
tp_val    = None if chart_tp == "-" else float(chart_tp)
zone_val  = None if chart_zone == "-" else chart_zone

cur.execute("""
INSERT INTO supreme_grid_log (
    timestamp, price, signal, reason, entry_zone, entry, take_profit, status
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp) DO UPDATE SET
    price = EXCLUDED.price,
    signal = EXCLUDED.signal,
    reason = EXCLUDED.reason,
    entry_zone = EXCLUDED.entry_zone,
    entry = EXCLUDED.entry,
    take_profit = EXCLUDED.take_profit;
""", (
    ts,
    round(price, 0),
    signal,
    reason,
    zone_val,
    entry_val,
    tp_val,
    "PENDING" if signal != "WAIT" else "IGNORED"
))

cur.close()
conn.close()

# ------------------------------------------------------------
# REPORT & TELEGRAM NOTIFICATION
# ------------------------------------------------------------
print("\n" + "="*64)
print("US30 QUANTUM SUPREME — CLOUD ENGINE (UNLEASHED)")
print("="*64)
print(f"UTC Time : {time_stamp}")
print(f"Current  : {price:.0f}")
print(f"ATR (14) : {atr:.0f}")
print(f"RSI (14) : {rsi:.1f}")
print("-" * 64)
print(f"PROJECTED BUY ZONE  : {buy_zone_str} (Limit: {buy_limit:.0f})")
print(f"PROJECTED SELL ZONE : {sell_zone_str} (Limit: {sell_limit:.0f})")
print(f"EQUILIBRIUM TP      : {midline:.0f}")
print("-" * 64)
print(f"FINAL SIGNAL  : {signal}")
print(f"REASON        : {reason}")
if signal != "WAIT":
    print(f">> EXECUTING LIMIT ORDER AT {chart_entry:.0f} <<")
print("="*64)

bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")

if bot_token and chat_id and signal in ["BUY", "SELL"]:
    msg = f"🦅 *US30 QUANTUM SUPREME*\n"
    msg += f"========================\n"
    msg += f"💰 *Current Price:* {price:.0f}\n"
    msg += f"📊 *ATR:* {atr:.0f} | *RSI:* {rsi:.1f}\n\n"
    
    if signal == "BUY":
        msg += f"🟢 *SIGNAL: BUY*\n"
        msg += f"🎯 *LIMIT ENTRY:* {chart_entry:.0f}\n"
        msg += f"🏁 *TAKE PROFIT:* {chart_tp:.0f}\n\n"
        msg += f"📍 _Zone:_ {chart_zone}\n"
    elif signal == "SELL":
        msg += f"🔴 *SIGNAL: SELL*\n"
        msg += f"🎯 *LIMIT ENTRY:* {chart_entry:.0f}\n"
        msg += f"🏁 *TAKE PROFIT:* {chart_tp:.0f}\n\n"
        msg += f"📍 _Zone:_ {chart_zone}\n"

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({'chat_id': chat_id, 'text': msg, 'parse_mode': 'Markdown'}).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req)
        print("Telegram alert sent successfully!")
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")
