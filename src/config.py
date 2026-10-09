import os

# ============================================================
# US30 QUANTUM — CONFIGURATION
# ============================================================

SYMBOL_NAME = "#US30"
SYMBOL_ID = 1287
ACCOUNT_ID = 25145646
TIMEFRAME = "M5"

# ============================================================
# SUPABASE DATABASE
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL GitHub Secret is missing."
    )

# ============================================================
# cTRADER API
# ============================================================

CTRADER_CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID")
CTRADER_CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET")
CTRADER_ACCESS_TOKEN = os.environ.get("CTRADER_ACCESS_TOKEN")

if not CTRADER_CLIENT_ID:
    raise RuntimeError("CTRADER_CLIENT_ID GitHub Secret is missing.")

if not CTRADER_CLIENT_SECRET:
    raise RuntimeError("CTRADER_CLIENT_SECRET GitHub Secret is missing.")

if not CTRADER_ACCESS_TOKEN:
    raise RuntimeError("CTRADER_ACCESS_TOKEN GitHub Secret is missing.")

# ============================================================
# PATH CONFIGURATION
# ============================================================
# __file__ points to src/config.py
# dirname(__file__) points to src/
# dirname(dirname(__file__)) points to the repository root

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# MODEL FILES
# ============================================================

UPSIDE_MODEL_PATH = os.path.join(MODEL_DIR, "upside30.joblib")
DOWNSIDE_MODEL_PATH = os.path.join(MODEL_DIR, "downside30.joblib")
RANGE_MODEL_PATH = os.path.join(MODEL_DIR, "range30.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")
