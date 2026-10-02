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
        "DATABASE_URL GitHub Secret is not configured."
    )


# ============================================================
# cTRADER API
# ============================================================

CTRADER_CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID")

CTRADER_CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET")

if not CTRADER_CLIENT_ID:
    raise RuntimeError(
        "CTRADER_CLIENT_ID GitHub Secret is not configured."
    )

if not CTRADER_CLIENT_SECRET:
    raise RuntimeError(
        "CTRADER_CLIENT_SECRET GitHub Secret is not configured."
    )


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)


# ============================================================
# MODEL FILES
# ============================================================

UPSIDE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "upside30.joblib"
)

DOWNSIDE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "downside30.joblib"
)

RANGE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "range30.joblib"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "metadata.joblib"
)
