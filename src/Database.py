import pandas as pd
from sqlalchemy import create_engine
from src.config import DATABASE_URL

def get_engine():
    """
    Creates and returns a SQLAlchemy engine connected to Supabase.
    SQLAlchemy requires 'postgresql://', so we safely replace 'postgres://' 
    if Supabase provided the older format in the DATABASE_URL.
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set.")
        
    db_url = DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

def get_latest_timestamp(table_name="m5_bars"):
    """
    Finds the most recent bar we have saved, so cTrader knows exactly 
    where to resume downloading from. (Replaces Step 4 in Cell 2).
    """
    engine = get_engine()
    query = f"SELECT MAX(timestamp) AS ts FROM {table_name}"
    
    try:
        df = pd.read_sql_query(query, engine)
        if not df.empty and pd.notna(df.ts.iloc[0]):
            return pd.to_datetime(df.ts.iloc[0], utc=True)
    except Exception as e:
        print(f"Table {table_name} might not exist yet or is empty: {e}")
        
    return None

def save_new_bars(df, table_name="m5_bars"):
    """
    Appends freshly downloaded M5 bars to the Supabase database.
    """
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"✓ Saved {len(df)} new rows to {table_name}")

def load_recent_bars(limit=200):
    """
    Loads just enough recent history to calculate EMA50 and Daily VWAP.
    This replaces downloading the entire DB every 30 minutes!
    """
    engine = get_engine()
    query = f"""
        SELECT timestamp, open, high, low, close, volume 
        FROM m5_bars 
        ORDER BY timestamp DESC 
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, engine)
    
    # Sort chronologically (oldest to newest) for feature math
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    return df
