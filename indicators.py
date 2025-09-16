import pandas as pd
import numpy as np
import logging

def add_indicators(df, cfg=None):
    """
    Add core technical indicators.
    - Preserves DatetimeIndex (NO dropna/reset_index here)
    - Accepts optional `cfg` dict (e.g., config["technical_indicators"])
    """
    logging.info("Adding technical indicators")
    df = df.copy()
    cfg = (cfg or {})

    # windows (fall back to your previous defaults)
    ma_w      = int(cfg.get("ma_window", 14))
    ema_w     = int(cfg.get("ema_window", 14))
    bb_w      = int(cfg.get("bb_window", 20))
    bb_std    = float(cfg.get("bb_std", 2))
    vol_w     = int(cfg.get("vol_window", 10))
    mom_w     = int(cfg.get("mom_window", 10))
    macd_fast = int(cfg.get("macd_fast", 12))
    macd_slow = int(cfg.get("macd_slow", 26))

    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Log return (needed for volatility)
    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # MACD
    fast_ema = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = fast_ema - slow_ema

    # RSI (simple rolling mean version, matches your old logic)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(ma_w).mean()
    roll_down = down.rolling(ma_w).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving averages
    df["SMA"] = df["Close"].rolling(window=ma_w).mean()
    df["EMA"] = df["Close"].ewm(span=ema_w, adjust=False).mean()

    # Bollinger Bands
    ma_bb  = df["Close"].rolling(window=bb_w).mean()
    std_bb = df["Close"].rolling(window=bb_w).std()
    df["BB_upper"] = ma_bb + (bb_std * std_bb)
    df["BB_lower"] = ma_bb - (bb_std * std_bb)

    # Volatility (rolling std of log returns)
    df["RollingVolatility"] = df["Log_Return"].rolling(window=vol_w).std()

    # Momentum (keep your previous definition: price minus rolling mean)
    df["Momentum"] = df["Close"] - df["Close"].rolling(window=mom_w).mean()

    # DO NOT dropna() or reset_index() here; imputation happens next
    logging.info(f"Indicators added. DataFrame shape: {df.shape}")
    return df
