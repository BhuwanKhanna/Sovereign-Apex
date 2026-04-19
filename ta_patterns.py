import pandas as pd
import numpy as np

def calculate_ema(df, column='Close', span=20):
    return df[column].ewm(span=span, adjust=False).mean()

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    """Calculates Average Directional Index (ADX)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm[minus_dm > -plus_dm] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def check_hhhl(df, lookback=10):
    """
    Checks for Higher-High Higher-Low structure.
    Simple approach: The maximum and minimum of the recent half must be 
    higher than the maximum and minimum of the older half of the lookback.
    """
    if len(df) < lookback: return False
    
    recent_half = df.iloc[-lookback//2:]
    older_half = df.iloc[-lookback: -lookback//2]
    
    hh = recent_half['High'].max() > older_half['High'].max()
    hl = recent_half['Low'].min() > older_half['Low'].min()
    
    return hh and hl

def check_lhll(df, lookback=10):
    """Checks for Lower-High Lower-Low structure."""
    if len(df) < lookback: return False
    
    recent_half = df.iloc[-lookback//2:]
    older_half = df.iloc[-lookback: -lookback//2]
    
    lh = recent_half['High'].max() < older_half['High'].max()
    ll = recent_half['Low'].min() < older_half['Low'].min()
    
    return lh and ll

def detect_consolidation_breakout(df, lookback=20, threshold_pct=0.02):
    """
    Detects if the price was consolidating (tight range) recently,
    and the current candle is breaking out of that range.
    Returns: 'bullish', 'bearish', or None
    """
    if len(df) < lookback + 1: return None
    
    # Analyze the previous 'lookback' candles (excluding the current one)
    past_period = df.iloc[-(lookback+1):-1]
    highest = past_period['High'].max()
    lowest = past_period['Low'].min()
    
    # Range of the consolidation block as a percentage of the low
    range_pct = (highest - lowest) / lowest
    
    # Check if the past period was relatively tight/consolidating
    is_consolidating = range_pct <= threshold_pct
    
    if is_consolidating:
        curr_candle = df.iloc[-1]
        if curr_candle['Close'] > highest:
            return 'bullish'
        elif curr_candle['Close'] < lowest:
            return 'bearish'
            
    return None

def detect_candlestick_patterns(df):
    """
    Checks for basic bullish and bearish single/double candle patterns.
    Returns: list of pattern strings
    """
    if len(df) < 2: return []
    
    patterns = []
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Basic definitions
    body_size = abs(curr['Close'] - curr['Open'])
    upper_wick = curr['High'] - max(curr['Close'], curr['Open'])
    lower_wick = min(curr['Close'], curr['Open']) - curr['Low']
    total_size = curr['High'] - curr['Low']
    
    is_bullish_candle = curr['Close'] > curr['Open']
    is_bearish_candle = curr['Close'] < curr['Open']
    
    prev_is_bullish = prev['Close'] > prev['Open']
    prev_is_bearish = prev['Close'] < prev['Open']
    
    # 1. Bullish Engulfing
    if prev_is_bearish and is_bullish_candle:
        if curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
            patterns.append('bullish_engulfing')
            
    # 2. Bearish Engulfing
    if prev_is_bullish and is_bearish_candle:
        if curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']:
            patterns.append('bearish_engulfing')
            
    # 3. Hammer (Bullish Reversal / Continuation)
    if lower_wick > (2 * body_size) and upper_wick < (0.2 * total_size):
        patterns.append('hammer')
        
    # 4. Shooting Star (Bearish Reversal / Continuation)
    if upper_wick > (2 * body_size) and lower_wick < (0.2 * total_size):
        patterns.append('shooting_star')
        
    return patterns

def calculate_rsi(df, period=14):
    """Calculates Relative Strength Index (RSI)"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
