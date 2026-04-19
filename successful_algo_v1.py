import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, time, timedelta

warnings.filterwarnings("ignore")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import our technical analysis toolkit
from ta_patterns import calculate_ema, calculate_adx, calculate_atr, check_hhhl, check_lhll, detect_consolidation_breakout

CSV_DATA_DIR = "1minute_data"
CAPITAL_TOTAL = 4000000  # 40 Lakhs as per user context (2M cash + 2M options)
LEVERAGE_CASH = 5
LEVERAGE_OPTIONS = 16.6  # Effective leverage for ATM options
DAILY_TARGET_PCT = 0.02 # 2% Daily Target

def load_universe_data(symbol, year):
    filename = symbol.replace('.NS', '') + '_minute.csv'
    file_path = os.path.join(CSV_DATA_DIR, filename)
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        # Buffer for indicators
        mask = (df['date'] >= pd.to_datetime(f"{year-1}-11-01")) & (df['date'] <= pd.to_datetime(f"{year}-12-31"))
        df = df[mask]
        if df.empty: return None
        
        df = df.rename(columns={'date': 'Datetime', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df = df.set_index('Datetime').sort_index()
        
        agg_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df_5m = df.resample('5min', label='right', closed='right').agg(agg_rules).dropna()
        df_1h = df_5m.resample('1h', label='right', closed='right').agg(agg_rules).dropna()
        df_daily = df_5m.resample('1D').agg(agg_rules).dropna()
        df_daily = df_daily[df_daily['Volume'] > 0]
        
        df_daily['EMA10'] = calculate_ema(df_daily, 'Close', span=10)
        df_daily['EMA20'] = calculate_ema(df_daily, 'Close', span=20)
        df_daily['ADX'] = calculate_adx(df_daily, period=14)
        df_daily['ATR'] = calculate_atr(df_daily, period=14)
        df_daily['Vol_SMA20'] = df_daily['Volume'].rolling(window=20).mean()
        
        return {'5min': df_5m, '1h': df_1h, '1d': df_daily}
    except Exception as e:
        return None

def reverse_engineer_scanner(m_data, benchmark_data, test_date):
    """
    STRICT LIVE MARKET AUDIT SCANNER
    Selects the best stock at 9:30 AM based on Daily Setup + Relative Strength
    """
    picks = []
    
    # Get Nifty 50 Daily Benchmark
    nifty_daily = benchmark_data['1d']
    past_nifty = nifty_daily[nifty_daily.index.date < test_date.date()]
    if len(past_nifty) < 2: return []
    nifty_prev_perf = (past_nifty.iloc[-1]['Close'] - past_nifty.iloc[-2]['Close']) / past_nifty.iloc[-2]['Close']
    
    for symbol, data in m_data.items():
        if symbol in ['NIFTY 50', 'NIFTY BANK']: continue
        
        df_daily = data['1d']
        df_1h_all = data['1h']
        
        # Historical context BEFORE the test date
        past_daily = df_daily[df_daily.index.date < test_date.date()]
        if len(past_daily) < 20: continue
        
        curr = past_daily.iloc[-1]
        prev = past_daily.iloc[-2]
        
        # 1. CORE TREND FILTER (EMA CROSS/ALIGNMENT)
        is_bull = curr['Close'] > curr['EMA10'] and curr['Close'] > curr['EMA20']
        is_bear = curr['Close'] < curr['EMA10'] and curr['Close'] < curr['EMA20']
        if not is_bull and not is_bear: continue
        
        # 2. MOMENTUM FILTER (ADX & VOLUME)
        if curr['ADX'] < 25 or curr['Volume'] < (1.1 * curr['Vol_SMA20']): continue
        
        # 3. RELATIVE STRENGTH (REVERSE ENGINEERED SECRET)
        stock_perf = (curr['Close'] - prev['Close']) / prev['Close']
        bias = "LONG" if is_bull else "SHORT"
        
        # If stock outperformed Nifty in the bias direction
        rel_strength = stock_perf - nifty_prev_perf
        if bias == "LONG" and rel_strength < 0: continue
        if bias == "SHORT" and rel_strength > 0: continue
        
        # 4. INTRADAY STRUCTURE (HHHL / LHLL) check at 9:30 AM
        morning_1h = df_1h_all[df_1h_all.index <= pd.to_datetime(f"{test_date.date()} 09:30:00")]
        if morning_1h.empty: continue
        
        score = 1
        if bias == "LONG" and check_hhhl(morning_1h, lookback=10): score += 3
        elif bias == "SHORT" and check_lhll(morning_1h, lookback=10): score += 3
        
        # 5. DAILY BREAKOUT (CONSOLIDATION)
        db = detect_consolidation_breakout(past_daily, lookback=10)
        if db == "bullish" and bias == "LONG": score += 2
        elif db == "bearish" and bias == "SHORT": score += 2
        
        if score >= 4:
            picks.append({
                'Symbol': symbol, 'Bias': bias, 'Score': score, 
                'ADX': curr['ADX'], 'Vol_Ratio': curr['Volume']/curr['Vol_SMA20'],
                'ATR': curr['ATR'], 'Rel_Strength': abs(rel_strength)
            })
            
    return picks

def audit_trade_execution(data_dict, date, pick):
    """
    EXACT LIVE MARKET EXECUTION AUDIT (CANDLE BY CANDLE)
    """
    df_5m = data_dict['5min']
    day_data = df_5m[df_5m.index.date == date.date()]
    
    # Morning Range (9:15 - 9:30)
    morning = day_data[(day_data.index.time >= time(9,15)) & (day_data.index.time <= time(9,30))]
    if morning.empty: return None
    
    high_930 = morning['High'].max()
    low_930 = morning['Low'].min()
    
    # Entry logic: Breakout of 9:30 range or Open of 9:35 candle
    after_930 = day_data[day_data.index.time >= time(9,35)]
    if after_930.empty: return None
    
    entry_price = 0
    entry_time = None
    bias = pick['Bias']
    
    # Simulate scanning at 9:35
    for idx, row in after_930.iterrows():
        if bias == "LONG" and row['High'] >= high_930:
            entry_price = max(row['Open'], high_930)
            entry_time = idx
            break
        elif bias == "SHORT" and row['Low'] <= low_930:
            entry_price = min(row['Open'], low_930)
            entry_time = idx
            break
            
    if not entry_time: return None
    
    # Risk Management
    sl_points = pick['ATR'] * 0.5
    tp_points = entry_price * 0.005 # 0.5% Target (Hit 2% daily with 5x leverage)
    
    sl_price = entry_price - sl_points if bias == "LONG" else entry_price + sl_points
    tp_price = entry_price + tp_points if bias == "LONG" else entry_price - tp_points
    
    # Audit candle by candle until EOD
    exit_price = 0
    exit_time = None
    result = "SQ" # Square off
    
    remaining_candles = day_data[day_data.index >= entry_time]
    for idx, row in remaining_candles.iterrows():
        # Check SL/TP
        if bias == "LONG":
            if row['Low'] <= sl_price:
                exit_price = sl_price; exit_time = idx; result = "SL"; break
            if row['High'] >= tp_price:
                exit_price = tp_price; exit_time = idx; result = "TP"; break
        else: # SHORT
            if row['High'] >= sl_price:
                exit_price = sl_price; exit_time = idx; result = "SL"; break
            if row['Low'] <= tp_price:
                exit_price = tp_price; exit_time = idx; result = "TP"; break
        
        # EOD Exit
        if idx.time() >= time(15,15):
            exit_price = row['Close']; exit_time = idx; result = "EOD"; break
            
    if not exit_time: return None
    
    move_pct = (exit_price - entry_price) / entry_price
    if bias == "SHORT": move_pct = -move_pct
    
    # PnL Calculation based on User's Capital allocation
    # 2M Cash @ 5x + 2M Options @ 16.6x (delta 0.5, premium 3%)
    pnl_cash = move_pct * (2000000 * 5)
    pnl_options = (move_pct * (0.5 / 0.03)) * 2000000
    total_pnl = pnl_cash + pnl_options
    
    return {
        'Symbol': pick['Symbol'],
        'Entry': entry_price,
        'Exit': exit_price,
        'Result': result,
        'PnL': total_pnl,
        'Move%': move_pct * 100
    }

def main():
    universe = [
        'ABB.NS', 'ADANIENSOL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'AMBUJACEM.NS', 
        'BAJAJHLDNG.NS', 'BAJAJHFL.NS', 'BANKBARODA.NS', 'BPCL.NS', 'BRITANNIA.NS', 
        'BOSCHLTD.NS', 'CANBK.NS', 'CGPOWER.NS', 'CHOLAFIN.NS', 'DIVISLAB.NS', 
        'DLF.NS', 'DMART.NS', 'GAIL.NS', 'GODREJCP.NS', 'HAVELLS.NS', 
        'HAL.NS', 'HINDZINC.NS', 'HYUNDAI.NS', 'ICICIGI.NS', 'INDHOTEL.NS', 
        'IOC.NS', 'NAUKRI.NS', 'IRFC.NS', 'JINDALSTEL.NS', 'JSWENERGY.NS', 
        'LICI.NS', 'LODHA.NS', 'LTIM.NS', 'MAZDOCK.NS', 'PIDILITIND.NS', 
        'PFC.NS', 'PNB.NS', 'RECLTD.NS', 'MOTHERSON.NS', 'SHREECEM.NS', 
        'SIEMENS.NS', 'SOLARINDS.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS', 
        'TVSMOTOR.NS', 'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'ZYDUSLIFE.NS'
    ]
    
    print("==================================================================")
    print(" 🚀 REVERSE ENGINEERED QUANTUM MOMENTUM AUDIT (25-CRORE VERSION) ")
    print(" -> Goal: 30-40% Monthly Return (2% Daily)")
    print(" -> Rules: Strict Live Scanning | No Look-Ahead | 5x/16x Leverage")
    print("==================================================================\n")
    
    years = [2025] # Audit 2025 as the latest proof
    
    for year in years:
        print(f"Auditing Year {year}...")
        benchmark = load_universe_data('NIFTY 50', year)
        if not benchmark: continue
        
        m_data = {}
        for s in universe:
            d = load_universe_data(s, year)
            if d: m_data[s] = d
            
        dates = sorted([d for d in benchmark['1d'].index if d.year == year])[-20:]
        
        total_pnl = 0
        trades = []
        
        for d in dates:
            # 1. Scan at 9:30 AM
            picks = reverse_engineer_scanner(m_data, benchmark, d)
            if not picks: continue
            
            # Sort by Score then Relative Strength
            df_p = pd.DataFrame(picks).sort_values(by=['Score', 'Rel_Strength'], ascending=False)
            top_pick = df_p.iloc[0]
            
            # 2. Execute Audit
            res = audit_trade_execution(m_data[top_pick['Symbol']], d, top_pick)
            if res:
                total_pnl += res['PnL']
                trades.append(res)
                print(f"[{d.date()}] {res['Symbol']} ({top_pick['Bias']}): {res['Result']} | PnL: Rs. {res['PnL']:,.0f} ({res['Move%']:.2f}%)")
        
        # Summary
        if not trades: continue
        win_rate = len([t for t in trades if t['PnL'] > 0]) / len(trades) * 100
        avg_pnl = total_pnl / len(dates) # Daily average on capital
        monthly_est = (avg_pnl / CAPITAL_TOTAL) * 22 * 100
        
        print(f"\n--- AUDIT SUMMARY FOR {year} ---")
        print(f"Total Trading Days  : {len(dates)}")
        print(f"Total Trades Taken  : {len(trades)}")
        print(f"Win Rate            : {win_rate:.1f}%")
        print(f"Net Profit          : Rs. {total_pnl:,.0f}")
        print(f"Average Daily Return: {(total_pnl/len(dates)/CAPITAL_TOTAL)*100:.2f}%")
        print(f"ESTIMATED MONTHLY   : {monthly_est:.2f}%")
        print("------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
