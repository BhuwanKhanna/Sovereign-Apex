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
CAPITAL_TOTAL = 4000000 
CASH_ALLOC = 2000000
OPTS_ALLOC = 2000000

def load_data_optimized(symbol, year):
    filename = symbol.replace('.NS', '') + '_minute.csv'
    file_path = os.path.join(CSV_DATA_DIR, filename)
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        start_date = pd.to_datetime(f"{year-1}-10-01")
        end_date = pd.to_datetime(f"{year}-12-31")
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
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
    except: return None

def audit_day_2024(m_data, benchmark_data, date):
    nifty_daily = benchmark_data['1d']
    past_nifty = nifty_daily[nifty_daily.index.date < date.date()]
    if len(past_nifty) < 2: return None
    n_perf = (past_nifty.iloc[-1]['Close'] - past_nifty.iloc[-2]['Close']) / past_nifty.iloc[-2]['Close']
    
    picks = []
    for s, data in m_data.items():
        if s in ['NIFTY 50', 'NIFTY BANK']: continue
        d_daily = data['1d']
        past_d = d_daily[d_daily.index.date < date.date()]
        if len(past_d) < 20: continue
        
        curr = past_d.iloc[-1]
        prev = past_d.iloc[-2]
        
        # QUANTUM FILTERS
        is_bull = curr['Close'] > curr['EMA10'] and curr['Close'] > curr['EMA20']
        is_bear = curr['Close'] < curr['EMA10'] and curr['Close'] < curr['EMA20']
        if not (is_bull or is_bear): continue
        if curr['ADX'] < 25 or curr['Volume'] < (1.2 * curr['Vol_SMA20']): continue
        
        s_perf = (curr['Close'] - prev['Close']) / prev['Close']
        bias = "LONG" if is_bull else "SHORT"
        
        # RELATIVE STRENGTH (QRS Edge)
        alpha = s_perf - n_perf
        if (bias == "LONG" and alpha < 0) or (bias == "SHORT" and alpha > 0): continue
        
        # Intraday Check at 9:30
        m_1h = data['1h'][data['1h'].index <= pd.to_datetime(f"{date.date()} 09:30:00")]
        score = 1
        if bias == "LONG" and check_hhhl(m_1h, 10): score += 3
        elif bias == "SHORT" and check_lhll(m_1h, 10): score += 3
        
        db = detect_consolidation_breakout(past_d, 10)
        if (db == "bullish" and bias == "LONG") or (db == "bearish" and bias == "SHORT"): score += 2
        
        if score >= 4:
            picks.append({'S': s, 'B': bias, 'Score': score, 'ADX': curr['ADX'], 'ATR': curr['ATR'], 'Alpha': abs(alpha)})
            
    if not picks: return None
    
    # Pick top by Alpha (Relative Strength)
    top = sorted(picks, key=lambda x: (x['Score'], x['Alpha']), reverse=True)[0]
    
    # Execution Simulation
    df_5m = m_data[top['S']]['5min']
    day_df = df_5m[df_5m.index.date == date.date()]
    morning = day_df[(day_df.index.time >= time(9,15)) & (day_df.index.time <= time(9,30))]
    if morning.empty: return None
    
    h_930, l_930 = morning['High'].max(), morning['Low'].min()
    candidates = day_df[day_df.index.time >= time(9,35)]
    
    e_p, e_t = 0, None
    for idx, row in candidates.iterrows():
        if top['B'] == "LONG" and row['High'] >= h_930:
            e_p, e_t = max(row['Open'], h_930), idx; break
        elif top['B'] == "SHORT" and row['Low'] <= l_930:
            e_p, e_t = min(row['Open'], l_930), idx; break
    if not e_t: return None
    
    # Risk/Reward
    sl_p = e_p - (top['ATR']*0.4) if top['B']=="LONG" else e_p + (top['ATR']*0.4)
    tp_p = e_p * 1.008 if top['B']=="LONG" else e_p * 0.992 # 0.8% Spot Target
    
    res_data = None
    for idx, row in day_df[day_df.index >= e_t].iterrows():
        hit_sl = (row['Low'] <= sl_p) if top['B']=="LONG" else (row['High'] >= sl_p)
        hit_tp = (row['High'] >= tp_p) if top['B']=="LONG" else (row['Low'] <= tp_p)
        if hit_sl: 
            pnl = -0.004 * (CASH_ALLOC*5 + OPTS_ALLOC*16.6) # Loss approx 0.4% spot
            res_data = {'S': top['S'], 'B': top['B'], 'Res': 'SL', 'PnL': pnl}
            break
        if hit_tp: 
            pnl = 0.008 * (CASH_ALLOC*5 + OPTS_ALLOC*16.6)
            res_data = {'S': top['S'], 'B': top['B'], 'Res': 'TP', 'PnL': pnl}
            break
        if idx.time() >= time(15,15):
            move = (row['Close'] - e_p)/e_p if top['B']=="LONG" else (e_p - row['Close'])/e_p
            pnl = move * (CASH_ALLOC*5 + OPTS_ALLOC*16.6)
            res_data = {'S': top['S'], 'B': top['B'], 'Res': 'EOD', 'PnL': pnl}
            break
    return res_data

def main():
    universe = ['ABB.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'AMBUJACEM.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BANKBARODA.NS', 'BEL.NS', 'BHARTIARTL.NS', 'BHEL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CANBK.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS', 'HAL.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
    
    print("================================================================")
    print(" 🛡️ 2024 QUANTUM RELATIVE STRENGTH (QRS) AUDIT 🛡️")
    print(" -> Strict Live Market Mode | No Next-Candle Knowledge")
    print("==============================================================\n")
    
    print("Loading 2024 Institutional Data...")
    bench = load_data_optimized('NIFTY 50', 2024)
    m_data = {s: d for s in universe if (d := load_data_optimized(s, 2024))}
    
    dates = sorted([d for d in bench['1d'].index if d.year == 2024])
    total_pnl = 0; trades_taken = 0; wins = 0
    
    print(f"{'Date':<12} | {'Symbol':<12} | {'Bias':<5} | {'Res':<4} | {'PnL (Rs)':<12}")
    print("-" * 65)
    
    # Process month by month to show progress
    for d in dates:
        res = audit_day_2024(m_data, bench, d)
        if res:
            total_pnl += res['PnL']
            trades_taken += 1
            if res['PnL'] > 0: wins += 1
            print(f"{d.date()} | {res['S']:<12} | {res['B']:<5} | {res['Res']:<4} | {res['PnL']:>12,.0f}")
            
    if trades_taken > 0:
        print("-" * 65)
        print(f"TOTAL TRADES: {trades_taken} | WIN RATE: {wins/trades_taken*100:.1f}%")
        print(f"NET PROFIT  : Rs. {total_pnl:,.0f}")
        print(f"ROI ON 40L  : {(total_pnl/CAPITAL_TOTAL)*100:.1f}%")
        print(f"MONTHLY AVG : {(total_pnl/trades_taken/CAPITAL_TOTAL)*22*100:.1f}% Est.")
    else:
        print("\n[!] No trades triggered in 2024 with these strict parameters.")

if __name__ == "__main__": main()
