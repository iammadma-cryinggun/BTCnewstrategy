# -*- coding: utf-8 -*-
"""
第一步：收集半年数据并找到所有开仓信号（完整验证5逻辑 + DXY燃料）
- 使用验证5完整逻辑（含DXY燃料增强）
- 记录所有4H收盘价和信号
- 区分：原始信号 vs V7.0.5过滤后的开仓信号
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.signal import detrend, hilbert
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("第一步：收集半年数据和开仓信号（完整验证5逻辑 + DXY燃料）")
print("=" * 80)
print()

# ==================== 配置 ====================
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=180)

print(f"时间范围: {START_DATE.strftime('%Y-%m-%d')} 到 {END_DATE.strftime('%Y-%m-%d')}")
print()

# ==================== 获取BTC 4H数据 ====================
print("正在获取BTC 4H数据...")

def fetch_btc_4h_data(start_date, end_date):
    """从Binance获取BTC 4H数据"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        all_data = []

        current_start = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        while current_start < end_timestamp:
            params = {
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'startTime': current_start,
                'endTime': end_timestamp,
                'limit': 1000
            }

            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            current_start = data[-1][6] + 1

            if len(data) < 1000:
                break

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        print(f"[ERROR] 获取数据失败: {e}")
        return None

df = fetch_btc_4h_data(START_DATE, END_DATE)

if df is None or len(df) < 100:
    print("[ERROR] 数据获取失败")
    exit(1)

print(f"[OK] 获取数据: {len(df)}条")
print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
print()

# ==================== 获取DXY数据 ====================
print("正在获取DXY数据...")

def fetch_dxy_data(start_date, end_date):
    """从FRED获取DXY美元指数数据"""
    try:
        from io import StringIO

        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS'
        resp = requests.get(url, timeout=15)

        if resp.status_code != 200:
            print("[WARNING] DXY数据获取失败")
            return None

        dxy_df = pd.read_csv(StringIO(resp.text))
        dxy_df['observation_date'] = pd.to_datetime(dxy_df['observation_date'])
        dxy_df.set_index('observation_date', inplace=True)
        dxy_df.rename(columns={'DTWEXBGS': 'Close'}, inplace=True)
        dxy_df = dxy_df.dropna()
        dxy_df['Close'] = pd.to_numeric(dxy_df['Close'], errors='coerce')

        # 过滤时间范围
        mask = (dxy_df.index >= start_date) & (dxy_df.index <= end_date)
        dxy_df = dxy_df[mask]

        print(f"[OK] DXY数据: {len(dxy_df)}条")
        return dxy_df

    except Exception as e:
        print(f"[WARNING] 获取DXY数据失败: {e}")
        return None

dxy_df = fetch_dxy_data(START_DATE, END_DATE)

if dxy_df is None:
    print("[WARNING] 将不使用DXY燃料增强")
    dxy_df = pd.DataFrame()  # 空DataFrame

print()

# ==================== 计算物理指标（验证5逻辑） ====================
print("正在计算物理指标...")

def calculate_physics_metrics(df):
    """计算物理指标：张力、加速度"""
    prices = df['close'].values

    # FFT + Hilbert
    d_prices = detrend(prices)
    coeffs = fft(d_prices)
    coeffs[8:] = 0
    filtered = ifft(coeffs).real

    analytic = hilbert(filtered)
    tension = np.imag(analytic)

    # 标准化
    if len(tension) > 1 and np.std(tension) > 0:
        tension_normalized = (tension - np.mean(tension)) / np.std(tension)
    else:
        tension_normalized = tension

    # 加速度
    acceleration = np.zeros_like(tension_normalized)
    for i in range(2, len(tension_normalized)):
        velocity = tension_normalized[i] - tension_normalized[i-1]
        prev_velocity = tension_normalized[i-1] - tension_normalized[i-2]
        acceleration[i] = velocity - prev_velocity

    return tension_normalized, acceleration

tension, acceleration = calculate_physics_metrics(df)

df_metrics = pd.DataFrame({
    'close': df['close'].values,
    'volume': df['volume'].values,
    'tension': tension,
    'acceleration': acceleration
}, index=df.index)

print(f"[OK] 计算完成: {len(df_metrics)}条")
print()

# ==================== 计算DXY燃料 ====================
print("正在计算DXY燃料...")

def calculate_dxy_fuel(dxy_df, current_date):
    """计算DXY燃料（验证5逻辑）"""
    if dxy_df is None or dxy_df.empty:
        return 0.0

    try:
        mask = dxy_df.index <= current_date
        available = dxy_df[mask]

        if len(available) < 3:
            return 0.0

        recent = available.tail(5)

        if len(recent) < 3:
            return 0.0

        closes = recent['Close'].values.astype(float)

        change_1 = (closes[-1] - closes[-2]) / closes[-2]
        change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1

        acceleration = change_1 - change_2
        fuel = -acceleration * 100

        return float(fuel)

    except Exception as e:
        return 0.0

# 为每个K线计算DXY燃料
dxy_fuels = []
for idx in df_metrics.index:
    fuel = calculate_dxy_fuel(dxy_df, idx)
    dxy_fuels.append(fuel)

df_metrics['dxy_fuel'] = dxy_fuels

print(f"[OK] DXY燃料计算完成")
print(f"  DXY燃料>0的K线数: {sum(1 for f in dxy_fuels if f > 0)}条")
print(f"  平均DXY燃料: {np.mean(dxy_fuels):.4f}")
print()

# ==================== 检测所有信号（验证5逻辑 + DXY） ====================
print("正在检测信号...")

def diagnose_regime(tension, acceleration, dxy_fuel=0.0):
    """诊断市场状态（完整验证5逻辑 + DXY燃料）"""
    TENSION_THRESHOLD = 0.35
    ACCEL_THRESHOLD = 0.02

    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        confidence = 0.7
        description = f"奇点看空(T={tension:.2f}≥{TENSION_THRESHOLD})"
        signal_type = 'BEARISH_SINGULARITY'

        if dxy_fuel > 0:
            confidence_boost = min(dxy_fuel * 0.01, 0.25)
            confidence += confidence_boost
            description += f" +DXY燃料({dxy_fuel:.2f})"

        return signal_type, confidence, description

    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        confidence = 0.6
        description = f"奇点看涨(T={tension:.2f}≤-{TENSION_THRESHOLD})"
        signal_type = 'BULLISH_SINGULARITY'

        if dxy_fuel > 0:
            confidence_boost = min(dxy_fuel * 0.01, 0.25)
            confidence += confidence_boost
            description += f" +DXY燃料({dxy_fuel:.2f})"

        return signal_type, confidence, description

    elif tension > 0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        description = f"高位震荡(T={tension:.2f}>0.3)"
        signal_type = 'HIGH_OSCILLATION'
        return signal_type, confidence, description

    elif tension < -0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        description = f"低位震荡(T={tension:.2f}<-0.3)"
        signal_type = 'LOW_OSCILLATION'
        return signal_type, confidence, description

    return None, 0.0, "无信号"

# 检测信号
all_signals = []
for idx, row in df_metrics.iterrows():
    tension_val = row['tension']
    accel_val = row['acceleration']
    dxy_fuel_val = row['dxy_fuel']

    signal_type, confidence, description = diagnose_regime(tension_val, accel_val, dxy_fuel_val)

    if signal_type is not None and confidence >= 0.6:
        all_signals.append({
            '时间': idx,
            '收盘价': row['close'],
            '信号类型': signal_type,
            '置信度': confidence,
            '描述': description,
            '张力': tension_val,
            '加速度': accel_val,
            '量能比率': 0.0,  # 稍后计算
            'EMA偏离%': 0.0,   # 稍后计算
            'DXY燃料': dxy_fuel_val,
            'V705通过': False,
            '过滤原因': None
        })

df_signals = pd.DataFrame(all_signals)

print(f"[OK] 检测到信号: {len(df_signals)}个")
print()

# ==================== 计算EMA和量能 ====================
print("正在计算EMA和量能...")

df_metrics['ema20'] = df_metrics['close'].ewm(span=20, adjust=False).mean()
df_metrics['price_vs_ema'] = (df_metrics['close'] - df_metrics['ema20']) / df_metrics['ema20']
df_metrics['avg_volume_20'] = df_metrics['volume'].rolling(20).mean()
df_metrics['volume_ratio'] = df_metrics['volume'] / df_metrics['avg_volume_20']

# 填充到信号DataFrame
for idx, signal in df_signals.iterrows():
    signal_time = signal['时间']
    if signal_time in df_metrics.index:
        df_signals.at[idx, '量能比率'] = df_metrics.loc[signal_time, 'volume_ratio']
        df_signals.at[idx, 'EMA偏离%'] = df_metrics.loc[signal_time, 'price_vs_ema']

print(f"[OK] EMA和量能计算完成")
print()

# ==================== V7.0.5过滤器 ====================
print("正在应用V7.0.5过滤器...")

def apply_v705_filter(signal_type, acceleration, volume_ratio, price_vs_ema):
    """V7.0.5入场过滤器"""
    if signal_type == 'HIGH_OSCILLATION':
        if price_vs_ema > 0.02:
            return False, f"牛市回调({price_vs_ema*100:.1f}%)"
        if acceleration >= 0:
            return False, f"无向下动能(a={acceleration:.3f})"
        if volume_ratio > 1.1:
            return False, f"高位放量({volume_ratio:.2f})"
        return True, "通过V7.0.5"

    elif signal_type == 'LOW_OSCILLATION':
        return True, "通过V7.0.5"

    elif signal_type == 'BULLISH_SINGULARITY':
        if volume_ratio > 0.95:
            return False, f"量能放大({volume_ratio:.2f})"
        if price_vs_ema > 0.05:
            return False, f"主升浪({price_vs_ema*100:.1f}%)"
        return True, "通过V7.0.5"

    elif signal_type == 'BEARISH_SINGULARITY':
        if price_vs_ema < -0.05:
            return False, f"主跌浪({price_vs_ema*100:.1f}%)"
        return True, "通过V7.0.5"

    return True, "通过V7.0.5"

# 应用过滤器
for idx, signal in df_signals.iterrows():
    should_pass, reason = apply_v705_filter(
        signal['信号类型'],
        signal['加速度'],
        signal['量能比率'],
        signal['EMA偏离%']
    )
    df_signals.at[idx, 'V705通过'] = should_pass
    df_signals.at[idx, '过滤原因'] = reason if not should_pass else None

v705_passed = df_signals[df_signals['V705通过'] == True]
v705_filtered = df_signals[df_signals['V705通过'] == False]

print(f"[OK] V7.0.5过滤结果:")
print(f"  通过V7.0.5: {len(v705_passed)}个 ({len(v705_passed)/len(df_signals)*100:.1f}%)")
print(f"  被过滤: {len(v705_filtered)}个 ({len(v705_filtered)/len(df_signals)*100:.1f}%)")
print()

# ==================== 保存数据 ====================
print("正在保存数据...")

# 1. 所有信号（原始）
df_signals.to_csv('step1_all_signals_with_dxy.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 已保存所有信号: step1_all_signals_with_dxy.csv ({len(df_signals)}个)")

# 2. V7.0.5通过的信号
v705_passed.to_csv('step1_entry_signals_with_dxy.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 已保存V7.0.5通过信号: step1_entry_signals_with_dxy.csv ({len(v705_passed)}个)")

# 3. 完整数据（含DXY燃料）
df_full = df_metrics.copy()
df_full.to_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
print(f"[OK] 已保存完整数据: step1_full_data_with_dxy.csv ({len(df_full)}条)")

print()
print("=" * 80)
print("[OK] 第一步完成！数据收集完成（含DXY燃料）")
print("=" * 80)
