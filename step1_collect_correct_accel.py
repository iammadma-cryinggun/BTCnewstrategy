# -*- coding: utf-8 -*-
"""
重新收集数据 - 使用正确的加速度计算
===================================

加速度 = 价格变化率的变化率（不是张力的二阶差分！）
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert
import requests
from io import StringIO
from scipy.signal import detrend
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("重新收集数据 - 使用正确的加速度计算")
print("=" * 80)

# ==================== 获取数据 ====================
START_DATE = '2025-07-25'
END_DATE = '2026-01-12'

def fetch_btc_data(start_date, end_date):
    """获取BTC 4H数据"""
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '4h',
        'startTime': int(pd.Timestamp(start_date).timestamp() * 1000),
        'endTime': int(pd.Timestamp(end_date).timestamp() * 1000),
        'limit': 1000
    }

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        raise Exception(f"获取数据失败: {resp.status_code}")

    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['open', 'high', 'low', 'close', 'volume']]

def fetch_dxy_data(start_date, end_date):
    """获取DXY数据"""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None

        dxy_df = pd.read_csv(StringIO(resp.text))
        dxy_df['observation_date'] = pd.to_datetime(dxy_df['observation_date'])
        dxy_df.set_index('observation_date', inplace=True)
        dxy_df.rename(columns={'DTWEXBGS': 'Close'}, inplace=True)
        dxy_df = dxy_df.dropna()
        dxy_df['Close'] = pd.to_numeric(dxy_df['Close'], errors='coerce')

        mask = (dxy_df.index >= start_date) & (dxy_df.index <= end_date)
        dxy_df = dxy_df[mask]

        return dxy_df
    except:
        return None

print("正在获取数据...")
df = fetch_btc_data(START_DATE, END_DATE)
print(f"[OK] BTC数据: {len(df)}条")

dxy_df = fetch_dxy_data(START_DATE, END_DATE)
if dxy_df is not None:
    print(f"[OK] DXY数据: {len(dxy_df)}条")
else:
    print("[WARNING] DXY数据获取失败")
    dxy_df = pd.DataFrame()

# ==================== 计算物理指标（验证5逻辑） ====================
print("\n正在计算物理指标...")

def calculate_physics_metrics_correct(df):
    """计算物理指标：张力、加速度（正确方法）"""

    prices = df['close'].values

    # 1. 张力：FFT + Hilbert
    d_prices = detrend(prices)
    coeffs = fft(d_prices)
    coeffs[8:] = 0
    filtered = ifft(coeffs).real

    analytic = hilbert(filtered)
    tension = np.imag(analytic)

    # 标准化张力
    if len(tension) > 1 and np.std(tension) > 0:
        tension_normalized = (tension - np.mean(tension)) / np.std(tension)
    else:
        tension_normalized = tension

    # 2. 加速度：价格变化率的变化率（正确方法！）
    # 速度 = 价格变化率
    price_velocity = np.zeros_like(prices, dtype=float)
    for i in range(1, len(prices)):
        price_velocity[i] = (prices[i] - prices[i-1]) / prices[i-1]

    # 加速度 = 速度的变化率
    acceleration = np.zeros_like(prices, dtype=float)
    for i in range(2, len(prices)):
        acceleration[i] = price_velocity[i] - price_velocity[i-1]

    return tension_normalized, acceleration

tension, acceleration = calculate_physics_metrics_correct(df)

df_metrics = pd.DataFrame({
    'close': df['close'].values,
    'volume': df['volume'].values,
    'tension': tension,
    'acceleration': acceleration
}, index=df.index)

print(f"[OK] 计算完成: {len(df_metrics)}条")
print(f"  张力范围: [{tension.min():.4f}, {tension.max():.4f}]")
print(f"  加速度范围: [{acceleration.min():.6f}, {acceleration.max():.6f}]")

# ==================== 计算DXY燃料 ====================
print("\n正在计算DXY燃料...")

def calculate_dxy_fuel(dxy_df, current_date):
    """计算DXY燃料"""
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

        accel = change_1 - change_2
        fuel = -accel * 100

        return float(fuel)
    except:
        return 0.0

dxy_fuels = []
for idx in df_metrics.index:
    fuel = calculate_dxy_fuel(dxy_df, idx)
    dxy_fuels.append(fuel)

df_metrics['dxy_fuel'] = dxy_fuels
print(f"[OK] DXY燃料计算完成")

# ==================== 检测所有信号（验证5逻辑 + 正确加速度） ====================
print("\n正在检测信号...")

TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02

print(f"阈值设置:")
print(f"  TENSION_THRESHOLD = {TENSION_THRESHOLD}")
print(f"  ACCEL_THRESHOLD = {ACCEL_THRESHOLD}")

def diagnose_regime(tension, acceleration, dxy_fuel=0.0):
    """诊断市场状态（验证5逻辑 + 正确加速度）"""

    # 1. 看跌奇点
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        confidence = 0.7
        signal_type = 'BEARISH_SINGULARITY'
        if dxy_fuel > 0:
            confidence_boost = min(dxy_fuel * 0.01, 0.25)
            confidence += confidence_boost
        return signal_type, confidence, '做空'

    # 2. 看涨奇点
    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        confidence = 0.6
        signal_type = 'BULLISH_SINGULARITY'
        if dxy_fuel > 0:
            confidence_boost = min(dxy_fuel * 0.01, 0.25)
            confidence += confidence_boost
        return signal_type, confidence, '做多'

    # 3. 高位震荡
    elif tension > 0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        signal_type = 'HIGH_OSCILLATION'
        return signal_type, confidence, '做空'

    # 4. 低位震荡
    elif tension < -0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        signal_type = 'LOW_OSCILLATION'
        return signal_type, confidence, '做多'

    return None, 0.0, None

# 检测信号
all_signals = []
for idx, row in df_metrics.iterrows():
    signal_type, confidence, direction = diagnose_regime(
        row['tension'], row['acceleration'], row['dxy_fuel']
    )

    if signal_type is not None and confidence >= 0.6:
        all_signals.append({
            '时间': idx,
            '收盘价': row['close'],
            '信号类型': signal_type,
            '方向': direction,
            '置信度': confidence,
            '张力': row['tension'],
            '加速度': row['acceleration'],
            'DXY燃料': row['dxy_fuel'],
            '量能比率': 0.0,
            'EMA偏离%': 0.0,
        })

df_all_signals = pd.DataFrame(all_signals)

print(f"\n检测到信号总数: {len(df_all_signals)}个")
print(f"信号类型分布:")
print(df_all_signals['信号类型'].value_counts())

# ==================== 计算EMA和量能 ====================
print("\n正在计算EMA和量能...")
df_metrics['ema20'] = df_metrics['close'].ewm(span=20, adjust=False).mean()
df_metrics['price_vs_ema'] = (df_metrics['close'] - df_metrics['ema20']) / df_metrics['ema20'] * 100
df_metrics['avg_volume_20'] = df_metrics['volume'].rolling(20).mean()
df_metrics['volume_ratio'] = df_metrics['volume'] / df_metrics['avg_volume_20']

for idx, signal in df_all_signals.iterrows():
    signal_time = signal['时间']
    if signal_time in df_metrics.index:
        df_all_signals.at[idx, '量能比率'] = df_metrics.loc[signal_time, 'volume_ratio']
        df_all_signals.at[idx, 'EMA偏离%'] = df_metrics.loc[signal_time, 'price_vs_ema']

# ==================== V7.0.5过滤器 ====================
print("\n正在应用V7.0.5过滤器...")

def apply_v705_filter(signal_type, volume_ratio, price_vs_ema):
    """V7.0.5入场过滤器"""
    # 1. 量能检查
    if volume_ratio < 0.5:
        return False, '量能过低'

    # 2. EMA偏离检查
    if abs(price_vs_ema) > 5:
        return False, 'EMA偏离过大'

    # 3. 趋势检查
    if signal_type in ['BEARISH_SINGULARITY', 'HIGH_OSCILLATION']:
        if price_vs_ema < -2:
            return False, '趋势不符'
    else:
        if price_vs_ema > 2:
            return False, '趋势不符'

    return True, None

for idx, signal in df_all_signals.iterrows():
    v705_pass, filter_reason = apply_v705_filter(
        signal['信号类型'], signal['量能比率'], signal['EMA偏离%']
    )
    df_all_signals.at[idx, 'V705通过'] = v705_pass
    df_all_signals.at[idx, '过滤原因'] = filter_reason

df_entry_signals = df_all_signals[df_all_signals['V705通过'] == True]

print(f"V7.0.5通过信号: {len(df_entry_signals)}个")
print(f"通过信号类型分布:")
print(df_entry_signals['信号类型'].value_counts())

# ==================== 保存数据 ====================
df_metrics.to_csv('step1_full_data_correct_accel.csv', index=True, encoding='utf-8-sig')
print(f"\n已保存: step1_full_data_correct_accel.csv")

df_all_signals.to_csv('step1_all_signals_correct_accel.csv', index=False, encoding='utf-8-sig')
print(f"已保存: step1_all_signals_correct_accel.csv")

df_entry_signals.to_csv('step1_entry_signals_correct_accel.csv', index=False, encoding='utf-8-sig')
print(f"已保存: step1_entry_signals_correct_accel.csv")

print("\n" + "=" * 80)
print("[OK] 数据收集完成（使用正确的加速度计算）")
print("=" * 80)
