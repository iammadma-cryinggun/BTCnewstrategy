# -*- coding: utf-8 -*-
"""
使用验证5的2年时间范围收集数据
=================================
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert
import requests
from scipy.signal import detrend
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("使用验证5的2年时间范围重新收集数据")
print("=" * 80)

# 验证5使用的时间范围
START_DATE = '2024-01-01'
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

    all_data = []
    current = params['startTime']

    while current < params['endTime']:
        resp = requests.get(url, params={**params, 'startTime': current}, timeout=15)
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1][0] + 1

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['open', 'high', 'low', 'close', 'volume']]

print("正在获取BTC数据...")
df = fetch_btc_data(START_DATE, END_DATE)
print(f"[OK] BTC数据: {len(df)}条")
print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")

# ==================== 使用验证5的计算方法 ====================
print("\n正在计算物理指标（验证5方法）...")

def calculate_physics_metrics_v5(df):
    """使用验证5的计算方法"""

    prices = df['close'].values

    # FFT + Hilbert
    d_prices = detrend(prices)
    coeffs = fft(d_prices)
    coeffs[8:] = 0
    filtered = ifft(coeffs).real

    analytic = hilbert(filtered)
    tension = np.imag(analytic)

    # 标准化（使用全部数据的统计量）
    if len(tension) > 1 and np.std(tension) > 0:
        tension_normalized = (tension - np.mean(tension)) / np.std(tension)
    else:
        tension_normalized = tension

    # 加速度：张力的二阶差分（验证5方法）
    acceleration = np.zeros_like(tension_normalized)
    for i in range(2, len(tension_normalized)):
        velocity = tension_normalized[i] - tension_normalized[i-1]
        prev_velocity = tension_normalized[i-1] - tension_normalized[i-2]
        acceleration[i] = velocity - prev_velocity

    return tension_normalized, acceleration

tension, acceleration = calculate_physics_metrics_v5(df)

print(f"[OK] 计算完成")
print(f"  张力范围: [{tension.min():.4f}, {tension.max():.4f}]")
print(f"  加速度范围: [{acceleration.min():.6f}, {acceleration.max():.6f}]")

# ==================== 检测信号 ====================
print("\n正在检测信号...")

TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02

print(f"阈值: TENSION={TENSION_THRESHOLD}, ACCEL={ACCEL_THRESHOLD}")

# 统计满足条件的次数
bearish_count = sum((tension > TENSION_THRESHOLD) & (acceleration < -ACCEL_THRESHOLD))
bullish_count = sum((tension < -TENSION_THRESHOLD) & (acceleration > ACCEL_THRESHOLD))
high_osc_count = sum((tension > 0.3) & (np.abs(acceleration) < 0.01))
low_osc_count = sum((tension < -0.3) & (np.abs(acceleration) < 0.01))

print(f"\n满足信号条件次数:")
print(f"  BEARISH_SINGULARITY（奇点看空）: {bearish_count}次")
print(f"  BULLISH_SINGULARITY（奇点看涨）: {bullish_count}次")
print(f"  HIGH_OSCILLATION（高位震荡）: {high_osc_count}次")
print(f"  LOW_OSCILLATION（低位震荡）: {low_osc_count}次")
print(f"  总计: {bearish_count + bullish_count + high_osc_count + low_osc_count}次")

# 检查加速度分布
print(f"\n加速度分布:")
print(f"  <-0.02: {sum(acceleration < -0.02)}次")
print(f"  >0.02: {sum(acceleration > 0.02)}次")
print(f"  <-0.01: {sum(acceleration < -0.01)}次")
print(f"  >0.01: {sum(acceleration > 0.01)}次")

# 保存数据
df_metrics = pd.DataFrame({
    'close': df['close'].values,
    'volume': df['volume'].values,
    'tension': tension,
    'acceleration': acceleration
}, index=df.index)

df_metrics.to_csv('step1_full_data_2years.csv', index=True, encoding='utf-8-sig')
print(f"\n已保存: step1_full_data_2years.csv")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
