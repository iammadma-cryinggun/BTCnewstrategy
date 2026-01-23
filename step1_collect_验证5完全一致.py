# -*- coding: utf-8 -*-
"""
按照验证5.py逻辑完全一致的版本
=================================
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("按照验证5.py逻辑收集数据")
print("=" * 80)

# ==================== 配置 ====================
START_DATE = '2025-08-10'
END_DATE = '2026-01-20'

# 验证5的参数
TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02
OSCILLATION_BAND = 0.5

print(f"\n时间范围: {START_DATE} 到 {END_DATE}")
print(f"参数: TENSION_THRESHOLD={TENSION_THRESHOLD}, ACCEL_THRESHOLD={ACCEL_THRESHOLD}")
print()

# ==================== 获取数据 ====================
print("正在获取BTC 4H数据...")

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
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df[['open', 'high', 'low', 'close', 'volume']]

df = fetch_btc_data(START_DATE, END_DATE)
print(f"[OK] BTC数据: {len(df)}条")
print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")

# ==================== 获取DXY数据 ====================
print("\n正在获取DXY数据...")

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

dxy_df = fetch_dxy_data(START_DATE, END_DATE)
if dxy_df is not None:
    print(f"[OK] DXY数据: {len(dxy_df)}条")
else:
    print("[WARNING] DXY数据获取失败")

# ==================== 计算物理指标（验证5逻辑） ====================
print("\n正在计算物理指标（完全按照验证5.py）...")

def calculate_tension_acceleration_verification5(prices):
    """
    计算张力和加速度（验证5逻辑）

    关键：加速度 = 张力的二阶差分（不是价格变化率的变化率！）
    """
    if len(prices) < 60:
        return None, None

    try:
        prices_array = np.array(prices, dtype=np.float64)
        d_prices = detrend(prices_array)

        # FFT滤波
        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real

        # Hilbert变换
        analytic = hilbert(filtered)
        tension = np.imag(analytic)

        # 标准化张力
        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension

        # 【关键】计算加速度：张力的二阶差分（验证5的逻辑）
        acceleration = np.zeros_like(norm_tension)
        for i in range(2, len(norm_tension)):
            current_tension = norm_tension[i]
            prev_tension = norm_tension[i-1]
            prev2_tension = norm_tension[i-2]

            # 速度 = 张力的一阶差分
            velocity = current_tension - prev_tension

            # 加速度 = 速度的一阶差分（张力的二阶差分）
            acceleration[i] = velocity - (prev_tension - prev2_tension)

        return norm_tension, acceleration

    except:
        return None, None

# 计算所有K线的物理指标
tensions = []
accelerations = []

# 使用滚动窗口计算（每个时刻用前100个数据点）
window_size = 100
for i in range(window_size, len(df)):
    window_prices = df['close'].iloc[i-window_size:i].values
    tension_window, accel_window = calculate_tension_acceleration_verification5(window_prices)

    if tension_window is not None and accel_window is not None:
        tensions.append(tension_window[-1])
        accelerations.append(accel_window[-1])

# 前window_size个点用全部可用数据计算
for i in range(window_size):
    window_prices = df['close'].iloc[:i+1].values
    if len(window_prices) >= 3:
        tension_window, accel_window = calculate_tension_acceleration_verification5(window_prices)
        if tension_window is not None and accel_window is not None:
            tensions.insert(i, tension_window[-1])
            accelerations.insert(i, accel_window[-1])

# 创建指标DataFrame
df_metrics = pd.DataFrame({
    'close': df['close'].values,
    'volume': df['volume'].values,
    'tension': tensions,
    'acceleration': accelerations
}, index=df.index)

print(f"[OK] 计算完成: {len(df_metrics)}条")
print(f"  张力范围: [{df_metrics['tension'].min():.4f}, {df_metrics['tension'].max():.4f}]")
print(f"  加速度范围: [{df_metrics['acceleration'].min():.6f}, {df_metrics['acceleration'].max():.6f}]")

# ==================== 计算DXY燃料 ====================
print("\n正在计算DXY燃料...")

def calculate_dxy_fuel(dxy_df, current_date):
    """计算DXY燃料（验证5逻辑）"""
    if dxy_df is None or dxy_df.empty:
        return 0.0

    try:
        mask = dxy_df.index <= current_date
        recent = dxy_df[mask].tail(5)

        if len(recent) < 3:
            return 0.0

        closes = recent['Close'].values.astype(float)

        change_1 = (closes[-1] - closes[-2]) / closes[-2]
        change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1

        acceleration = change_1 - change_2
        fuel = -acceleration * 100

        return float(fuel)
    except:
        return 0.0

# 为每个K线计算DXY燃料
dxy_fuels = []
for idx in df_metrics.index:
    fuel = calculate_dxy_fuel(dxy_df, idx)
    dxy_fuels.append(fuel)

df_metrics['dxy_fuel'] = dxy_fuels

print(f"[OK] DXY燃料计算完成")

# ==================== 检测所有信号（验证5逻辑） ====================
print("\n正在检测信号（验证5逻辑）...")

def diagnose_regime_verification5(tension, acceleration, dxy_fuel=0.0):
    """诊断市场状态（验证5逻辑）"""

    # 1. 奇点看空
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        if dxy_fuel > 0.1:
            return "BEARISH_SINGULARITY", "强奇点看空 (宏观失速)", 0.9
        else:
            return "BEARISH_SINGULARITY", "奇点看空 (动力失速)", 0.7

    # 2. 奇点看涨
    if tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        if dxy_fuel > 0.2:
            return "BULLISH_SINGULARITY", "超强奇点看涨 (燃料爆炸)", 0.95
        elif dxy_fuel > 0:
            return "BULLISH_SINGULARITY", "强奇点看涨 (动力回归)", 0.8
        else:
            return "BULLISH_SINGULARITY", "奇点看涨 (弹性释放)", 0.6

    # 3. 震荡
    if abs(tension) < OSCILLATION_BAND and abs(acceleration) < 0.02:
        return "OSCILLATION", "系统平衡 (震荡收敛)", 0.8

    # 4. 高位震荡
    if tension > 0.3 and abs(acceleration) < 0.01:
        return "HIGH_OSCILLATION", "高位震荡 (风险积聚)", 0.6

    # 5. 低位震荡
    if tension < -0.3 and abs(acceleration) < 0.01:
        return "LOW_OSCILLATION", "低位震荡 (机会积聚)", 0.6

    # 6. 过渡状态
    if tension > 0 and acceleration > 0:
        return "TRANSITION_UP", "向上过渡 (蓄力)", 0.4
    elif tension < 0 and acceleration < 0:
        return "TRANSITION_DOWN", "向下过渡 (泄力)", 0.4

    return "TRANSITION", "体制切换中", 0.3

# 检测信号
all_signals = []
for idx, row in df_metrics.iterrows():
    tension_val = row['tension']
    accel_val = row['acceleration']
    dxy_fuel_val = row['dxy_fuel']

    signal_type, description, confidence = diagnose_regime_verification5(tension_val, accel_val, dxy_fuel_val)

    # 只记录置信度>=0.6的信号
    if confidence >= 0.6:
        all_signals.append({
            '时间': idx,
            '收盘价': row['close'],
            '信号类型': signal_type,
            '置信度': confidence,
            '描述': description,
            '张力': tension_val,
            '加速度': accel_val,
            'DXY燃料': dxy_fuel_val
        })

df_signals = pd.DataFrame(all_signals)

print(f"[OK] 检测到信号: {len(df_signals)}个")
print(f"\n信号类型分布:")
print(df_signals['信号类型'].value_counts())

# ==================== 保存数据 ====================
print("\n正在保存数据...")

# 1. 完整数据
df_full = df_metrics.copy()
df_full.to_csv('step1_full_data_验证5标准.csv', encoding='utf-8-sig')
print(f"[OK] 已保存完整数据: step1_full_data_验证5标准.csv ({len(df_full)}条)")

# 2. 所有信号
df_signals.to_csv('step1_all_signals_验证5标准.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 已保存所有信号: step1_all_signals_验证5标准.csv ({len(df_signals)}个)")

print("\n" + "=" * 80)
print("[OK] 数据收集完成（完全按照验证5.py逻辑）")
print("=" * 80)
