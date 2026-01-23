# -*- coding: utf-8 -*-
"""
使用合理阈值重新收集数据
===========================

基于数据分布的动态阈值
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
print("使用合理阈值重新收集数据")
print("=" * 80)

# 读取已有的正确加速度数据
df_full = pd.read_csv('step1_full_data_correct_accel.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

print(f"\n数据量: {len(df_full)}条")

# 分析加速度分布
accel = df_full['acceleration'].values
accel_std = np.std(accel)
accel_95 = np.percentile(np.abs(accel), 95)

print(f"\n加速度统计:")
print(f"  标准差: {accel_std:.6f}")
print(f"  95%分位绝对值: {accel_95:.6f}")

# 使用合理的阈值（基于95%分位数）
TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = accel_95  # 约0.024（动态阈值）

print(f"\n使用的阈值:")
print(f"  TENSION_THRESHOLD = {TENSION_THRESHOLD}")
print(f"  ACCEL_THRESHOLD = {ACCEL_THRESHOLD:.6f} (95%分位数)")

# 检测信号
def diagnose_regime(tension, acceleration):
    """诊断市场状态（合理阈值）"""

    # 1. 看跌奇点
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        return 'BEARISH_SINGULARITY', '做空', 0.7

    # 2. 看涨奇点
    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        return 'BULLISH_SINGULARITY', '做多', 0.6

    # 3. 高位震荡
    elif tension > 0.3 and abs(acceleration) < 0.01:
        return 'HIGH_OSCILLATION', '做空', 0.6

    # 4. 低位震荡
    elif tension < -0.3 and abs(acceleration) < 0.01:
        return 'LOW_OSCILLATION', '做多', 0.6

    return None, None, 0.0

# 检测所有信号
all_signals = []
for idx, row in df_full.iterrows():
    signal_type, direction, confidence = diagnose_regime(
        row['tension'], row['acceleration']
    )

    if signal_type is not None and confidence >= 0.6:
        all_signals.append({
            '时间': row['timestamp'],
            '收盘价': row['close'],
            '信号类型': signal_type,
            '方向': direction,
            '置信度': confidence,
            '张力': row['tension'],
            '加速度': row['acceleration'],
            '量能比率': row.get('volume_ratio', 1.0),
            'EMA偏离%': 0.0,
            'DXY燃料': row.get('dxy_fuel', 0.0),
        })

df_all_signals = pd.DataFrame(all_signals)

print(f"\n检测到信号总数: {len(df_all_signals)}个")
print(f"信号类型分布:")
print(df_all_signals['信号类型'].value_counts())

# 计算EMA
df_full['ema20'] = df_full['close'].ewm(span=20, adjust=False).mean()
df_full['price_vs_ema'] = (df_full['close'] - df_full['ema20']) / df_full['ema20'] * 100

# 填充EMA和量能
for idx, signal in df_all_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    if signal_time in df_full['timestamp'].values:
        df_all_signals.at[idx, 'EMA偏离%'] = df_full.loc[df_full['timestamp'] == signal_time, 'price_vs_ema'].values[0]

# V7.0.5过滤器
def apply_v705_filter(signal_type, volume_ratio, price_vs_ema):
    """V7.0.5入场过滤器"""
    if volume_ratio < 0.5:
        return False, '量能过低'
    if abs(price_vs_ema) > 5:
        return False, 'EMA偏离过大'
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

print(f"\nV7.0.5通过信号: {len(df_entry_signals)}个")
print(f"通过信号类型分布:")
print(df_entry_signals['信号类型'].value_counts())

# 保存数据
df_all_signals.to_csv('step1_all_signals_reasonable.csv', index=False, encoding='utf-8-sig')
df_entry_signals.to_csv('step1_entry_signals_reasonable.csv', index=False, encoding='utf-8-sig')

print(f"\n已保存:")
print(f"  step1_all_signals_reasonable.csv")
print(f"  step1_entry_signals_reasonable.csv")

print("\n" + "=" * 80)
print("[OK] 数据收集完成（使用合理阈值）")
print("=" * 80)
