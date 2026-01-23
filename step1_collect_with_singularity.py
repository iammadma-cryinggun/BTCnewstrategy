# -*- coding: utf-8 -*-
"""
重新收集数据 - 包含奇点信号
==========================

调整阈值到合理范围（基于数据分布）
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
print("重新收集数据 - 包含奇点信号（调整阈值）")
print("=" * 80)

# 读取已有数据
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

# 计算EMA
df_full['ema20'] = df_full['close'].ewm(span=20, adjust=False).mean()
df_full['ema_deviation'] = (df_full['close'] - df_full['ema20']) / df_full['ema20'] * 100

# 计算DXY趋势（用于V7.0.5过滤）
df_full['dxy_trend'] = df_full['dxy_fuel'].rolling(5).mean()

print(f"\n完整数据: {len(df_full)}条")

# ==================== 重新检测信号（合理阈值） ====================
print("\n正在重新检测信号...")

# 基于数据分布的合理阈值
tension_75 = df_full['tension'].quantile(0.75)
tension_25 = df_full['tension'].quantile(0.25)
accel_75 = df_full['acceleration'].quantile(0.75)
accel_25 = df_full['acceleration'].quantile(0.25)

print(f"\n实际数据分布:")
print(f"张力 75%分位: {tension_75:.4f}, 25%分位: {tension_25:.4f}")
print(f"加速度 75%分位: {accel_75:.6f}, 25%分位: {accel_25:.6f}")

# 设置合理阈值（使用75%分位数）
TENSION_THRESHOLD = abs(tension_75)  # 约0.35-0.4
ACCEL_THRESHOLD = abs(accel_75) * 2  # 约0.0005-0.0006

print(f"\n使用的阈值:")
print(f"TENSION_THRESHOLD = {TENSION_THRESHOLD:.4f}")
print(f"ACCEL_THRESHOLD = {ACCEL_THRESHOLD:.6f}")

def diagnose_regime_with_singularity(tension, acceleration, dxy_fuel=0.0):
    """诊断市场状态（完整验证5逻辑 + 奇点信号）"""

    # 1. 看跌奇点（高张力 + 负加速度）
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        confidence = 0.7
        signal_type = 'BEARISH_SINGULARITY'
        return signal_type, confidence, '做空'

    # 2. 看涨奇点（低张力 + 正加速度）
    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        confidence = 0.7
        signal_type = 'BULLISH_SINGULARITY'
        return signal_type, confidence, '做多'

    # 3. 高位震荡（高张力 + 低加速度）
    elif tension > 0.3 and abs(acceleration) < 0.0003:
        confidence = 0.6
        signal_type = 'HIGH_OSCILLATION'
        return signal_type, confidence, '做空'

    # 4. 低位震荡（低张力 + 低加速度）
    elif tension < -0.3 and abs(acceleration) < 0.0003:
        confidence = 0.6
        signal_type = 'LOW_OSCILLATION'
        return signal_type, confidence, '做多'

    return None, 0.0, None

# 检测所有信号
all_signals = []
for idx, row in df_full.iterrows():
    tension_val = row['tension']
    accel_val = row['acceleration']
    dxy_fuel_val = row['dxy_fuel']
    volume_ratio = row['volume_ratio']
    ema_dev = row['ema_deviation']

    signal_type, confidence, direction = diagnose_regime_with_singularity(
        tension_val, accel_val, dxy_fuel_val
    )

    if signal_type is not None:
        # V7.0.5 过滤器
        v705_pass = True
        filter_reason = None

        # 1. 量能检查
        if volume_ratio < 0.5:
            v705_pass = False
            filter_reason = '量能过低'

        # 2. EMA偏离检查
        if abs(ema_dev) > 5:
            v705_pass = False
            filter_reason = 'EMA偏离过大'

        # 3. 趋势检查
        if signal_type in ['BEARISH_SINGULARITY', 'HIGH_OSCILLATION']:
            # 做空信号：EMA应该向上（价格在高位）
            if ema_dev < -2:
                v705_pass = False
                filter_reason = '趋势不符'
        else:
            # 做多信号：EMA应该向下（价格在低位）
            if ema_dev > 2:
                v705_pass = False
                filter_reason = '趋势不符'

        all_signals.append({
            '时间': row['timestamp'],
            '收盘价': row['close'],
            '信号类型': signal_type,
            '方向': direction,
            '置信度': confidence,
            '张力': tension_val,
            '加速度': accel_val,
            '量能比率': volume_ratio,
            'EMA偏离%': ema_dev,
            'DXY燃料': dxy_fuel_val,
            'V705通过': v705_pass,
            '过滤原因': filter_reason
        })

df_all_signals = pd.DataFrame(all_signals)

print(f"\n检测到信号总数: {len(df_all_signals)}个")
print(f"信号类型分布:")
print(df_all_signals['信号类型'].value_counts())

# 保存所有信号
df_all_signals.to_csv('step1_all_signals_with_singularity.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: step1_all_signals_with_singularity.csv")

# 只保留V7.0.5通过的信号
df_entry_signals = df_all_signals[df_all_signals['V705通过'] == True].copy()

print(f"\nV7.0.5通过信号: {len(df_entry_signals)}个")
print(f"通过信号类型分布:")
print(df_entry_signals['信号类型'].value_counts())

df_entry_signals.to_csv('step1_entry_signals_with_singularity.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: step1_entry_signals_with_singularity.csv")

print("\n" + "=" * 80)
print("[OK] 数据收集完成")
print("=" * 80)
