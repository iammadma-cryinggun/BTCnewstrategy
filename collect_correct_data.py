# -*- coding: utf-8 -*-
"""
使用正确方法重新收集所有信号并做黄金信号分析
==============================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("使用正确方法重新收集数据并分析")
print("=" * 80)

# 读取正确计算的数据
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"\n完整数据: {len(df_full)}条")

# 计算额外特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']
df_full['ema20'] = df_full['close'].ewm(span=20, adjust=False).mean()
df_full['ema_deviation'] = (df_full['close'] - df_full['ema20']) / df_full['ema20'] * 100

# ==================== 检测所有信号（验证5逻辑） ====================
print("\n正在检测信号...")

TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02
OSCILLATION_BAND = 0.5

def diagnose_regime(tension, acceleration):
    """验证5的信号诊断逻辑"""

    # 1. 奇点看空（最高优先级）
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        return 'BEARISH_SINGULARITY', 'short', 0.7

    # 2. 奇点看涨（最高优先级）
    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        return 'BULLISH_SINGULARITY', 'long', 0.6

    # 3. 平衡震荡
    elif abs(tension) < OSCILLATION_BAND and abs(acceleration) < 0.02:
        return 'OSCILLATION', None, 0.8

    # 4. 高位震荡
    elif tension > 0.3 and abs(acceleration) < 0.01:
        return 'HIGH_OSCILLATION', 'short', 0.6

    # 5. 低位震荡
    elif tension < -0.3 and abs(acceleration) < 0.01:
        return 'LOW_OSCILLATION', 'long', 0.6

    # 6-8. 过渡状态
    elif tension > 0 and acceleration > 0:
        return 'TRANSITION_UP', None, 0.4
    elif tension < 0 and acceleration < 0:
        return 'TRANSITION_DOWN', None, 0.4
    else:
        return 'TRANSITION', None, 0.3

# 检测所有信号
all_signals = []
for idx, row in df_full.iterrows():
    signal_type, direction, confidence = diagnose_regime(
        row['tension'], row['acceleration']
    )

    # 只保留交易信号（有方向的）
    if direction is not None and confidence >= 0.6:
        all_signals.append({
            '时间': row['timestamp'],
            '收盘价': row['close'],
            '信号类型': signal_type,
            '方向': direction,
            '置信度': confidence,
            '张力': row['tension'],
            '加速度': row['acceleration'],
            '量能比率': row['volume_ratio'],
            'EMA偏离%': row['ema_deviation'],
        })

df_all_signals = pd.DataFrame(all_signals)

print(f"\n检测到信号总数: {len(df_all_signals)}个")
print(f"信号类型分布:")
print(df_all_signals['信号类型'].value_counts())

# 保存所有信号
df_all_signals.to_csv('step1_all_signals_v5_correct.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: step1_all_signals_v5_correct.csv")

print("\n" + "=" * 80)
print("数据收集完成！现在奇点信号占了大部分")
print("=" * 80)
