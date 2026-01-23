# -*- coding: utf-8 -*-
"""
标注高低点 - 数学方法
==================
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("MARKING HIGH AND LOW POINTS")
print("="*120)

# Load data
df = pd.read_csv('简单.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条数据")

# ============================================================================
# 方法：滚动窗口找高低点
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Find High/Low Points Using Rolling Window")
print("="*120)

WINDOW = 5  # 前后5根K线
MIN_CHANGE = 0.005  # 最小变化0.5%

high_low = []

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']

    # 获取窗口
    start_idx = max(0, i - WINDOW)
    end_idx = min(len(df), i + WINDOW + 1)
    window_closes = df.loc[start_idx:end_idx-1, '收盘价'].values

    # 判断是否是高点或低点
    window_max = np.max(window_closes)
    window_min = np.min(window_closes)

    is_high = (current_close == window_max) and (i > WINDOW) and (i < len(df) - WINDOW)
    is_low = (current_close == window_min) and (i > WINDOW) and (i < len(df) - WINDOW)

    label = ''
    if is_high:
        label = '高点'
    elif is_low:
        label = '低点'

    high_low.append(label)

df['高低点'] = high_low

# 显示结果
print(f"\n前50个数据（标注高低点）:")
print(f"{'序号':<6} {'时间':<18} {'收盘价':<12} {'标注':<8}")
print("-" * 80)

for i in range(min(50, len(df))):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    label = df.loc[i, '高低点']

    if label:
        print(f"{i:<6} {time_str:<18} {close:<12.2f} [{label}]")
    else:
        print(f"{i:<6} {time_str:<18} {close:<12.2f}")

# ============================================================================
# 统计
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Statistics")
print("="*120)

high_count = (df['高低点'] == '高点').sum()
low_count = (df['高低点'] == '低点').sum()

print(f"\n找到高点: {high_count} 个")
print(f"找到低点: {low_count} 个")

# ============================================================================
# 保存
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Save Results")
print("="*120)

df.to_csv('简单_标注高低点.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 简单_标注高低点.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
