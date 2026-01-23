# -*- coding: utf-8 -*-
"""
标注局部波峰波谷 - 小窗口
========================
"""

import pandas as pd
import numpy as np

print("="*120)
print("MARKING LOCAL PEAKS AND VALLEYS")
print("="*120)

# Load data
df = pd.read_csv('简单.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条数据")

# ============================================================================
# 局部极值：使用scipy + 手动检查用户期望的4个点
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Find Local Peaks and Valleys")
print("="*120)

from scipy.signal import argrelextrema

# 使用order=2找局部极值
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

print(f"\n使用scipy找到 {len(local_max_indices)} 个局部高点, {len(local_min_indices)} 个局部低点")

# 标注
peak_valley_labels = []
for i in range(len(df)):
    if i in local_max_indices:
        peak_valley_labels.append('高点')
    elif i in local_min_indices:
        peak_valley_labels.append('低点')
    else:
        peak_valley_labels.append('')

df['高低点标注'] = peak_valley_labels

# 检查用户期望的4个点
print("\n检查用户期望的关键点:")
print(f"{'行号':<6} {'时间':<18} {'收盘价':<10} {'标注':<10}")
print("-" * 80)

test_times = ['2025-08-20 16:00', '2025-08-21 16:00', '2025-08-22 04:00', '2025-08-22 08:00']
for time_str in test_times:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        label = df.loc[idx, '高低点标注']
        print(f"{idx:<6} {time_str:<18} {df.loc[idx, '收盘价']:<10.2f} {label:<10}")

actions = []
positions = []
current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']

    is_local_peak = (df.loc[i, '高低点标注'] == '高点')
    is_local_valley = (df.loc[i, '高低点标注'] == '低点')

    # 计算盈亏
    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (current_close - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_close) / entry_price * 100
    else:
        pnl_pct = 0

    # 决策逻辑
    action = ''

    if current_position == 'NONE':
        # 初始状态
        if i == 0:
            # 第一根，开多
            action = '开多'
            current_position = 'LONG'
            entry_price = current_close
        else:
            action = '观望'

    elif current_position == 'LONG':
        # 持多仓
        if is_local_peak:
            # 局部高点 → 平多反空
            action = f'平多/反空 (局部高点,盈利{pnl_pct:.2f}%)'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = f'继续持多 ({pnl_pct:+.2f}%)'

    elif current_position == 'SHORT':
        # 持空仓
        if is_local_valley:
            # 局部低点 → 平空反多
            action = f'平空/反多 (局部低点,盈利{pnl_pct:.2f}%)'
            current_position = 'LONG'
            entry_price = current_close
        else:
            action = f'继续持空 ({pnl_pct:+.2f}%)'

    actions.append(action)
    positions.append(current_position)

df['最优动作'] = actions
df['持仓状态'] = positions

# ============================================================================
# 验证用户例子
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Verify User's Example")
print("="*120)

test_cases = [
    ('2025-08-20 16:00', '平多/反空'),
    ('2025-08-21 16:00', '平空/反多'),
    ('2025-08-22 04:00', '平多/反空'),
    ('2025-08-22 08:00', '平空/反多')
]

print("\n验证用户标注:")
for time_str, expected in test_cases:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        print(f"\n{df.loc[idx, '时间']}: 收盘{df.loc[idx, '收盘价']:.2f}")
        print(f"  用户期望: {expected}")
        print(f"  算法结果: {df.loc[idx, '最优动作'][:30]}")

# ============================================================================
# 显示前20个
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Results (First 20)")
print("="*120)

print(f"\n{'序号':<6} {'时间':<18} {'收盘价':<10} {'持仓':<8} {'动作':<35}")
print("-" * 120)

for i in range(20):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    pos = df.loc[i, '持仓状态']
    action = df.loc[i, '最优动作'][:32]

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {pos:<8} {action:<35}")

# ============================================================================
# 保存
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Save Results")
print("="*120)

df[['时间', '收盘价', '持仓状态', '最优动作']].to_csv('局部极值标注.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 局部极值标注.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
