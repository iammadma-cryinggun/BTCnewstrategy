# -*- coding: utf-8 -*-
"""
数学方法找真正的波峰波谷
========================

纯数学视角：
1. 计算价格的一阶差分（变化率）
2. 找变化率的符号反转点
3. 找局部极大值和极小值
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("MATHEMATICAL PEAK/VALLEY DETECTION")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 方法1: 局部极值（scipy.signal.argrelextrema）
# ============================================================================
print("\n" + "="*120)
print("METHOD 1: Local Extrema (Mathematical)")
print("="*120)

# 找局部高点（order=N表示前后N个点）
order = 5
local_max = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

print(f"\n找到 {len(local_max)} 个局部高点，{len(local_min)} 个局部低点")

# 显示前20个数据，标注波峰波谷
print(f"\n前30个数据点（标注波峰波谷）:")
print(f"{'序号':<6} {'时间':<18} {'收盘价':<10} {'类型':<10}")
print("-" * 80)

for i in range(30):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']

    marker = ''
    if i in local_max:
        marker = '波峰★'
    elif i in local_min:
        marker = '波谷★'

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {marker:<10}")

# ============================================================================
# 方法2: 一阶差分符号变化
# ============================================================================
print("\n" + "="*120)
print("METHOD 2: First Derivative Sign Change")
print("="*120)

# 计算一阶差分
df['价格变化'] = df['收盘价'].diff()
df['变化方向'] = df['价格变化'].apply(lambda x: 'UP' if x > 0 else 'DOWN' if x < 0 else 'FLAT')

# 找方向变化的点
direction_changes = []
for i in range(1, len(df)):
    if df.loc[i, '变化方向'] != df.loc[i-1, '变化方向'] and df.loc[i, '变化方向'] != 'FLAT':
        direction_changes.append(i)

print(f"\n找到 {len(direction_changes)} 个方向变化点")
print("\n前20个方向变化点:")
print(f"{'序号':<6} {'时间':<18} {'收盘价':<10} {'从':<6} {'到':<6}")
print("-" * 80)

for i in direction_changes[:20]:
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    from_dir = df.loc[i-1, '变化方向']
    to_dir = df.loc[i, '变化方向']

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {from_dir:<6} {to_dir:<6}")

# ============================================================================
# 方法3: 用户例子的价格序列分析
# ============================================================================
print("\n" + "="*120)
print("METHOD 3: Analyzing User's Example Period")
print("="*120)

# 用户说：
# 8/19 20:00 (112873) → 开多
# 8/20 16:00 (114277) → 平多反空
# 一直到 8/22 8:00 (112320) → 平空反多

start_date = '2025-08-19'
end_date = '2025-08-23'

period = df[(df['时间'] >= start_date) & (df['时间'] < end_date)].copy()

print(f"\n8月19-22日期间的价格数据:")
print(f"{'序号':<6} {'时间':<18} {'收盘价':<10} {'变化%':<10} {'极值':<10}")
print("-" * 80)

period['变化%'] = period['收盘价'].pct_change() * 100

for idx, row in period.iterrows():
    time_str = str(row['时间'])[:16]
    close = row['收盘价']
    change = row['变化%'] if not pd.isna(row['变化%']) else 0

    marker = ''
    if idx in local_max:
        marker = '波峰'
    elif idx in local_min:
        marker = '波谷'

    print(f"{idx:<6} {time_str:<18} {close:<10.2f} {change:+.2f}%     {marker:<10}")

# 找这个期间的最高点和最低点
if len(period) > 0:
    max_idx = period['收盘价'].idxmax()
    min_idx = period['收盘价'].idxmin()

    print(f"\n数学上的极值:")
    print(f"  最高点: Row {max_idx}, {df.loc[max_idx, '时间']}, 收盘 {df.loc[max_idx, '收盘价']:.2f}")
    print(f"  最低点: Row {min_idx}, {df.loc[min_idx, '时间']}, 收盘 {df.loc[min_idx, '收盘价']:.2f}")

# ============================================================================
# 基于数学极值的最优交易路径
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Optimal Trading Path Based on Math Extrema")
print("="*120)

actions = []
positions = []
current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    close = df.loc[i, '收盘价']

    # 判断是否是极值点
    is_peak = i in local_max
    is_valley = i in local_min

    # 计算盈亏
    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (close - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - close) / entry_price * 100
    else:
        pnl_pct = 0

    action = ''

    if current_position == 'NONE':
        if is_valley:
            action = '开多（波谷）'
            current_position = 'LONG'
            entry_price = close
        elif is_peak:
            action = '开空（波峰）'
            current_position = 'SHORT'
            entry_price = close
        else:
            action = '观望'

    elif current_position == 'LONG':
        if is_peak:
            action = f'平多/反空（波峰,盈利{pnl_pct:.2f}%）'
            current_position = 'SHORT'
            entry_price = close
        else:
            action = f'继续持多 ({pnl_pct:+.2f}%)'

    elif current_position == 'SHORT':
        if is_valley:
            action = f'平空/反多（波谷,盈利{pnl_pct:.2f}%）'
            current_position = 'LONG'
            entry_price = close
        else:
            action = f'继续持空 ({pnl_pct:+.2f}%)'

    actions.append(action)
    positions.append(current_position)

df['数学最优动作'] = actions
df['数学持仓'] = positions

# 验证用户例子
print("\n用户例子验证:")
test_cases = [
    ('2025-08-19 20:00', '开多'),
    ('2025-08-20 16:00', '平多反空'),
    ('2025-08-22 08:00', '平空反多')
]

for time_str, expected in test_cases:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        actual = df.loc[idx, '数学最优动作'][:30]
        print(f"{df.loc[idx, '时间']} 收盘{df.loc[idx, '收盘价']:.2f}")
        print(f"  期望: {expected}")
        print(f"  算法: {actual}")
        print()

# 保存
df[['时间', '收盘价', '数学持仓', '数学最优动作']].to_csv('数学波峰波谷.csv', index=False, encoding='utf-8-sig')
print("结果已保存至: 数学波峰波谷.csv")
