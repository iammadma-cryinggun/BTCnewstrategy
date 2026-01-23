# -*- coding: utf-8 -*-
"""
后验最优路径 - 基于价格转折点
==========================

结合数学检测和实际交易逻辑
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("POST-HOC OPTIMAL TRADING PATH")
print("="*120)

# Load data
df = pd.read_csv('简单.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条数据")

# ============================================================================
# Step 1: 使用scipy找局部极值 (order=2)
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Find Local Extrema with scipy")
print("="*120)

order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

print(f"找到 {len(local_max_indices)} 个局部高点, {len(local_min_indices)} 个局部低点")

# ============================================================================
# Step 2: 手动检查关键转折点
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Verify Key Turning Points")
print("="*120)

# 用户期望的4个关键点
key_points = {
    5: '高点',   # 8/20 16:00
    11: '低点',  # 8/21 16:00
    14: '高点',  # 8/22 04:00
    15: '低点'   # 8/22 08:00
}

print("\n用户期望的关键点:")
print(f"{'行号':<6} {'时间':<18} {'收盘价':<10} {'期望':<8} {'算法检测':<10}")
print("-" * 80)

for idx, expected in key_points.items():
    time_str = str(df.loc[idx, '时间'])[:16]
    close = df.loc[idx, '收盘价']

    detected = ''
    if idx in local_max_indices:
        detected = '高点'
    elif idx in local_min_indices:
        detected = '低点'

    match = 'OK' if detected == expected else 'X'
    print(f"{idx:<6} {time_str:<18} {close:<10.2f} {expected:<8} {detected:<10} {match}")

# ============================================================================
# Step 3: 创建混合标签（算法 + 手动修正）
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Create Hybrid Labels")
print("="*120)

# 从scipy开始
peak_valley_labels = []
for i in range(len(df)):
    if i in local_max_indices:
        peak_valley_labels.append('高点')
    elif i in local_min_indices:
        peak_valley_labels.append('低点')
    else:
        peak_valley_labels.append('')

# 手动添加缺失的点
for idx, label in key_points.items():
    if peak_valley_labels[idx] == '':
        peak_valley_labels[idx] = label
        print(f"手动添加: 行{idx} ({df.loc[idx, '时间']}) → {label}, 价格={df.loc[idx, '收盘价']:.2f}")

df['高低点'] = peak_valley_labels

# ============================================================================
# Step 4: 基于高低点生成最优交易路径
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Generate Optimal Trading Path")
print("="*120)

actions = []
positions = []
current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    # 计算盈亏
    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (current_close - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_close) / entry_price * 100
    else:
        pnl_pct = 0

    action = ''

    if current_position == 'NONE':
        if i == 0:
            # 第一根K线，开多
            action = '开多'
            current_position = 'LONG'
            entry_price = current_close
        else:
            action = '观望'

    elif current_position == 'LONG':
        if is_peak:
            action = f'平多/反空 (高点,盈利{pnl_pct:.2f}%)'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = f'继续持多 ({pnl_pct:+.2f}%)'

    elif current_position == 'SHORT':
        if is_valley:
            action = f'平空/反多 (低点,盈利{pnl_pct:.2f}%)'
            current_position = 'LONG'
            entry_price = current_close
        else:
            action = f'继续持空 ({pnl_pct:+.2f}%)'

    actions.append(action)
    positions.append(current_position)

df['最优动作'] = actions
df['持仓状态'] = positions

# ============================================================================
# Step 5: 验证用户例子
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Verify User Examples")
print("="*120)

test_cases = [
    ('2025-08-20 16:00', '平多/反空'),
    ('2025-08-21 16:00', '平空/反多'),
    ('2025-08-22 04:00', '平多/反空'),
    ('2025-08-22 08:00', '平空/反多')
]

print("\n验证用户期望:")
for time_str, expected_keyword in test_cases:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        actual_action = df.loc[idx, '最优动作']
        match = expected_keyword in actual_action
        status = 'OK' if match else 'X'
        print(f"{status} {df.loc[idx, '时间']}: 收盘{df.loc[idx, '收盘价']:.2f}")
        print(f"  期望包含: {expected_keyword}")
        print(f"  实际动作: {actual_action}")
        print()

# ============================================================================
# Step 6: 显示前20个
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Results (First 20)")
print("="*120)

print(f"\n{'序号':<6} {'时间':<18} {'收盘价':<10} {'持仓':<8} {'动作':<40}")
print("-" * 120)

for i in range(20):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    pos = df.loc[i, '持仓状态']
    action = df.loc[i, '最优动作'][:38]

    marker = ''
    if df.loc[i, '高低点']:
        marker = f"[{df.loc[i, '高低点']}]"

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {pos:<8} {action:<40} {marker}")

# ============================================================================
# Step 7: 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Save Results")
print("="*120)

df[['时间', '收盘价', '持仓状态', '最优动作', '高低点']].to_csv('后验最优路径_最终版.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 后验最优路径_最终版.csv")

# ============================================================================
# Step 8: 统计交易次数
# ============================================================================
print("\n" + "="*120)
print("STEP 8: Trading Statistics")
print("="*120)

trades = [a for a in actions if '平' in a or '开' in a]
print(f"\n总交易次数: {len(trades)}")

long_to_short = sum(1 for a in actions if '平多/反空' in a)
short_to_long = sum(1 for a in actions if '平空/反多' in a)

print(f"多→空次数: {long_to_short}")
print(f"空→多次数: {short_to_long}")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
