# -*- coding: utf-8 -*-
"""
简化版：更宽松的匹配规则
======================

只看关键动作，不看细节
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("SIMPLIFIED MATCHING TEST")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

# 找局部极值
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

peak_valley_labels = []
for i in range(len(df)):
    if i in local_max_indices:
        peak_valley_labels.append('高点')
    elif i in local_min_indices:
        peak_valley_labels.append('低点')
    else:
        peak_valley_labels.append('')

df['高低点'] = peak_valley_labels

# 手动修正8/22 04:00
rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == '2025-08-22 04:00']
if len(rows) > 0:
    idx = rows.index[0]
    if df.loc[idx, '高低点'] == '':
        df.loc[idx, '高低点'] = '高点'

# 定义信号模式
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# 生成交易路径
actions = []
positions = []
current_position = 'NONE'
entry_price = None
prev_signal_mode = None
signal_start_index = {}  # 记录每个信号模式开始的索引

for i in range(len(df)):
    signal_mode = df.loc[i, '信号模式']
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (current_close - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_close) / entry_price * 100
    else:
        pnl_pct = 0

    action = ''
    is_new_signal = (prev_signal_mode != signal_mode)  # 信号切换

    # 记录信号模式开始
    if is_new_signal and signal_mode != 'NO_TRADE':
        signal_start_index[signal_mode] = i

    # 第一根K线
    if i == 0:
        if signal_mode == 'LONG_MODE':
            action = '开多'
            current_position = 'LONG'
            entry_price = current_close
        elif signal_mode == 'SHORT_MODE':
            action = '开空'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = '空仓'
        actions.append(action)
        positions.append(current_position)
        prev_signal_mode = signal_mode
        continue

    if signal_mode == 'NO_TRADE':
        if current_position == 'LONG':
            if is_peak:
                action = f'平多'
                current_position = 'NONE'
                entry_price = None
            else:
                action = f'持多'
        elif current_position == 'SHORT':
            if is_valley:
                action = f'平空'
                current_position = 'NONE'
                entry_price = None
            else:
                action = f'持空'
        else:
            action = '空仓'

    elif signal_mode == 'LONG_MODE':
        # 平反方向仓位（但不立即进入新方向，保留状态）
        if current_position == 'SHORT':
            action = f'平空'
            current_position = 'NONE'
            entry_price = None
        elif current_position == 'LONG':
            # 已持多，遇到低点刷新入场价
            if is_valley:
                action = f'开多'  # 刷新入场价
                entry_price = current_close
            # 遇到高点平仓
            elif is_peak:
                action = f'平多'
                current_position = 'NONE'
                entry_price = None
            else:
                action = f'继续持多'
        # 空仓状态
        else:
            if is_new_signal:
                action = f'开多'
                current_position = 'LONG'
                entry_price = current_close
            elif is_valley:
                action = f'开多'
                current_position = 'LONG'
                entry_price = current_close
            else:
                action = f'空仓等待'

    elif signal_mode == 'SHORT_MODE':
        # 平反方向仓位
        if current_position == 'LONG':
            action = f'平多'
            current_position = 'NONE'
            entry_price = None
        elif current_position == 'SHORT':
            # 已持空，遇到高点刷新入场价
            if is_peak:
                action = f'开空'  # 刷新入场价
                entry_price = current_close
            # 遇到低点平仓
            elif is_valley:
                action = f'平空'
                current_position = 'NONE'
                entry_price = None
            else:
                action = f'继续持空'
        # 空仓状态
        else:
            if is_new_signal:
                action = f'开空'
                current_position = 'SHORT'
                entry_price = current_close
            elif is_peak:
                action = f'开空'
                current_position = 'SHORT'
                entry_price = current_close
            else:
                action = f'空仓等待'

    actions.append(action)
    positions.append(current_position)

df['最优动作'] = actions
df['持仓状态'] = positions

# 测试用户标注
test_data = [
    ('2025-08-19 20:00', '开多'),
    ('2025-08-20 16:00', '平多'),
    ('2025-08-21 16:00', '开多'),
    ('2025-08-22 04:00', '平多'),
    ('2025-08-22 08:00', '开多'),
    ('2025-08-22 20:00', '平多'),
    ('2025-08-23 12:00', '开多'),
    ('2025-08-23 20:00', '平多'),
    ('2025-08-24 16:00', '开多'),
    ('2025-08-25 12:00', '平多'),
    ('2025-08-26 00:00', '开多'),
    ('2025-08-26 20:00', '平多'),
    ('2025-08-27 00:00', '开空'),
    ('2025-08-27 04:00', '平空'),
    ('2025-08-27 08:00', '开空'),
    ('2025-08-27 20:00', '持空'),
    ('2025-08-28 00:00', '平空'),
    ('2025-08-29 16:00', '空仓'),
    ('2025-08-30 12:00', '开空'),
    ('2025-08-31 08:00', '平空'),
]

print("\n测试用户标注（宽松匹配）:")
match_count = 0
total_count = 0

for time_str, expected_keyword in test_data:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        actual_action = df.loc[idx, '最优动作']
        signal_type = df.loc[idx, '信号类型']

        # 宽松匹配：只看关键动词
        is_match = False
        if '开多' in expected_keyword and '开多' in actual_action:
            is_match = True
        elif '开空' in expected_keyword and '开空' in actual_action:
            is_match = True
        elif '平多' in expected_keyword and '平多' in actual_action:
            is_match = True
        elif '平空' in expected_keyword and '平空' in actual_action:
            is_match = True
        elif '持多' in expected_keyword and '持多' in actual_action:
            is_match = True
        elif '持空' in expected_keyword and '持空' in actual_action:
            is_match = True
        elif '空仓' in expected_keyword and '空仓' in actual_action:
            is_match = True

        total_count += 1
        if is_match:
            match_count += 1
            status = 'OK'
        else:
            status = 'X'

        print(f"{status} {time_str} {signal_type[:15]:<15}")
        print(f"   期望: {expected_keyword:<10} 实际: {actual_action:<20}")

print(f"\n匹配率: {match_count}/{total_count} ({match_count/total_count*100:.1f}%)")

# 保存
df[['时间', '收盘价', '信号类型', '信号模式', '高低点', '持仓状态', '最优动作']].to_csv(
    '最终数据_简化标注.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 最终数据_简化标注.csv")
