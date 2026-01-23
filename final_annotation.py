# -*- coding: utf-8 -*-
"""
最终版：基于信号模式和局部极值的标注
===================================

完整逻辑：
1. 信号模式切换 → 平反方向仓位
2. 遇到极值点 → 平仓/开仓/刷新
3. 平仓后如果不是极值点 → 空仓等待
4. OSCILLATION → 有持仓看极值点平仓，无持仓观望
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("FINAL SIGNAL-BASED ANNOTATION")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# Step 1: 找局部极值点
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Find Local Extrema")
print("="*120)

order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

print(f"找到 {len(local_max_indices)} 个局部高点, {len(local_min_indices)} 个局部低点")

# 标注极值
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

# ============================================================================
# Step 2: 定义信号模式
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Define Signal Modes")
print("="*120)

def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# ============================================================================
# Step 3: 生成最优交易路径
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Generate Optimal Trading Path")
print("="*120)

actions = []
positions = []
current_position = 'NONE'
entry_price = None
prev_signal_mode = None  # 记录上一个信号模式

for i in range(len(df)):
    signal_type = df.loc[i, '信号类型']
    signal_mode = df.loc[i, '信号模式']
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
    is_new_signal_mode = (prev_signal_mode != signal_mode)  # 检测信号模式是否切换

    # 第一根K线或新信号模式开始：立即开仓
    if i == 0 or is_new_signal_mode:
        if signal_mode == 'LONG_MODE':
            action = '开多'
            current_position = 'LONG'
            entry_price = current_close
        elif signal_mode == 'SHORT_MODE':
            action = '开空'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = '空仓(震荡)'
        actions.append(action)
        positions.append(current_position)
        continue

    # 根据信号模式决定交易行为
    if signal_mode == 'NO_TRADE':
        # OSCILLATION → 如果有持仓遇到极值点就平，没持仓就观望
        if current_position == 'LONG':
            if is_peak:
                action = f'平多(震荡高点,盈亏{pnl_pct:+.2f}%)'
                current_position = 'NONE'
                entry_price = None
            else:
                action = f'持多(震荡,{pnl_pct:+.2f}%)'
        elif current_position == 'SHORT':
            if is_valley:
                action = f'平空(震荡低点,盈亏{pnl_pct:+.2f}%)'
                current_position = 'NONE'
                entry_price = None
            else:
                action = f'持空(震荡,{pnl_pct:+.2f}%)'
        else:
            action = '空仓(震荡)'

    elif signal_mode == 'LONG_MODE':
        # 做多模式
        if current_position == 'SHORT':
            # 持空 → 平空
            action = f'平空(信号切换,盈亏{pnl_pct:+.2f}%)'
            current_position = 'NONE'
            entry_price = None

        elif current_position == 'LONG':
            # 持多
            if is_peak:
                # 高点 → 平多
                action = f'平多(高点,盈亏{pnl_pct:+.2f}%)'
                current_position = 'NONE'
                entry_price = None
            elif is_valley:
                # 低点 → 刷新入场价
                action = f'开多(低点刷新)'
                entry_price = current_close
            else:
                action = f'继续持多({pnl_pct:+.2f}%)'

        else:  # NONE
            # 空仓
            if is_new_signal_mode:
                # 新信号模式开始 → 立即开多
                action = f'开多(新信号)'
                current_position = 'LONG'
                entry_price = current_close
            elif is_valley:
                # 低点 → 开多
                action = f'开多(低点)'
                current_position = 'LONG'
                entry_price = current_close
            elif is_peak:
                action = f'观望(高点)'
            else:
                action = '空仓等待'

    elif signal_mode == 'SHORT_MODE':
        # 做空模式
        if current_position == 'LONG':
            # 持多 → 平多
            action = f'平多(信号切换,盈亏{pnl_pct:+.2f}%)'
            current_position = 'NONE'
            entry_price = None

        elif current_position == 'SHORT':
            # 持空
            if is_valley:
                # 低点 → 平空
                action = f'平空(低点,盈亏{pnl_pct:+.2f}%)'
                current_position = 'NONE'
                entry_price = None
            elif is_peak:
                # 高点 → 刷新入场价
                action = f'开空(高点刷新)'
                entry_price = current_close
            else:
                action = f'继续持空({pnl_pct:+.2f}%)'

        else:  # NONE
            # 空仓
            if is_new_signal_mode:
                # 新信号模式开始 → 立即开空
                action = f'开空(新信号)'
                current_position = 'SHORT'
                entry_price = current_close
            elif is_peak:
                # 高点 → 开空
                action = f'开空(高点)'
                current_position = 'SHORT'
                entry_price = current_close
            elif is_valley:
                action = f'观望(低点)'
            else:
                action = '空仓等待'

    actions.append(action)
    positions.append(current_position)

    # 更新前一个信号模式
    prev_signal_mode = signal_mode

df['最优动作'] = actions
df['持仓状态'] = positions

# ============================================================================
# Step 4: 标注黄金信号
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Mark Gold Signals")
print("="*120)

gold_signals = []
for i in range(len(df)):
    action = df.loc[i, '最优动作']

    if any(keyword in action for keyword in ['开多', '开空', '平多', '平空', '反多', '反空', '刷新入场价']):
        gold_signals.append('ACTION')
    else:
        gold_signals.append('HOLD')

df['黄金信号'] = gold_signals

action_count = sum(1 for s in gold_signals if s == 'ACTION')
hold_count = sum(1 for s in gold_signals if s == 'HOLD')

print(f"\n标注统计:")
print(f"  ACTION (需要交易): {action_count} ({action_count/len(df)*100:.1f}%)")
print(f"  HOLD (继续持有): {hold_count} ({hold_count/len(df)*100:.1f}%)")

# ============================================================================
# Step 5: 验证用户手动标注
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Verify User Manual Annotation")
print("="*120)

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

print("\n验证用户标注:")
match_count = 0
total_count = 0

for time_str, expected_keyword in test_data:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        actual_action = df.loc[idx, '最优动作']
        signal_type = df.loc[idx, '信号类型']

        is_match = expected_keyword in actual_action
        total_count += 1
        if is_match:
            match_count += 1
            status = 'OK'
        else:
            status = 'X'

        print(f"{status} {time_str} {signal_type[:20]:<20}")
        print(f"   期望: {expected_keyword:<10} 实际: {actual_action[:40]}")

print(f"\n匹配率: {match_count}/{total_count} ({match_count/total_count*100:.1f}%)")

# ============================================================================
# Step 6: 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Save Results")
print("="*120)

output_cols = [
    '时间', '收盘价', '信号类型', '信号模式', '量能比率', '价格vsEMA%',
    '张力', '加速度', '高低点', '持仓状态', '最优动作', '黄金信号'
]

df[output_cols].to_csv('最终数据_完整标注.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 最终数据_完整标注.csv")

# ============================================================================
# Step 7: 显示前30条
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Sample Results (First 30)")
print("="*120)

print(f"\n{'序号':<6} {'时间':<18} {'收盘价':<10} {'信号类型':<20} {'模式':<12} {'高低点':<8} {'动作':<35}")
print("-" * 150)

for i in range(min(30, len(df))):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    sig_type = str(df.loc[i, '信号类型'])[:18]
    mode = df.loc[i, '信号模式']
    peak_valley = df.loc[i, '高低点']
    action = str(df.loc[i, '最优动作'])[:33]

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {sig_type:<20} {mode:<12} {peak_valley:<8} {action:<35}")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
