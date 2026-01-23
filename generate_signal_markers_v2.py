# -*- coding: utf-8 -*-
"""
生成带信号标记的完整数据表（完全重写版）
使用两阶段处理：第一阶段检测信号，第二阶段执行交易
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("生成带信号标记的完整数据表（完全重写版）")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

# 计算下影线
df['下影线'] = df.apply(lambda row: (row['收盘价'] - row['最低价']) / row['收盘价']
                        if row['收盘价'] > row['最低价'] else 0, axis=1)

# 识别极值点
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

df['高低点'] = ''
for i in local_max_indices:
    df.loc[i, '高低点'] = '高点'
for i in local_min_indices:
    df.loc[i, '高低点'] = '低点'

# 定义信号模式
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# 黑天鹅信号识别
df['是黑天鹅'] = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.15) &
    (df['张力'] >= 0.60) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

# ==============================================================================
# 阶段1：检测所有潜在的开仓信号
# ==============================================================================

print("\n[阶段1] 检测开仓信号...")

entry_signals = []  # 记录所有入场信号 (信号索引, 入场索引, 类型)

for i in range(len(df) - order):
    signal_mode = df.loc[i, '信号模式']

    # 只在空仓状态下检测开仓信号
    # 信号切换到LONG_MODE时触发
    if i == 0:
        prev_mode = 'NO_TRADE'
    else:
        prev_mode = df.loc[i-1, '信号模式']

    # 检测信号模式切换
    if signal_mode == 'LONG_MODE' and prev_mode != 'LONG_MODE':
        # 滞后order根K线入场
        entry_idx = i + order
        is_bs = df.loc[i, '是黑天鹅']

        entry_signals.append({
            'signal_idx': i,
            'entry_idx': entry_idx,
            'entry_price': df.loc[entry_idx, '收盘价'],
            'is_black_swan': is_bs,
            'signal_time': df.loc[i, '时间'],
            'entry_time': df.loc[entry_idx, '时间']
        })

print(f"  检测到 {len(entry_signals)} 个潜在入场信号")

# ==============================================================================
# 阶段2：执行交易模拟
# ==============================================================================

print("\n[阶段2] 执行交易模拟...")

LEVERAGE = 5.0
NORMAL_POSITION_PCT = 0.30
BS_POSITION_PCT = 0.50
NORMAL_STOP_LOSS_PCT = 0.015
BS_STOP_LOSS_PCT = 0.02

# 初始化状态列
df['策略状态'] = '空仓'
df['持仓类型'] = ''
df['入场价'] = 0.0
df['当前盈亏%'] = 0.0
df['信号动作'] = ''

current_position = 'NONE'
entry_price = None
entry_idx = None
is_black_swan_trade = False
stop_loss_pct = None
signal_queue = entry_signals.copy()  # 信号队列
current_signal = None

for i in range(len(df)):
    signal_mode = df.loc[i, '信号模式']
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    action = ''

    # ==================== 检查是否到达入场点 ====================

    if current_position == 'NONE' and signal_queue:
        # 检查队列中的第一个信号
        if signal_queue[0]['entry_idx'] == i:
            sig = signal_queue.pop(0)  # 从队列中取出
            entry_price = sig['entry_price']
            entry_idx = i
            is_black_swan_trade = sig['is_black_swan']

            if is_black_swan_trade:
                position_size_pct = BS_POSITION_PCT * LEVERAGE * 100
                stop_loss_pct = BS_STOP_LOSS_PCT
                action = f'开多(黑天鹅 {position_size_pct:.0f}%仓位)'
                current_position = 'LONG'
            else:
                position_size_pct = NORMAL_POSITION_PCT * LEVERAGE * 100
                stop_loss_pct = NORMAL_STOP_LOSS_PCT
                action = f'开多(正常 {position_size_pct:.0f}%仓位)'
                current_position = 'LONG'

            current_signal = sig

    # ==================== 平仓逻辑 ====================

    if current_position == 'LONG':
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 止损平仓
        if unrealized_pnl <= -stop_loss_pct:
            action = f'平仓(止损{unrealized_pnl*100:.2f}%)'
            current_position = 'NONE'
            entry_price = None
            is_black_swan_trade = False
            stop_loss_pct = None
            current_signal = None

        # 高点平仓
        elif is_peak:
            action = '平仓(高点)'
            current_position = 'NONE'
            entry_price = None
            is_black_swan_trade = False
            stop_loss_pct = None
            current_signal = None

        # 信号切换平仓
        elif signal_mode in ['SHORT_MODE', 'NO_TRADE']:
            if signal_mode == 'SHORT_MODE':
                action = '平仓(转SHORT)'
            else:
                action = '平仓(OSCILLATION)'
            current_position = 'NONE'
            entry_price = None
            is_black_swan_trade = False
            stop_loss_pct = None
            current_signal = None

    # ==================== 更新状态 ====================

    if current_position == 'LONG':
        df.loc[i, '策略状态'] = '持仓'
        df.loc[i, '持仓类型'] = 'LONG黑天鹅' if is_black_swan_trade else 'LONG正常'
        df.loc[i, '入场价'] = entry_price
        df.loc[i, '当前盈亏%'] = (current_close - entry_price) / entry_price * 100
    else:
        df.loc[i, '策略状态'] = '空仓'
        df.loc[i, '持仓类型'] = ''
        df.loc[i, '入场价'] = 0.0
        df.loc[i, '当前盈亏%'] = 0.0

    if action:
        df.loc[i, '信号动作'] = action

# ==============================================================================
# 生成输出表
# ==============================================================================

# 选择关键列
output_columns = [
    '时间', '收盘价', '最高价', '最低价',
    '信号类型', '信号模式',
    '加速度', '张力', '下影线', '量能比率',
    '高低点', '是黑天鹅',
    '策略状态', '持仓类型', '入场价', '当前盈亏%', '信号动作'
]

output_df = df[output_columns].copy()

# 格式化显示
output_df['是黑天鹅'] = output_df['是黑天鹅'].map({True: '★黑天鹅', False: ''})
output_df['当前盈亏%'] = output_df['当前盈亏%'].round(2)

print("\n正在保存完整数据表...")

# 保存完整版本
output_df.to_csv('带信号标记_完整数据_V2.csv', index=False, encoding='utf-8-sig')
print("[OK] 已保存: 带信号标记_完整数据_V2.csv")
print(f"  总行数: {len(output_df)}")

# 保存精简版
action_df = output_df[
    (output_df['信号动作'] != '') |
    (output_df['策略状态'] == '持仓')
].copy()
action_df.to_csv('带信号标记_关键行_V2.csv', index=False, encoding='utf-8-sig')
print("[OK] 已保存: 带信号标记_关键行_V2.csv")
print(f"  关键行数: {len(action_df)}")

# 统计
print("\n" + "="*120)
print("统计信息")
print("="*120)

print(f"\n黑天鹅信号数: {len(output_df[output_df['是黑天鹅'] == '★黑天鹅'])}")

open_actions = output_df[output_df['信号动作'].str.contains('开多', na=False)]
print(f"实际开仓: {len(open_actions)}")

close_actions = output_df[output_df['信号动作'].str.contains('平仓', na=False)]
print(f"实际平仓: {len(close_actions)}")

# 验证数据一致性
print("\n" + "="*120)
print("数据一致性验证")
print("="*120)

invalid_opens = output_df[
    (output_df['信号动作'].str.contains('开多', na=False)) &
    ((output_df['入场价'] == 0) | (output_df['策略状态'] == '空仓'))
]

if len(invalid_opens) > 0:
    print(f"\n⚠️ 警告：发现 {len(invalid_opens)} 个无效的开仓记录！")
    for idx, row in invalid_opens.iterrows():
        print(f"  行{idx}: {row['时间']} 入场价={row['入场价']} 状态={row['策略状态']}")
else:
    print("\n✓ 所有开仓记录都有效！")

print("\n" + "="*120)
print("[OK] 完成！")
print("="*120)
