# -*- coding: utf-8 -*-
"""
生成带信号标记的完整数据表（修复版）
修复了平仓逻辑导致的状态覆盖问题
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("生成带信号标记的完整数据表（修复版）")
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
# 策略执行模拟（修复版）
# ==============================================================================

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
position_size_pct = 0.0
entry_signal_idx = None

prev_signal_mode = None

for i in range(len(df)):
    signal_mode = df.loc[i, '信号模式']
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    # 信号模式切换检测
    is_new_signal = (prev_signal_mode != signal_mode) and (signal_mode != 'NO_TRADE')

    action = ''

    # ==================== 平仓逻辑 ====================

    if current_position == 'LONG':
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 黑天鹅交易的张力释放离场
        if is_black_swan_trade:
            current_tension = df.loc[i, '张力']
            if current_tension < 0.30:
                action = f'平仓(张力释放{current_tension:.3f})'
                current_position = 'NONE'
                entry_price = None
                is_black_swan_trade = False
                stop_loss_pct = None

        # 止损平仓
        if current_position == 'LONG' and unrealized_pnl <= -stop_loss_pct:
            action = f'平仓(止损{unrealized_pnl*100:.2f}%)'
            current_position = 'NONE'
            entry_price = None
            is_black_swan_trade = False
            stop_loss_pct = None

        # 正常平仓：高点或信号切换
        elif current_position == 'LONG' and (is_peak or signal_mode in ['SHORT_MODE', 'NO_TRADE']):
            if is_peak:
                action = '平仓(高点)'
            elif signal_mode == 'SHORT_MODE':
                action = '平仓(转SHORT)'
            else:
                action = '平仓(OSCILLATION)'

            current_position = 'NONE'
            entry_price = None
            is_black_swan_trade = False
            stop_loss_pct = None

    # ==================== 开仓逻辑 ====================

    if current_position == 'NONE':
        # 只在信号模式切换到LONG_MODE时开多仓
        if signal_mode == 'LONG_MODE' and prev_signal_mode != 'LONG_MODE':
            # 滞后2根K线入场
            if i + order < len(df):
                entry_idx = i + order
                entry_price = df.loc[entry_idx, '收盘价']
                entry_signal_idx = i

                # 检查原始信号是否是黑天鹅
                if df.loc[entry_signal_idx, '是黑天鹅']:
                    position_size_pct = BS_POSITION_PCT * LEVERAGE * 100  # 250%
                    stop_loss_pct = BS_STOP_LOSS_PCT
                    is_black_swan_trade = True
                    current_position = 'LONG'
                    df.loc[entry_idx, '信号动作'] = f'开多(黑天鹅 {position_size_pct:.0f}%仓位)'
                else:
                    position_size_pct = NORMAL_POSITION_PCT * LEVERAGE * 100  # 150%
                    stop_loss_pct = NORMAL_STOP_LOSS_PCT
                    is_black_swan_trade = False
                    current_position = 'LONG'
                    df.loc[entry_idx, '信号动作'] = f'开多(正常 {position_size_pct:.0f}%仓位)'

    # ==================== 更新状态（逐行更新）====================

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

    # 如果有动作，记录动作（覆盖之前的空值）
    if action:
        df.loc[i, '信号动作'] = action

    prev_signal_mode = signal_mode

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
output_df.to_csv('带信号标记_完整数据_修复版.csv', index=False, encoding='utf-8-sig')

print("[OK] 已保存: 带信号标记_完整数据_修复版.csv")
print(f"  总行数: {len(output_df)}")

# 保存精简版（只有动作和持仓）
action_df = output_df[
    (output_df['信号动作'] != '') |
    (output_df['策略状态'] == '持仓')
].copy()

action_df.to_csv('带信号标记_关键行_修复版.csv', index=False, encoding='utf-8-sig')

print("[OK] 已保存: 带信号标记_关键行_修复版.csv")
print(f"  关键行数: {len(action_df)}")

# 统计信息
print("\n" + "="*120)
print("统计信息")
print("="*120)

print(f"\n黑天鹅信号数: {len(output_df[output_df['是黑天鹅'] == '★黑天鹅'])}")

open_long_actions = output_df[output_df['信号动作'].str.contains('开多', na=False)]
print(f"开多次数: {len(open_long_actions)}")

bs_trades = output_df[output_df['信号动作'].str.contains('黑天鹅', na=False)]
print(f"黑天鹅交易: {len(bs_trades)}")

normal_trades = output_df[output_df['信号动作'].str.contains('正常', na=False)]
print(f"正常交易: {len(normal_trades)}")

print("\n" + "="*120)
print("关键信号列表")
print("="*120)

# 显示所有开仓和平仓动作
all_actions = output_df[output_df['信号动作'] != ''].copy()

if len(all_actions) > 0:
    print(f"\n{'时间':<20} {'收盘价':<12} {'信号类型':<20} {'黑天鹅':<10} {'策略状态':<10} {'入场价':<10} {'动作':<30}")
    print("-"*140)

    for idx, row in all_actions.iterrows():
        entry_price_display = f"{row['入场价']:.2f}" if row['入场价'] > 0 else "N/A"
        print(f"{str(row['时间'])[:18]:<20} "
              f"${row['收盘价']:>10.2f} "
              f"{row['信号类型']:<20} "
              f"{row['是黑天鹅']:<10} "
              f"{row['策略状态']:<10} "
              f"{entry_price_display:<10} "
              f"{row['信号动作']:<30}")

print("\n" + "="*120)
print("[OK] 完成！已修复状态覆盖问题")
print("="*120)
