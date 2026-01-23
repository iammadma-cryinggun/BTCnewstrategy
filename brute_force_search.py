# -*- coding: utf-8 -*-
"""
暴力搜索：寻找利润最大化参数
用完整OHLC数据回测
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("暴力搜索：利润最大化参数")
print("="*120)

# 加载完整OHLC数据
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

# ==============================================================================
# 参数池
# ==============================================================================

accel_options = [-0.12, -0.15, -0.20, -0.25]          # 入场加速度门槛
tension_entry_options = [0.50, 0.60, 0.70]           # 入场张力门槛
tension_exit_options = [0.40, 0.20, 0.05]            # 离场张力阈值
stop_loss_options = [0.02]                           # 固定2%止损
take_profit_options = [0.02, 0.03, 0.05, 0.08, 0.10] # 固定止盈选项

INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.50
COMMISSION_PCT = 0.0005
HOLD_MAX_PERIODS = 20

print(f"""
参数搜索空间：
- 加速度: {accel_options}
- 入场张力: {tension_entry_options}
- 离场张力: {tension_exit_options}
- 止盈: {take_profit_options}
- 止损: {stop_loss_options}

总组合数: {len(accel_options) * len(tension_entry_options) * len(tension_exit_options) * len(take_profit_options)}
""")

results = []

total_combinations = len(accel_options) * len(tension_entry_options) * len(tension_exit_options) * len(take_profit_options)
current = 0

# ==============================================================================
# 暴力搜索
# ==============================================================================

for accel in accel_options:
    for t_in in tension_entry_options:
        for t_out in tension_exit_options:
            for tp in take_profit_options:

                current += 1
                if current % 10 == 0:
                    print(f"进度: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")

                # 回测变量
                capital = INITIAL_CAPITAL
                trades = []
                last_exit_idx = -100

                # 信号检测
                signal_conditions = (
                    (df['信号模式'] == 'LONG_MODE') &
                    (df['加速度'] <= accel) &
                    (df['张力'] >= t_in) &
                    (df['下影线'] < 0.35) &
                    (df['量能比率'] > 1.0)
                )

                potential_signals = df[signal_conditions].copy()

                # 遍历信号
                for idx, row in potential_signals.iterrows():
                    # 冷却期：距离上次离场 > 6根K线
                    if idx - last_exit_idx <= 6:
                        continue

                    signal_high = row['最高价']
                    entry_price = signal_high * 1.0001

                    # 检查成交（后续10根K线）
                    filled = False
                    fill_bar = None
                    fill_price = None

                    for i in range(idx + 1, min(idx + 11, len(df))):
                        if df.loc[i, '最高价'] >= entry_price:
                            filled = True
                            fill_bar = i
                            fill_price = df.loc[i, '收盘价']
                            break

                    if not filled:
                        continue

                    entry_time = df.loc[fill_bar, '时间']

                    # 离场逻辑
                    exit_triggered = False
                    exit_price = None
                    exit_bar = None
                    exit_reason = None

                    for i in range(fill_bar + 1, min(fill_bar + HOLD_MAX_PERIODS + 1, len(df))):
                        current_price = df.loc[i, '收盘价']
                        current_low = df.loc[i, '最低价']
                        current_tension = df.loc[i, '张力']

                        # 计算盈亏
                        unrealized_pnl = (current_price - fill_price) / fill_price

                        # 止盈
                        if unrealized_pnl >= tp:
                            exit_price = current_price
                            exit_bar = i
                            exit_reason = f'止盈({tp*100:.0f}%)'
                            exit_triggered = True
                            break

                        # 止损
                        if current_low <= fill_price * (1 - stop_loss_options[0]):
                            exit_price = current_price
                            exit_bar = i
                            exit_reason = f'止损({stop_loss_options[0]*100:.0f}%)'
                            exit_triggered = True
                            break

                        # 张力释放离场
                        if current_tension < t_out:
                            exit_price = current_price
                            exit_bar = i
                            exit_reason = f'张力释放({current_tension:.3f})'
                            exit_triggered = True
                            break

                    # 到期离场
                    if not exit_triggered:
                        exit_bar = min(fill_bar + HOLD_MAX_PERIODS, len(df) - 1)
                        exit_price = df.loc[exit_bar, '收盘价']
                        exit_reason = '到期'

                    exit_time = df.loc[exit_bar, '时间']

                    pnl_pct = (exit_price - fill_price) / fill_price * 100
                    position_size = capital * POSITION_SIZE_PCT
                    pnl_usd = position_size * (pnl_pct / 100)
                    pnl_usd -= position_size * COMMISSION_PCT
                    capital += pnl_usd

                    trades.append({
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason,
                        'hold_bars': exit_bar - fill_bar
                    })

                    last_exit_idx = exit_bar

                # 统计结果
                if len(trades) == 0:
                    continue

                total_pnl = sum(t['pnl_usd'] for t in trades)
                total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

                winning_trades = [t for t in trades if t['pnl_pct'] > 0]
                losing_trades = [t for t in trades if t['pnl_pct'] <= 0]

                win_rate = len(winning_trades) / len(trades) * 100

                days = (df['时间'].max() - df['时间'].min()).days
                annualized_return = (1 + total_return/100) ** (365/days) - 1

                results.append({
                    'Accel': accel,
                    'Tension_In': t_in,
                    'Tension_Out': t_out,
                    'Take_Profit': tp,
                    'Trades': len(trades),
                    'Win_Rate': win_rate,
                    'Total_Return': total_return,
                    'Annual_Return': annualized_return * 100
                })

# ==============================================================================
# 结果展示
# ==============================================================================

print("\n" + "="*120)
print("暴力搜索结果（按年化收益率排序）")
print("="*120)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Annual_Return', ascending=False)

print(f"\n{'排名':<4} {'Acc':<6} {'T_In':<6} {'T_Out':<6} {'TP':<6} {'交易':<6} {'胜率%':<8} {'总收益%':<10} {'年化%':<10}")
print("-"*80)

for i in range(min(15, len(results_df))):
    res = results_df.iloc[i]
    print(f"#{i+1:<3} {res['Accel']:<6} {res['Tension_In']:<6} {res['Tension_Out']:<6} "
          f"{res['Take_Profit']:<6} {int(res['Trades']):<6} {res['Win_Rate']:<7.1f} "
          f"{res['Total_Return']:>8.2f} {res['Annual_Return']:>9.2f}")

print("\n" + "="*120)
print("TOP 3 策略详情")
print("="*120)

for i in range(min(3, len(results_df))):
    res = results_df.iloc[i]
    print(f"\n第{i+1}名：年化收益 {res['Annual_Return']:.2f}%")
    print(f"  参数：Acc={res['Accel']}, T_In={res['Tension_In']}, T_Out={res['Tension_Out']}, TP={res['Take_Profit']}")
    print(f"  表现：{int(res['Trades'])}笔交易, 胜率{res['Win_Rate']:.1f}%, 总收益{res['Total_Return']:.2f}%")

# 找到年化收益 > 20% 的策略
high_return = results_df[results_df['Annual_Return'] > 20]

if len(high_return) > 0:
    print(f"\n" + "="*120)
    print(f"找到 {len(high_return)} 个年化收益 > 20% 的策略！")
    print("="*120)
else:
    print(f"\n" + "="*120)
    print("警告：没有找到年化收益 > 20% 的策略")
    print("="*120)

print("\n" + "="*120)
