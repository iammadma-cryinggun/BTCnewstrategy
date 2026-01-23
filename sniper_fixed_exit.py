# -*- coding: utf-8 -*-
"""
狙击手战术手册 - 修正版
固定离场策略（不使用张力释放）
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("狙击手战术手册 - 修正版（固定离场）")
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

# ==============================================================================
# 信号检测
# ==============================================================================

signal_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.20) &
    (df['张力'] >= 0.70) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

potential_signals = df[signal_conditions].copy()

print(f"\n找到潜在信号: {len(potential_signals)} 个")

# ==============================================================================
# 回测配置
# ==============================================================================

INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.50
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.03  # 3%止盈
COMMISSION_PCT = 0.0005

print(f"""
回测参数:
- 止损: {STOP_LOSS_PCT*100}%
- 止盈: {TAKE_PROFIT_PCT*100}%
- 仓位: {POSITION_SIZE_PCT*100}%
""")

# ==============================================================================
# 回测主函数
# ==============================================================================

confirmed_trades = []
last_exit_idx = -100

for idx, row in potential_signals.iterrows():
    # 冷却期检查
    bars_since_exit = idx - last_exit_idx
    if bars_since_exit <= 6:  # 放宽到6根K线（24小时）
        continue

    signal_high = row['最高价']
    entry_price = signal_high * 1.0001

    # 检查成交
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

    # 固定止盈止损离场
    exit_triggered = False
    exit_price = None
    exit_bar = None
    exit_reason = None

    for i in range(fill_bar + 1, min(fill_bar + 42, len(df))):
        current_price = df.loc[i, '收盘价']
        current_low = df.loc[i, '最低价']

        # 计算盈亏
        unrealized_pnl = (current_price - fill_price) / fill_price

        # 止盈
        if unrealized_pnl >= TAKE_PROFIT_PCT:
            exit_price = current_price
            exit_bar = i
            exit_reason = f'止盈({TAKE_PROFIT_PCT*100}%)'
            exit_triggered = True
            break

        # 止损
        if current_low <= fill_price * (1 - STOP_LOSS_PCT):
            exit_price = current_price
            exit_bar = i
            exit_reason = f'止损({STOP_LOSS_PCT*100}%)'
            exit_triggered = True
            break

    # 到期离场
    if not exit_triggered:
        exit_bar = min(fill_bar + 41, len(df) - 1)
        exit_price = df.loc[exit_bar, '收盘价']
        exit_reason = '到期(42周期)'

    exit_time = df.loc[exit_bar, '时间']
    hold_bars = exit_bar - fill_bar
    hold_hours = hold_bars * 4

    pnl_pct = (exit_price - fill_price) / fill_price * 100
    position_size = INITIAL_CAPITAL * POSITION_SIZE_PCT
    pnl_usd = position_size * (pnl_pct / 100)
    pnl_usd -= position_size * COMMISSION_PCT

    confirmed_trades.append({
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': fill_price,
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'pnl_usd': pnl_usd,
        'exit_reason': exit_reason,
        'hold_bars': hold_bars,
        'hold_hours': hold_hours
    })

    last_exit_idx = exit_bar

# ==============================================================================
# 统计结果
# ==============================================================================

print("\n" + "="*120)
print("统计结果")
print("="*120)

if len(confirmed_trades) == 0:
    print("\n没有成交的交易！")
else:
    trades_df = pd.DataFrame(confirmed_trades)

    capital = INITIAL_CAPITAL
    for trade in confirmed_trades:
        capital += trade['pnl_usd']

    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    total_pnl = trades_df['pnl_usd'].sum()
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    days = (df['时间'].max() - df['时间'].min()).days
    annualized_return = (1 + total_return/100) ** (365/days) - 1

    print(f"\n【总体表现】")
    print(f"初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"最终资金: ${capital:,.2f}")
    print(f"总盈亏: ${total_pnl:+,.2f}")
    print(f"总收益率: {total_return:+.2f}%")
    print(f"年化收益率: {annualized_return*100:+.2f}%")
    print(f"测试周期: {days}天 ({days/30:.1f}个月)")
    print(f"总交易次数: {total_trades}")

    print(f"\n【胜率统计】")
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    print(f"胜率: {win_rate:.1f}%")
    print(f"盈利交易: {len(winning_trades)} 笔")
    print(f"亏损交易: {len(losing_trades)} 笔")

    if len(winning_trades) > 0:
        avg_win = winning_trades['pnl_pct'].mean()
        max_win = winning_trades['pnl_pct'].max()
        avg_win_hours = winning_trades['hold_hours'].mean()
        print(f"平均盈利: {avg_win:+.2f}%")
        print(f"最大盈利: {max_win:+.2f}%")
        print(f"平均持仓: {avg_win_hours:.1f}小时")

    if len(losing_trades) > 0:
        avg_loss = losing_trades['pnl_pct'].mean()
        max_loss = losing_trades['pnl_pct'].min()
        avg_loss_hours = losing_trades['hold_hours'].mean()
        print(f"平均亏损: {avg_loss:+.2f}%")
        print(f"最大亏损: {max_loss:+.2f}%")
        print(f"平均持仓: {avg_loss_hours:.1f}小时")

    if len(winning_trades) > 0 and len(losing_trades) > 0:
        profit_factor = abs(winning_trades['pnl_usd'].sum() / losing_trades['pnl_usd'].sum())
        print(f"盈亏比: {profit_factor:.2f}")

    # 显示所有交易
    print(f"\n【所有交易记录】")
    print(f"{'#':<4} {'入场时间':<18} {'出场时间':<18} {'入场价':<10} {'出场价':<10} "
          f"{'盈亏%':<10} {'盈亏$':<12} {'原因':<15} {'持仓h':<8}")
    print("-"*120)

    for idx, trade in trades_df.iterrows():
        print(f"{idx+1:<4} {str(trade['entry_time'])[:16]:<18} {str(trade['exit_time'])[:16]:<18} "
              f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} {trade['pnl_pct']:+<10.2f} "
              f"${trade['pnl_usd']:+<11.2f} {trade['exit_reason']:<15} {trade['hold_hours']:<8.1f}")

print("\n" + "="*120)
print("回测完成")
print("="*120)
