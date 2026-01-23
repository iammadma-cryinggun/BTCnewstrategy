# -*- coding: utf-8 -*-
"""
混合策略：简单极值点 + 黑天鹅增强
- 正常信号：简单策略（70%胜率基础）
- 极端信号：黑天鹅增强（重仓、宽止损、特殊离场）
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("混合策略：简单极值点 + 黑天鹅增强")
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
# 黑天鹅信号识别（用于增强）
# ==============================================================================

df['是黑天鹅'] = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.15) &
    (df['张力'] >= 0.60) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

black_swan_count = df['是黑天鹅'].sum()
print(f"\n检测到黑天鹅信号: {black_swan_count} 个 (占{black_swan_count/len(df)*100:.1f}%)")

# ==============================================================================
# 混合策略回测
# ==============================================================================

print("""
策略说明：
【正常信号】
- 入场：BEARISH + 低点
- 仓位：50%
- 止损：1.5%
- 离场：高点或最大持仓10周期

【黑天鹅增强信号】
- 入场：BEARISH + 低点 + 黑天鹅指标
- 仓位：80%（重仓）
- 止损：2.0%（宽止损）
- 离场：张力释放(< 0.3) 或止盈3% 或高点
""")

INITIAL_CAPITAL = 10000
COMMISSION_PCT = 0.0005
HOLD_MAX_PERIODS_NORMAL = 10
HOLD_MAX_PERIODS_BLACKSWAN = 20

capital = INITIAL_CAPITAL
capital_curve = [INITIAL_CAPITAL]
trades = []

current_position = 'NONE'
entry_price = None
entry_idx = None
entry_time = None
position_size = 0
stop_loss_pct = None
is_black_swan_trade = False

peak_set = set(df[df['高低点'] == '高点'].index)
valley_set = set(df[df['高低点'] == '低点'].index)

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']
    is_peak = (i in peak_set)
    is_valley = (i in valley_set)
    signal_mode = df.loc[i, '信号模式']
    is_black_swan = df.loc[i, '是黑天鹅']

    # ==================== 开仓逻辑 ====================

    if current_position == 'NONE':
        # 做多信号：BEARISH + 低点
        if signal_mode == 'LONG_MODE' and is_valley:
            # 滞后2根K线入场
            if i + order < len(df):
                entry_price = df.loc[i + order, '收盘价']
                entry_idx = i + order
                entry_time = df.loc[entry_idx, '时间']

                # 检查是否是黑天鹅信号
                if df.loc[i, '是黑天鹅']:
                    # 黑天鹅增强模式
                    position_size = capital * 0.80
                    stop_loss_pct = 0.02
                    is_black_swan_trade = True
                    current_position = 'LONG_BLACKSWAN'
                else:
                    # 正常模式
                    position_size = capital * 0.50
                    stop_loss_pct = 0.015
                    is_black_swan_trade = False
                    current_position = 'LONG'

    # ==================== 平仓逻辑 ====================

    if current_position.startswith('LONG'):
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 检查止损
        if unrealized_pnl <= -stop_loss_pct:
            exit_price = current_close
            pnl = (exit_price - entry_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT
            capital += pnl

            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': '止损',
                'hold_bars': i - entry_idx,
                'hold_hours': max(0, (i - entry_idx) * 4),
                'is_black_swan': is_black_swan_trade,
                'position_size': position_size
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0
            is_black_swan_trade = False

        # 黑天鹅模式的特殊离场逻辑
        elif current_position == 'LONG_BLACKSWAN':
            current_tension = df.loc[i, '张力']

            # 止盈3%
            if unrealized_pnl >= 0.03:
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl_usd': pnl,
                    'exit_reason': '止盈(3%)',
                    'hold_bars': i - entry_idx,
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'is_black_swan': is_black_swan_trade,
                    'position_size': position_size
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0
                is_black_swan_trade = False

            # 张力释放离场
            elif current_tension < 0.30:
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl_usd': pnl,
                    'exit_reason': f'张力释放({current_tension:.3f})',
                    'hold_bars': i - entry_idx,
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'is_black_swan': is_black_swan_trade,
                    'position_size': position_size
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0
                is_black_swan_trade = False

            # 高点离场
            elif is_peak:
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl_usd': pnl,
                    'exit_reason': '高点',
                    'hold_bars': i - entry_idx,
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'is_black_swan': is_black_swan_trade,
                    'position_size': position_size
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0
                is_black_swan_trade = False

            # 最大持仓时间
            elif i - entry_idx >= HOLD_MAX_PERIODS_BLACKSWAN:
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl_usd': pnl,
                    'exit_reason': '最大持仓(20周期)',
                    'hold_bars': i - entry_idx,
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'is_black_swan': is_black_swan_trade,
                    'position_size': position_size
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0
                is_black_swan_trade = False

        # 正常模式的离场逻辑
        elif current_position == 'LONG':
            # 高点离场或最大持仓
            if is_peak or (i - entry_idx >= HOLD_MAX_PERIODS_NORMAL):
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                reason = '高点' if is_peak else f'最大持仓({HOLD_MAX_PERIODS_NORMAL}周期)'

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl_usd': pnl,
                    'exit_reason': reason,
                    'hold_bars': i - entry_idx,
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'is_black_swan': is_black_swan_trade,
                    'position_size': position_size
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0
                is_black_swan_trade = False

    capital_curve.append(capital)

# ==============================================================================
# 统计结果
# ==============================================================================

print("\n" + "="*120)
print("回测结果")
print("="*120)

if len(trades) == 0:
    print("\n没有交易！")
else:
    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    normal_trades = trades_df[~trades_df['is_black_swan']]
    black_swan_trades = trades_df[trades_df['is_black_swan']]

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
    print(f"总交易次数: {total_trades}")

    print(f"\n【交易类型分布】")
    print(f"正常交易: {len(normal_trades)} 笔")
    if len(normal_trades) > 0:
        normal_win_rate = len(normal_trades[normal_trades['pnl_pct'] > 0]) / len(normal_trades) * 100
        normal_total_pnl = normal_trades['pnl_usd'].sum()
        print(f"  胜率: {normal_win_rate:.1f}%")
        print(f"  总盈亏: ${normal_total_pnl:+,.2f}")

    print(f"黑天鹅交易: {len(black_swan_trades)} 笔")
    if len(black_swan_trades) > 0:
        bs_win_rate = len(black_swan_trades[black_swan_trades['pnl_pct'] > 0]) / len(black_swan_trades) * 100
        bs_total_pnl = black_swan_trades['pnl_usd'].sum()
        print(f"  胜率: {bs_win_rate:.1f}%")
        print(f"  总盈亏: ${bs_total_pnl:+,.2f}")

    print(f"\n【胜率统计】")
    win_rate = len(winning_trades) / total_trades * 100
    print(f"总胜率: {win_rate:.1f}%")
    print(f"盈利交易: {len(winning_trades)} 笔")
    print(f"亏损交易: {len(losing_trades)} 笔")

    if len(winning_trades) > 0:
        avg_win = winning_trades['pnl_pct'].mean()
        print(f"平均盈利: {avg_win:+.2f}%")

    if len(losing_trades) > 0:
        avg_loss = losing_trades['pnl_pct'].mean()
        print(f"平均亏损: {avg_loss:+.2f}%")

    if len(winning_trades) > 0 and len(losing_trades) > 0:
        profit_factor = abs(winning_trades['pnl_usd'].sum() / losing_trades['pnl_usd'].sum())
        print(f"盈亏比: {profit_factor:.2f}")

    # 最大回撤
    peak = capital_curve[0]
    max_drawdown = 0
    for cap in capital_curve:
        if cap > peak:
            peak = cap
        drawdown = (peak - cap) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    print(f"\n【风险指标】")
    print(f"最大回撤: {max_drawdown:.2f}%")

    returns = pd.Series(capital_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252*6/4) if returns.std() > 0 else 0
    print(f"夏普比率: {sharpe:.2f}")

    # 保存结果
    trades_df.to_csv('混合策略_回测结果.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 混合策略_回测结果.csv")

print("\n" + "="*120)
print("回测完成")
print("="*120)
