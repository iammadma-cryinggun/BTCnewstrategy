# -*- coding: utf-8 -*-
"""
融合策略 - 取两套系统之长
简单极值点策略 + 黑天鹅系统的风控精华
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("融合策略：简单极值点 + 黑天鹅风控")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

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
# 融合策略：
# 1. 入场：简单极值点（BEARISH + 低点）
# 2. 风控：黑天鹅的宽止损（2%）
# 3. 离场：下一个极值点
# ==============================================================================

print("""
策略说明：
【入场】简单极值点
- BEARISH_SINGULARITY信号 + 低点确认
- 滞后2根K线入场（避免未来函数）

【风控】黑天鹅精华
- 固定2%止损（扛住流动性猎杀）

【离场】物理规律
- 遇到高点离场
- 最大持仓10周期
""")

# ==============================================================================
# 回测配置
# ==============================================================================

INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.50
STOP_LOSS_PCT = 0.02
COMMISSION_PCT = 0.0005
HOLD_MAX_PERIODS = 10

print(f"回测参数：")
print(f"- 初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"- 仓位: {POSITION_SIZE_PCT*100}%")
print(f"- 止损: {STOP_LOSS_PCT*100}%")
print(f"- 最大持仓: {HOLD_MAX_PERIODS}周期")
print(f"- 手续费: {COMMISSION_PCT*100}%")

# ==============================================================================
# 回测主函数
# ==============================================================================

capital = INITIAL_CAPITAL
capital_curve = [INITIAL_CAPITAL]
trades = []

current_position = 'NONE'
entry_price = None
entry_idx = None
entry_time = None
position_size = 0

# 极值点集合
peak_set = set(df[df['高低点'] == '高点'].index)
valley_set = set(df[df['高低点'] == '低点'].index)

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']
    is_peak = (i in peak_set)
    is_valley = (i in valley_set)
    signal_mode = df.loc[i, '信号模式']

    # 开仓逻辑：简单极值点
    if current_position == 'NONE':
        # 做多：BEARISH信号 + 低点
        if signal_mode == 'LONG_MODE' and is_valley:
            # 滞后2根K线入场
            if i + order < len(df):
                entry_price = df.loc[i + order, '收盘价']
                entry_idx = i + order
                entry_time = df.loc[entry_idx, '时间']
                position_size = capital * POSITION_SIZE_PCT
                current_position = 'LONG'

    # 平仓逻辑
    if current_position == 'LONG':
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 止损
        if unrealized_pnl <= -STOP_LOSS_PCT:
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
                'hold_hours': max(0, (i - entry_idx) * 4)
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

        # 正常平仓：高点或最大持仓
        elif is_peak or (i - entry_idx >= HOLD_MAX_PERIODS):
            exit_price = current_close
            pnl = (exit_price - entry_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT
            capital += pnl

            reason = '高点' if is_peak else f'最大持仓({HOLD_MAX_PERIODS}周期)'

            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': reason,
                'hold_bars': i - entry_idx,
                'hold_hours': max(0, (i - entry_idx) * 4)
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

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

    print(f"\n【胜率统计】")
    win_rate = len(winning_trades) / total_trades * 100
    print(f"胜率: {win_rate:.1f}%")
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
    trades_df.to_csv('融合策略_回测结果.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 融合策略_回测结果.csv")

print("\n" + "="*120)
print("回测完成")
print("="*120)
