# -*- coding: utf-8 -*-
"""
混合策略V2 - 5倍杠杆版本
基于用户实际交易习惯
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("混合策略V2 - 5倍杠杆版本")
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

black_swan_count = df['是黑天鹅'].sum()
print(f"\n检测到黑天鹅信号: {black_swan_count} 个")

print("""
策略说明：
【基础逻辑】严格复现简单策略（70%胜率）
- 入场：信号模式切换（LONG/SHORT）
- 离场：高点或信号切换
- 滞后确认：2根K线

【黑天鹅增强】当入场信号是黑天鹅时
- 正常：30%仓位（5倍杠杆=150%实际），1.5%止损
- 黑天鹅：50%仓位（5倍杠杆=250%实际），2.0%止损，张力释放离场
""")

# ==============================================================================
# 回测参数
# ==============================================================================

INITIAL_CAPITAL = 10000
LEVERAGE = 5.0  # 5倍杠杆
COMMISSION_PCT = 0.0005

# 正常交易参数
NORMAL_POSITION_PCT = 0.30  # 30%仓位
NORMAL_STOP_LOSS_PCT = 0.015

# 黑天鹅交易参数
BS_POSITION_PCT = 0.50  # 50%仓位（黑天鹅时重仓）
BS_STOP_LOSS_PCT = 0.02

print(f"回测参数：")
print(f"- 初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"- 杠杆倍数: {LEVERAGE}x")
print(f"- 正常仓位: {NORMAL_POSITION_PCT*100}% (实际{NORMAL_POSITION_PCT*LEVERAGE*100:.0f}%), 止损: {NORMAL_STOP_LOSS_PCT*100}%")
print(f"- 黑天鹅仓位: {BS_POSITION_PCT*100}% (实际{BS_POSITION_PCT*LEVERAGE*100:.0f}%), 止损: {BS_STOP_LOSS_PCT*100}%")

# ==============================================================================
# 回测主循环
# ==============================================================================

trades = []
capital = INITIAL_CAPITAL
current_position = 'NONE'
entry_price = None
entry_time = None
entry_idx = None
position_size = None  # 绝对仓位大小
stop_loss_pct = None
is_black_swan_trade = False
entry_signal_idx = None

capital_curve = [INITIAL_CAPITAL]
capital_times = [df.loc[0, '时间']]

prev_signal_mode = None

for i in range(len(df)):
    signal_type = df.loc[i, '信号类型']
    signal_mode = df.loc[i, '信号模式']
    current_time = df.loc[i, '时间']
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    # 信号模式切换检测
    is_new_signal = (prev_signal_mode != signal_mode) and (signal_mode != 'NO_TRADE')

    # ==================== 平仓逻辑 ====================

    if current_position == 'LONG':
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 黑天鹅交易的张力释放离场
        if is_black_swan_trade:
            current_tension = df.loc[i, '张力']
            if current_tension < 0.30:
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                trades.append({
                    'type': 'LONG',
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'pnl_usd': pnl,
                    'exit_reason': f'张力释放({current_tension:.3f})',
                    'hold_bars': i - entry_idx,
                    'hold_hours': (i - entry_idx) * 4,
                    'is_black_swan': True
                })

                current_position = 'NONE'
                entry_price = None
                position_size = None
                is_black_swan_trade = False
                stop_loss_pct = None

        # 止损平仓
        if current_position == 'LONG' and unrealized_pnl <= -stop_loss_pct:
            exit_price = current_close
            pnl = (exit_price - entry_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT
            capital += pnl

            trades.append({
                'type': 'LONG',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': '止损',
                'hold_bars': i - entry_idx,
                'hold_hours': (i - entry_idx) * 4,
                'is_black_swan': is_black_swan_trade
            })

            current_position = 'NONE'
            entry_price = None
            position_size = None
            is_black_swan_trade = False
            stop_loss_pct = None

        # 正常平仓：高点或信号切换
        elif current_position == 'LONG' and (is_peak or signal_mode in ['SHORT_MODE', 'NO_TRADE']):
            exit_price = current_close
            pnl = (exit_price - entry_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT
            capital += pnl

            reason = '高点' if is_peak else ('SHORT模式' if signal_mode == 'SHORT_MODE' else 'OSCILLATION')

            trades.append({
                'type': 'LONG',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': reason,
                'hold_bars': i - entry_idx,
                'hold_hours': (i - entry_idx) * 4,
                'is_black_swan': is_black_swan_trade
            })

            current_position = 'NONE'
            entry_price = None
            position_size = None
            is_black_swan_trade = False
            stop_loss_pct = None

    # ==================== 开仓逻辑 ====================

    if current_position == 'NONE' and is_new_signal:
        # 做多信号
        if signal_mode == 'LONG_MODE':
            # 滞后2根K线入场
            if i + order < len(df):
                entry_price = df.loc[i + order, '收盘价']
                entry_idx = i + order
                entry_time = df.loc[entry_idx, '时间']
                entry_signal_idx = i

                # 检查原始信号是否是黑天鹅
                if df.loc[entry_signal_idx, '是黑天鹅']:
                    # 黑天鹅：50%仓位 × 5倍杠杆 = 250%实际仓位
                    position_size = capital * BS_POSITION_PCT * LEVERAGE
                    stop_loss_pct = BS_STOP_LOSS_PCT
                    is_black_swan_trade = True
                    current_position = 'LONG'
                else:
                    # 正常：30%仓位 × 5倍杠杆 = 150%实际仓位
                    position_size = capital * NORMAL_POSITION_PCT * LEVERAGE
                    stop_loss_pct = NORMAL_STOP_LOSS_PCT
                    is_black_swan_trade = False
                    current_position = 'LONG'

    prev_signal_mode = signal_mode
    capital_curve.append(capital)
    capital_times.append(current_time)

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
    print(f"正常交易: {len(normal_trades)} 笔 (150%实际仓位)")
    if len(normal_trades) > 0:
        normal_win_rate = len(normal_trades[normal_trades['pnl_pct'] > 0]) / len(normal_trades) * 100
        normal_total_pnl = normal_trades['pnl_usd'].sum()
        print(f"  胜率: {normal_win_rate:.1f}%")
        print(f"  总盈亏: ${normal_total_pnl:+,.2f}")

    print(f"黑天鹅交易: {len(black_swan_trades)} 笔 (250%实际仓位)")
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
        max_win = winning_trades['pnl_pct'].max()
        print(f"平均盈利: {avg_win:+.2f}%")
        print(f"最大盈利: {max_win:+.2f}%")

    if len(losing_trades) > 0:
        avg_loss = losing_trades['pnl_pct'].mean()
        max_loss = losing_trades['pnl_pct'].min()
        print(f"平均亏损: {avg_loss:+.2f}%")
        print(f"最大亏损: {max_loss:+.2f}%")

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
    trades_df.to_csv('混合策略V2_5倍杠杆_回测结果.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 混合策略V2_5倍杠杆_回测结果.csv")

print("\n" + "="*120)
print("回测完成")
print("="*120)
