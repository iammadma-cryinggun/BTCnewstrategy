# -*- coding: utf-8 -*-
"""
最优策略实战测试
================
严格模拟实战，使用滞后确认
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("最优策略实战测试 - 严格模拟")
print("="*120)

# ============================================================================
# 1. 加载数据
# ============================================================================
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条4小时K线")
print(f"时间范围: {df['时间'].min()} 到 {df['时间'].max()}")

# ============================================================================
# 2. 识别局部极值点（使用scipy）
# ============================================================================
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

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
    df.loc[idx, '高低点'] = '高点'

# ============================================================================
# 3. 定义信号模式
# ============================================================================
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# ============================================================================
# 4. 回测参数
# ============================================================================
INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.03
STOP_LOSS_PCT = 0.02
COMMISSION_PCT = 0.0005

print(f"\n回测参数:")
print(f"  初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"  仓位大小: {POSITION_SIZE_PCT*100}%")
print(f"  止损幅度: {STOP_LOSS_PCT*100}%")
print(f"  手续费: {COMMISSION_PCT*100}%")
print(f"  极值确认滞后: {order}根K线 ({order*4}小时)")

# ============================================================================
# 5. 回测主循环
# ============================================================================
print("\n" + "="*120)
print("执行回测...")
print("="*120)

trades = []
capital = INITIAL_CAPITAL
current_position = 'NONE'
entry_price = None
entry_time = None
entry_idx = None
position_size = 0
prev_signal_mode = None

capital_curve = [INITIAL_CAPITAL]
capital_times = [df.loc[0, '时间']]

for i in range(len(df)):
    signal_type = df.loc[i, '信号类型']
    signal_mode = df.loc[i, '信号模式']
    current_time = df.loc[i, '时间']
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    # 信号模式切换检测
    is_new_signal = (prev_signal_mode != signal_mode) and (signal_mode != 'NO_TRADE')

    # ========== 平仓逻辑 ==========
    if current_position == 'LONG':
        # 计算未实现盈亏
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 止损平仓
        if unrealized_pnl <= -STOP_LOSS_PCT:
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
                'hold_hours': (i - entry_idx) * 4
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

        # 正常平仓：高点或信号切换
        elif is_peak or (signal_mode in ['SHORT_MODE', 'NO_TRADE']):
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
                'hold_hours': (i - entry_idx) * 4
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

    elif current_position == 'SHORT':
        # 计算未实现盈亏
        unrealized_pnl = (entry_price - current_close) / entry_price

        # 止损平仓
        if unrealized_pnl <= -STOP_LOSS_PCT:
            exit_price = current_close
            pnl = (entry_price - exit_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT
            capital += pnl

            trades.append({
                'type': 'SHORT',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (entry_price - exit_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': '止损',
                'hold_bars': i - entry_idx,
                'hold_hours': (i - entry_idx) * 4
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

        # 正常平仓：低点或信号切换
        elif is_valley or (signal_mode in ['LONG_MODE', 'NO_TRADE']):
            exit_price = current_close
            pnl = (entry_price - exit_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT
            capital += pnl

            reason = '低点' if is_valley else ('LONG模式' if signal_mode == 'LONG_MODE' else 'OSCILLATION')

            trades.append({
                'type': 'SHORT',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (entry_price - exit_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': reason,
                'hold_bars': i - entry_idx,
                'hold_hours': (i - entry_idx) * 4
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

    # ========== 开仓逻辑 ==========
    if current_position == 'NONE':
        # 做多模式
        if signal_mode == 'LONG_MODE':
            # 条件1: 新信号立即开多
            if is_new_signal:
                # 等待确认：在i+order时刻入场
                if i + order < len(df):
                    entry_price = df.loc[i + order, '收盘价']
                    entry_time = df.loc[i + order, '时间']
                    entry_idx = i + order
                    position_size = capital * POSITION_SIZE_PCT
                    current_position = 'LONG'

            # 条件2: 遇到局部低点开多
            elif is_valley:
                # 等待确认：在i+order时刻入场
                if i + order < len(df):
                    entry_price = df.loc[i + order, '收盘价']
                    entry_time = df.loc[i + order, '时间']
                    entry_idx = i + order
                    position_size = capital * POSITION_SIZE_PCT
                    current_position = 'LONG'

        # 做空模式
        elif signal_mode == 'SHORT_MODE':
            # 条件1: 新信号立即开空
            if is_new_signal:
                # 等待确认：在i+order时刻入场
                if i + order < len(df):
                    entry_price = df.loc[i + order, '收盘价']
                    entry_time = df.loc[i + order, '时间']
                    entry_idx = i + order
                    position_size = capital * POSITION_SIZE_PCT
                    current_position = 'SHORT'

            # 条件2: 遇到局部高点开空
            elif is_peak:
                # 等待确认：在i+order时刻入场
                if i + order < len(df):
                    entry_price = df.loc[i + order, '收盘价']
                    entry_time = df.loc[i + order, '时间']
                    entry_idx = i + order
                    position_size = capital * POSITION_SIZE_PCT
                    current_position = 'SHORT'

    # 更新信号模式
    prev_signal_mode = signal_mode

    # 记录资金曲线
    capital_curve.append(capital)
    capital_times.append(current_time)

# ============================================================================
# 6. 计算统计结果
# ============================================================================
print("\n" + "="*120)
print("回测结果统计")
print("="*120)

trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("\n没有执行任何交易！")
else:
    total_trades = len(trades_df)
    long_trades = trades_df[trades_df['type'] == 'LONG']
    short_trades = trades_df[trades_df['type'] == 'SHORT']

    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    total_pnl = trades_df['pnl_usd'].sum()
    final_capital = capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 计算年化收益
    days = (df['时间'].max() - df['时间'].min()).days
    annualized_return = (1 + total_return/100) ** (365/days) - 1

    print(f"\n【总体表现】")
    print(f"初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"最终资金: ${final_capital:,.2f}")
    print(f"总盈亏: ${total_pnl:+,.2f}")
    print(f"总收益率: {total_return:+.2f}%")
    print(f"年化收益率: {annualized_return*100:+.2f}%")
    print(f"测试周期: {days}天 ({days/30:.1f}个月)")
    print(f"总交易次数: {total_trades}")

    print(f"\n【胜率统计】")
    win_rate = len(winning_trades) / total_trades * 100
    print(f"胜率: {win_rate:.1f}%")
    print(f"盈利交易: {len(winning_trades)} 笔")
    print(f"亏损交易: {len(losing_trades)} 笔")

    if len(winning_trades) > 0:
        avg_win = winning_trades['pnl_pct'].mean()
        max_win = winning_trades['pnl_pct'].max()
        avg_win_hours = winning_trades['hold_hours'].mean()
        print(f"平均盈利: {avg_win:+.2f}%")
        print(f"最大盈利: {max_win:+.2f}%")
        print(f"平均持仓时间: {avg_win_hours:.1f}小时")

    if len(losing_trades) > 0:
        avg_loss = losing_trades['pnl_pct'].mean()
        max_loss = losing_trades['pnl_pct'].min()
        avg_loss_hours = losing_trades['hold_hours'].mean()
        print(f"平均亏损: {avg_loss:+.2f}%")
        print(f"最大亏损: {max_loss:+.2f}%")
        print(f"平均持仓时间: {avg_loss_hours:.1f}小时")

    # 盈亏比
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        profit_factor = abs(winning_trades['pnl_usd'].sum() / losing_trades['pnl_usd'].sum())
        print(f"盈亏比: {profit_factor:.2f}")

    print(f"\n【做多交易】")
    if len(long_trades) > 0:
        long_win_rate = len(long_trades[long_trades['pnl_pct'] > 0]) / len(long_trades) * 100
        long_avg_pnl = long_trades['pnl_pct'].mean()
        long_avg_hold = long_trades['hold_hours'].mean()
        long_total_pnl = long_trades['pnl_usd'].sum()
        print(f"交易次数: {len(long_trades)}")
        print(f"胜率: {long_win_rate:.1f}%")
        print(f"平均盈亏: {long_avg_pnl:+.2f}%")
        print(f"总盈亏: ${long_total_pnl:+,.2f}")
        print(f"平均持仓: {long_avg_hold:.1f}小时")

    print(f"\n【做空交易】")
    if len(short_trades) > 0:
        short_win_rate = len(short_trades[short_trades['pnl_pct'] > 0]) / len(short_trades) * 100
        short_avg_pnl = short_trades['pnl_pct'].mean()
        short_avg_hold = short_trades['hold_hours'].mean()
        short_total_pnl = short_trades['pnl_usd'].sum()
        print(f"交易次数: {len(short_trades)}")
        print(f"胜率: {short_win_rate:.1f}%")
        print(f"平均盈亏: {short_avg_pnl:+.2f}%")
        print(f"总盈亏: ${short_total_pnl:+,.2f}")
        print(f"平均持仓: {short_avg_hold:.1f}小时")

    # 风险指标
    print(f"\n【风险指标】")
    # 最大回撤
    peak = capital_curve[0]
    max_drawdown = 0
    max_drawdown_idx = 0
    for i, cap in enumerate(capital_curve):
        if cap > peak:
            peak = cap
        drawdown = (peak - cap) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_idx = i

    print(f"最大回撤: {max_drawdown:.2f}%")

    # 夏普比率（假设无风险利率=0）
    returns = pd.Series(capital_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252*6/4) if returns.std() > 0 else 0
    print(f"夏普比率: {sharpe:.2f}")

    # 保存交易记录
    output_cols = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price',
                   'pnl_pct', 'pnl_usd', 'exit_reason', 'hold_bars', 'hold_hours']
    trades_df[output_cols].to_csv('最优策略_实战测试.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 最优策略_实战测试.csv")

    # 显示所有交易
    print(f"\n【所有交易记录】")
    print(f"{'#':<4} {'类型':<8} {'入场时间':<18} {'出场时间':<18} {'入场价':<10} {'出场价':<10} "
          f"{'盈亏%':<10} {'盈亏$':<12} {'原因':<12} {'持仓h':<8}")
    print("-" * 140)

    for idx, trade in trades_df.iterrows():
        print(f"{idx+1:<4} {trade['type']:<8} {str(trade['entry_time'])[:16]:<18} {str(trade['exit_time'])[:16]:<18} "
              f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} {trade['pnl_pct']:+<10.2f} "
              f"${trade['pnl_usd']:+<11.2f} {trade['exit_reason']:<12} {trade['hold_hours']:<8.1f}")

print("\n" + "="*120)
print("测试完成")
print("="*120)
