# -*- coding: utf-8 -*-
"""
黄金信号实战回测 - 宽容参数（鲁棒性版本）
基于实战优化的多空不对称策略
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("黄金信号实战回测 - 宽容参数（鲁棒性版本）")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条4小时K线")
print(f"时间范围: {df['时间'].min()} 到 {df['时间'].max()}")

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
# 实战级黄金信号检测（宽容参数）
# ==============================================================================

print("\n" + "="*120)
print("实战级黄金信号检测标准（宽容参数）")
print("="*120)

print("""
【多头策略：强力回撤接多】
- 加速度: ≤ -0.08（急刹车，有恐慌盘）
- 乖离率: ≤ -0.88%（实质性跌破均线近1%）
- 张力: ≥ 0.48（势能积蓄过半）

【空头策略：缩量阴跌狙击】
- 加速度: 0.04 ~ 0.12（温水煮青蛙，涨不动）
- 量能: ≤ 0.62（没人玩了，一票否决）
- 张力: ≤ -0.39（顶部有压力堆积）
""")

# 检测实战级黄金做多信号
golden_long_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.08) &
    (df['价格vsEMA%'] <= -0.88) &
    (df['张力'] >= 0.48) &
    (df['高低点'] == '低点')
)

# 检测实战级黄金做空信号
golden_short_conditions = (
    (df['信号模式'] == 'SHORT_MODE') &
    (df['加速度'] >= 0.04) &
    (df['加速度'] <= 0.12) &
    (df['量能比率'] <= 0.62) &
    (df['张力'] <= -0.39) &
    (df['高低点'] == '高点')
)

golden_long_signals = df[golden_long_conditions].copy()
golden_short_signals = df[golden_short_conditions].copy()

print(f"\n检测到黄金做多信号: {len(golden_long_signals)} 个")
print(f"检测到黄金做空信号: {len(golden_short_signals)} 个")
print(f"总计: {len(golden_long_signals) + len(golden_short_signals)} 个")

# ==============================================================================
# 回测配置
# ==============================================================================

INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.50  # 50%仓位
STOP_LOSS_PCT = 0.02      # 2%止损
COMMISSION_PCT = 0.0005   # 0.05%手续费
HOLD_MAX_PERIODS = 10     # 最多持有10个周期

print(f"\n回测参数:")
print(f"  初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"  仓位大小: {POSITION_SIZE_PCT*100}%")
print(f"  止损幅度: {STOP_LOSS_PCT*100}%")
print(f"  手续费: {COMMISSION_PCT*100}%")
print(f"  最大持仓周期: {HOLD_MAX_PERIODS}")

# ==============================================================================
# 回测主函数
# ==============================================================================

def backtest_golden_signals(df, long_signals_idx, short_signals_idx):
    """回测黄金信号"""

    capital = INITIAL_CAPITAL
    capital_curve = [INITIAL_CAPITAL]
    trades = []

    current_position = 'NONE'
    entry_price = None
    entry_idx = None
    entry_time = None
    position_size = 0
    entry_type = None

    # 创建信号集合
    long_signal_set = set(long_signals_idx)
    short_signal_set = set(short_signals_idx)

    for i in range(len(df)):
        current_close = df.loc[i, '收盘价']
        current_time = df.loc[i, '时间']
        is_peak = (df.loc[i, '高低点'] == '高点')
        is_valley = (df.loc[i, '高低点'] == '低点')

        # 开仓逻辑
        if current_position == 'NONE':
            if i in long_signal_set:
                # 等待确认（滞后order根K线）
                if i + order < len(df):
                    entry_price = df.loc[i + order, '收盘价']
                    entry_idx = i + order
                    entry_time = df.loc[entry_idx, '时间']
                    position_size = capital * POSITION_SIZE_PCT
                    current_position = 'LONG'
                    entry_type = 'GOLDEN_LONG'

            elif i in short_signal_set:
                # 等待确认（滞后order根K线）
                if i + order < len(df):
                    entry_price = df.loc[i + order, '收盘价']
                    entry_idx = i + order
                    entry_time = df.loc[entry_idx, '时间']
                    position_size = capital * POSITION_SIZE_PCT
                    current_position = 'SHORT'
                    entry_type = 'GOLDEN_SHORT'

        # 平仓逻辑
        if current_position == 'LONG':
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
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'signal_type': entry_type
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0

            # 正常平仓：高点或超过最大持仓周期
            elif is_peak or (i - entry_idx >= HOLD_MAX_PERIODS):
                exit_price = current_close
                pnl = (exit_price - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                reason = '高点' if is_peak else f'最大持仓({HOLD_MAX_PERIODS}周期)'

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
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'signal_type': entry_type
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0

        elif current_position == 'SHORT':
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
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'signal_type': entry_type
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0

            # 正常平仓：低点或超过最大持仓周期
            elif is_valley or (i - entry_idx >= HOLD_MAX_PERIODS):
                exit_price = current_close
                pnl = (entry_price - exit_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl

                reason = '低点' if is_valley else f'最大持仓({HOLD_MAX_PERIODS}周期)'

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
                    'hold_hours': max(0, (i - entry_idx) * 4),
                    'signal_type': entry_type
                })

                current_position = 'NONE'
                entry_price = None
                position_size = 0

        capital_curve.append(capital)

    return trades, capital, capital_curve

# ==============================================================================
# 执行回测
# ==============================================================================

print("\n" + "="*120)
print("执行回测...")
print("="*120)

trades, final_capital, capital_curve = backtest_golden_signals(
    df,
    golden_long_signals.index.tolist(),
    golden_short_signals.index.tolist()
)

# ==============================================================================
# 计算统计结果
# ==============================================================================

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
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

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
    peak = capital_curve[0]
    max_drawdown = 0
    for cap in capital_curve:
        if cap > peak:
            peak = cap
        drawdown = (peak - cap) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    print(f"最大回撤: {max_drawdown:.2f}%")

    returns = pd.Series(capital_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252*6/4) if returns.std() > 0 else 0
    print(f"夏普比率: {sharpe:.2f}")

    # 保存交易记录
    output_cols = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price',
                   'pnl_pct', 'pnl_usd', 'exit_reason', 'hold_bars', 'hold_hours', 'signal_type']
    trades_df[output_cols].to_csv('黄金信号实战回测_宽容参数.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 黄金信号实战回测_宽容参数.csv")

    # 显示所有交易
    print(f"\n【所有交易记录】")
    print(f"{'#':<4} {'类型':<8} {'入场时间':<18} {'出场时间':<18} {'入场价':<10} {'出场价':<10} "
          f"{'盈亏%':<10} {'盈亏$':<12} {'原因':<15} {'持仓h':<8}")
    print("-" * 140)

    for idx, trade in trades_df.iterrows():
        print(f"{idx+1:<4} {trade['type']:<8} {str(trade['entry_time'])[:16]:<18} {str(trade['exit_time'])[:16]:<18} "
              f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} {trade['pnl_pct']:+<10.2f} "
              f"${trade['pnl_usd']:+<11.2f} {trade['exit_reason']:<15} {trade['hold_hours']:<8.1f}")

print("\n" + "="*120)
print("回测完成")
print("="*120)
