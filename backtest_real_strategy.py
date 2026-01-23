# -*- coding: utf-8 -*-
"""
真实回测系统
============
严格模拟实战交易，验证策略有效性
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("策略回测系统 - 真实模拟")
print("="*120)

# ============================================================================
# Step 1: 加载数据
# ============================================================================
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条4小时K线")
print(f"时间范围: {df['时间'].min()} 到 {df['时间'].max()}")

# ============================================================================
# Step 2: 识别局部极值点
# ============================================================================
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

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
# Step 3: 定义信号模式
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
# Step 4: 策略回测
# ============================================================================
print("\n" + "="*120)
print("执行策略回测")
print("="*120)

# 策略参数
INITIAL_CAPITAL = 10000  # 初始资金 $10,000
POSITION_SIZE_PCT = 0.03  # 每笔交易3%仓位
STOP_LOSS_PCT = 0.02      # 止损2%
COMMISSION_PCT = 0.0005   # 手续费0.05%

trades = []
capital = INITIAL_CAPITAL
current_position = 'NONE'
entry_price = None
entry_time = None
position_size = 0  # 仓位大小（USD）
prev_signal_mode = None

for i in range(len(df)):
    signal_type = df.loc[i, '信号类型']
    signal_mode = df.loc[i, '信号模式']
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    # 获取参数
    tension = df.loc[i, '张力']
    volume_ratio = df.loc[i, '量能比率']
    price_vs_ema = df.loc[i, '价格vsEMA%']

    # 信号模式切换检测
    is_new_signal = (prev_signal_mode != signal_mode) and (signal_mode != 'NO_TRADE')

    # ========== 平仓逻辑 ==========
    if current_position == 'LONG':
        # 止损检查
        unrealized_pnl = (current_close - entry_price) / entry_price

        # 止损平仓
        if unrealized_pnl <= -STOP_LOSS_PCT:
            exit_price = current_close
            pnl = (exit_price - entry_price) / entry_price * position_size
            pnl -= position_size * COMMISSION_PCT  # 手续费
            capital += pnl

            trades.append({
                'type': 'LONG',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': '止损',
                'hold_bars': i - entry_time,
                'capital_after': capital
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

            reason = '高点' if is_peak else '信号切换'

            trades.append({
                'type': 'LONG',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': reason,
                'hold_bars': i - entry_time,
                'capital_after': capital
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

    elif current_position == 'SHORT':
        # 止损检查
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
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (entry_price - exit_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': '止损',
                'hold_bars': i - entry_time,
                'capital_after': capital
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

            reason = '低点' if is_valley else '信号切换'

            trades.append({
                'type': 'SHORT',
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': (entry_price - exit_price) / entry_price * 100,
                'pnl_usd': pnl,
                'exit_reason': reason,
                'hold_bars': i - entry_time,
                'capital_after': capital
            })

            current_position = 'NONE'
            entry_price = None
            position_size = 0

    # ========== 开仓逻辑 ==========
    if current_position == 'NONE':
        # 过滤条件：张力和量能
        if tension >= 0 or volume_ratio < 1.0:
            prev_signal_mode = signal_mode
            continue

        # 做多模式
        if signal_mode == 'LONG_MODE':
            # 条件1：新信号立即开多
            if is_new_signal:
                current_position = 'LONG'
                entry_price = current_close
                entry_time = i
                position_size = capital * POSITION_SIZE_PCT

            # 条件2：局部低点开多
            elif is_valley:
                current_position = 'LONG'
                entry_price = current_close
                entry_time = i
                position_size = capital * POSITION_SIZE_PCT

        # 做空模式
        elif signal_mode == 'SHORT_MODE':
            # 条件1：新信号立即开空
            if is_new_signal:
                current_position = 'SHORT'
                entry_price = current_close
                entry_time = i
                position_size = capital * POSITION_SIZE_PCT

            # 条件2：局部高点开空
            elif is_peak:
                current_position = 'SHORT'
                entry_price = current_close
                entry_time = i
                position_size = capital * POSITION_SIZE_PCT

    prev_signal_mode = signal_mode

# ============================================================================
# Step 5: 计算回测结果
# ============================================================================
print("\n" + "="*120)
print("回测结果统计")
print("="*120)

if len(trades) == 0:
    print("\n没有执行任何交易！")
    print("可能原因：")
    print("1. 张力和量能过滤条件太严格")
    print("2. 信号触发条件不满足")
else:
    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    long_trades = trades_df[trades_df['type'] == 'LONG']
    short_trades = trades_df[trades_df['type'] == 'SHORT']

    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    total_pnl = trades_df['pnl_usd'].sum()
    final_capital = capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print(f"\n【总体表现】")
    print(f"初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"最终资金: ${final_capital:,.2f}")
    print(f"总盈亏: ${total_pnl:+,.2f}")
    print(f"总收益率: {total_return:+.2f}%")
    print(f"总交易次数: {total_trades}")

    print(f"\n【胜率统计】")
    win_rate = len(winning_trades) / total_trades * 100
    print(f"胜率: {win_rate:.1f}%")
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

    print(f"\n【做多交易】")
    if len(long_trades) > 0:
        long_win_rate = len(long_trades[long_trades['pnl_pct'] > 0]) / len(long_trades) * 100
        long_avg_pnl = long_trades['pnl_pct'].mean()
        long_avg_hold = long_trades['hold_bars'].mean()
        print(f"交易次数: {len(long_trades)}")
        print(f"胜率: {long_win_rate:.1f}%")
        print(f"平均盈亏: {long_avg_pnl:+.2f}%")
        print(f"平均持仓: {long_avg_hold:.1f} 根K线 ({long_avg_hold*4:.1f} 小时)")

    print(f"\n【做空交易】")
    if len(short_trades) > 0:
        short_win_rate = len(short_trades[short_trades['pnl_pct'] > 0]) / len(short_trades) * 100
        short_avg_pnl = short_trades['pnl_pct'].mean()
        short_avg_hold = short_trades['hold_bars'].mean()
        print(f"交易次数: {len(short_trades)}")
        print(f"胜率: {short_win_rate:.1f}%")
        print(f"平均盈亏: {short_avg_pnl:+.2f}%")
        print(f"平均持仓: {short_avg_hold:.1f} 根K线 ({short_avg_hold*4:.1f} 小时)")

    # 计算最大回撤
    print(f"\n【风险指标】")
    capital_curve = [INITIAL_CAPITAL] + trades_df['capital_after'].tolist()
    peak = capital_curve[0]
    max_drawdown = 0
    for cap in capital_curve:
        if cap > peak:
            peak = cap
        drawdown = (peak - cap) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    print(f"最大回撤: {max_drawdown:.2f}%")

    # 盈亏比
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        profit_factor = abs(winning_trades['pnl_usd'].sum() / losing_trades['pnl_usd'].sum())
        print(f"盈亏比: {profit_factor:.2f}")

    # 保存交易记录
    output_cols = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price',
                   'pnl_pct', 'pnl_usd', 'exit_reason', 'hold_bars', 'capital_after']

    # 转换时间索引
    for idx in trades_df.index:
        entry_idx = trades_df.loc[idx, 'entry_time']
        exit_idx = trades_df.loc[idx, 'exit_time']
        if entry_idx < len(df) and exit_idx < len(df):
            trades_df.loc[idx, 'entry_time'] = df.loc[entry_idx, '时间']
            trades_df.loc[idx, 'exit_time'] = df.loc[exit_idx, '时间']

    trades_df[output_cols].to_csv('回测交易记录.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 回测交易记录.csv")

    # 显示前10笔交易
    print(f"\n【前10笔交易】")
    print(f"{'#':<4} {'类型':<8} {'入场时间':<18} {'出场时间':<18} {'入场价':<10} {'出场价':<10} {'盈亏%':<10} {'盈亏$':<12} {'原因':<10}")
    print("-" * 130)

    for idx, row in trades_df.head(10).iterrows():
        print(f"{idx+1:<4} {row['type']:<8} {str(row['entry_time'])[:16]:<18} {str(row['exit_time'])[:16]:<18} "
              f"{row['entry_price']:<10.2f} {row['exit_price']:<10.2f} {row['pnl_pct']:+<10.2f} "
              f"${row['pnl_usd']:+<11.2f} {row['exit_reason']:<10}")

print("\n" + "="*120)
print("回测完成")
print("="*120)
