# -*- coding: utf-8 -*-
"""
狙击手战术手册 (The Sniper Protocol) - 完整回测
经过实战检验的钢铁法则
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("狙击手战术手册 (The Sniper Protocol)")
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
# 模块1: 信号 (Trigger)
# ==============================================================================

print("\n" + "="*120)
print("模块1: 信号检测 (Trigger)")
print("="*120)

print("""
核心参数：
1. 加速度 ≤ -0.20 (极速撞击)
2. 张力 ≥ 0.70 (极限拉伸)
3. 下影线 < 0.35 (光脚大阴线)
4. 冷却期 > 12根K线 (48小时缓冲)
""")

signal_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.20) &
    (df['张力'] >= 0.70) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

potential_signals = df[signal_conditions].copy()

print(f"通过物理过滤器: {len(potential_signals)} 个")

# ==============================================================================
# 模块2: 入场 (Entry) + 模块3: 风控 (Shield)
# ==============================================================================

print("\n" + "="*120)
print("模块2&3: 入场与风控 (Entry & Shield)")
print("="*120)

INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.50
STOP_LOSS_PCT = 0.02  # 固定2%止损
COMMISSION_PCT = 0.0005

print(f"""
入场规则：
- Buy Stop @ 最高价 × 1.0001
- 只有突破才成交

风控规则：
- 固定 {STOP_LOSS_PCT*100}% 止损
- 初始资金: ${INITIAL_CAPITAL:,.2f}
- 仓位大小: {POSITION_SIZE_PCT*100}%
""")

confirmed_trades = []
last_exit_idx = -100  # 上次离场的位置，初始化为很早的值

for idx, row in potential_signals.iterrows():
    # 冷却期检查：距离上次离场必须 > 12根K线
    bars_since_exit = idx - last_exit_idx
    if bars_since_exit <= 12:
        print(f"[冷却中] {row['时间']} - 距离上次离场仅{bars_since_exit}根K线，需要>12根")
        continue

    signal_high = row['最高价']
    entry_price = signal_high * 1.0001

    # 检查后续10根K线是否成交
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
        print(f"[未成交] {row['时间']} - 挂单 ${entry_price:,.2f} 未被触发")
        continue

    # 成交了，记录入场
    entry_time = df.loc[fill_bar, '时间']
    print(f"\n[成交] {entry_time}")
    print(f"  信号: 加速度={row['加速度']:.4f}, 张力={row['张力']:.3f}, 下影线={row['下影线']:.3f}")
    print(f"  入场: ${fill_price:,.2f}")
    print(f"  止损: ${fill_price * (1 - STOP_LOSS_PCT):,.2f} ({STOP_LOSS_PCT*100}%)")

    # ========================================================================
    # 模块4: 离场 (Exit) - 张力释放
    # ========================================================================

    exit_triggered = False
    exit_price = None
    exit_bar = None
    exit_reason = None

    # 检查后续K线，直到触发离场条件
    for i in range(fill_bar + 1, min(fill_bar + 42, len(df))):
        current_price = df.loc[i, '收盘价']
        current_tension = df.loc[i, '张力']
        current_low = df.loc[i, '最低价']

        # 止损检查
        unrealized_pnl = (current_price - fill_price) / fill_price
        if unrealized_pnl <= -STOP_LOSS_PCT:
            exit_price = current_price
            exit_bar = i
            exit_reason = f'止损({STOP_LOSS_PCT*100}%)'
            exit_triggered = True
            break

        # 张力释放离场
        if current_tension < 0.40:
            exit_price = current_price
            exit_bar = i
            exit_reason = f'张力释放({current_tension:.3f})'
            exit_triggered = True
            break

    # 如果没有触发离场，用最后的价格
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
        'entry_idx': fill_bar,
        'exit_idx': exit_bar,
        'entry_price': fill_price,
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'pnl_usd': pnl_usd,
        'exit_reason': exit_reason,
        'hold_bars': hold_bars,
        'hold_hours': hold_hours,
        'signal_accel': row['加速度'],
        'signal_tension': row['张力']
    })

    print(f"  离场: {exit_time} @ ${exit_price:,.2f}")
    print(f"  原因: {exit_reason}")
    print(f"  收益: {pnl_pct:+.2f}% (${pnl_usd:+,.2f})")
    print(f"  持仓: {hold_bars}周期 ({hold_hours}小时)")

    # 更新最后离场位置（用于冷却期计算）
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

    # 最大回撤
    capital_curve = [INITIAL_CAPITAL]
    for trade in confirmed_trades:
        capital_curve.append(capital_curve[-1] + trade['pnl_usd'])

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

    # 保存交易记录
    output_cols = ['entry_time', 'exit_time', 'entry_price', 'exit_price',
                   'pnl_pct', 'pnl_usd', 'exit_reason', 'hold_bars', 'hold_hours']
    trades_df[output_cols].to_csv('狙击手战术手册_回测结果.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存: 狙击手战术手册_回测结果.csv")

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
print("狙击手战术手册 - 回测完成")
print("="*120)
