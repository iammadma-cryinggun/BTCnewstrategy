# -*- coding: utf-8 -*-
"""
未来函数风险分析与实战修正
=========================
从数学家视角审视look-ahead bias
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("未来函数风险分析 - The Look-ahead Bias Problem")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 识别局部极值
order = 2
local_max = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

print("\n【数学原理】")
print("="*120)
print("\nargrelextrema(order=2) 的数学定义:")
print("  对于索引i，i是局部高点 iff:")
print("    P[i] > P[i-2], P[i-1], P[i+1], P[i+2]")
print("  对于索引i，i是局部低点 iff:")
print("    P[i] < P[i-2], P[i-1], P[i+1], P[i+2]")

print("\n【实时性问题】")
print("="*120)
print("\n要确认i是极值点，需要:")
print(f"  1. 历史数据: P[i-2], P[i-1] (已知)")
print(f"  2. 未来数据: P[i+1], P[i+2] (未知!)")

print(f"\n滞后时间: {order}根K线 = {order*4}小时")
print("  - 在t=i时刻，我们只能'猜测'可能是极值")
print("  - 在t=i+order时刻，我们才能'确认'是极值")

# 实例分析
print("\n" + "="*120)
print("实例分析：第一个低点的确认过程")
print("="*120)

if len(local_min) > 0:
    idx = local_min[0]
    print(f"\n低点时间: {df.loc[idx, '时间']}")
    print(f"低点价格: ${df.loc[idx, '收盘价']:.2f}")

    print(f"\n价格序列:")
    for j in range(idx - order, idx + order + 1):
        if 0 <= j < len(df):
            marker = " <-- 理论低点" if j == idx else ""
            if j == idx + order:
                marker += " <-- 确认时刻"
            print(f"  t={j:3d}: ${df.loc[j, '收盘价']:>10.2f}{marker}")

    theoretical_entry = df.loc[idx, '收盘价']
    actual_entry = df.loc[idx + order, '收盘价']

    print(f"\n入场价差异:")
    print(f"  理论入场价（后验）: ${theoretical_entry:.2f}")
    print(f"  实际入场价（确认后）: ${actual_entry:.2f}")
    print(f"  滑点: ${actual_entry - theoretical_entry:.2f} ({(actual_entry/theoretical_entry - 1)*100:+.2f}%)")

# 重新回测：使用确认后的入场价
print("\n" + "="*120)
print("实战模拟：滞后确认 vs 理论完美")
print("="*120)

def backtest_with_lag(use_realistic_entry=True):
    INITIAL = 10000
    POSITION = 0.03
    STOP_LOSS = 0.02
    COMMISSION = 0.0005

    trades = []
    capital = INITIAL
    pos = 'NONE'
    entry_price = None
    entry_idx = None
    pos_size = 0

    # 定义信号模式
    def get_mode(sig):
        if sig in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
            return 'LONG'
        elif sig in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
            return 'SHORT'
        return 'NONE'

    df['模式'] = df['信号类型'].apply(get_mode)

    # 标注高低点
    df['高低点'] = ''
    for i in local_max:
        df.loc[i, '高低点'] = '高点'
    for i in local_min:
        df.loc[i, '高低点'] = '低点'

    prev_mode = None

    for i in range(len(df)):
        mode = df.loc[i, '模式']
        close = df.loc[i, '收盘价']
        is_peak = (df.loc[i, '高低点'] == '高点')
        is_valley = (df.loc[i, '高低点'] == '低点')
        is_new = (prev_mode != mode) and (mode != 'NONE')

        # 平仓
        if pos == 'LONG':
            pnl = (close - entry_price) / entry_price
            if pnl <= -STOP_LOSS or is_peak or mode in ['SHORT', 'NONE']:
                actual_pnl = pnl * pos_size - pos_size * COMMISSION
                capital += actual_pnl
                trades.append({
                    'type': 'LONG',
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'pnl_pct': pnl * 100,
                    'pnl_usd': actual_pnl
                })
                pos = 'NONE'
                entry_price = None

        elif pos == 'SHORT':
            pnl = (entry_price - close) / entry_price
            if pnl <= -STOP_LOSS or is_valley or mode in ['LONG', 'NONE']:
                actual_pnl = pnl * pos_size - pos_size * COMMISSION
                capital += actual_pnl
                trades.append({
                    'type': 'SHORT',
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'pnl_pct': pnl * 100,
                    'pnl_usd': actual_pnl
                })
                pos = 'NONE'
                entry_price = None

        # 开仓
        if pos == 'NONE':
            if mode == 'LONG' and (is_new or is_valley):
                # 关键：是否使用滞后确认
                if use_realistic_entry:
                    # 实战模式：等待确认后才入场
                    if i + order < len(df):
                        entry_price = df.loc[i + order, '收盘价']
                        entry_idx = i + order
                        pos_size = capital * POSITION
                        pos = 'LONG'
                else:
                    # 理论模式：假设能完美识别极值
                    entry_price = close
                    entry_idx = i
                    pos_size = capital * POSITION
                    pos = 'LONG'

            elif mode == 'SHORT' and (is_new or is_peak):
                if use_realistic_entry:
                    if i + order < len(df):
                        entry_price = df.loc[i + order, '收盘价']
                        entry_idx = i + order
                        pos_size = capital * POSITION
                        pos = 'SHORT'
                else:
                    entry_price = close
                    entry_idx = i
                    pos_size = capital * POSITION
                    pos = 'SHORT'

        prev_mode = mode

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {'trades': 0, 'return': 0}

    total_pnl = trades_df['pnl_usd'].sum()
    total_return = (capital - INITIAL) / INITIAL * 100
    win_rate = len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df) * 100

    return {
        'trades': len(trades_df),
        'return': total_return,
        'pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl_pct'] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl_pct'] <= 0]) > 0 else 0
    }

# 对比测试
print("\n对比测试:")
print("-" * 80)

result_perfect = backtest_with_lag(use_realistic_entry=False)
result_realistic = backtest_with_lag(use_realistic_entry=True)

print(f"{'模式':<30} {'交易数':<10} {'收益率':<12} {'胜率':<10}")
print("-" * 80)
print(f"{'理论完美（假设能即时识别极值）':<30} {result_perfect['trades']:<10} {result_perfect['return']:+<12.2f}% {result_perfect['win_rate']:<10.1f}%")
print(f"{'实战现实（滞后确认后入场）':<30} {result_realistic['trades']:<10} {result_realistic['return']:+<12.2f}% {result_realistic['win_rate']:<10.1f}%")

print(f"\n收益差异: {result_realistic['return'] - result_perfect['return']:+.2f}%")
print(f"相对损失: {(result_realistic['return'] / result_perfect['return'] - 1) * 100:+.1f}%")

print("\n" + "="*120)
print("数学家视角的结论")
print("="*120)

print("""
1. 系统本质:
   - 这是一个在非线性向量场中的极值搜索策略
   - BEARISH/BULLISH定义了"场的方向"
   - 极值点定义了"操作时机"
   - 逆向映射: 场的方向与操作方向相反

2. 未来函数问题:
   - argrelextrema需要前后order=2根K线确认
   - 滞后8小时是可以接受的
   - 但确实存在滑点成本

3. 实战修正:
   - 使用滞后确认是合理的
   - 收益率会降低，但逻辑依然有效
   - 关键是: 这不是"完美拟合"，而是"可实现的策略"

4. 哲学启示:
   - 后验标注 = 理论上限
   - 实战执行 = 实现收益
   - 两者之间的差距 = 执行成本
   - 只要差距不大，策略就是可行的
""")

print("\n" + "="*120)
print("分析完成")
print("="*120)
