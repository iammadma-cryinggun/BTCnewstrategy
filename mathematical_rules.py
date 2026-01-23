# -*- coding: utf-8 -*-
"""
数学化交易规律分析
==================
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*140)
print("MATHEMATICAL TRADING RULES ANALYSIS")
print("="*140)

# ============================================================================
# Step 1: 定义数学概念
# ============================================================================
print("\n" + "="*140)
print("STEP 1: Mathematical Definitions")
print("="*140)

print("【局部极值的数学定义】")
print("")
print("给定价格序列 P[0], P[1], ..., P[n-1]")
print("")
print("对于位置 i (2 <= i <= n-3):")
print("  P[i] 是局部高点 当且仅当 P[i] > P[i-2], P[i] > P[i-1], P[i] > P[i+1], P[i] > P[i+2]")
print("  P[i] 是局部低点 当且仅当 P[i] < P[i-2], P[i] < P[i-1], P[i] < P[i+1], P[i] < P[i+2]")
print("")
print("使用 scipy.signal.argrelextrema(order=2) 实现:")
print("  局部高点索引集 H = {i | P[i] = max(P[i-2:i+3])}")
print("  局部低点索引集 L = {i | P[i] = min(P[i-2:i+3])}")
print("")

# ============================================================================
# Step 2: 加载数据并验证
# ============================================================================
print("\n" + "="*140)
print("STEP 2: Load Data and Verify")
print("="*140)

df = pd.read_csv('最终数据_完整合并.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 找出所有交易回合
trades = []  # (入场索引, 出场索引, 类型, 入场价, 出场价, 盈亏%)

i = 0
while i < len(df):
    action = df.loc[i, '最优动作']

    # 检测开仓
    if '开多' in action or '开空' in action:
        entry_idx = i
        entry_type = 'LONG' if '开多' in action else 'SHORT'
        entry_price = df.loc[i, '收盘价']

        # 找到对应的平仓点
        exit_idx = None
        exit_price = None
        exit_action = None

        for j in range(i + 1, len(df)):
            future_action = df.loc[j, '最优动作']

            if entry_type == 'LONG':
                if '平多' in future_action:
                    exit_idx = j
                    exit_price = df.loc[j, '收盘价']
                    exit_action = future_action
                    break
            else:  # SHORT
                if '平空' in future_action:
                    exit_idx = j
                    exit_price = df.loc[j, '收盘价']
                    exit_action = future_action
                    break

        if exit_idx is not None:
            # 计算盈亏
            if entry_type == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            # 持仓K线数
            hold_bars = exit_idx - entry_idx

            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'type': entry_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'hold_bars': hold_bars,
                'entry_signal': df.loc[i, '信号类型'],
                'exit_signal': df.loc[exit_idx, '信号类型'],
                'entry_action': action,
                'exit_action': exit_action
            })

            i = exit_idx  # 跳到平仓后
        else:
            i += 1
    else:
        i += 1

print(f"\n找到 {len(trades)} 个完整交易回合")

# ============================================================================
# Step 3: 交易规律数学化
# ============================================================================
print("\n" + "="*140)
print("STEP 3: Mathematical Trading Rules")
print("="*140)

print("""
【交易规则的形式化定义】

令 S[i] ∈ {BEARISH, BULLISH, OSCILLATION} 为信号类型
令 M[i] ∈ {LONG_MODE, SHORT_MODE, NO_TRADE} 为信号模式
令 H, L 为局部高点、低点索引集

信号模式映射:
  M[i] = LONG_MODE     if S[i] ∈ {BEARISH_SINGULARITY, LOW_OSCILLATION}
  M[i] = SHORT_MODE    if S[i] ∈ {BULLISH_SINGULARITY, HIGH_OSCILLATION}
  M[i] = NO_TRADE      if S[i] = OSCILLATION

持仓状态 X[i] ∈ {NONE, LONG, SHORT}

交易决策函数 F(i, X[i], M[i]):
""")

# 分析交易规律
long_trades = [t for t in trades if t['type'] == 'LONG']
short_trades = [t for t in trades if t['type'] == 'SHORT']

print(f"\n做多交易统计 (共{len(long_trades)}笔):")
if len(long_trades) > 0:
    avg_pnl = np.mean([t['pnl_pct'] for t in long_trades])
    avg_hold = np.mean([t['hold_bars'] for t in long_trades])
    win_rate = len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades) * 100

    print(f"  平均盈利: {avg_pnl:+.2f}%")
    print(f"  平均持仓: {avg_hold:.1f} 根K线 ({avg_hold*4:.1f} 小时)")
    print(f"  胜率: {win_rate:.1f}%")

    # 盈亏分布
    profits = [t['pnl_pct'] for t in long_trades if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in long_trades if t['pnl_pct'] < 0]

    if profits:
        print(f"  平均盈利: {np.mean(profits):+.2f}% (共{len(profits)}笔)")
    if losses:
        print(f"  平均亏损: {np.mean(losses):+.2f}% (共{len(losses)}笔)")

print(f"\n做空交易统计 (共{len(short_trades)}笔):")
if len(short_trades) > 0:
    avg_pnl = np.mean([t['pnl_pct'] for t in short_trades])
    avg_hold = np.mean([t['hold_bars'] for t in short_trades])
    win_rate = len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades) * 100

    print(f"  平均盈利: {avg_pnl:+.2f}%")
    print(f"  平均持仓: {avg_hold:.1f} 根K线 ({avg_hold*4:.1f} 小时)")
    print(f"  胜率: {win_rate:.1f}%")

    # 盈亏分布
    profits = [t['pnl_pct'] for t in short_trades if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in short_trades if t['pnl_pct'] < 0]

    if profits:
        print(f"  平均盈利: {np.mean(profits):+.2f}% (共{len(profits)}笔)")
    if losses:
        print(f"  平均亏损: {np.mean(losses):+.2f}% (共{len(losses)}笔)")

# ============================================================================
# Step 4: 开仓/平仓点的数学特征
# ============================================================================
print("\n" + "="*140)
print("STEP 4: Entry/Exit Point Characteristics")
print("="*140)

# 分析做多交易的开仓/平仓特征
if len(long_trades) > 0:
    print("\n【做多交易 LONG_MODE】")

    entry_points = []
    exit_points = []

    for t in long_trades:
        entry_idx = t['entry_idx']
        exit_idx = t['exit_idx']

        entry_points.append({
            'is_valley': df.loc[entry_idx, '高低点'] == '低点',
            'volume_ratio': df.loc[entry_idx, '量能比率'],
            'price_vs_ema': df.loc[entry_idx, '价格vsEMA%'],
            'tension': df.loc[entry_idx, '张力'],
            'acceleration': df.loc[entry_idx, '加速度']
        })

        exit_points.append({
            'is_peak': df.loc[exit_idx, '高低点'] == '高点',
            'volume_ratio': df.loc[exit_idx, '量能比率'],
            'price_vs_ema': df.loc[exit_idx, '价格vsEMA%'],
            'tension': df.loc[exit_idx, '张力'],
            'acceleration': df.loc[exit_idx, '加速度']
        })

    entry_df = pd.DataFrame(entry_points)
    exit_df = pd.DataFrame(exit_points)

    print(f"\n开仓点特征 (n={len(entry_df)}):")
    print(f"  在局部低点开仓: {entry_df['is_valley'].sum()} / {len(entry_df)} ({entry_df['is_valley'].mean()*100:.1f}%)")
    print(f"  量能比率: {entry_df['volume_ratio'].mean():.4f} ± {entry_df['volume_ratio'].std():.4f}")
    print(f"  价格vsEMA: {entry_df['price_vs_ema'].mean():.4f}% ± {entry_df['price_vs_ema'].std():.4f}%")
    print(f"  张力: {entry_df['tension'].mean():.4f} ± {entry_df['tension'].std():.4f}")
    print(f"  加速度: {entry_df['acceleration'].mean():.4f} ± {entry_df['acceleration'].std():.4f}")

    print(f"\n平仓点特征 (n={len(exit_df)}):")
    print(f"  在局部高点平仓: {exit_df['is_peak'].sum()} / {len(exit_df)} ({exit_df['is_peak'].mean()*100:.1f}%)")
    print(f"  量能比率: {exit_df['volume_ratio'].mean():.4f} ± {exit_df['volume_ratio'].std():.4f}")
    print(f"  价格vsEMA: {exit_df['price_vs_ema'].mean():.4f}% ± {exit_df['price_vs_ema'].std():.4f}%")
    print(f"  张力: {exit_df['tension'].mean():.4f} ± {exit_df['tension'].std():.4f}")
    print(f"  加速度: {exit_df['acceleration'].mean():.4f} ± {exit_df['acceleration'].std():.4f}")

# 分析做空交易的开仓/平仓特征
if len(short_trades) > 0:
    print("\n【做空交易 SHORT_MODE】")

    entry_points = []
    exit_points = []

    for t in short_trades:
        entry_idx = t['entry_idx']
        exit_idx = t['exit_idx']

        entry_points.append({
            'is_peak': df.loc[entry_idx, '高低点'] == '高点',
            'volume_ratio': df.loc[entry_idx, '量能比率'],
            'price_vs_ema': df.loc[entry_idx, '价格vsEMA%'],
            'tension': df.loc[entry_idx, '张力'],
            'acceleration': df.loc[entry_idx, '加速度']
        })

        exit_points.append({
            'is_valley': df.loc[exit_idx, '高低点'] == '低点',
            'volume_ratio': df.loc[exit_idx, '量能比率'],
            'price_vs_ema': df.loc[exit_idx, '价格vsEMA%'],
            'tension': df.loc[exit_idx, '张力'],
            'acceleration': df.loc[exit_idx, '加速度']
        })

    entry_df = pd.DataFrame(entry_points)
    exit_df = pd.DataFrame(exit_points)

    print(f"\n开仓点特征 (n={len(entry_df)}):")
    print(f"  在局部高点开仓: {entry_df['is_peak'].sum()} / {len(entry_df)} ({entry_df['is_peak'].mean()*100:.1f}%)")
    print(f"  量能比率: {entry_df['volume_ratio'].mean():.4f} ± {entry_df['volume_ratio'].std():.4f}")
    print(f"  价格vsEMA: {entry_df['price_vs_ema'].mean():.4f}% ± {entry_df['price_vs_ema'].std():.4f}%")
    print(f"  张力: {entry_df['tension'].mean():.4f} ± {entry_df['tension'].std():.4f}")
    print(f"  加速度: {entry_df['acceleration'].mean():.4f} ± {entry_df['acceleration'].std():.4f}")

    print(f"\n平仓点特征 (n={len(exit_df)}):")
    print(f"  在局部低点平仓: {exit_df['is_valley'].sum()} / {len(exit_df)} ({exit_df['is_valley'].mean()*100:.1f}%)")
    print(f"  量能比率: {exit_df['volume_ratio'].mean():.4f} ± {exit_df['volume_ratio'].std():.4f}")
    print(f"  价格vsEMA: {exit_df['price_vs_ema'].mean():.4f}% ± {exit_df['price_vs_ema'].std():.4f}%")
    print(f"  张力: {exit_df['tension'].mean():.4f} ± {exit_df['tension'].std():.4f}")
    print(f"  加速度: {exit_df['acceleration'].mean():.4f} ± {exit_df['acceleration'].std():.4f}")

# ============================================================================
# Step 5: 总结数学规律
# ============================================================================
print("\n" + "="*140)
print("STEP 5: Mathematical Trading Rules Summary")
print("="*140)

print("""
【完整的交易规则数学表达】

规则1: 信号模式映射
  M[i] = f(S[i])
         = LONG_MODE  if S[i] ∈ {BEARISH, LOW_OSC}
         = SHORT_MODE if S[i] ∈ {BULLISH, HIGH_OSC}
         = NO_TRADE   if S[i] = OSCILLATION

规则2: 做多交易 (M[i] = LONG_MODE)

  开仓条件 (X[i] = NONE):
    ├─ 信号模式刚切换: M[i] ≠ M[i-1] → 立即开多
    └─ 或 遇到局部低点: i ∈ L → 开多

  持仓条件 (X[i] = LONG):
    ├─ 遇到局部低点: i ∈ L → 刷新入场价 (平多/开多)
    └─ 遇到局部高点: i ∈ H → 平多

  数学表达:
    IF X[i-1] = NONE AND (M[i] ≠ M[i-1] OR i ∈ L):
      X[i] = LONG, entry_price = P[i]
    ELSE IF X[i-1] = LONG:
      IF i ∈ H: X[i] = NONE (平仓)
      ELSE IF i ∈ L: entry_price = P[i] (刷新)
      ELSE: X[i] = LONG (继续持有)

规则3: 做空交易 (M[i] = SHORT_MODE)

  开仓条件 (X[i] = NONE):
    ├─ 信号模式刚切换: M[i] ≠ M[i-1] → 立即开空
    └─ 或 遇到局部高点: i ∈ H → 开空

  持仓条件 (X[i] = SHORT):
    ├─ 遇到局部高点: i ∈ H → 刷新入场价 (平空/开空)
    └─ 遇到局部低点: i ∈ L → 平空

  数学表达:
    IF X[i-1] = NONE AND (M[i] ≠ M[i-1] OR i ∈ H):
      X[i] = SHORT, entry_price = P[i]
    ELSE IF X[i-1] = SHORT:
      IF i ∈ L: X[i] = NONE (平仓)
      ELSE IF i ∈ H: entry_price = P[i] (刷新)
      ELSE: X[i] = SHORT (继续持有)

规则4: OSCILLATION模式 (M[i] = NO_TRADE)

  IF X[i-1] = LONG AND i ∈ H: X[i] = NONE
  IF X[i-1] = SHORT AND i ∈ L: X[i] = NONE
  ELSE: 保持当前持仓
""")

# ============================================================================
# Step 6: 保存交易记录
# ============================================================================
print("\n" + "="*140)
print("STEP 6: Save Trade Records")
print("="*140)

if trades:
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('交易记录_完整.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存 {len(trades_df)} 笔交易到: 交易记录_完整.csv")

    # 显示前10笔交易
    print("\n前10笔交易:")
    print(f"{'#':<4} {'类型':<8} {'入场时间':<18} {'出场时间':<18} {'入场价':<10} {'出场价':<10} {'盈亏%':<10} {'持仓K线':<10}")
    print("-" * 120)

    for i, trade in enumerate(trades[:10]):
        entry_time = str(df.loc[trade['entry_idx'], '时间'])[:16]
        exit_time = str(df.loc[trade['exit_idx'], '时间'])[:16]
        trade_type = trade['type']

        print(f"{i+1:<4} {trade_type:<8} {entry_time:<18} {exit_time:<18} "
              f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} "
              f"{trade['pnl_pct']:+<10.2f} {trade['hold_bars']:<10}")

print("\n" + "="*140)
print("COMPLETE")
print("="*140)
