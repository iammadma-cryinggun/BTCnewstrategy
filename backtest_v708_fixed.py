# -*- coding: utf-8 -*-
"""
V7.0.8修复后的完整回测（使用正确信号方向+DXY支持）
验证目标：
1. 使用正确的反向策略（BEARISH→long, BULLISH→short）
2. 验证信号方向修复是否正确
3. 测试DXY燃料增强功能

时间范围：最近1000条4H数据（约5个月）
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入V7.0.7系统组件
from v707_trader_main import DataFetcher, PhysicsSignalCalculator, V707TraderConfig

print("=" * 80)
print("V7.0.8修复后完整回测")
print("=" * 80)
print(f"验证目标：")
print(f"  1. 使用正确的反向策略信号方向")
print(f"  2. 验证DXY燃料增强功能")
print(f"  3. 统计正确信号的盈亏表现")
print("=" * 80)
print()

# ==================== 配置 ====================
config = V707TraderConfig()
fetcher = DataFetcher(config)
calculator = PhysicsSignalCalculator(config)

# ==================== 获取历史数据 ====================
print("正在获取历史数据...")
print("正在从Binance获取历史数据...")

df = fetcher.fetch_btc_data(interval='4h', limit=1000)

if df is None or len(df) < 100:
    print("[ERROR] 数据不足，无法回测")
    sys.exit(1)

print(f"[OK] 获取数据：{len(df)}条")
print(f"时间范围：{df.index[0]} 到 {df.index[-1]}")
print()

# ==================== 获取DXY数据 ====================
print("正在获取DXY数据...")
dxy_df = fetcher.fetch_dxy_data(limit=100)
if dxy_df is not None:
    print(f"[OK] DXY数据：{len(dxy_df)}条")
else:
    print("[WARNING] DXY数据获取失败，将使用dxy_fuel=0.0")
print()

# ==================== 计算物理学指标 ====================
print("正在计算物理学指标...")
df_metrics = calculator.calculate_physics_metrics(df)

if df_metrics is None or len(df_metrics) < 100:
    print("[ERROR] 指标计算失败")
    sys.exit(1)

print(f"[OK] 计算完成：{len(df_metrics)}条")
print()

# ==================== 回测主逻辑 ====================
print("开始回测...")
print()

# 统计数据
all_signals = []
trades = []

for i in range(50, len(df_metrics)):
    current_time = df_metrics.index[i]
    current_price = df_metrics['close'].iloc[i]
    current_tension = df_metrics['tension'].iloc[i]
    current_accel = df_metrics['acceleration'].iloc[i]
    current_volume = df_metrics['volume'].iloc[i]

    # 计算量能比率
    avg_volume = df_metrics['volume'].iloc[i-20:i].mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    # 计算DXY燃料
    dxy_fuel = 0.0
    if dxy_df is not None:
        dxy_fuel = calculator.calculate_dxy_fuel(dxy_df, current_time)

    # ==================== 检测信号（使用DXY增强） ====================
    signal_type, confidence, description = calculator.diagnose_regime(
        tension=current_tension,
        acceleration=current_accel,
        dxy_fuel=dxy_fuel
    )

    if signal_type is None:
        continue

    # 记录信号
    all_signals.append({
        'time': current_time,
        'price': current_price,
        'signal_type': signal_type,
        'confidence': confidence,
        'description': description,
        'dxy_fuel': dxy_fuel
    })

    # 确定交易方向（使用正确的反向策略）
    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'  # 系统看涨/高位 → 我们反向做空
    else:  # BEARISH_SINGULARITY, LOW_OSCILLATION
        direction = 'long'   # 系统看空/低位 → 我们反向做多

    # ==================== 模拟交易 ====================
    # 找平仓点（固定止盈止损）
    for j in range(i+1, min(i+30, len(df_metrics))):
        hold_periods = j - i
        exit_p = df_metrics['close'].iloc[j]

        # 计算PNL
        if direction == 'short':
            pnl = (current_price - exit_p) / current_price * 100
            tp_price = current_price * 0.95
            sl_price = current_price * 1.025
        else:  # long
            pnl = (exit_p - current_price) / current_price * 100
            tp_price = current_price * 1.05
            sl_price = current_price * 0.975

        # 止盈止损或超时
        should_exit = False
        exit_reason = ""

        if direction == 'short':
            if exit_p <= tp_price:
                should_exit = True
                exit_reason = "止盈(+5%)"
            elif exit_p >= sl_price:
                should_exit = True
                exit_reason = "止损(-2.5%)"
        else:  # long
            if exit_p >= tp_price:
                should_exit = True
                exit_reason = "止盈(+5%)"
            elif exit_p <= sl_price:
                should_exit = True
                exit_reason = "止损(-2.5%)"

        if should_exit or hold_periods >= 10:
            if not should_exit:
                exit_reason = f"超时({hold_periods}周期)"

            trades.append({
                'direction': direction,
                'signal_type': signal_type,
                'entry_time': current_time,
                'entry_price': current_price,
                'exit_price': exit_p,
                'pnl': pnl,
                'hold_periods': hold_periods,
                'exit_reason': exit_reason,
                'confidence': confidence,
                'dxy_fuel': dxy_fuel
            })
            break

# ==================== 生成报告 ====================
print("=" * 80)
print("回测完成！生成报告...")
print("=" * 80)
print()

# 1. 信号统计
print("[一] 信号统计")
print("-" * 80)
print(f"总信号数: {len(all_signals)}")
for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    count = sum(1 for s in all_signals if s['signal_type'] == sig_type)
    if count > 0:
        print(f"  {sig_type}: {count}个")
print()

# 2. 交易统计（按方向）
print("[二] 交易统计（按交易方向）")
print("-" * 80)

for direction in ['long', 'short']:
    direction_cn = '做多LONG' if direction == 'long' else '做空SHORT'
    dir_trades = [t for t in trades if t['direction'] == direction]

    if len(dir_trades) > 0:
        avg_pnl = np.mean([t['pnl'] for t in dir_trades])
        win_rate = sum(1 for t in dir_trades if t['pnl'] > 0) / len(dir_trades) * 100

        print(f"\n{direction_cn}:")
        print(f"  交易次数: {len(dir_trades)}")
        print(f"  平均盈亏: {avg_pnl:+.2f}%")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  最大盈利: {max(t['pnl'] for t in dir_trades):+.2f}%")
        print(f"  最大亏损: {min(t['pnl'] for t in dir_trades):+.2f}%")

# 3. 交易统计（按信号类型）
print("\n" + "=" * 80)
print("[三] 交易统计（按信号类型）")
print("-" * 80)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    sig_trades = [t for t in trades if t['signal_type'] == sig_type]

    if len(sig_trades) > 0:
        avg_pnl = np.mean([t['pnl'] for t in sig_trades])
        win_rate = sum(1 for t in sig_trades if t['pnl'] > 0) / len(sig_trades) * 100

        print(f"\n{sig_type}:")
        print(f"  交易次数: {len(sig_trades)}")
        print(f"  平均盈亏: {avg_pnl:+.2f}%")
        print(f"  胜率: {win_rate:.1f}%")

# 4. DXY增强效果
print("\n" + "=" * 80)
print("[四] DXY燃料增强效果")
print("-" * 80)

trades_with_dxy = [t for t in trades if t['dxy_fuel'] > 0]
trades_without_dxy = [t for t in trades if t['dxy_fuel'] == 0]

if len(trades_with_dxy) > 0:
    avg_pnl_with = np.mean([t['pnl'] for t in trades_with_dxy])
    win_rate_with = sum(1 for t in trades_with_dxy if t['pnl'] > 0) / len(trades_with_dxy) * 100
    print(f"\n使用DXY增强 ({len(trades_with_dxy)}笔):")
    print(f"  平均盈亏: {avg_pnl_with:+.2f}%")
    print(f"  胜率: {win_rate_with:.1f}%")

if len(trades_without_dxy) > 0:
    avg_pnl_without = np.mean([t['pnl'] for t in trades_without_dxy])
    win_rate_without = sum(1 for t in trades_without_dxy if t['pnl'] > 0) / len(trades_without_dxy) * 100
    print(f"\n未使用DXY增强 ({len(trades_without_dxy)}笔):")
    print(f"  平均盈亏: {avg_pnl_without:+.2f}%")
    print(f"  胜率: {win_rate_without:.1f}%")

# 5. 总体统计
print("\n" + "=" * 80)
print("[五] 总体统计")
print("-" * 80)

if len(trades) > 0:
    total_pnl = np.mean([t['pnl'] for t in trades])
    total_win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100

    print(f"\n总交易次数: {len(trades)}")
    print(f"平均盈亏: {total_pnl:+.2f}%")
    print(f"总胜率: {total_win_rate:.1f}%")

# 6. 保存详细结果
print("\n" + "=" * 80)
print("保存详细结果...")
print()

output_file = 'v708_fixed_backtest_results.csv'
if trades:
    df_results = pd.DataFrame(trades)
    df_results.to_csv(output_file, index=False)
    print(f"[OK] 详细结果已保存到: {output_file}")
    print(f"   共 {len(trades)} 笔交易")

print("\n" + "=" * 80)
print("回测完成！")
print("=" * 80)
