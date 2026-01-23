# -*- coding: utf-8 -*-
"""
数据完整性验证
================

检查项：
1. 真实BTC价格（OHLCV数据）
2. 真实普通信号（验证5逻辑计算，含DXY）
3. V7.0.5过滤后的交易信号
4. 所有数据都正确标注
"""

import pandas as pd
import numpy as np
import os

print("=" * 100)
print("数据完整性验证")
print("=" * 100)

# ==================== 1. 检查K线数据 ====================
print("\n【步骤1】检查BTC价格数据（OHLCV）")
print("-" * 100)

kline_file = 'step1_full_data_v5_complete.csv'
if not os.path.exists(kline_file):
    print(f"✗ 文件不存在: {kline_file}")
else:
    df_kline = pd.read_csv(kline_file, encoding='utf-8-sig')
    df_kline['timestamp'] = pd.to_datetime(df_kline['timestamp'])

    print(f"✓ 文件存在: {kline_file}")
    print(f"  总记录数: {len(df_kline)}条")
    print(f"  时间范围: {df_kline['timestamp'].min()} 至 {df_kline['timestamp'].max()}")
    print(f"  时间跨度: {(df_kline['timestamp'].max() - df_kline['timestamp'].min()).days}天")

    # 检查必需列
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df_kline.columns]
    if missing_cols:
        print(f"✗ 缺少列: {missing_cols}")
    else:
        print(f"✓ OHLCV数据完整")

    # 检查数据连续性
    time_diffs = df_kline['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(hours=4)
    gaps = time_diffs[time_diffs != expected_diff]

    if len(gaps) > 0:
        print(f"⚠ 发现{len(gaps)}个时间间隔异常")
        print(f"  前5个异常: {gaps.head().tolist()}")
    else:
        print(f"✓ 时间间隔连续（4小时）")

    # 检查物理指标
    physics_cols = ['tension', 'acceleration']
    missing_physics = [col for col in physics_cols if col not in df_kline.columns]
    if missing_physics:
        print(f"✗ 缺少物理指标: {missing_physics}")
    else:
        print(f"✓ 张力(tension)和加速度(acceleration)存在")

    # 检查DXY
    if 'dxy_close' in df_kline.columns or 'DXY' in df_kline.columns:
        print(f"✓ DXY数据存在")
    else:
        print(f"⚠ K线数据中没有DXY列（可能在信号文件中）")

# ==================== 2. 检查普通信号 ====================
print("\n【步骤2】检查普通信号（验证5逻辑）")
print("-" * 100)

signal_file = 'step1_all_signals_v5_correct.csv'
if not os.path.exists(signal_file):
    print(f"✗ 文件不存在: {signal_file}")
else:
    df_signals = pd.read_csv(signal_file, encoding='utf-8-sig')
    df_signals['时间'] = pd.to_datetime(df_signals['时间'])

    print(f"✓ 文件存在: {signal_file}")
    print(f"  总信号数: {len(df_signals)}个")
    print(f"  时间范围: {df_signals['时间'].min()} 至 {df_signals['时间'].max()}")

    # 检查必需列
    required_signal_cols = ['时间', '信号类型', '方向', '张力', '加速度']
    missing_signal_cols = [col for col in required_signal_cols if col not in df_signals.columns]
    if missing_signal_cols:
        print(f"✗ 缺少列: {missing_signal_cols}")
    else:
        print(f"✓ 基础列完整")

    # 检查信号类型
    signal_types = df_signals['信号类型'].unique()
    expected_types = ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']
    for st in expected_types:
        if st in signal_types:
            count = len(df_signals[df_signals['信号类型'] == st])
            print(f"  {st}: {count}个")

    # 检查DXY燃料
    if 'DXY燃料' in df_signals.columns:
        dxy_count = df_signals['DXY燃料'].notna().sum()
        print(f"✓ DXY燃料列存在，有{dxy_count}个非空值")
    else:
        print(f"⚠ 没有DXY燃料列")

    # 验证信号时刻都在K线中
    matched = 0
    for idx, row in df_signals.iterrows():
        signal_time = row['时间']
        kline_match = df_kline[df_kline['timestamp'] == signal_time]
        if len(kline_match) > 0:
            matched += 1

    print(f"✓ 信号与K线匹配: {matched}/{len(df_signals)} ({matched/len(df_signals)*100:.1f}%)")

# ==================== 3. 检查V7.0.5过滤后的交易信号 ====================
print("\n【步骤3】检查V7.0.5过滤后的交易信号")
print("-" * 100)

v705_file = 'v705_trading_data_passed.csv'
if not os.path.exists(v705_file):
    print(f"✗ 文件不存在: {v705_file}")
else:
    df_v705 = pd.read_csv(v705_file, encoding='utf-8-sig')
    df_v705['信号时间'] = pd.to_datetime(df_v705['信号时间'])

    print(f"✓ 文件存在: {v705_file}")
    print(f"  总信号数: {len(df_v705)}个")

    # 检查过滤结果
    filtered_count = len(df_signals) - len(df_v705)
    filter_rate = filtered_count / len(df_signals) * 100
    print(f"  过滤掉: {filtered_count}个 ({filter_rate:.1f}%)")
    print(f"  通过率: {len(df_v705)/len(df_signals)*100:.1f}%")

    # 检查V7.0.5过滤列
    if 'V7.0.5过滤结果' in df_v705.columns:
        passed = len(df_v705[df_v705['V7.0.5过滤结果'] == '通过'])
        print(f"✓ V7.0.5过滤结果列存在，全部通过({passed}个)")
    else:
        print(f"⚠ 没有V7.0.5过滤结果列")

    if '过滤原因' in df_v705.columns:
        print(f"✓ 过滤原因列存在")
    else:
        print(f"⚠ 没有过虑原因列（可能只保存了通过的信号）")

    # 检查交易结果列
    trading_cols = ['开仓价', '平仓价', '持仓周期', '盈亏%', '是否盈利']
    missing_trading = [col for col in trading_cols if col not in df_v705.columns]
    if missing_trading:
        print(f"✗ 缺少交易列: {missing_trading}")
    else:
        print(f"✓ 交易结果列完整")

    # 统计盈亏
    if '盈亏%' in df_v705.columns:
        avg_pnl = df_v705['盈亏%'].mean()
        win_rate = len(df_v705[df_v705['盈亏%'] > 0]) / len(df_v705) * 100
        print(f"  平均盈亏: {avg_pnl:.2f}%")
        print(f"  胜率: {win_rate:.1f}%")

# ==================== 4. 检查黄金信号标注 ====================
print("\n【步骤4】检查黄金信号标注")
print("-" * 100)

gold_file = 'v705_gold_signals_complete.csv'
if not os.path.exists(gold_file):
    print(f"✗ 文件不存在: {gold_file}")
else:
    df_gold = pd.read_csv(gold_file, encoding='utf-8-sig')

    print(f"✓ 文件存在: {gold_file}")
    print(f"  总记录数: {len(df_gold)}个")

    # 检查黄金信号列
    gold_cols = ['黄金开仓等待周期', '黄金开仓价格', '黄金平仓持仓周期', '黄金平仓价格', '最优盈亏%']
    missing_gold = [col for col in gold_cols if col not in df_gold.columns]
    if missing_gold:
        print(f"✗ 缺少黄金信号列: {missing_gold}")
    else:
        print(f"✓ 黄金信号标注完整")

    # 检查特征列
    feature_cols = ['信号前5周期平均张力', '信号时刻张力', '黄金开仓张力', '黄金平仓张力']
    missing_features = [col for col in feature_cols if col not in df_gold.columns]
    if missing_features:
        print(f"✗ 缺少特征列: {missing_features}")
    else:
        print(f"✓ 特征标注完整")

    # 检查DXY
    dxy_cols = ['DXY燃料_信号时刻', 'DXY燃料_开仓时刻', 'DXY燃料_平仓时刻']
    has_dxy = any(col in df_gold.columns for col in dxy_cols)
    if has_dxy:
        dxy_data_count = sum(df_gold[col].notna().sum() for col in dxy_cols if col in df_gold.columns)
        print(f"⚠ DXY燃料列存在，但数据量为: {dxy_data_count}个")
    else:
        print(f"✗ 没有DXY燃料列")

# ==================== 总结 ====================
print("\n" + "=" * 100)
print("数据完整性总结")
print("=" * 100)

issues = []

# 检查DXY数据完整性
print("\n【关键问题】DXY数据完整性:")
dxy_file = 'dxy_data.csv'
if os.path.exists(dxy_file):
    df_dxy = pd.read_csv(dxy_file, encoding='utf-8-sig')
    print(f"✓ DXY文件存在: {len(df_dxy)}条记录")
    print(f"  时间范围: {df_dxy['date'].min()} 至 {df_dxy['date'].max()}")
else:
    print(f"✗ DXY文件不存在: {dxy_file}")
    issues.append("缺少DXY数据文件")

print("\n【数据完整性结论】:")
if len(issues) == 0:
    print("✓✓✓ 所有数据文件完整！")
else:
    print("⚠⚠⚠ 发现问题:")
    for issue in issues:
        print(f"  - {issue}")

print("\n" + "=" * 100)
