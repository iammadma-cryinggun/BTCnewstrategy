# -*- coding: utf-8 -*-
"""
黄金信号分析 - 逐周期检查方法（正确版）
=====================================

按照正确的定义：
1. 最优开仓价：从信号开始逐周期检查，在方向改变前找到的最有利价格
2. 最优平仓价：从开仓后逐周期检查，在趋势反转前找到的最有利价格
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("黄金信号分析 - 逐周期检查方法（正确版）")
print("=" * 80)

# ==================== 读取数据 ====================
print("\n读取数据...")

df_signals = pd.read_csv('step1_entry_signals_with_dxy.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"[OK] 开仓信号: {len(df_signals)}个")
print(f"[OK] 完整数据: {len(df_full)}条")

# ==================== 分析每个信号 ====================
print("\n正在逐周期分析每个信号...")

analysis_results = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    signal_tension = signal['张力']

    # 确定交易方向
    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'
    else:
        direction = 'long'

    # 在完整数据中找到这个信号的位置
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx_list) == 0:
        continue

    signal_idx = signal_idx_list[0]

    # 分析后续30个周期
    look_ahead_periods = 30

    if signal_idx + look_ahead_periods >= len(df_full):
        continue

    # ==================== 第一阶段：找最优开仓价 ====================
    # 从信号开始逐周期检查，在方向改变前找最有利价格

    best_entry_price = signal_price  # 初始值就是首次价格
    best_entry_period = 0  # 0表示立即开仓
    direction_changed = False
    change_period = 0

    if direction == 'short':
        # 做空：找最高价（在价格开始下跌之前）
        for period in range(1, min(11, look_ahead_periods + 1)):  # 最多看10个周期
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            # 如果出现更高的价格，更新最优开仓价
            if future_price > best_entry_price:
                best_entry_price = future_price
                best_entry_period = period

            # 检测方向改变：价格开始下跌（连续2个周期下跌）
            if period >= 2:
                prev1_price = df_full.loc[signal_idx + period - 1, 'close']
                prev2_price = df_full.loc[signal_idx + period - 2, 'close']

                if (prev1_price < signal_price and prev2_price < signal_price):
                    direction_changed = True
                    change_period = period - 2
                    break

    else:  # long
        # 做多：找最低价（在价格开始上涨之前）
        for period in range(1, min(11, look_ahead_periods + 1)):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            # 如果出现更低的价格，更新最优开仓价
            if future_price < best_entry_price:
                best_entry_price = future_price
                best_entry_period = period

            # 检测方向改变：价格开始上涨（连续2个周期上涨）
            if period >= 2:
                prev1_price = df_full.loc[signal_idx + period - 1, 'close']
                prev2_price = df_full.loc[signal_idx + period - 2, 'close']

                if (prev1_price > signal_price and prev2_price > signal_price):
                    direction_changed = True
                    change_period = period - 2
                    break

    # 计算价格优势
    price_advantage = (best_entry_price - signal_price) / signal_price * 100

    # ==================== 第二阶段：找最优平仓价 ====================
    # 从最优开仓点开始，在趋势反转前找最有利平仓价

    entry_idx = signal_idx + best_entry_period
    best_exit_price = best_entry_price
    best_exit_pnl = 0.0
    best_exit_period = 0
    trend_reversed = False

    # 继续往后看（最多30周期）
    max_look = look_ahead_periods - best_entry_period

    if direction == 'short':
        # 做空：从开仓后找最低价（在价格反弹之前）
        min_price = best_entry_price

        for period in range(1, max_look + 1):
            future_idx = entry_idx + period
            if future_idx >= len(df_full):
                break

            future_price = df_full.loc[future_idx, 'close']

            # 如果出现更低的价格，更新最优平仓价
            if future_price < min_price:
                min_price = future_price
                best_exit_price = future_price
                best_exit_period = period

            # 检测趋势反转：价格开始反弹（连续2个周期上涨）
            if period >= 2:
                prev1_price = df_full.loc[entry_idx + period - 1, 'close']
                prev2_price = df_full.loc[entry_idx + period - 2, 'close']

                if (future_price > prev1_price and prev1_price > prev2_price):
                    trend_reversed = True
                    break

    else:  # long
        # 做多：从开仓后找最高价（在价格回调之前）
        max_price = best_entry_price

        for period in range(1, max_look + 1):
            future_idx = entry_idx + period
            if future_idx >= len(df_full):
                break

            future_price = df_full.loc[future_idx, 'close']

            # 如果出现更高的价格，更新最优平仓价
            if future_price > max_price:
                max_price = future_price
                best_exit_price = future_price
                best_exit_period = period

            # 检测趋势反转：价格开始回调（连续2个周期下跌）
            if period >= 2:
                prev1_price = df_full.loc[entry_idx + period - 1, 'close']
                prev2_price = df_full.loc[entry_idx + period - 2, 'close']

                if (future_price < prev1_price and prev1_price < prev2_price):
                    trend_reversed = True
                    break

    # 计算盈亏
    if direction == 'short':
        best_exit_pnl = (best_entry_price - best_exit_price) / best_entry_price * 100
    else:
        best_exit_pnl = (best_exit_price - best_entry_price) / best_entry_price * 100

    # 首次开仓盈亏（第一个周期的价格）
    first_price = df_full.loc[signal_idx + 1, 'close']
    if direction == 'short':
        first_pnl = (signal_price - first_price) / signal_price * 100
    else:
        first_pnl = (first_price - signal_price) / signal_price * 100

    # 记录结果
    analysis_results.append({
        '信号时间': signal_time,
        '信号类型': signal_type,
        '方向': direction,
        '首次开仓价': signal_price,
        '首次开仓盈亏%': first_pnl,

        '最优开仓价': best_entry_price,
        '最优开仓周期': best_entry_period,
        '价格优势%': price_advantage,

        '最优平仓价': best_exit_price,
        '最优平仓周期': best_exit_period,
        '最优平仓盈亏%': best_exit_pnl,

        '方向改变': direction_changed,
        '方向改变周期': change_period,
        '趋势反转': trend_reversed,
    })

df_analysis = pd.DataFrame(analysis_results)

print(f"[OK] 分析完成: {len(df_analysis)}个信号")
print()

# ==================== 统计摘要 ====================
print("=" * 80)
print("统计摘要")
print("=" * 80)

# 好机会定义：最优平仓盈利>0
df_analysis['是好机会'] = df_analysis['最优平仓盈亏%'] > 0

good = df_analysis[df_analysis['是好机会'] == True]
bad = df_analysis[df_analysis['是好机会'] == False]

print(f"\n总信号数: {len(df_analysis)}个")
print(f"好机会: {len(good)}个 ({len(good)/len(df_analysis)*100:.1f}%)")
print(f"坏机会: {len(bad)}个 ({len(bad)/len(df_analysis)*100:.1f}%)")

print(f"\n按方向统计:")
for direction in ['short', 'long']:
    df_dir = df_analysis[df_analysis['方向'] == direction]
    good_dir = df_dir[df_dir['是好机会'] == True]

    print(f"\n{direction.upper()}信号:")
    print(f"  总数: {len(df_dir)}个")
    print(f"  好机会: {len(good_dir)}个 ({len(good_dir)/len(df_dir)*100:.1f}%)")
    print(f"  平均首次盈亏: {df_dir['首次开仓盈亏%'].mean():+.2f}%")
    print(f"  平均最优盈亏: {df_dir['最优平仓盈亏%'].mean():+.2f}%")
    print(f"  平均最优开仓周期: {df_dir['最优开仓周期'].mean():.1f}周期")
    print(f"  平均最优平仓周期: {df_dir['最优平仓周期'].mean():.1f}周期")

print()
print("=" * 80)
print("[OK] 分析完成")
print("=" * 80)

# ==================== 保存数据 ====================
print("\n正在保存数据...")
df_analysis.to_csv('step2_3_correct_golden_analysis.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 已保存: step2_3_correct_golden_analysis.csv")
