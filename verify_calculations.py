# -*- coding: utf-8 -*-
"""
黄金信号标注计算验证
==================

目的：验证计算逻辑是否准确（没有bug）
- 逐行重新计算，对比结果是否一致
- 确认算法逻辑正确
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("黄金信号标注 - 计算逻辑验证")
print("=" * 100)

# 读取数据
df_gold = pd.read_csv('v705_gold_signals_complete.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"\n总信号数: {len(df_gold)}")

# 抽样验证前10个信号
sample_size = min(10, len(df_gold))
print(f"\n抽样验证前{sample_size}个信号的计算逻辑...")

all_match = True

for i in range(sample_size):
    idx = i
    row = df_gold.iloc[idx]

    signal_time = pd.to_datetime(row['信号时间'])
    direction = row['方向']
    signal_type = row['信号类型']

    print(f"\n{'=' * 100}")
    print(f"信号 #{idx+1}: {signal_type} @ {signal_time}")
    print(f"方向: {direction.upper()}")
    print(f"{'=' * 100}")

    # 找到信号时刻索引
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx_list) == 0:
        print(f"  ⚠ 找不到信号时刻")
        all_match = False
        continue

    signal_idx = signal_idx_list[0]
    signal_price = df_full.loc[signal_idx, 'close']

    print(f"\n【步骤1: 验证黄金开仓计算】")
    print(f"信号时刻: 索引{signal_idx}, 价格 {signal_price:.2f}")

    # 重新计算黄金开仓
    look_ahead = min(10, len(df_full) - signal_idx - 1)
    print(f"向前看: {look_ahead}周期")

    claimed_entry_price = row['黄金开仓价格']
    claimed_wait_period = int(row['黄金开仓等待周期'])

    if direction == 'short':
        # 做空：找最高价格
        best_price = signal_price
        best_period = 0
        print(f"\n做空逻辑: 寻找最高价格")

        for period in range(1, look_ahead + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if period <= 5 or future_price > best_price:
                marker = " [新高]" if future_price > best_price else ""
                print(f"  周期{period}: 索引{future_idx}, 价格{future_price:.2f}{marker}")

            if future_price > best_price:
                best_price = future_price
                best_period = period
    else:  # long
        # 做多：找最低价格
        best_price = signal_price
        best_period = 0
        print(f"\n做多逻辑: 寻找最低价格")

        for period in range(1, look_ahead + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if period <= 5 or future_price < best_price:
                marker = " [新低]" if future_price < best_price else ""
                print(f"  周期{period}: 索引{future_idx}, 价格{future_price:.2f}{marker}")

            if future_price < best_price:
                best_price = future_price
                best_period = period

    print(f"\n计算结果:")
    print(f"  最优等待周期: {best_period}")
    print(f"  最优开仓价格: {best_price:.2f}")

    print(f"\n标注结果:")
    print(f"  等待周期: {claimed_wait_period}")
    print(f"  开仓价格: {claimed_entry_price:.2f}")

    entry_match = np.isclose(claimed_entry_price, best_price, rtol=1e-5) and (claimed_wait_period == best_period)

    if entry_match:
        print(f"  [OK] 黄金开仓计算一致")
    else:
        print(f"  [ERROR] 黄金开仓计算不一致!")
        all_match = False

    # ==================== 验证黄金平仓 ====================
    print(f"\n【步骤2: 验证黄金平仓计算】")

    entry_idx = signal_idx + claimed_wait_period
    entry_price = claimed_entry_price

    print(f"开仓时刻: 索引{entry_idx}, 价格 {entry_price:.2f}")

    look_ahead_exit = min(30, len(df_full) - entry_idx - 1)
    print(f"向前看: {look_ahead_exit}周期")

    claimed_exit_price = row['黄金平仓价格']
    claimed_hold_period = int(row['黄金平仓持仓周期'])
    claimed_pnl = row['最优盈亏%']

    # 重新计算黄金平仓
    best_pnl = -999
    best_exit_price = entry_price
    best_period_exit = 0

    if direction == 'short':
        print(f"\n做空逻辑: 寻找最大盈利点")
        for period in range(1, look_ahead_exit + 1):
            future_idx = entry_idx + period
            future_price = df_full.loc[future_idx, 'close']
            pnl = (entry_price - future_price) / entry_price * 100

            if period <= 5 or pnl > best_pnl:
                marker = " [新高]" if pnl > best_pnl else ""
                print(f"  周期{period}: 价格{future_price:.2f}, 盈亏{pnl:.2f}%{marker}")

            if pnl > best_pnl:
                best_pnl = pnl
                best_exit_price = future_price
                best_period_exit = period
    else:  # long
        print(f"\n做多逻辑: 寻找最大盈利点")
        for period in range(1, look_ahead_exit + 1):
            future_idx = entry_idx + period
            future_price = df_full.loc[future_idx, 'close']
            pnl = (future_price - entry_price) / entry_price * 100

            if period <= 5 or pnl > best_pnl:
                marker = " [新高]" if pnl > best_pnl else ""
                print(f"  周期{period}: 价格{future_price:.2f}, 盈亏{pnl:.2f}%{marker}")

            if pnl > best_pnl:
                best_pnl = pnl
                best_exit_price = future_price
                best_period_exit = period

    print(f"\n计算结果:")
    print(f"  最优持仓周期: {best_period_exit}")
    print(f"  最优平仓价格: {best_exit_price:.2f}")
    print(f"  最优盈亏: {best_pnl:.2f}%")

    print(f"\n标注结果:")
    print(f"  持仓周期: {claimed_hold_period}")
    print(f"  平仓价格: {claimed_exit_price:.2f}")
    print(f"  盈亏: {claimed_pnl:.2f}%")

    exit_match = (np.isclose(claimed_exit_price, best_exit_price, rtol=1e-5) and
                  claimed_hold_period == best_period_exit and
                  np.isclose(claimed_pnl, best_pnl, rtol=1e-3, atol=0.01))

    if exit_match:
        print(f"  [OK] 黄金平仓计算一致")
    else:
        print(f"  [ERROR] 黄金平仓计算不一致!")
        all_match = False

# ==================== 总结 ====================
print("\n" + "=" * 100)
print("计算验证总结")
print("=" * 100)

if all_match:
    print("\n[OK][OK][OK] 抽样验证全部通过！计算逻辑准确无误！")
    print("\n可以进行统计学分析了。")
else:
    print("\n[ERROR] 发现计算不一致，需要检查代码逻辑！")

print("\n" + "=" * 100)
