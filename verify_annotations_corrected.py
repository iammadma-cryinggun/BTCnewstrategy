# -*- coding: utf-8 -*-
"""
黄金信号标注验证（修正版）
========================

从数学角度验证标注的正确性
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("黄金信号标注数学验证（修正版）")
print("=" * 100)

# 读取数据
df_gold = pd.read_csv('v705_gold_signals_complete.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"\n总信号数: {len(df_gold)}")

# ==================== 验证黄金开仓标注 ====================
print("\n" + "=" * 100)
print("步骤1: 验证黄金开仓标注")
print("=" * 100)

entry_errors = []
entry_correct = 0

for idx, row in df_gold.iterrows():
    signal_time = pd.to_datetime(row['信号时间'])
    direction = row['方向']
    claimed_wait_period = int(row['黄金开仓等待周期'])
    claimed_entry_price = row['黄金开仓价格']

    # 找到信号时刻索引
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx_list) == 0:
        entry_errors.append(f"信号#{idx}: 找不到信号时刻")
        continue

    signal_idx = signal_idx_list[0]
    signal_price = df_full.loc[signal_idx, 'close']

    # 向前看最多10个周期
    look_ahead = min(10, len(df_full) - signal_idx - 1)

    # 手动计算最优开仓价格
    if direction == 'short':
        # 做空：找最高价格
        best_price = signal_price
        best_period = 0
        best_idx = signal_idx

        for period in range(1, look_ahead + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if future_price > best_price:
                best_price = future_price
                best_period = period
                best_idx = future_idx
    else:  # long
        # 做多：找最低价格
        best_price = signal_price
        best_period = 0
        best_idx = signal_idx

        for period in range(1, look_ahead + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if future_price < best_price:
                best_price = future_price
                best_period = period
                best_idx = future_idx

    # 验证标注是否正确
    price_match = np.isclose(claimed_entry_price, best_price, rtol=1e-5)
    period_match = (claimed_wait_period == best_period)

    if price_match and period_match:
        entry_correct += 1
    else:
        entry_errors.append({
            '索引': idx,
            '方向': direction,
            '声称等待周期': claimed_wait_period,
            '实际最优周期': best_period,
            '声称价格': claimed_entry_price,
            '实际最优价格': best_price,
            '信号价格': signal_price
        })

print(f"\n黄金开仓验证结果:")
print(f"  正确: {entry_correct}/{len(df_gold)} ({entry_correct/len(df_gold)*100:.1f}%)")
print(f"  错误: {len(entry_errors)}个")

if len(entry_errors) > 0 and len(entry_errors) <= 10:
    print(f"\n前{len(entry_errors)}个错误详情:")
    for err in entry_errors[:10]:
        if isinstance(err, dict):
            print(f"\n信号#{err['索引']}: {err['方向']}")
            print(f"  声称: 等待{err['声称等待周期']}周期, 价格{err['声称价格']:.2f}")
            print(f"  实际: 等待{err['实际最优周期']}周期, 价格{err['实际最优价格']:.2f}")
            print(f"  信号价格: {err['信号价格']:.2f}")
        else:
            print(err)

# ==================== 验证黄金平仓标注 ====================
print("\n" + "=" * 100)
print("步骤2: 验证黄金平仓标注")
print("=" * 100)

exit_errors = []
exit_correct = 0

for idx, row in df_gold.iterrows():
    signal_time = pd.to_datetime(row['信号时间'])
    direction = row['方向']
    entry_wait_period = int(row['黄金开仓等待周期'])

    # 找到开仓时刻
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx_list) == 0:
        exit_errors.append(f"信号#{idx}: 找不到信号时刻")
        continue

    signal_idx = signal_idx_list[0]
    entry_idx = signal_idx + entry_wait_period
    entry_price = row['黄金开仓价格']

    claimed_hold_period = int(row['黄金平仓持仓周期'])
    claimed_exit_price = row['黄金平仓价格']
    claimed_pnl = row['最优盈亏%']

    # 向前看最多30个周期
    look_ahead = min(30, len(df_full) - entry_idx - 1)

    # 手动计算最优平仓
    best_pnl = -999
    best_exit_price = entry_price
    best_period = 0

    for period in range(1, look_ahead + 1):
        future_idx = entry_idx + period
        future_price = df_full.loc[future_idx, 'close']

        if direction == 'short':
            pnl = (entry_price - future_price) / entry_price * 100
        else:  # long
            pnl = (future_price - entry_price) / entry_price * 100

        if pnl > best_pnl:
            best_pnl = pnl
            best_exit_price = future_price
            best_period = period

    # 验证标注是否正确
    price_match = np.isclose(claimed_exit_price, best_exit_price, rtol=1e-5)
    period_match = (claimed_hold_period == best_period)
    pnl_match = np.isclose(claimed_pnl, best_pnl, rtol=1e-3, atol=0.01)

    if price_match and period_match and pnl_match:
        exit_correct += 1
    else:
        exit_errors.append({
            '索引': idx,
            '方向': direction,
            '声称持仓周期': claimed_hold_period,
            '实际最优周期': best_period,
            '声称价格': claimed_exit_price,
            '实际最优价格': best_exit_price,
            '声称盈亏': claimed_pnl,
            '实际盈亏': best_pnl,
            '开仓价': entry_price
        })

print(f"\n黄金平仓验证结果:")
print(f"  正确: {exit_correct}/{len(df_gold)} ({exit_correct/len(df_gold)*100:.1f}%)")
print(f"  错误: {len(exit_errors)}个")

if len(exit_errors) > 0 and len(exit_errors) <= 10:
    print(f"\n前{len(exit_errors)}个错误详情:")
    for err in exit_errors[:10]:
        print(f"\n信号#{err['索引']}: {err['方向']}, 开仓价{err['开仓价']:.2f}")
        print(f"  声称: 持仓{err['声称持仓周期']}周期, 平仓价{err['声称价格']:.2f}, 盈亏{err['声称盈亏']:.2f}%")
        print(f"  实际: 持仓{err['实际最优周期']}周期, 平仓价{err['实际最优价格']:.2f}, 盈亏{err['实际盈亏']:.2f}%")

# ==================== 总结 ====================
print("\n" + "=" * 100)
print("验证总结")
print("=" * 100)

print(f"\n黄金开仓标注正确率: {entry_correct/len(df_gold)*100:.2f}%")
print(f"黄金平仓标注正确率: {exit_correct/len(df_gold)*100:.2f}%")

if entry_correct == len(df_gold) and exit_correct == len(df_gold):
    print("\n✓ 所有标注完全正确，可以进行统计分析！")
else:
    print(f"\n⚠ 发现{len(entry_errors)}个开仓错误, {len(exit_errors)}个平仓错误")
    print("需要修复标注代码")

print("\n" + "=" * 100)
