# -*- coding: utf-8 -*-
"""
从数据中学习真实的信号-方向对应关系
====================================

原则：
1. 不假设任何对应关系
2. 分析每个信号类型在不同方向下的真实盈亏
3. 找出最优的交易方向
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("从数据中学习真实的信号-方向对应关系")
print("=" * 100)

# 读取完整生命周期数据
df = pd.read_csv('complete_signal_lifecycle.csv', encoding='utf-8-sig')

print(f"\n总信号数: {len(df)}")
print(f"\n信号类型分布:")
print(df['信号类型'].value_counts())

# ==================== 关键：不假设对应关系，分析所有可能性 ====================
print("\n" + "=" * 100)
print("关键分析：每个信号类型在不同方向下的真实盈亏")
print("=" * 100)

for signal_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df[df['信号类型'] == signal_type].copy()

    if len(df_type) == 0:
        continue

    print(f"\n{'=' * 100}")
    print(f"{signal_type}")
    print(f"{'=' * 100}")
    print(f"总信号数: {len(df_type)}个")

    # 分析原始标记的方向
    for direction in ['long', 'short']:
        df_dir = df_type[df_type['方向'] == direction]

        if len(df_dir) > 0:
            print(f"\n【原始标记为{direction.upper()}】")
            print(f"  数量: {len(df_dir)}个")
            print(f"  平均盈亏: {df_dir['最优盈亏%'].mean():.2f}%")
            print(f"  盈利数: {len(df_dir[df_dir['最优盈亏%'] > 0])}个 ({len(df_dir[df_dir['最优盈亏%'] > 0])/len(df_dir)*100:.1f}%)")
            print(f"  亏损数: {len(df_dir[df_dir['最优盈亏%'] < 0])}个 ({len(df_dir[df_dir['最优盈亏%'] < 0])/len(df_dir)*100:.1f}%)")
            print(f"  最大盈利: {df_dir['最优盈亏%'].max():.2f}%")
            print(f"  最大亏损: {df_dir['最优盈亏%'].min():.2f}%")

    # ==================== 反向测试：如果做反方向会怎样？ ====================
    print(f"\n【反向测试分析】")

    # 如果原始是short，我们测试如果做long会怎样
    df_original_short = df_type[df_type['方向'] == 'short']
    if len(df_original_short) > 0:
        # 原始是short，如果做long，盈亏就是负的
        reverse_pnl = -df_original_short['最优盈亏%']
        print(f"\n原始标记为SHORT，如果实际做LONG:")
        print(f"  平均盈亏: {reverse_pnl.mean():.2f}%")
        print(f"  盈利数: {len(reverse_pnl[reverse_pnl > 0])}个 ({len(reverse_pnl[reverse_pnl > 0])/len(reverse_pnl)*100:.1f}%)")
        print(f"  亏损数: {len(reverse_pnl[reverse_pnl < 0])}个 ({len(reverse_pnl[reverse_pnl < 0])/len(reverse_pnl)*100:.1f}%)")
        print(f"  最大盈利: {reverse_pnl.max():.2f}%")
        print(f"  最大亏损: {reverse_pnl.min():.2f}%")

    # 如果原始是long，我们测试如果做short会怎样
    df_original_long = df_type[df_type['方向'] == 'long']
    if len(df_original_long) > 0:
        # 原始是long，如果做short，盈亏就是负的
        reverse_pnl = -df_original_long['最优盈亏%']
        print(f"\n原始标记为LONG，如果实际做SHORT:")
        print(f"  平均盈亏: {reverse_pnl.mean():.2f}%")
        print(f"  盈利数: {len(reverse_pnl[reverse_pnl > 0])}个 ({len(reverse_pnl[reverse_pnl > 0])/len(reverse_pnl)*100:.1f}%)")
        print(f"  亏损数: {len(reverse_pnl[reverse_pnl < 0])}个 ({len(reverse_pnl[reverse_pnl < 0])/len(reverse_pnl)*100:.1f}%)")
        print(f"  最大盈利: {reverse_pnl.max():.2f}%")
        print(f"  最大亏损: {reverse_pnl.min():.2f}%")

print("\n" + "=" * 100)
print("[完成] 真实对应关系分析完成！")
print("=" * 100)
