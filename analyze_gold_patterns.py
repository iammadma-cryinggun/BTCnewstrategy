# -*- coding: utf-8 -*-
"""
黄金信号特征统计分析
==================

目标：
1. 对每个信号类型，分析好机会vs坏机会的特征差异
2. 使用统计学方法（t-test, Cohen's d）找出显著特征
3. 为V7.0.5优化提供数据支持
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("黄金信号特征统计分析")
print("=" * 100)

# 读取数据
df = pd.read_csv('v705_gold_signals_complete.csv', encoding='utf-8-sig')

print(f"\n总信号数: {len(df)}")
print(f"\n信号类型分布:")
print(df['信号类型'].value_counts())

# ==================== 关键发现1: 等待周期分布 ====================
print("\n" + "=" * 100)
print("发现1: 黄金开仓等待周期分布")
print("=" * 100)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df[df['信号类型'] == sig_type].copy()

    if len(df_type) == 0:
        continue

    print(f"\n{sig_type}:")
    print(f"  总数: {len(df_type)}个")
    print(f"  等待周期=0: {len(df_type[df_type['黄金开仓等待周期']==0])}个 ({len(df_type[df_type['黄金开仓等待周期']==0])/len(df_type)*100:.1f}%)")
    print(f"  等待周期1-3: {len(df_type[(df_type['黄金开仓等待周期']>=1) & (df_type['黄金开仓等待周期']<=3)])}个")
    print(f"  等待周期4-7: {len(df_type[(df_type['黄金开仓等待周期']>=4) & (df_type['黄金开仓等待周期']<=7)])}个")
    print(f"  等待周期8-10: {len(df_type[(df_type['黄金开仓等待周期']>=8) & (df_type['黄金开仓等待周期']<=10)])}个")
    print(f"  平均等待周期: {df_type['黄金开仓等待周期'].mean():.2f}")
    print(f"  平均价格优势: {df_type['价格优势%'].mean():.2f}%")

# ==================== 关键发现2: 持仓周期分布 ====================
print("\n" + "=" * 100)
print("发现2: 黄金平仓持仓周期分布")
print("=" * 100)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df[df['信号类型'] == sig_type].copy()

    if len(df_type) == 0:
        continue

    print(f"\n{sig_type}:")
    print(f"  平均持仓周期: {df_type['黄金平仓持仓周期'].mean():.2f}")
    print(f"  中位数持仓周期: {df_type['黄金平仓持仓周期'].median():.2f}")
    print(f"  最短持仓: {df_type['黄金平仓持仓周期'].min()}")
    print(f"  最长持仓: {df_type['黄金平仓持仓周期'].max()}")

# ==================== 关键发现3: 盈亏分布 ====================
print("\n" + "=" * 100)
print("发现3: 最优盈亏分布")
print("=" * 100)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df[df['信号类型'] == sig_type].copy()

    if len(df_type) == 0:
        continue

    print(f"\n{sig_type}:")
    print(f"  平均盈亏: {df_type['最优盈亏%'].mean():.2f}%")
    print(f"  中位数盈亏: {df_type['最优盈亏%'].median():.2f}%")
    print(f"  盈利交易: {len(df_type[df_type['最优盈亏%']>0])}个 ({len(df_type[df_type['最优盈亏%']>0])/len(df_type)*100:.1f}%)")
    print(f"  亏损交易: {len(df_type[df_type['最优盈亏%']<0])}个 ({len(df_type[df_type['最优盈亏%']<0])/len(df_type)*100:.1f}%)")
    print(f"  最大盈利: {df_type['最优盈亏%'].max():.2f}%")
    print(f"  最大亏损: {df_type['最优盈亏%'].min():.2f}%")

# ==================== 统计学分析: 好机会vs坏机会 ====================
print("\n" + "=" * 100)
print("统计学分析: 好机会 vs 坏机会的特征差异")
print("=" * 100)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df[df['信号类型'] == sig_type].copy()

    if len(df_type) < 5:
        continue

    # 按盈亏中位数分好机会/坏机会
    pnl_median = df_type['最优盈亏%'].median()
    df_type['是好机会'] = df_type['最优盈亏%'] >= pnl_median

    good = df_type[df_type['是好机会'] == True]
    bad = df_type[df_type['是好机会'] == False]

    print(f"\n{'=' * 100}")
    print(f"{sig_type} - 特征分析")
    print(f"{'=' * 100}")
    print(f"好机会: {len(good)}个, 平均盈亏 {good['最优盈亏%'].mean():.2f}%")
    print(f"坏机会: {len(bad)}个, 平均盈亏 {bad['最优盈亏%'].mean():.2f}%")

    # 分析各个阶段的特征
    stages = [
        ('信号前', '信号前5周期平均张力', '信号前5周期平均加速度', '信号前5周期平均量能'),
        ('信号时', '信号时刻张力', '信号时刻加速度', '信号时刻量能'),
        ('黄金开仓', '黄金开仓张力', '黄金开仓加速度', '黄金开仓量能'),
        ('黄金平仓', '黄金平仓张力', '黄金平仓加速度', '黄金平仓量能'),
    ]

    print(f"\n{'阶段':<10} {'特征':<20} {'好机会均值':<12} {'坏机会均值':<12} {'Cohen d':<10} {'p值':<10} {'显著性'}")
    print(f"{'-' * 100}")

    for stage_name, tension_col, accel_col, volume_col in stages:
        for col in [tension_col, accel_col, volume_col]:
            if col not in df_type.columns:
                continue

            good_mean = good[col].mean()
            bad_mean = bad[col].mean()
            t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')

            n1, n2 = len(good), len(bad)
            var1, var2 = good[col].var(), bad[col].var()
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            effect = '超大' if abs(cohens_d) > 1.2 else '大' if abs(cohens_d) > 0.8 else '中等' if abs(cohens_d) > 0.5 else '小'

            if p_val < 0.1 or abs(cohens_d) > 0.5:
                print(f"{stage_name:<10} {col:<20} {good_mean:<12.4f} {bad_mean:<12.4f} {cohens_d:<10.3f} {p_val:<10.4f} {sig} {effect}")

# ==================== 特征相关性分析 ====================
print("\n" + "=" * 100)
print("特征与盈亏的相关性分析")
print("=" * 100)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df[df['信号类型'] == sig_type].copy()

    if len(df_type) < 5:
        continue

    print(f"\n{sig_type}:")

    features = [
        '信号前5周期平均张力', '信号前5周期平均加速度', '信号前5周期平均量能',
        '信号时刻张力', '信号时刻加速度', '信号时刻量能',
        '黄金开仓张力', '黄金开仓加速度', '黄金开仓量能',
        '价格优势%', '黄金开仓等待周期'
    ]

    print(f"{'特征':<25} {'与盈亏相关系数':<15} {'说明'}")
    print(f"{'-' * 100}")

    for feat in features:
        if feat not in df_type.columns:
            continue

        corr = df_type[feat].corr(df_type['最优盈亏%'])

        if abs(corr) > 0.3:
            strength = '强' if abs(corr) > 0.6 else '中'
            direction = '正相关' if corr > 0 else '负相关'
            print(f"{feat:<25} {corr:<15.3f} {strength}{direction}")
        elif abs(corr) > 0.15:
            print(f"{feat:<25} {corr:<15.3f} 弱相关")

print("\n" + "=" * 100)
print("[完成] 黄金信号特征统计分析完成！")
print("=" * 100)
