# -*- coding: utf-8 -*-
"""
第二步+第三步：分析开仓信号的最优时机并重新总结黄金信号

- 分析每个开仓信号后的最优开仓时机（首次 vs 等待）
- 分析最优平仓时机
- 用数学统计视角重新总结黄金信号逻辑
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("第二步+第三步：分析最优时机并重新总结黄金信号")
print("=" * 80)
print()

# ==================== 读取数据 ====================
print("读取数据...")

df_signals = pd.read_csv('step1_entry_signals.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"[OK] 开仓信号: {len(df_signals)}个")
print(f"[OK] 完整数据: {len(df_full)}条")
print()

# ==================== 分析每个开仓信号的后续走势 ====================
print("正在分析每个信号的后续走势...")

analysis_results = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    signal_tension = signal['张力']

    # 确定交易方向（验证5反向策略）
    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'  # 系统看涨/高位 → 我们反向做空
    else:  # BEARISH_SINGULARITY, LOW_OSCILLATION
        direction = 'long'   # 系统看空/低位 → 我们反向做多

    # 在完整数据中找到这个信号的位置
    signal_idx = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx) == 0:
        continue

    signal_idx = signal_idx[0]

    # 分析后续30个周期（5天）
    look_ahead_periods = 30

    if signal_idx + look_ahead_periods >= len(df_full):
        continue

    # 记录后续每个周期的表现
    future_performance = []

    for period in range(1, look_ahead_periods + 1):
        future_idx = signal_idx + period
        future_price = df_full.loc[future_idx, 'close']

        # 计算盈亏
        if direction == 'short':
            pnl = (signal_price - future_price) / signal_price * 100
        else:  # long
            pnl = (future_price - signal_price) / signal_price * 100

        future_performance.append({
            'period': period,
            'price': future_price,
            'pnl': pnl
        })

    # 找最优平仓点（最大盈亏）
    max_pnl = max(p['pnl'] for p in future_performance)
    best_exit_period = next(p['period'] for p in future_performance if p['pnl'] == max_pnl)

    # 找首次开仓的盈亏（第1周期）
    first_pnl = future_performance[0]['pnl']

    # 找最差平仓点
    min_pnl = min(p['pnl'] for p in future_performance)

    # 分析：哪些周期是"好机会"（盈亏>3%）
    good_periods = [p for p in future_performance if p['pnl'] > 3.0]
    is_good_opportunity = len(good_periods) > 0

    # 张力变化（信号时刻到最优平仓）
    best_exit_tension = df_full.loc[signal_idx + best_exit_period, 'tension']
    tension_change = (best_exit_tension - signal_tension) / abs(signal_tension) * 100 if signal_tension != 0 else 0

    # 价格优势（首次开仓价格 vs 最低价）
    if direction == 'short':
        # 做空：后续最高价越低越好
        future_prices = [df_full.loc[signal_idx + p, 'close'] for p in range(1, look_ahead_periods + 1)]
        price_advantage = (signal_price - min(future_prices)) / signal_price * 100
    else:  # long
        # 做多：后续最低价越高越好
        future_prices = [df_full.loc[signal_idx + p, 'close'] for p in range(1, look_ahead_periods + 1)]
        price_advantage = (max(future_prices) - signal_price) / signal_price * 100

    analysis_results.append({
        '信号时间': signal_time,
        '信号类型': signal_type,
        '交易方向': direction,
        '开仓价': signal_price,
        '开仓张力': signal_tension,

        '首次开仓盈亏': first_pnl,
        '最优平仓盈亏': max_pnl,
        '最优平仓周期': best_exit_period,
        '最差平仓盈亏': min_pnl,
        '盈亏波动': max_pnl - min_pnl,

        '是好机会': is_good_opportunity,
        '好机会数': len(good_periods),
        '好机会平均盈亏': np.mean([p['pnl'] for p in good_periods]) if good_periods else 0,

        '最优平仓张力': best_exit_tension,
        '张力变化%': tension_change,
        '价格优势': price_advantage
    })

print(f"[OK] 分析完成: {len(analysis_results)}个信号")
print()

# ==================== 统计分析 ====================
df_analysis = pd.DataFrame(analysis_results)

print("=" * 80)
print("统计分析")
print("=" * 80)
print()

# 1. 分别统计SHORT和LONG
for direction in ['short', 'long']:
    direction_cn = '做空SHORT' if direction == 'short' else '做多LONG'
    df_dir = df_analysis[df_analysis['交易方向'] == direction]

    print(f"\n{direction_cn}:")
    print("-" * 40)

    if len(df_dir) == 0:
        print("  无数据")
        continue

    print(f"样本数: {len(df_dir)}个")

    # 首次开仓 vs 最优开仓
    first_avg = df_dir['首次开仓盈亏'].mean()
    best_avg = df_dir['最优平仓盈亏'].mean()
    improvement = best_avg - first_avg

    print(f"\n首次开仓（立即开）:")
    print(f"  平均盈亏: {first_avg:+.2f}%")
    print(f"  胜率: {(df_dir['首次开仓盈亏'] > 0).sum() / len(df_dir) * 100:.1f}%")

    print(f"\n最优开仓（等待最佳时机）:")
    print(f"  平均盈亏: {best_avg:+.2f}%")
    print(f"  胜率: {(df_dir['最优平仓盈亏'] > 0).sum() / len(df_dir) * 100:.1f}%")
    print(f"  平均最优周期: {df_dir['最优平仓周期'].mean():.1f}周期")

    print(f"\n改善: {improvement:+.2f}%")

    # 好机会统计
    good_count = df_dir['是好机会'].sum()
    print(f"\n好机会（盈亏>3%）:")
    print(f"  数量: {good_count}个 ({good_count/len(df_dir)*100:.1f}%)")
    if good_count > 0:
        print(f"  平均盈亏: {df_dir[df_dir['是好机会']]['最优平仓盈亏'].mean():+.2f}%")

# 2. 黄金开仓条件分析（Youden Index）
print("\n" + "=" * 80)
print("黄金开仓条件分析（寻找最优阈值）")
print("=" * 80)
print()

# 分别分析SHORT和LONG的最优开仓条件
for direction in ['short', 'long']:
    direction_cn = '做空SHORT' if direction == 'short' else '做多LONG'
    df_dir = df_analysis[df_analysis['交易方向'] == direction]

    if len(df_dir) == 0:
        continue

    print(f"\n{direction_cn} 黄金开仓条件:")
    print("-" * 40)

    # 标记是否为好机会
    df_dir['是好机会_binary'] = df_dir['是好机会'].astype(int)

    # 条件1：张力变化
    print("\n条件1: 张力变化（开仓→最优平仓）")
    for threshold in [2, 3, 4, 5, 6, 7, 8]:
        df_dir[f'张力变化_{threshold}'] = (df_dir['张力变化%'].abs() >= threshold).astype(int)

        # 计算Youden Index
        TP = ((df_dir[f'张力变化_{threshold}'] == 1) & (df_dir['是好机会_binary'] == 1)).sum()
        FP = ((df_dir[f'张力变化_{threshold}'] == 1) & (df_dir['是好机会_binary'] == 0)).sum()
        TN = ((df_dir[f'张力变化_{threshold}'] == 0) & (df_dir['是好机会_binary'] == 0)).sum()
        FN = ((df_dir[f'张力变化_{threshold}'] == 0) & (df_dir['是好机会_binary'] == 1)).sum()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        youden = sensitivity + specificity - 1

        print(f"  阈值≥{threshold}%: 敏感度={sensitivity:.3f}, 特异度={specificity:.3f}, Youden={youden:.3f}, 样本={df_dir[f'张力变化_{threshold}'].sum()}个")

    # 条件2：价格优势
    print("\n条件2: 价格优势")
    for threshold in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        df_dir[f'价格优势_{threshold}'] = (df_dir['价格优势'] >= threshold).astype(int)

        TP = ((df_dir[f'价格优势_{threshold}'] == 1) & (df_dir['是好机会_binary'] == 1)).sum()
        FP = ((df_dir[f'价格优势_{threshold}'] == 1) & (df_dir['是好机会_binary'] == 0)).sum()
        TN = ((df_dir[f'价格优势_{threshold}'] == 0) & (df_dir['是好机会_binary'] == 0)).sum()
        FN = ((df_dir[f'价格优势_{threshold}'] == 0) & (df_dir['是好机会_binary'] == 1)).sum()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        youden = sensitivity + specificity - 1

        print(f"  阈值≥{threshold}%: 敏感度={sensitivity:.3f}, 特异度={specificity:.3f}, Youden={youden:.3f}, 样本={df_dir[f'价格优势_{threshold}'].sum()}个")

# 3. 黄金平仓条件分析
print("\n" + "=" * 80)
print("黄金平仓条件分析")
print("=" * 80)
print()

# 分析好机会的平仓特征
df_good = df_analysis[df_analysis['是好机会'] == True]

for direction in ['short', 'long']:
    direction_cn = '做空SHORT' if direction == 'short' else '做多LONG'
    df_dir = df_good[df_good['交易方向'] == direction]

    if len(df_dir) == 0:
        continue

    print(f"\n{direction_cn} 好机会的平仓特征:")
    print("-" * 40)

    print(f"样本数: {len(df_dir)}个")

    # 最优平仓周期
    print(f"\n最优平仓周期:")
    print(f"  平均: {df_dir['最优平仓周期'].mean():.1f}周期")
    print(f"  中位数: {df_dir['最优平仓周期'].median():.0f}周期")

    period_dist = df_dir['最优平仓周期'].value_counts().sort_index()
    for period, count in period_dist.head(10).items():
        print(f"    第{period}周期: {count}个 ({count/len(df_dir)*100:.1f}%)")

    # 张力变化
    print(f"\n张力变化特征:")
    print(f"  平均: {df_dir['张力变化%'].mean():+.2f}%")
    print(f"  中位数: {df_dir['张力变化%'].median():+.2f}%")
    print(f"  最小: {df_dir['张力变化%'].min():+.2f}%")
    print(f"  最大: {df_dir['张力变化%'].max():+.2f}%")

    # 盈亏分布
    print(f"\n最优平仓盈亏:")
    print(f"  平均: {df_dir['最优平仓盈亏'].mean():+.2f}%")
    print(f"  中位数: {df_dir['最优平仓盈亏'].median():+.2f}%")

# ==================== 保存结果 ====================
print("\n" + "=" * 80)
print("保存结果...")
print()

df_analysis.to_csv('step2_3_analysis_results.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 分析结果已保存到: step2_3_analysis_results.csv")

print("\n" + "=" * 80)
print("第二步+第三步完成！")
print("=" * 80)
print()
print("下一步：根据Youden Index最优阈值，重新定义黄金信号逻辑")
