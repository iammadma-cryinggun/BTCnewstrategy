# -*- coding: utf-8 -*-
"""
找出所有信号类型的最优筛选条件（开仓+平仓）
==========================================

目标：
1. 分析所有4种信号类型
2. 使用Youden指数找最优阈值
3. 区分普通信号、开仓信号、平仓信号的筛选作用
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("所有信号类型的最优筛选条件分析")
print("=" * 100)

# 读取数据
df = pd.read_csv('complete_signal_lifecycle.csv', encoding='utf-8-sig')

print(f"\n总信号数: {len(df)}")
print(f"\n信号类型分布:")
print(df['信号类型'].value_counts())

# ==================== 分析函数 ====================
def find_optimal_thresholds(df_analysis, signal_type, stage):
    """
    使用Youden指数找最优阈值

    stage: '普通信号', '开仓', '平仓'
    """

    df_type = df_analysis.copy()

    # 按盈亏中位数分好机会/坏机会
    pnl_median = df_type['最优盈亏%'].median()
    df_type['是好机会'] = df_type['最优盈亏%'] >= pnl_median

    good = df_type[df_type['是好机会'] == True]
    bad = df_type[df_type['是好机会'] == False]

    if len(good) == 0 or len(bad) == 0:
        return

    print(f"\n{'=' * 100}")
    print(f"{signal_type} - {stage}阶段筛选分析")
    print(f"{'=' * 100}")
    print(f"好机会: {len(good)}个, 平均盈亏 {good['最优盈亏%'].mean():.2f}%")
    print(f"坏机会: {len(bad)}个, 平均盈亏 {bad['最优盈亏%'].mean():.2f}%")

    # 确定要分析的特征
    if stage == '普通信号':
        features = {
            '张力': '普通信号张力',
            '加速度': '普通信号加速度',
            '量能比率': '普通信号量能比率'
        }
    elif stage == '开仓':
        features = {
            '张力': '开仓张力',
            '加速度': '开仓加速度',
            '量能比率': '开仓量能比率'
        }
    else:  # 平仓
        features = {
            '张力': '平仓张力',
            '加速度': '平仓加速度',
            '量能比率': '平仓量能比率'
        }

    print(f"\n{'特征':<15} {'Cohen''s d':<12} {'p值':<12} {'最优阈值':<12} {'Youden':<12} {'筛选效果':<40}")
    print(f"{'-' * 100}")

    for name, col in features.items():
        if col not in df_type.columns:
            continue

        # 统计显著性
        good_mean = good[col].mean()
        bad_mean = bad[col].mean()
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')

        n1, n2 = len(good), len(bad)
        var1, var2 = good[col].var(), bad[col].var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        effect = '超大' if abs(cohens_d) > 1.2 else '大' if abs(cohens_d) > 0.8 else '中等' if abs(cohens_d) > 0.5 else '小'

        # Youden指数找最优阈值
        try:
            fpr, tpr, thresholds = roc_curve(df_type['是好机会'], df_type[col])
            youden = tpr - fpr
            optimal_idx = np.argmax(youden)
            optimal_threshold = thresholds[optimal_idx]
            optimal_youden = youden[optimal_idx]

            # 筛选效果
            above = df_type[df_type[col] >= optimal_threshold]
            below = df_type[df_type[col] < optimal_threshold]

            if len(above) > 0 and len(below) > 0:
                above_rate = len(above[above['是好机会']==True])/len(above)*100
                below_rate = len(below[below['是好机会']==True])/len(below)*100
                filter_effect = f"阈值以上{above_rate:.1f}% vs 阈值以下{below_rate:.1f}%"

                # 只有p<0.1或者Youden>0.1才显示
                if p_val < 0.1 or optimal_youden > 0.1:
                    print(f"{name:<15} {cohens_d:>8.3f} {sig:<4} {p_val:<12.4f} {optimal_threshold:<12.4f} {optimal_youden:<12.4f} {filter_effect:<40}")
        except:
            pass

# ==================== 分析所有信号类型 ====================
signal_types = ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']
stages = ['普通信号', '开仓', '平仓']

for signal_type in signal_types:
    df_type = df[df['信号类型'] == signal_type]

    if len(df_type) < 5:
        print(f"\n{signal_type}: 样本量太少({len(df_type)}个)，跳过")
        continue

    for stage in stages:
        find_optimal_thresholds(df_type, signal_type, stage)

print("\n" + "=" * 100)
print("[完成] 所有信号类型的最优筛选条件分析完成！")
print("=" * 100)
