# -*- coding: utf-8 -*-
"""
黄金信号完整统计学分析 - 正确方法（逐周期检查）
================================================

基于正确的黄金信号定义进行统计学分析
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("黄金信号完整统计学分析 - 正确方法")
print("=" * 80)

# ==================== 读取数据 ====================
df = pd.read_csv('step2_3_correct_golden_analysis.csv', encoding='utf-8')

print(f"\n数据范围：{df['信号时间'].min()} 至 {df['信号时间'].max()}")
print(f"总信号数：{len(df)}个")

# ==================== 分SHORT和LONG ====================
df_short = df[df['方向'] == 'short'].copy()
df_long = df[df['方向'] == 'long'].copy()

print(f"\nSHORT信号：{len(df_short)}个")
print(f"LONG信号：{len(df_long)}个")

# ==================== 定义辅助函数 ====================
def calculate_cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

def calculate_ci(data, confidence=0.95):
    """计算95%置信区间"""
    n = len(data)
    mean = data.mean()
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    return (mean - h, mean + h)

def t_test(group_good, group_bad):
    """独立样本t检验"""
    t_stat, p_value = stats.ttest_ind(group_good, group_bad, equal_var=False)
    return t_stat, p_value

# ==================== 分析SHORT信号 ====================
print("\n" + "=" * 80)
print("SHORT信号统计学分析")
print("=" * 80)

if len(df_short) > 0:
    # 使用盈亏中位数区分大盈和小盈
    pnl_median = df_short['最优平仓盈亏%'].median()
    df_short['大盈'] = df_short['最优平仓盈亏%'] >= pnl_median

    good_short = df_short[df_short['大盈'] == True]
    bad_short = df_short[df_short['大盈'] == False]

    print(f"\n大盈（>={pnl_median:.2f}%）: {len(good_short)}个 ({len(good_short)/len(df_short)*100:.1f}%)")
    print(f"小盈（<{pnl_median:.2f}%）: {len(bad_short)}个 ({len(bad_short)/len(df_short)*100:.1f}%)")

    # 特征统计
    print(f"\n特征统计：")

    feature_cols = ['最优开仓周期', '价格优势%', '最优平仓周期', '最优平仓盈亏%']
    feature_names = ['最优开仓周期', '价格优势%', '最优平仓周期', '最优平仓盈亏%']

    for feat_col, feat_name in zip(feature_cols, feature_names):
        big_vals = df_short[df_short['大盈'] == True][feat_col]
        small_vals = df_short[df_short['大盈'] == False][feat_col]

        if len(big_vals) > 0 and len(small_vals) > 0:
            t_stat, p_value = t_test(big_vals, small_vals)
            cohens_d = calculate_cohens_d(big_vals, small_vals)
            ci_big = calculate_ci(big_vals)
            ci_small = calculate_ci(small_vals)

            print(f"\n{feat_name}:")
            print(f"  大盈: {big_vals.mean():.4f} [{ci_big[0]:.4f}, {ci_big[1]:.4f}]")
            print(f"  小盈: {small_vals.mean():.4f} [{ci_small[0]:.4f}, {ci_small[1]:.4f}]")
            print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.4f}",
                  f"{'***显著***' if p_value < 0.05 else '不显著'}")
            print(f"  Cohen's d: {cohens_d:.4f}",
                  f"{'超大效应' if abs(cohens_d) > 1.2 else '大效应' if abs(cohens_d) > 0.8 else '中等效应' if abs(cohens_d) > 0.5 else '小效应'}")

    # 价格优势分布
    print(f"\n价格优势分布：")
    print(f"  等待0周期（立即开仓）: {sum(df_short['最优开仓周期'] == 0)}个 ({sum(df_short['最优开仓周期'] == 0)/len(df_short)*100:.1f}%)")
    print(f"  等待1-3周期: {sum((df_short['最优开仓周期'] >= 1) & (df_short['最优开仓周期'] <= 3))}个 ({sum((df_short['最优开仓周期'] >= 1) & (df_short['最优开仓周期'] <= 3))/len(df_short)*100:.1f}%)")
    print(f"  等待4-6周期: {sum((df_short['最优开仓周期'] >= 4) & (df_short['最优开仓周期'] <= 6))}个 ({sum((df_short['最优开仓周期'] >= 4) & (df_short['最优开仓周期'] <= 6))/len(df_short)*100:.1f}%)")
    print(f"  等待7+周期: {sum(df_short['最优开仓周期'] >= 7)}个 ({sum(df_short['最优开仓周期'] >= 7)/len(df_short)*100:.1f}%)")

# ==================== 分析LONG信号 ====================
print("\n" + "=" * 80)
print("LONG信号统计学分析")
print("=" * 80)

if len(df_long) > 0:
    # 使用盈亏中位数区分大盈和小盈
    pnl_median = df_long['最优平仓盈亏%'].median()
    df_long['大盈'] = df_long['最优平仓盈亏%'] >= pnl_median

    good_long = df_long[df_long['大盈'] == True]
    bad_long = df_long[df_long['大盈'] == False]

    print(f"\n大盈（>={pnl_median:.2f}%）: {len(good_long)}个 ({len(good_long)/len(df_long)*100:.1f}%)")
    print(f"小盈（<{pnl_median:.2f}%）: {len(bad_long)}个 ({len(bad_long)/len(df_long)*100:.1f}%)")

    # 特征统计
    print(f"\n特征统计：")

    feature_cols = ['最优开仓周期', '价格优势%', '最优平仓周期', '最优平仓盈亏%']
    feature_names = ['最优开仓周期', '价格优势%', '最优平仓周期', '最优平仓盈亏%']

    for feat_col, feat_name in zip(feature_cols, feature_names):
        big_vals = df_long[df_long['大盈'] == True][feat_col]
        small_vals = df_long[df_long['大盈'] == False][feat_col]

        if len(big_vals) > 0 and len(small_vals) > 0:
            t_stat, p_value = t_test(big_vals, small_vals)
            cohens_d = calculate_cohens_d(big_vals, small_vals)
            ci_big = calculate_ci(big_vals)
            ci_small = calculate_ci(small_vals)

            print(f"\n{feat_name}:")
            print(f"  大盈: {big_vals.mean():.4f} [{ci_big[0]:.4f}, {ci_big[1]:.4f}]")
            print(f"  小盈: {small_vals.mean():.4f} [{ci_small[0]:.4f}, {ci_small[1]:.4f}]")
            print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.4f}",
                  f"{'***显著***' if p_value < 0.05 else '不显著'}")
            print(f"  Cohen's d: {cohens_d:.4f}",
                  f"{'超大效应' if abs(cohens_d) > 1.2 else '大效应' if abs(cohens_d) > 0.8 else '中等效应' if abs(cohens_d) > 0.5 else '小效应'}")

    # 价格优势分布
    print(f"\n价格优势分布：")
    print(f"  等待0周期（立即开仓）: {sum(df_long['最优开仓周期'] == 0)}个 ({sum(df_long['最优开仓周期'] == 0)/len(df_long)*100:.1f}%)")
    print(f"  等待1-3周期: {sum((df_long['最优开仓周期'] >= 1) & (df_long['最优开仓周期'] <= 3))}个 ({sum((df_long['最优开仓周期'] >= 1) & (df_long['最优开仓周期'] <= 3))/len(df_long)*100:.1f}%)")
    print(f"  等待4-6周期: {sum((df_long['最优开仓周期'] >= 4) & (df_long['最优开仓周期'] <= 6))}个 ({sum((df_long['最优开仓周期'] >= 4) & (df_long['最优开仓周期'] <= 6))/len(df_long)*100:.1f}%)")
    print(f"  等待7+周期: {sum(df_long['最优开仓周期'] >= 7)}个 ({sum(df_long['最优开仓周期'] >= 7)/len(df_long)*100:.1f}%)")

print("\n" + "=" * 80)
print("[OK] 统计学分析完成")
print("=" * 80)
