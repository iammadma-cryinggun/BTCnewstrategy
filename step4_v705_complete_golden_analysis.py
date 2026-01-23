# -*- coding: utf-8 -*-
"""
V7.0.5黄金信号完整统计学分析 - 专业数学家方法
==========================================

基于完整交易数据的统计学分析：
- p值显著性检验（t检验，p<0.05）
- Cohen's d效应量（>0.5中等，>0.8大，>1.2超大）
- Youden Index（最优判别阈值，>0.8优秀）
- 95%置信区间

数据：step2_3_analysis_results.csv（2025-08-05至2026-01-20）
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V7.0.5黄金信号完整统计学分析 - 专业数学家方法")
print("=" * 80)

# ==================== 读取数据 ====================
df = pd.read_csv('step2_3_analysis_results.csv', encoding='utf-8')

# 重命名列
df.columns = ['信号时间', '信号类型', '方向', '开仓价', '张力', '首次开仓盈亏%',
              '最终平仓盈亏%', '最终平仓周期', '黄金平仓盈亏%', '盈亏类型',
              '是好机会', '好机会类型', '好机会盈亏%', '好机会平仓周期', '张力变化%', '价格优势%']

# 转换数据类型
df['信号时间'] = pd.to_datetime(df['信号时间'])
for col in ['张力', '首次开仓盈亏%', '最终平仓盈亏%', '黄金平仓盈亏%', '好机会盈亏%', '张力变化%', '价格优势%']:
    df[col] = df[col].astype(float)

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

def find_youden_threshold(feature, good_opportunities, all_data):
    """使用Youden Index找最优阈值"""
    fpr, tpr, thresholds = roc_curve(all_data['是好机会'], feature)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_index[optimal_idx]
    return optimal_threshold, optimal_youden

# ==================== 分析SHORT信号 ====================
print("\n" + "=" * 80)
print("SHORT信号统计学分析")
print("=" * 80)

if len(df_short) > 0:
    df_short['是好机会_binary'] = df_short['好机会盈亏%'] > 0
    good_short = df_short[df_short['是好机会_binary'] == True]
    bad_short = df_short[df_short['是好机会_binary'] == False]

    print(f"\n好机会数：{len(good_short)}个 ({len(good_short)/len(df_short)*100:.1f}%)")
    print(f"坏机会数：{len(bad_short)}个 ({len(bad_short)/len(df_short)*100:.1f}%)")

    # 特征统计
    features = {
        '张力': df_short['张力'],
        '张力变化%': df_short['张力变化%'],
        '价格优势%': df_short['价格优势%']
    }

    print(f"\n特征统计：")
    for feat_name, feat_data in features.items():
        good_vals = df_short[df_short['是好机会_binary'] == True][feat_name]
        bad_vals = df_short[df_short['是好机会_binary'] == False][feat_name]

        if len(good_vals) > 0 and len(bad_vals) > 0:
            t_stat, p_value = t_test(good_vals, bad_vals)
            cohens_d = calculate_cohens_d(good_vals, bad_vals)
            ci_good = calculate_ci(good_vals)
            ci_bad = calculate_ci(bad_vals)

            print(f"\n{feat_name}:")
            print(f"  好机会: {good_vals.mean():.4f} [{ci_good[0]:.4f}, {ci_good[1]:.4f}]")
            print(f"  坏机会: {bad_vals.mean():.4f} [{ci_bad[0]:.4f}, {ci_bad[1]:.4f}]")
            print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.4f}",
                  f"{'***显著***' if p_value < 0.05 else '不显著'}")
            print(f"  Cohen's d: {cohens_d:.4f}",
                  f"{'超大效应' if abs(cohens_d) > 1.2 else '大效应' if abs(cohens_d) > 0.8 else '中等效应' if abs(cohens_d) > 0.5 else '小效应'}")

            # Youden Index
            try:
                threshold, youden = find_youden_threshold(df_short[feat_name], good_short, df_short)
                print(f"  Youden阈值: {threshold:.4f}, Youden指数: {youden:.4f}",
                      f"{'优秀' if youden > 0.8 else '良好' if youden > 0.5 else '一般'}")
            except:
                pass

# ==================== 分析LONG信号 ====================
print("\n" + "=" * 80)
print("LONG信号统计学分析")
print("=" * 80)

if len(df_long) > 0:
    df_long['是好机会_binary'] = df_long['好机会盈亏%'] > 0
    good_long = df_long[df_long['是好机会_binary'] == True]
    bad_long = df_long[df_long['是好机会_binary'] == False]

    print(f"\n好机会数：{len(good_long)}个 ({len(good_long)/len(df_long)*100:.1f}%)")
    print(f"坏机会数：{len(bad_long)}个 ({len(bad_long)/len(df_long)*100:.1f}%)")

    # 特征统计
    features = {
        '张力': df_long['张力'],
        '张力变化%': df_long['张力变化%'],
        '价格优势%': df_long['价格优势%']
    }

    print(f"\n特征统计：")
    for feat_name, feat_data in features.items():
        good_vals = df_long[df_long['是好机会_binary'] == True][feat_name]
        bad_vals = df_long[df_long['是好机会_binary'] == False][feat_name]

        if len(good_vals) > 0 and len(bad_vals) > 0:
            t_stat, p_value = t_test(good_vals, bad_vals)
            cohens_d = calculate_cohens_d(good_vals, bad_vals)
            ci_good = calculate_ci(good_vals)
            ci_bad = calculate_ci(bad_vals)

            print(f"\n{feat_name}:")
            print(f"  好机会: {good_vals.mean():.4f} [{ci_good[0]:.4f}, {ci_good[1]:.4f}]")
            print(f"  坏机会: {bad_vals.mean():.4f} [{ci_bad[0]:.4f}, {ci_bad[1]:.4f}]")
            print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.4f}",
                  f"{'***显著***' if p_value < 0.05 else '不显著'}")
            print(f"  Cohen's d: {cohens_d:.4f}",
                  f"{'超大效应' if abs(cohens_d) > 1.2 else '大效应' if abs(cohens_d) > 0.8 else '中等效应' if abs(cohens_d) > 0.5 else '小效应'}")

            # Youden Index
            try:
                threshold, youden = find_youden_threshold(df_long[feat_name], good_long, df_long)
                print(f"  Youden阈值: {threshold:.4f}, Youden指数: {youden:.4f}",
                      f"{'优秀' if youden > 0.8 else '良好' if youden > 0.5 else '一般'}")
            except:
                pass

print("\n" + "=" * 80)
print("[OK] V7.0.5黄金信号统计学分析完成")
print("=" * 80)
