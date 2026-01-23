# -*- coding: utf-8 -*-
"""
黄金平仓信号统计学分析 - 含DXY燃料
=====================================

基于数据规律分析最优平仓条件：
- 平仓时的张力特征
- 平仓时的量能特征
- 平仓时的加速度特征
- 平仓周期分布

统计方法：p值检验、Cohen's d、Youden指数
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("黄金平仓信号统计学分析 - 含DXY燃料")
print("=" * 80)

# ==================== 读取数据 ====================
print("\n读取数据...")

df_signals = pd.read_csv('step1_entry_signals_with_dxy.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"[OK] 开仓信号: {len(df_signals)}个")
print(f"[OK] 完整数据: {len(df_full)}条")

# ==================== 分析每个信号的最优平仓点特征 ====================
print("\n正在分析最优平仓点的市场特征...")

exit_analysis = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    signal_tension = signal['张力']
    signal_accel = signal['加速度']

    # 确定交易方向
    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'
    else:
        direction = 'long'

    # 在完整数据中找到这个信号的位置
    signal_idx = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx) == 0:
        continue

    signal_idx = signal_idx[0]

    # 分析后续30个周期
    look_ahead_periods = 30

    if signal_idx + look_ahead_periods >= len(df_full):
        continue

    # 找到最优平仓点（最大盈利）
    best_pnl = -999
    best_period = 0

    for period in range(1, look_ahead_periods + 1):
        future_idx = signal_idx + period
        future_price = df_full.loc[future_idx, 'close']

        if direction == 'short':
            pnl = (signal_price - future_price) / signal_price * 100
        else:
            pnl = (future_price - signal_price) / signal_price * 100

        if pnl > best_pnl:
            best_pnl = pnl
            best_period = period

    # 记录最优平仓点的市场特征
    best_exit_idx = signal_idx + best_period

    exit_data = {
        '信号时间': signal_time,
        '信号类型': signal_type,
        '方向': direction,
        '开仓价': signal_price,
        '开仓张力': signal_tension,
        '开仓加速度': signal_accel,

        '最优平仓周期': best_period,
        '最优平仓盈亏%': best_pnl,

        # 平仓时的市场特征
        '平仓张力': df_full.loc[best_exit_idx, 'tension'],
        '平仓加速度': df_full.loc[best_exit_idx, 'acceleration'],
        '平仓量能比率': 0.0,  # 需要计算
        '平仓DXY燃料': df_full.loc[best_exit_idx, 'dxy_fuel'],

        # 特征变化
        '张力变化': df_full.loc[best_exit_idx, 'tension'] - signal_tension,
        '张力变化%': (df_full.loc[best_exit_idx, 'tension'] - signal_tension) / abs(signal_tension) * 100 if signal_tension != 0 else 0,
    }

    exit_analysis.append(exit_data)

df_exit = pd.DataFrame(exit_analysis)

print(f"[OK] 分析完成: {len(df_exit)}个信号")
print()

# ==================== 计算量能比率 ====================
print("正在计算量能比率...")

df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

# 为每个平仓点填充量能比率
for idx, row in df_exit.iterrows():
    signal_time = row['信号时间']
    best_period = int(row['最优平仓周期'])

    # 找到平仓时间
    signal_idx = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx) > 0:
        exit_idx = signal_idx[0] + best_period
        if exit_idx < len(df_full):
            df_exit.at[idx, '平仓量能比率'] = df_full.loc[exit_idx, 'volume_ratio']

print(f"[OK] 量能比率计算完成")
print()

# ==================== 统计学分析 ====================
print("=" * 80)
print("黄金平仓信号统计学分析")
print("=" * 80)

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

# ==================== 按方向分析 ====================
for direction in ['short', 'long']:
    df_dir = df_exit[df_exit['方向'] == direction].copy()

    if len(df_dir) == 0:
        continue

    print(f"\n{'=' * 80}")
    print(f"{direction.upper()}信号平仓分析")
    print(f"{'=' * 80}")

    # 定义大盈和小盈
    pnl_threshold = df_dir['最优平仓盈亏%'].median()
    df_dir['大盈'] = df_dir['最优平仓盈亏%'] >= pnl_threshold

    big_win = df_dir[df_dir['大盈'] == True]
    small_win = df_dir[df_dir['大盈'] == False]

    print(f"\n大盈（>={pnl_threshold:.2f}%）: {len(big_win)}个 ({len(big_win)/len(df_dir)*100:.1f}%)")
    print(f"小盈（<{pnl_threshold:.2f}%）: {len(small_win)}个 ({len(small_win)/len(df_dir)*100:.1f}%)")

    # 特征统计
    print(f"\n特征统计：")

    feature_cols = ['最优平仓周期', '平仓张力', '平仓加速度', '平仓量能比率', '张力变化%']
    feature_names = ['平仓周期', '平仓张力', '平仓加速度', '平仓量能比率', '张力变化%']

    for feat_col, feat_name in zip(feature_cols, feature_names):
        big_vals = df_dir[df_dir['大盈'] == True][feat_col]
        small_vals = df_dir[df_dir['大盈'] == False][feat_col]

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

    # 平仓周期分布
    print(f"\n平仓周期分布：")
    period_dist = df_dir['最优平仓周期'].value_counts().sort_index()
    for period, count in period_dist.head(10).items():
        print(f"  第{period}周期: {count}个 ({count/len(df_dir)*100:.1f}%)")

print("\n" + "=" * 80)
print("[OK] 黄金平仓信号分析完成")
print("=" * 80)
