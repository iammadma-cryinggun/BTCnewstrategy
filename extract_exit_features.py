# -*- coding: utf-8 -*-
"""
提取平仓时的市场特征（DXY燃料、张力、加速度、量能等）
=====================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("提取平仓时的市场特征")
print("=" * 80)

# ==================== 读取数据 ====================
df_signals = pd.read_csv('step1_entry_signals_with_dxy.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算量能比率
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

# 计算张力/加速度比
df_full['tension_accel_ratio'] = np.abs(df_full['tension'] / df_full['acceleration'].replace(0, np.nan))

print(f"\n信号数据: {len(df_signals)}个")
print(f"完整数据: {len(df_full)}条")

# ==================== 提取每个信号的平仓时特征 ====================
print("\n正在提取平仓时的市场特征...")

exit_features = []

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
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx_list) == 0:
        continue

    signal_idx = signal_idx_list[0]

    # 从step2_3_correct_golden_analysis.csv读取最优平仓周期
    # 简化：直接找未来30周期内的最优平仓点
    look_ahead_periods = 30

    if signal_idx + look_ahead_periods >= len(df_full):
        continue

    # 找最优平仓点
    best_pnl = -999
    best_exit_period = 0

    for period in range(1, look_ahead_periods + 1):
        future_idx = signal_idx + period
        future_price = df_full.loc[future_idx, 'close']

        if direction == 'short':
            pnl = (signal_price - future_price) / signal_price * 100
        else:
            pnl = (future_price - signal_price) / signal_price * 100

        if pnl > best_pnl:
            best_pnl = pnl
            best_exit_period = period

    # 记录平仓时的市场特征
    exit_idx = signal_idx + best_exit_period

    exit_features.append({
        '信号时间': signal_time,
        '信号类型': signal_type,
        '方向': direction,
        '开仓张力': signal_tension,
        '开仓加速度': signal_accel,

        # 平仓时的特征
        '平仓周期': best_exit_period,
        '平仓盈亏%': best_pnl,

        '平仓时张力': df_full.loc[exit_idx, 'tension'],
        '平仓时加速度': df_full.loc[exit_idx, 'acceleration'],
        '平仓时量能比率': df_full.loc[exit_idx, 'volume_ratio'],
        '平仓时DXY燃料': df_full.loc[exit_idx, 'dxy_fuel'],
        '平仓时张力加速度比': df_full.loc[exit_idx, 'tension_accel_ratio'],

        # 特征变化
        '张力变化': df_full.loc[exit_idx, 'tension'] - signal_tension,
        '张力变化%': (df_full.loc[exit_idx, 'tension'] - signal_tension) / abs(signal_tension) * 100 if signal_tension != 0 else 0,
        '加速度变化': df_full.loc[exit_idx, 'acceleration'] - signal_accel,
    })

df_exit = pd.DataFrame(exit_features)

print(f"[OK] 提取完成: {len(df_exit)}个信号")

# ==================== 统计学分析 ====================
print("\n" + "=" * 80)
print("平仓时市场特征的统计学分析")
print("=" * 80)

def analyze_features(df_dir, direction_name):
    """分析某个方向的特征"""
    print(f"\n{'=' * 80}")
    print(f"{direction_name}信号分析")
    print(f"{'=' * 80}")

    # 按盈亏中位数分组
    pnl_median = df_dir['平仓盈亏%'].median()
    df_dir['大盈'] = df_dir['平仓盈亏%'] >= pnl_median

    good = df_dir[df_dir['大盈'] == True]
    bad = df_dir[df_dir['大盈'] == False]

    print(f"\n大盈组 (n={len(good)}): 平均盈亏 {good['平仓盈亏%'].mean():.2f}%")
    print(f"小盈组 (n={len(bad)}): 平均盈亏 {bad['平仓盈亏%'].mean():.2f}%")

    # 分析各个特征
    features = [
        ('平仓周期', '平仓周期'),
        ('平仓时张力', '平仓时张力'),
        ('平仓时加速度', '平仓时加速度'),
        ('平仓时量能比率', '平仓时量能比率'),
        ('平仓时DXY燃料', '平仓时DXY燃料'),
        ('平仓时张力加速度比', '平仓时张力加速度比'),
        ('张力变化%', '张力变化%'),
    ]

    print(f"\n特征对比（大盈 vs 小盈）:")
    print(f"{'特征':<20} {'大盈均值':<15} {'小盈均值':<15} {'Cohen\\'s d':<12} {'p值':<10} {'显著性'}")
    print("-" * 100)

    for feat_col, feat_name in features:
        good_vals = good[feat_col]
        bad_vals = bad[feat_col]

        # Cohen's d
        n1, n2 = len(good_vals), len(bad_vals)
        var1, var2 = good_vals.var(), bad_vals.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good_vals.mean() - bad_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # t检验
        t_stat, p_value = stats.ttest_ind(good_vals, bad_vals, equal_var=False, nan_policy='omit')

        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''

        print(f"{feat_name:<20} {good_vals.mean():<15.4f} {bad_vals.mean():<15.4f} {cohens_d:<12.4f} {p_value:<10.4f} {significance}")

    # 找出最优阈值
    print(f"\n最优阈值分析（找最能区分大盈小盈的阈值）:")
    print(f"{'特征':<20} {'最优阈值':<15} {'大盈组均值':<15} {'小盈组均值':<15}")
    print("-" * 80)

    for feat_col, feat_name in features:
        if feat_col == '平仓周期':
            continue  # 跳过周期，因为用户说不是关键

        good_vals = good[feat_col]
        bad_vals = bad[feat_col]

        # 简单方法：用大盈组和小盈组的中位数作为阈值
        threshold = (good_vals.median() + bad_vals.median()) / 2

        print(f"{feat_name:<20} {threshold:<15.4f} {good_vals.mean():<15.4f} {bad_vals.mean():<15.4f}")

# 分析SHORT和LONG
df_short = df_exit[df_exit['方向'] == 'short'].copy()
analyze_features(df_short, "SHORT")

df_long = df_exit[df_exit['方向'] == 'long'].copy()
analyze_features(df_long, "LONG")

print("\n" + "=" * 80)
print("[OK] 分析完成")
print("=" * 80)

# 保存数据
df_exit.to_csv('exit_features_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: exit_features_analysis.csv")
