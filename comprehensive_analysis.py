# -*- coding: utf-8 -*-
"""
综合统计分析：最佳入场时机与错误信号识别
========================================
"""

import pandas as pd
import numpy as np
from scipy import stats

print("="*140)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("Optimal Entry/Exit Timing & False Signal Detection")
print("="*140)

# ============================================================================
# Step 1: 加载数据
# ============================================================================
print("\n" + "="*140)
print("STEP 1: Load Data")
print("="*140)

df = pd.read_csv('最终数据_完整合并.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

print(f"\n数据集: {len(df)} 条信号")
print(f"ACTION: {len(df[df['黄金信号']=='ACTION'])} 条")
print(f"HOLD: {len(df[df['黄金信号']=='HOLD'])} 条")

# ============================================================================
# Step 2: 分析ACTION时的特征
# ============================================================================
print("\n" + "="*140)
print("STEP 2: ACTION Signal Characteristics")
print("="*140)

action_df = df[df['黄金信号'] == 'ACTION'].copy()
hold_df = df[df['黄金信号'] == 'HOLD'].copy()

print("\n【ACTION类型分布】")
action_by_type = action_df['最优动作'].value_counts()
for action, count in action_by_type.items():
    pct = count / len(action_df) * 100
    print(f"  {action}: {count} ({pct:.1f}%)")

print("\n【ACTION与HOLD的参数对比】")
print(f"{'参数':<15} {'ACTION均值':<12} {'HOLD均值':<12} {'差异':<12} {'Cohen d':<12} {'统计显著性':<15}")
print("-" * 140)

params = ['量能比率', '价格vsEMA%', '张力', '加速度', 'DXY燃料']
for param in params:
    if param in df.columns:
        action_values = action_df[param].dropna()
        hold_values = hold_df[param].dropna()

        action_mean = action_values.mean()
        hold_mean = hold_values.mean()
        diff = action_mean - hold_mean

        # Cohen's d (效应量)
        pooled_std = np.sqrt((action_values.std()**2 + hold_values.std()**2) / 2)
        cohens_d = abs(diff / pooled_std) if pooled_std > 0 else 0

        # t-test (统计显著性)
        t_stat, p_value = stats.ttest_ind(action_values, hold_values)

        action_str = f"{action_mean:.4f}"
        hold_str = f"{hold_mean:.4f}"
        diff_str = f"{diff:+.4f}"
        d_str = f"{cohens_d:.4f}"

        if p_value < 0.001:
            sig_str = "*** (p<0.001)"
        elif p_value < 0.01:
            sig_str = "** (p<0.01)"
        elif p_value < 0.05:
            sig_str = "* (p<0.05)"
        else:
            sig_str = "ns"

        print(f"{param:<15} {action_str:<12} {hold_str:<12} {diff_str:<12} {d_str:<12} {sig_str:<15}")

# ============================================================================
# Step 3: 分析高低点时的特征
# ============================================================================
print("\n" + "="*140)
print("STEP 3: Peak/Valley Characteristics")
print("="*140)

peaks = df[df['高低点'] == '高点']
valleys = df[df['高低点'] == '低点']

print(f"\n高点数量: {len(peaks)}")
print(f"低点数量: {len(valleys)}")

print("\n【高点时的参数特征】")
if len(peaks) > 0:
    for param in params:
        if param in df.columns:
            values = peaks[param].dropna()
            if len(values) > 0:
                print(f"  {param}: {values.mean():.4f} ± {values.std():.4f}")

print("\n【低点时的参数特征】")
if len(valleys) > 0:
    for param in params:
        if param in df.columns:
            values = valleys[param].dropna()
            if len(values) > 0:
                print(f"  {param}: {values.mean():.4f} ± {values.std():.4f}")

# ============================================================================
# Step 4: 按信号类型分析成功率
# ============================================================================
print("\n" + "="*140)
print("STEP 4: Success Rate by Signal Type")
print("="*140)

for signal_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'OSCILLATION']:
    subset = df[df['信号类型'] == signal_type]
    if len(subset) > 0:
        action_count = len(subset[subset['黄金信号'] == 'ACTION'])
        total_count = len(subset)
        action_rate = action_count / total_count * 100

        print(f"\n{signal_type}:")
        print(f"  总数: {total_count}")
        print(f"  ACTION: {action_count} ({action_rate:.1f}%)")
        print(f"  HOLD: {total_count - action_count} ({100-action_rate:.1f}%)")

        # 该信号类型的参数均值
        print(f"  平均参数:")
        for param in params:
            if param in df.columns:
                values = subset[param].dropna()
                if len(values) > 0:
                    print(f"    {param}: {values.mean():.4f}")

# ============================================================================
# Step 5: 最佳入场时机分析
# ============================================================================
print("\n" + "="*140)
print("STEP 5: Optimal Entry Timing Analysis")
print("="*140)

# 分析信号模式开始时的特征
signal_starts = []
prev_mode = None
for i in range(len(df)):
    current_mode = df.loc[i, '信号模式']
    if i == 0 or current_mode != prev_mode:
        signal_starts.append(i)
    prev_mode = current_mode

print(f"\n信号模式切换次数: {len(signal_starts)}")

# 统计信号开始后第1根K线是ACTION的概率
first_kline_actions = []
for start_idx in signal_starts:
    if start_idx < len(df):
        gold_signal = df.loc[start_idx, '黄金信号']
        first_kline_actions.append(gold_signal)

if first_kline_actions:
    action_pct = first_kline_actions.count('ACTION') / len(first_kline_actions) * 100
    print(f"信号开始第1根K线是ACTION的概率: {action_pct:.1f}%")

# 分析在信号模式内，等待极值点的效果
print("\n【等待极值点 vs 立即入场】")
for signal_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY']:
    subset = df[df['信号类型'] == signal_type].copy()

    if len(subset) > 0:
        # 找到该信号类型的所有连续段
        segments = []
        start_idx = subset.index[0]
        prev_idx = start_idx

        for idx in subset.index[1:]:
            if idx != prev_idx + 1:  # 不连续
                segments.append((start_idx, prev_idx))
                start_idx = idx
            prev_idx = idx
        segments.append((start_idx, prev_idx))

        # 分析每一段
        total_entries = 0
        immediate_entries = 0  # 第1根K线就入场
        extreme_entries = 0    # 等到极值点入场

        for start, end in segments:
            first_action = df.loc[start, '黄金信号']

            # 统计开仓动作
            long_entries = 0
            short_entries = 0

            for idx in range(start, end + 1):
                action = df.loc[idx, '最优动作']
                if '开多' in action:
                    long_entries += 1
                    if idx == start:
                        immediate_entries += 1
                    if df.loc[idx, '高低点'] in ['低点', '高点']:
                        extreme_entries += 1
                elif '开空' in action:
                    short_entries += 1
                    if idx == start:
                        immediate_entries += 1
                    if df.loc[idx, '高低点'] in ['高点', '低点']:
                        extreme_entries += 1

            total_entries += long_entries + short_entries

        print(f"\n{signal_type}:")
        print(f"  总入场次数: {total_entries}")
        print(f"  信号开始立即入场: {immediate_entries} ({immediate_entries/max(total_entries,1)*100:.1f}%)")
        print(f"  等待极值点入场: {extreme_entries} ({extreme_entries/max(total_entries,1)*100:.1f}%)")

# ============================================================================
# Step 6: 识别可能错误信号的特征
# ============================================================================
print("\n" + "="*140)
print("STEP 6: False Signal Detection")
print("="*140)

# 定义"可能错误"的信号：信号类型明确但最终是HOLD（没有交易机会）
potential_false = df[
    (df['信号类型'].isin(['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY'])) &
    (df['黄金信号'] == 'HOLD')
]

good_signals = df[
    (df['信号类型'].isin(['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY'])) &
    (df['黄金信号'] == 'ACTION')
]

print(f"\n可能错误的信号（SINGULARITY但HOLD）: {len(potential_false)}")
print(f"良好的信号（SINGULARITY且ACTION）: {len(good_signals)}")

if len(potential_false) > 0 and len(good_signals) > 0:
    print("\n【错误信号 vs 良好信号的参数对比】")
    print(f"{'参数':<15} {'错误信号均值':<15} {'良好信号均值':<15} {'差异':<12} {'Cohen d':<12}")
    print("-" * 100)

    for param in params:
        if param in df.columns:
            false_values = potential_false[param].dropna()
            good_values = good_signals[param].dropna()

            if len(false_values) > 0 and len(good_values) > 0:
                false_mean = false_values.mean()
                good_mean = good_values.mean()
                diff = false_mean - good_mean

                pooled_std = np.sqrt((false_values.std()**2 + good_values.std()**2) / 2)
                cohens_d = abs(diff / pooled_std) if pooled_std > 0 else 0

                false_str = f"{false_mean:.4f}"
                good_str = f"{good_mean:.4f}"
                diff_str = f"{diff:+.4f}"
                d_str = f"{cohens_d:.4f}"

                print(f"{param:<15} {false_str:<15} {good_str:<15} {diff_str:<12} {d_str:<12}")

# ============================================================================
# Step 7: 推荐的入场阈值
# ============================================================================
print("\n" + "="*140)
print("STEP 7: Recommended Entry Thresholds")
print("="*140)

print("\n基于统计分析，推荐的入场条件：")
print("\n1. 量能比率阈值:")

# 分析不同量能比率下的ACTION概率
volume_bins = [0, 0.5, 0.8, 1.0, 1.2, 5.0]
volume_labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.2', '>1.2']

df_temp = df.copy()
df_temp['量能区间'] = pd.cut(df_temp['量能比率'], bins=volume_bins, labels=volume_labels)

for label in volume_labels:
    subset = df_temp[df_temp['量能区间'] == label]
    if len(subset) > 0:
        action_rate = len(subset[subset['黄金信号']=='ACTION']) / len(subset) * 100
        print(f"  量能比率 {label}: ACTION率 = {action_rate:.1f}% ({len(subset)}条)")

print("\n2. 价格vsEMA%阈值:")
price_bins = [-5, -1, -0.5, 0, 0.5, 1, 5]
price_labels = ['<-1%', '-1~-0.5%', '-0.5~0%', '0~0.5%', '0.5~1%', '>1%']

df_temp['价格区间'] = pd.cut(df_temp['价格vsEMA%'], bins=price_bins, labels=price_labels)

for label in price_labels:
    subset = df_temp[df_temp['价格区间'] == label]
    if len(subset) > 0:
        action_rate = len(subset[subset['黄金信号']=='ACTION']) / len(subset) * 100
        print(f"  价格vsEMA {label}: ACTION率 = {action_rate:.1f}% ({len(subset)}条)")

# ============================================================================
# Step 8: 保存分析结果
# ============================================================================
print("\n" + "="*140)
print("STEP 8: Save Analysis Results")
print("="*140)

# 创建特征统计表
summary_data = {
    '参数': params,
    'ACTION均值': [action_df[param].mean() for param in params],
    'HOLD均值': [hold_df[param].mean() for param in params],
    '差异': [action_df[param].mean() - hold_df[param].mean() for param in params],
    'ACTION标准差': [action_df[param].std() for param in params],
    'HOLD标准差': [hold_df[param].std() for param in params],
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('统计结果_ACTION_vs_HOLD.csv', index=False, encoding='utf-8-sig')
print("\n已保存: 统计结果_ACTION_vs_HOLD.csv")

print("\n" + "="*140)
print("COMPLETE")
print("="*140)
