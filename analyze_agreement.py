# -*- coding: utf-8 -*-
"""
策略信号 vs 手动标注符合度分析
==============================
"""

import pandas as pd
import numpy as np

print("="*120)
print("STRATEGY SIGNAL vs MANUAL ANNOTATION - AGREEMENT ANALYSIS")
print("="*120)

# Load data with V8.0 results
df = pd.read_csv('V8_0_strategy_results.csv', encoding='utf-8-sig')

# Extract ideal actions
df['Ideal_Action'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'UNKNOWN'
)

# Filter valid annotations
df_valid = df[df['Ideal_Action'].isin(['ACTION', 'HOLD'])].copy()

print(f"\n总样本数: {len(df_valid)}")
print(f"手动标注 ACTION: {sum(df_valid['Ideal_Action']=='ACTION')}")
print(f"手动标注 HOLD: {sum(df_valid['Ideal_Action']=='HOLD')}")

# ============================================================================
# 1. 总体符合度
# ============================================================================
print("\n" + "="*120)
print("1. 总体符合度 (OVERALL AGREEMENT)")
print("="*120)

# Binary encoding
ideal_binary = (df_valid['Ideal_Action'] == 'ACTION').astype(int)
v8_binary = (df_valid['V8_Action'] == 'ACTION').astype(int)
v705_binary = df_valid['V7.0.5通过'].isin(['TRUE', True]).astype(int)

# Calculate agreement
v8_agree = (v8_binary == ideal_binary).sum()
v705_agree = (v705_binary == ideal_binary).sum()

print(f"\nV8.0策略符合度:")
print(f"  一致数量: {v8_agree}/{len(df_valid)}")
print(f"  符合率: {v8_agree/len(df_valid)*100:.2f}%")

print(f"\nV7.0.5策略符合度:")
print(f"  一致数量: {v705_agree}/{len(df_valid)}")
print(f"  符合率: {v705_agree/len(df_valid)*100:.2f}%")

# ============================================================================
# 2. 混淆矩阵分析
# ============================================================================
print("\n" + "="*120)
print("2. 混淆矩阵分析 (CONFUSION MATRIX)")
print("="*120)

# V8.0 Confusion Matrix
tp_v8 = ((v8_binary == 1) & (ideal_binary == 1)).sum()
tn_v8 = ((v8_binary == 0) & (ideal_binary == 0)).sum()
fp_v8 = ((v8_binary == 1) & (ideal_binary == 0)).sum()
fn_v8 = ((v8_binary == 0) & (ideal_binary == 1)).sum()

print("\n【V8.0策略混淆矩阵】")
print(f"                    策略预测")
print(f"            HOLD            ACTION")
print(f"实际 HOLD   {tn_v8:4d} (正确)    {fp_v8:4d} (误报)")
print(f"     ACTION {fn_v8:4d} (漏报)    {tp_v8:4d} (正确)")

# V7.0.5 Confusion Matrix
tp_v7 = ((v705_binary == 1) & (ideal_binary == 1)).sum()
tn_v7 = ((v705_binary == 0) & (ideal_binary == 0)).sum()
fp_v7 = ((v705_binary == 1) & (ideal_binary == 0)).sum()
fn_v7 = ((v705_binary == 0) & (ideal_binary == 1)).sum()

print("\n【V7.0.5策略混淆矩阵】")
print(f"                    策略预测")
print(f"            HOLD            ACTION")
print(f"实际 HOLD   {tn_v7:4d} (正确)    {fp_v7:4d} (误报)")
print(f"     ACTION {fn_v7:4d} (漏报)    {tp_v7:4d} (正确)")

# ============================================================================
# 3. 详细符合度分解
# ============================================================================
print("\n" + "="*120)
print("3. 详细符合度分解 (DETAILED AGREEMENT BREAKDOWN)")
print("="*120)

# 对于ACTION信号的符合度
action_mask = (ideal_binary == 1)
v8_action_correct = ((v8_binary == 1) & action_mask).sum()
v705_action_correct = ((v705_binary == 1) & action_mask).sum()
total_action = action_mask.sum()

print(f"\n【ACTION信号符合度】(手动标注为ACTION的情况)")
print(f"  总ACTION数: {total_action}")
print(f"  V8.0捕获: {v8_action_correct} ({v8_action_correct/total_action*100:.1f}%)")
print(f"  V7.0.5捕获: {v705_action_correct} ({v705_action_correct/total_action*100:.1f}%)")

# 对于HOLD信号的符合度
hold_mask = (ideal_binary == 0)
v8_hold_correct = ((v8_binary == 0) & hold_mask).sum()
v705_hold_correct = ((v705_binary == 0) & hold_mask).sum()
total_hold = hold_mask.sum()

print(f"\n【HOLD信号符合度】(手动标注为HOLD/继续持仓的情况)")
print(f"  总HOLD数: {total_hold}")
print(f"  V8.0正确: {v8_hold_correct} ({v8_hold_correct/total_hold*100:.1f}%)")
print(f"  V7.0.5正确: {v705_hold_correct} ({v705_hold_correct/total_hold*100:.1f}%)")

# ============================================================================
# 4. 分类指标
# ============================================================================
print("\n" + "="*120)
print("4. 分类性能指标 (CLASSIFICATION METRICS)")
print("="*120)

# V8.0 Metrics
precision_v8 = tp_v8 / (tp_v8 + fp_v8) if (tp_v8 + fp_v8) > 0 else 0
recall_v8 = tp_v8 / (tp_v8 + fn_v8) if (tp_v8 + fn_v8) > 0 else 0
f1_v8 = 2 * precision_v8 * recall_v8 / (precision_v8 + recall_v8) if (precision_v8 + recall_v8) > 0 else 0
specificity_v8 = tn_v8 / (tn_v8 + fp_v8) if (tn_v8 + fp_v8) > 0 else 0

# V7.0.5 Metrics
precision_v7 = tp_v7 / (tp_v7 + fp_v7) if (tp_v7 + fp_v7) > 0 else 0
recall_v7 = tp_v7 / (tp_v7 + fn_v7) if (tp_v7 + fn_v7) > 0 else 0
f1_v7 = 2 * precision_v7 * recall_v7 / (precision_v7 + recall_v7) if (precision_v7 + recall_v7) > 0 else 0
specificity_v7 = tn_v7 / (tn_v7 + fp_v7) if (tn_v7 + fp_v7) > 0 else 0

print(f"\n{'指标':<15} {'V8.0':<12} {'V7.0.5':<12} {'更优':<10}")
print("-"*60)
print(f"{'精确率':<15} {precision_v8:<12.4f} {precision_v7:<12.4f} {'V8.0' if precision_v8 > precision_v7 else 'V7.0.5'}")
print(f"{'召回率':<15} {recall_v8:<12.4f} {recall_v7:<12.4f} {'V8.0' if recall_v8 > recall_v7 else 'V7.0.5'}")
print(f"{'F1分数':<15} {f1_v8:<12.4f} {f1_v7:<12.4f} {'V8.0' if f1_v8 > f1_v7 else 'V7.0.5'}")
print(f"{'特异度':<15} {specificity_v8:<12.4f} {specificity_v7:<12.4f} {'V8.0' if specificity_v8 > specificity_v7 else 'V7.0.5'}")

# ============================================================================
# 5. Cohen's Kappa - 统计学一致性检验
# ============================================================================
print("\n" + "="*120)
print("5. Cohen's Kappa - 统计学一致性")
print("="*120)

def calculate_kappa(tp, tn, fp, fn, n):
    """Calculate Cohen's Kappa"""
    # Observed agreement
    po = (tp + tn) / n

    # Expected agreement by chance
    pa_yes = ((tp + fp) * (tp + fn)) / (n * n)
    pa_no = ((tn + fn) * (tn + fp)) / (n * n)
    pe = pa_yes + pa_no

    # Kappa
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    return kappa

kappa_v8 = calculate_kappa(tp_v8, tn_v8, fp_v8, fn_v8, len(df_valid))
kappa_v7 = calculate_kappa(tp_v7, tn_v7, fp_v7, fn_v7, len(df_valid))

print(f"\nV8.0 Kappa: {kappa_v8:.4f}")
print(f"  ({'Substantial' if kappa_v8 > 0.6 else 'Moderate' if kappa_v8 > 0.4 else 'Fair' if kappa_v8 > 0.2 else 'Slight'})")

print(f"\nV7.0.5 Kappa: {kappa_v7:.4f}")
print(f"  ({'Substantial' if kappa_v7 > 0.6 else 'Moderate' if kappa_v7 > 0.4 else 'Fair' if kappa_v7 > 0.2 else 'Slight'})")

# ============================================================================
# 6. 逐行对比分析 - 符合/不符合的案例
# ============================================================================
print("\n" + "="*120)
print("6. 案例分析 - 符合与不符合的典型样本")
print("="*120)

# 完全符合 (V8 = 手动标注)
perfect_agree_v8 = df_valid[
    (df_valid['V8_Action'] == df_valid['Ideal_Action'])
].copy()

# V8.0符合但V7.0.5不符合
v8_unique_correct = df_valid[
    (df_valid['V8_Action'] == df_valid['Ideal_Action']) &
    (df_valid['V7.0.5通过'].isin(['FALSE', False])) &
    (df_valid['Ideal_Action'] == 'ACTION')
].copy()

# 两个策略都错了
both_wrong = df_valid[
    (df_valid['V8_Action'] != df_valid['Ideal_Action']) &
    (df_valid['V7.0.5通过'].isin(['FALSE', False])) &
    (df_valid['Ideal_Action'] == 'ACTION')
].copy()

print(f"\n【V8.0完全符合的案例】: {len(perfect_agree_v8)} 个")

if len(v8_unique_correct) > 0:
    print(f"\n【V8.0独有正确捕获】(V7.0.5漏掉但V8.0抓对): {len(v8_unique_correct)} 个")
    print("\n示例:")
    cols = ['时间', '信号类型', '量能比率', '价格vsEMA%', 'V8_Score', 'V7.0.5通过', 'V8_Action', '黄金信号']
    print(v8_unique_correct[cols].head(5).to_string(index=False))

if len(both_wrong) > 0:
    print(f"\n【两个策略都漏掉的黄金信号】: {len(both_wrong)} 个")
    print("\n示例:")
    print(both_wrong[cols].head(5).to_string(index=False))

# ============================================================================
# 7. 按信号类型分组分析
# ============================================================================
print("\n" + "="*120)
print("7. 按信号类型分组的符合度")
print("="*120)

for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'OSCILLATION']:
    subset = df_valid[df_valid['信号类型'] == sig_type]
    if len(subset) > 0:
        ideal_subset = (subset['Ideal_Action'] == 'ACTION').astype(int)
        v8_subset = (subset['V8_Action'] == 'ACTION').astype(int)
        v705_subset = subset['V7.0.5通过'].isin(['TRUE', True]).astype(int)

        agree_v8 = (v8_subset == ideal_subset).sum()
        agree_v7 = (v705_subset == ideal_subset).sum()

        print(f"\n{sig_type}:")
        print(f"  样本数: {len(subset)}")
        print(f"  V8.0符合率: {agree_v8}/{len(subset)} ({agree_v8/len(subset)*100:.1f}%)")
        print(f"  V7.0.5符合率: {agree_v7}/{len(subset)} ({agree_v7/len(subset)*100:.1f}%)")

# ============================================================================
# 8. 最终总结
# ============================================================================
print("\n" + "="*120)
print("最终总结 (FINAL SUMMARY)")
print("="*120)

print(f"\n总体符合度排名:")
if v8_agree > v705_agree:
    print(f"  🥇 第1名: V8.0 ({v8_agree/len(df_valid)*100:.2f}%)")
    print(f"  🥈 第2名: V7.0.5 ({v705_agree/len(df_valid)*100:.2f}%)")
else:
    print(f"  🥇 第1名: V7.0.5 ({v705_agree/len(df_valid)*100:.2f}%)")
    print(f"  🥈 第2名: V8.0 ({v8_agree/len(df_valid)*100:.2f}%)")

print(f"\n关键发现:")
print(f"  1. V8.0在精确率上{'优于' if precision_v8 > precision_v7 else '不及'}V7.0.5")
print(f"  2. V8.0在召回率上{'优于' if recall_v8 > recall_v7 else '不及'}V7.0.5")
print(f"  3. V8.0成功捕获了{len(v8_unique_correct)}个V7.0.5漏掉的黄金信号")
print(f"  4. 两个策略都漏掉了{len(both_wrong)}个黄金信号")

# ============================================================================
# 9. 符合度提升建议
# ============================================================================
print("\n" + "="*120)
print("符合度提升建议")
print("="*120)

print(f"\n当前问题:")
if recall_v8 < 0.6:
    print(f"  - 召回率偏低({recall_v8*100:.1f}%)，建议降低阈值至0.45")
if precision_v8 < 0.2:
    print(f"  - 精确率偏低({precision_v8*100:.1f}%)，建议增加'恐慌因子'权重")
if len(both_wrong) > 20:
    print(f"  - 有{len(both_wrong)}个黄金信号被完全漏掉，需要新特征")

print(f"\n优化方向:")
print(f"  1. 混合策略: V8.0(突变检测) + V7.0.5(趋势跟随)")
print(f"  2. 降低V8.0阈值至0.45，提高召回率")
print(f"  3. 增加方向性判断(Delta_EMA的正负)")
print(f"  4. 引入Z-Score自适应阈值")

print("\n" + "="*120)
print("分析完成")
print("="*120)
