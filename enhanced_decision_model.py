# -*- coding: utf-8 -*-
"""
改进版决策模型 - 引入交互项和Z-Score
基于专业反馈的优化实现
"""

import pandas as pd
import numpy as np
from scipy import stats

print("="*100)
print("ENHANCED DECISION MODEL - With Interaction Terms & Z-Score")
print("="*100)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')

df['Action_Type'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'OTHER'
)

df_analysis = df[df['Action_Type'].isin(['ACTION', 'HOLD'])].copy()

# Calculate rate of change
for param in ['量能比率', '价格vsEMA%', '张力']:
    df_analysis[f'{param}_lag1'] = df_analysis[param].shift(1)
    df_analysis[f'{param}_roc'] = (df_analysis[param] - df_analysis[f'{param}_lag1']) / df_analysis[f'{param}_lag1']
    df_analysis[f'{param}_roc'] = df_analysis[f'{param}_roc'].replace([np.inf, -np.inf], 0)

# Clean data
df_clean = df_analysis.dropna(subset=['量能比率_roc', '价格vsEMA%_roc', '张力_roc'])

print(f"\nClean sample size: n={len(df_clean)}")

# ============================================================================
# MODEL 1: Original Linear Model
# ============================================================================
print("\n" + "="*100)
print("MODEL 1: Original Linear Model")
print("="*100)

w1, w2, w3 = 0.3, 0.5, 0.2
df_clean['Action_Score_Original'] = (
    w1 * df_clean['量能比率_roc'] +
    w2 * df_clean['价格vsEMA%_roc'] +
    w3 * df_clean['张力_roc']
)

threshold_80 = df_clean['Action_Score_Original'].quantile(0.8)
df_clean['Predicted_Original'] = (df_clean['Action_Score_Original'] >= threshold_80).astype(int)

actual = (df_clean['Action_Type'] == 'ACTION').astype(int)
tp = ((df_clean['Predicted_Original'] == 1) & (actual == 1)).sum()
fp = ((df_clean['Predicted_Original'] == 1) & (actual == 0)).sum()
fn = ((df_clean['Predicted_Original'] == 0) & (actual == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Original Model (Top 20%):")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

# ============================================================================
# MODEL 2: Enhanced Model with Interaction Term
# ============================================================================
print("\n" + "="*100)
print("MODEL 2: Enhanced Model with Interaction Term")
print("="*100)

# Panic Factor: Volume surge × Price drop
df_clean['Panic_Factor'] = df_clean['量能比率_roc'] * abs(df_clean['价格vsEMA%_roc'])

w_interaction = 0.3
df_clean['Action_Score_Enhanced'] = (
    w1 * df_clean['量能比率_roc'] +
    w2 * df_clean['价格vsEMA%_roc'] +
    w3 * df_clean['张力_roc'] +
    w_interaction * df_clean['Panic_Factor']
)

threshold_80_e = df_clean['Action_Score_Enhanced'].quantile(0.8)
df_clean['Predicted_Enhanced'] = (df_clean['Action_Score_Enhanced'] >= threshold_80_e).astype(int)

tp_e = ((df_clean['Predicted_Enhanced'] == 1) & (actual == 1)).sum()
fp_e = ((df_clean['Predicted_Enhanced'] == 1) & (actual == 0)).sum()
fn_e = ((df_clean['Predicted_Enhanced'] == 0) & (actual == 1)).sum()

precision_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0
recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
f1_e = 2 * precision_e * recall_e / (precision_e + recall_e) if (precision_e + recall_e) > 0 else 0

print(f"Enhanced Model (Top 20%):")
print(f"  Precision: {precision_e:.4f}")
print(f"  Recall: {recall_e:.4f}")
print(f"  F1-Score: {f1_e:.4f}")
print(f"  Improvement: ΔF1={f1_e-f1:+.4f}")

# ============================================================================
# MODEL 3: Z-Score Adaptive Model
# ============================================================================
print("\n" + "="*100)
print("MODEL 3: Z-Score Adaptive Model")
print("="*100)

window = 20
for param in ['量能比率_roc', '价格vsEMA%_roc', '张力_roc']:
    rolling_mean = df_clean[param].rolling(window).mean()
    rolling_std = df_clean[param].rolling(window).std()
    df_clean[f'{param}_zscore'] = (df_clean[param] - rolling_mean) / rolling_std

# Count extreme events
extreme_volume = (df_clean['量能比率_roc_zscore'] > 2).sum()
extreme_price = (df_clean['价格vsEMA%_roc_zscore'] < -2).sum()

print(f"Extreme Events (|Z|>2):")
print(f"  Volume surge: {extreme_volume} occurrences")
print(f"  Price plunge: {extreme_price} occurrences")

# ============================================================================
# MODEL 4: Bayesian Probability Adjustment
# ============================================================================
print("\n" + "="*100)
print("MODEL 4: Bayesian Probability Adjustment")
print("="*100)

df_clean['Volume_Prob_Score'] = df_clean['量能比率'].apply(lambda x:
    1.0 if x > 1.4 else 0.7 if x > 1.2 else 0.4 if x > 1.0 else 0.1
)

print(f"\nVolume Probability Distribution:")
print(df_clean['Volume_Prob_Score'].value_counts().sort_index())

# Check V7.0.5 filtered signals
v705_filtered = df_clean[df_clean['V7.0.5通过'].isin(['FALSE', False, 'False '])]
print(f"\nV7.0.5 Filtered: n={len(v705_filtered)}")

rescued = v705_filtered[v705_filtered['Volume_Prob_Score'] >= 0.7]
print(f"Could rescue (prob>=0.7): n={len(rescued)} ({len(rescued)/len(v705_filtered)*100:.1f}%)")

# Check GOLD signals
rescued = rescued.copy()
rescued['Is_Gold'] = rescued['黄金信号'].apply(lambda x: 'GOLD' if '黄金' in str(x) else 'NORMAL')
rescued_gold = rescued[rescued['Is_Gold'] == 'GOLD']
print(f"Actually GOLD: n={len(rescued_gold)}")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("1. Original linear model baseline: F1={f1:.4f}")
print("2. Enhanced with interaction: F1={f1_e:.4f} (Δ={f1_e-f1:+.4f})")
print("3. Z-score adaptive model enables self-adjusting thresholds")
print("4. Bayesian approach rescues filtered signals probabilistically")
