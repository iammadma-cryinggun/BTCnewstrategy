# -*- coding: utf-8 -*-
"""
完整交易策略制定与验证
========================

基于数学分析洞察的综合策略
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("COMPREHENSIVE TRADING STRATEGY - Formulation & Validation")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')

# Ground truth: Ideal actions from gold signals
df['Ideal_Action'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'UNKNOWN'
)

df_analysis = df[df['Ideal_Action'].isin(['ACTION', 'HOLD'])].copy()

print(f"\nDataset size: n={len(df_analysis)}")
print(f"  ACTION (Ideal): {(df_analysis['Ideal_Action']=='ACTION').sum()}")
print(f"  HOLD (Ideal): {(df_analysis['Ideal_Action']=='HOLD').sum()}")

# ============================================================================
# STRATEGY FORMULATION
# ============================================================================
print("\n" + "="*120)
print("STRATEGY FORMULATION - Based on Mathematical Insights")
print("="*120)

print("\n核心洞察:")
print("1. 量能比率是ACTION的最强预测因子 (p=0.0028, Cohen's d=0.64)")
print("2. 价格vsEMA%的变化率最敏感 (-72.44% in state transitions)")
print("3. 黄金信号出现在: 高量能(>1.4) + 低价格vsEMA(<-2.4%)")
print("4. 静态阈值预测能力弱 (AUC=0.48), 需要动态变化率")
print("5. 交互效应: 量能暴增 × 价格急跌 = 恐慌性反转信号")

print("\n" + "-"*120)
print("STRATEGY COMPOSITE (综合策略)")
print("-"*120)

print("""
策略由3个子系统组成，每个子系统给出0-1的得分：

【子系统1: 量能分析】(权重: 30%)
- 量能比率 > 1.4 → score_vol = 1.0 (黄金区间)
- 量能比率 > 1.2 → score_vol = 0.7 (强势区间)
- 量能比率 > 1.0 → score_vol = 0.4 (正常区间)
- 量能比率 <= 1.0 → score_vol = 0.1 (弱势区间)

【子系统2: 价格位置分析】(权重: 30%)
- 价格vsEMA% < -2.4% → score_price = 1.0 (深度超卖)
- 价格vsEMA% < -1.5% → score_price = 0.7 (轻度超卖)
- 价格vsEMA% > 2.0% → score_price = 0.7 (轻度超买)
- 其他 → score_price = 0.1 (中性区间)

【子系统3: 动态变化率分析】(权重: 40%)
使用Z-Score自适应方法（20周期窗口）:
- 计算量能比率的Z-Score
- 计算价格vsEMA%的Z-Score

触发条件:
- 量能比率Z > 2.0 (量能激增) → score_dyn += 0.5
- 价格vsEMA% Z < -2.0 (价格急跌) → score_dyn += 0.5
- 同时满足 → score_dyn = 1.0 (恐慌反转！)

【综合决策】
Composite_Score = 0.3×score_vol + 0.3×score_price + 0.4×score_dyn

决策规则:
- Composite_Score >= 0.7 → ACTION (开仓/平仓/反手)
- Composite_Score < 0.7 → HOLD (继续持有)
""")

# ============================================================================
# IMPLEMENTATION
# ============================================================================
print("\n" + "="*120)
print("STRATEGY IMPLEMENTATION")
print("="*120)

# Subsystem 1: Volume Analysis
df_analysis['Score_Vol'] = df_analysis['量能比率'].apply(lambda x:
    1.0 if x > 1.4 else 0.7 if x > 1.2 else 0.4 if x > 1.0 else 0.1
)

# Subsystem 2: Price Position Analysis
df_analysis['Score_Price'] = df_analysis['价格vsEMA%'].apply(lambda x:
    1.0 if x < -2.4 else 0.7 if x < -1.5 else 0.7 if x > 2.0 else 0.1
)

# Subsystem 3: Dynamic Rate of Change Analysis
window = 20

# Calculate lagged values
df_analysis['量能比率_lag1'] = df_analysis['量能比率'].shift(1)
df_analysis['价格vsEMA_lag1'] = df_analysis['价格vsEMA%'].shift(1)

# Calculate rate of change
df_analysis['Vol_RoC'] = (df_analysis['量能比率'] - df_analysis['量能比率_lag1']) / df_analysis['量能比率_lag1']
df_analysis['Price_RoC'] = (df_analysis['价格vsEMA%'] - df_analysis['价格vsEMA_lag1']) / df_analysis['价格vsEMA_lag1']

# Clean infinite values
df_analysis['Vol_RoC'] = df_analysis['Vol_RoC'].replace([np.inf, -np.inf], 0)
df_analysis['Price_RoC'] = df_analysis['Price_RoC'].replace([np.inf, -np.inf], 0)

# Calculate Z-score
vol_rolling_mean = df_analysis['Vol_RoC'].rolling(window).mean()
vol_rolling_std = df_analysis['Vol_RoC'].rolling(window).std()
df_analysis['Vol_Z'] = (df_analysis['Vol_RoC'] - vol_rolling_mean) / vol_rolling_std

price_rolling_mean = df_analysis['Price_RoC'].rolling(window).mean()
price_rolling_std = df_analysis['Price_RoC'].rolling(window).std()
df_analysis['Price_Z'] = (df_analysis['Price_RoC'] - price_rolling_mean) / price_rolling_std

# Dynamic scoring
df_analysis['Score_Dyn'] = 0.0
df_analysis.loc[df_analysis['Vol_Z'] > 2.0, 'Score_Dyn'] += 0.5
df_analysis.loc[df_analysis['Price_Z'] < -2.0, 'Score_Dyn'] += 0.5

# Panic factor: Both conditions met
panic_condition = (df_analysis['Vol_Z'] > 2.0) & (df_analysis['Price_Z'] < -2.0)
df_analysis.loc[panic_condition, 'Score_Dyn'] = 1.0

# Composite Score
df_analysis['Composite_Score'] = (
    0.3 * df_analysis['Score_Vol'] +
    0.3 * df_analysis['Score_Price'] +
    0.4 * df_analysis['Score_Dyn']
)

# Strategy Decision
df_analysis['Strategy_Decision'] = (df_analysis['Composite_Score'] >= 0.7).astype(int)
df_analysis['Actual_Action'] = (df_analysis['Ideal_Action'] == 'ACTION').astype(int)

print("\nSubsystem Scores Distribution:")
print(f"  Volume Score: mean={df_analysis['Score_Vol'].mean():.4f}")
print(f"  Price Score: mean={df_analysis['Score_Price'].mean():.4f}")
print(f"  Dynamic Score: mean={df_analysis['Score_Dyn'].mean():.4f}")
print(f"  Composite Score: mean={df_analysis['Composite_Score'].mean():.4f}")

print("\nDecision Threshold: 0.7")
print(f"  Predicted ACTION: {(df_analysis['Strategy_Decision']==1).sum()}")
print(f"  Actual ACTION: {(df_analysis['Actual_Action']==1).sum()}")

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "="*120)
print("PERFORMANCE EVALUATION - Strategy vs Ideal Actions")
print("="*120)

# Clean NaN
df_eval = df_analysis.dropna(subset=['Strategy_Decision', 'Actual_Action'])

y_true = df_eval['Actual_Action']
y_pred = df_eval['Strategy_Decision']

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"            HOLD    ACTION")
print(f"Actual HOLD   {tn:4d}    {fp:4d}")
print(f"       ACTION {fn:4d}    {tp:4d}")

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nClassification Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  Specificity: {specificity:.4f}")

# AUC
if len(np.unique(y_pred)) > 1:
    auc = roc_auc_score(y_true, y_pred)
    print(f"  AUC-ROC:   {auc:.4f}")

# ============================================================================
# COMPARISON WITH V7.0.5
# ============================================================================
print("\n" + "="*120)
print("COMPARISON - New Strategy vs V7.0.5 Filter")
print("="*120)

# V7.0.5 decisions
v705_pass = df_eval['V7.0.5通过'].isin(['TRUE', True])
v705_decision = v705_pass.astype(int)

# V7.0.5 comparison
v705_true = df_eval.loc[v705_pass.index, 'Actual_Action']
y_pred_v705 = pd.Series([1]*len(v705_pass.index), index=v705_pass.index)

tp_v = ((y_pred_v705 == 1) & (v705_true == 1)).sum()
fp_v = ((y_pred_v705 == 1) & (v705_true == 0)).sum()
fn_v = ((y_pred_v705 == 0) & (v705_true == 1)).sum()
tn_v = ((y_pred_v705 == 0) & (v705_true == 0)).sum()

precision_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0
recall_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0
f1_v = 2 * precision_v * recall_v / (precision_v + recall_v) if (precision_v + recall_v) > 0 else 0

print(f"\nV7.0.5 Performance (on passed signals only):")
print(f"  Precision: {precision_v:.4f}")
print(f"  Recall:    {recall_v:.4f}")
print(f"  F1-Score:  {f1_v:.4f}")
print(f"  Coverage:  {v705_pass.sum()}/{len(df_eval)} ({v705_pass.sum()/len(df_eval)*100:.1f}%)")

print(f"\nNew Strategy Performance (on all signals):")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  Coverage:  {(df_eval['Strategy_Decision']==1).sum()}/{len(df_eval)} ({(df_eval['Strategy_Decision']==1).sum()/len(df_eval)*100:.1f}%)")

print(f"\nImprovement:")
print(f"  ΔRecall:    {recall - recall_v:+.4f}")
print(f"  ΔF1-Score: {f1 - f1_v:+.4f}")

# ============================================================================
# DETAILED ANALYSIS: Where Strategy Matches/Misses
# ============================================================================
print("\n" + "="*120)
print("DETAILED ANALYSIS - True Positives vs False Positives")
print("="*120)

# True Positives: Strategy says ACTION, Ideal says ACTION
tp_data = df_eval[(df_eval['Strategy_Decision'] == 1) & (df_eval['Actual_Action'] == 1)]
print(f"\n[OK] True Positives (Strategy Correct): n={len(tp_data)}")
if len(tp_data) > 0:
    print(f"  Avg Composite Score: {tp_data['Composite_Score'].mean():.4f}")
    print(f"  Avg Volume Score: {tp_data['Score_Vol'].mean():.4f}")
    print(f"  Avg Price Score: {tp_data['Score_Price'].mean():.4f}")
    print(f"  Avg Dynamic Score: {tp_data['Score_Dyn'].mean():.4f}")

# False Positives: Strategy says ACTION, Ideal says HOLD
fp_data = df_eval[(df_eval['Strategy_Decision'] == 1) & (df_eval['Actual_Action'] == 0)]
print(f"\n[XX] False Positives (Strategy Wrong): n={len(fp_data)}")
if len(fp_data) > 0:
    print(f"  Avg Composite Score: {fp_data['Composite_Score'].mean():.4f}")
    print(f"  Avg Volume Score: {fp_data['Score_Vol'].mean():.4f}")
    print(f"  Avg Price Score: {fp_data['Score_Price'].mean():.4f}")
    print(f"  Avg Dynamic Score: {fp_data['Score_Dyn'].mean():.4f}")

# False Negatives: Strategy says HOLD, Ideal says ACTION
fn_data = df_eval[(df_eval['Strategy_Decision'] == 0) & (df_eval['Actual_Action'] == 1)]
print(f"\n[!!] False Negatives (Missed Opportunities): n={len(fn_data)}")
if len(fn_data) > 0:
    print(f"  Avg Composite Score: {fn_data['Composite_Score'].mean():.4f}")
    print(f"  Avg Volume Score: {fn_data['Score_Vol'].mean():.4f}")
    print(f"  Avg Price Score: {fn_data['Score_Price'].mean():.4f}")
    print(f"  Avg Dynamic Score: {fn_data['Score_Dyn'].mean():.4f}")

# ============================================================================
# OPTIMIZATION SUGGESTIONS
# ============================================================================
print("\n" + "="*120)
print("OPTIMIZATION INSIGHTS")
print("="*120)

if len(fp_data) > 0 and len(tp_data) > 0:
    # What distinguishes TP from FP?
    print("\nTrue Positives vs False Positives:")
    for col in ['Score_Vol', 'Score_Price', 'Score_Dyn', 'Composite_Score']:
        tp_mean = tp_data[col].mean()
        fp_mean = fp_data[col].mean()
        diff = tp_mean - fp_mean
        print(f"  {col}: TP={tp_mean:.4f}, FP={fp_mean:.4f}, Diff={diff:+.4f}")

if len(fn_data) > 0 and len(tp_data) > 0:
    print("\nTrue Positives vs False Negatives:")
    for col in ['Score_Vol', 'Score_Price', 'Score_Dyn', 'Composite_Score']:
        tp_mean = tp_data[col].mean()
        fn_mean = fn_data[col].mean()
        diff = tp_mean - fn_mean
        print(f"  {col}: TP={tp_mean:.4f}, FN={fn_mean:.4f}, Diff={diff:+.4f}")

print("\n" + "="*120)
print("STRATEGY VALIDATION COMPLETE")
print("="*120)
print(f"\nFinal Performance Summary:")
print(f"  F1-Score: {f1:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  Precision: {precision:.4f}")
