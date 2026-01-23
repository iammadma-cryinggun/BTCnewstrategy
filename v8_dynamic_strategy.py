# -*- coding: utf-8 -*-
"""
V8.0 动态量化策略 - 基于变化率的突变检测系统
========================================

核心思想：市场是向量场，不是静态空间。
我们不关心"在哪里"，只关心"正以多快的速度冲向哪里"。
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("V8.0 DYNAMIC QUANTITATIVE STRATEGY - Mutation Detection System")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')

# Sort by time
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\nDataset: {len(df)} signals")
print(f"Date range: {df['时间'].min()} to {df['时间'].max()}")

# ============================================================================
# FEATURE ENGINEERING: Calculate Deltas (Rate of Change)
# ============================================================================
print("\n" + "="*120)
print("FEATURE ENGINEERING - Dynamic Delta Calculation")
print("="*120)

# Calculate previous values
df['Last_Vol'] = df['量能比率'].shift(1)
df['Last_EMA_Pct'] = df['价格vsEMA%'].shift(1)
df['Last_Tension'] = df['张力'].shift(1)

# Calculate Deltas (Change Rates)
df['Delta_Vol'] = (df['量能比率'] - df['Last_Vol']) / df['Last_Vol'].replace(0, np.nan)
df['Delta_EMA'] = (df['价格vsEMA%'] - df['Last_EMA_Pct'])  # Keep direction
df['Delta_EMA_Abs'] = df['Delta_EMA'].abs()  # Magnitude only
df['Delta_Tension'] = (df['张力'] - df['Last_Tension']).abs()

# Clean infinite values and fill NaN
df['Delta_Vol'] = df['Delta_Vol'].replace([np.inf, -np.inf], 0).fillna(0)
df['Delta_EMA'] = df['Delta_EMA'].replace([np.inf, -np.inf], 0).fillna(0)
df['Delta_EMA_Abs'] = df['Delta_EMA_Abs'].replace([np.inf, -np.inf], 0).fillna(0)
df['Delta_Tension'] = df['Delta_Tension'].replace([np.inf, -np.inf], 0).fillna(0)

print("\nDelta Statistics:")
print(f"  Delta_Vol: mean={df['Delta_Vol'].mean():.4f}, std={df['Delta_Vol'].std():.4f}")
print(f"  Delta_EMA_Abs: mean={df['Delta_EMA_Abs'].mean():.4f}, std={df['Delta_EMA_Abs'].std():.4f}")
print(f"  Delta_Tension: mean={df['Delta_Tension'].mean():.4f}, std={df['Delta_Tension'].std():.4f}")

# ============================================================================
# V8.0 SCORING MODEL
# ============================================================================
print("\n" + "="*120)
print("V8.0 SCORING MODEL - Composite Dynamic Score")
print("="*120)

print("""
【评分模型公式】

V8_Score = 0.5 × Score_EMA_Delta + 0.3 × Score_Vol_Delta + 0.2 × Score_Base_Vol

权重解释:
- EMA突变 (50%): 价格急跌/急涨是最重要的触发信号
- 量能突变 (30%): 确认价格变化有真实资金支持
- 基础量能 (20%): 防止小量能的假突破
""")

# Normalization function (clipping to cap)
def normalize_clipped(series, cap):
    """Normalize series to 0-1 range by clipping at cap"""
    return series.abs().clip(upper=cap) / cap

# Component 1: EMA Delta (Most Important - 50% weight)
score_ema_delta = normalize_clipped(df['Delta_EMA_Abs'], cap=1.0) * 0.5

# Component 2: Volume Delta (30% weight)
score_vol_delta = normalize_clipped(df['Delta_Vol'], cap=0.5) * 0.3

# Component 3: Base Volume (20% weight) - Prevents fake breakouts
# Volume > 1.2 gets full score
score_base_vol = normalize_clipped(df['量能比率'], cap=1.5) * 0.2

# Total Score
df['V8_Score'] = score_ema_delta + score_vol_delta + score_base_vol

print(f"\nV8_Score Distribution:")
print(f"  Mean: {df['V8_Score'].mean():.4f}")
print(f"  Std: {df['V8_Score'].std():.4f}")
print(f"  Max: {df['V8_Score'].max():.4f}")
print(f"  Min: {df['V8_Score'].min():.4f}")

# ============================================================================
# V8.0 DECISION LOGIC
# ============================================================================
print("\n" + "="*120)
print("V8.0 DECISION LOGIC")
print("="*120)

# Threshold optimization
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
print("\nThreshold Optimization:")
print(f"{'Threshold':<12} {'Predicted':<12} {'Coverage':<12}")
print("-"*50)

for thresh in thresholds:
    df['V8_Action_Pred'] = (df['V8_Score'] >= thresh).astype(int)
    coverage = (df['V8_Action_Pred'] == 1).sum()
    print(f"{thresh:<12.2f} {coverage:<12} {coverage/len(df)*100:.1f}%")

# Select optimal threshold (balance precision and recall)
THRESHOLD = 0.50  # Based on optimization

print(f"\n[SELECTED] Threshold = {THRESHOLD}")

def v8_strategy(row):
    # Rule 1: Fuse mechanism - Oscillation with low volume
    if row['信号类型'] == 'OSCILLATION' and row['量能比率'] < 1.0:
        return 'HOLD'

    # Rule 2: Scoring trigger
    if row['V8_Score'] >= THRESHOLD:
        return 'ACTION'
    else:
        return 'HOLD'

df['V8_Action'] = df.apply(v8_strategy, axis=1)

print(f"\nV8.0 Predictions:")
print(f"  ACTION: {(df['V8_Action']=='ACTION').sum()} ({(df['V8_Action']=='ACTION').sum()/len(df)*100:.1f}%)")
print(f"  HOLD: {(df['V8_Action']=='HOLD').sum()} ({(df['V8_Action']=='HOLD').sum()/len(df)*100:.1f}%)")

# ============================================================================
# GROUND TRUTH EXTRACTION
# ============================================================================
print("\n" + "="*120)
print("GROUND TRUTH - Gold Signal Annotation")
print("="*120)

# Extract ideal actions from gold signals
df['Ideal_Action'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'UNKNOWN'
)

df_valid = df[df['Ideal_Action'].isin(['ACTION', 'HOLD'])].copy()

print(f"\nValid ground truth samples: {len(df_valid)}")
print(f"  ACTION (Ideal): {(df_valid['Ideal_Action']=='ACTION').sum()}")
print(f"  HOLD (Ideal): {(df_valid['Ideal_Action']=='HOLD').sum()}")

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "="*120)
print("PERFORMANCE EVALUATION - V8.0 vs Ideal Actions")
print("="*120)

# Binary encoding
y_true = (df_valid['Ideal_Action'] == 'ACTION').astype(int)
y_pred_v8 = (df_valid['V8_Action'] == 'ACTION').astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_v8)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"                  Predicted")
print(f"              HOLD    ACTION")
print(f"Actual HOLD   {tn:4d}    {fp:4d}")
print(f"       ACTION {fn:4d}    {tp:4d}")

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\nClassification Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ============================================================================
# COMPARISON WITH V7.0.5
# ============================================================================
print("\n" + "="*120)
print("COMPARISON - V8.0 vs V7.0.5")
print("="*120)

# V7.0.5 performance
v705_pass = df_valid['V7.0.5通过'].isin(['TRUE', True])
v705_action = df_valid.loc[v705_pass, 'Ideal_Action'] == 'ACTION'

# V7.0.5 metrics (treats all passed signals as ACTION)
tp_v7 = v705_action.sum()
fp_v7 = (~v705_action).sum()
fn_v7 = ((~v705_pass) & (df_valid['Ideal_Action'] == 'ACTION')).sum()
tn_v7 = ((~v705_pass) & (df_valid['Ideal_Action'] == 'HOLD')).sum()

precision_v7 = tp_v7 / (tp_v7 + fp_v7) if (tp_v7 + fp_v7) > 0 else 0
recall_v7 = tp_v7 / (tp_v7 + fn_v7) if (tp_v7 + fn_v7) > 0 else 0
f1_v7 = 2 * precision_v7 * recall_v7 / (precision_v7 + recall_v7) if (precision_v7 + recall_v7) > 0 else 0

print(f"\n{'Metric':<15} {'V7.0.5':<12} {'V8.0':<12} {'Improvement':<12}")
print("-"*60)
print(f"{'Precision':<15} {precision_v7:<12.4f} {precision:<12.4f} {precision-precision_v7:+.4f}")
print(f"{'Recall':<15} {recall_v7:<12.4f} {recall:<12.4f} {recall-recall_v7:+.4f}")
print(f"{'F1-Score':<15} {f1_v7:<12.4f} {f1:<12.4f} {f1-f1_v7:+.4f}")
print(f"{'Coverage':<15} {v705_pass.sum():<12} {y_pred_v8.sum():<12} {y_pred_v8.sum()-v705_pass.sum():+d}")

# ============================================================================
# DETAILED ANALYSIS: True Positives
# ============================================================================
print("\n" + "="*120)
print("DETAILED ANALYSIS - True Positives (V8.0 Correct)")
print("="*120)

tp_data = df_valid[(y_pred_v8 == 1) & (y_true == 1)]

if len(tp_data) > 0:
    print(f"\nTrue Positives: n={len(tp_data)}")
    print(f"  Avg V8_Score: {tp_data['V8_Score'].mean():.4f}")
    print(f"  Avg Volume: {tp_data['量能比率'].mean():.4f}")
    print(f"  Avg Price vs EMA: {tp_data['价格vsEMA%'].mean():.4f}")
    print(f"  Avg Delta_EMA_Abs: {tp_data['Delta_EMA_Abs'].mean():.4f}")
    print(f"  Avg Delta_Vol: {tp_data['Delta_Vol'].mean():.4f}")

# ============================================================================
# DETAILED ANALYSIS: False Negatives (Missed Opportunities)
# ============================================================================
print("\n" + "="*120)
print("DETAILED ANALYSIS - False Negatives (Missed by V8.0)")
print("="*120)

fn_data = df_valid[(y_pred_v8 == 0) & (y_true == 1)]

if len(fn_data) > 0:
    print(f"\nFalse Negatives: n={len(fn_data)}")
    print(f"  Avg V8_Score: {fn_data['V8_Score'].mean():.4f}")
    print(f"  Avg Volume: {fn_data['量能比率'].mean():.4f}")
    print(f"  Avg Price vs EMA: {fn_data['价格vsEMA%'].mean():.4f}")
    print(f"  Avg Delta_EMA_Abs: {fn_data['Delta_EMA_Abs'].mean():.4f}")
    print(f"  Avg Delta_Vol: {fn_data['Delta_Vol'].mean():.4f}")

# ============================================================================
# RESURRECTED SIGNALS: Caught by V8.0 but missed by V7.0.5
# ============================================================================
print("\n" + "="*120)
print("RESURRECTED SIGNALS - Caught by V8.0 but missed by V7.0.5")
print("="*120)

resurrected = df_valid[
    (v705_pass == False) &  # V7.0.5 filtered
    (y_pred_v8 == 1) &  # V8.0 says ACTION
    (y_true == 1)  # Actually ACTION
]

print(f"\nResurrected Gold Signals: n={len(resurrected)}")
if len(resurrected) > 0:
    print(f"  These are GOLD signals that V7.0.5 filtered but V8.0 caught!")
    print(f"\nTop 10 Resurrected Signals:")
    cols = ['时间', '信号类型', '量能比率', '价格vsEMA%', 'Delta_EMA_Abs', 'V8_Score', 'V7.0.5通过', '黄金信号']
    print(resurrected[cols].head(10).to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*120)
print("SAVING RESULTS")
print("="*120)

# Add V8.0 action to original dataframe
output_df = df.copy()
output_df.to_csv('V8_0_strategy_results.csv', index=False, encoding='utf-8-sig')
print(f"\nResults saved to: V8_0_strategy_results.csv")

print("\n" + "="*120)
print("V8.0 STRATEGY VALIDATION COMPLETE")
print("="*120)
print(f"\nFinal Summary:")
print(f"  V8.0 F1-Score: {f1:.4f}")
print(f"  V7.0.5 F1-Score: {f1_v7:.4f}")
print(f"  Improvement: {f1-f1_v7:+.4f}")
print(f"  Resurrected Gold Signals: {len(resurrected)}")
