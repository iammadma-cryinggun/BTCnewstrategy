# -*- coding: utf-8 -*-
"""
最终调整版V7.0.5 - 基于真实反手操作特征
========================================

核心发现：
1. 96.5%的操作是REVERSE_LONG（反手多）
2. 漏掉的52个反手多：平均量能0.95，平均价格vsEMA -0.53%
3. 需要降低阈值来提高匹配率
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

print("="*120)
print("FINAL ADJUSTED V7.0.5 - Optimized for REVERSE_LONG Actions")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 1. EXTRACT ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Extract Action Type")
print("="*120)

def classify_action(gold_signal):
    """分类操作类型"""
    if pd.isna(gold_signal):
        return 'HOLD'

    signal_str = str(gold_signal)

    # 反手操作
    if '反手' in signal_str or '反' in signal_str:
        if '多' in signal_str:
            return 'REVERSE_LONG'
        elif '空' in signal_str:
            return 'REVERSE_SHORT'

    # 开仓操作
    if '开' in signal_str:
        return 'OPEN'

    # 平仓操作
    if '平' in signal_str and '反' not in signal_str:
        return 'CLOSE'

    # 持仓
    if '继续持' in signal_str or '持仓' in signal_str:
        return 'HOLD'

    return 'HOLD'

df['Manual_Action'] = df['黄金信号'].apply(classify_action)

print(f"\n操作类型统计:")
action_counts = df['Manual_Action'].value_counts()
for action_type, count in action_counts.items():
    print(f"  {action_type}: {count} 次")

# ============================================================================
# 2. CHARACTERISTICS OF REVERSE_LONG (YOUR MAIN OPERATION)
# ============================================================================
print("\n" + "="*120)
print("STEP 2: REVERSE_LONG Characteristics (Your Main Operation)")
print("="*120)

reverse_long = df[df['Manual_Action'] == 'REVERSE_LONG']
hold = df[df['Manual_Action'] == 'HOLD']

print(f"\n{'参数':<15} {'REVERSE_LONG均值':<20} {'HOLD均值':<20} {'差异':<15}")
print("-"*90)

for param in ['量能比率', '价格vsEMA%', '张力', '加速度']:
    if param in reverse_long.columns:
        rev_mean = reverse_long[param].mean()
        hold_mean = hold[param].mean()
        diff = rev_mean - hold_mean
        print(f"{param:<15} {rev_mean:<20.4f} {hold_mean:<20.4f} {diff:<15.4f}")

# ============================================================================
# 3. THRESHOLD OPTIMIZATION - FIND BEST CUTOFF FOR REVERSE_LONG
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Threshold Optimization for REVERSE_LONG")
print("="*120)

print("\n测试不同量能和价格阈值组合...")

# Test different threshold combinations
vol_thresholds = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
price_thresholds = [-0.3, -0.4, -0.5, -0.6, -0.7]

print(f"\n{'量能阈值':<10} {'价格阈值':<12} {'命中REVERSE':<15} {'误报HOLD':<15} {'精确率':<12} {'召回率':<12}")
print("-"*100)

best_f1 = -1
best_config = None

for vol_th in vol_thresholds:
    for price_th in price_thresholds:
        # Predict REVERSE_LONG
        def predict_reverse(row):
            if row['量能比率'] > vol_th and row['价格vsEMA%'] < price_th:
                return 'REVERSE_LONG'
            return 'HOLD'

        df['Pred'] = df.apply(predict_reverse, axis=1)

        # Calculate metrics
        tp = ((df['Pred'] == 'REVERSE_LONG') & (df['Manual_Action'] == 'REVERSE_LONG')).sum()
        fp = ((df['Pred'] == 'REVERSE_LONG') & (df['Manual_Action'] == 'HOLD')).sum()
        fn = ((df['Pred'] == 'HOLD') & (df['Manual_Action'] == 'REVERSE_LONG')).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{vol_th:<10.2f} {price_th:<12.1f} {tp:<15} {fp:<15} {precision:<12.2%} {recall:<12.2%}")

        if f1 > best_f1:
            best_f1 = f1
            best_config = (vol_th, price_th, precision, recall, f1, tp, fp, fn)

print(f"\n最佳配置:")
print(f"  量能阈值: {best_config[0]}")
print(f"  价格阈值: {best_config[1]}")
print(f"  精确率: {best_config[2]:.2%}")
print(f"  召回率: {best_config[3]:.2%}")
print(f"  F1分数: {best_config[4]:.4f}")

# ============================================================================
# 4. FINAL OPTIMIZED RULES
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Final Optimized Rules")
print("="*120)

OPTIMAL_VOL = best_config[0]
OPTIMAL_PRICE = best_config[1]

print(f"""
【最终优化规则】

基于{len(reverse_long)}个REVERSE_LONG时刻的真实特征：

核心规则（针对REVERSE_LONG）:
IF (量能比率 > {OPTIMAL_VOL}) AND (价格vsEMA < {OPTIMAL_PRICE}%):
    ACTION = 'REVERSE_LONG'

调整说明:
1. 量能阈值: 1.1 → {OPTIMAL_VOL} (降低以覆盖更多反手操作)
2. 价格阈值: -0.3% → {OPTIMAL_PRICE}% (放宽以覆盖轻度超卖)
3. 专门针对REVERSE_LONG优化（您的主要操作）

预期性能:
  精确率: {best_config[2]:.2%}
  召回率: {best_config[3]:.2%}
  F1分数: {best_config[4]:.4f}
""")

def final_optimized_rules(row):
    """
    最终优化的规则

    Returns: (Action, Reason)
    """
    vol = row['量能比率']
    price_vs_ema = row['价格vsEMA%']
    tension = row['张力']
    signal_type = row['信号类型']

    # 规则1: 震荡过滤
    if signal_type == 'OSCILLATION':
        if vol > 1.8 and price_vs_ema < -2.0:
            return ('REVERSE_LONG', '震荡-极端超卖')
        else:
            return ('HOLD', '震荡-观望')

    # 规则2: 高张力过滤
    if abs(tension) > 0.8:
        return ('HOLD', '高张力-避免')

    # 规则3: 核心REVERSE_LONG（优化后）
    if vol > OPTIMAL_VOL and price_vs_ema < OPTIMAL_PRICE:
        return ('REVERSE_LONG', f'反多-量能>{OPTIMAL_VOL}+价格<{OPTIMAL_PRICE}%')

    # 规则4: 辅助条件（保留V7.0.5通过的信号）
    if row['V7.0.5通过'] in ['TRUE', True] and vol > 0.8:
        return ('ACTION', 'V7通过-保留')

    # 默认观望
    return ('HOLD', '默认观望')

# Apply final rules
results = df.apply(final_optimized_rules, axis=1, result_type='expand')
df['Final_Prediction'] = results[0]
df['Final_Reason'] = results[1]

# ============================================================================
# 5. PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Performance Evaluation")
print("="*120)

# Binary classification
y_true = df['Manual_Action'].apply(lambda x: 1 if x == 'REVERSE_LONG' else 0)
y_pred_final = df['Final_Prediction'].apply(lambda x: 1 if x == 'REVERSE_LONG' else 0)
y_pred_orig = df['V7.0.5通过'].apply(lambda x: 1 if x in ['TRUE', True] else 0)

# Confusion Matrix - Final
cm_final = confusion_matrix(y_true, y_pred_final)
tn_f, fp_f, fn_f, tp_f = cm_final.ravel()

# Confusion Matrix - Original
cm_orig = confusion_matrix(y_true, y_pred_orig)
tn_o, fp_o, fn_o, tp_o = cm_orig.ravel()

# Metrics
precision_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
recall_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
f1_f = 2 * precision_f * recall_f / (precision_f + recall_f) if (precision_f + recall_f) > 0 else 0

precision_o = tp_o / (tp_o + fp_o) if (tp_o + fp_o) > 0 else 0
recall_o = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0
f1_o = 2 * precision_o * recall_o / (precision_o + recall_o) if (precision_o + recall_o) > 0 else 0

print(f"\n{'指标':<15} {'原始V7.0.5':<15} {'最终优化版':<15} {'改进':<15}")
print("-"*80)
print(f"{'精确率':<15} {precision_o:<15.4f} {precision_f:<15.4f} {precision_f-precision_o:+.4f}")
print(f"{'召回率':<15} {recall_o:<15.4f} {recall_f:<15.4f} {recall_f-recall_o:+.4f}")
print(f"{'F1分数':<15} {f1_o:<15.4f} {f1_f:<15.4f} {f1_f-f1_o:+.4f}")
print(f"{'触发次数':<15} {y_pred_orig.sum():<15} {y_pred_final.sum():<15} {y_pred_final.sum()-y_pred_orig.sum():+d}")

# ============================================================================
# 6. DETAILED ANALYSIS - MATCHED vs MISSED
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Detailed Analysis - Matched vs Missed REVERSE_LONG")
print("="*120)

# True Positives: Correctly predicted REVERSE_LONG
tp_data = df[(y_pred_final == 1) & (y_true == 1)]
print(f"\n正确预测的REVERSE_LONG: {len(tp_data)}个")
if len(tp_data) > 0:
    print(f"  平均量能: {tp_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {tp_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {tp_data['张力'].mean():.4f}")

# False Negatives: Missed REVERSE_LONG
fn_data = df[(y_pred_final == 0) & (y_true == 1)]
print(f"\n漏掉的REVERSE_LONG: {len(fn_data)}个")
if len(fn_data) > 0:
    print(f"  平均量能: {fn_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {fn_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {fn_data['张力'].mean():.4f}")

    # Why missed?
    low_vol = (fn_data['量能比率'] <= OPTIMAL_VOL).sum()
    price_high = (fn_data['价格vsEMA%'] >= OPTIMAL_PRICE).sum()
    print(f"\n  漏掉原因:")
    print(f"    量能≤{OPTIMAL_VOL}: {low_vol} ({low_vol/len(fn_data)*100:.1f}%)")
    print(f"    价格≥{OPTIMAL_PRICE}%: {price_high} ({price_high/len(fn_data)*100:.1f}%)")

# False Positives: Predicted REVERSE_LONG but HOLD
fp_data = df[(y_pred_final == 1) & (y_true == 0)]
print(f"\n误报(HOLD预测为REVERSE_LONG): {len(fp_data)}个")
if len(fp_data) > 0:
    print(f"  平均量能: {fp_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {fp_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {fp_data['张力'].mean():.4f}")

# ============================================================================
# 7. SAVE FINAL RESULTS
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Save Final Results")
print("="*120)

output_cols = [
    '时间', '信号类型',
    '量能比率', '价格vsEMA%', '张力',
    'Manual_Action', 'Final_Prediction', 'Final_Reason',
    'V7.0.5通过', '黄金信号'
]

df[output_cols].to_csv('Final_Optimized_V7.csv', index=False, encoding='utf-8-sig')
print(f"\n最终优化结果已保存至: Final_Optimized_V7.csv")

print("\n" + "="*120)
print("OPTIMIZATION COMPLETE")
print("="*120)

print(f"""
最终总结:

优化后的V7.0.5规则:
  量能阈值: {OPTIMAL_VOL} (从1.1降低)
  价格阈值: {OPTIMAL_PRICE}% (从-0.3%放宽)
  专门针对: REVERSE_LONG操作（您的主要操作）

性能对比:
  F1分数: {f1_f:.4f} (vs 原始 {f1_o:.4f}, 变化: {f1_f-f1_o:+.4f})
  精确率: {precision_f:.4f} (vs 原始 {precision_o:.4f}, 变化: {precision_f-precision_o:+.4f})
  召回率: {recall_f:.4f} (vs 原始 {recall_o:.4f}, 变化: {recall_f-recall_o:+.4f})

改进:
  正确预测: {tp_f}个REVERSE_LONG操作
  漏掉: {fn_f}个REVERSE_LONG操作
  误报: {fp_f}个HOLD误判为REVERSE_LONG

建议:
  如果需要更高召回率: 可以进一步降低量能阈值到0.90
  如果需要更高精确率: 可以提高量能阈值到1.05
  当前配置: 平衡召回率和精确率
""")
