# -*- coding: utf-8 -*-
"""
修正版V7.0.5 - 基于真实的反手操作特征
======================================

真实发现：
1. 96.5%的操作是反手（不是开仓或平仓）
2. 反手时刻量能均值1.14（不是1.4！）
3. 反手时刻价格vsEMA均值-0.44%（不在-2.4到-1.5区间！）
4. 几乎没有单独的平仓操作
"""

import pandas as pd
import numpy as np

print("="*120)
print("CORRECTED V7.0.5 - Based on Real REVERSE Action Characteristics")
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
    """简化分类：只分ACTION vs HOLD"""
    if pd.isna(gold_signal):
        return 'HOLD'

    signal_str = str(gold_signal)

    # 任何涉及操作的都是ACTION
    if any(k in signal_str for k in ['反', '开']):
        return 'ACTION'
    elif '继续持' in signal_str or '持仓' in signal_str:
        return 'HOLD'
    else:
        return 'HOLD'

df['Manual_Action'] = df['黄金信号'].apply(classify_action)

print(f"\n手动标注统计:")
print(f"  ACTION: {(df['Manual_Action']=='ACTION').sum()} 次")
print(f"  HOLD: {(df['Manual_Action']=='HOLD').sum()} 次")

# ============================================================================
# 2. WHAT DO REVERSE MOMENTS LOOK LIKE?
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Reverse Moments Characteristics")
print("="*120)

reverse_moments = df[df['Manual_Action'] == 'ACTION'].copy()
hold_moments = df[df['Manual_Action'] == 'HOLD'].copy()

print(f"\n{'参数':<15} {'ACTION均值':<15} {'HOLD均值':<15} {'差异':<12} {'提升%':<10}")
print("-"*90)

for param in ['量能比率', '价格vsEMA%', '张力', '加速度']:
    action_mean = reverse_moments[param].mean()
    hold_mean = hold_moments[param].mean()
    diff = action_mean - hold_mean
    lift_pct = (action_mean / hold_mean - 1) * 100 if hold_mean != 0 else 0

    print(f"{param:<15} {action_mean:<15.4f} {hold_mean:<15.4f} {diff:<12.4f} {lift_pct:+.1f}%")

# ============================================================================
# 3. CORRECTED V7.0.5 RULES - 降低阈值
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Corrected V7.0.5 Rules")
print("="*120)

print("""
[Corrected V7.0.5 Rules]

Based on 85 ACTION moments:

[Adjusted Thresholds]
1. Volume Ratio: > 1.1 (lowered from >1.4, ACTION mean=1.13)
2. Price vs EMA: < -0.3% (raised from <-1.5%, ACTION mean=-0.44%)
3. Avoid High Tension: Tension < 0.5

[Failed Rules]
1. Volume > 1.4 (too high)
2. Price vs EMA < -1.5% (too strict)
3. Tension > 1.0 (ACTION moments have low tension)
""")

def corrected_v7_rules(row):
    """
    修正后的V7.0.5规则

    Returns: (Action, Reason)
    """
    vol = row['量能比率']
    price_vs_ema = row['价格vsEMA%']
    tension = row['张力']
    signal_type = row['信号类型']

    # 规则1: 强制过滤 - OSCILLATION
    if signal_type == 'OSCILLATION':
        # 震荡中只有极端情况才考虑
        if vol > 1.8 and price_vs_ema < -2.0:
            return ('ACTION', '震荡-深度超卖+放量')
        else:
            return ('HOLD', '震荡-观望')

    # 规则2: 避免高张力
    if abs(tension) > 0.8:
        return ('HOLD', '高张力-避免')

    # 规则3: 核心条件 - 基于真实特征
    # 量能>1.1 AND 价格轻度超卖(<-0.3%)
    if vol > 1.1 and price_vs_ema < -0.3:
        return ('ACTION', '反多-量能>1.1+轻度超卖')

    # 规则4: 量能稍低但价格更超卖
    if vol > 0.9 and price_vs_ema < -1.0:
        return ('ACTION', '反多-量能>0.9+超卖')

    # 规则5: 保留原V7.0.5通过（但要检查）
    if row['V7.0.5通过'] in ['TRUE', True]:
        if vol > 0.8:  # 量能不能太低
            return ('ACTION', 'V7通过-保留')

    # 默认观望
    return ('HOLD', '默认观望')

# Apply corrected rules
results = df.apply(corrected_v7_rules, axis=1, result_type='expand')
df['Corr_V7_Action'] = results[0]
df['Corr_V7_Reason'] = results[1]

# ============================================================================
# 4. PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Performance Evaluation")
print("="*120)

from sklearn.metrics import confusion_matrix

# Binary
y_true = df['Manual_Action'].apply(lambda x: 1 if x == 'ACTION' else 0)
y_pred_corr = df['Corr_V7_Action'].apply(lambda x: 1 if x == 'ACTION' else 0)
y_pred_orig = df['V7.0.5通过'].apply(lambda x: 1 if x in ['TRUE', True] else 0)

# Confusion Matrix
cm_corr = confusion_matrix(y_true, y_pred_corr)
tn_c, fp_c, fn_c, tp_c = cm_corr.ravel()

cm_orig = confusion_matrix(y_true, y_pred_orig)
tn_o, fp_o, fn_o, tp_o = cm_orig.ravel()

# Metrics
precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
f1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0

precision_o = tp_o / (tp_o + fp_o) if (tp_o + fp_o) > 0 else 0
recall_o = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0
f1_o = 2 * precision_o * recall_o / (precision_o + recall_o) if (precision_o + recall_o) > 0 else 0

print(f"\n{'指标':<15} {'原始V7.0.5':<15} {'修正V7.0.5':<15} {'改进':<15}")
print("-"*75)
print(f"{'精确率':<15} {precision_o:<15.4f} {precision_c:<15.4f} {precision_c-precision_o:+.4f}")
print(f"{'召回率':<15} {recall_o:<15.4f} {recall_c:<15.4f} {recall_c-recall_o:+.4f}")
print(f"{'F1分数':<15} {f1_o:<15.4f} {f1_c:<15.4f} {f1_c-f1_o:+.4f}")
print(f"{'触发次数':<15} {y_pred_orig.sum():<15} {y_pred_corr.sum():<15} {y_pred_corr.sum()-y_pred_orig.sum():+d}")

# ============================================================================
# 5. DETAILED ANALYSIS - True Positives
# ============================================================================
print("\n" + "="*120)
print("STEP 5: True Positives Analysis")
print("="*120)

tp_mask = (y_pred_corr == 1) & (y_true == 1)
tp_data = df[tp_mask]

if len(tp_data) > 0:
    print(f"\n修正V7.0.5 - 正确预测的ACTION ({len(tp_data)}个):")
    print(f"\n触发原因分布:")
    reason_counts = tp_data['Corr_V7_Reason'].value_counts()
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count} 次 ({count/len(tp_data)*100:.1f}%)")

    print(f"\n数据特征:")
    print(f"  平均量能: {tp_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {tp_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {tp_data['张力'].mean():.4f}")

# ============================================================================
# 6. FALSE NEGATIVES - 为何漏掉？
# ============================================================================
print("\n" + "="*120)
print("STEP 6: False Negatives - Why Did We Miss?")
print("="*120)

fn_mask = (y_pred_corr == 0) & (y_true == 1)
fn_data = df[fn_mask]

if len(fn_data) > 0:
    print(f"\n修正V7.0.5 - 漏掉的ACTION ({len(fn_data)}个):")

    print(f"\n数据特征:")
    print(f"  平均量能: {fn_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {fn_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {fn_data['张力'].mean():.4f}")

    print(f"\n为何被漏掉?")
    low_vol = (fn_data['量能比率'] < 1.1).sum()
    price_not_oversold = (fn_data['价格vsEMA%'] >= -0.3).sum()
    high_tension = (fn_data['张力'] > 0.8).sum()

    print(f"  量能<1.1: {low_vol} ({low_vol/len(fn_data)*100:.1f}%)")
    print(f"  价格vsEMA≥-0.3%: {price_not_oversold} ({price_not_oversold/len(fn_data)*100:.1f}%)")
    print(f"  张力>0.8: {high_tension} ({high_tension/len(fn_data)*100:.1f}%)")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Save Results")
print("="*120)

output_cols = [
    '时间', '信号类型',
    '量能比率', '价格vsEMA%', '张力',
    'V7.0.5通过', 'Corr_V7_Action', 'Corr_V7_Reason',
    'Manual_Action', '黄金信号'
]

df[output_cols].to_csv('V7_Corrected_vs_Manual.csv', index=False, encoding='utf-8-sig')
print(f"\n修正版对比结果已保存至: V7_Corrected_vs_Manual.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
print(f"""
总结:

修正V7.0.5基于真实ACTION时刻特征:
1. 降低量能阈值: 1.4 → 1.1
2. 放宽价格条件: <-1.5% → <-0.3%
3. 降低张力过滤: >1.0 → >0.8

性能:
  精确率: {precision_c:.4f}
  召回率: {recall_c:.4f}
  F1分数: {f1_c:.4f}

与原始V7.0.5对比:
  F1分数变化: {f1_c-f1_o:+.4f}
""")
