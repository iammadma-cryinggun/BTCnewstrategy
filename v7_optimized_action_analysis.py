# -*- coding: utf-8 -*-
"""
优化版V7.0.5 - 基于ACTION时刻的真实特征
==========================================

核心发现：
1. 轻度超卖(-2.4%到-1.5%) > 深度超卖(<-2.4%)
2. 量能>1.4是关键阈值
3. 高张力(>1.0)是负信号
4. 需要区分开仓/平仓/反手的不同条件
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

print("="*120)
print("V7.0.5 OPTIMIZED - Based on Real ACTION Moment Characteristics")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 1. EXTRACT ACTION DETAILS - 区分开仓/平仓/反手
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Extract Action Details")
print("="*120)

def classify_action_detail(gold_signal):
    """详细分类黄金信号"""
    if pd.isna(gold_signal):
        return 'NO_ACTION'

    signal_str = str(gold_signal)

    # 开仓信号
    if '开' in signal_str and '平' not in signal_str and '反' not in signal_str:
        if '多' in signal_str:
            return 'OPEN_LONG'
        elif '空' in signal_str:
            return 'OPEN_SHORT'

    # 平仓信号
    elif '平' in signal_str and '反' not in signal_str and '开' not in signal_str:
        if '多' in signal_str:
            return 'CLOSE_LONG'
        elif '空' in signal_str:
            return 'CLOSE_SHORT'

    # 反手信号
    elif '反' in signal_str:
        if '多' in signal_str:
            return 'REVERSE_TO_LONG'
        elif '空' in signal_str:
            return 'REVERSE_TO_SHORT'

    # 继续持有
    elif '继续持' in signal_str or '持仓' in signal_str:
        if '多' in signal_str:
            return 'HOLD_LONG'
        elif '空' in signal_str:
            return 'HOLD_SHORT'

    return 'OTHER'

df['Action_Detail'] = df['黄金信号'].apply(classify_action_detail)

# Count each action type
action_counts = df['Action_Detail'].value_counts()
print("\n手动标注操作统计:")
print(action_counts)

# ============================================================================
# 2. OPTIMIZED V7.0.5 RULES - 基于真实特征
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Optimized V7.0.5 Rules")
print("="*120)

print("""
【优化后的V7.0.5规则】

基于85个ACTION时刻的统计分析：

✅ 有效规则:
1. 量能比率 > 1.4 (ACTION均值1.13 vs HOLD均值0.98)
2. 轻度超卖: -2.4% < 价格vsEMA < -1.5% (31.25% ACTION概率)
3. 避免高张力: 张力 < 1.0 (张力>1.0时ACTION概率0%)

❌ 无效规则:
1. 深度超卖(<-2.4%): ACTION概率仅6.25%
2. 高张力(>1.0): ACTION概率0%
3. DXY燃料: 无显著差异(p=0.899)
""")

def optimized_v7_rules(row):
    """
    优化后的V7.0.5规则

    Returns: (Action_Type, Reason)
    """
    vol = row['量能比率']
    price_vs_ema = row['价格vsEMA%']
    tension = row['张力']
    signal_type = row['信号类型']

    # 规则1: 强制过滤 - OSCILLATION
    if signal_type == 'OSCILLATION':
        # 震荡中，只有极端情况才考虑
        if vol > 2.0 and abs(price_vs_ema) > 3.0:
            return ('ACTION', '震荡-极端突破')
        else:
            return ('HOLD', '震荡-观望')

    # 规则2: 强制过滤 - 高张力
    if abs(tension) > 1.0:
        return ('HOLD', '高张力-避免')

    # 规则3: 核心开仓条件 - 高量能+轻度超卖
    if vol > 1.4:
        # 轻度超卖区间 (最佳区间!)
        if -2.4 < price_vs_ema < -1.5:
            return ('ACTION', '开仓-高量能+轻度超卖')

        # 轻度超买
        if price_vs_ema > 2.0:
            return ('ACTION', '开仓-高量能+轻度超买')

        # 深度超卖 (避免!)
        if price_vs_ema < -2.4:
            return ('HOLD', '深度超卖-避免')

    # 规则4: 中等量能+轻度超卖
    if vol > 1.2:
        if -2.0 < price_vs_ema < -1.0:
            return ('ACTION', '开仓-中量能+轻度超卖')

    # 规则5: 保留原有V7.0.5通过的情况（保守）
    if row['V7.0.5通过'] in ['TRUE', True]:
        # 但要检查是否是负面条件
        if vol < 1.0:
            return ('HOLD', 'V7通过但量能低-降级')
        else:
            return ('ACTION', 'V7原始通过-保留')

    # 默认观望
    return ('HOLD', '默认观望')

# Apply optimized rules
results = df.apply(optimized_v7_rules, axis=1, result_type='expand')
df['Opt_V7_Action'] = results[0]
df['Opt_V7_Reason'] = results[1]

# ============================================================================
# 3. CONVERT TO ACTION/NON-ACTION FOR COMPARISON
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Convert to Binary Classification")
print("="*120)

# Binary classification for comparison
df['Manual_Action'] = df['Action_Detail'].apply(lambda x: 1 if x.startswith(('OPEN', 'CLOSE', 'REVERSE')) else 0)
df['Opt_V7_Binary'] = df['Opt_V7_Action'].apply(lambda x: 1 if x == 'ACTION' else 0)
df['Original_V7_Binary'] = df['V7.0.5通过'].apply(lambda x: 1 if x in ['TRUE', True] else 0)

print("\n操作类型统计:")
print(f"  手动标注ACTION: {df['Manual_Action'].sum()} 次")
print(f"  优化V7预测ACTION: {df['Opt_V7_Binary'].sum()} 次")
print(f"  原始V7预测ACTION: {df['Original_V7_Binary'].sum()} 次")

# ============================================================================
# 4. PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Performance Evaluation - Optimized vs Original V7.0.5")
print("="*120)

# Optimized V7.0.5
y_true = df['Manual_Action']
y_pred_opt = df['Opt_V7_Binary']
y_pred_orig = df['Original_V7_Binary']

# Clean NaN (if any)
valid_idx = df['Manual_Action'].notna()
y_true_clean = y_true[valid_idx]
y_pred_opt_clean = y_pred_opt[valid_idx]
y_pred_orig_clean = y_pred_orig[valid_idx]

# Confusion Matrix - Optimized
cm_opt = confusion_matrix(y_true_clean, y_pred_opt_clean)
tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()

# Confusion Matrix - Original
cm_orig = confusion_matrix(y_true_clean, y_pred_orig_clean)
tn_orig, fp_orig, fn_orig, tp_orig = cm_orig.ravel()

# Metrics
precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
accuracy_opt = (tp_opt + tn_opt) / (tp_opt + tn_opt + fp_opt + fn_opt)

precision_orig = tp_orig / (tp_orig + fp_orig) if (tp_orig + fp_orig) > 0 else 0
recall_orig = tp_orig / (tp_orig + fn_orig) if (tp_orig + fn_orig) > 0 else 0
f1_orig = 2 * precision_orig * recall_orig / (precision_orig + recall_orig) if (precision_orig + recall_orig) > 0 else 0
accuracy_orig = (tp_orig + tn_orig) / (tp_orig + tn_orig + fp_orig + fn_orig)

print(f"\n{'指标':<15} {'原始V7.0.5':<15} {'优化V7.0.5':<15} {'改进':<15}")
print("-"*80)
print(f"{'精确率':<15} {precision_orig:<15.4f} {precision_opt:<15.4f} {precision_opt-precision_orig:+.4f}")
print(f"{'召回率':<15} {recall_orig:<15.4f} {recall_opt:<15.4f} {recall_opt-recall_orig:+.4f}")
print(f"{'F1分数':<15} {f1_orig:<15.4f} {f1_opt:<15.4f} {f1_opt-f1_orig:+.4f}")
print(f"{'准确率':<15} {accuracy_orig:<15.4f} {accuracy_opt:<15.4f} {accuracy_opt-accuracy_orig:+.4f}")
print(f"{'触发次数':<15} {y_pred_orig_clean.sum():<15} {y_pred_opt_clean.sum():<15} {y_pred_opt_clean.sum()-y_pred_orig_clean.sum():+d}")

# ============================================================================
# 5. DETAILED ANALYSIS - True Positives
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Detailed Analysis - True Positives")
print("="*120)

# Optimized V7 True Positives
tp_mask_opt = (y_pred_opt_clean == 1) & (y_true_clean == 1)
tp_data_opt = df[valid_idx][tp_mask_opt]

if len(tp_data_opt) > 0:
    print(f"\n优化V7.0.5 - 正确预测的ACTION ({len(tp_data_opt)}个):")
    print(f"\n触发原因分布:")
    reason_counts = tp_data_opt['Opt_V7_Reason'].value_counts()
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count} 次")

    print(f"\n数据特征:")
    print(f"  平均量能: {tp_data_opt['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {tp_data_opt['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {tp_data_opt['张力'].mean():.4f}")

# ============================================================================
# 6. DETAILED ANALYSIS - False Negatives (Missed Opportunities)
# ============================================================================
print("\n" + "="*120)
print("STEP 6: False Negatives - Missed ACTION Moments")
print("="*120)

fn_mask_opt = (y_pred_opt_clean == 0) & (y_true_clean == 1)
fn_data_opt = df[valid_idx][fn_mask_opt]

if len(fn_data_opt) > 0:
    print(f"\n优化V7.0.5 - 漏掉的ACTION ({len(fn_data_opt)}个):")

    print(f"\n数据特征:")
    print(f"  平均量能: {fn_data_opt['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {fn_data_opt['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {fn_data_opt['张力'].mean():.4f}")

    print(f"\n这些ACTION为何被漏掉?")
    print(f"  量能<1.2: {(fn_data_opt['量能比率'] < 1.2).sum()} 个")
    print(f"  价格vsEMA不在-2.4%到-1.5%: {((fn_data_opt['价格vsEMA%'] >= -1.5) | (fn_data_opt['价格vsEMA%'] <= -2.4)).sum()} 个")
    print(f"  张力>1.0: {(fn_data_opt['张力'] > 1.0).sum()} 个")

# ============================================================================
# 7. ACTION TYPE MATCHING - 开仓/平仓/反手详细对比
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Action Type Matching - Open/Close/Reverse")
print("="*120)

# Filter to only ACTION moments
action_moments = df[df['Manual_Action'] == 1].copy()

if len(action_moments) > 0:
    action_moments['Opt_V7_Predicted'] = action_moments['Opt_V7_Binary'] == 1
    action_moments['Orig_V7_Predicted'] = action_moments['Original_V7_Binary'] == 1

    print(f"\n按操作类型统计:")
    print(f"{'操作类型':<20} {'总数':<10} {'优化V7命中':<15} {'原始V7命中':<15}")
    print("-"*80)

    for action_type in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', 'REVERSE_TO_LONG', 'REVERSE_TO_SHORT']:
        subset = action_moments[action_moments['Action_Detail'] == action_type]
        if len(subset) > 0:
            opt_hit = subset['Opt_V7_Predicted'].sum()
            orig_hit = subset['Orig_V7_Predicted'].sum()
            print(f"{action_type:<20} {len(subset):<10} {opt_hit}/{len(subset)} ({opt_hit/len(subset)*100:.1f}%)      {orig_hit}/{len(subset)} ({orig_hit/len(subset)*100:.1f}%)")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*120)
print("STEP 8: Save Detailed Results")
print("="*120)

output_cols = [
    '时间', '信号类型',
    '量能比率', '价格vsEMA%', '张力', '加速度', 'DXY燃料',
    'V7.0.5通过', 'Opt_V7_Action', 'Opt_V7_Reason',
    'Action_Detail', '黄金信号'
]

df[output_cols].to_csv('V7_Optimized_vs_Manual.csv', index=False, encoding='utf-8-sig')
print(f"\n详细对比结果已保存至: V7_Optimized_vs_Manual.csv")

print("\n" + "="*120)
print("OPTIMIZATION COMPLETE")
print("="*120)
print(f"""
最终总结:

优化V7.0.5性能:
  精确率: {precision_opt:.4f} ({'+' if precision_opt > precision_orig else ''}{precision_opt-precision_orig:.4f})
  召回率: {recall_opt:.4f} ({'+' if recall_opt > recall_orig else ''}{recall_opt-recall_orig:.4f})
  F1分数: {f1_opt:.4f} ({'+' if f1_opt > f1_orig else ''}{f1_opt-f1_orig:.4f})

核心改进:
1. 提高量能阈值: 从无明确阈值 → >1.4
2. 改变价格条件: 从深度超卖 → 轻度超卖(-2.4%到-1.5%)
3. 增加张力过滤: 张力>1.0时避免
4. 保留V7.0.5通过的信号(保守策略)

建议:
- 如果精确率提升: 优化成功，减少误报
- 如果召回率下降: 需要权衡，可能需要降低阈值
""")
