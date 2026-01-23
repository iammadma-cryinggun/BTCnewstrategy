# -*- coding: utf-8 -*-
"""
状态跟踪版V7.0.5优化 - 区分开仓/平仓/反手
=======================================

关键洞察：
1. 开仓：从无仓位 → 有仓位
2. 平仓：从有仓位 → 无仓位
3. 反手：从多→空 或 从空→多（状态翻转）

需要跟踪持仓状态来预测不同操作！
"""

import pandas as pd
import numpy as np

print("="*120)
print("STATE-BASED V7.0.5 OPTIMIZATION - Open/Close/Reverse Analysis")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 1. EXTRACT CURRENT POSITION STATE - 追踪持仓状态
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Track Position State")
print("="*120)

def classify_action_detail(gold_signal):
    """详细分类黄金信号"""
    if pd.isna(gold_signal):
        return 'NO_ACTION'

    signal_str = str(gold_signal)

    # 反手信号（最高优先级）
    if '反手' in signal_str or '反' in signal_str:
        if '多' in signal_str:
            return 'REVERSE_TO_LONG'
        elif '空' in signal_str:
            return 'REVERSE_TO_SHORT'

    # 开仓信号
    elif '开' in signal_str:
        if '多' in signal_str:
            return 'OPEN_LONG'
        elif '空' in signal_str:
            return 'OPEN_SHORT'

    # 平仓信号
    elif '平' in signal_str:
        if '多' in signal_str:
            return 'CLOSE_LONG'
        elif '空' in signal_str:
            return 'CLOSE_SHORT'

    # 继续持有
    elif '继续持' in signal_str or '持仓' in signal_str:
        if '多' in signal_str:
            return 'HOLD_LONG'
        elif '空' in signal_str:
            return 'HOLD_SHORT'

    return 'OTHER'

df['Action_Detail'] = df['黄金信号'].apply(classify_action_detail)

# Track position state
position_state = []  # 'NONE', 'LONG', 'SHORT'
current_state = 'NONE'

for _, row in df.iterrows():
    action = row['Action_Detail']

    if action in ['OPEN_LONG', 'REVERSE_TO_LONG']:
        current_state = 'LONG'
    elif action in ['OPEN_SHORT', 'REVERSE_TO_SHORT']:
        current_state = 'SHORT'
    elif action in ['CLOSE_LONG', 'CLOSE_SHORT']:
        current_state = 'NONE'
    # HOLD_* 不改变状态

    position_state.append(current_state)

df['Position_State'] = position_state

# Count action types
action_counts = df['Action_Detail'].value_counts()
print("\n手动标注操作统计:")
for action_type, count in action_counts.items():
    print(f"  {action_type}: {count} 次")

# ============================================================================
# 2. ANALYZE CHARACTERISTICS BY ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Analyze Characteristics by Action Type")
print("="*120)

action_types_to_analyze = [
    'OPEN_LONG', 'OPEN_SHORT',
    'CLOSE_LONG', 'CLOSE_SHORT',
    'REVERSE_TO_LONG', 'REVERSE_TO_SHORT'
]

print(f"\n{'操作类型':<20} {'数量':<8} {'平均量能':<12} {'平均价格vsEMA':<15} {'平均张力':<12}")
print("-"*100)

for action_type in action_types_to_analyze:
    subset = df[df['Action_Detail'] == action_type]
    if len(subset) > 0:
        avg_vol = subset['量能比率'].mean()
        avg_price = subset['价格vsEMA%'].mean()
        avg_tension = subset['张力'].mean()
        print(f"{action_type:<20} {len(subset):<8} {avg_vol:<12.4f} {avg_price:<15.4f} {avg_tension:<12.4f}")

# ============================================================================
# 3. STATE-BASED PREDICTION RULES
# ============================================================================
print("\n" + "="*120)
print("STEP 3: State-Based Prediction Rules")
print("="*120)

print("""
【基于状态的预测规则】

1. 当前状态: NONE
   → 预测操作类型: OPEN_LONG 或 OPEN_SHORT 或 HOLD

2. 当前状态: LONG
   → 预测操作类型: CLOSE_LONG 或 REVERSE_TO_SHORT 或 HOLD_LONG

3. 当前状态: SHORT
   → 预测操作类型: CLOSE_SHORT 或 REVERSE_TO_LONG 或 HOLD_SHORT
""")

def predict_action_based_on_state(row, current_position_state):
    """
    基于当前状态预测下一个操作

    row: 当前行数据
    current_position_state: 'NONE', 'LONG', 'SHORT'

    Returns: (Predicted_Action, Reason)
    """
    vol = row['量能比率']
    price_vs_ema = row['价格vsEMA%']
    tension = row['张力']
    signal_type = row['信号类型']

    # 强制过滤 - 震荡
    if signal_type == 'OSCILLATION':
        if vol > 2.0 and abs(price_vs_ema) > 3.0:
            # 极端突破，考虑反手
            if current_position_state == 'LONG' and price_vs_ema < -2.0:
                return ('REVERSE_TO_SHORT', '震荡突破-反手空')
            elif current_position_state == 'SHORT' and price_vs_ema > 2.0:
                return ('REVERSE_TO_LONG', '震荡突破-反手多')
        return ('HOLD', '震荡-观望')

    # 强制过滤 - 高张力
    if abs(tension) > 1.0:
        return ('HOLD', '高张力-避免操作')

    # === 当前状态: NONE (考虑开仓) ===
    if current_position_state == 'NONE':
        if vol > 1.4:
            # 轻度超卖 → 开多
            if -2.4 < price_vs_ema < -1.5:
                return ('OPEN_LONG', '开多-高量能+轻度超卖')

            # 轻度超买 → 开空
            if price_vs_ema > 2.0:
                return ('OPEN_SHORT', '开空-高量能+轻度超买')

        # 中等量能
        if vol > 1.2:
            if -2.0 < price_vs_ema < -1.0:
                return ('OPEN_LONG', '开多-中量能+轻度超卖')

        return ('HOLD', '观望-无明确信号')

    # === 当前状态: LONG (考虑平仓或反手) ===
    elif current_position_state == 'LONG':
        # 平多条件: 价格反弹到EMA以上或超买
        if price_vs_ema > 1.5:
            if vol > 1.2:
                return ('CLOSE_LONG', '平多-价格反弹+量能')
            else:
                return ('HOLD_LONG', '持多-价格反弹但量能低')

        # 反手空条件: 价格大幅下跌+放量
        if vol > 1.4 and price_vs_ema < -2.0:
            return ('REVERSE_TO_SHORT', '反手空-价格急跌+放量')

        # 继续持多
        return ('HOLD_LONG', '持多-趋势未变')

    # === 当前状态: SHORT (考虑平仓或反手) ===
    elif current_position_state == 'SHORT':
        # 平空条件: 价格回落到EMA以下或超卖
        if price_vs_ema < -1.5:
            if vol > 1.2:
                return ('CLOSE_SHORT', '平空-价格回落+量能')
            else:
                return ('HOLD_SHORT', '持空-价格回落但量能低')

        # 反手多条件: 价格大幅上涨+放量
        if vol > 1.4 and price_vs_ema > 2.0:
            return ('REVERSE_TO_LONG', '反手多-价格急涨+放量')

        # 继续持空
        return ('HOLD_SHORT', '持空-趋势未变')

    # Default
    return ('HOLD', '默认观望')

# Apply state-based prediction
predicted_actions = []
predicted_reasons = []

current_state = 'NONE'

for idx, row in df.iterrows():
    pred_action, pred_reason = predict_action_based_on_state(row, current_state)
    predicted_actions.append(pred_action)
    predicted_reasons.append(pred_reason)

    # Update state based on manual annotation (for next prediction)
    actual_action = row['Action_Detail']
    if actual_action in ['OPEN_LONG', 'REVERSE_TO_LONG']:
        current_state = 'LONG'
    elif actual_action in ['OPEN_SHORT', 'REVERSE_TO_SHORT']:
        current_state = 'SHORT'
    elif actual_action in ['CLOSE_LONG', 'CLOSE_SHORT']:
        current_state = 'NONE'

df['Predicted_Action'] = predicted_actions
df['Predicted_Reason'] = predicted_reasons

# ============================================================================
# 4. PERFORMANCE EVALUATION - BY ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Performance Evaluation by Action Type")
print("="*120)

# Filter to moments where manual action is not 'OTHER' or 'NO_ACTION'
valid_actions = df[df['Action_Detail'].isin(action_types_to_analyze + ['HOLD_LONG', 'HOLD_SHORT'])].copy()

if len(valid_actions) > 0:
    print(f"\n有效操作时刻: {len(valid_actions)} 个")

    # Match prediction
    valid_actions['Prediction_Match'] = (valid_actions['Predicted_Action'] == valid_actions['Action_Detail'])

    print(f"\n{'操作类型':<20} {'总数':<8} {'预测正确':<10} {'准确率':<12}")
    print("-"*70)

    for action_type in action_types_to_analyze + ['HOLD_LONG', 'HOLD_SHORT']:
        subset = valid_actions[valid_actions['Action_Detail'] == action_type]
        if len(subset) > 0:
            correct = subset['Prediction_Match'].sum()
            accuracy = correct / len(subset)
            print(f"{action_type:<20} {len(subset):<8} {correct:<10} {accuracy:<12.2%}")

    # Overall accuracy
    overall_accuracy = valid_actions['Prediction_Match'].sum() / len(valid_actions)
    print(f"\n整体准确率: {overall_accuracy:.2%}")

# ============================================================================
# 5. DETAILED ANALYSIS - OPEN vs CLOSE vs REVERSE
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Detailed Analysis - Open vs Close vs Reverse")
print("="*120)

# 开仓 vs 平仓 vs 反手
open_actions = df[df['Action_Detail'].isin(['OPEN_LONG', 'OPEN_SHORT'])]
close_actions = df[df['Action_Detail'].isin(['CLOSE_LONG', 'CLOSE_SHORT'])]
reverse_actions = df[df['Action_Detail'].isin(['REVERSE_TO_LONG', 'REVERSE_TO_SHORT'])]

print(f"\n开仓操作 ({len(open_actions)}个):")
if len(open_actions) > 0:
    print(f"  量能: 均值={open_actions['量能比率'].mean():.4f}, 中位数={open_actions['量能比率'].median():.4f}")
    print(f"  价格vsEMA: 均值={open_actions['价格vsEMA%'].mean():.4f}%, 中位数={open_actions['价格vsEMA%'].median():.4f}%")
    print(f"  张力: 均值={open_actions['张力'].mean():.4f}")

print(f"\n平仓操作 ({len(close_actions)}个):")
if len(close_actions) > 0:
    print(f"  量能: 均值={close_actions['量能比率'].mean():.4f}, 中位数={close_actions['量能比率'].median():.4f}")
    print(f"  价格vsEMA: 均值={close_actions['价格vsEMA%'].mean():.4f}%, 中位数={close_actions['价格vsEMA%'].median():.4f}%")
    print(f"  张力: 均值={close_actions['张力'].mean():.4f}")

print(f"\n反手操作 ({len(reverse_actions)}个):")
if len(reverse_actions) > 0:
    print(f"  量能: 均值={reverse_actions['量能比率'].mean():.4f}, 中位数={reverse_actions['量能比率'].median():.4f}")
    print(f"  价格vsEMA: 均值={reverse_actions['价格vsEMA%'].mean():.4f}%, 中位数={reverse_actions['价格vsEMA%'].median():.4f}%")
    print(f"  张力: 均值={reverse_actions['张力'].mean():.4f}")

# ============================================================================
# 6. CONFUSION MATRIX - FOR ACTION PREDICTION
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Confusion Matrix - Action Prediction")
print("="*120)

# Binary classification: ACTION (any action) vs HOLD
y_true = df['Action_Detail'].apply(lambda x: 1 if x in action_types_to_analyze else 0)
y_pred = df['Predicted_Action'].apply(lambda x: 1 if x in action_types_to_analyze else 0)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'预测':^15} {'实际HOLD':^15} {'实际ACTION':^15}")
print("-"*50)
print(f"{'HOLD':^15} {tn:^15} {fp:^15}")
print(f"{'ACTION':^15} {fn:^15} {tp:^15}")

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n性能指标:")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Save Results")
print("="*120)

output_cols = [
    '时间', '信号类型', 'Position_State',
    '量能比率', '价格vsEMA%', '张力',
    'Action_Detail', 'Predicted_Action', 'Predicted_Reason',
    '黄金信号'
]

df[output_cols].to_csv('State_Based_Prediction_Results.csv', index=False, encoding='utf-8-sig')
print(f"\n状态跟踪预测结果已保存至: State_Based_Prediction_Results.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
