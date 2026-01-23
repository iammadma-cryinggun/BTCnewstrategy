# -*- coding: utf-8 -*-
"""
详细操作类型匹配分析
====================

分类:
1. 开多/开空
2. 平多/平空
3. 反手多/反手空（类似开仓）
4. 持多/持空（HOLD）
"""

import pandas as pd
import numpy as np

print("="*120)
print("DETAILED ACTION TYPE MATCHING ANALYSIS")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 1. DETAILED ACTION CLASSIFICATION
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Detailed Action Classification")
print("="*120)

def classify_detailed_action(gold_signal):
    """详细分类操作类型"""
    if pd.isna(gold_signal):
        return 'NO_SIGNAL'

    signal_str = str(gold_signal)

    # 反手操作（最高优先级，因为包含"平"和"开"）
    if '反手' in signal_str or '反' in signal_str:
        if '多' in signal_str:
            return 'REVERSE_LONG'  # 反手多
        elif '空' in signal_str:
            return 'REVERSE_SHORT'  # 反手空

    # 开仓操作
    if '开' in signal_str and '平' not in signal_str:
        if '多' in signal_str:
            return 'OPEN_LONG'
        elif '空' in signal_str:
            return 'OPEN_SHORT'

    # 平仓操作
    if '平' in signal_str and '反' not in signal_str and '开' not in signal_str:
        if '多' in signal_str or '空' in signal_str:
            return 'CLOSE'  # 平仓（不区分多空）

    # 持仓操作
    if '继续持' in signal_str or '持仓' in signal_str:
        if '多' in signal_str:
            return 'HOLD_LONG'
        elif '空' in signal_str:
            return 'HOLD_SHORT'

    return 'OTHER'

df['Action_Detail'] = df['黄金信号'].apply(classify_detailed_action)

# Count each type
action_counts = df['Action_Detail'].value_counts()
print("\n手动标注操作类型统计:")
print(f"{'操作类型':<20} {'数量':<10} {'占比':<10}")
print("-"*60)
total_actions = len(df)
for action_type, count in action_counts.items():
    pct = count / total_actions * 100
    print(f"{action_type:<20} {count:<10} {pct:<10.2f}%")

# ============================================================================
# 2. CHARACTERISTICS BY ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Characteristics by Action Type")
print("="*120)

action_types = ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'REVERSE_LONG', 'REVERSE_SHORT']

print(f"\n{'操作类型':<20} {'数量':<8} {'平均量能':<12} {'平均价格vsEMA%':<15} {'平均张力':<12}")
print("-"*100)

for action_type in action_types:
    subset = df[df['Action_Detail'] == action_type]
    if len(subset) > 0:
        avg_vol = subset['量能比率'].mean()
        avg_price = subset['价格vsEMA%'].mean()
        avg_tension = subset['张力'].mean()
        print(f"{action_type:<20} {len(subset):<8} {avg_vol:<12.4f} {avg_price:<15.4f} {avg_tension:<12.4f}")

# ============================================================================
# 3. PREDICTION RULES FOR EACH ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Prediction Rules Based on Characteristics")
print("="*120)

print("""
【基于特征的预测规则】

1. OPEN_LONG (开多):
   - 量能 > 1.0
   - 价格vsEMA 在 -2% 到 0% 之间（轻度超卖到中性）

2. OPEN_SHORT (开空):
   - 量能 > 1.0
   - 价格vsEMA > 1%（超买）

3. CLOSE (平仓):
   - 量能 > 0.8
   - 价格回归EMA附近（|价格vsEMA| < 0.5%）

4. REVERSE_LONG (反手多，从空翻多):
   - 量能 > 1.1
   - 价格vsEMA < -0.3%（轻度超卖）

5. REVERSE_SHORT (反手空，从多翻空):
   - 量能 > 1.2
   - 价格vsEMA < -1.0%（深度超卖）
""")

def predict_action_type(row):
    """
    基于特征预测操作类型

    Returns: Predicted_Action
    """
    vol = row['量能比率']
    price_vs_ema = row['价格vsEMA%']
    tension = row['张力']
    signal_type = row['信号类型']

    # 震荡过滤
    if signal_type == 'OSCILLATION':
        if vol > 2.0 and abs(price_vs_ema) > 3.0:
            # 极端突破，可能反手
            if price_vs_ema < -2.0:
                return 'REVERSE_LONG'
            elif price_vs_ema > 2.0:
                return 'REVERSE_SHORT'
        return 'HOLD'

    # 高张力过滤
    if abs(tension) > 0.8:
        return 'HOLD'

    # 规则1: 反手多（量能>1.1 + 轻度超卖）
    if vol > 1.1 and price_vs_ema < -0.3:
        return 'REVERSE_LONG'

    # 规则2: 反手空（量能>1.2 + 深度超卖）- 这个看起来不合理，应该是超买
    # 但根据您的手动标注，反手空是在价格下跌时（平多反空）
    # 所以可能逻辑是：价格跌得不够深，趋势反转了

    # 规则3: 开多（低量能 + 中性偏空）
    if vol > 0.8 and -2.0 < price_vs_ema < 0:
        return 'OPEN_LONG'

    # 规则4: 开空（量能 + 超买）
    if vol > 1.0 and price_vs_ema > 1.0:
        return 'OPEN_SHORT'

    # 规则5: 平仓（价格回归EMA）
    if abs(price_vs_ema) < 0.5:
        return 'CLOSE'

    # 默认
    return 'HOLD'

# Apply prediction
df['Predicted_Action'] = df.apply(predict_action_type, axis=1)

# ============================================================================
# 4. MATCHING ANALYSIS - BY ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Matching Analysis by Action Type")
print("="*120)

# Filter to only action moments (not HOLD)
action_types_full = action_types + ['HOLD_LONG', 'HOLD_SHORT', 'HOLD']

print(f"\n{'操作类型':<20} {'总数':<8} {'预测正确':<10} {'准确率':<12}")
print("-"*70)

for action_type in action_types_full:
    subset = df[df['Action_Detail'] == action_type]
    if len(subset) > 0:
        correct = (subset['Predicted_Action'] == subset['Action_Detail']).sum()
        accuracy = correct / len(subset)
        print(f"{action_type:<20} {len(subset):<8} {correct:<10} {accuracy:<12.2%}")

# ============================================================================
# 5. CONFUSION MATRIX - DETAILED
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Detailed Confusion Matrix")
print("="*120)

# Binary: Any ACTION vs HOLD
df['Is_Action'] = df['Action_Detail'].apply(lambda x: 1 if x in action_types else 0)
df['Predicted_Is_Action'] = df['Predicted_Action'].apply(lambda x: 1 if x in action_types else 0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df['Is_Action'], df['Predicted_Is_Action'])
tn, fp, fn, tp = cm.ravel()

print(f"\n{'预测':<15} {'实际HOLD':<15} {'实际ACTION':<15}")
print("-"*50)
print(f"{'HOLD':<15} {tn:<15} {fp:<15}")
print(f"{'ACTION':<15} {fn:<15} {tp:<15}")

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n性能指标:")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

# ============================================================================
# 6. DETAILED BREAKDOWN - REVERSE vs OPEN vs CLOSE
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Detailed Breakdown - REVERSE vs OPEN vs CLOSE")
print("="*120)

# REVERSE operations
reverse_data = df[df['Action_Detail'].isin(['REVERSE_LONG', 'REVERSE_SHORT'])]
if len(reverse_data) > 0:
    reverse_correct = (reverse_data['Predicted_Action'] == reverse_data['Action_Detail']).sum()
    print(f"\n反手操作 ({len(reverse_data)}个):")
    print(f"  预测正确: {reverse_correct}/{len(reverse_data)} ({reverse_correct/len(reverse_data)*100:.1f}%)")

    print(f"\n  特征:")
    print(f"    平均量能: {reverse_data['量能比率'].mean():.4f}")
    print(f"    平均价格vsEMA: {reverse_data['价格vsEMA%'].mean():.4f}%")
    print(f"    平均张力: {reverse_data['张力'].mean():.4f}")

# OPEN operations
open_data = df[df['Action_Detail'].isin(['OPEN_LONG', 'OPEN_SHORT'])]
if len(open_data) > 0:
    open_correct = (open_data['Predicted_Action'] == open_data['Action_Detail']).sum()
    print(f"\n开仓操作 ({len(open_data)}个):")
    print(f"  预测正确: {open_correct}/{len(open_data)} ({open_correct/len(open_data)*100:.1f}%)")

    print(f"\n  特征:")
    print(f"    平均量能: {open_data['量能比率'].mean():.4f}")
    print(f"    平均价格vsEMA: {open_data['价格vsEMA%'].mean():.4f}%")
    print(f"    平均张力: {open_data['张力'].mean():.4f}")

# CLOSE operations
close_data = df[df['Action_Detail'] == 'CLOSE']
if len(close_data) > 0:
    close_correct = (close_data['Predicted_Action'] == close_data['Action_Detail']).sum()
    print(f"\n平仓操作 ({len(close_data)}个):")
    print(f"  预测正确: {close_correct}/{len(close_data)} ({close_correct/len(close_data)*100:.1f}%)")

    print(f"\n  特征:")
    print(f"    平均量能: {close_data['量能比率'].mean():.4f}")
    print(f"    平均价格vsEMA: {close_data['价格vsEMA%'].mean():.4f}%")
    print(f"    平均张力: {close_data['张力'].mean():.4f}")

# ============================================================================
# 7. WHAT DO WE MISS?
# ============================================================================
print("\n" + "="*120)
print("STEP 7: What Do We Miss? (False Negatives)")
print("="*120)

fn_mask = (df['Predicted_Is_Action'] == 0) & (df['Is_Action'] == 1)
fn_data = df[fn_mask]

if len(fn_data) > 0:
    print(f"\n漏掉的操作: {len(fn_data)}个")

    # Break down by action type
    print(f"\n{'操作类型':<20} {'漏掉数量':<15}")
    print("-"*50)
    for action_type in action_types:
        count = (fn_data['Action_Detail'] == action_type).sum()
        if count > 0:
            print(f"{action_type:<20} {count:<15}")

    print(f"\n漏掉操作的特征:")
    print(f"  平均量能: {fn_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {fn_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {fn_data['张力'].mean():.4f}")

# ============================================================================
# 8. WHAT DO WE GET WRONG? (False Positives)
# ============================================================================
print("\n" + "="*120)
print("STEP 8: What Do We Get Wrong? (False Positives)")
print("="*120)

fp_mask = (df['Predicted_Is_Action'] == 1) & (df['Is_Action'] == 0)
fp_data = df[fp_mask]

if len(fp_data) > 0:
    print(f"\n误报的操作: {len(fp_data)}个")
    print(f"  平均量能: {fp_data['量能比率'].mean():.4f}")
    print(f"  平均价格vsEMA: {fp_data['价格vsEMA%'].mean():.4f}%")
    print(f"  平均张力: {fp_data['张力'].mean():.4f}")

# ============================================================================
# 9. SAVE DETAILED RESULTS
# ============================================================================
print("\n" + "="*120)
print("STEP 9: Save Detailed Results")
print("="*120)

output_cols = [
    '时间', '信号类型',
    '量能比率', '价格vsEMA%', '张力',
    'Action_Detail', 'Predicted_Action',
    '黄金信号'
]

df[output_cols].to_csv('Detailed_Action_Matching.csv', index=False, encoding='utf-8-sig')
print(f"\n详细匹配结果已保存至: Detailed_Action_Matching.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
