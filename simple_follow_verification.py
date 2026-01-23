# -*- coding: utf-8 -*-
"""
终极验证：简单跟随信号策略
========================

假设：REVERSE操作就是直接跟随验证5的信号类型
- BEARISH_SINGULARITY → 反手多
- BULLISH_SINGULARITY → 反手空
- OSCILLATION → 观望
"""

import pandas as pd
from sklearn.metrics import confusion_matrix

print("="*120)
print("ULTIMATE VERIFICATION: Follow Signal Type Strategy")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

# Classify action
def classify_action(gold_signal):
    if pd.isna(gold_signal):
        return 'HOLD'
    signal_str = str(gold_signal)
    if '反手' in signal_str or '反' in signal_str:
        if '多' in signal_str:
            return 'REVERSE_LONG'
        elif '空' in signal_str:
            return 'REVERSE_SHORT'
    if '继续持' in signal_str or '持仓' in signal_str:
        return 'HOLD'
    return 'OTHER'

df['Manual_Action'] = df['黄金信号'].apply(classify_action)

# Simple strategy: Follow signal type
def simple_follow_strategy(row):
    signal_type = row['信号类型']

    if signal_type == 'BEARISH_SINGULARITY':
        return 'REVERSE_LONG'
    elif signal_type == 'BULLISH_SINGULARITY':
        return 'REVERSE_SHORT'
    elif signal_type == 'OSCILLATION':
        return 'HOLD'
    else:
        return 'HOLD'

df['Simple_Prediction'] = df.apply(simple_follow_strategy, axis=1)

# Evaluate
y_true = df['Manual_Action'].apply(lambda x: 1 if x == 'REVERSE_LONG' else 0)
y_pred_simple = df['Simple_Prediction'].apply(lambda x: 1 if x == 'REVERSE_LONG' else 0)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_simple)
tn, fp, fn, tp = cm.ravel()

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n性能指标:")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

print(f"\n混淆矩阵:")
print(f"  True Positives: {tp}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Negatives: {tn}")

# Check breakdown by signal type
print(f"\n按信号类型统计:")
for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'OSCILLATION']:
    subset = df[df['信号类型'] == sig_type]
    if len(subset) > 0:
        actual_reverse = (subset['Manual_Action'] == 'REVERSE_LONG').sum()
        predicted_reverse = (subset['Simple_Prediction'] == 'REVERSE_LONG').sum()
        print(f"  {sig_type}:")
        print(f"    实际REVERSE_LONG: {actual_reverse}/{len(subset)}")
        print(f"    预测REVERSE_LONG: {predicted_reverse}/{len(subset)}")

print("\n" + "="*120)
print("VERIFICATION COMPLETE")
print("="*120)
print(f"""
结论:

简单跟随信号策略:
- BEARISH_SINGULARITY → REVERSE_LONG
- BULLISH_SINGULARITY → REVERSE_SHORT
- OSCILLATION → HOLD

性能: F1={f1:.4f}

这个策略是否有效？
- 如果F1高: 说明REVERSE操作就是直接跟随验证5信号
- 如果F1低: 说明还有其他因素影响
""")
