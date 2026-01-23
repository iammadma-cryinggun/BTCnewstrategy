# -*- coding: utf-8 -*-
"""
分析漏掉的REVERSE_LONG案例 - 为什么漏掉？
======================================
"""

import pandas as pd
import numpy as np

print("="*120)
print("MISSED REVERSE_LONG CASES ANALYSIS")
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
    if '开' in signal_str:
        return 'OPEN'
    if '平' in signal_str and '反' not in signal_str:
        return 'CLOSE'
    if '继续持' in signal_str or '持仓' in signal_str:
        return 'HOLD'
    return 'HOLD'

df['Manual_Action'] = df['黄金信号'].apply(classify_action)

# Predict using the optimized rule (vol>0.85, price<-0.5)
def predict(row):
    if row['量能比率'] > 0.85 and row['价格vsEMA%'] < -0.5:
        return 'REVERSE_LONG'
    return 'HOLD'

df['Predicted'] = df.apply(predict, axis=1)

# Get missed cases
missed = df[(df['Predicted'] == 'HOLD') & (df['Manual_Action'] == 'REVERSE_LONG')].copy()

print(f"\n漏掉的REVERSE_LONG案例: {len(missed)}个")
print("\n前20个漏掉的案例:")
print("-"*120)

cols = ['时间', '信号类型', '量能比率', '价格vsEMA%', '张力', '加速度', '黄金信号']

print(missed[cols].head(20).to_string(index=False))

print(f"\n\n统计特征:")
print(f"  量能均值: {missed['量能比率'].mean():.4f}")
print(f"  价格vsEMA均值: {missed['价格vsEMA%'].mean():.4f}%")
print(f"  张力均值: {missed['张力'].mean():.4f}")
print(f"  加速度均值: {missed['加速度'].mean():.4f}")

# Check price vs EMA distribution
print(f"\n价格vsEMA分布:")
print(f"  <-1.0%: {(missed['价格vsEMA%'] < -1.0).sum()} ({(missed['价格vsEMA%'] < -1.0).sum()/len(missed)*100:.1f}%)")
print(f"  -1.0% to -0.5%: {((missed['价格vsEMA%'] >= -1.0) & (missed['价格vsEMA%'] < -0.5)).sum()} ({((missed['价格vsEMA%'] >= -1.0) & (missed['价格vsEMA%'] < -0.5)).sum()/len(missed)*100:.1f}%)")
print(f"  -0.5% to 0%: {((missed['价格vsEMA%'] >= -0.5) & (missed['价格vsEMA%'] < 0)).sum()} ({((missed['价格vsEMA%'] >= -0.5) & (missed['价格vsEMA%'] < 0)).sum()/len(missed)*100:.1f}%)")
print(f"  >= 0%: {(missed['价格vsEMA%'] >= 0).sum()} ({(missed['价格vsEMA%'] >= 0).sum()/len(missed)*100:.1f}%)")

# Check volume distribution
print(f"\n量能分布:")
print(f"  >1.2: {(missed['量能比率'] > 1.2).sum()} ({(missed['量能比率'] > 1.2).sum()/len(missed)*100:.1f}%)")
print(f"  1.0-1.2: {((missed['量能比率'] >= 1.0) & (missed['量能比率'] <= 1.2)).sum()} ({((missed['量能比率'] >= 1.0) & (missed['量能比率'] <= 1.2)).sum()/len(missed)*100:.1f}%)")
print(f"  0.85-1.0: {((missed['量能比率'] >= 0.85) & (missed['量能比率'] < 1.0)).sum()} ({((missed['量能比率'] >= 0.85) & (missed['量能比率'] < 1.0)).sum()/len(missed)*100:.1f}%)")
print(f"  <0.85: {(missed['量能比率'] < 0.85).sum()} ({(missed['量能比率'] < 0.85).sum()/len(missed)*100:.1f}%)")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
