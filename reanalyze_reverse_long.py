# -*- coding: utf-8 -*-
"""
重新分析REVERSE_LONG - 究竟什么条件？
==================================
"""

import pandas as pd
import numpy as np

print("="*120)
print("RE-ANALYZING REVERSE_LONG - What are the REAL conditions?")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

# Classify
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

# Get REVERSE_LONG data
reverse_long = df[df['Manual_Action'] == 'REVERSE_LONG'].copy()

print(f"\nREVERSE_LONG总数: {len(reverse_long)}")

print(f"\n当前时刻特征:")
print(f"  平均量能: {reverse_long['量能比率'].mean():.4f}")
print(f"  平均价格vsEMA: {reverse_long['价格vsEMA%'].mean():.4f}%")
print(f"  平均张力: {reverse_long['张力'].mean():.4f}")
print(f"  平均加速度: {reverse_long['加速度'].mean():.4f}")

print(f"\n信号类型分布:")
print(reverse_long['信号类型'].value_counts())

print(f"\n价格vsEMA分布:")
print(f"  < -2.0%: {(reverse_long['价格vsEMA%'] < -2.0).sum()} ({(reverse_long['价格vsEMA%'] < -2.0).sum()/len(reverse_long)*100:.1f}%)")
print(f"  -2.0% to -1.0%: {((reverse_long['价格vsEMA%'] >= -2.0) & (reverse_long['价格vsEMA%'] < -1.0)).sum()} ({((reverse_long['价格vsEMA%'] >= -2.0) & (reverse_long['价格vsEMA%'] < -1.0)).sum()/len(reverse_long)*100:.1f}%)")
print(f"  -1.0% to 0%: {((reverse_long['价格vsEMA%'] >= -1.0) & (reverse_long['价格vsEMA%'] < 0)).sum()} ({((reverse_long['价格vsEMA%'] >= -1.0) & (reverse_long['价格vsEMA%'] < 0)).sum()/len(reverse_long)*100:.1f}%)")
print(f"  >= 0%: {(reverse_long['价格vsEMA%'] >= 0).sum()} ({(reverse_long['价格vsEMA%'] >= 0).sum()/len(reverse_long)*100:.1f}%)")

print(f"\n量能分布:")
print(f"  > 1.4: {(reverse_long['量能比率'] > 1.4).sum()} ({(reverse_long['量能比率'] > 1.4).sum()/len(reverse_long)*100:.1f}%)")
print(f"  1.2-1.4: {((reverse_long['量能比率'] >= 1.2) & (reverse_long['量能比率'] <= 1.4)).sum()} ({((reverse_long['量能比率'] >= 1.2) & (reverse_long['量能比率'] <= 1.4)).sum()/len(reverse_long)*100:.1f}%)")
print(f"  1.0-1.2: {((reverse_long['量能比率'] >= 1.0) & (reverse_long['量能比率'] < 1.2)).sum()} ({((reverse_long['量能比率'] >= 1.0) & (reverse_long['量能比率'] < 1.2)).sum()/len(reverse_long)*100:.1f}%)")
print(f"  < 1.0: {(reverse_long['量能比率'] < 1.0).sum()} ({(reverse_long['量能比率'] < 1.0).sum()/len(reverse_long)*100:.1f}%)")

# Show first 20 REVERSE_LONG cases
print(f"\n前20个REVERSE_LONG案例:")
print("-"*120)
cols = ['时间', '信号类型', '量能比率', '价格vsEMA%', '张力', '加速度', '黄金信号']
print(reverse_long[cols].head(20).to_string(index=False))

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
