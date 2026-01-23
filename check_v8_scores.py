# -*- coding: utf-8 -*-
"""
检查V8.0评分分布
"""

import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('带信号标记_完整数据_修复版.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

print("="*100)
print("V8.0 Score Distribution Analysis")
print("="*100)

# 计算V8.0评分
def calculate_v8_score(row):
    accel = abs(row.get('加速度', 0))
    tension = abs(row.get('张力', 0))
    volume = row.get('量能比率', 1.0)

    score = (
        min(accel / 0.3, 1.0) * 0.5 +
        min(tension / 1.0, 1.0) * 0.3 +
        min(volume / 2.0, 1.0) * 0.2
    )

    return score

df['V8_Score'] = df.apply(calculate_v8_score, axis=1)

# 统计
print(f"\n[V8_Score Statistics]")
print(f"  Mean: {df['V8_Score'].mean():.4f}")
print(f"  Median: {df['V8_Score'].median():.4f}")
print(f"  Max: {df['V8_Score'].max():.4f}")
print(f"  Min: {df['V8_Score'].min():.4f}")
print(f"  Std: {df['V8_Score'].std():.4f}")

# 分位数
print(f"\n[V8_Score Percentiles]")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}%: {df['V8_Score'].quantile(p/100):.4f}")

# 检查信号动作分布
print(f"\n[Signal Action Distribution]")
print(df['信号动作'].value_counts())

# 检查高V8_Score的行
high_score = df[df['V8_Score'] >= 0.3]
print(f"\n[High Score Rows (V8 >= 0.3)]")
print(f"  Count: {len(high_score)}")
if len(high_score) > 0:
    print(f"\n  Top 10 rows:")
    print(high_score[['时间', 'V8_Score', '信号类型', '信号动作', '收盘价']].head(10).to_string())

# 检查有信号动作的行
signal_rows = df[df['信号动作'].notna() & (df['信号动作'] != '')]
print(f"\n[Rows with Signal Action]")
print(f"  Count: {len(signal_rows)}")
if len(signal_rows) > 0:
    print(f"\n  V8_Score distribution for signal rows:")
    print(f"    Mean: {signal_rows['V8_Score'].mean():.4f}")
    print(f"    Median: {signal_rows['V8_Score'].median():.4f}")
    print(f"    Max: {signal_rows['V8_Score'].max():.4f}")
    print(f"    Min: {signal_rows['V8_Score'].min():.4f}")

print("\n" + "="*100)
