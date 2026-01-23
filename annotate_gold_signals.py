# -*- coding: utf-8 -*-
"""
黄金信号自动标注程序
===================

根据用户预标注的逻辑模式，自动分析价格序列并标注所有信号

逻辑：基于价格序列寻找最优进出场点
"""

import pandas as pd
import numpy as np
from io import StringIO

print("=" * 100)
print("黄金信号自动标注")
print("=" * 100)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
print(f"  总记录数: {len(df)}条")

# 检查现有标注
print("\n步骤2: 检查现有标注...")
has_annotation = df['黄金信号'].notna()
print(f"  已标注: {has_annotation.sum()}条")
print(f"  未标注: {(~has_annotation).sum()}条")

# 显示已有标注
if has_annotation.sum() > 0:
    print("\n  已有的标注样例:")
    annotated = df[has_annotation][['时间', '收盘价', '信号类型', '黄金信号']].head(10)
    for idx, row in annotated.iterrows():
        print(f"    行{idx+2}: {row['时间']} | 价格={row['收盘价']:.2f} | {row['信号类型'][:15]:15s} | {row['黄金信号']}")

# 价格序列分析逻辑
print("\n步骤3: 分析价格序列并标注...")

# 准备数据
prices = df['收盘价'].values
dates = df['时间'].values
signal_types = df['信号类型'].values

# 初始化黄金信号列
gold_signals = df['黄金信号'].copy()

# 分析价格序列寻找关键转折点
def find_turning_points(prices, window=5):
    """寻找价格转折点"""
    turning_points = []

    for i in range(window, len(prices) - window):
        # 局部极值点
        before = prices[i-window:i]
        after = prices[i+1:i+window+1]

        # 局部高点
        if prices[i] == max(before) and prices[i] > max(after):
            turning_points.append((i, 'peak', prices[i]))

        # 局部低点
        elif prices[i] == min(before) and prices[i] < min(after):
            turning_points.append((i, 'trough', prices[i]))

    return turning_points

# 寻找转折点
turning_points = find_turning_points(prices, window=3)
print(f"  发现{len(turning_points)}个价格转折点")

# 基于转折点和价格趋势标注
print("\n步骤4: 生成黄金信号标注...")

# 状态变量
current_position = None  # 'long' or 'short'
entry_price = None
entry_idx = None

# 只标注未标注的行
for i in range(len(df)):
    if pd.notna(gold_signals.iloc[i]):
        # 保留用户已有的标注
        current_position = None  # 重置，让用户标注决定
        continue

    # 简化逻辑：基于价格变化和趋势
    if i > 0:
        price_change = (prices[i] - prices[i-1]) / prices[i-1] * 100

        # 如果是V7.0.5通过的信号，才进行黄金信号分析
        if df['V7.0.5通过'].iloc[i] == 'TRUE':
            # 做多信号（BEARISH的反向）
            if df['交易方向'].iloc[i] == '做多(反向)':
                # 检查是否处于低位
                if i >= 5:
                    recent_prices = prices[max(0, i-5):i+1]
                    if prices[i] <= min(recent_prices):
                        gold_signals.iloc[i] = "黄金/开多"
                    else:
                        gold_signals.iloc[i] = "开多"
                else:
                    gold_signals.iloc[i] = "开多"

            # 做空信号（BULLISH的反向）
            elif df['交易方向'].iloc[i] == '做空(反向)':
                # 检查是否处于高位
                if i >= 5:
                    recent_prices = prices[max(0, i-5):i+1]
                    if prices[i] >= max(recent_prices):
                        gold_signals.iloc[i] = "黄金/开空"
                    else:
                        gold_signals.iloc[i] = "开空"
                else:
                    gold_signals.iloc[i] = "开空"

# 更新数据框
df['黄金信号'] = gold_signals

# 保存结果
print("\n步骤5: 保存结果...")
df.to_csv('最终数据_普通信号_完整含DXY_OHLC.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 已保存到: 最终数据_普通信号_完整含DXY_OHLC.csv")

# 统计结果
print("\n步骤6: 标注统计...")
annotated_count = df['黄金信号'].notna().sum()
print(f"  已标注: {annotated_count}/{len(df)}条")

if annotated_count > 0:
    print("\n  标注类型分布:")
    annotation_counts = df[df['黄金信号'].notna()]['黄金信号'].value_counts()
    for annotation, count in annotation_counts.items():
        print(f"    {annotation}: {count}个")

print("\n" + "=" * 100)
print("[完成] 黄金信号标注完成")
print("=" * 100)

# 显示前50行的标注结果
print("\n前50行标注结果预览:")
preview = df[['时间', '收盘价', '信号类型', 'V7.0.5通过', '黄金信号']].head(50)
for idx, row in preview.iterrows():
    if pd.notna(row['黄金信号']):
        print(f"行{idx+2}: {row['时间']} | {row['收盘价']:>10.2f} | {row['信号类型'][:20]:20s} | V7.0.5={row['V7.0.5通过']:5s} | {row['黄金信号']}")
