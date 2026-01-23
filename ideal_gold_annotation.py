# -*- coding: utf-8 -*-
"""
黄金信号理想标注程序
===================

标注基于价格走势的理想多空切换点（黄金标准）

逻辑：
1. 分析价格序列，找出上涨段和下跌段
2. 价格下跌 → 应该做多
3. 价格上涨 → 应该做空
4. 趋势转折点 → 平仓并反手
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("黄金信号理想标注 - 基于价格走势的最佳多空切换点")
print("=" * 100)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
print(f"  总记录数: {len(df)}条")

# 提取价格序列
prices = df['收盘价'].values
dates = df['时间'].values
signal_types = df['信号类型'].values
v705_passed = df['V7.0.5通过'].values

print("\n步骤2: 分析价格走势，找出趋势转折点...")

# 计算价格的移动平均和趋势
window_short = 5
window_long = 15

# 计算短期和长期均线
ma_short = pd.Series(prices).rolling(window_short).mean().values
ma_long = pd.Series(prices).rolling(window_long).mean().values

# 判断趋势
trend = []  # 1=上涨, -1=下跌, 0=震荡
for i in range(len(prices)):
    if i < window_long:
        trend.append(0)
    elif ma_short[i] > ma_long[i]:
        trend.append(1)  # 上涨趋势
    elif ma_short[i] < ma_long[i]:
        trend.append(-1)  # 下跌趋势
    else:
        trend.append(0)  # 震荡

# 找出趋势转折点
turning_points = []
for i in range(window_long, len(prices) - 1):
    if trend[i] != trend[i+1] and trend[i] != 0 and trend[i+1] != 0:
        turning_points.append(i)

print(f"  发现{len(turning_points)}个趋势转折点")

# 生成理想交易信号
print("\n步骤3: 生成理想多空切换点...")

gold_signals = [''] * len(df)
current_position = None  # 'long' or 'short'
entry_price = None
entry_idx = None

# 遍历所有信号点（只处理V7.0.5通过的信号）
for i in range(len(df)):
    # 保留用户已有的标注
    if pd.notna(df.loc[i, '黄金信号']) and df.loc[i, '黄金信号'] != '':
        gold_signals[i] = df.loc[i, '黄金信号']
        # 从用户标注中推断当前持仓
        annotation = str(df.loc[i, '黄金信号']).lower()
        if '开多' in annotation or '多' in annotation:
            current_position = 'long'
            entry_price = prices[i]
            entry_idx = i
        elif '开空' in annotation or '空' in annotation:
            current_position = 'short'
            entry_price = prices[i]
            entry_idx = i
        continue

    # 只处理V7.0.5通过的信号
    if v705_passed[i] != 'TRUE':
        continue

    # 获取当前趋势
    current_trend = trend[i] if i < len(trend) else 0

    # 判断是否是转折点
    is_turning_point = i in turning_points

    # 当前价格与入场价格的变化
    if entry_price is not None and current_position is not None:
        if current_position == 'long':
            price_change_pct = (prices[i] - entry_price) / entry_price * 100
        else:  # short
            price_change_pct = (entry_price - prices[i]) / entry_price * 100
    else:
        price_change_pct = 0

    # 根据趋势决定应该持有的仓位
    if current_trend == -1:
        # 下跌趋势 → 应该做多
        if current_position == 'short':
            # 当前是空单，需要平空反多
            if price_change_pct > 2.0 or is_turning_point:
                gold_signals[i] = '黄金/空平反多'
            else:
                gold_signals[i] = '空平/反手多'
            current_position = 'long'
            entry_price = prices[i]
            entry_idx = i
        elif current_position == 'long':
            # 已经是多单，继续持有
            gold_signals[i] = '持仓多'
        else:
            # 首次开多
            gold_signals[i] = '开多'
            current_position = 'long'
            entry_price = prices[i]
            entry_idx = i

    elif current_trend == 1:
        # 上涨趋势 → 应该做空
        if current_position == 'long':
            # 当前是多单，需要平多反空
            if price_change_pct > 2.0 or is_turning_point:
                gold_signals[i] = '黄金/多平反空'
            else:
                gold_signals[i] = '多平/反手空'
            current_position = 'short'
            entry_price = prices[i]
            entry_idx = i
        elif current_position == 'short':
            # 已经是空单，继续持有
            gold_signals[i] = '持仓空'
        else:
            # 首次开空
            gold_signals[i] = '开空'
            current_position = 'short'
            entry_price = prices[i]
            entry_idx = i

    else:
        # 震荡趋势
        gold_signals[i] = '观察'
        # 保持当前持仓不变

# 更新数据框
df['黄金信号'] = gold_signals

# 保存结果
print("\n步骤4: 保存结果...")
df.to_csv('最终数据_普通信号_完整含DXY_OHLC.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 已保存到: 最终数据_普通信号_完整含DXY_OHLC.csv")

# 统计结果
print("\n步骤5: 标注统计...")
annotated_count = df['黄金信号'].notna().sum()
non_empty_count = (df['黄金信号'] != '').sum()
print(f"  已标注: {non_empty_count}/{len(df)}条 ({non_empty_count/len(df)*100:.1f}%)")

if non_empty_count > 0:
    print("\n  标注类型分布:")
    annotation_counts = df[df['黄金信号'] != '']['黄金信号'].value_counts()
    for annotation, count in annotation_counts.items():
        print(f"    {annotation}: {count}个")

# 显示用户标注的样例
print("\n用户预标注的样例（用于验证）:")
user_annotated = df[df['黄金信号'].notna() & (df['黄金信号'] != '')].head(20)
for idx, row in user_annotated.iterrows():
    # 检查是否是用户原始标注（包含"/"和"反手"等关键字）
    annotation = str(row['黄金信号'])
    if '/' in annotation or '开多' in annotation or '开空' in annotation:
        print(f"  行{idx+2}: {row['时间']} | 价格={row['收盘价']:>10.2f} | 趋势={trend[idx]:2d} | {annotation}")

print("\n" + "=" * 100)
print("[完成] 黄金信号理想标注完成")
print("=" * 100)

# 显示前80行的标注结果
print("\n前80行标注结果预览:")
print("-" * 120)
print(f"{'行号':>4s} | {'时间':>16s} | {'价格':>10s} | {'趋势':>4s} | {'V7.0.5':>6s} | {'黄金信号'}")
print("-" * 120)

for idx in range(min(80, len(df))):
    row = df.iloc[idx]
    trend_mark = {1: '↑上涨', -1: '↓下跌', 0: '~震荡'}.get(trend[idx], '  无')
    v705_mark = 'TRUE' if row['V7.0.5通过'] == 'TRUE' or row['V7.0.5通过'] == True else 'FALSE'
    gold_signal = row['黄金信号'] if row['黄金信号'] != '' else ''
    print(f"{idx+2:4d} | {row['时间']:16s} | {row['收盘价']:10.2f} | {trend_mark:6s} | {v705_mark:6s} | {gold_signal}")

# 显示后续数据预览
if len(df) > 80:
    print(f"\n... (还有{len(df)-80}行数据) ...")
