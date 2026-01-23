# -*- coding: utf-8 -*-
"""
黄金信号理想标注 V3 - 完整版
=============================

基于价格走势标注所有信号的理想多空切换点

逻辑：
1. 遍历所有价格点，找出趋势转折点
2. 在V7.0.5通过的信号上标注应该采取的行动
3. 保留用户已有的标注
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("黄金信号理想标注 V3 - 基于价格走势标注所有信号")
print("=" * 120)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
print(f"  总记录数: {len(df)}条")

# 提取数据
prices = df['收盘价'].values
v705_values = df['V7.0.5通过'].values

# 检查V7.0.5通过的函数
def is_v705_passed(v):
    """检查是否V7.0.5通过"""
    return v == 'TRUE' or v == True or (isinstance(v, str) and v.upper() == 'TRUE')

# 读取用户已有的标注
print("\n步骤2: 读取用户已有的标注...")
user_annotations = {}
for i in range(len(df)):
    if pd.notna(df.loc[i, '黄金信号']) and df.loc[i, '黄金信号'] != '':
        user_annotations[i] = df.loc[i, '黄金信号']
print(f"  用户已标注: {len(user_annotations)}个")

# 分析价格走势
print("\n步骤3: 分析价格走势，找出理想切换点...")

# 计算价格变化率
price_changes = np.zeros(len(prices))
for i in range(1, len(prices)):
    price_changes[i] = (prices[i] - prices[i-1]) / prices[i-1] * 100

# 找出局部极值点（简化版：使用更大的窗口）
window = 8
local_maxima = []
local_minima = []

for i in range(window, len(prices) - window):
    window_prices = prices[i-window:i+window+1]
    if prices[i] == max(window_prices):
        local_maxima.append(i)
    elif prices[i] == min(window_prices):
        local_minima.append(i)

print(f"  发现{len(local_maxima)}个局部高点, {len(local_minima)}个局部低点")

# 生成理想标注
print("\n步骤4: 生成理想标注...")

gold_signals = [''] * len(df)

# 保留用户标注
for idx, annotation in user_annotations.items():
    gold_signals[idx] = annotation

# 当前状态
current_position = None  # 'long' or 'short'
entry_price = None
entry_idx = None

# 从用户标注中推断初始状态
for i in range(len(df)):
    if i in user_annotations:
        annotation = str(user_annotations[i]).lower()
        if '开多' in annotation or '持多' in annotation:
            current_position = 'long'
            entry_price = prices[i]
            entry_idx = i
        elif '开空' in annotation or '持空' in annotation:
            current_position = 'short'
            entry_price = prices[i]
            entry_idx = i
        break

# 如果没有用户标注，从第一个V7.0.5信号开始
if current_position is None:
    for i in range(len(df)):
        if is_v705_passed(v705_values[i]):
            if df.loc[i, '交易方向'] == '做多(反向)':
                current_position = 'long'
                gold_signals[i] = '开多'
            else:
                current_position = 'short'
                gold_signals[i] = '开空'
            entry_price = prices[i]
            entry_idx = i
            break

# 现在遍历所有V7.0.5通过的信号并标注
for i in range(len(df)):
    # 跳过已有标注
    if gold_signals[i] != '':
        # 更新状态
        annotation = str(gold_signals[i]).lower()
        if '反手空' in annotation or '开空' in annotation:
            current_position = 'short'
            entry_price = prices[i]
            entry_idx = i
        elif '反手多' in annotation or '开多' in annotation:
            current_position = 'long'
            entry_price = prices[i]
            entry_idx = i
        continue

    # 只标注V7.0.5通过的信号
    if not is_v705_passed(v705_values[i]):
        continue

    # 计算从入场到现在的价格变化
    if entry_price is not None and current_position is not None:
        if current_position == 'long':
            price_change = (prices[i] - entry_price) / entry_price * 100
        else:
            price_change = (entry_price - prices[i]) / entry_price * 100
    else:
        price_change = 0

    # 检查是否接近极值点
    is_near_peak = any(abs(i - p) <= 3 for p in local_maxima)
    is_near_trough = any(abs(i - t) <= 3 for t in local_minima)

    # 检查近期价格趋势
    if i >= 10:
        recent_prices = prices[max(0, i-8):i+1]
        x_vals = np.arange(len(recent_prices))
        trend_slope = np.polyfit(x_vals, recent_prices, 1)[0]

        # 判断趋势
        is_uptrend = trend_slope > 50
        is_downtrend = trend_slope < -50
    else:
        is_uptrend = False
        is_downtrend = False

    # 根据当前位置和市场状态决定行动
    if current_position == 'long':
        # 当前持多
        if is_near_peak or (is_uptrend and price_change > 1.5):
            # 高点或明显上涨 → 平多反空
            gold_signals[i] = '多平/反空' if price_change < 2.5 else '黄金/多平反空'
            current_position = 'short'
            entry_price = prices[i]
            entry_idx = i
        else:
            # 继续持多
            gold_signals[i] = '继续持多'

    elif current_position == 'short':
        # 当前持空
        if is_near_trough or (is_downtrend and price_change > 1.5):
            # 低点或明显下跌 → 平空反多
            gold_signals[i] = '空平/反多' if price_change < 2.5 else '黄金/空平反多'
            current_position = 'long'
            entry_price = prices[i]
            entry_idx = i
        else:
            # 继续持空
            gold_signals[i] = '继续持空'

# 更新数据框
df['黄金信号'] = gold_signals

# 保存结果
print("\n步骤5: 保存结果...")
df.to_csv('最终数据_普通信号_完整含DXY_OHLC.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 已保存到: 最终数据_普通信号_完整含DXY_OHLC.csv")

# 统计结果
print("\n步骤6: 标注统计...")
non_empty_count = (df['黄金信号'] != '').sum()
v705_count = sum(1 for i in range(len(df)) if is_v705_passed(v705_values[i]))
v705_annotated = sum(1 for i in range(len(df)) if is_v705_passed(v705_values[i]) and df.loc[i, '黄金信号'] != '')
print(f"  已标注: {non_empty_count}/{len(df)}条 ({non_empty_count/len(df)*100:.1f}%)")
print(f"  V7.0.5通过: {v705_count}条")
print(f"  V7.0.5已标注: {v705_annotated}条 ({v705_annotated/v705_count*100:.1f}%)")

if non_empty_count > 0:
    print("\n  标注类型分布:")
    annotation_counts = df[df['黄金信号'] != '']['黄金信号'].value_counts()
    for annotation, count in annotation_counts.items():
        print(f"    {annotation}: {count}个")

# 验证用户标注
print("\n用户预标注验证:")
for idx in sorted(user_annotations.keys())[:10]:
    row = df.iloc[idx]
    v705_mark = '[OK]' if is_v705_passed(row['V7.0.5通过']) else '[--]'
    print(f"  行{idx+2}: {row['时间']} | V7.0.5={v705_mark:6s} | 价格={row['收盘价']:>10.2f} | {user_annotations[idx]}")

print("\n" + "=" * 120)
print("[完成] 黄金信号理想标注完成")
print("=" * 120)

# 显示标注预览
print("\n标注结果预览（前100行）:")
print("-" * 150)
print(f"{'行号':>4s} | {'时间':>16s} | {'价格':>10s} | {'V7.0.5':>6s} | {'信号类型':>20s} | {'黄金信号'}")
print("-" * 150)

for idx in range(min(100, len(df))):
    row = df.iloc[idx]
    v705_mark = '[OK]' if is_v705_passed(row['V7.0.5通过']) else '[--]'
    gold_signal = row['黄金信号'] if row['黄金信号'] != '' else ''
    signal_type = row['信号类型'][:18]

    # 只显示V7.0.5通过的或已标注的
    if is_v705_passed(row['V7.0.5通过']) or gold_signal != '':
        print(f"{idx+2:4d} | {row['时间']:16s} | {row['收盘价']:10.2f} | {v705_mark:6s} | {signal_type:20s} | {gold_signal}")

if len(df) > 100:
    print(f"\n... (还有{len(df)-100}行) ...")
