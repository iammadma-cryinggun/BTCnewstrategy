# -*- coding: utf-8 -*-
"""
黄金信号理想标注程序 V2
========================

基于价格走势标注理想多空切换点（行动点）

逻辑：
1. 找出所有局部高点和低点
2. 高点 → 做多的人应该平仓并反手做空
3. 低点 → 做空的人应该平仓并反手做多
4. 只在V7.0.5通过的信号上标注
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("黄金信号理想标注 V2 - 基于价格局部极值点")
print("=" * 100)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
print(f"  总记录数: {len(df)}条")

# 提取价格和信号数据
prices = df['收盘价'].values
dates = df['时间'].values
signal_types = df['信号类型'].values
v705_passed = df['V7.0.5通过'].values

# 找出V7.0.5通过的信号索引
v705_indices = []
for i, v in enumerate(v705_passed):
    if v == 'TRUE' or v == True or (isinstance(v, str) and v.upper() == 'TRUE'):
        v705_indices.append(i)
v705_indices = np.array(v705_indices)
print(f"  V7.0.5通过的信号: {len(v705_indices)}个")

print("\n步骤2: 分析价格序列，找出局部极值点...")

# 找出局部高点和低点
def find_local_extremes(prices, window=3):
    """找出局部高点和低点"""
    peaks = []  # 局部高点
    troughs = []  # 局部低点

    for i in range(window, len(prices) - window):
        window_prices = prices[i-window:i+window+1]

        # 检查是否是局部高点
        if prices[i] == max(window_prices):
            peaks.append(i)

        # 检查是否是局部低点
        elif prices[i] == min(window_prices):
            troughs.append(i)

    return peaks, troughs

peaks, troughs = find_local_extremes(prices, window=3)
print(f"  发现{len(peaks)}个局部高点, {len(troughs)}个局部低点")

print("\n步骤3: 为V7.0.5通过的信号标注理想行动点...")

gold_signals = [''] * len(df)

# 保留用户已有的标注
user_annotated = df['黄金信号'].notna() & (df['黄金信号'] != '')
print(f"  用户已标注: {user_annotated.sum()}个")

# 从用户标注推断当前状态
current_position = None
for i in range(len(df)):
    if user_annotated.iloc[i]:
        annotation = str(df.loc[i, '黄金信号']).lower()
        if '开多' in annotation or '持仓多' in annotation:
            current_position = 'long'
        elif '开空' in annotation or '持仓空' in annotation:
            current_position = 'short'
        gold_signals[i] = df.loc[i, '黄金信号']

# 现在标注所有V7.0.5通过的信号
for idx in v705_indices:
    # 跳过用户已标注的
    if gold_signals[idx] != '':
        continue

    # 检查这个点是否接近局部高点或低点
    look_forward = 5
    look_backward = 5

    # 检查前方是否是高点
    is_near_peak = False
    is_near_trough = False

    for p in peaks:
        if 0 <= p - idx <= look_forward:
            is_near_peak = True
            break

    for t in troughs:
        if 0 <= t - idx <= look_forward:
            is_near_trough = True
            break

    # 同时检查后方是否刚过高点或低点
    for p in peaks:
        if 0 <= idx - p <= look_backward:
            is_near_peak = True
            break

    for t in troughs:
        if 0 <= idx - t <= look_backward:
            is_near_trough = True
            break

    # 根据是否接近极值点来标注
    if is_near_peak and current_position == 'long':
        # 高点 → 平多反空
        gold_signals[idx] = '多平/反手空'
        current_position = 'short'
    elif is_near_trough and current_position == 'short':
        # 低点 → 平空反多
        gold_signals[idx] = '空平/反手多'
        current_position = 'long'
    elif not is_near_peak and not is_near_trough:
        # 既不接近高点也不接近低点，检查价格趋势
        if idx >= 10:
            recent_prices = prices[idx-10:idx+1]
            x_values = list(range(len(recent_prices)))
            price_trend = np.polyfit(x_values, recent_prices, 1)[0]

            if price_trend > 100 and current_position == 'long':
                # 明显上涨 → 平多反空
                gold_signals[idx] = '多平/反手空'
                current_position = 'short'
            elif price_trend < -100 and current_position == 'short':
                # 明显下跌 → 平空反多
                gold_signals[idx] = '空平/反手多'
                current_position = 'long'

    # 如果仍然没有标注，根据当前位置标记为继续持有
    if gold_signals[idx] == '' and current_position is not None:
        if current_position == 'long':
            gold_signals[idx] = '继续持多'
        else:
            gold_signals[idx] = '继续持空'

# 更新数据框
df['黄金信号'] = gold_signals

# 保存结果
print("\n步骤4: 保存结果...")
df.to_csv('最终数据_普通信号_完整含DXY_OHLC.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 已保存到: 最终数据_普通信号_完整含DXY_OHLC.csv")

# 统计结果
print("\n步骤5: 标注统计...")
non_empty_count = (df['黄金信号'] != '').sum()
print(f"  已标注: {non_empty_count}/{len(df)}条 ({non_empty_count/len(df)*100:.1f}%)")

if non_empty_count > 0:
    print("\n  标注类型分布:")
    annotation_counts = df[df['黄金信号'] != '']['黄金信号'].value_counts()
    for annotation, count in annotation_counts.items():
        print(f"    {annotation}: {count}个")

# 显示用户预标注的样例验证
print("\n用户预标注样例验证:")
user_annotated_rows = df[df['黄金信号'].notna() & (df['黄金信号'] != '') & df['黄金信号'].str.contains('/', na=False)].head(10)
for idx, row in user_annotated_rows.iterrows():
    print(f"  行{idx+2}: {row['时间']} | 价格={row['收盘价']:>10.2f} | {row['黄金信号']}")

print("\n" + "=" * 100)
print("[完成] 黄金信号理想标注完成")
print("=" * 100)

# 显示前100行的标注结果
print("\n前100行标注结果预览:")
print("-" * 140)
print(f"{'行号':>4s} | {'时间':>16s} | {'价格':>10s} | {'信号类型':>22s} | {'V7.0.5':>6s} | {'黄金信号'}")
print("-" * 140)

for idx in range(min(100, len(df))):
    row = df.iloc[idx]
    v705_value = row['V7.0.5通过']
    v705_mark = 'TRUE' if v705_value == 'TRUE' or v705_value == True or (isinstance(v705_value, str) and v705_value.upper() == 'TRUE') else 'FALSE'
    gold_signal = row['黄金信号'] if row['黄金信号'] != '' else ''
    signal_type = row['信号类型'][:20] if len(row['信号类型']) > 20 else row['信号类型']

    # 只显示V7.0.5通过的或已标注的行
    if v705_mark == 'TRUE' or gold_signal != '':
        print(f"{idx+2:4d} | {row['时间']:16s} | {row['收盘价']:10.2f} | {signal_type:22s} | {v705_mark:6s} | {gold_signal}")
