# -*- coding: utf-8 -*-
"""
黄金信号完整标注程序
===================

根据用户预标注的逻辑模式，自动完成所有707行剩余标注

逻辑规律：
1. 首次信号 → "开仓多" 或 "开仓空"
2. 价格上涨 → "平多/反手空"
3. 价格下跌 → "平空/反手多"
4. 极端价格位置 → 标记"黄金"
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("黄金信号完整标注")
print("=" * 100)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
print(f"  总记录数: {len(df)}条")

# 检查现有标注
print("\n步骤2: 检查现有标注...")
print(f"  已标注: {df['黄金信号'].notna().sum()}条")
print(f"  未标注: {df['黄金信号'].isna().sum()}条")

# 显示已有标注
print("\n  用户已预标注的样例:")
annotated = df[df['黄金信号'].notna()][['时间', '收盘价', '信号类型', 'V7.0.5通过', '黄金信号']]
for idx, row in annotated.iterrows():
    print(f"    行{idx+2}: {row['时间']} | 价格={row['收盘价']:>10.2f} | {row['信号类型'][:20]:20s} | V7.0.5={row['V7.0.5通过']:5s} | {row['黄金信号']}")

# 自动标注逻辑
print("\n步骤3: 应用标注逻辑到剩余行...")

# 状态变量
current_position = None  # 'long' or 'short'
entry_price = None
entry_idx = None

# 保留用户已有的标注，只标注空白行
for i in range(len(df)):
    # 如果已有标注，跳过
    if pd.notna(df.loc[i, '黄金信号']):
        # 更新当前位置
        annotation = df.loc[i, '黄金信号']
        if '开多' in annotation or '多' in annotation:
            current_position = 'long'
            entry_price = df.loc[i, '收盘价']
            entry_idx = i
        elif '开空' in annotation or '空' in annotation:
            current_position = 'short'
            entry_price = df.loc[i, '收盘价']
            entry_idx = i
        continue

    # 只标注V7.0.5通过的信号
    if df.loc[i, 'V7.0.5通过'] != 'TRUE':
        continue

    current_price = df.loc[i, '收盘价']
    signal_direction = df.loc[i, '交易方向']  # 做多(反向) 或 做空(反向)

    # 首次开仓
    if current_position is None:
        if signal_direction == '做多(反向)':
            df.loc[i, '黄金信号'] = '开多'
            current_position = 'long'
        elif signal_direction == '做空(反向)':
            df.loc[i, '黄金信号'] = '开空'
            current_position = 'short'
        entry_price = current_price
        entry_idx = i

    # 当前持有多单
    elif current_position == 'long':
        price_change = (current_price - entry_price) / entry_price * 100

        if signal_direction == '做多(反向)':
            # 继续持多或平多反多
            if price_change < -1.0:
                # 价格下跌较多，可能需要调整
                df.loc[i, '黄金信号'] = '持仓观察'
            else:
                df.loc[i, '黄金信号'] = '持仓多'
        elif signal_direction == '做空(反向)':
            # 平多反空
            if price_change > 2.0:
                df.loc[i, '黄金信号'] = '黄金/多平反空'
            else:
                df.loc[i, '黄金信号'] = '多平/反手空'
            current_position = 'short'
            entry_price = current_price
            entry_idx = i

    # 当前持有空单
    elif current_position == 'short':
        price_change = (entry_price - current_price) / entry_price * 100

        if signal_direction == '做空(反向)':
            # 继续持空
            if price_change < -1.0:
                df.loc[i, '黄金信号'] = '持仓观察'
            else:
                df.loc[i, '黄金信号'] = '持仓空'
        elif signal_direction == '做多(反向)':
            # 平空反多
            if price_change > 3.0:
                df.loc[i, '黄金信号'] = '黄金/空平反多'
            else:
                df.loc[i, '黄金信号'] = '空平/反手多'
            current_position = 'long'
            entry_price = current_price
            entry_idx = i

# 保存结果
print("\n步骤4: 保存结果...")
df.to_csv('最终数据_普通信号_完整含DXY_OHLC.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 已保存到: 最终数据_普通信号_完整含DXY_OHLC.csv")

# 统计结果
print("\n步骤5: 标注统计...")
annotated_count = df['黄金信号'].notna().sum()
print(f"  已标注: {annotated_count}/{len(df)}条 ({annotated_count/len(df)*100:.1f}%)")

if annotated_count > 0:
    print("\n  标注类型分布:")
    annotation_counts = df[df['黄金信号'].notna()]['黄金信号'].value_counts()
    for annotation, count in annotation_counts.items():
        print(f"    {annotation}: {count}个")

print("\n" + "=" * 100)
print("[完成] 黄金信号标注完成")
print("=" * 100)

# 显示标注结果预览
print("\n前60行标注结果预览:")
print("-" * 100)
preview = df.head(60)
for idx, row in preview.iterrows():
    if pd.notna(row['黄金信号']):
        signal_mark = f"[{row['黄金信号']}]"
    else:
        signal_mark = ""
    print(f"行{idx+2}: {row['时间']} | {row['收盘价']:>10.2f} | {row['信号类型'][:22]:22s} | {signal_mark}")
