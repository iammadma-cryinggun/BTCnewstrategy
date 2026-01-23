# -*- coding: utf-8 -*-
"""
检查信号和K线的对应关系
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("检查信号和K线的对应关系")
print("=" * 100)

# 读取数据
df_signals = pd.read_csv('v705_trading_data_passed.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"\nK线数据: {len(df_full)}条")
print(f"通过过滤的信号: {len(df_signals)}个")

# 检查每个信号是否能在K线数据中找到
unmatched = []
matched = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['信号时间'])
    signal_type = signal['信号类型']
    signal_price = signal['信号价']

    # 在K线数据中查找
    kline = df_full[df_full['timestamp'] == signal_time]

    if len(kline) == 0:
        unmatched.append({
            '索引': idx,
            '时间': signal_time,
            '类型': signal_type,
            '信号价': signal_price
        })
    else:
        kline_close = kline.iloc[0]['close']
        price_diff = abs(signal_price - kline_close)

        matched.append({
            '索引': idx,
            '时间': signal_time,
            '类型': signal_type,
            '信号价': signal_price,
            'K线收盘价': kline_close,
            '价格差异': price_diff
        })

print(f"\n对应关系检查:")
print(f"  匹配: {len(matched)}个")
print(f"  不匹配: {len(unmatched)}个")

if len(unmatched) > 0:
    print(f"\n前10个不匹配的信号:")
    for item in unmatched[:10]:
        print(f"  索引{item['索引']}: {item['时间']} {item['类型']} 信号价{item['信号价']}")

# 检查K线数据的完整性
print(f"\nK线数据时间间隔检查:")
df_full_sorted = df_full.sort_values('timestamp')
time_diffs = df_full_sorted['timestamp'].diff()
print(f"  时间间隔统计:")
print(f"  {time_diffs.value_counts().head(10)}")

print(f"\n" + "=" * 100)
