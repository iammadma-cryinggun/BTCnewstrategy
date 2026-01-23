# -*- coding: utf-8 -*-
"""
确认用户手动标注的正确逻辑
==========================
"""

import pandas as pd

print("="*120)
print("VERIFYING USER'S ANNOTATION LOGIC")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

# 只看前20个有标注的
annotated = df[df['黄金信号'].notna()].head(20)

print(f"\n{'序号':<6} {'时间':<20} {'收盘价':<12} {'信号类型':<20} {'当前标注':<25} {'价格变化':<12}")
print("-" * 120)

prev_close = None
for idx, row in annotated.iterrows():
    time_str = str(row['时间'])
    close = row['收盘价']
    signal_type = str(row['信号类型'])[:18]
    annotation = str(row['黄金信号'])[:23]

    # 计算价格变化
    if prev_close is not None:
        price_change_pct = (close - prev_close) / prev_close * 100
        change_str = f"{price_change_pct:+.2f}%"
    else:
        change_str = "起点"

    print(f"{idx:<6} {time_str:<20} {close:<12.2f} {signal_type:<20} {annotation:<25} {change_str:<12}")

    prev_close = close

print("\n" + "="*120)
print("关键问题：请确认以下理解是否正确")
print("="*120)

print("""
从您的描述"8月20日 0:00 开多，8月20日 20:00 平多反空，一直做到8月21日 20:00"：

应该的交易路径：
- 8/20 0:00   → 开多
- 8/20 4:00   → 继续持多
- 8/20 8:00   → 继续持多
- 8/20 12:00  → 继续持多
- 8/20 16:00  → 继续持多
- 8/20 20:00  → 平多/反空
- 8/21 0:00   → 继续持空
- 8/21 4:00   → 继续持空  (不是反多!)
- 8/21 8:00   → 继续持空
- 8/21 12:00  → 继续持空
- 8/21 16:00  → 继续持空
- 8/21 20:00  → 平空/反多

但您的CSV显示：
- Row 9 (8/21 4:00): 标注"空平/反多" ← 这与您描述矛盾

请确认：
1. Row 9的"空平/反多"是标错了吗？应该是"继续持空"？
2. 还是您的描述和CSV标注有其他我理解不对的地方？
3. 正确的标注逻辑到底应该看什么？(价格涨跌幅? 信号类型? 还是其他?)
""")

print("\n请告诉我正确的标注逻辑，我会重新标注整个表格。")
