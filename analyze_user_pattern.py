# -*- coding: utf-8 -*-
"""
分析用户手动标注的真实模式
===========================
"""

import pandas as pd
import numpy as np

print("="*120)
print("ANALYZING USER'S MANUAL ANNOTATION PATTERN")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

# 只看前50个有标注的
annotated = df[df['黄金信号'].notna()].head(50)

print(f"\n前50个手动标注的详细分析:")
print(f"\n{'行号':<6} {'时间':<18} {'收盘价':<12} {'价格变化':<12} {'标注':<30} {'信号类型':<20}")
print("-" * 120)

prev_close = None
for idx, row in annotated.iterrows():
    time_str = str(row['时间'])[:16]
    close = row['收盘价']
    signal = str(row['黄金信号'])[:25] if not pd.isna(row['黄金信号']) else ''
    signal_type = str(row['信号类型'])[:18]

    # 计算价格变化
    if prev_close is not None:
        price_change = close - prev_close
        price_change_pct = price_change / prev_close * 100
        change_str = f"{price_change_pct:+.2f}%"
    else:
        change_str = "N/A"

    print(f"{idx:<6} {time_str:<18} {close:<12.2f} {change_str:<12} {signal:<30} {signal_type:<20}")

    prev_close = close

# ============================================================================
# 分析标注变化的规律
# ============================================================================
print("\n" + "="*120)
print("PATTERN ANALYSIS - When does user change position?")
print("="*120)

# 解析持仓状态
def get_position_from_annotation(annotation):
    if pd.isna(annotation):
        return 'NONE'
    annotation = str(annotation)
    if '持多' in annotation:
        return 'LONG'
    elif '持空' in annotation:
        return 'SHORT'
    elif '开多' in annotation or '反多' in annotation:
        return 'LONG_ENTRY'
    elif '开空' in annotation or '反空' in annotation:
        return 'SHORT_ENTRY'
    else:
        return 'NONE'

annotated = df[df['黄金信号'].notna()].copy()
annotated['持仓状态'] = annotated['黄金信号'].apply(get_position_from_annotation)

# 找出所有切换点
print("\n持仓切换点:")
print(f"{'时间':<18} {'收盘价':<12} {'从':<15} {'到':<15} {'价格变化%':<12}")
print("-" * 100)

prev_pos = 'NONE'
for idx, row in annotated.iterrows():
    curr_pos = row['持仓状态']
    if curr_pos != prev_pos and curr_pos != 'NONE':
        if 'LONG' in curr_pos:
            to_pos = '做多'
        elif 'SHORT' in curr_pos:
            to_pos = '做空'
        else:
            to_pos = curr_pos

        if 'LONG' in str(prev_pos):
            from_pos = '做多'
        elif 'SHORT' in str(prev_pos):
            from_pos = '做空'
        else:
            from_pos = '空仓'

        # 找到之前的价格
        if idx > 0:
            prev_close = df.loc[idx-1, '收盘价']
            price_change = (row['收盘价'] - prev_close) / prev_close * 100
        else:
            price_change = 0

        print(f"{str(row['时间'])[:18]:<18} {row['收盘价']:<12.2f} {from_pos:<15} {to_pos:<15} {price_change:>+10.2f}%")

        prev_pos = curr_pos if 'ENTRY' not in str(curr_pos) else curr_pos.replace('_ENTRY', '')

# ============================================================================
# 关键洞察：用户看什么？
# ============================================================================
print("\n" + "="*120)
print("KEY INSIGHT - What is the user looking at?")
print("="*120)

print("""
从标注来看，用户的逻辑可能是：

1. **看信号类型**：
   - BEARISH_SINGULARITY → 做多（反向交易）
   - BULLISH_SINGULARITY → 做空（反向交易）

2. **看价格位置**：
   - 价格连续上涨后 → 考虑平多反空
   - 价格连续下跌后 → 考虑平空反多

3. **看量能**：
   - 量能放大 → 可能是反转点

4. **不是纯技术指标**：
   - 用户标注包含主观判断
   - 结合了多个因素
   - 不是简单的峰值检测

需要用户确认真实的标注逻辑！
""")
