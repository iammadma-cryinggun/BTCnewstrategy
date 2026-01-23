# -*- coding: utf-8 -*-
"""
基于价格波动的正确标注 - 后验最优交易路径
========================================

逻辑：
1. 找价格低点 → 开多/反手多
2. 找价格高点 → 开空/反手空
3. 持仓期间 → 继续持有
4. 价格反转 → 平仓/反手

这是基于未来价格走势的后验分析，标注的是"本应该做什么"
"""

import pandas as pd
import numpy as np

print("="*120)
print("CORRECT ANNOTATION BASED ON PRICE MOVEMENTS")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# ALGORITHM: 找波峰波谷，标注最优交易路径
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Find Peaks and Valleys - Optimal Trading Path")
print("="*120)

# Initialize
actions = []
positions = []  # 当前持仓: 'LONG', 'SHORT', 'NONE'
entry_prices = []  # 开仓价格

# 参数
WINDOW = 6  # 前后N根K线判断波峰波谷
PEAK_THRESHOLD = 0.015  # 1.5%变化才算反转

current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']

    # 获取前后窗口的价格
    start_idx = max(0, i - WINDOW)
    end_idx = min(len(df), i + WINDOW + 1)
    window_prices = df.loc[start_idx:end_idx-1, '收盘价'].values

    # 判断是否是局部高点
    is_peak = (current_close == np.max(window_prices)) and (i > 0) and (i < len(df)-1)
    is_valley = (current_close == np.min(window_prices)) and (i > 0) and (i < len(df)-1)

    # 计算从entry的涨跌幅
    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (current_close - entry_price) / entry_price * 100
        elif current_position == 'SHORT':
            pnl_pct = (entry_price - current_close) / entry_price * 100
        else:
            pnl_pct = 0
    else:
        pnl_pct = 0

    # 决策逻辑
    action = ''
    new_position = current_position

    if current_position == 'NONE':
        # 无持仓，判断是否进场
        if is_valley:
            action = '开多'
            new_position = 'LONG'
            entry_price = current_close
        elif is_peak:
            action = '开空'
            new_position = 'SHORT'
            entry_price = current_close
        else:
            action = '观望'

    elif current_position == 'LONG':
        # 持有多仓
        if is_peak and pnl_pct > PEAK_THRESHOLD * 100:
            # 价格到高点，平多反空
            action = f'平多/反空 (盈利{pnl_pct:.2f}%)'
            new_position = 'SHORT'
            entry_price = current_close
        else:
            action = f'继续持多 (盈亏{pnl_pct:.2f}%)'

    elif current_position == 'SHORT':
        # 持有空仓
        if is_valley and pnl_pct > PEAK_THRESHOLD * 100:
            # 价格到低点，平空反多
            action = f'平空/反多 (盈利{pnl_pct:.2f}%)'
            new_position = 'LONG'
            entry_price = current_close
        else:
            action = f'继续持空 (盈亏{pnl_pct:.2f}%)'

    actions.append(action)
    positions.append(new_position)
    entry_prices.append(entry_price)

df['最优动作'] = actions
df['持仓状态'] = positions

# ============================================================================
# 统计分析
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Statistics")
print("="*120)

# 动作类型统计
action_stats = {}
for action in actions:
    if '开多' in action or '反多' in action:
        key = '做多相关'
    elif '开空' in action or '反空' in action:
        key = '做空相关'
    elif '持多' in action:
        key = '持多'
    elif '持空' in action:
        key = '持空'
    else:
        key = '观望'
    action_stats[key] = action_stats.get(key, 0) + 1

print("\n动作类型统计:")
for key, count in action_stats.items():
    print(f"  {key}: {count} 次 ({count/len(df)*100:.1f}%)")

# ============================================================================
# 对比用户手动标注
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Compare with User Manual Annotations")
print("="*120)

# 解析用户标注
def parse_user_annotation(annotation):
    if pd.isna(annotation):
        return 'NONE'
    annotation = str(annotation)
    if '开多' in annotation or '反多' in annotation or '持多' in annotation:
        return 'LONG'
    elif '开空' in annotation or '反空' in annotation or '持空' in annotation:
        return 'SHORT'
    else:
        return 'NONE'

df['用户持仓'] = df['黄金信号'].apply(parse_user_annotation)

# 计算匹配度
match_count = 0
for i in range(len(df)):
    if df.loc[i, '持仓状态'] == df.loc[i, '用户持仓']:
        match_count += 1

print(f"\n持仓状态匹配度: {match_count}/{len(df)} = {match_count/len(df)*100:.1f}%")

# 详细对比（前50个）
print("\n前50个信号对比:")
print(f"{'时间':<20} {'收盘价':<12} {'最优动作':<25} {'用户标注':<20} {'匹配':<6}")
print("-" * 120)

for i in range(min(50, len(df))):
    time_str = str(df.loc[i, '时间'])[:16]
    close_price = f"{df.loc[i, '收盘价']:.2f}"
    optimal = df.loc[i, '最优动作'][:20]
    user = str(df.loc[i, '黄金信号'])[:15] if not pd.isna(df.loc[i, '黄金信号']) else 'NaN'
    match = '✓' if df.loc[i, '持仓状态'] == df.loc[i, '用户持仓'] else '✗'

    print(f"{time_str:<20} {close_price:<12} {optimal:<25} {user:<20} {match:<6}")

# ============================================================================
# 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Save Results")
print("="*120)

output_cols = [
    '时间', '开盘价', '最高价', '最低价', '收盘价',
    '信号类型', '置信度', '量能比率', '价格vsEMA%', '张力', '加速度',
    '最优动作', '持仓状态', '黄金信号'
]

df[output_cols].to_csv('价格最优路径标注.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 价格最优路径标注.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)

print(f"""
总结：

基于价格波动的后验最优路径：
- 使用 {WINDOW} 根K线窗口判断波峰波谷
- 盈利阈值: {PEAK_THRESHOLD*100}%
- 总信号数: {len(df)}

与用户手动标注的匹配度: {match_count/len(df)*100:.1f}%

这个算法自动化了用户的标注逻辑：
1. 在价格低点开多/反多
2. 在价格高点开空/反空
3. 持仓期间继续持有
4. 价格反转时平仓反手

这是"后验最优"，不是实时预测！
""")
