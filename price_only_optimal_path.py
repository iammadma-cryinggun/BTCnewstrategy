# -*- coding: utf-8 -*-
"""
纯价格驱动的后验最优交易路径
============================

逻辑：
1. 找价格波峰 → 平多反空
2. 找价格波谷 → 平空反多
3. 持仓期间 → 继续持有
"""

import pandas as pd
import numpy as np

print("="*120)
print("PRICE-ONLY OPTIMAL TRADING PATH")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# ALGORITHM: 找波峰波谷
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Find Peaks and Valleys")
print("="*120)

WINDOW = 5  # 前后5根K线判断
MIN_CHANGE = 0.008  # 最小涨跌幅0.8%

actions = []
positions = []  # 'LONG', 'SHORT', 'NONE'
entry_prices = []
peaks_valleys = []  # 标记波峰波谷

current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']

    # 获取窗口内的价格
    start_idx = max(0, i - WINDOW)
    end_idx = min(len(df), i + WINDOW + 1)
    window_closes = df.loc[start_idx:end_idx-1, '收盘价'].values
    window_max = np.max(window_closes)
    window_min = np.min(window_closes)

    # 判断是否是波峰或波谷
    is_peak = (current_close == window_max) and (i > WINDOW) and (i < len(df) - WINDOW)
    is_valley = (current_close == window_min) and (i > WINDOW) and (i < len(df) - WINDOW)

    # 计算从entry的盈亏
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
    pv_type = ''

    if current_position == 'NONE':
        # 初始状态，判断是波谷还是波峰
        if is_valley:
            action = '开多'
            current_position = 'LONG'
            entry_price = current_close
            pv_type = '波谷'
        elif is_peak:
            action = '开空'
            current_position = 'SHORT'
            entry_price = current_close
            pv_type = '波峰'
        else:
            action = '观望'
            pv_type = ''

    elif current_position == 'LONG':
        # 持有多仓
        if is_peak and pnl_pct > MIN_CHANGE * 100:
            # 价格到波峰且盈利足够 → 平多反空
            action = f'平多/反空 (盈利{pnl_pct:.2f}%)'
            current_position = 'SHORT'
            entry_price = current_close
            pv_type = '波峰★'
        else:
            action = f'继续持多 (盈亏{pnl_pct:+.2f}%)'
            pv_type = ''

    elif current_position == 'SHORT':
        # 持有空仓
        if is_valley and pnl_pct > MIN_CHANGE * 100:
            # 价格到波谷且盈利足够 → 平空反多
            action = f'平空/反多 (盈利{pnl_pct:.2f}%)'
            current_position = 'LONG'
            entry_price = current_close
            pv_type = '波谷★'
        else:
            action = f'继续持空 (盈亏{pnl_pct:+.2f}%)'
            pv_type = ''

    actions.append(action)
    positions.append(current_position)
    entry_prices.append(entry_price)
    peaks_valleys.append(pv_type)

df['最优动作'] = actions
df['持仓状态'] = positions
df['波峰波谷'] = peaks_valleys

# ============================================================================
# 显示前30个结果
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Results (First 30)")
print("="*120)

print(f"\n{'序号':<6} {'时间':<18} {'收盘价':<10} {'波峰波谷':<8} {'持仓':<6} {'动作':<30}")
print("-" * 120)

for i in range(30):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    pv = df.loc[i, '波峰波谷']
    pos = df.loc[i, '持仓状态']
    action = df.loc[i, '最优动作'][:28]

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {pv:<8} {pos:<6} {action:<30}")

# ============================================================================
# 验证用户给出的例子
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Verify User's Example")
print("="*120)

print("\n验证用户例子:")
print("8/19 20:00 收盘112873 → 开多")
print("8/20 16:00 收盘114277 → 平多反空")
print("8/22 8:00 收盘112320 → 平空反多")
print()

# 找到这些行
row_8_19_20 = df[(df['时间'].dt.strftime('%Y-%m-%d %H:%M') == '2025-08-19 20:00')].index
row_8_20_16 = df[(df['时间'].dt.strftime('%Y-%m-%d %H:%M') == '2025-08-20 16:00')].index
row_8_22_08 = df[(df['时间'].dt.strftime('%Y-%m-%d %H:%M') == '2025-08-22 08:00')].index

if len(row_8_19_20) > 0:
    idx = row_8_19_20[0]
    print(f"Row {idx}: {df.loc[idx, '时间']} 收盘{df.loc[idx, '收盘价']:.2f} → {df.loc[idx, '最优动作']}")

if len(row_8_20_16) > 0:
    idx = row_8_20_16[0]
    print(f"Row {idx}: {df.loc[idx, '时间']} 收盘{df.loc[idx, '收盘价']:.2f} → {df.loc[idx, '最优动作']}")

if len(row_8_22_08) > 0:
    idx = row_8_22_08[0]
    print(f"Row {idx}: {df.loc[idx, '时间']} 收盘{df.loc[idx, '收盘价']:.2f} → {df.loc[idx, '最优动作']}")

# ============================================================================
# 统计
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Statistics")
print("="*120)

action_count = {}
for action in actions:
    if '开多' in action:
        key = '开多'
    elif '开空' in action:
        key = '开空'
    elif '平多反空' in action:
        key = '平多反空'
    elif '平空反多' in action:
        key = '平空反多'
    elif '持多' in action:
        key = '继续持多'
    elif '持空' in action:
        key = '继续持空'
    else:
        key = '观望'
    action_count[key] = action_count.get(key, 0) + 1

print("\n动作统计:")
for key, count in action_count.items():
    print(f"  {key}: {count} 次")

# ============================================================================
# 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Save Results")
print("="*120)

output_cols = [
    '时间', '开盘价', '最高价', '最低价', '收盘价',
    '信号类型', '量能比率', '价格vsEMA%', '张力', '加速度',
    '波峰波谷', '持仓状态', '最优动作', '黄金信号'
]

df[output_cols].to_csv('价格最优路径_纯价格版.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 价格最优路径_纯价格版.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
