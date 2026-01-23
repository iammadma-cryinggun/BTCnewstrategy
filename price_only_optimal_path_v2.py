# -*- coding: utf-8 -*-
"""
纯价格驱动的后验最优交易路径 V2
==============================

改进：
1. 第一行直接根据趋势方向开仓
2. 更宽松的波峰波谷识别
"""

import pandas as pd
import numpy as np

print("="*120)
print("PRICE-ONLY OPTIMAL TRADING PATH V2")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 改进的波峰波谷识别
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Improved Peak/Valley Detection")
print("="*120)

actions = []
positions = []
entry_prices = []

current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']

    # 计算盈亏
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

    if current_position == 'NONE':
        # 判断开仓方向 - 看前后几根K线的趋势
        if i == 0:
            # 第一行，看后3根
            future_closes = df.loc[1:min(4, len(df)), '收盘价'].values
            if len(future_closes) > 0 and future_closes[0] > current_close:
                # 后面涨，开多
                action = '开多'
                current_position = 'LONG'
                entry_price = current_close
            else:
                # 后面跌，开空
                action = '开空'
                current_position = 'SHORT'
                entry_price = current_close
        else:
            # 看前后趋势
            before_closes = df.loc[max(0, i-3):i, '收盘价'].values
            after_closes = df.loc[i+1:min(i+4, len(df)), '收盘价'].values

            # 判断是否是局部高点或低点
            is_local_peak = True
            is_local_valley = True

            if len(before_closes) > 0:
                for c in before_closes:
                    if current_close <= c:
                        is_local_peak = False
                    if current_close >= c:
                        is_local_valley = False

            if len(after_closes) > 0:
                for c in after_closes:
                    if current_close <= c:
                        is_local_peak = False
                    if current_close >= c:
                        is_local_valley = False

            if is_local_valley:
                action = '开多'
                current_position = 'LONG'
                entry_price = current_close
            elif is_local_peak:
                action = '开空'
                current_position = 'SHORT'
                entry_price = current_close
            else:
                action = '观望'

    elif current_position == 'LONG':
        # 持多仓 - 判断是否该平仓
        # 看后面几根K线，如果开始下跌就平多反空
        if i < len(df) - 3:
            future_closes = df.loc[i+1:i+4, '收盘价'].values
            # 如果后面连续下跌，且当前盈利
            if all(c < current_close for c in future_closes) and pnl_pct > 0.3:
                action = f'平多/反空 (盈利{pnl_pct:.2f}%)'
                current_position = 'SHORT'
                entry_price = current_close
            else:
                action = f'继续持多 (盈亏{pnl_pct:+.2f}%)'
        else:
            action = f'继续持多 (盈亏{pnl_pct:+.2f}%)'

    elif current_position == 'SHORT':
        # 持空仓 - 判断是否该平仓
        # 看后面几根K线，如果开始上涨就平空反多
        if i < len(df) - 3:
            future_closes = df.loc[i+1:i+4, '收盘价'].values
            # 如果后面连续上涨，且当前盈利
            if all(c > current_close for c in future_closes) and pnl_pct > 0.3:
                action = f'平空/反多 (盈利{pnl_pct:.2f}%)'
                current_position = 'LONG'
                entry_price = current_close
            else:
                action = f'继续持空 (盈亏{pnl_pct:+.2f}%)'
        else:
            action = f'继续持空 (盈亏{pnl_pct:+.2f}%)'

    actions.append(action)
    positions.append(current_position)
    entry_prices.append(entry_price)

df['最优动作'] = actions
df['持仓状态'] = positions

# ============================================================================
# 验证用户例子
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Verify User's Example")
print("="*120)

print("\n用户例子验证:")
print("8/19 20:00 收盘112873 → 应该: 开多")
print("8/20 16:00 收盘114277 → 应该: 平多反空")
print("8/22 8:00 收盘112320 → 应该: 平空反多")
print()

# 找到这些时间点
test_times = ['2025-08-19 20:00', '2025-08-20 16:00', '2025-08-22 08:00']

for test_time in test_times:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == test_time]
    if len(rows) > 0:
        idx = rows.index[0]
        print(f"{df.loc[idx, '时间']} 收盘{df.loc[idx, '收盘价']:.2f}")
        print(f"  算法: {df.loc[idx, '最优动作']}")
        print()

# ============================================================================
# 显示前20个
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Results (First 20)")
print("="*120)

print(f"\n{'序号':<6} {'时间':<18} {'收盘价':<10} {'持仓':<6} {'动作':<35}")
print("-" * 120)

for i in range(20):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    pos = df.loc[i, '持仓状态']
    action = df.loc[i, '最优动作'][:33]

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {pos:<6} {action:<35}")

# ============================================================================
# 保存
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Save Results")
print("="*120)

output_cols = [
    '时间', '开盘价', '最高价', '最低价', '收盘价',
    '信号类型', '量能比率', '价格vsEMA%', '张力', '加速度',
    '持仓状态', '最优动作', '黄金信号'
]

df[output_cols].to_csv('价格最优路径_V2.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 价格最优路径_V2.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
