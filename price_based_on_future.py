# -*- coding: utf-8 -*-
"""
基于未来收盘价的后验最优路径
==========================

逻辑：
- 看未来N根K线的价格变化
- 如果未来暴涨 → 当前应该平空反多
- 如果未来暴跌 → 当前应该平多反空
"""

import pandas as pd
import numpy as np

print("="*120)
print("FUTURE-BASED OPTIMAL TRADING PATH")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# 基于未来价格变化的决策
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Look at Future Price Changes")
print("="*120)

LOOKAHEAD = 4  # 看未来4根K线
MIN_CHANGE = 0.015  # 最小涨跌幅1.5%

actions = []
positions = []
current_position = 'NONE'
entry_price = None

for i in range(len(df)):
    current_close = df.loc[i, '收盘价']
    current_time = df.loc[i, '时间']

    # 计算未来价格变化
    if i < len(df) - LOOKAHEAD:
        future_closes = df.loc[i+1:i+LOOKAHEAD+1, '收盘价'].values
        future_max = np.max(future_closes)
        future_min = np.min(future_closes)

        # 未来涨幅
        future_up_pct = (future_max - current_close) / current_close
        # 未来跌幅
        future_down_pct = (current_close - future_min) / current_close
    else:
        future_up_pct = 0
        future_down_pct = 0

    # 计算当前盈亏
    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (current_close - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_close) / entry_price * 100
    else:
        pnl_pct = 0

    # 决策逻辑
    action = ''

    if current_position == 'NONE':
        # 初始判断：看未来是涨还是跌
        if future_up_pct > MIN_CHANGE:
            action = f'开多（未来将涨{future_up_pct*100:.1f}%）'
            current_position = 'LONG'
            entry_price = current_close
        elif future_down_pct > MIN_CHANGE:
            action = f'开空（未来将跌{future_down_pct*100:.1f}%）'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = '观望'

    elif current_position == 'LONG':
        # 持多仓：看未来是否将暴跌
        if future_down_pct > MIN_CHANGE:
            action = f'平多/反空（未来将跌{future_down_pct*100:.1f}%,当前盈亏{pnl_pct:+.2f}%）'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = f'继续持多（盈亏{pnl_pct:+.2f}%）'

    elif current_position == 'SHORT':
        # 持空仓：看未来是否将暴涨
        if future_up_pct > MIN_CHANGE:
            action = f'平空/反多（未来将涨{future_up_pct*100:.1f}%,当前盈亏{pnl_pct:+.2f}%）'
            current_position = 'LONG'
            entry_price = current_close
        else:
            action = f'继续持空（盈亏{pnl_pct:+.2f}%）'

    actions.append(action)
    positions.append(current_position)

df['后验最优动作'] = actions
df['后验持仓'] = positions

# ============================================================================
# 验证用户例子
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Verify User's Example")
print("="*120)

test_cases = [
    ('2025-08-19 20:00', '开多'),
    ('2025-08-20 16:00', '平多反空'),
    ('2025-08-22 08:00', '平空反多')
]

print("\n验证用户例子:")
for time_str, expected in test_cases:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        print(f"\n{df.loc[idx, '时间']}: 收盘{df.loc[idx, '收盘价']:.2f}")
        print(f"  用户期望: {expected}")

        # 显示未来价格
        if idx < len(df) - LOOKAHEAD:
            future_closes = df.loc[idx+1:idx+LOOKAHEAD+1, '收盘价'].values
            future_max = np.max(future_closes)
            future_min = np.min(future_closes)
            print(f"  未来{LOOKAHEAD}根K线: 最高{future_max:.2f}, 最低{future_min:.2f}")

        print(f"  算法结果: {df.loc[idx, '后验最优动作'][:40]}")

# ============================================================================
# 显示前20个结果
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Results (First 20)")
print("="*120)

print(f"\n{'序号':<6} {'时间':<18} {'收盘价':<10} {'持仓':<8} {'动作':<45}")
print("-" * 120)

for i in range(20):
    time_str = str(df.loc[i, '时间'])[:16]
    close = df.loc[i, '收盘价']
    pos = df.loc[i, '后验持仓']
    action = df.loc[i, '后验最优动作'][:42]

    print(f"{i:<6} {time_str:<18} {close:<10.2f} {pos:<8} {action:<45}")

# ============================================================================
# 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Save Results")
print("="*120)

output_cols = [
    '时间', '开盘价', '最高价', '最低价', '收盘价',
    '信号类型', '量能比率', '价格vsEMA%', '张力', '加速度',
    '后验持仓', '后验最优动作', '黄金信号'
]

df[output_cols].to_csv('后验最优路径_基于未来价格.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 后验最优路径_基于未来价格.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)

print(f"""
算法逻辑：
- 看未来{LOOKAHEAD}根K线
- 如果未来涨 > {MIN_CHANGE*100}% → 当前应该做多/平空反多
- 如果未来跌 > {MIN_CHANGE*100}% → 当前应该做空/平多反空
- 这是基于未来价格的后验最优路径

这个算法是否正确？
""")
