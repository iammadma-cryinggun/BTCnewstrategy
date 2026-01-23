# -*- coding: utf-8 -*-
"""
黑天鹅猎手 - 真实参数回测
基于用户的完整设计
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("黑天鹅猎手 - 真实参数回测")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 计算下影线
df['下影线'] = df.apply(lambda row: (row['收盘价'] - row['最低价']) / row['收盘价'] if row['收盘价'] > row['最低价'] else 0, axis=1)

# 识别极值点
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

df['高低点'] = ''
for i in local_max_indices:
    df.loc[i, '高低点'] = '高点'
for i in local_min_indices:
    df.loc[i, '高低点'] = '低点'

# 定义信号模式
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# ==============================================================================
# 黑天鹅信号检测（用户真实参数）
# ==============================================================================

print("\n" + "="*120)
print("黑天鹅信号检测（用户真实参数）")
print("="*120)

print("""
核心信号（4个条件，全部满足）：
1. 加速度 < -0.20（极速撞击）
2. 张力 > 0.70（高强度）
3. 下影线 < 0.35（光脚）
4. 量能 > 1.0（恐慌放量）
""")

black_swan_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] < -0.20) &
    (df['张力'] > 0.70) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

signals = df[black_swan_conditions].copy()

print(f"找到信号: {len(signals)} 个")

if len(signals) > 0:
    print(f"\n{'时间':<20} {'收盘价':<12} {'最高价':<12} {'下影线':<10} {'加速度':<12} {'张力':<10} {'量能':<10}")
    print("-" * 120)

    for idx, row in signals.iterrows():
        signal_high = row['最高价']
        entry_price = signal_high * 1.0001

        print(f"{str(row['时间'])[:18]:<20} "
              f"${row['收盘价']:>10.2f} "
              f"${signal_high:>10.2f} "
              f"{row['下影线']:>8.3f} "
              f"{row['加速度']:>10.4f} "
              f"{row['张力']:>8.3f} "
              f"{row['量能比率']:>8.2f}")

        print(f"    → 挂单价: ${entry_price:.2f} (最高价×1.0001)")

        # 检查是否成交
        # 在后续10根K线内查找是否有突破
        filled = False
        fill_price = None
        fill_bar = None

        for i in range(idx + 1, min(idx + 11, len(df))):
            if df.loc[i, '最高价'] >= entry_price:
                filled = True
                fill_price = df.loc[i, '收盘价']  # 用收盘价作为成交价
                fill_bar = i
                break

        if filled:
            fill_time = df.loc[fill_bar, '时间']
            hold_bars = fill_bar - idx

            # 计算后续表现
            max_pnl = 0
            max_pnl_bar = 0
            final_pnl = 0

            for i in range(fill_bar, min(fill_bar + 21, len(df))):
                future_close = df.loc[i, '收盘价']
                pnl = (future_close - fill_price) / fill_price * 100

                if pnl > max_pnl:
                    max_pnl = pnl
                    max_pnl_bar = i - fill_bar

                if i - fill_bar == 20:
                    final_pnl = pnl

            print(f"    [OK] 成交! {fill_time} @ ${fill_price:.2f}")
            print(f"    后续20周期最大收益: +{max_pnl:.2f}% (第{max_pnl_bar}周期)")
            print(f"    第20周期收益: {final_pnl:+.2f}%")
        else:
            print(f"    [X] 未成交（没有突破）")

        print()
else:
    print("\n未找到符合条件的信号")
