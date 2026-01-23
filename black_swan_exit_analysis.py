# -*- coding: utf-8 -*-
"""
黑天鹅信号 - 最优离场策略分析
基于唯一的100%胜率信号（2025-08-19 20:00）
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("黑天鹅信号 - 最优离场策略分析")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

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
# 识别黑天鹅信号（严格参数）
# ==============================================================================

print("\n" + "="*120)
print("黑天鹅信号识别（严格参数 + 双层防御）")
print("="*120)

# 计算下影线
df['下影线'] = df.apply(lambda row: (row['收盘价'] - row['最低价']) / row['收盘价'] if row['收盘价'] > row['最低价'] else 0, axis=1)

# 第一层防御：物理过滤器（用户真实参数）
black_swan_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] < -0.20) &
    (df['张力'] > 0.70) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

potential_signals = df[black_swan_conditions].copy()

print(f"\n通过第一层防御（物理过滤器）: {len(potential_signals)} 个")

if len(potential_signals) > 0:
    print(f"\n{'时间':<20} {'收盘价':<12} {'下影线':<10} {'加速度':<12} {'张力':<10} {'量能':<10}")
    print("-" * 100)

    for idx, row in potential_signals.iterrows():
        print(f"{str(row['时间'])[:18]:<20} "
              f"${row['收盘价']:>10.2f} "
              f"{row['下影线']:>8.3f} "
              f"{row['加速度']:>10.4f} "
              f"{row['张力']:>8.3f} "
              f"{row['量能比率']:>8.2f}")

# ==============================================================================
# 第二层防御：结构过滤器（需要确认）
# ==============================================================================

print("\n" + "="*120)
print("第二层防御：结构过滤器（需要下一根K线突破）")
print("="*120)

confirmed_signals = []
rejected_signals = []

for idx, row in potential_signals.iterrows():
    signal_time = row['时间']
    signal_price = row['收盘价']
    signal_high = df.loc[idx, '最高价']  # 当前K线的最高价

    # 检查下一根K线是否突破
    if idx + 1 < len(df):
        next_bar = df.loc[idx + 1]
        next_high = next_bar['最高价']
        next_close = next_bar['收盘价']

        # 突破判断：下一根K线的最高价 > 当前K线的最高价
        if next_high > signal_high:
            confirmed_signals.append({
                'signal_idx': idx,
                'entry_idx': idx + 1,
                'entry_time': next_bar['时间'],
                'entry_price': next_close,  # 用下一根K线的收盘价作为入场价
                'signal_time': signal_time,
                'signal_price': signal_price,
                'acceleration': row['加速度'],
                'tension': row['张力'],
                'volume_ratio': row['量能比率'],
                'shadow': row['下影线']
            })
            print(f"\n[OK] {signal_time} - 确认！下一根K线突破 (${signal_high:.2f} -> ${next_high:.2f})")
        else:
            rejected_signals.append({
                'signal_idx': idx,
                'signal_time': signal_time,
                'signal_price': signal_price,
                'signal_high': signal_high,
                'next_high': next_high,
                'reason': '下一根K线未突破'
            })
            print(f"\n[X] {signal_time} - 拒绝！下一根K线未突破 (${signal_high:.2f} -> ${next_high:.2f})")
    else:
        rejected_signals.append({
            'signal_idx': idx,
            'signal_time': signal_time,
            'signal_price': signal_price,
            'reason': '没有下一根K线'
        })
        print(f"\n[X] {signal_time} - 拒绝！没有下一根K线")

print(f"\n通过第二层防御（结构过滤器）: {len(confirmed_signals)} 个")
print(f"被第二层防御拒绝: {len(rejected_signals)} 个")

# ==============================================================================
# 最优离场策略回测
# ==============================================================================

if len(confirmed_signals) == 0:
    print("\n没有确认的黑天鹅信号，无法进行离场策略分析。")
else:
    print("\n" + "="*120)
    print("最优离场策略回测")
    print("="*120)

    for signal in confirmed_signals:
        entry_idx = signal['entry_idx']
        entry_price = signal['entry_price']
        entry_time = signal['entry_time']
        signal_accel = signal['acceleration']
        signal_tension = signal['tension']

        print(f"\n{'='*120}")
        print(f"分析信号: {entry_time}")
        print(f"入场价: ${entry_price:.2f}")
        print(f"信号加速度: {signal_accel:.4f}")
        print(f"信号张力: {signal_tension:.3f}")
        print(f"{'='*120}")

        # 获取后续数据
        future_data = df[entry_idx+1:min(entry_idx+42, len(df))].copy()  # 最多看42个周期（7天）

        if len(future_data) == 0:
            print("没有后续数据")
            continue

        # 初始化
        INITIAL_CAPITAL = 10000
        POSITION_SIZE = INITIAL_CAPITAL * 0.50  # 50%仓位

        # ====================================================================
        # 策略1: 固定止盈止损
        # ====================================================================

        print(f"\n【策略1: 固定止盈止损】")

        tp_pcts = [0.02, 0.03, 0.05, 0.08, 0.10]  # 2%, 3%, 5%, 8%, 10%
        sl_pct = 0.025  # 2.5%止损

        for tp_pct in tp_pcts:
            exit_price = None
            exit_idx = None
            exit_reason = None
            hold_bars = 0

            for i, (_, row) in enumerate(future_data.iterrows()):
                hold_bars = i + 1
                pnl = (row['收盘价'] - entry_price) / entry_price

                # 止盈
                if pnl >= tp_pct:
                    exit_price = row['收盘价']
                    exit_idx = entry_idx + i + 1
                    exit_reason = f'止盈({tp_pct*100:.0f}%)'
                    break

                # 止损
                if pnl <= -sl_pct:
                    exit_price = row['收盘价']
                    exit_idx = entry_idx + i + 1
                    exit_reason = f'止损({sl_pct*100:.1f}%)'
                    break

            # 如果没有触发止盈止损，用最后的价格
            if exit_price is None:
                exit_price = future_data.iloc[-1]['收盘价']
                exit_idx = entry_idx + len(future_data)
                exit_reason = '到期(42周期)'

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_usd = POSITION_SIZE * (pnl_pct / 100)

            print(f"  TP{tp_pct*100:.0f}%/SL{sl_pct*100:.0f}%: "
                  f"持有{hold_bars}周期 → {exit_reason}, "
                  f"收益{pnl_pct:+.2f}%, "
                  f"${pnl_usd:+.2f}")

        # ====================================================================
        # 策略2: 张力释放
        # ====================================================================

        print(f"\n【策略2: 张力释放】")

        # 计算后续的张力
        for i, (_, row) in enumerate(future_data.iterrows()):
            hold_bars = i + 1
            current_tension = row['张力']
            pnl_pct = (row['收盘价'] - entry_price) / entry_price * 100

            # 张力从正变负（释放完成）
            if current_tension < 0:
                exit_price = row['收盘价']
                exit_idx = entry_idx + i + 1
                exit_reason = f'张力释放({signal_tension:.3f}→{current_tension:.3f})'

                pnl_usd = POSITION_SIZE * (pnl_pct / 100)

                print(f"  张力释放: "
                      f"持有{hold_bars}周期 → {exit_reason}, "
                      f"收益{pnl_pct:+.2f}%, "
                      f"${pnl_usd:+.2f}")
                break
        else:
            # 如果张力没有释放，用最后的价格
            pnl_pct = (future_data.iloc[-1]['收盘价'] - entry_price) / entry_price * 100
            pnl_usd = POSITION_SIZE * (pnl_pct / 100)
            print(f"  张力未释放: 持有{len(future_data)}周期 → 收益{pnl_pct:+.2f}%, ${pnl_usd:+.2f}")

        # ====================================================================
        # 策略3: 局部高点平仓
        # ====================================================================

        print(f"\n【策略3: 局部高点平仓】")

        # 在未来数据中找高点
        future_peaks = future_data[future_data['高低点'] == '高点']

        if len(future_peaks) > 0:
            first_peak = future_peaks.iloc[0]
            peak_idx = future_data.index.get_loc(first_peak.name) + 1
            exit_price = first_peak['收盘价']
            exit_reason = '局部高点'
            hold_bars = peak_idx

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_usd = POSITION_SIZE * (pnl_pct / 100)

            print(f"  第一个高点: "
                  f"持有{hold_bars}周期 → {exit_reason}, "
                  f"收益{pnl_pct:+.2f}%, "
                  f"${pnl_usd:+.2f}")
        else:
            pnl_pct = (future_data.iloc[-1]['收盘价'] - entry_price) / entry_price * 100
            pnl_usd = POSITION_SIZE * (pnl_pct / 100)
            print(f"  无高点: 持有{len(future_data)}周期 → 收益{pnl_pct:+.2f}%, ${pnl_usd:+.2f}")

        # ====================================================================
        # 策略4: 时间止盈（持有N个周期）
        # ====================================================================

        print(f"\n【策略4: 时间止盈】")

        hold_periods_options = [1, 2, 3, 5, 7, 10, 14, 21]  # 周期数

        for hold_periods in hold_periods_options:
            if hold_periods <= len(future_data):
                exit_bar = future_data.iloc[hold_periods - 1]
                exit_price = exit_bar['收盘价']
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_usd = POSITION_SIZE * (pnl_pct / 100)
                print(f"  持有{hold_periods}周期: 收益{pnl_pct:+.2f}%, ${pnl_usd:+.2f}")
            else:
                exit_price = future_data.iloc[-1]['收盘价']
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_usd = POSITION_SIZE * (pnl_pct / 100)
                print(f"  持有{hold_periods}周期: 数据不足 → 收益{pnl_pct:+.2f}%, ${pnl_usd:+.2f}")

        # ====================================================================
        # 策略5: 动态追踪止损
        # ====================================================================

        print(f"\n【策略5: 动态追踪止损】")

        trailing_pcts = [0.01, 0.02, 0.03]  # 1%, 2%, 3%

        for trailing_pct in trailing_pcts:
            highest_pnl = 0
            exit_price = None
            exit_idx = None

            for i, (_, row) in enumerate(future_data.iterrows()):
                current_pnl = (row['收盘价'] - entry_price) / entry_price

                # 更新最高盈利
                if current_pnl > highest_pnl:
                    highest_pnl = current_pnl

                # 回撤止损
                if (highest_pnl - current_pnl) >= trailing_pct:
                    exit_price = row['收盘价']
                    exit_idx = entry_idx + i + 1
                    hold_bars = i + 1
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    pnl_usd = POSITION_SIZE * (pnl_pct / 100)
                    print(f"  追踪止损{trailing_pct*100:.0f}%: "
                          f"持有{hold_bars}周期 → 回撤{trailing_pct*100:.0f}%, "
                          f"收益{pnl_pct:+.2f}%, "
                          f"${pnl_usd:+.2f}")
                    break
            else:
                # 没有触发止损
                pnl_pct = (future_data.iloc[-1]['收盘价'] - entry_price) / entry_price * 100
                pnl_usd = POSITION_SIZE * (pnl_pct / 100)
                print(f"  追踪止损{trailing_pct*100:.0f}%: 未触发 → 收益{pnl_pct:+.2f}%, ${pnl_usd:+.2f}")

print("\n" + "="*120)
print("离场策略分析完成")
print("="*120)
