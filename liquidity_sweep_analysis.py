# -*- coding: utf-8 -*-
"""
流动性猎杀分析 - Aug 19 神单完整复盘
验证宽止损的必要性
"""

import pandas as pd
import numpy as np

df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 找到2025-08-19信号位置
signal_idx = df[df['时间'] == '2025-08-19 20:00:00'].index[0]

print('='*120)
print('Aug 19 神单 - 流动性猎杀完整复盘')
print('='*120)

signal = df.loc[signal_idx]
print(f'\n【信号K线】2025-08-19 20:00')
print(f'收盘价: ${signal["收盘价"]:,.2f}')
print(f'最高价: ${signal["最高价"]:,.2f}')
print(f'最低价: ${signal["最低价"]:,.2f}')
print(f'挂单价位: ${signal["最高价"] * 1.0001:,.2f} (最高价×1.0001)')

# 检查后续10根K线
print(f'\n【后续K线微观结构】')
print(f'{'时间':<20} {'开盘':<12} {'最高':<12} {'最低':<15} {'收盘':<12} {'振幅':<10}')
print('-'*120)

entry_bar = None
for i in range(signal_idx + 1, min(signal_idx + 11, len(df))):
    bar = df.loc[i]
    high = bar['最高价']
    low = bar['最低价']
    close = bar['收盘价']
    amplitude = (high - low) / close * 100

    status = ''
    if entry_bar is None and high >= signal['最高价'] * 1.0001:
        entry_bar = i
        status = ' [成交]'

    print(f'{str(bar["时间"]):<20} ${bar["开盘价"]:>10,.2f} ${high:>10,.2f} ${low:>13,.2f} ${close:>10,.2f} {amplitude:>6.2f}%{status}')

if entry_bar:
    entry_price = df.loc[entry_bar, '收盘价']
    print(f'\n【成交时刻】2025-08-20 00:00')
    print(f'成交价: ${entry_price:,.2f}')
    print(f'止损价(2.5%): ${entry_price * 0.975:,.2f}')
    print(f'止损价(2.0%): ${entry_price * 0.98:,.2f}')
    print(f'止损价(1.5%): ${entry_price * 0.985:,.2f}')

    # 检查是否被扫损
    print(f'\n【扫损分析】')
    min_low = df.loc[entry_bar:min(entry_bar+5, len(df)-1), '最低价'].min()
    min_idx = df.loc[entry_bar:min(entry_bar+5, len(df)-1), '最低价'].idxmin()
    sweep_time = df.loc[min_idx, '时间']

    print(f'前5根K线最低价: ${min_low:,.2f} (发生在 {sweep_time})')

    sl_25 = entry_price * 0.975
    sl_20 = entry_price * 0.98
    sl_15 = entry_price * 0.985

    if min_low <= sl_25:
        loss_25 = (min_low - entry_price) / entry_price * 100
        print('[X] 2.5%止损被扫！最低价 ${:.2f} < ${:.2f} (亏损{:.2f}%)'.format(min_low, sl_25, loss_25))
    else:
        print('[OK] 2.5%止损存活！最低价 ${:.2f} > ${:.2f}'.format(min_low, sl_25))

    if min_low <= sl_20:
        loss_20 = (min_low - entry_price) / entry_price * 100
        print('[X] 2.0%止损被扫！最低价 ${:.2f} < ${:.2f} (亏损{:.2f}%)'.format(min_low, sl_20, loss_20))
    else:
        print('[OK] 2.0%止损存活！最低价 ${:.2f} > ${:.2f}'.format(min_low, sl_20))

    if min_low <= sl_15:
        loss_15 = (min_low - entry_price) / entry_price * 100
        print('[X] 1.5%止损被扫！最低价 ${:.2f} < ${:.2f} (亏损{:.2f}%)'.format(min_low, sl_15, loss_15))
    else:
        print('[OK] 1.5%止损存活！最低价 ${:.2f} > ${:.2f}'.format(min_low, sl_15))

    # 检查最终涨幅
    max_high = df.loc[entry_bar:min(entry_bar+21, len(df)-1), '最高价'].max()
    max_idx = df.loc[entry_bar:min(entry_bar+21, len(df)-1), '最高价'].idxmax()
    max_time = df.loc[max_idx, '时间']
    max_gain = (max_high - entry_price) / entry_price * 100

    print(f'\n【最终暴涨】')
    print(f'最高价: ${max_high:,.2f} (发生在 {max_time})')
    print(f'最大涨幅: +{max_gain:.2f}%')
    print(f'最大收益(50%仓位): +${max_gain/100 * 5000:,.2f}')

    # 张力释放分析
    print(f'\n【张力释放离场分析】')
    for i in range(entry_bar, min(entry_bar+42, len(df))):
        tension = df.loc[i, '张力']
        if tension < 0.40:
            exit_bar = df.loc[i]
            exit_price = exit_bar['收盘价']
            exit_gain = (exit_price - entry_price) / entry_price * 100
            print(f'张力释放时刻: {exit_bar["时间"]}')
            print(f'张力值: {tension:.3f}')
            print(f'离场价: ${exit_price:,.2f}')
            print(f'收益: +{exit_gain:.2f}%')
            print(f'收益(50%仓位): +${exit_gain/100 * 5000:,.2f}')
            break

print('\n' + '='*120)
print('结论: 宽止损(2%)是必须的，否则会被流动性猎杀')
print('='*120)
