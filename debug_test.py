# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

peak_valley_labels = []
for i in range(len(df)):
    if i in local_max_indices:
        peak_valley_labels.append('高点')
    elif i in local_min_indices:
        peak_valley_labels.append('低点')
    else:
        peak_valley_labels.append('')

df['高低点'] = peak_valley_labels

def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# Test first few iterations
actions = []
positions = []
current_position = 'NONE'
entry_price = None
prev_signal_mode = None

for i in range(10):
    signal_type = df.loc[i, '信号类型']
    signal_mode = df.loc[i, '信号模式']
    current_close = df.loc[i, '收盘价']
    is_peak = (df.loc[i, '高低点'] == '高点')
    is_valley = (df.loc[i, '高低点'] == '低点')

    if entry_price is not None:
        if current_position == 'LONG':
            pnl_pct = (current_close - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_close) / entry_price * 100
    else:
        pnl_pct = 0

    action = ''
    is_new_signal_mode = (prev_signal_mode != signal_mode)

    if i == 0:
        if signal_mode == 'LONG_MODE':
            action = '开多'
            current_position = 'LONG'
            entry_price = current_close
        elif signal_mode == 'SHORT_MODE':
            action = '开空'
            current_position = 'SHORT'
            entry_price = current_close
        else:
            action = '空仓(震荡)'
        actions.append(action)
        positions.append(current_position)
        prev_signal_mode = signal_mode
        continue

    print(f'i={i}, time={df.loc[i, "时间"]}, signal_mode={signal_mode}, prev={prev_signal_mode}, is_new={is_new_signal_mode}, pos={current_position}, peak={is_peak}, valley={is_valley}')

    if signal_mode == 'LONG_MODE':
        if current_position == 'SHORT':
            action = f'平空(信号切换,盈亏{pnl_pct:+.2f}%)'
            current_position = 'NONE'
            entry_price = None
        elif current_position == 'LONG':
            if is_peak:
                action = f'平多(高点,盈亏{pnl_pct:+.2f}%)'
                current_position = 'NONE'
                entry_price = None
            elif is_valley:
                action = f'开多(低点刷新)'
                entry_price = current_close
            else:
                action = f'继续持多({pnl_pct:+.2f}%)'
        else:
            if is_new_signal_mode:
                action = f'开多(新信号)'
                current_position = 'LONG'
                entry_price = current_close
            elif is_valley:
                action = f'开多(低点)'
                current_position = 'LONG'
                entry_price = current_close
            elif is_peak:
                action = f'观望(高点)'
            else:
                action = '空仓等待'

    print(f'  action="{action}"')
    actions.append(action)
    positions.append(current_position)
    prev_signal_mode = signal_mode

print('\nFirst 10 actions:', actions)
