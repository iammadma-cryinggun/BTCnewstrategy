# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('带信号标记_完整数据.csv', encoding='utf-8-sig')

print('='*80)
print('完整策略回测分析')
print('='*80)

print(f'\n数据范围: {df["时间"].min()} 到 {df["时间"].max()}')
print(f'总K线数: {len(df)}')

# 信号统计
print('\n' + '='*80)
print('信号类型统计')
print('='*80)
signal_counts = df['信号类型'].value_counts()
for signal, count in signal_counts.items():
    print(f'{signal}: {count}')

# 黑天鹅统计
black_swans = df[df['是黑天鹅'] == '★黑天鹅']
print(f'\n黑天鹅信号: {len(black_swans)} 个')

# 交易统计
trades = df[df['信号动作'].notna()]
print(f'\n交易动作总数: {len(trades)}')

print('\n' + '='*80)
print('交易明细')
print('='*80)

open_trades = df[df['信号动作'].str.contains('开多|开空', na=False)]
close_trades = df[df['信号动作'].str.contains('平仓', na=False)]

print(f'\n开仓次数: {len(open_trades)}')
print(f'平仓次数: {len(close_trades)}')

# 计算每笔交易的盈亏
print('\n' + '='*80)
print('平仓交易盈亏明细')
print('='*80)
print(f'{"时间":<20} {"动作":<25} {"入场价":>10} {"出场价":>10} {"盈亏%":>10}')
print('-'*80)

total_pnl = 0
win_count = 0
loss_count = 0

for idx, row in close_trades.iterrows():
    action = row['信号动作']
    entry_price = row['入场价']
    exit_price = row['收盘价']

    if pd.notna(entry_price) and entry_price > 0:
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        total_pnl += pnl_pct

        if pnl_pct > 0:
            win_count += 1
        else:
            loss_count += 1

        print(f'{str(row["时间"])[:19]:<20} {action:<25} {entry_price:>10.2f} {exit_price:>10.2f} {pnl_pct:>+9.2f}%')

print('\n' + '='*80)
print('总体统计')
print('='*80)
print(f'总交易次数: {len(close_trades)}')
print(f'盈利交易: {win_count}')
print(f'亏损交易: {loss_count}')
if len(close_trades) > 0:
    win_rate = win_count / len(close_trades) * 100
    print(f'胜率: {win_rate:.1f}%')
    print(f'平均每笔盈亏: {total_pnl/len(close_trades):+.2f}%')
    print(f'总盈亏: {total_pnl:+.2f}%')
