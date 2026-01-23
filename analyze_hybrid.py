# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('混合策略_回测结果.csv', encoding='utf-8-sig')

print('='*80)
print('正常交易 vs 黑天鹅交易对比')
print('='*80)

normal = df[~df['is_black_swan']]
bs = df[df['is_black_swan']]

print(f'\n正常交易: {len(normal)}笔')
if len(normal) > 0:
    print(f'胜率: {(len(normal[normal["pnl_pct"]>0])/len(normal)*100):.1f}%')
    print(f'平均盈亏: {normal["pnl_pct"].mean():+.2f}%')
    print(f'总盈亏: ${normal["pnl_usd"].sum():+,.2f}')

print(f'\n黑天鹅交易: {len(bs)}笔')
if len(bs) > 0:
    print(f'胜率: {(len(bs[bs["pnl_pct"]>0])/len(bs)*100):.1f}%')
    print(f'平均盈亏: {bs["pnl_pct"].mean():+.2f}%')
    print(f'总盈亏: ${bs["pnl_usd"].sum():+,.2f}')

print('\n' + '='*80)
print('黑天鹅交易详情:')
print('='*80)
print(bs[['entry_time', 'entry_price', 'exit_price', 'pnl_pct', 'pnl_usd', 'exit_reason', 'hold_hours']])

print('\n' + '='*80)
print('亏损交易详情（正常）:')
print('='*80)
losing_normal = normal[normal['pnl_pct'] <= 0].sort_values('pnl_pct')
print(losing_normal[['entry_time', 'pnl_pct', 'pnl_usd', 'exit_reason', 'hold_hours']].head(10))
