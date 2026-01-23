# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print('='*100)
print('CORRECTING STRATEGY ERRORS IN CSV FILES')
print('='*100)

# 读取文件
df_key = pd.read_csv('带信号标记_关键行.csv', encoding='utf-8-sig')
df_full = pd.read_csv('带信号标记_完整数据.csv', encoding='utf-8-sig')

print('\n[1] Processing: 带信号标记_关键行.csv')
print(f'Original rows: {len(df_key)}')

errors_fixed = 0

# 错误1: 2025-09-19 00:00 BULLISH_SINGULARITY 开多 -> 应该开空
mask1 = (df_key['时间'] == '2025-09-19 00:00:00') & (df_key['信号类型'] == 'BULLISH_SINGULARITY')
if mask1.any():
    idx = df_key[mask1].index[0]
    print(f'\n  Fixing Error 1: 2025-09-19 00:00')
    print(f'    Before: {df_key.loc[idx, "信号动作"]}')
    df_key.loc[idx, '信号动作'] = '开空(BULLISH反转 150%仓位)'
    df_key.loc[idx, '策略状态'] = '空仓'
    df_key.loc[idx, '持仓类型'] = ''
    print(f'    After: {df_key.loc[idx, "信号动作"]}')
    errors_fixed += 1

# 错误2: 2025-12-28 00:00 BULLISH_SINGULARITY 开多 -> 应该开空
mask2 = (df_key['时间'] == '2025-12-28 00:00:00') & (df_key['信号类型'] == 'BULLISH_SINGULARITY')
if mask2.any():
    idx = df_key[mask2].index[0]
    print(f'\n  Fixing Error 2: 2025-12-28 00:00')
    print(f'    Before: {df_key.loc[idx, "信号动作"]}')
    df_key.loc[idx, '信号动作'] = '开空(BULLISH反转 150%仓位)'
    df_key.loc[idx, '策略状态'] = '空仓'
    df_key.loc[idx, '持仓类型'] = ''
    print(f'    After: {df_key.loc[idx, "信号动作"]}')
    errors_fixed += 1

# 错误3: 2025-12-09 20:00 OSCILLATION+NO_TRADE 开多 -> 应该不交易
mask3 = (df_key['时间'] == '2025-12-09 20:00:00') & (df_key['信号类型'] == 'OSCILLATION')
if mask3.any():
    idx = df_key[mask3].index[0]
    print(f'\n  Fixing Error 3: 2025-12-09 20:00')
    print(f'    Before: {df_key.loc[idx, "信号动作"]}')
    df_key.loc[idx, '信号动作'] = ''
    df_key.loc[idx, '策略状态'] = '空仓'
    df_key.loc[idx, '持仓类型'] = ''
    print(f'    After: (no action)')
    errors_fixed += 1

print(f'\nFixed {errors_fixed} errors in 带信号标记_关键行.csv')

# 保存修正后的文件
df_key.to_csv('带信号标记_关键行_修复版.csv', index=False, encoding='utf-8-sig')
print('Saved: 带信号标记_关键行_修复版.csv')

# 修正带信号标记_完整数据.csv
print('\n[2] Processing: 带信号标记_完整数据.csv')
print(f'Original rows: {len(df_full)}')

errors_fixed_full = 0

# 错误1: 2025-09-19 00:00
mask1_full = (df_full['时间'] == '2025-09-19 00:00:00') & (df_full['信号类型'] == 'BULLISH_SINGULARITY')
if mask1_full.any():
    idx = df_full[mask1_full].index[0]
    print(f'\n  Fixing Error 1: 2025-09-19 00:00')
    print(f'    Before: {df_full.loc[idx, "信号动作"]}')
    df_full.loc[idx, '信号动作'] = '开空(BULLISH反转 150%仓位)'
    df_full.loc[idx, '策略状态'] = '空仓'
    df_full.loc[idx, '持仓类型'] = ''
    print(f'    After: {df_full.loc[idx, "信号动作"]}')
    errors_fixed_full += 1

# 错误2: 2025-12-28 00:00
mask2_full = (df_full['时间'] == '2025-12-28 00:00:00') & (df_full['信号类型'] == 'BULLISH_SINGULARITY')
if mask2_full.any():
    idx = df_full[mask2_full].index[0]
    print(f'\n  Fixing Error 2: 2025-12-28 00:00')
    print(f'    Before: {df_full.loc[idx, "信号动作"]}')
    df_full.loc[idx, '信号动作'] = '开空(BULLISH反转 150%仓位)'
    df_full.loc[idx, '策略状态'] = '空仓'
    df_full.loc[idx, '持仓类型'] = ''
    print(f'    After: {df_full.loc[idx, "信号动作"]}')
    errors_fixed_full += 1

# 错误3: 2025-12-09 20:00
mask3_full = (df_full['时间'] == '2025-12-09 20:00:00') & (df_full['信号类型'] == 'OSCILLATION')
if mask3_full.any():
    idx = df_full[mask3_full].index[0]
    print(f'\n  Fixing Error 3: 2025-12-09 20:00')
    print(f'    Before: {df_full.loc[idx, "信号动作"]}')
    df_full.loc[idx, '信号动作'] = ''
    df_full.loc[idx, '策略状态'] = '空仓'
    df_full.loc[idx, '持仓类型'] = ''
    print(f'    After: (no action)')
    errors_fixed_full += 1

print(f'\nFixed {errors_fixed_full} errors in 带信号标记_完整数据.csv')

# 保存修正后的文件
df_full.to_csv('带信号标记_完整数据_修复版.csv', index=False, encoding='utf-8-sig')
print('Saved: 带信号标记_完整数据_修复版.csv')

print('\n' + '='*100)
print('CORRECTION COMPLETE')
print('='*100)
print(f'Total errors fixed: {errors_fixed}')
print(f'Output files:')
print(f'  - 带信号标记_关键行_修复版.csv')
print(f'  - 带信号标记_完整数据_修复版.csv')

# 验证修正结果
print('\n' + '='*100)
print('VERIFICATION')
print('='*100)

dates = ['2025-09-19 00:00:00', '2025-12-28 00:00:00', '2025-12-09 20:00:00']
for date in dates:
    row = df_key[df_key['时间'] == date]
    if not row.empty:
        row = row.iloc[0]
        print(f'\n{date}:')
        print(f'  Signal: {row["信号类型"]}')
        print(f'  Action: {row["信号动作"]}')

# 统计信号
print('\n' + '='*100)
print('FINAL STATISTICS')
print('='*100)
open_long = df_key[df_key['信号动作'].str.contains('开多', na=False)]
open_short = df_key[df_key['信号动作'].str.contains('开空', na=False)]
print(f'\nOpen LONG: {len(open_long)}')
print(f'Open SHORT: {len(open_short)}')
