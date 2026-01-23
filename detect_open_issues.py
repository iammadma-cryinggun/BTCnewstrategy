# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('带信号标记_关键行_修复版.csv', encoding='utf-8-sig')

print('检查开仓动作的问题:\n')
print('序号  时间                  动作                    状态    入场价')
print('-'*90)

for i, row in df.iterrows():
    action = row['信号动作']
    if pd.notna(action) and '开多' in action:
        entry = row['入场价']
        status = row['策略状态']
        print(f'{i+2:<4} {str(row["时间"]):<20} {action:<25} {status:<8} {entry}')

print('\n\n问题分析：')
print('有些"开多"动作的入场价为0或状态为空仓，说明这些信号并没有真正开仓')
print('原因：开仓逻辑在信号切换时触发，但某些切换是无效的')
