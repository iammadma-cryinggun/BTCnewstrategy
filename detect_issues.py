# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('带信号标记_关键行.csv', encoding='utf-8-sig')

print('='*120)
print('检测开仓平仓逻辑问题')
print('='*120)

issues = []

for i in range(len(df)):
    row = df.iloc[i]
    action = row['信号动作']

    if pd.notna(action) and '开多' in action:
        entry_price = row['入场价']
        status = row['策略状态']

        print(f"\n行{i+2}: {row['时间']}")
        print(f"  动作: {action}")
        print(f"  入场价: {entry_price}")
        print(f"  策略状态: {status}")

        # 检查问题
        if entry_price == 0:
            issues.append({
                'row': i+2,
                'time': row['时间'],
                'issue': '开仓但入场价为0',
                'action': action
            })
            print(f"  ⚠️ 问题: 开仓动作但入场价为0")

        if status == '空仓':
            issues.append({
                'row': i+2,
                'time': row['时间'],
                'issue': '开仓但策略状态为空仓',
                'action': action
            })
            print(f"  ⚠️ 问题: 开仓动作但策略状态为空仓")

print('\n' + '='*120)
print(f'发现 {len(issues)} 个问题')
print('='*120)

for issue in issues:
    print(f"行{issue['row']} ({issue['time']}): {issue['issue']} - {issue['action']}")
