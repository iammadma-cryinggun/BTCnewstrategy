# -*- coding: utf-8 -*-
"""
检查交易序列是否有问题
"""
import pandas as pd

df = pd.read_csv('带信号标记_关键行.csv', encoding='utf-8-sig')

print("="*120)
print("交易序列分析")
print("="*120)

# 提取所有有动作的行
actions = df[df['信号动作'] != ''].copy()

print(f"\n总动作数: {len(actions)}")

# 按时间顺序列出所有动作
print(f"\n{'序号':<4} {'时间':<20} {'动作':<30} {'入场价':<12} {'当前盈亏%':<10}")
print("-" * 120)

sequence = []
for idx, row in actions.iterrows():
    action = row['信号动作']
    entry_price = row['入场价']
    pnl = row['当前盈亏%']

    print(f"{len(sequence)+1:<4} {str(row['时间'])[:18]:<20} {action:<30} "
          f"{entry_price:<12.2f} {pnl:<10.2f}")

    sequence.append({
        'time': row['时间'],
        'action': action,
        'entry_price': entry_price,
        'pnl': pnl
    })

# 检查异常情况
print("\n" + "="*120)
print("异常检测")
print("="*120)

errors = []
for i in range(len(sequence) - 1):
    curr = sequence[i]
    next_act = sequence[i + 1]

    # 检查1: 连续开仓
    if '开多' in curr['action'] and '开多' in next_act['action']:
        errors.append(f"[异常] 第{i+1}和{i+2}行连续开仓！")

    # 检查2: 连续平仓
    if '平仓' in curr['action'] and '平仓' in next_act['action']:
        errors.append(f"[异常] 第{i+1}和{i+2}行连续平仓！")

    # 检查3: 平仓后立即开仓（同一天）
    if '平仓' in curr['action'] and '开多' in next_act['action']:
        time1 = pd.to_datetime(curr['time'])
        time2 = pd.to_datetime(next_act['time'])
        if (time2 - time1).total_seconds() <= 4 * 3600:  # 4小时内
            errors.append(f"[注意] 第{i+1}平仓后{i+2}立即开仓（间隔<4小时）")

if len(errors) > 0:
    print(f"\n发现{len(errors)}个问题：")
    for error in errors:
        print(error)
else:
    print("\n[OK] 未发现连续开仓或连续平仓的异常情况")

# 统计开平仓配对
print("\n" + "="*120)
print("开平仓配对分析")
print("="*120)

opens = [i for i, x in enumerate(sequence) if '开多' in x['action']]
closes = [i for i, x in enumerate(sequence) if '平仓' in x['action']]

print(f"\n开仓次数: {len(opens)}")
print(f"平仓次数: {len(closes)}")

if len(opens) == len(closes):
    print("[OK] 开平仓次数匹配")

    # 分析每个交易
    print(f"\n{'交易#':<6} {'开仓时间':<20} {'平仓时间':<20} {'持仓小时':<10} {'盈亏%':<10}")
    print("-"*80)

    for i in range(len(opens)):
        open_idx = opens[i]
        close_idx = closes[i]

        open_time = pd.to_datetime(sequence[open_idx]['time'])
        close_time = pd.to_datetime(sequence[close_idx]['time'])
        hold_hours = (close_time - open_time).total_seconds() / 3600
        pnl = sequence[close_idx]['pnl']

        print(f"{i+1:<6} {str(open_time)[:18]:<20} {str(close_time)[:18]:<20} "
              f"{hold_hours:<10.0f} {pnl:<10.2f}")
else:
    print(f"[警告] 开平仓次数不匹配！开{len(opens)}次，平{len(closes)}次")
