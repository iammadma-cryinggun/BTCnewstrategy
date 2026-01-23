# -*- coding: utf-8 -*-
"""
为简化标注添加黄金信号
====================
"""

import pandas as pd

print("="*120)
print("ADD GOLD SIGNALS TO SIMPLIFIED ANNOTATION")
print("="*120)

# Load simplified annotation
df = pd.read_csv('最终数据_简化标注.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# Step 1: 标注黄金信号
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Mark Gold Signals")
print("="*120)

gold_signals = []
for i in range(len(df)):
    action = df.loc[i, '最优动作']

    # 任何涉及开仓、平仓、刷新的动作都是ACTION
    if any(keyword in action for keyword in ['开多', '开空', '平多', '平空']):
        gold_signals.append('ACTION')
    else:
        gold_signals.append('HOLD')

df['黄金信号'] = gold_signals

action_count = sum(1 for s in gold_signals if s == 'ACTION')
hold_count = sum(1 for s in gold_signals if s == 'HOLD')

print(f"\n标注统计:")
print(f"  ACTION (需要交易): {action_count} ({action_count/len(df)*100:.1f}%)")
print(f"  HOLD (继续持有): {hold_count} ({hold_count/len(df)*100:.1f}%)")

# ============================================================================
# Step 2: 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Save Results")
print("="*120)

output_cols = [
    '时间', '收盘价', '信号类型', '信号模式', '高低点', '持仓状态', '最优动作', '黄金信号'
]

df[output_cols].to_csv('最终数据_完整标注_黄金信号.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: 最终数据_完整标注_黄金信号.csv")

# ============================================================================
# Step 3: 验证用户标注
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Verify User Annotation")
print("="*120)

test_data = [
    ('2025-08-19 20:00', '开多'),
    ('2025-08-20 16:00', '平多'),
    ('2025-08-21 16:00', '开多'),
    ('2025-08-22 04:00', '平多'),
    ('2025-08-22 08:00', '开多'),
    ('2025-08-22 20:00', '平多'),
    ('2025-08-23 12:00', '开多'),
    ('2025-08-23 20:00', '平多'),
    ('2025-08-24 16:00', '开多'),
    ('2025-08-25 12:00', '平多'),
    ('2025-08-26 00:00', '开多'),
    ('2025-08-26 20:00', '平多'),
    ('2025-08-27 00:00', '开空'),
    ('2025-08-27 04:00', '平空'),
    ('2025-08-27 08:00', '开空'),
    ('2025-08-27 20:00', '持空'),
    ('2025-08-28 00:00', '平空'),
    ('2025-08-29 16:00', '空仓'),
    ('2025-08-30 12:00', '开空'),
    ('2025-08-31 08:00', '平空'),
]

print("\n验证用户标注:")
match_count = 0
total_count = 0

for time_str, expected_keyword in test_data:
    rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == time_str]
    if len(rows) > 0:
        idx = rows.index[0]
        actual_action = df.loc[idx, '最优动作']
        signal_type = df.loc[idx, '信号类型']
        gold_signal = df.loc[idx, '黄金信号']

        # 宽松匹配：只看关键动词
        is_match = False
        if '开多' in expected_keyword and '开多' in actual_action:
            is_match = True
        elif '开空' in expected_keyword and '开空' in actual_action:
            is_match = True
        elif '平多' in expected_keyword and '平多' in actual_action:
            is_match = True
        elif '平空' in expected_keyword and '平空' in actual_action:
            is_match = True
        elif '持多' in expected_keyword and '持多' in actual_action:
            is_match = True
        elif '持空' in expected_keyword and '持空' in actual_action:
            is_match = True
        elif '空仓' in expected_keyword and '空仓' in actual_action:
            is_match = True

        total_count += 1
        if is_match:
            match_count += 1
            status = 'OK'
        else:
            status = 'X'

        print(f"{status} {time_str} {signal_type[:15]:<15} 期望:{expected_keyword:<8} 实际:{actual_action:<12} 黄金:{gold_signal}")

print(f"\n匹配率: {match_count}/{total_count} ({match_count/total_count*100:.1f}%)")

# ============================================================================
# Step 4: 显示ACTION信号的分布
# ============================================================================
print("\n" + "="*120)
print("STEP 4: ACTION Signal Distribution")
print("="*120)

action_df = df[df['黄金信号'] == 'ACTION']
print(f"\nACTION信号按信号类型分布:")
action_by_type = action_df['信号类型'].value_counts()
for signal_type, count in action_by_type.items():
    pct = count / len(action_df) * 100
    print(f"  {signal_type}: {count} ({pct:.1f}%)")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
