# -*- coding: utf-8 -*-
"""
合并标注结果与原始数据
====================
"""

import pandas as pd

print("="*120)
print("MERGE ANNOTATION WITH ORIGINAL DATA")
print("="*120)

# ============================================================================
# Step 1: 加载数据
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Load Data")
print("="*120)

# 原始数据（包含所有参数）
original_df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
original_df['时间'] = pd.to_datetime(original_df['时间'])
original_df = original_df.sort_values('时间').reset_index(drop=True)

print(f"\n原始数据: {len(original_df)} 条")
print(f"列数: {len(original_df.columns)}")
print(f"列名: {list(original_df.columns)}")

# 标注数据（包含最优动作和黄金信号）
annotation_df = pd.read_csv('最终数据_完整标注_黄金信号.csv', encoding='utf-8-sig')
annotation_df['时间'] = pd.to_datetime(annotation_df['时间'])
annotation_df = annotation_df.sort_values('时间').reset_index(drop=True)

print(f"\n标注数据: {len(annotation_df)} 条")
print(f"列名: {list(annotation_df.columns)}")

# ============================================================================
# Step 2: 合并数据
# ============================================================================
print("\n" + "="*120)
print("STEP 2: Merge Data")
print("="*120)

# 从标注数据中提取关键列
annotation_cols = ['时间', '信号模式', '高低点', '持仓状态', '最优动作', '黄金信号']

# 合并到原始数据
merged_df = original_df.copy()

for col in annotation_cols:
    if col != '时间':
        merged_df[col] = annotation_df[col].values

print(f"\n合并后数据: {len(merged_df)} 条")
print(f"列数: {len(merged_df.columns)}")

# ============================================================================
# Step 3: 保存结果
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Save Results")
print("="*120)

output_file = '最终数据_完整合并.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n已保存至: {output_file}")

# ============================================================================
# Step 4: 显示列信息
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Column Information")
print("="*120)

print(f"\n所有列名 ({len(merged_df.columns)} 列):")
for i, col in enumerate(merged_df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# Step 5: 显示前10条样本
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Sample Data (First 10)")
print("="*120)

display_cols = [
    '时间', '收盘价', '信号类型', '信号模式', '量能比率', '价格vsEMA%',
    '张力', '加速度', '高低点', '持仓状态', '最优动作', '黄金信号'
]

print(f"\n{'时间':<18} {'收盘价':<10} {'信号类型':<20} {'信号模式':<12} {'量能比率':<10} {'张力':<10} {'加速度':<10} {'黄金信号':<8}")
print("-" * 130)

for i in range(min(10, len(merged_df))):
    row = merged_df.iloc[i]
    time_str = str(row['时间'])[:16]
    close = f"{row['收盘价']:.2f}"
    signal_type = str(row['信号类型'])[:18]
    signal_mode = str(row['信号模式'])[:10]
    volume_ratio = f"{row['量能比率']:.2f}" if pd.notna(row['量能比率']) else 'N/A'
    tension = f"{row['张力']:.2f}" if pd.notna(row['张力']) else 'N/A'
    acceleration = f"{row['加速度']:.2f}" if pd.notna(row['加速度']) else 'N/A'
    gold_signal = str(row['黄金信号'])

    print(f"{time_str:<18} {close:<10} {signal_type:<20} {signal_mode:<12} {volume_ratio:<10} {tension:<10} {acceleration:<10} {gold_signal:<8}")

# ============================================================================
# Step 6: 统计ACTION vs HOLD的各参数均值
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Statistical Summary by Gold Signal")
print("="*120)

action_df = merged_df[merged_df['黄金信号'] == 'ACTION']
hold_df = merged_df[merged_df['黄金信号'] == 'HOLD']

print(f"\n参数统计 (ACTION vs HOLD):")
print(f"{'参数':<20} {'ACTION均值':<15} {'HOLD均值':<15} {'差异':<15}")
print("-" * 80)

params = ['量能比率', '价格vsEMA%', '张力', '加速度', 'DXY燃料']
for param in params:
    if param in merged_df.columns:
        action_mean = action_df[param].mean()
        hold_mean = hold_df[param].mean()
        diff = action_mean - hold_mean

        action_str = f"{action_mean:.4f}" if pd.notna(action_mean) else "N/A"
        hold_str = f"{hold_mean:.4f}" if pd.notna(hold_mean) else "N/A"
        diff_str = f"{diff:.4f}" if pd.notna(diff) else "N/A"

        print(f"{param:<20} {action_str:<15} {hold_str:<15} {diff_str:<15}")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
print(f"\n最终文件: {output_file}")
print(f"包含 {len(merged_df)} 条信号，{len(merged_df.columns)} 列完整数据")
