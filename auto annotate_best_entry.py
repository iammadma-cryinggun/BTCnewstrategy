import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("自动标注2025年6-12月的好机会")
print("="*100)

# 读取信号数据
df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 筛选6-12月
df_2025 = df[
    (df['时间'] >= '2025-06-01') &
    (df['时间'] <= '2025-12-31')
].copy().reset_index(drop=True)

print(f"\n总信号数: {len(df_2025)}条")

# 找所有首次信号
first_signals_indices = []

for idx in range(len(df_2025)):
    tension = df_2025.iloc[idx, 5]  # 张力
    accel = df_2025.iloc[idx, 6]    # 加速度

    if tension > 0.5 and accel < 0:
        first_signals_indices.append(idx)

print(f"首次SHORT信号: {len(first_signals_indices)}条")

# 对每个首次信号，找最佳开仓点
annotated_trades = []

for first_idx in first_signals_indices:
    first_row = df_2025.iloc[first_idx]
    first_time = first_row.iloc[0]
    first_price = first_row.iloc[1]
    first_tension = first_row.iloc[5]
    first_accel = first_row.iloc[6]
    first_energy = first_row.iloc[3]

    # 在后续1-7个周期内找最佳开仓点
    best_entry_idx = None
    best_entry_price = None
    best_entry_tension = None
    wait_periods = 0
    price_advantage = 0

    # 检查后续1-7个周期
    for wait_period in range(1, 8):
        check_idx = first_idx + wait_period
        if check_idx >= len(df_2025):
            break

        check_row = df_2025.iloc[check_idx]
        check_price = check_row.iloc[1]
        check_tension = check_row.iloc[5]
        check_accel = check_row.iloc[6]
        check_energy = check_row.iloc[3]

        # 确认条件：张力保持高位，加速度仍然为负，量能<1.0
        if check_tension > 0.45 and check_accel < 0 and check_energy < 1.0:
            # 计算价格优势（SHORT：价格越低越好）
            current_advantage = (first_price - check_price) / first_price * 100

            # 找价格优势最大的点
            if current_advantage > price_advantage:
                price_advantage = current_advantage
                best_entry_idx = check_idx
                best_entry_price = check_price
                best_entry_tension = check_tension
                wait_periods = wait_period

    # 如果找到最佳开仓点
    if best_entry_idx is not None:
        entry_row = df_2025.iloc[best_entry_idx]
        entry_time = entry_row.iloc[0]

        # 张力变化
        tension_change = (best_entry_tension - first_tension) / first_tension * 100

        # 判断是否是好机会
        # 标准1：张力上升>5%
        # 标准2：价格优势>0.5%
        is_good = (tension_change > 5) or (price_advantage > 0.5)

        annotated_trades.append({
            '首次信号时间': first_time,
            '首次价格': first_price,
            '首次张力': first_tension,
            '首次加速度': first_accel,
            '首次能量': first_energy,

            '最佳开仓时间': entry_time,
            '等待周期': wait_periods,
            '开仓价格': best_entry_price,
            '开仓张力': best_entry_tension,

            '张力变化%': tension_change,
            '价格优势%': price_advantage,

            '是好机会': '是' if is_good else '否',
            '好机会原因': ('张力上升' if tension_change > 5 else '') + (' 价格优势' if price_advantage > 0.5 else '')
        })

# 转换为DataFrame
results_df = pd.DataFrame(annotated_trades)

print(f"\n找到最佳开仓点: {len(results_df)}笔")

# 统计好机会
good_trades = results_df[results_df['是好机会'] == '是']
bad_trades = results_df[results_df['是好机会'] == '否']

print(f"\n好机会: {len(good_trades)}笔 ({len(good_trades)/len(results_df)*100:.1f}%)")
print(f"差机会: {len(bad_trades)}笔 ({len(bad_trades)/len(results_df)*100:.1f}%)")

# 保存到Excel
output_file = '2025年6-12月_自动标注结果.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 所有找到最佳开仓点的交易
    results_df.to_excel(writer, sheet_name='所有标注', index=False)

    # 好机会
    if len(good_trades) > 0:
        good_trades.to_excel(writer, sheet_name='好机会', index=False)

    # 差机会
    if len(bad_trades) > 0:
        bad_trades.to_excel(writer, sheet_name='差机会', index=False)

    # 统计汇总
    summary = pd.DataFrame([
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['首次信号数', len(first_signals_indices)],
        ['找到最佳开仓点', len(results_df)],
        ['', ''],
        ['好机会数量', len(good_trades)],
        ['好机会比例', f'{len(good_trades)/len(results_df)*100:.1f}%'],
        ['', ''],
        ['好机会平均张力变化', f'{good_trades["张力变化%"].mean():+.2f}%' if len(good_trades) > 0 else 'N/A'],
        ['好机会平均价格优势', f'{good_trades["价格优势%"].mean():+.3f}%' if len(good_trades) > 0 else 'N/A'],
        ['', ''],
        ['差机会平均张力变化', f'{bad_trades["张力变化%"].mean():+.2f}%' if len(bad_trades) > 0 else 'N/A'],
        ['差机会平均价格优势', f'{bad_trades["价格优势%"].mean():+.3f}%' if len(bad_trades) > 0 else 'N/A'],
    ])
    summary.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

print(f"\n已保存到: {output_file}")

# 显示好机会示例
print("\n" + "="*100)
print("好机会示例（前10笔）")
print("="*100)

for idx, row in good_trades.head(10).iterrows():
    print(f"\n{row['首次信号时间']}")
    print(f"  首次: 价格{row['首次价格']:.2f}, 张力{row['首次张力']:.4f}")
    print(f"  等待{row['等待周期']:.0f}周期后: {row['最佳开仓时间']}")
    print(f"  开仓: 价格{row['开仓价格']:.2f}, 张力{row['开仓张力']:.4f}")
    print(f"  张力变化: {row['张力变化%']:+.2f}%, 价格优势: {row['价格优势%']:+.3f}%")
    print(f"  原因: {row['好机会原因']}")
