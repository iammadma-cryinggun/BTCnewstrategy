import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("自动标注2025年6-12月的好机会（SHORT + LONG）")
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
short_signals = []
long_signals = []

for idx in range(len(df_2025)):
    tension = df_2025.iloc[idx, 5]  # 张力
    accel = df_2025.iloc[idx, 6]    # 加速度

    if tension > 0.5 and accel < 0:
        short_signals.append(idx)
    elif tension < -0.5 and accel > 0:
        long_signals.append(idx)

print(f"\n首次SHORT信号: {len(short_signals)}条")
print(f"首次LONG信号: {len(long_signals)}条")

# 处理SHORT信号
print("\n" + "="*100)
print("处理SHORT信号...")
print("="*100)

short_annotated = []

for first_idx in short_signals:
    first_row = df_2025.iloc[first_idx]
    first_time = first_row.iloc[0]
    first_price = first_row.iloc[1]
    first_tension = first_row.iloc[5]
    first_accel = first_row.iloc[6]
    first_energy = first_row.iloc[3]

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
            # SHORT：价格越低越好
            current_advantage = (first_price - check_price) / first_price * 100

            if current_advantage > price_advantage:
                price_advantage = current_advantage
                best_entry_idx = check_idx
                best_entry_price = check_price
                best_entry_tension = check_tension
                wait_periods = wait_period

    if best_entry_idx is not None:
        entry_row = df_2025.iloc[best_entry_idx]
        entry_time = entry_row.iloc[0]

        tension_change = (best_entry_tension - first_tension) / first_tension * 100

        # 好机会标准
        is_good = (tension_change > 5) or (price_advantage > 0.5)

        short_annotated.append({
            '交易方向': 'SHORT',
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

# 处理LONG信号
print("\n处理LONG信号...")

long_annotated = []

for first_idx in long_signals:
    first_row = df_2025.iloc[first_idx]
    first_time = first_row.iloc[0]
    first_price = first_row.iloc[1]
    first_tension = first_row.iloc[5]
    first_accel = first_row.iloc[6]
    first_energy = first_row.iloc[3]

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

        # 确认条件：张力保持低位（绝对值），加速度仍然为正，量能<1.0
        if check_tension < -0.45 and check_accel > 0 and check_energy < 1.0:
            # LONG：价格越高越好
            current_advantage = (check_price - first_price) / first_price * 100

            if current_advantage > price_advantage:
                price_advantage = current_advantage
                best_entry_idx = check_idx
                best_entry_price = check_price
                best_entry_tension = check_tension
                wait_periods = wait_period

    if best_entry_idx is not None:
        entry_row = df_2025.iloc[best_entry_idx]
        entry_time = entry_row.iloc[0]

        # 对于LONG，张力是负数，变化计算要特别处理
        tension_change = abs((best_entry_tension - first_tension) / first_tension * 100)

        # 好机会标准
        is_good = (tension_change > 5) or (price_advantage > 0.5)

        long_annotated.append({
            '交易方向': 'LONG',
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

# 合并结果
all_annotated = short_annotated + long_annotated
results_df = pd.DataFrame(all_annotated)

print(f"\n找到最佳开仓点: {len(results_df)}笔")
print(f"  - SHORT: {len(short_annotated)}笔")
print(f"  - LONG: {len(long_annotated)}笔")

# 分别统计
short_df = results_df[results_df['交易方向'] == 'SHORT']
long_df = results_df[results_df['交易方向'] == 'LONG']

short_good = short_df[short_df['是好机会'] == '是']
long_good = long_df[long_df['是好机会'] == '是']

print(f"\nSHORT好机会: {len(short_good)}笔 ({len(short_good)/len(short_df)*100:.1f}%)")
print(f"LONG好机会: {len(long_good)}笔 ({len(long_good)/len(long_df)*100:.1f}%)")

# 保存到Excel
output_file = '2025年6-12月_完整标注结果_SHORT+LONG.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 所有标注
    results_df.to_excel(writer, sheet_name='所有标注', index=False)

    # SHORT
    short_df.to_excel(writer, sheet_name='SHORT所有', index=False)
    if len(short_good) > 0:
        short_good.to_excel(writer, sheet_name='SHORT好机会', index=False)

    # LONG
    long_df.to_excel(writer, sheet_name='LONG所有', index=False)
    if len(long_good) > 0:
        long_good.to_excel(writer, sheet_name='LONG好机会', index=False)

    # 统计汇总
    summary_data = [
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['', ''],
        ['SHORT统计', ''],
        ['首次信号数', len(short_signals)],
        ['找到最佳开仓点', len(short_df)],
        ['好机会数量', len(short_good)],
        ['好机会比例', f'{len(short_good)/len(short_df)*100:.1f}%'],
        ['好机会平均张力变化', f'{short_good["张力变化%"].mean():+.2f}%' if len(short_good) > 0 else 'N/A'],
        ['好机会平均价格优势', f'{short_good["价格优势%"].mean():+.3f}%' if len(short_good) > 0 else 'N/A'],
        ['好机会平均等待周期', f'{short_good["等待周期"].mean():.1f}' if len(short_good) > 0 else 'N/A'],
        ['', ''],
        ['LONG统计', ''],
        ['首次信号数', len(long_signals)],
        ['找到最佳开仓点', len(long_df)],
        ['好机会数量', len(long_good)],
        ['好机会比例', f'{len(long_good)/len(long_df)*100:.1f}%'],
        ['好机会平均张力变化', f'{long_good["张力变化%"].mean():+.2f}%' if len(long_good) > 0 else 'N/A'],
        ['好机会平均价格优势', f'{long_good["价格优势%"].mean():+.3f}%' if len(long_good) > 0 else 'N/A'],
        ['好机会平均等待周期', f'{long_good["等待周期"].mean():.1f}' if len(long_good) > 0 else 'N/A'],
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

print(f"\n已保存到: {output_file}")

# 总结规律
print("\n" + "="*100)
print("规律总结")
print("="*100)

print("\n【SHORT信号规律】")
if len(short_good) > 0:
    print(f"好机会数量: {len(short_good)}笔")
    print(f"平均张力变化: {short_good['张力变化%'].mean():+.2f}%")
    print(f"平均价格优势: {short_good['价格优势%'].mean():+.3f}%")
    print(f"平均等待周期: {short_good['等待周期'].mean():.1f}个")

    # 张力变化分布
    tension_rising = short_good[short_good['张力变化%'] > 5]
    tension_small = short_good[short_good['张力变化%'] <= 5]

    print(f"\n张力大幅上升(>5%): {len(tension_rising)}笔 ({len(tension_rising)/len(short_good)*100:.1f}%)")
    print(f"张力小幅上升(0-5%): {len(tension_small)}笔 ({len(tension_small)/len(short_good)*100:.1f}%)")

    # 价格优势分布
    price_big = short_good[short_good['价格优势%'] > 1.0]
    price_small = short_good[(short_good['价格优势%'] > 0.5) & (short_good['价格优势%'] <= 1.0)]
    price_none = short_good[short_good['价格优势%'] <= 0.5]

    print(f"\n大幅价格优势(>1%): {len(price_big)}笔")
    print(f"小幅价格优势(0.5-1%): {len(price_small)}笔")
    print(f"主要靠张力优势(≤0.5%): {len(price_none)}笔")

print("\n【LONG信号规律】")
if len(long_good) > 0:
    print(f"好机会数量: {len(long_good)}笔")
    print(f"平均张力变化: {long_good['张力变化%'].mean():+.2f}%")
    print(f"平均价格优势: {long_good['价格优势%'].mean():+.3f}%")
    print(f"平均等待周期: {long_good['等待周期'].mean():.1f}个")

    # 张力变化分布
    tension_rising = long_good[long_good['张力变化%'] > 5]
    tension_small = long_good[long_good['张力变化%'] <= 5]

    print(f"\n张力大幅上升(>5%): {len(tension_rising)}笔 ({len(tension_rising)/len(long_good)*100:.1f}%)")
    print(f"张力小幅上升(0-5%): {len(tension_small)}笔 ({len(tension_small)/len(long_good)*100:.1f}%)")

    # 价格优势分布
    price_big = long_good[long_good['价格优势%'] > 1.0]
    price_small = long_good[(long_good['价格优势%'] > 0.5) & (long_good['价格优势%'] <= 1.0)]
    price_none = long_good[long_good['价格优势%'] <= 0.5]

    print(f"\n大幅价格优势(>1%): {len(price_big)}笔")
    print(f"小幅价格优势(0.5-1%): {len(price_small)}笔")
    print(f"主要靠张力优势(≤0.5%): {len(price_none)}笔")

print("\n" + "="*100)
print("核心结论")
print("="*100)

print("\n【好机会的3个特征】")
print("1. 张力大幅上升(>5%) - 表示趋势在加强")
print("2. 价格优势明显(>0.5%) - 等待后获得更好的开仓价格")
print("3. 等待1-4个周期 - 不要等太久，错过机会")

if len(short_good) > 0 and len(long_good) > 0:
    print(f"\n【SHORT vs LONG对比】")
    print(f"SHORT好机会比例: {len(short_good)/len(short_df)*100:.1f}%")
    print(f"LONG好机会比例: {len(long_good)/len(long_df)*100:.1f}%")

    if len(short_good) > 0 and len(long_good) > 0:
        short_tension = short_good['张力变化%'].mean()
        long_tension = long_good['张力变化%'].mean()
        print(f"\nSHORT平均张力变化: {short_tension:+.2f}%")
        print(f"LONG平均张力变化: {long_tension:+.2f}%")

        if short_tension > long_tension:
            print(f"→ SHORT信号的张力变化更明显，更容易识别好机会")
        else:
            print(f"→ LONG信号的张力变化更明显，更容易识别好机会")
