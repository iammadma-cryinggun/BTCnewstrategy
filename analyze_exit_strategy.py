import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("平仓策略分析：好机会的最优平仓位置")
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

# 读取好机会标注
good_opportunities = pd.read_excel('2025年6-12月_完整标注结果_SHORT+LONG.xlsx', sheet_name='所有标注')
good_opportunities = good_opportunities[good_opportunities['是好机会'] == '是'].copy()

print(f"好机会数量: {len(good_opportunities)}笔")

# 分析每个好机会的开仓后走势
exit_analysis = []

for idx, row in good_opportunities.iterrows():
    direction = row['交易方向']
    entry_time = pd.to_datetime(row['最佳开仓时间'])
    entry_price = row['开仓价格']
    entry_tension = row['开仓张力']

    # 找到在数据中的位置
    entry_idx = df_2025[df_2025['时间'] == entry_time].index

    if len(entry_idx) == 0:
        continue

    entry_idx = entry_idx[0]

    # 跟踪后续10个周期
    max_pnl = -999 if direction == 'SHORT' else -999
    min_pnl = 999 if direction == 'SHORT' else 999
    best_exit_idx = None
    worst_exit_idx = None

    for look_ahead in range(1, min(11, len(df_2025) - entry_idx)):
        future_idx = entry_idx + look_ahead
        future_row = df_2025.iloc[future_idx]

        future_price = future_row.iloc[1]  # 收盘价
        future_tension = future_row.iloc[5]  # 张力
        future_accel = future_row.iloc[6]  # 加速度
        future_energy = future_row.iloc[3]  # 量能

        if direction == 'SHORT':
            pnl = (entry_price - future_price) / entry_price * 100
        else:  # LONG
            pnl = (future_price - entry_price) / entry_price * 100

        # 记录最大盈亏
        if pnl > max_pnl:
            max_pnl = pnl
            best_exit_idx = future_idx
            best_exit_period = look_ahead
            best_exit_tension = future_tension
            best_exit_accel = future_accel
            best_exit_energy = future_energy

        if pnl < min_pnl:
            min_pnl = pnl
            worst_exit_idx = future_idx

    # 张力变化（开仓到最优平仓）
    if best_exit_idx is not None:
        tension_change = (best_exit_tension - entry_tension) / entry_tension * 100

        exit_analysis.append({
            '交易方向': direction,
            '开仓时间': entry_time,
            '开仓价格': entry_price,
            '开仓张力': entry_tension,

            '最优平仓周期': best_exit_period,
            '最优平仓盈亏': max_pnl,
            '最优平仓张力': best_exit_tension,
            '最优平仓加速度': best_exit_accel,
            '最优平仓量能': future_energy,

            '张力变化%': tension_change,

            '最差平仓盈亏': min_pnl,
            '盈亏波动': max_pnl - min_pnl
        })

exit_df = pd.DataFrame(exit_analysis)

print(f"\n成功跟踪到后续走势: {len(exit_df)}笔")

# 分别分析SHORT和LONG
for direction in ['SHORT', 'LONG']:
    print("\n" + "="*100)
    print(f"【{direction}信号】平仓策略分析")
    print("="*100)

    data = exit_df[exit_df['交易方向'] == direction].copy()

    if len(data) == 0:
        continue

    print(f"\n样本数: {len(data)}笔")

    # 1. 最优平仓周期分布
    print("\n" + "-"*100)
    print("1. 最优平仓周期分布")
    print("-"*100)

    for period in range(1, 11):
        count = (data['最优平仓周期'] == period).sum()
        if count > 0:
            avg_pnl = data[data['最优平仓周期'] == period]['最优平仓盈亏'].mean()
            print(f"  第{period}周期: {count}笔 ({count/len(data)*100:.1f}%), 平均盈亏{avg_pnl:+.2f}%")

    print(f"\n平均最优平仓周期: {data['最优平仓周期'].mean():.1f}")
    print(f"中位数最优平仓周期: {data['最优平仓周期'].median():.1f}")

    # 2. 最优平仓时的特征
    print("\n" + "-"*100)
    print("2. 最优平仓时的市场特征")
    print("-"*100)

    print(f"\n最优平仓时的张力:")
    print(f"  平均: {data['最优平仓张力'].mean():.4f}")
    print(f"  中位数: {data['最优平仓张力'].median():.4f}")

    if direction == 'SHORT':
        print(f"\n最优平仓时的加速度（SHORT应为负或接近0）:")
        accel_data = data['最优平仓加速度']
    else:
        print(f"\n最优平仓时的加速度（LONG应为正或接近0）:")
        accel_data = data['最优平仓加速度']

    print(f"  平均: {accel_data.mean():.6f}")
    print(f"  中位数: {accel_data.median():.6f}")

    # 加速度转正的比例
    if direction == 'SHORT':
        accel_positive = (accel_data > 0).sum()
    else:
        accel_positive = (accel_data < 0).sum()

    print(f"  加速度转{('正' if direction == 'SHORT' else '负')}: {accel_positive}/{len(accel_data)} = {accel_positive/len(accel_data)*100:.1f}%")

    print(f"\n最优平仓时的量能:")
    print(f"  平均: {data['最优平仓量能'].mean():.2f}")
    print(f"  中位数: {data['最优平仓量能'].median():.2f}")

    # 3. 张力变化特征
    print("\n" + "-"*100)
    print("3. 从开仓到最优平仓的张力变化")
    print("-"*100)

    if direction == 'SHORT':
        tension_change = data['张力变化%']
    else:
        tension_change = abs(data['张力变化%'])

    print(f"  平均: {tension_change.mean():+.2f}%")
    print(f"  中位数: {tension_change.median():+.2f}%")

    # 张力下降/上升的比例
    if direction == 'SHORT':
        tension_drop = (tension_change < 0).sum()
        print(f"  张力下降: {tension_drop}/{len(tension_change)} = {tension_drop/len(tension_change)*100:.1f}%")
    else:
        tension_drop = (tension_change < 0).sum()
        print(f"  张力绝对值下降: {tension_drop}/{len(tension_change)} = {tension_drop/len(tension_change)*100:.1f}%")

    # 4. 最优平仓盈亏分布
    print("\n" + "-"*100)
    print("4. 最优平仓盈亏分布")
    print("-"*100)

    print(f"  平均: {data['最优平仓盈亏'].mean():+.2f}%")
    print(f"  中位数: {data['最优平仓盈亏'].median():+.2f}%")
    print(f"  最大: {data['最优平仓盈亏'].max():+.2f}%")
    print(f"  最小: {data['最优平仓盈亏'].min():+.2f}%")

    # 盈亏分布
    print(f"\n盈亏区间分布:")
    bins = [(-999, -2), (-2, 0), (0, 2), (2, 5), (5, 999)]
    for min_p, max_p in bins:
        if max_p == 999:
            subset = data[data['最优平仓盈亏'] >= min_p]
            label = f"≥{min_p}%"
        elif min_p == -999:
            subset = data[data['最优平仓盈亏'] < max_p]
            label = f"<{max_p}%"
        else:
            subset = data[(data['最优平仓盈亏'] >= min_p) & (data['最优平仓盈亏'] < max_p)]
            label = f"{min_p}%~{max_p}%"

        if len(subset) > 0:
            print(f"  {label}: {len(subset)}笔 ({len(subset)/len(data)*100:.1f}%)")

    # 5. 按平仓周期分组的盈亏
    print("\n" + "-"*100)
    print("5. 按平仓周期分组的盈亏表现")
    print("-"*100)

    for period in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        subset = data[data['最优平仓周期'] == period]
        if len(subset) > 0:
            print(f"\n  第{period}周期平仓 ({len(subset)}笔):")
            print(f"    平均盈亏: {subset['最优平仓盈亏'].mean():+.2f}%")
            print(f"    盈利概率: {(subset['最优平仓盈亏'] > 0).sum()/len(subset)*100:.1f}%")
            print(f"    大赚(>3%): {(subset['最优平仓盈亏'] > 3).sum()}笔 ({(subset['最优平仓盈亏'] > 3).sum()/len(subset)*100:.1f}%)")

    # 6. 寻找最优平仓信号
    print("\n" + "-"*100)
    print("6. 最优平仓信号特征总结")
    print("-"*100)

    # 张力下降40%的比例
    if direction == 'SHORT':
        big_drop = (data['张力变化%'] < -40).sum()
        print(f"\n张力下降≥40%: {big_drop}/{len(data)} = {big_drop/len(data)*100:.1f}%")
    else:
        big_drop = (abs(data['张力变化%']) < -40).sum()
        print(f"\n张力绝对值下降≥40%: {big_drop}/{len(data)} = {big_drop/len(data)*100:.1f}%")

    # 量能放大的比例
    energy_expand = (data['最优平仓量能'] > 1.0).sum()
    print(f"量能放大(>1.0): {energy_expand}/{len(data)} = {energy_expand/len(data)*100:.1f}%")

    # 加速度转正的比例
    if direction == 'SHORT':
        accel_turn = (data['最优平仓加速度'] > 0).sum()
        print(f"加速度转正: {accel_turn}/{len(data)} = {accel_turn/len(data)*100:.1f}%")
    else:
        accel_turn = (data['最优平仓加速度'] < 0).sum()
        print(f"加速度转负: {accel_turn}/{len(data)} = {accel_turn/len(data)*100:.1f}%")

    # 7. 综合平仓策略
    print("\n" + "-"*100)
    print("7. 综合平仓策略建议")
    print("-"*100)

    # 找最常见的平仓周期
    mode_period = data['最优平仓周期'].mode()[0]
    print(f"\n最常见最优平仓周期: 第{mode_period}周期")

    # 找盈亏最好的周期
    best_period_data = data.groupby('最优平仓周期')['最优平仓盈亏'].mean().sort_values(ascending=False)
    best_period = best_period_data.index[0]
    best_pnl = best_period_data.iloc[0]
    print(f"盈亏最好的周期: 第{best_period}周期 (平均{best_pnl:+.2f}%)")

# 保存结果
print("\n" + "="*100)
print("保存平仓分析结果...")
print("="*100)

output_file = '平仓策略分析_好机会最优平仓.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    exit_df.to_excel(writer, sheet_name='所有平仓分析', index=False)

    short_exit = exit_df[exit_df['交易方向'] == 'SHORT']
    long_exit = exit_df[exit_df['交易方向'] == 'LONG']

    short_exit.to_excel(writer, sheet_name='SHORT平仓', index=False)
    long_exit.to_excel(writer, sheet_name='LONG平仓', index=False)

print(f"\n已保存到: {output_file}")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
