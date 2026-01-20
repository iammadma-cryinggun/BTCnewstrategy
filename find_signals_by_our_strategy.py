import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("按你总结的完整规律分析2025年6-12月信号")
print("="*100)

# 1. 读取信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

signals_2025 = signals_df[
    (signals_df['时间'] >= '2025-06-01') &
    (signals_df['时间'] <= '2025-12-31')
].copy().reset_index(drop=True)

print(f"\n总信号数: {len(signals_2025)}条")

# 2. 找出所有首次信号（张力>0.5, 加速度<0）
first_signals = signals_2025[
    (signals_2025['张力'] > 0.5) &
    (signals_2025['加速度'] < 0)
].copy().reset_index(drop=True)

print(f"\n首次信号（张力>0.5, 加速度<0）: {len(first_signals)}个")

# 3. 对每个首次信号，检查后续1-2个周期的确认条件
opportunities = []

for idx, first_sig in first_signals.iterrows():
    signal_time = first_sig['时间']
    first_tension = first_sig['张力']
    first_accel = first_sig['加速度']
    first_vol = first_sig['量能比率']
    first_price = first_sig['收盘价']

    # 找到在原始数据中的位置
    original_idx = signals_2025[signals_2025['时间'] == signal_time].index[0]

    # 检查后续1-2个周期
    best_entry = None
    best_entry_period = None

    for wait_period in range(1, 3):  # 等待1-2个周期
        check_idx = original_idx + wait_period
        if check_idx >= len(signals_2025):
            break

        check_row = signals_2025.iloc[check_idx]
        check_tension = check_row['张力']
        check_accel = check_row['加速度']
        check_vol = check_row['量能比率']
        check_price = check_row['收盘价']

        # 计算张力变化
        tension_change = (check_tension - first_tension) / first_tension

        # 确认条件（按照你总结的）
        # 1. 张力保持合理范围（可以稍微下降）
        # 2. 加速度保持负值或负值加大
        # 3. 量能比率 < 1.0
        # 4. 张力变化趋势（上升或小幅下降）

        confirm_conditions = {
            '张力合理': check_tension > 0.45,  # 允许小幅下降
            '加速度负值': check_accel < 0,
            '量能缩量': check_vol < 1.0,
            '张力上升': tension_change >= -0.1,  # 允许小幅下降
        }

        all_pass = all(confirm_conditions.values())

        if all_pass:
            best_entry = {
                'period': wait_period,
                '时间': check_row['时间'],
                '张力': check_tension,
                '加速度': check_accel,
                '量能': check_vol,
                '价格': check_price,
                '张力变化%': tension_change * 100,
                '确认条件': confirm_conditions
            }
            best_entry_period = wait_period
            break  # 找到第一个满足条件的就停止

    # 如果找到确认的开仓点，跟踪后续表现
    if best_entry:
        entry_idx = original_idx + best_entry_period
        entry_tension = best_entry['张力']
        entry_price = best_entry['价格']

        # 检查后续最多10个周期
        max_look_ahead = min(10, len(signals_2025) - entry_idx - 1)
        exit_found = False
        best_pnl = -999
        best_exit_info = None

        for look_ahead in range(1, max_look_ahead + 1):
            future_idx = entry_idx + look_ahead
            future_row = signals_2025.iloc[future_idx]

            future_tension = future_row['张力']
            future_accel = future_row['加速度']
            future_vol = future_row['量能比率']
            future_price = future_row['收盘价']

            # 计算盈亏
            pnl = (entry_price - future_price) / entry_price * 100

            # 检查平仓条件
            tension_drop = (entry_tension - future_tension) / entry_tension

            # 平仓：张力下降40%+ 且 加速度转正或接近0 且 量能放大
            should_exit = (
                tension_drop > 0.4 and
                (future_accel > 0 or abs(future_accel) < 0.001) and
                future_vol > 1.0
            )

            if should_exit:
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_exit_info = {
                        '周期': look_ahead,
                        '时间': future_row['时间'],
                        '张力': future_tension,
                        '加速度': future_accel,
                        '量能': future_vol,
                        '价格': future_price,
                        '张力下降%': tension_drop * 100,
                        '盈亏%': pnl
                    }
                exit_found = True
                break

        # 记录机会
        opportunities.append({
            '首次信号时间': signal_time,
            '首次信号张力': first_tension,
            '首次信号加速度': first_accel,
            '首次信号量能': first_vol,
            '首次信号价格': first_price,
            '等待周期': best_entry_period,
            '最佳开仓时间': best_entry['时间'],
            '开仓张力': best_entry['张力'],
            '开仓加速度': best_entry['加速度'],
            '开仓量能': best_entry['量能'],
            '开仓价格': best_entry['价格'],
            '张力变化%': best_entry['张力变化%'],
            '平仓时间': best_exit_info['时间'] if best_exit_info else None,
            '平仓价格': best_exit_info['价格'] if best_exit_info else None,
            '盈亏%': best_exit_info['盈亏%'] if best_exit_info else None,
            '持仓周期': best_exit_info['周期'] if best_exit_info else None,
            '是好机会': '是' if best_exit_info and best_exit_info['盈亏%'] > 2 else '否' if best_exit_info else '未平仓'
        })

# 转换为DataFrame
opportunities_df = pd.DataFrame(opportunities)

# 统计
print(f"\n通过确认的交易机会: {len(opportunities_df)}个")

if len(opportunities_df) > 0:
    good_opp = opportunities_df[opportunities_df['是好机会'] == '是']
    bad_opp = opportunities_df[opportunities_df['是好机会'] == '否']

    print(f"  好机会（盈利>2%）: {len(good_opp)}个")
    print(f"  差机会（盈利≤2%）: {len(bad_opp)}个")

    closed_trades = opportunities_df[opportunities_df['平仓时间'].notna()]
    if len(closed_trades) > 0:
        print(f"\n整体表现:")
        print(f"  总盈亏: {closed_trades['盈亏%'].sum():.2f}%")
        print(f"  胜率: {(closed_trades['盈亏%'] > 0).sum() / len(closed_trades) * 100:.2f}%")
        print(f"  平均盈亏: {closed_trades['盈亏%'].mean():.2f}%")

# 保存到Excel
with pd.ExcelWriter('按完整规律分析_2025年6-12月.xlsx', engine='openpyxl') as writer:

    # 所有符合条件的机会
    opportunities_df.to_excel(writer, sheet_name='所有机会', index=False)

    # 只显示好机会
    if len(good_opp) > 0:
        good_opp.to_excel(writer, sheet_name='好机会_盈利>2%', index=False)

    # 只显示差机会
    if len(bad_opp) > 0:
        bad_opp.to_excel(writer, sheet_name='差机会_盈利≤2%', index=False)

    # 统计汇总
    summary = pd.DataFrame([
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['首次信号数', len(first_signals)],
        ['通过确认数', len(opportunities_df)],
        ['好机会数', len(good_opp)],
        ['差机会数', len(bad_opp)],
        ['', ''],
        ['好机会占比', f'{len(good_opp)/len(opportunities_df)*100:.1f}%' if len(opportunities_df) > 0 else '0%'],
        ['', ''],
        ['确认条件', ''],
        ['1. 张力合理', '> 0.45（允许小幅下降）'],
        ['2. 加速度负值', '< 0'],
        ['3. 量能缩量', '< 1.0'],
        ['4. 张力趋势', '变化 >= -10%'],
        ['', ''],
        ['平仓条件', ''],
        ['张力下降', '> 40%'],
        ['加速度', '转正 或 接近0'],
        ['量能', '> 1.0（放量）'],
    ])
    summary.to_excel(writer, sheet_name='策略说明', index=False, header=False)

print("\n" + "="*100)
print("分析完成！已保存到: 按完整规律分析_2025年6-12月.xlsx")
print("="*100)

print(f"\n关键发现:")
print(f"  在{len(first_signals)}个首次信号中:")
print(f"  - 通过确认的: {len(opportunities_df)}个")
print(f"  - 其中好机会: {len(good_opp)}个 ({len(good_opp)/len(opportunities_df)*100:.1f}%)")
print(f"  - 其中差机会: {len(bad_opp)}个 ({len(bad_opp)/len(opportunities_df)*100:.1f}%)")
