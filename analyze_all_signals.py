import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("分析所有符合我们条件的信号")
print("检查哪些是真正的好入场点")
print("="*100)

# 1. 读取信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

signals_2025 = signals_df[
    (signals_df['时间'] >= '2025-06-01') &
    (signals_df['时间'] <= '2025-12-31')
].copy().reset_index(drop=True)

print(f"\n总信号数: {len(signals_2025)}条")

# 2. 找出所有符合SHORT条件的信号（张力>0.5, 加速度<0）
short_signals = signals_2025[
    (signals_2025['张力'] > 0.5) &
    (signals_2025['加速度'] < 0)
].copy().reset_index(drop=True)

print(f"\n符合SHORT条件的信号: {len(short_signals)}条")

# 3. 对每个SHORT信号，分析后续表现
short_analysis = []

for sig_idx, sig_row in short_signals.iterrows():
    signal_time = sig_row['时间']
    signal_price = sig_row['收盘价']
    signal_tension = sig_row['张力']
    signal_accel = sig_row['加速度']
    signal_vol_ratio = sig_row['量能比率']

    # 找到这个信号在原始数据中的索引
    original_idx = signals_2025[signals_2025['时间'] == signal_time].index[0]

    # 检查后1个周期的条件（确认）
    if original_idx + 1 < len(signals_2025):
        next_row = signals_2025.iloc[original_idx + 1]
        next_tension = next_row['张力']
        next_accel = next_row['加速度']
        next_vol_ratio = next_row['量能比率']
        next_price = next_row['收盘价']

        # 判断是否满足确认条件
        confirm_condition = (
            next_tension > 0.5 and
            next_accel < 0 and
            next_vol_ratio < 1.0
        )

        # 如果确认，检查后续10个周期的盈亏
        if confirm_condition:
            entry_price = next_price
            entry_tension = next_tension

            # 检查后续最多10个周期（40小时）
            max_periods = min(10, len(signals_2025) - original_idx - 1)
            best_pnl = -999
            best_exit_idx = -1
            exit_reason = ""

            for look_ahead in range(1, max_periods + 1):
                check_idx = original_idx + 1 + look_ahead
                if check_idx >= len(signals_2025):
                    break

                future_row = signals_2025.iloc[check_idx]
                future_price = future_row['收盘价']
                future_tension = future_row['张力']
                future_accel = future_row['加速度']
                future_vol_ratio = future_row['量能比率']

                # 计算当前盈亏
                pnl = (entry_price - future_price) / entry_price * 100

                # 检查平仓条件
                tension_change = (entry_tension - future_tension) / entry_tension

                # 张力下降40%或加速度转正
                if tension_change > 0.4 or future_accel > 0:
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_exit_idx = check_idx
                        exit_reason = "张力下降40%" if tension_change > 0.4 else "加速度转正"
                    break  # 平仓

                # 记录最佳盈亏点
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_exit_idx = check_idx
                    exit_reason = f"第{look_ahead}周期"

            short_analysis.append({
                '首次信号时间': signal_time,
                '首次信号张力': signal_tension,
                '首次信号加速度': signal_accel,
                '首次信号量能': signal_vol_ratio,
                '首次信号价格': signal_price,
                '确认通过': '是',
                '开仓时间': next_row['时间'],
                '开仓价格': entry_price,
                '开仓张力': entry_tension,
                '开仓加速度': next_accel,
                '开仓量能': next_vol_ratio,
                '平仓时间': signals_2025.iloc[best_exit_idx]['时间'] if best_exit_idx > 0 else None,
                '平仓价格': signals_2025.iloc[best_exit_idx]['收盘价'] if best_exit_idx > 0 else None,
                '盈亏%': best_pnl if best_pnl != -999 else 0,
                '平仓原因': exit_reason,
                '持仓周期': best_exit_idx - original_idx - 1 if best_exit_idx > 0 else 0,
                '是好机会': '是' if best_pnl > 2 else '否'
            })
        else:
            # 不满足确认条件
            short_analysis.append({
                '首次信号时间': signal_time,
                '首次信号张力': signal_tension,
                '首次信号加速度': signal_accel,
                '首次信号量能': signal_vol_ratio,
                '首次信号价格': signal_price,
                '确认通过': '否',
                '开仓时间': None,
                '开仓价格': None,
                '开仓张力': None,
                '开仓加速度': None,
                '开仓量能': None,
                '平仓时间': None,
                '平仓价格': None,
                '盈亏%': None,
                '平仓原因': None,
                '持仓周期': None,
                '是好机会': None
            })

# 转换为DataFrame
short_analysis_df = pd.DataFrame(short_analysis)

# 统计
print(f"\nSHORT信号分析:")
print(f"  总信号数: {len(short_signals)}")
print(f"  通过确认: {len(short_analysis_df[short_analysis_df['确认通过'] == '是'])}")
print(f"  实际开仓: {len(short_analysis_df[short_analysis_df['确认通过'] == '是'])}")

confirmed_trades = short_analysis_df[short_analysis_df['确认通过'] == '是']
if len(confirmed_trades) > 0:
    good_trades = confirmed_trades[confirmed_trades['是好机会'] == '是']
    bad_trades = confirmed_trades[confirmed_trades['是好机会'] == '否']

    print(f"\n  好机会（盈利>2%）: {len(good_trades)}笔")
    print(f"  差机会（盈利≤2%）: {len(bad_trades)}笔")

    if len(confirmed_trades) > 0:
        print(f"\n  总盈亏: {confirmed_trades['盈亏%'].sum():.2f}%")
        print(f"  胜率: {(confirmed_trades['盈亏%'] > 0).sum() / len(confirmed_trades) * 100:.2f}%")
        print(f"  平均盈亏: {confirmed_trades['盈亏%'].mean():.2f}%")

# 保存到Excel
with pd.ExcelWriter('所有SHORT信号分析.xlsx', engine='openpyxl') as writer:
    # 所有SHORT信号
    short_analysis_df.to_excel(writer, sheet_name='所有SHORT信号分析', index=False)

    # 只显示好机会
    if len(good_trades) > 0:
        good_trades.to_excel(writer, sheet_name='好机会_盈利>2%', index=False)

    # 只显示差机会
    if len(bad_trades) > 0:
        bad_trades.to_excel(writer, sheet_name='差机会_盈利≤2%', index=False)

    # 统计汇总
    summary = pd.DataFrame([
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['SHORT信号总数', len(short_signals)],
        ['通过确认的信号', len(confirmed_trades)],
        ['好机会（盈利>2%）', len(good_trades)],
        ['差机会（盈利≤2%）', len(bad_trades)],
        ['', ''],
        ['好机会占比', f'{len(good_trades)/len(confirmed_trades)*100:.1f}%' if len(confirmed_trades) > 0 else '0%'],
        ['总盈亏', f'{confirmed_trades["盈亏%"].sum():.2f}%' if len(confirmed_trades) > 0 else '0%'],
        ['胜率', f'{(confirmed_trades["盈亏%"]>0).sum()/len(confirmed_trades)*100:.1f}%' if len(confirmed_trades) > 0 else '0%'],
        ['', ''],
        ['关键发现', ''],
        ['结论1', '不是所有张力>0.5的信号都是好机会'],
        ['结论2', f'只有{len(good_trades)}/{len(confirmed_trades)}={len(good_trades)/len(confirmed_trades)*100:.1f}%的信号是好机会'],
    ])
    summary.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

print("\n" + "="*100)
print("分析完成！已保存到: 所有SHORT信号分析.xlsx")
print("="*100)

print("\n关键发现:")
print(f"  在{len(confirmed_trades)}笔通过确认的交易中:")
print(f"  - 只有{len(good_trades)}笔（{len(good_trades)/len(confirmed_trades)*100:.1f}%）是好机会（盈利>2%）")
print(f"  - 有{len(bad_trades)}笔（{len(bad_trades)/len(confirmed_trades)*100:.1f}%）是差机会")
print(f"\n  结论: 需要更精确的过滤条件！")
