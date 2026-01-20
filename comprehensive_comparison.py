import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("完整对比分析：首次信号直接开仓 vs 等待确认后开仓")
print("2025年6-12月，所有282个首次信号")
print("="*100)

# 1. 读取信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

signals_2025 = signals_df[
    (signals_df['时间'] >= '2025-06-01') &
    (signals_df['时间'] <= '2025-12-31')
].copy().reset_index(drop=True)

print(f"\n总信号数: {len(signals_2025)}条")

# 2. 找出所有首次信号
first_signals = signals_2025[
    (signals_2025['张力'] > 0.5) &
    (signals_2025['加速度'] < 0)
].copy().reset_index(drop=True)

print(f"首次信号（张力>0.5, 加速度<0）: {len(first_signals)}个")

# 3. 对每个首次信号，对比两种策略
comparison_results = []

for idx, first_sig in first_signals.iterrows():
    signal_time = first_sig['时间']
    first_tension = first_sig['张力']
    first_accel = first_sig['加速度']
    first_vol = first_sig['量能比率']
    first_price = first_sig['收盘价']

    # 找到在原始数据中的位置
    original_idx = signals_2025[signals_2025['时间'] == signal_time].index[0]

    # ===== 策略A：首次信号直接开仓 =====
    strategy_a_pnl = None
    strategy_a_periods = None
    strategy_a_exit_reason = None

    # 检查后续10个周期
    for look_ahead in range(1, min(11, len(signals_2025) - original_idx)):
        future_idx = original_idx + look_ahead
        future_row = signals_2025.iloc[future_idx]

        future_price = future_row['收盘价']
        future_tension = future_row['张力']
        future_accel = future_row['加速度']

        # 计算盈亏
        pnl = (first_price - future_price) / first_price * 100

        # 平仓条件：张力下降40%或加速度转正
        tension_drop = (first_tension - future_tension) / first_tension if first_tension > 0 else 0

        if tension_drop > 0.4 or future_accel > 0:
            strategy_a_pnl = pnl
            strategy_a_periods = look_ahead
            strategy_a_exit_reason = "张力下降40%" if tension_drop > 0.4 else "加速度转正"
            break

    # 如果10个周期都没平仓，记录第10个周期的盈亏
    if strategy_a_pnl is None:
        if original_idx + 10 < len(signals_2025):
            future_price = signals_2025.iloc[original_idx + 10]['收盘价']
            strategy_a_pnl = (first_price - future_price) / first_price * 100
            strategy_a_periods = 10
            strategy_a_exit_reason = "超时"

    # ===== 策略B：等待确认后开仓 =====
    strategy_b_entry_found = False
    strategy_b_entry_price = None
    strategy_b_entry_tension = None
    strategy_b_wait_periods = None
    strategy_b_pnl = None
    strategy_b_periods = None
    strategy_b_exit_reason = None

    # 检查后续1-2个周期是否有确认
    for wait_period in range(1, 3):
        check_idx = original_idx + wait_period
        if check_idx >= len(signals_2025):
            break

        check_row = signals_2025.iloc[check_idx]
        check_tension = check_row['张力']
        check_accel = check_row['加速度']
        check_vol = check_row['量能比率']
        check_price = check_row['收盘价']

        # 确认条件
        confirm = (
            check_tension > 0.45 and
            check_accel < 0 and
            check_vol < 1.0 and
            (check_tension - first_tension) / first_tension >= -0.1
        )

        if confirm:
            strategy_b_entry_found = True
            strategy_b_entry_price = check_price
            strategy_b_entry_tension = check_tension
            strategy_b_wait_periods = wait_period

            # 从开仓点跟踪后续10个周期
            for look_ahead in range(1, min(11, len(signals_2025) - check_idx)):
                future_idx = check_idx + look_ahead
                future_row = signals_2025.iloc[future_idx]

                future_price = future_row['收盘价']
                future_tension = future_row['张力']
                future_accel = future_row['加速度']
                future_vol = future_row['量能比率']

                # 计算盈亏
                pnl = (strategy_b_entry_price - future_price) / strategy_b_entry_price * 100

                # 平仓条件
                tension_drop = (strategy_b_entry_tension - future_tension) / strategy_b_entry_tension

                if tension_drop > 0.4 and (future_accel > 0 or abs(future_accel) < 0.001) and future_vol > 1.0:
                    strategy_b_pnl = pnl
                    strategy_b_periods = look_ahead
                    strategy_b_exit_reason = "正常平仓"
                    break

            # 如果没触发平仓，记录第10个周期
            if strategy_b_pnl is None and check_idx + 10 < len(signals_2025):
                future_price = signals_2025.iloc[check_idx + 10]['收盘价']
                strategy_b_pnl = (strategy_b_entry_price - future_price) / strategy_b_entry_price * 100
                strategy_b_periods = 10
                strategy_b_exit_reason = "超时"

            break  # 找到确认就停止

    # 价格优势
    if strategy_b_entry_found:
        price_advantage = (first_price - strategy_b_entry_price) / first_price * 100
    else:
        price_advantage = None

    comparison_results.append({
        '首次信号时间': signal_time,
        '首次信号张力': first_tension,
        '首次信号加速度': first_accel,
        '首次信号量能': first_vol,
        '首次信号价格': first_price,

        '策略A_直接开仓_盈亏%': strategy_a_pnl,
        '策略A_持仓周期': strategy_a_periods,
        '策略A_平仓原因': strategy_a_exit_reason,

        '策略B_确认开仓_找到确认': strategy_b_entry_found,
        '策略B_等待周期': strategy_b_wait_periods,
        '策略B_开仓价格': strategy_b_entry_price,
        '策略B_开仓张力': strategy_b_entry_tension,
        '策略B_盈亏%': strategy_b_pnl,
        '策略B_总持仓周期': strategy_b_periods,
        '策略B_平仓原因': strategy_b_exit_reason,

        '价格优势%': price_advantage,
        '盈亏差异%': (strategy_b_pnl - strategy_a_pnl) if (strategy_a_pnl is not None and strategy_b_pnl is not None) else None,
        '哪个更好': 'B更好' if (strategy_a_pnl is not None and strategy_b_pnl is not None and strategy_b_pnl > strategy_a_pnl) else 'A更好' if (strategy_a_pnl is not None and strategy_b_pnl is not None and strategy_b_pnl < strategy_a_pnl) else '持平'
    })

# 转换为DataFrame
comparison_df = pd.DataFrame(comparison_results)

# 统计分析
print("\n" + "="*100)
print("策略对比统计")
print("="*100)

# 策略A：直接开仓
strategy_a_valid = comparison_df[comparison_df['策略A_直接开仓_盈亏%'].notna()]
print(f"\n策略A - 首次信号直接开仓:")
print(f"  有效交易: {len(strategy_a_valid)}笔")
print(f"  总盈亏: {strategy_a_valid['策略A_直接开仓_盈亏%'].sum():.2f}%")
print(f"  胜率: {(strategy_a_valid['策略A_直接开仓_盈亏%'] > 0).sum() / len(strategy_a_valid) * 100:.2f}%")
print(f"  平均盈亏: {strategy_a_valid['策略A_直接开仓_盈亏%'].mean():.2f}%")

# 策略B：确认后开仓
strategy_b_valid = comparison_df[
    comparison_df['策略B_确认开仓_找到确认'] == True
][comparison_df['策略B_盈亏%'].notna()]

print(f"\n策略B - 等待确认后开仓:")
print(f"  通过确认: {(comparison_df['策略B_确认开仓_找到确认'] == True).sum()}个")
print(f"  有盈亏数据: {len(strategy_b_valid)}笔")
print(f"  总盈亏: {strategy_b_valid['策略B_盈亏%'].sum():.2f}%")
print(f"  胜率: {(strategy_b_valid['策略B_盈亏%'] > 0).sum() / len(strategy_b_valid) * 100:.2f}%")
print(f"  平均盈亏: {strategy_b_valid['策略B_盈亏%'].mean():.2f}%")

# 直接对比
both_valid = comparison_df[
    (comparison_df['策略A_直接开仓_盈亏%'].notna()) &
    (comparison_df['策略B_盈亏%'].notna())
]

print(f"\n直接对比（两种策略都有盈亏的{len(both_valid)}笔）:")
better_b = (both_valid['策略B_盈亏%'] > both_valid['策略A_直接开仓_盈亏%']).sum()
better_a = (both_valid['策略B_盈亏%'] < both_valid['策略A_直接开仓_盈亏%']).sum()
equal = (both_valid['策略B_盈亏%'] == both_valid['策略A_直接开仓_盈亏%']).sum()

print(f"  策略B更好: {better_b}笔 ({better_b/len(both_valid)*100:.1f}%)")
print(f"  策略A更好: {better_a}笔 ({better_a/len(both_valid)*100:.1f}%)")
print(f"  持平: {equal}笔 ({equal/len(both_valid)*100:.1f}%)")

avg_diff = (both_valid['策略B_盈亏%'] - both_valid['策略A_直接开仓_盈亏%']).mean()
print(f"  平均盈亏差异: {avg_diff:+.2f}%")

# 保存到Excel
with pd.ExcelWriter('完整对比_直接开仓vs确认开仓.xlsx', engine='openpyxl') as writer:

    # 所有对比结果
    comparison_df.to_excel(writer, sheet_name='所有信号对比', index=False)

    # 只显示两种策略都有效的
    both_valid.to_excel(writer, sheet_name='两种策略都有效', index=False)

    # 策略A更好的
    a_better = both_valid[both_valid['策略B_盈亏%'] < both_valid['策略A_直接开仓_盈亏%']]
    if len(a_better) > 0:
        a_better.to_excel(writer, sheet_name='策略A更好', index=False)

    # 策略B更好的
    b_better = both_valid[both_valid['策略B_盈亏%'] > both_valid['策略A_直接开仓_盈亏%']]
    if len(b_better) > 0:
        b_better.to_excel(writer, sheet_name='策略B更好', index=False)

    # 统计汇总
    summary = pd.DataFrame([
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['首次信号总数', len(first_signals)],
        ['', ''],
        ['策略A：首次信号直接开仓', ''],
        ['有效交易数', len(strategy_a_valid)],
        ['总盈亏%', f"{strategy_a_valid['策略A_直接开仓_盈亏%'].sum():.2f}"],
        ['胜率%', f"{(strategy_a_valid['策略A_直接开仓_盈亏%'] > 0).sum() / len(strategy_a_valid) * 100:.2f}"],
        ['平均盈亏%', f"{strategy_a_valid['策略A_直接开仓_盈亏%'].mean():.2f}"],
        ['', ''],
        ['策略B：等待确认后开仓', ''],
        ['通过确认数', (comparison_df['策略B_确认开仓_找到确认'] == True).sum()],
        ['有效交易数', len(strategy_b_valid)],
        ['总盈亏%', f"{strategy_b_valid['策略B_盈亏%'].sum():.2f}"],
        ['胜率%', f"{(strategy_b_valid['策略B_盈亏%'] > 0).sum() / len(strategy_b_valid) * 100:.2f}"],
        ['平均盈亏%', f"{strategy_b_valid['策略B_盈亏%'].mean():.2f}"],
        ['', ''],
        ['直接对比', ''],
        ['两种策略都有效', len(both_valid)],
        ['策略B更好', f"{better_b}笔 ({better_b/len(both_valid)*100:.1f}%)"],
        ['策略A更好', f"{better_a}笔 ({better_a/len(both_valid)*100:.1f}%)"],
        ['平均盈亏差异', f"{avg_diff:+.2f}%"],
        ['', ''],
        ['结论', '策略B' if avg_diff > 0 else '策略A' if avg_diff < 0 else '持平'],
    ])
    summary.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

print("\n" + "="*100)
print("详细对比已保存到: 完整对比_直接开仓vs确认开仓.xlsx")
print("="*100)

# 显示一些具体案例
print("\n" + "="*100)
print("具体案例展示（前10个两种策略都有效的）")
print("="*100)

sample_cases = both_valid.head(10)
for idx, row in sample_cases.iterrows():
    print(f"\n案例 {idx+1}: {row['首次信号时间']}")
    print(f"  首次信号: 张力={row['首次信号张力']:.4f}, 加速度={row['首次信号加速度']:.4f}")
    print(f"  策略A（直接开仓）: 盈亏{row['策略A_直接开仓_盈亏%']:+.2f}% ({row['策略A_持仓周期']:.0f}周期)")
    print(f"  策略B（确认开仓）: 盈亏{row['策略B_盈亏%']:+.2f}% ({row['策略B_总持仓周期']:.0f}周期, 等待{row['策略B_等待周期']:.0f}周期)")
    print(f"  差异: {row['盈亏差异%']:+.2f}% → {row['哪个更好']}")
