import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("创建对比分析Excel：原始信号 vs 总结规律")
print("="*100)

# 1. 读取原始信号数据
signals_df = pd.read_csv('btc_4h_signals_matching_log.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

print(f"\n原始信号数据: {len(signals_df)}条")
print(f"时间范围: {signals_df['时间'].min()} 至 {signals_df['时间'].max()}")

# 2. 定义我们总结的5笔交易
summary_trades = [
    {
        'trade_id': 1,
        'direction': 'short',
        'first_signal': '2025-12-02 12:00:00',
        'best_entry': '2025-12-03 16:00:00',
        'best_exit': '2025-12-11 20:00:00',
        'entry_tension': 1.014203,
        'exit_tension': 0.612816
    },
    {
        'trade_id': 2,
        'direction': 'short',
        'first_signal': '2025-12-12 08:00:00',
        'best_entry': '2025-12-12 12:00:00',
        'best_exit': '2025-12-16 00:00:00',
        'entry_tension': 0.473982,
        'exit_tension': 0.131206
    },
    {
        'trade_id': 3,
        'direction': 'long',
        'first_signal': '2025-12-26 16:00:00',
        'best_entry': '2025-12-26 20:00:00',
        'best_exit': '2026-01-06 16:00:00',
        'entry_tension': -0.553677,
        'exit_tension': 0.229676
    },
    {
        'trade_id': 4,
        'direction': 'short',
        'first_signal': '2026-01-07 08:00:00',
        'best_entry': '2026-01-07 12:00:00',
        'best_exit': '2026-01-12 16:00:00',
        'entry_tension': 0.850444,
        'exit_tension': 0.451525
    },
    {
        'trade_id': 5,
        'direction': 'short',
        'first_signal': '2026-01-16 08:00:00',
        'best_entry': '2026-01-16 12:00:00',
        'best_exit': None,  # 未平仓
        'entry_tension': 0.658092,
        'exit_tension': None
    }
]

# 3. 创建对比分析DataFrame
analysis_data = []

for idx, row in signals_df.iterrows():
    signal_time = pd.to_datetime(row['时间'])

    # 检查这个时间点是否在我们总结的交易中
    trade_notes = []

    for trade in summary_trades:
        # 转换时间为datetime
        first_sig_time = pd.to_datetime(trade['first_signal'])
        best_entry_time = pd.to_datetime(trade['best_entry'])

        # 检查是否匹配
        if signal_time == first_sig_time:
            trade_notes.append(f"【总结-交易{trade['trade_id']}】首次信号")
            trade_notes.append(f"  我们认为的最佳开仓: {trade['best_entry']}")
            trade_notes.append(f"  我们认为的最佳平仓: {trade['best_exit']}")
            trade_notes.append(f"  方向: {trade['direction'].upper()}")

        elif signal_time == best_entry_time:
            trade_notes.append(f"【总结-交易{trade['trade_id']}】最佳开仓位置⭐")
            trade_notes.append(f"  首次信号: {trade['first_signal']}")
            trade_notes.append(f"  张力: {trade['entry_tension']:.6f} (我们总结的最佳张力)")
            trade_notes.append(f"  方向: {trade['direction'].upper()}")

    # 如果有标注，添加到分析数据
    if trade_notes:
        analysis_data.append({
            '时间': row['时间'],
            '收盘价': row['收盘价'],
            '张力': row['张力'],
            '加速度': row['加速度'],
            '量能比率': row['量能比率'],
            'EMA偏离%': row['EMA偏离%'],
            '信号类型': row['信号类型'],
            '置信度': row['置信度'],
            '信号描述': row['信号描述'],
            '交易方向': row['交易方向'],
            '是否开单': row['是否开单'],
            '标注': '\n'.join(trade_notes)
        })

# 创建对比分析的DataFrame
comparison_df = pd.DataFrame(analysis_data)

print(f"\n找到 {len(comparison_df)} 个与我们总结相关的信号点")

# 4. 读取V7.0.7系统的实际交易数据
v707_trades = pd.read_csv('backtest_results_2024_2025.csv')
v707_trades['入场时间'] = pd.to_datetime(v707_trades['入场时间'])
v707_trades_2025 = v707_trades[
    (v707_trades['入场时间'] >= '2025-12-01') &
    (v707_trades['入场时间'] <= '2026-01-19')
]

print(f"\nV7.0.7系统2025年12月-1月的实际交易: {len(v707_trades_2025)}笔")

# 5. 创建完整的对比Excel
with pd.ExcelWriter('信号对比分析_原始vs总结.xlsx', engine='openpyxl') as writer:

    # Sheet1: 原始信号数据（所有）
    signals_df.to_excel(writer, sheet_name='原始信号_全部', index=False)

    # Sheet2: 我们总结的关键点标注
    comparison_df.to_excel(writer, sheet_name='总结规律_标注点', index=False)

    # Sheet3: V7.0.7系统实际交易
    v707_trades_2025.to_excel(writer, sheet_name='V707系统_实际交易', index=False)

    # Sheet4: 详细对比分析
    detailed_comparison = []

    for trade in summary_trades:
        first_sig_time = pd.to_datetime(trade['first_signal'])
        best_entry_time = pd.to_datetime(trade['best_entry'])

        # 找到原始信号数据
        first_signal_data = signals_df[signals_df['时间'] == first_sig_time]
        best_entry_data = signals_df[signals_df['时间'] == best_entry_time]

        if len(first_signal_data) > 0:
            fs = first_signal_data.iloc[0]
            detailed_comparison.append({
                '交易ID': trade['trade_id'],
                '关键时间点': '首次信号',
                '时间': trade['first_signal'],
                '收盘价': fs['收盘价'],
                '张力': fs['张力'],
                '加速度': fs['加速度'],
                '量能比率': fs['量能比率'],
                '信号类型': fs['信号类型'],
                '置信度': fs['置信度'],
                '是否开单': fs['是否开单'],
                '我们总结的位置': '⭐ 是 - 最佳开仓应等1周期',
            })
        else:
            detailed_comparison.append({
                '交易ID': trade['trade_id'],
                '关键时间点': '首次信号',
                '时间': trade['first_signal'],
                '备注': '未在原始信号中找到'
            })

        if len(best_entry_data) > 0:
            be = best_entry_data.iloc[0]
            detailed_comparison.append({
                '交易ID': trade['trade_id'],
                '关键时间点': '最佳开仓⭐',
                '时间': trade['best_entry'],
                '收盘价': be['收盘价'],
                '张力': be['张力'],
                '加速度': be['加速度'],
                '量能比率': be['量能比率'],
                '信号类型': be['信号类型'],
                '置信度': be['置信度'],
                '是否开单': be['是否开单'],
                '我们总结的位置': '⭐⭐⭐ 这里是最佳开仓！',
            })

        # 平仓点（如果有的话）
        if trade['best_exit']:
            best_exit_time = pd.to_datetime(trade['best_exit'])
            best_exit_data = signals_df[signals_df['时间'] == best_exit_time]

            if len(best_exit_data) > 0:
                ex = best_exit_data.iloc[0]
                detailed_comparison.append({
                    '交易ID': trade['trade_id'],
                    '关键时间点': '最佳平仓⭐',
                    '时间': trade['best_exit'],
                    '收盘价': ex['收盘价'],
                    '张力': ex['张力'],
                    '加速度': ex['加速度'],
                    '量能比率': ex['量能比率'],
                    '信号类型': ex['信号类型'],
                    '置信度': ex['置信度'],
                    '是否开单': ex['是否开单'],
                    '我们总结的位置': '⭐⭐⭐ 这里是最佳平仓！',
                })

    pd.DataFrame(detailed_comparison).to_excel(writer, sheet_name='详细对比分析', index=False)

    # Sheet5: 统计总结
    summary_stats = pd.DataFrame([
        ['数据周期', '2025-12-01 至 2026-01-19'],
        ['总信号数', len(signals_df)],
        ['我们总结的交易数', len(summary_trades)],
        ['V7.0.7系统实际交易数', len(v707_trades_2025)],
        ['V7.0.7系统总盈亏%', f"{v707_trades_2025['盈亏%'].sum():.2f}"],
        ['V7.0.7系统胜率%', f"{(v707_trades_2025['盈亏%']>0).sum()/len(v707_trades_2025)*100:.1f}%"],
        ['', ''],
        ['关键发现', ''],
        ['1. 我们只总结了5笔特殊情况', ''],
        ['2. V7.0.7系统实际交易更多', ''],
        ['3. 我们总结的规律不能代表系统整体逻辑', ''],
        ['4. 需要用V7.0.5过滤器和ZigZag出场重新理解', ''],
    ])
    summary_stats.to_excel(writer, sheet_name='统计总结', index=False, header=False)

print("\n✅ Excel文件已生成: 信号对比分析_原始vs总结.xlsx")
print("\n包含以下Sheet:")
print("  1. 原始信号_全部 - 所有原始信号数据")
print("  2. 总结规律_标注点 - 我们总结的关键点标注")
print("  3. V707系统_实际交易 - V7.0.7系统的实际交易记录")
print("  4. 详细对比分析 - 逐笔对比分析")
print("  5. 统计总结 - 关键发现和统计数据")

# 6. 输出一些关键对比
print("\n" + "="*100)
print("关键对比分析")
print("="*100)

print("\n我们总结的交易 vs 原始信号:")
for trade in summary_trades:
    print(f"\n【交易 {trade['trade_id']}】{trade['direction'].upper()}")

    # 首次信号
    first_signal_data = signals_df[signals_df['时间'] == pd.to_datetime(trade['first_signal'])]
    if len(first_signal_data) > 0:
        fs = first_signal_data.iloc[0]
        print(f"  首次信号 ({trade['first_signal']}):")
        print(f"    张力: {fs['张力']:.6f}")
        print(f"    加速度: {fs['加速度']:.6f}")
        print(f"    信号类型: {fs['信号类型']}")
        print(f"    是否开单: {fs['是否开单']}")

    # 最佳开仓
    best_entry_data = signals_df[signals_df['时间'] == pd.to_datetime(trade['best_entry'])]
    if len(best_entry_data) > 0:
        be = best_entry_data.iloc[0]
        print(f"  最佳开仓 ({trade['best_entry']}) ⭐:")
        print(f"    张力: {be['张力']:.6f}")
        print(f"    加速度: {be['加速度']:.6f}")
        print(f"    信号类型: {be['信号类型']}")
        print(f"    是否开单: {be['是否开单']}")

    # 对比
    if len(first_signal_data) > 0 and len(best_entry_data) > 0:
        fs = first_signal_data.iloc[0]
        be = best_entry_data.iloc[0]
        tension_change = be['张力'] - fs['张力']
        accel_change = be['加速度'] - fs['加速度']
        periods = (pd.to_datetime(trade['best_entry']) - pd.to_datetime(trade['first_signal'])).total_seconds() / 3600 / 4

        print(f"  差异分析:")
        print(f"    等待周期: {periods:.0f}个4小时")
        print(f"    张力变化: {tension_change:+.6f}")
        print(f"    加速度变化: {accel_change:+.6f}")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
