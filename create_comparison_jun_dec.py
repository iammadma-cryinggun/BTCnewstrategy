import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("创建对比分析Excel：2025年6-12月")
print("  - 原始信号 vs V7.0.7系统实际交易")
print("="*100)

# 1. 读取2025年6-12月的信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
print(f"\n原始信号数据: {len(signals_df)}条")
print(f"时间范围: {signals_df['时间'].min()} 至 {signals_df['时间'].max()}")

# 2. 读取V7.0.7系统回测结果，筛选2025年6-12月
all_trades = pd.read_csv('backtest_results_2024_2025.csv')
all_trades['入场时间'] = pd.to_datetime(all_trades['入场时间'])

v707_trades_2025 = all_trades[
    (all_trades['入场时间'] >= '2025-06-01') &
    (all_trades['入场时间'] <= '2025-12-31')
]

print(f"\nV7.0.7系统2025年6-12月实际交易: {len(v707_trades_2025)}笔")
print(f"时间范围: {v707_trades_2025['入场时间'].min()} 至 {v707_trades_2025['入场时间'].max()}")

# 3. 分析信号类型分布
print("\n" + "="*100)
print("信号类型分布")
print("="*100)

# 信号类型统计
signal_counts = signals_df['信号类型'].value_counts()
print("\n原始信号类型分布:")
for sig_type, count in signal_counts.items():
    print(f"  {sig_type}: {count}条")

# V7.0.7实际交易方向统计
direction_counts = v707_trades_2025['方向'].value_counts()
print("\nV7.0.7实际交易方向分布:")
for direction, count in direction_counts.items():
    print(f"  {direction}: {count}笔")

# 4. 创建对比Excel
with pd.ExcelWriter('信号对比分析_2025年6-12月.xlsx', engine='openpyxl') as writer:

    # Sheet1: 所有原始信号
    signals_df.to_excel(writer, sheet_name='原始信号_全部', index=False)

    # Sheet2: V7.0.7系统实际交易
    v707_trades_2025.to_excel(writer, sheet_name='V707系统_实际交易', index=False)

    # Sheet3: 信号类型详细分析
    signal_analysis = []

    for sig_type in signals_df['信号类型'].unique():
        sig_data = signals_df[signals_df['信号类型'] == sig_type]

        signal_analysis.append({
            '信号类型': sig_type,
            '信号数量': len(sig_data),
            '平均张力': sig_data['张力'].mean(),
            '张力最小值': sig_data['张力'].min(),
            '张力最大值': sig_data['张力'].max(),
            '平均加速度': sig_data['加速度'].mean(),
            '平均置信度': sig_data['置信度'].mean(),
        })

    pd.DataFrame(signal_analysis).to_excel(writer, sheet_name='信号类型分析', index=False)

    # Sheet4: V7.0.7交易统计
    trade_stats = []

    for direction in ['LONG', 'SHORT']:
        dir_trades = v707_trades_2025[v707_trades_2025['方向'] == direction]

        if len(dir_trades) > 0:
            winning_trades = dir_trades[dir_trades['盈亏%'] > 0]
            losing_trades = dir_trades[dir_trades['盈亏%'] < 0]

            # 计算平均持仓时间（小时）
            dir_trades_copy = dir_trades.copy()
            dir_trades_copy['入场时间'] = pd.to_datetime(dir_trades_copy['入场时间'])
            dir_trades_copy['出场时间'] = pd.to_datetime(dir_trades_copy['出场时间'])
            avg_hold_hours = (dir_trades_copy['出场时间'] - dir_trades_copy['入场时间']).dt.total_seconds().mean() / 3600

            trade_stats.append({
                '方向': direction,
                '交易数': len(dir_trades),
                '总盈亏%': dir_trades['盈亏%'].sum(),
                '胜率%': (len(winning_trades) / len(dir_trades) * 100) if len(dir_trades) > 0 else 0,
                '平均盈亏%': dir_trades['盈亏%'].mean(),
                '最大盈利%': dir_trades['盈亏%'].max(),
                '最大亏损%': dir_trades['盈亏%'].min(),
                '平均持仓小时': avg_hold_hours,
                '盈利交易数': len(winning_trades),
                '亏损交易数': len(losing_trades)
            })

    pd.DataFrame(trade_stats).to_excel(writer, sheet_name='V707交易统计', index=False)

    # Sheet5: 信号与交易匹配
    # 将V7.0.7的实际交易与信号匹配
    signal_trade_matches = []

    for idx, trade in v707_trades_2025.iterrows():
        entry_time = pd.to_datetime(trade['入场时间'])

        # 查找对应的信号（时间相近的）
        # 将交易时间转换为4H周期对齐
        trade_time_4h = entry_time

        # 查找同一时间的信号
        matching_signal = signals_df[
            pd.to_datetime(signals_df['时间']) == trade_time_4h
        ]

        if len(matching_signal) > 0:
            sig = matching_signal.iloc[0]

            # 计算持仓小时数
            exit_time = pd.to_datetime(trade['出场时间'])
            hold_hours = (exit_time - entry_time).total_seconds() / 3600

            signal_trade_matches.append({
                '交易时间': trade['入场时间'],
                '信号时间': sig['时间'],
                '信号类型': sig['信号类型'],
                '张力': sig['张力'],
                '加速度': sig['加速度'],
                '置信度': sig['置信度'],
                '交易方向': trade['方向'],
                '入场价': trade['入场价'],
                '出场价': trade['出场价'],
                '盈亏%': trade['盈亏%'],
                '持仓小时': hold_hours,
                '出场原因': trade['出场原因']
            })

    pd.DataFrame(signal_trade_matches).to_excel(
        writer, sheet_name='信号交易匹配', index=False
    )

    # Sheet6: 总结统计
    summary_stats = pd.DataFrame([
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['', ''],
        ['原始信号数据', ''],
        ['总信号数', len(signals_df)],
        ['信号类型数', signals_df['信号类型'].nunique()],
        ['', ''],
        ['V7.0.7系统交易', ''],
        ['总交易数', len(v707_trades_2025)],
        ['LONG交易数', len(v707_trades_2025[v707_trades_2025['方向'] == 'LONG'])],
        ['SHORT交易数', len(v707_trades_2025[v707_trades_2025['方向'] == 'SHORT'])],
        ['总盈亏%', f"{v707_trades_2025['盈亏%'].sum():.2f}"],
        ['胜率%', f"{(v707_trades_2025['盈亏%']>0).sum()/len(v707_trades_2025)*100:.2f}" if len(v707_trades_2025) > 0 else "0"],
        ['', ''],
        ['关键发现', ''],
        ['1. 信号 vs 实际交易', f'{len(signals_df)}个信号 → {len(v707_trades_2025)}笔交易'],
        ['2. 信号匹配', f'匹配到{len(signal_trade_matches)}笔交易与信号的对应关系'],
    ])
    summary_stats.to_excel(writer, sheet_name='统计总结', index=False, header=False)

print("\n" + "="*100)
print("Excel文件已生成: 信号对比分析_2025年6-12月.xlsx")
print("="*100)

print("\n包含以下Sheet:")
print("  1. 原始信号_全部 - 所有原始信号数据")
print("  2. V707系统_实际交易 - V7.0.7系统的实际交易记录")
print("  3. 信号类型分析 - 各信号类型的统计信息")
print("  4. V707交易统计 - LONG vs SHORT的详细统计")
print("  5. 信号交易匹配 - 信号与交易的对应关系")
print("  6. 统计总结 - 关键数据和发现")

# 输出关键对比
print("\n" + "="*100)
print("关键对比分析")
print("="*100)

print(f"\n【原始信号】")
print(f"  总数: {len(signals_df)}")

# 信号类型统计
print(f"\n【信号类型分布】")
for sig_type, count in signals_df['信号类型'].value_counts().items():
    sig_data = signals_df[signals_df['信号类型'] == sig_type]
    print(f"  {sig_type}: {count}条 (平均张力={sig_data['张力'].mean():.2f})")

print(f"\n【V7.0.7系统交易】")
print(f"  总交易: {len(v707_trades_2025)}笔")
print(f"  总盈亏: {v707_trades_2025['盈亏%'].sum():.2f}%")
print(f"  胜率: {(v707_trades_2025['盈亏%']>0).sum()/len(v707_trades_2025)*100:.2f}%")

long_trades = v707_trades_2025[v707_trades_2025['方向'] == 'LONG']
short_trades = v707_trades_2025[v707_trades_2025['方向'] == 'SHORT']

print(f"\n  LONG: {len(long_trades)}笔, 盈亏{long_trades['盈亏%'].sum():.2f}%, 胜率{(long_trades['盈亏%']>0).sum()/len(long_trades)*100:.2f}%")
print(f"  SHORT: {len(short_trades)}笔, 盈亏{short_trades['盈亏%'].sum():.2f}%, 胜率{(short_trades['盈亏%']>0).sum()/len(short_trades)*100:.2f}%")

print(f"\n【信号交易匹配】")
print(f"  匹配成功: {len(signal_trade_matches)}笔")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
