import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.precision', 6)

print("=" * 120)
print("BTC 4小时信号 - 最佳开仓/平仓位置分析")
print("=" * 120)

# 读取信号日志
df = pd.read_csv('btc_4h_signals_matching_log.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n总数据: {len(df)}条")
print(f"时间范围: {df['时间'].min()} 至 {df['时间'].max()}")

# 定义5笔交易的关键时间点
trades = [
    {
        'trade_id': 1,
        'first_signal': '2025-12-02 12:00:00',
        'best_entry': '2025-12-03 16:00:00',
        'best_exit': '2025-12-11 20:00:00',
        'direction': 'short'
    },
    {
        'trade_id': 2,
        'first_signal': '2025-12-12 08:00:00',
        'best_entry': '2025-12-12 12:00:00',
        'best_exit': '2025-12-16 00:00:00',
        'direction': 'short'
    },
    {
        'trade_id': 3,
        'first_signal': '2025-12-26 16:00:00',
        'best_entry': '2025-12-26 20:00:00',
        'best_exit': '2026-01-06 16:00:00',
        'direction': 'long'
    },
    {
        'trade_id': 4,
        'first_signal': '2026-01-07 08:00:00',
        'best_entry': '2026-01-07 12:00:00',
        'best_exit': '2026-01-12 16:00:00',
        'direction': 'short'
    },
    {
        'trade_id': 5,
        'first_signal': '2026-01-16 08:00:00',
        'best_entry': '2026-01-16 12:00:00',
        'best_exit': None,  # 未平仓
        'direction': 'short'
    }
]

# 提取关键时间点的数据
results = []

for trade in trades:
    print(f"\n{'='*120}")
    print(f"交易 #{trade['trade_id']} - {trade['direction'].upper()}")
    print(f"{'='*120}")

    # 首次信号点
    first_signal_data = df[df['时间'] == pd.to_datetime(trade['first_signal'])]
    if len(first_signal_data) == 0:
        print(f"警告: 未找到首次信号时间 {trade['first_signal']}")
        continue

    # 最佳开仓点
    best_entry_data = df[df['时间'] == pd.to_datetime(trade['best_entry'])]
    if len(best_entry_data) == 0:
        print(f"警告: 未找到最佳开仓时间 {trade['best_entry']}")
        continue

    # 最佳平仓点
    best_exit_data = None
    if trade['best_exit']:
        best_exit_data = df[df['时间'] == pd.to_datetime(trade['best_exit'])]
        if len(best_exit_data) == 0:
            print(f"警告: 未找到最佳平仓时间 {trade['best_exit']}")

    # 打印数据
    print(f"\n【首次信号】{trade['first_signal']}")
    fs = first_signal_data.iloc[0]
    print(f"  收盘价: {fs['收盘价']:.2f}")
    print(f"  张力: {fs['张力']:.6f}")
    print(f"  加速度: {fs['加速度']:.6f}")
    print(f"  量能比率: {fs['量能比率']:.4f}")
    print(f"  EMA偏离%: {fs['EMA偏离%']:.4f}")
    print(f"  信号类型: {fs['信号类型']}")

    print(f"\n【最佳开仓】{trade['best_entry']}")
    be = best_entry_data.iloc[0]
    print(f"  收盘价: {be['收盘价']:.2f}")
    print(f"  张力: {be['张力']:.6f}")
    print(f"  加速度: {be['加速度']:.6f}")
    print(f"  量能比率: {be['量能比率']:.4f}")
    print(f"  EMA偏离%: {be['EMA偏离%']:.4f}")
    print(f"  信号类型: {be['信号类型']}")

    if best_exit_data is not None and len(best_exit_data) > 0:
        print(f"\n【最佳平仓】{trade['best_exit']}")
        ex = best_exit_data.iloc[0]
        print(f"  收盘价: {ex['收盘价']:.2f}")
        print(f"  张力: {ex['张力']:.6f}")
        print(f"  加速度: {ex['加速度']:.6f}")
        print(f"  量能比率: {ex['量能比率']:.4f}")
        print(f"  EMA偏离%: {ex['EMA偏离%']:.4f}")
        print(f"  信号类型: {ex['信号类型']}")

    # 计算变化
    print(f"\n【变化分析】首次信号 -> 最佳开仓")
    tension_change_entry = be['张力'] - fs['张力']
    accel_change_entry = be['加速度'] - fs['加速度']
    price_change_entry = be['收盘价'] - fs['收盘价']
    periods_to_entry = (pd.to_datetime(trade['best_entry']) - pd.to_datetime(trade['first_signal'])).total_seconds() / 3600 / 4

    print(f"  时间间隔: {periods_to_entry:.0f}个4小时周期")
    print(f"  张力变化: {fs['张力']:.6f} -> {be['张力']:.6f} (变化量: {tension_change_entry:+.6f}, 变化率: {tension_change_entry/abs(fs['张力'])*100 if fs['张力']!=0 else 0:+.1f}%)")
    print(f"  加速度变化: {fs['加速度']:.6f} -> {be['加速度']:.6f} (变化量: {accel_change_entry:+.6f})")
    print(f"  价格变化: {fs['收盘价']:.2f} -> {be['收盘价']:.2f} (变化: {price_change_entry:+.2f})")

    if best_exit_data is not None and len(best_exit_data) > 0:
        print(f"\n【变化分析】最佳开仓 -> 最佳平仓")
        tension_change_exit = ex['张力'] - be['张力']
        accel_change_exit = ex['加速度'] - be['加速度']
        price_change_exit = ex['收盘价'] - be['收盘价']
        periods_to_exit = (pd.to_datetime(trade['best_exit']) - pd.to_datetime(trade['best_entry'])).total_seconds() / 3600 / 4

        print(f"  时间间隔: {periods_to_exit:.0f}个4小时周期")
        print(f"  张力变化: {be['张力']:.6f} -> {ex['张力']:.6f} (变化量: {tension_change_exit:+.6f}, 变化率: {tension_change_exit/abs(be['张力'])*100 if be['张力']!=0 else 0:+.1f}%)")
        print(f"  加速度变化: {be['加速度']:.6f} -> {ex['加速度']:.6f} (变化量: {accel_change_exit:+.6f})")
        print(f"  价格变化: {be['收盘价']:.2f} -> {ex['收盘价']:.2f} (变化: {price_change_exit:+.2f})")

    # 保存结果
    result = {
        '交易ID': trade['trade_id'],
        '方向': trade['direction'],
        '首次信号时间': trade['first_signal'],
        '最佳开仓时间': trade['best_entry'],
        '最佳平仓时间': trade['best_exit'] or '未平仓',
        '首次信号_张力': fs['张力'],
        '首次信号_加速度': fs['加速度'],
        '首次信号_量能': fs['量能比率'],
        '最佳开仓_张力': be['张力'],
        '最佳开仓_加速度': be['加速度'],
        '最佳开仓_量能': be['量能比率'],
        '张力变化_信号到开仓': tension_change_entry,
        '加速度变化_信号到开仓': accel_change_entry,
        '周期数_信号到开仓': periods_to_entry,
    }

    if best_exit_data is not None and len(best_exit_data) > 0:
        result.update({
            '最佳平仓_张力': ex['张力'],
            '最佳平仓_加速度': ex['加速度'],
            '最佳平仓_量能': ex['量能比率'],
            '张力变化_开仓到平仓': tension_change_exit,
            '加速度变化_开仓到平仓': accel_change_exit,
            '周期数_开仓到平仓': periods_to_exit,
        })

    results.append(result)

# 创建分析DataFrame
analysis_df = pd.DataFrame(results)

print(f"\n{'='*120}")
print("综合分析表")
print(f"{'='*120}")
print("\n1. 首次信号特征:")
print(analysis_df[['交易ID', '方向', '首次信号时间', '首次信号_张力', '首次信号_加速度', '首次信号_量能']].to_string(index=False))

print("\n2. 最佳开仓特征:")
print(analysis_df[['交易ID', '方向', '最佳开仓时间', '最佳开仓_张力', '最佳开仓_加速度', '最佳开仓_量能']].to_string(index=False))

print("\n3. 从信号到开仓的变化:")
entry_change_cols = ['交易ID', '方向', '张力变化_信号到开仓', '加速度变化_信号到开仓', '周期数_信号到开仓']
print(analysis_df[entry_change_cols].to_string(index=False))

if '最佳平仓_张力' in analysis_df.columns:
    print("\n4. 最佳平仓特征:")
    exit_cols = ['交易ID', '方向', '最佳平仓时间', '最佳平仓_张力', '最佳平仓_加速度', '最佳平仓_量能']
    print(analysis_df[exit_cols].to_string(index=False))

    print("\n5. 从开仓到平仓的变化:")
    exit_change_cols = ['交易ID', '方向', '张力变化_开仓到平仓', '加速度变化_开仓到平仓', '周期数_开仓到平仓']
    print(analysis_df[exit_change_cols].to_string(index=False))

# 统计规律
print(f"\n{'='*120}")
print("关键规律总结")
print(f"{'='*120}")

print("\n【开仓规律】首次信号 -> 最佳开仓")
print("-" * 120)
for idx, row in analysis_df.iterrows():
    print(f"\n交易#{row['交易ID']} ({row['方向'].upper()}):")
    print(f"  张力: {row['首次信号_张力']:.6f} -> {row['最佳开仓_张力']:.6f}")
    print(f"    变化: {row['张力变化_信号到开仓']:+.6f} ({row['张力变化_信号到开仓']/abs(row['首次信号_张力'])*100 if row['首次信号_张力']!=0 else 0:+.1f}%)")
    print(f"  加速度: {row['首次信号_加速度']:.6f} -> {row['最佳开仓_加速度']:.6f}")
    print(f"    变化: {row['加速度变化_信号到开仓']:+.6f}")
    print(f"  等待: {row['周期数_信号到开仓']:.0f}个周期 ({row['周期数_信号到开仓']*4:.0f}小时)")

if '张力变化_开仓到平仓' in analysis_df.columns:
    print("\n\n【平仓规律】最佳开仓 -> 最佳平仓")
    print("-" * 120)
    for idx, row in analysis_df.iterrows():
        if pd.notna(row['张力变化_开仓到平仓']):
            print(f"\n交易#{row['交易ID']} ({row['方向'].upper()}):")
            print(f"  张力: {row['最佳开仓_张力']:.6f} -> {row['最佳平仓_张力']:.6f}")
            print(f"    变化: {row['张力变化_开仓到平仓']:+.6f} ({row['张力变化_开仓到平仓']/abs(row['最佳开仓_张力'])*100 if row['最佳开仓_张力']!=0 else 0:+.1f}%)")
            print(f"  加速度: {row['最佳开仓_加速度']:.6f} -> {row['最佳平仓_加速度']:.6f}")
            print(f"    变化: {row['加速度变化_开仓到平仓']:+.6f}")
            print(f"  持仓: {row['周期数_开仓到平仓']:.0f}个周期 ({row['周期数_开仓到平仓']*4:.0f}小时)")

# 统计分析
print(f"\n{'='*120}")
print("统计分析")
print(f"{'='*120}")

print("\nSHORT交易的规律:")
short_trades = analysis_df[analysis_df['方向'] == 'short']
if len(short_trades) > 0:
    print(f"  首次信号张力范围: {short_trades['首次信号_张力'].min():.6f} ~ {short_trades['首次信号_张力'].max():.6f}")
    print(f"  首次信号张力平均: {short_trades['首次信号_张力'].mean():.6f}")
    print(f"  最佳开仓张力范围: {short_trades['最佳开仓_张力'].min():.6f} ~ {short_trades['最佳开仓_张力'].max():.6f}")
    print(f"  最佳开仓张力平均: {short_trades['最佳开仓_张力'].mean():.6f}")
    print(f"  张力变化幅度: 平均{short_trades['张力变化_信号到开仓'].mean():.6f}")
    print(f"  等待周期数: 平均{short_trades['周期数_信号到开仓'].mean():.1f}个")

print("\nLONG交易的规律:")
long_trades = analysis_df[analysis_df['方向'] == 'long']
if len(long_trades) > 0:
    print(f"  首次信号张力范围: {long_trades['首次信号_张力'].min():.6f} ~ {long_trades['首次信号_张力'].max():.6f}")
    print(f"  首次信号张力平均: {long_trades['首次信号_张力'].mean():.6f}")
    print(f"  最佳开仓张力范围: {long_trades['最佳开仓_张力'].min():.6f} ~ {long_trades['最佳开仓_张力'].max():.6f}")
    print(f"  最佳开仓张力平均: {long_trades['最佳开仓_张力'].mean():.6f}")
    print(f"  张力变化幅度: 平均{long_trades['张力变化_信号到开仓'].mean():.6f}")
    print(f"  等待周期数: 平均{long_trades['周期数_信号到开仓'].mean():.1f}个")

# 保存结果
analysis_df.to_csv('最佳开仓平仓分析.csv', index=False, encoding='utf-8-sig')
print(f"\n详细分析已保存到: 最佳开仓平仓分析.csv")

print(f"\n{'='*120}")
print("分析完成！")
print(f"{'='*120}")
