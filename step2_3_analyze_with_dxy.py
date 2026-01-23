# -*- coding: utf-8 -*-
"""
第二步+第三步：分析开仓信号的最优时机（含DXY燃料）

- 分析每个开仓信号后的最优开仓时机（首次 vs 等待）
- 分析最优平仓时机
- 统计DXY燃料的影响
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("第二步+第三步：分析最优时机（含DXY燃料）")
print("=" * 80)
print()

# ==================== 读取数据 ====================
print("读取数据...")

df_signals = pd.read_csv('step1_entry_signals_with_dxy.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"[OK] 开仓信号: {len(df_signals)}个")
print(f"[OK] 完整数据: {len(df_full)}条")
print()

# ==================== 分析每个开仓信号的后续走势 ====================
print("正在分析每个信号的后续走势...")

analysis_results = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    signal_tension = signal['张力']
    signal_dxy_fuel = signal.get('DXY燃料', 0.0)

    # 确定交易方向（验证5反向策略）
    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'
    else:
        direction = 'long'

    # 在完整数据中找到这个信号的位置
    signal_idx = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx) == 0:
        continue

    signal_idx = signal_idx[0]

    # 分析后续30个周期（5天）
    look_ahead_periods = 30

    if signal_idx + look_ahead_periods >= len(df_full):
        continue

    future_performance = []

    for period in range(1, look_ahead_periods + 1):
        future_idx = signal_idx + period
        future_price = df_full.loc[future_idx, 'close']

        if direction == 'short':
            pnl = (signal_price - future_price) / signal_price * 100
        else:
            pnl = (future_price - signal_price) / signal_price * 100

        future_performance.append({
            'period': period,
            'price': future_price,
            'pnl': pnl
        })

    df_future = pd.DataFrame(future_performance)

    # ==================== 分析最优开仓时机 ====================
    # 首次开仓盈亏（第1周期）
    first_entry_pnl = df_future.iloc[0]['pnl']

    # 找最优开仓点（最大盈亏）
    best_entry_idx = df_future['pnl'].idxmax()
    best_entry_pnl = df_future.loc[best_entry_idx, 'pnl']
    best_entry_period = df_future.loc[best_entry_idx, 'period']

    # 价格优势 = (最优开仓价 - 首次开仓价) / 首次开仓价 * 100
    best_entry_price = df_future.loc[best_entry_idx, 'price']
    price_advantage = (best_entry_price - signal_price) / signal_price * 100

    # ==================== 分析最优平仓时机 ====================
    # 最优平仓点（未来30周期内最大盈利）
    best_exit_pnl = df_future['pnl'].max()
    best_exit_period = df_future.loc[df_future['pnl'].idxmax(), 'period']

    # ==================== 记录结果 ====================
    analysis_results.append({
        '信号时间': signal_time,
        '信号类型': signal_type,
        '方向': direction,
        '开仓价': signal_price,
        '张力': signal_tension,
        'DXY燃料': signal_dxy_fuel,
        '首次开仓盈亏%': first_entry_pnl,
        '最优开仓盈亏%': best_entry_pnl,
        '最优开仓周期': best_entry_period,
        '价格优势%': price_advantage,
        '最优平仓盈亏%': best_exit_pnl,
        '最优平仓周期': best_exit_period
    })

df_analysis = pd.DataFrame(analysis_results)

print(f"[OK] 分析完成: {len(df_analysis)}个信号")
print()

# ==================== 判断好机会 ====================
print("正在判断好机会...")

# 好机会定义：最优平仓盈利>0
df_analysis['是好机会'] = df_analysis['最优平仓盈亏%'] > 0

# 好机会类型
def classify_opportunity(row):
    if not row['是好机会']:
        return 0  # 不是好机会

    if row['首次开仓盈亏%'] > 0:
        return 1  # 首次开仓就是好机会

    if row['最优开仓盈亏%'] > 0:
        return 2  # 等待后是好机会

    return 3  # 其他情况

df_analysis['好机会类型'] = df_analysis.apply(classify_opportunity, axis=1)

# 好机会盈亏 = max(首次开仓, 最优开仓)
df_analysis['好机会盈亏%'] = df_analysis.apply(
    lambda row: max(row['首次开仓盈亏%'], row['最优开仓盈亏%'])
    if row['是好机会'] else row['首次开仓盈亏%'],
    axis=1
)

# 好机会平仓周期
def get_best_exit_period(row):
    if not row['是好机会']:
        return 1

    if row['首次开仓盈亏%'] > 0:
        return 1

    return row['最优开仓周期']

df_analysis['好机会平仓周期'] = df_analysis.apply(get_best_exit_period, axis=1)

# 张力变化 = (最优开仓时张力 - 首次张力) / |首次张力| * 100
# 由于我们没有记录每个周期的张力，这里简化处理
df_analysis['张力变化%'] = 0.0  # 暂时设为0，后续可优化

print(f"[OK] 好机会判断完成")
print()

# ==================== 统计摘要 ====================
print("=" * 80)
print("统计摘要")
print("=" * 80)

good_opportunities = df_analysis[df_analysis['是好机会'] == True]
bad_opportunities = df_analysis[df_analysis['是好机会'] == False]

print(f"\n总信号数: {len(df_analysis)}个")
print(f"好机会: {len(good_opportunities)}个 ({len(good_opportunities)/len(df_analysis)*100:.1f}%)")
print(f"坏机会: {len(bad_opportunities)}个 ({len(bad_opportunities)/len(df_analysis)*100:.1f}%)")

print(f"\n按方向统计:")
for direction in ['short', 'long']:
    df_dir = df_analysis[df_analysis['方向'] == direction]
    good_dir = df_dir[df_dir['是好机会'] == True]

    print(f"\n{direction.upper()}信号:")
    print(f"  总数: {len(df_dir)}个")
    print(f"  好机会: {len(good_dir)}个 ({len(good_dir)/len(df_dir)*100:.1f}%)")
    print(f"  平均首次盈亏: {df_dir['首次开仓盈亏%'].mean():+.2f}%")
    print(f"  平均最优盈亏: {df_dir['最优开仓盈亏%'].mean():+.2f}%")

print(f"\n按信号类型统计:")
for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_sig = df_analysis[df_analysis['信号类型'] == sig_type]
    good_sig = df_sig[df_sig['是好机会'] == True]

    if len(df_sig) > 0:
        print(f"\n{sig_type}:")
        print(f"  总数: {len(df_sig)}个")
        print(f"  好机会: {len(good_sig)}个 ({len(good_sig)/len(df_sig)*100:.1f}%)")
        print(f"  平均首次盈亏: {df_sig['首次开仓盈亏%'].mean():+.2f}%")
        print(f"  平均最优盈亏: {df_sig['最优开仓盈亏%'].mean():+.2f}%")

print(f"\n按DXY燃料统计:")
with_dxy = df_analysis[df_analysis['DXY燃料'] > 0]
without_dxy = df_analysis[df_analysis['DXY燃料'] == 0]

print(f"\n有DXY燃料: {len(with_dxy)}个")
if len(with_dxy) > 0:
    good_with_dxy = with_dxy[(with_dxy['是好机会'] == True)]
    print(f"  好机会: {len(good_with_dxy)}个 ({len(good_with_dxy)/len(with_dxy)*100:.1f}%)")
    print(f"  平均首次盈亏: {with_dxy['首次开仓盈亏%'].mean():+.2f}%")
    print(f"  平均最优盈亏: {with_dxy['最优开仓盈亏%'].mean():+.2f}%")

print(f"\n无DXY燃料: {len(without_dxy)}个")
if len(without_dxy) > 0:
    good_without_dxy = without_dxy[(without_dxy['是好机会'] == True)]
    print(f"  好机会: {len(good_without_dxy)}个 ({len(good_without_dxy)/len(without_dxy)*100:.1f}%)")
    print(f"  平均首次盈亏: {without_dxy['首次开仓盈亏%'].mean():+.2f}%")
    print(f"  平均最优盈亏: {without_dxy['最优开仓盈亏%'].mean():+.2f}%")

print()
print("=" * 80)
print("[OK] 第二步+第三步完成")
print("=" * 80)

# ==================== 保存数据 ====================
print("\n正在保存数据...")
df_analysis.to_csv('step2_3_analysis_results_with_dxy.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 已保存分析结果: step2_3_analysis_results_with_dxy.csv")
