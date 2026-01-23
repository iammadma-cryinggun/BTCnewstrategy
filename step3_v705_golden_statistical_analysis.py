# -*- coding: utf-8 -*-
"""
V7.0.5黄金信号完整统计学分析 - 专业数学家方法
==========================================

基于V7.0.5逻辑的完整统计分析：
1. 信号生成：验证5逻辑（含DXY燃料）
2. 入场过滤：V7.0.5过滤器
3. 出场策略：固定止盈止损+5%/-2.5%

统计学方法：
- Youden Index（最优判别阈值）
- Cohen's d（效应量）
- p值检验（统计显著性）
- 95%置信区间

数据范围：2025-08-05 至 2026-01-20（约5.5个月）
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V7.0.5黄金信号完整统计学分析 - 专业数学家方法")
print("=" * 80)

# ==================== 读取数据 ====================
print("\n读取数据...")
df_signals = pd.read_csv('step1_entry_signals.csv', encoding='utf-8')
df_trades = pd.read_csv('v707_backtest_results.csv', encoding='utf-8')

print(f"信号数据: {len(df_signals)}条")
print(f"交易数据: {len(df_trades)}笔")

# ==================== 数据预处理 ====================
print("\n预处理信号数据...")
df_signals['时间'] = pd.to_datetime(df_signals['时间'])

# 提取数值列
for col in ['张力', '加速度', '量能比率', '置信度', 'EMA偏离%']:
    df_signals[col] = df_signals[col].astype(float)

# 计算张力/加速度比
df_signals['张力_加速度比'] = np.abs(df_signals['张力'] / df_signals['加速度'].replace(0, np.nan))

print("\n预处理交易数据...")
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])

# 定义好机会：盈利的交易
df_trades['是好机会'] = df_trades['pnl_pct'] > 0

print(f"\n好机会统计:")
print(f"总交易数: {len(df_trades)}笔")
print(f"好机会数: {df_trades['是好机会'].sum()}笔 ({df_trades['是好机会'].mean()*100:.1f}%)")
print(f"坏机会数: {(~df_trades['是好机会']).sum()}笔 ({(~df_trades['是好机会']).mean()*100:.1f}%)")

# ==================== 按方向分组分析 ====================
print("\n" + "=" * 80)
print("按交易方向分组分析")
print("=" * 80)

for direction in ['short', 'long']:
    df_dir = df_trades[df_trades['direction'] == direction].copy()

    if len(df_dir) == 0:
        continue

    print(f"\n{'SHORT' if direction == 'short' else 'LONG'}信号:")
    print(f"  总交易: {len(df_dir)}笔")
    print(f"  好机会: {df_dir['是好机会'].sum()}笔 ({df_dir['是好机会'].mean()*100:.1f}%)")
    print(f"  平均盈亏: {df_dir['pnl_pct'].mean():+.2f}%")
    print(f"  中位盈亏: {df_dir['pnl_pct'].median():+.2f}%")
    print(f"  平均持仓: {df_dir['hold_periods'].mean():.1f}周期")

    # 按信号类型细分
    print(f"\n  按信号类型细分:")
    for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
        df_sig = df_dir[df_dir['signal_type'] == sig_type]
        if len(df_sig) > 0:
            print(f"    {sig_type}: {len(df_sig)}笔, 胜率{df_sig['是好机会'].mean()*100:.1f}%, 平均{df_sig['pnl_pct'].mean():+.2f}%")

print("\n" + "=" * 80)
print("[OK] V7.0.5基础统计分析完成")
print("=" * 80)
print("\n注意：由于v707_backtest_results.csv与step1_entry_signals.csv时间范围不匹配，")
print("      无法进行完整的黄金信号特征分析（张力、量能等）。")
print("\n建议：需要重新运行匹配时间范围的回测，生成包含信号特征的完整交易数据。")
