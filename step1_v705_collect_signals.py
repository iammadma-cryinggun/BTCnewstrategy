# -*- coding: utf-8 -*-
"""
V7.0.5黄金信号统计分析 - 专业统计学方法
==========================================

基于V7.0.5逻辑：
1. 信号生成：验证5逻辑（含DXY燃料）
2. V7.0.5入场过滤器
3. 固定止盈止损（+5%/-2.5%）

统计学方法：
- Youden Index（最优判别阈值）
- Cohen's d（效应量）
- p值检验（统计显著性）
- 95%置信区间

数据范围：2025-07-24 至 2026-01-20（约6个月）
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V7.0.5黄金信号统计分析 - 专业统计学方法")
print("=" * 80)

# ==================== 读取数据 ====================
df = pd.read_csv('step1_entry_signals.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\n数据范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")
print(f"总K线数：{len(df)}条")

# ==================== 计算V7.0.5过滤器所需指标 ====================
print("\n计算V7.0.5过滤器指标...")

# 计算EMA20
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

# 计算价格vs EMA
df['price_vs_ema'] = (df['close'] - df['ema20']) / df['ema20']

# 计算量能比率
df['avg_volume_20'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['avg_volume_20']

# 填充前20行的NaN
df['price_vs_ema'].fillna(0, inplace=True)
df['volume_ratio'].fillna(1.0, inplace=True)

print("✓ EMA20、价格偏离、量能比率计算完成")

# ==================== V7.0.5过滤器 ====================
def apply_v705_filter(row):
    """
    V7.0.5入场过滤器

    返回: (should_pass, reason)
    """
    signal_type = row['signal_type']
    acceleration = row['acceleration']
    price_vs_ema = row['price_vs_ema']
    volume_ratio = row['volume_ratio']

    if signal_type == 'HIGH_OSCILLATION':
        # 牛市回调阈值2%
        if price_vs_ema > 0.02:
            return False, f"牛市回调({price_vs_ema*100:.1f}%)"

        # 无向下动能
        if acceleration >= 0:
            return False, f"无向下动能(a={acceleration:.3f})"

        # 高位放量
        if volume_ratio > 1.1:
            return False, f"高位放量({volume_ratio:.2f})"

        return True, "通过V7.0.5"

    elif signal_type == 'LOW_OSCILLATION':
        # V7.0.5: 完全移除所有过滤
        return True, "通过V7.0.5"

    elif signal_type == 'BULLISH_SINGULARITY':
        # 量能阈值0.95
        if volume_ratio > 0.95:
            return False, f"量能放大({volume_ratio:.2f})"

        # 主升浪过滤
        if price_vs_ema > 0.05:
            return False, f"主升浪({price_vs_ema*100:.1f}%)"

        return True, "通过V7.0.5"

    elif signal_type == 'BEARISH_SINGULARITY':
        # 主跌浪过滤
        if price_vs_ema < -0.05:
            return False, f"主跌浪({price_vs_ema*100:.1f}%)"

        return True, "通过V7.0.5"

    return True, "通过V7.0.5"

# 应用V7.0.5过滤器
print("\n应用V7.0.5过滤器...")
df_signals = df[df['signal_type'].notna()].copy()

# 应用过滤
filter_results = df_signals.apply(apply_v705_filter, axis=1, result_type='expand')
df_signals['v705_should_trade'] = filter_results[0]
df_signals['v705_reason'] = filter_results[1]

# 统计
total_signals = len(df_signals)
v705_passed = len(df_signals[df_signals['v705_should_trade'] == True])
v705_filtered = len(df_signals[df_signals['v705_should_trade'] == False])

print(f"\nV7.0.5过滤结果：")
print(f"  总信号数：{total_signals}个")
print(f"  通过V7.0.5：{v705_passed}个 ({v705_passed/total_signals*100:.1f}%)")
print(f"  被过滤：{v705_filtered}个 ({v705_filtered/total_signals*100:.1f}%)")

# 按信号类型统计
print(f"\n按信号类型统计：")
for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    sig_data = df_signals[df_signals['signal_type'] == sig_type]
    passed = len(sig_data[sig_data['v705_should_trade'] == True])
    total = len(sig_data)
    if total > 0:
        print(f"  {sig_type}: {passed}/{total} ({passed/total*100:.1f}%)")

# ==================== 保存通过V7.0.5的信号 ====================
df_v705_passed = df_signals[df_signals['v705_should_trade'] == True].copy()
df_v705_passed.to_csv('v705_passed_signals.csv', index=False, encoding='utf-8')

print(f"\n✓ 已保存通过V7.0.5的信号到 v705_passed_signals.csv")
print(f"  共{len(df_v705_passed)}个信号准备进行黄金信号分析")

# ==================== 准备黄金信号分析 ====================
print("\n" + "=" * 80)
print("开始黄金信号统计学分析...")
print("=" * 80)

# 后续分析将在下一个脚本中完成
print("\n✓ 数据准备完成，请运行 step2_v705_golden_analysis.py 进行统计学分析")
