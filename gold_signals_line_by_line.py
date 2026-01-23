# -*- coding: utf-8 -*-
"""
黄金信号逐行标注：从信号往后找出最优开仓和平仓位置
====================================================

目标：
1. 对每个通过的V7.0.5信号，逐行往后看K线
2. 标注哪一根K线是最优开仓位置（黄金开仓）
3. 标注哪一根K线是最优平仓位置（黄金平仓）
4. 记录这些黄金位置的特征（张力、加速度、量能、DXY等）
5. 用统计学方法找出规律
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("黄金信号逐行标注分析")
print("=" * 100)

# ==================== 读取数据 ====================
print("\n步骤1: 读取数据...")

df_signals = pd.read_csv('v705_trading_data_passed.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"  通过V7.0.5过滤的信号: {len(df_signals)}个")
print(f"  K线数据: {len(df_full)}条")

# ==================== 计算额外特征 ====================
print("\n步骤2: 计算额外特征...")

df_full['ema20'] = df_full['close'].ewm(span=20, adjust=False).mean()
df_full['price_vs_ema'] = (df_full['close'] - df_full['ema20']) / df_full['ema20']
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

# 读取DXY数据（如果有）
try:
    df_dxy = pd.read_csv('dxy_data.csv', encoding='utf-8-sig')
    df_dxy['date'] = pd.to_datetime(df_dxy['date'])
    df_dxy = df_dxy.sort_values('date')
    print(f"  DXY数据: {len(df_dxy)}条")
    has_dxy = True
except:
    print("  DXY数据未找到，将跳过DXY分析")
    has_dxy = False
    df_dxy = None

# ==================== 逐行处理每个信号 ====================
print("\n步骤3: 逐行处理每个信号...")

all_gold_signals = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['信号时间'])
    signal_type = signal['信号类型']
    direction = signal['方向']
    signal_price = signal['信号价']

    # 在K线数据中找到信号时刻
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx_list) == 0:
        continue

    signal_idx = signal_idx_list[0]

    # ==================== 步骤1: 往后找最优开仓位置 ====================
    print(f"\n处理信号 #{idx+1}: {signal_type} @ {signal_time}")

    # 向前看最多10个周期，寻找最优开仓价格
    look_ahead_entry = min(10, len(df_full) - signal_idx - 1)

    best_entry_price = signal_price
    best_entry_period = 0
    best_entry_idx = signal_idx

    if direction == 'short':
        # 做空：找最高价格开仓
        for period in range(1, look_ahead_entry + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if future_price > best_entry_price:
                best_entry_price = future_price
                best_entry_period = period
                best_entry_idx = future_idx
    else:  # long
        # 做多：找最低价格开仓
        for period in range(1, look_ahead_entry + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if future_price < best_entry_price:
                best_entry_price = future_price
                best_entry_period = period
                best_entry_idx = future_idx

    # 记录信号前的特征
    pre_tension = df_full.loc[signal_idx-5:signal_idx, 'tension'].mean() if signal_idx >= 5 else df_full.loc[signal_idx, 'tension']
    pre_accel = df_full.loc[signal_idx-5:signal_idx, 'acceleration'].mean() if signal_idx >= 5 else df_full.loc[signal_idx, 'acceleration']
    pre_volume = df_full.loc[signal_idx-5:signal_idx, 'volume_ratio'].mean() if signal_idx >= 5 else df_full.loc[signal_idx, 'volume_ratio']

    # 记录信号时刻的特征
    signal_tension = signal['张力']
    signal_accel = signal['加速度']
    signal_volume = signal['量能比率']

    # 记录黄金开仓时刻的特征
    gold_entry_tension = df_full.loc[best_entry_idx, 'tension']
    gold_entry_accel = df_full.loc[best_entry_idx, 'acceleration']
    gold_entry_volume = df_full.loc[best_entry_idx, 'volume_ratio']

    # 价格优势（等待带来的价格改善）
    if direction == 'short':
        price_advantage = (best_entry_price - signal_price) / signal_price * 100
    else:
        price_advantage = (signal_price - best_entry_price) / signal_price * 100

    print(f"  黄金开仓: 等待{best_entry_period}周期, 价格优势{price_advantage:.2f}%")

    # ==================== 步骤2: 从开仓往后找最优平仓位置 ====================

    # 向前看最多30个周期，寻找最优平仓价格
    look_ahead_exit = min(30, len(df_full) - best_entry_idx - 1)

    best_exit_price = best_entry_price
    best_exit_period = 0
    best_exit_idx = best_entry_idx
    best_pnl = -999

    for period in range(1, look_ahead_exit + 1):
        future_idx = best_entry_idx + period
        future_price = df_full.loc[future_idx, 'close']

        if direction == 'short':
            pnl = (best_entry_price - future_price) / best_entry_price * 100
        else:  # long
            pnl = (future_price - best_entry_price) / best_entry_price * 100

        if pnl > best_pnl:
            best_pnl = pnl
            best_exit_price = future_price
            best_exit_period = period
            best_exit_idx = future_idx

    # 记录黄金平仓时刻的特征
    gold_exit_tension = df_full.loc[best_exit_idx, 'tension']
    gold_exit_accel = df_full.loc[best_exit_idx, 'acceleration']
    gold_exit_volume = df_full.loc[best_exit_idx, 'volume_ratio']

    print(f"  黄金平仓: 持仓{best_exit_period}周期, 盈亏{best_pnl:.2f}%")

    # ==================== 步骤3: 计算DXY燃料（如果有数据） ====================

    dxy_fuel_signal = None
    dxy_fuel_entry = None
    dxy_fuel_exit = None

    if has_dxy:
        # 信号时刻的DXY燃料
        dxy_fuel_signal = calculate_dxy_fuel(df_dxy, signal_time)
        # 开仓时刻的DXY燃料
        entry_time = df_full.loc[best_entry_idx, 'timestamp']
        dxy_fuel_entry = calculate_dxy_fuel(df_dxy, entry_time)
        # 平仓时刻的DXY燃料
        exit_time = df_full.loc[best_exit_idx, 'timestamp']
        dxy_fuel_exit = calculate_dxy_fuel(df_dxy, exit_time)

    # ==================== 保存黄金信号 ====================

    all_gold_signals.append({
        '信号索引': idx,
        '信号时间': signal_time,
        '信号类型': signal_type,
        '方向': direction,

        # 信号前特征
        '信号前5周期平均张力': pre_tension,
        '信号前5周期平均加速度': pre_accel,
        '信号前5周期平均量能': pre_volume,

        # 信号时刻特征
        '信号时刻张力': signal_tension,
        '信号时刻加速度': signal_accel,
        '信号时刻量能': signal_volume,
        '信号时刻价格': signal_price,

        # 黄金开仓特征
        '黄金开仓等待周期': best_entry_period,
        '黄金开仓时间': df_full.loc[best_entry_idx, 'timestamp'],
        '黄金开仓价格': best_entry_price,
        '黄金开仓张力': gold_entry_tension,
        '黄金开仓加速度': gold_entry_accel,
        '黄金开仓量能': gold_entry_volume,
        '价格优势%': price_advantage,

        # 黄金平仓特征
        '黄金平仓持仓周期': best_exit_period,
        '黄金平仓时间': df_full.loc[best_exit_idx, 'timestamp'],
        '黄金平仓价格': best_exit_price,
        '黄金平仓张力': gold_exit_tension,
        '黄金平仓加速度': gold_exit_accel,
        '黄金平仓量能': gold_exit_volume,
        '最优盈亏%': best_pnl,

        # DXY燃料
        'DXY燃料_信号时刻': dxy_fuel_signal,
        'DXY燃料_开仓时刻': dxy_fuel_entry,
        'DXY燃料_平仓时刻': dxy_fuel_exit,
    })

# ==================== 保存黄金信号数据 ====================
df_gold = pd.DataFrame(all_gold_signals)

output_file = 'v705_gold_signals_complete.csv'
df_gold.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[OK] 已保存黄金信号数据: {output_file}")

# ==================== 统计学分析 ====================
print("\n" + "=" * 100)
print("统计学分析：找出黄金位置的特征规律")
print("=" * 100)

for signal_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    df_type = df_gold[df_gold['信号类型'] == signal_type].copy()

    if len(df_type) < 5:
        continue

    # 按盈亏中位数分好机会/坏机会
    pnl_median = df_type['最优盈亏%'].median()
    df_type['是好机会'] = df_type['最优盈亏%'] >= pnl_median

    good = df_type[df_type['是好机会'] == True]
    bad = df_type[df_type['是好机会'] == False]

    print(f"\n{'=' * 100}")
    print(f"{signal_type} - 黄金位置特征分析")
    print(f"{'=' * 100}")
    print(f"好机会: {len(good)}个, 平均盈亏 {good['最优盈亏%'].mean():.2f}%")
    print(f"坏机会: {len(bad)}个, 平均盈亏 {bad['最优盈亏%'].mean():.2f}%")

    # 分析各个阶段的特征
    stages = [
        ('信号前', '信号前5周期平均张力', '信号前5周期平均加速度', '信号前5周期平均量能'),
        ('信号时', '信号时刻张力', '信号时刻加速度', '信号时刻量能'),
        ('黄金开仓', '黄金开仓张力', '黄金开仓加速度', '黄金开仓量能'),
        ('黄金平仓', '黄金平仓张力', '黄金平仓加速度', '黄金平仓量能'),
    ]

    for stage_name, tension_col, accel_col, volume_col in stages:
        print(f"\n【{stage_name}】")

        for col in [tension_col, accel_col, volume_col]:
            if col not in df_type.columns:
                continue

            good_mean = good[col].mean()
            bad_mean = bad[col].mean()
            t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')

            n1, n2 = len(good), len(bad)
            var1, var2 = good[col].var(), bad[col].var()
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            effect = '超大' if abs(cohens_d) > 1.2 else '大' if abs(cohens_d) > 0.8 else '中等' if abs(cohens_d) > 0.5 else '小'

            if p_val < 0.1 or abs(cohens_d) > 0.5:
                print(f"  {col}: 好={good_mean:.4f}, 坏={bad_mean:.4f}, d={cohens_d:.3f}({effect}), p={p_val:.4f} {sig}")

print("\n" + "=" * 100)
print("[完成] 黄金信号逐行标注完成！")
print("=" * 100)


def calculate_dxy_fuel(df_dxy, current_time):
    """计算DXY燃料"""
    try:
        mask = df_dxy['date'] <= current_time
        recent = df_dxy[mask].tail(5)

        if len(recent) < 3:
            return 0.0

        closes = recent['Close'].values.astype(float)

        change_1 = (closes[-1] - closes[-2]) / closes[-2]
        change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1

        acceleration = change_1 - change_2
        fuel = -acceleration * 100

        return float(fuel)
    except:
        return 0.0
