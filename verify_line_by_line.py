# -*- coding: utf-8 -*-
"""
逐行统计归纳：黄金信号完整生命周期验证
==========================================

目标：
1. 普通信号 vs 开仓信号 区分
2. 开仓信号发生后找最优价格
3. 最优价格后找翻前最优平仓价格
4. 完整的逐行统计归纳
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("黄金信号完整生命周期逐行验证")
print("=" * 100)

# 读取数据
df_signals = pd.read_csv('step1_all_signals_v5_correct.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算额外特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

print(f"\n数据加载完成:")
print(f"  - 信号总数: {len(df_signals)}个")
print(f"  - 完整K线数据: {len(df_full)}条")
print(f"\n信号类型分布:")
print(df_signals['信号类型'].value_counts())

# ==================== 定义：普通信号 vs 开仓信号 ====================
print("\n" + "=" * 100)
print("定义说明")
print("=" * 100)
print("""
【普通信号】 = 初始信号检测时刻
  - 定义：满足信号条件的第一个时间点
  - 特征：该时刻的张力、加速度、量能比率
  - 操作：此时只是观察，不立即交易

【开仓信号】 = 实际执行交易的时刻
  - 定义：普通信号出现后，找到最优价格的时机
  - 寻找方法：在前10周期内，寻找价格最优位置
  - 特征：开仓时刻的张力、加速度、量能比率
  - 操作：在此时刻实际开仓

【平仓信号】 = 实际平仓的时刻
  - 定义：开仓后，在未来30周期内找到的最优盈利点
  - 寻找方法：遍历未来30周期，找到盈亏最大的点
  - 特征：平仓时刻的张力、加速度、量能比率
  - 操作：在此时刻实际平仓
""")

# ==================== 逐行分析每个信号 ====================
print("\n" + "=" * 100)
print("开始逐行分析每个信号的完整生命周期...")
print("=" * 100)

all_signal_lifecycles = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    direction = signal['方向']

    # 在完整数据中找到这个信号的位置
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx_list) == 0 or signal_idx_list[0] < 5:
        continue

    signal_idx = signal_idx_list[0]

    # ========== 普通信号特征 ==========
    normal_tension = signal['张力']
    normal_accel = signal['加速度']
    normal_volume = signal.get('量能比率', 1.0)

    # 信号前5周期平均特征
    pre_tension = df_full.loc[signal_idx-5:signal_idx, 'tension'].mean()
    pre_accel = df_full.loc[signal_idx-5:signal_idx, 'acceleration'].mean()
    pre_volume = df_full.loc[signal_idx-5:signal_idx, 'volume_ratio'].mean()

    # ========== 寻找开仓信号（最优价格）==========
    look_ahead_entry = min(10, len(df_full) - signal_idx - 1)
    best_entry_price = signal_price
    best_entry_period = 0
    direction_changed = False

    if direction == 'short':
        # 做空：找最高价格开仓
        for period in range(1, look_ahead_entry + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if future_price > best_entry_price:
                best_entry_price = future_price
                best_entry_period = period

            # 检查方向是否改变（连续2周期价格突破信号价）
            if period >= 2:
                prev1_price = df_full.loc[signal_idx + period - 1, 'close']
                prev2_price = df_full.loc[signal_idx + period - 2, 'close']
                if (prev1_price < signal_price and prev2_price < signal_price):
                    direction_changed = True
                    break
    else:
        # 做多：找最低价格开仓
        for period in range(1, look_ahead_entry + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if future_price < best_entry_price:
                best_entry_price = future_price
                best_entry_period = period

            # 检查方向是否改变（连续2周期价格突破信号价）
            if period >= 2:
                prev1_price = df_full.loc[signal_idx + period - 1, 'close']
                prev2_price = df_full.loc[signal_idx + period - 2, 'close']
                if (prev1_price > signal_price and prev2_price > signal_price):
                    direction_changed = True
                    break

    # ========== 开仓信号特征 ==========
    entry_idx = signal_idx + best_entry_period
    entry_tension = df_full.loc[entry_idx, 'tension']
    entry_accel = df_full.loc[entry_idx, 'acceleration']
    entry_volume = df_full.loc[entry_idx, 'volume_ratio']

    price_advantage = (best_entry_price - signal_price) / signal_price * 100 if direction == 'short' else (signal_price - best_entry_price) / signal_price * 100

    # ========== 寻找平仓信号（最优盈利点）==========
    look_ahead_exit = min(30, len(df_full) - entry_idx - 1)
    best_exit_price = best_entry_price
    best_exit_period = 0
    best_pnl = -999

    for period in range(1, look_ahead_exit + 1):
        future_idx = entry_idx + period
        future_price = df_full.loc[future_idx, 'close']

        if direction == 'short':
            pnl = (best_entry_price - future_price) / best_entry_price * 100
        else:
            pnl = (future_price - best_entry_price) / best_entry_price * 100

        if pnl > best_pnl:
            best_pnl = pnl
            best_exit_price = future_price
            best_exit_period = period

    # ========== 平仓信号特征 ==========
    exit_idx = entry_idx + best_exit_period
    exit_tension = df_full.loc[exit_idx, 'tension']
    exit_accel = df_full.loc[exit_idx, 'acceleration']
    exit_volume = df_full.loc[exit_idx, 'volume_ratio']

    # ========== 保存完整生命周期 ==========
    all_signal_lifecycles.append({
        '信号索引': idx,
        '信号类型': signal_type,
        '方向': direction,

        # 普通信号
        '普通信号时间': signal_time,
        '普通信号价格': signal_price,
        '普通信号张力': normal_tension,
        '普通信号加速度': normal_accel,
        '普通信号量能比率': normal_volume,

        # 信号前状态
        '信号前5周期平均张力': pre_tension,
        '信号前5周期平均加速度': pre_accel,
        '信号前5周期平均量能比率': pre_volume,

        # 开仓信号
        '开仓等待周期': best_entry_period,
        '开仓时间': df_full.loc[entry_idx, 'timestamp'],
        '开仓价格': best_entry_price,
        '开仓张力': entry_tension,
        '开仓加速度': entry_accel,
        '开仓量能比率': entry_volume,
        '价格优势%': price_advantage,
        '方向是否改变': direction_changed,

        # 平仓信号
        '持仓周期': best_exit_period,
        '平仓时间': df_full.loc[exit_idx, 'timestamp'],
        '平仓价格': best_exit_price,
        '平仓张力': exit_tension,
        '平仓加速度': exit_accel,
        '平仓量能比率': exit_volume,
        '最优盈亏%': best_pnl,
    })

df_lifecycle = pd.DataFrame(all_signal_lifecycles)
print(f"\n[OK] 完成逐行分析: {len(df_lifecycle)}个信号")

# 保存完整生命周期数据
df_lifecycle.to_csv('complete_signal_lifecycle.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 已保存: complete_signal_lifecycle.csv")

# ==================== 逐行展示前10个信号示例 ====================
print("\n" + "=" * 100)
print("【示例：前10个信号的完整生命周期】")
print("=" * 100)

for i in range(min(10, len(df_lifecycle))):
    row = df_lifecycle.iloc[i]
    print(f"\n{'━' * 100}")
    print(f"信号 #{i+1}: {row['信号类型']} ({row['方向'].upper()})")
    print(f"{'━' * 100}")

    print(f"\n【阶段1：普通信号（初始检测）】")
    print(f"  时间: {row['普通信号时间']}")
    print(f"  价格: ${row['普通信号价格']:.2f}")
    print(f"  张力: {row['普通信号张力']:.4f}")
    print(f"  加速度: {row['普通信号加速度']:.4f}")
    print(f"  量能比率: {row['普通信号量能比率']:.2f}")

    print(f"\n【阶段2：信号前状态（前5周期平均）】")
    print(f"  张力: {row['信号前5周期平均张力']:.4f}")
    print(f"  加速度: {row['信号前5周期平均加速度']:.4f}")
    print(f"  量能比率: {row['信号前5周期平均量能比率']:.2f}")

    print(f"\n【阶段3：开仓信号（最优价格）】")
    print(f"  等待周期: {row['开仓等待周期']}周期")
    print(f"  时间: {row['开仓时间']}")
    print(f"  价格: ${row['开仓价格']:.2f}")
    print(f"  张力: {row['开仓张力']:.4f}")
    print(f"  加速度: {row['开仓加速度']:.4f}")
    print(f"  量能比率: {row['开仓量能比率']:.2f}")
    print(f"  价格优势: {row['价格优势%']:.2f}%")

    print(f"\n【阶段4：平仓信号（最优盈利点）】")
    print(f"  持仓周期: {row['持仓周期']}周期")
    print(f"  时间: {row['平仓时间']}")
    print(f"  价格: ${row['平仓价格']:.2f}")
    print(f"  张力: {row['平仓张力']:.4f}")
    print(f"  加速度: {row['平仓加速度']:.4f}")
    print(f"  量能比率: {row['平仓量能比率']:.2f}")
    print(f"  最优盈亏: {row['最优盈亏%']:.2f}%")

# ==================== 统计归纳：不同阶段的特征差异 ====================
print("\n" + "=" * 100)
print("【统计归纳：好机会 vs 坏机会 的特征差异】")
print("=" * 100)

def analyze_stage(df, stage_prefix, stage_name):
    """分析某个阶段的统计特征"""

    # 按盈亏中位数分好机会/坏机会
    pnl_median = df['最优盈亏%'].median()
    df['是好机会'] = df['最优盈亏%'] >= pnl_median

    good = df[df['是好机会'] == True]
    bad = df[df['是好机会'] == False]

    print(f"\n{'=' * 100}")
    print(f"{stage_name}")
    print(f"{'=' * 100}")
    print(f"好机会: {len(good)}个, 平均盈亏 {good['最优盈亏%'].mean():.2f}%")
    print(f"坏机会: {len(bad)}个, 平均盈亏 {bad['最优盈亏%'].mean():.2f}%")

    # 提取该阶段的特征
    features = {
        '张力': stage_prefix + '张力',
        '加速度': stage_prefix + '加速度',
        '量能比率': stage_prefix + '量能比率'
    }

    print(f"\n特征差异统计:")
    print(f"{'特征':<15} {'好机会均值':<12} {'坏机会均值':<12} {'Cohen\'s d':<12} {'p值':<12} {'显著性'}")
    print(f"{'-' * 100}")

    for name, col in features.items():
        if col not in df.columns:
            continue

        good_mean = good[col].mean()
        bad_mean = bad[col].mean()
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')

        # Cohen's d
        n1, n2 = len(good), len(bad)
        var1, var2 = good[col].var(), bad[col].var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        effect = '超大' if abs(cohens_d) > 1.2 else '大' if abs(cohens_d) > 0.8 else '中等' if abs(cohens_d) > 0.5 else '小'

        print(f"{name:<15} {good_mean:<12.4f} {bad_mean:<12.4f} {cohens_d:<12.3f} {p_val:<12.4f} {sig} {effect}")

# 分析BEARISH_SINGULARITY
df_short = df_lifecycle[df_lifecycle['信号类型'] == 'BEARISH_SINGULARITY']
analyze_stage(df_short, '普通信号', '【BEARISH_SINGULARITY - 普通信号阶段】')
analyze_stage(df_short, '开仓', '【BEARISH_SINGULARITY - 开仓信号阶段】')
analyze_stage(df_short, '平仓', '【BEARISH_SINGULARITY - 平仓信号阶段】')

# 分析BULLISH_SINGULARITY
df_long = df_lifecycle[df_lifecycle['信号类型'] == 'BULLISH_SINGULARITY']
analyze_stage(df_long, '普通信号', '【BULLISH_SINGULARITY - 普通信号阶段】')
analyze_stage(df_long, '开仓', '【BULLISH_SINGULARITY - 开仓信号阶段】')
analyze_stage(df_long, '平仓', '【BULLISH_SINGULARITY - 平仓信号阶段】')

# ==================== 关键周期统计 ====================
print("\n" + "=" * 100)
print("【关键周期统计】")
print("=" * 100)

for signal_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY']:
    df_type = df_lifecycle[df_lifecycle['信号类型'] == signal_type]

    print(f"\n{signal_type}:")

    # 好机会/坏机会
    pnl_median = df_type['最优盈亏%'].median()
    df_type['是好机会'] = df_type['最优盈亏%'] >= pnl_median

    print(f"\n开仓等待周期分布:")
    for wait in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        count = len(df_type[df_type['开仓等待周期'] == wait])
        if count > 0:
            good_count = len(df_type[(df_type['开仓等待周期'] == wait) & (df_type['是好机会'] == True)])
            print(f"  等待{wait}周期: {count}个, 好机会率 {good_count/count*100:.1f}%")

    print(f"\n持仓周期分布:")
    for period in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]:
        count = len(df_type[df_type['持仓周期'] == period])
        if count > 0:
            good_count = len(df_type[(df_type['持仓周期'] == period) & (df_type['是好机会'] == True)])
            print(f"  持仓{period}周期: {count}个, 好机会率 {good_count/count*100:.1f}%")

print("\n" + "=" * 100)
print("[COMPLETE] 逐行统计归纳完成!")
print("=" * 100)
