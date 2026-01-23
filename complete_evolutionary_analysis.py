# -*- coding: utf-8 -*-
"""
完整的黄金信号演变分析 - 统计学数学家方法
=====================================

完整分析链条：
1. 信号出现前的市场状态
2. 信号出现时的市场特征
3. 从信号到黄金开仓的演变过程
4. 黄金开仓点的数值共性
5. 从开仓到黄金平仓的演变
6. 黄金平仓点的数值共性
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("完整的黄金信号演变分析 - 统计学数学家方法")
print("=" * 80)

# ==================== 读取数据 ====================
df_signals = pd.read_csv('step1_entry_signals_with_dxy.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']
df_full['tension_accel_ratio'] = np.abs(df_full['tension'] / df_full['acceleration'].replace(0, np.nan))

print(f"\n总信号数: {len(df_signals)}个")

# ==================== 提取完整的信号演变数据 ====================
print("\n正在提取完整的信号演变数据...")

all_signal_data = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    signal_tension = signal['张力']
    signal_accel = signal['加速度']
    signal_volume_ratio = signal.get('量能比率', 1.0)
    signal_dxy_fuel = signal.get('DXY燃料', 0.0)

    # 确定交易方向
    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'
    else:
        direction = 'long'

    # 在完整数据中找到这个信号的位置
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx_list) == 0 or signal_idx_list[0] < 5:
        continue

    signal_idx = signal_idx_list[0]

    # 1. 信号出现前的市场状态（前5个周期）
    pre_tension = df_full.loc[signal_idx-5:signal_idx, 'tension'].mean()
    pre_accel = df_full.loc[signal_idx-5:signal_idx, 'acceleration'].mean()
    pre_volume = df_full.loc[signal_idx-5:signal_idx, 'volume_ratio'].mean()
    pre_dxy = df_full.loc[signal_idx-5:signal_idx, 'dxy_fuel'].mean()

    # 2. 找最优开仓点（前10周期内）
    look_ahead_entry = min(10, len(df_full) - signal_idx - 1)
    best_entry_price = signal_price
    best_entry_period = 0
    direction_changed = False

    if direction == 'short':
        # 做空：找最高价（在价格开始下跌之前）
        for period in range(1, look_ahead_entry + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']
            if future_price > best_entry_price:
                best_entry_price = future_price
                best_entry_period = period
            # 检测方向改变：价格开始下跌（连续2个周期下跌）
            if period >= 2:
                prev1_price = df_full.loc[signal_idx + period - 1, 'close']
                prev2_price = df_full.loc[signal_idx + period - 2, 'close']
                if (prev1_price < signal_price and prev2_price < signal_price):
                    direction_changed = True
                    break
    else:
        # 做多：找最低价（在价格开始上涨之前）
        for period in range(1, look_ahead_entry + 1):
            future_idx = signal_idx + period
            future_price = df_full.loc[future_idx, 'close']
            if future_price < best_entry_price:
                best_entry_price = future_price
                best_entry_period = period
            # 检测方向改变：价格开始上涨（连续2个周期上涨）
            if period >= 2:
                prev1_price = df_full.loc[signal_idx + period - 1, 'close']
                prev2_price = df_full.loc[signal_idx + period - 2, 'close']
                if (prev1_price > signal_price and prev2_price > signal_price):
                    direction_changed = True
                    break

    # 3. 黄金开仓点的特征
    entry_idx = signal_idx + best_entry_period
    entry_tension = df_full.loc[entry_idx, 'tension']
    entry_accel = df_full.loc[entry_idx, 'acceleration']
    entry_volume = df_full.loc[entry_idx, 'volume_ratio']
    entry_dxy = df_full.loc[entry_idx, 'dxy_fuel']
    price_advantage = (best_entry_price - signal_price) / signal_price * 100 if direction == 'short' else (signal_price - best_entry_price) / signal_price * 100

    # 4. 从开仓点找最优平仓点（未来30周期内）
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

    # 5. 黄金平仓点的特征
    exit_idx = entry_idx + best_exit_period
    exit_tension = df_full.loc[exit_idx, 'tension']
    exit_accel = df_full.loc[exit_idx, 'acceleration']
    exit_volume = df_full.loc[exit_idx, 'volume_ratio']
    exit_dxy = df_full.loc[exit_idx, 'dxy_fuel']

    all_signal_data.append({
        '信号类型': signal_type,
        '方向': direction,
        '最优平仓盈亏%': best_pnl,

        # 信号前状态
        '信号前张力': pre_tension,
        '信号前加速度': pre_accel,
        '信号前量能比率': pre_volume,
        '信号前DXY燃料': pre_dxy,

        # 信号时特征
        '信号时张力': signal_tension,
        '信号时加速度': signal_accel,
        '信号时量能比率': signal_volume_ratio,
        '信号时DXY燃料': signal_dxy_fuel,

        # 开仓特征
        '最优开仓周期': best_entry_period,
        '开仓时张力': entry_tension,
        '开仓时加速度': entry_accel,
        '开仓时量能比率': entry_volume,
        '开仓时DXY燃料': entry_dxy,
        '价格优势%': price_advantage,

        # 平仓特征
        '最优平仓周期': best_exit_period,
        '平仓时张力': exit_tension,
        '平仓时加速度': exit_accel,
        '平仓时量能比率': exit_volume,
        '平仓时DXY燃料': exit_dxy,

        # 方向改变
        '方向改变': direction_changed,
    })

df_complete = pd.DataFrame(all_signal_data)
print(f"[OK] 提取完成: {len(df_complete)}个信号")

# ==================== 系统性分析函数 ====================
def analyze_complete_evolution(df, direction_name, signal_types):
    """分析信号的完整演变过程"""
    print(f"\n{'=' * 80}")
    print(f"{direction_name} - {signal_types} 完整演变分析")
    print(f"{'=' * 80}")

    df_dir = df[df['信号类型'].isin(signal_types)].copy()
    print(f"\n信号总数: {len(df_dir)}个")

    # 找好机会和坏机会
    pnl_median = df_dir['最优平仓盈亏%'].median()
    df_dir['是好机会'] = df_dir['最优平仓盈亏%'] >= pnl_median

    good = df_dir[df_dir['是好机会'] == True]
    bad = df_dir[df_dir['是好机会'] == False]

    print(f"好机会: {len(good)}个, 平均盈亏 {good['最优平仓盈亏%'].mean():.2f}%")
    print(f"坏机会: {len(bad)}个, 平均盈亏 {bad['最优平仓盈亏%'].mean():.2f}%")

    # 1. 信号出现前的市场状态
    print(f"\n{'=' * 60}")
    print("【1. 信号出现前的市场状态（前5周期）")
    print(f"{'=' * 60}")

    print(f"信号前5周期平均:")
    for col in ['信号前张力', '信号前加速度', '信号前量能比率', '信号前DXY燃料']:
        good_mean = good[col].mean()
        bad_mean = bad[col].mean()
        print(f"  {col} - 好机会: {good_mean:.4f}, 坏机会: {bad_mean:.4f}")

        # 统计检验
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')
        print(f"    t检验: t={t_stat:.3f}, p={p_val:.4f}")

    # 2. 信号出现时的市场特征
    print(f"\n{'=' * 60}")
    print("【2. 信号出现时的市场特征")
    print(f"{'=' * 60}")

    print(f"信号出现时:")
    for col in ['信号时张力', '信号时加速度', '信号时量能比率', '信号时DXY燃料']:
        good_mean = good[col].mean()
        bad_mean = bad[col].mean()
        print(f"  {col} - 好机会: {good_mean:.4f}, 坏机会: {bad_mean:.4f}")

        # 统计检验
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')
        print(f"    t检验: t={t_stat:.3f}, p={p_val:.4f}")

    # 3. 从信号到黄金开仓的演变
    print(f"\n{'=' * 60}")
    print("【3. 从信号到黄金开仓的演变过程】")
    print(f"{'=' * 60}")

    print(f"等待最优开仓的周期分布:")
    for wait in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        count = len(df_dir[df_dir['最优开仓周期'] == wait])
        if count > 0:
            good_count = len(df_dir[(df_dir['最优开仓周期'] == wait) & (df_dir['是好机会'] == True)])
            print(f"  等待{wait}周期: {count}个, 好机会率 {good_count/count*100:.1f}%")

    # 4. 黄金开仓点的数值共性
    print(f"\n{'=' * 60}")
    print("【4. 黄金开仓点的数值共性（所有好机会）】")
    print(f"{'=' * 60}")

    print(f"好机会的开仓特征:")
    for col in ['开仓时张力', '开仓时加速度', '开仓时量能比率', '开仓时DXY燃料']:
        good_mean = good[col].mean()
        good_std = good[col].std()
        bad_mean = bad[col].mean()
        print(f"  {col} - 好机会: {good_mean:.4f} +/- {good_std:.4f}, 坏机会: {bad_mean:.4f}")

        # 统计检验
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')
        n1, n2 = len(good), len(bad)
        var1, var2 = good[col].var(), bad[col].var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0
        print(f"    Cohen's d: {cohens_d:.3f}, p: {p_val:.4f}")

    # 5. 从开仓到平仓的周期分布
    print(f"\n{'=' * 60}")
    print("【5. 从开仓到平仓的周期分布】")
    print(f"{'=' * 60}")

    print(f"平仓周期分布:")
    for period in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]:
        count = len(df_dir[df_dir['最优平仓周期'] == period])
        if count > 0:
            good_count = len(df_dir[(df_dir['最优平仓周期'] == period) & (df_dir['是好机会'] == True)])
            print(f"  持仓{period}周期: {count}个, 好机会率 {good_count/count*100:.1f}%")

    # 6. 黄金平仓点的数值共性
    print(f"\n{'=' * 60}")
    print("【6. 黄金平仓点的数值共性（所有好机会）】")
    print(f"{'=' * 60}")

    print(f"好机会的平仓特征:")
    for col in ['平仓时张力', '平仓时加速度', '平仓时量能比率', '平仓时DXY燃料']:
        good_mean = good[col].mean()
        good_std = good[col].std()
        bad_mean = bad[col].mean()
        print(f"  {col} - 好机会: {good_mean:.4f} +/- {good_std:.4f}, 坏机会: {bad_mean:.4f}")

        # 统计检验
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')
        n1, n2 = len(good), len(bad)
        var1, var2 = good[col].var(), bad[col].var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0
        print(f"    Cohen's d: {cohens_d:.3f}, p: {p_val:.4f}")

    # 7. Youden指数找最优阈值
    print(f"\n{'=' * 60}")
    print("【7. Youden指数最优阈值】")
    print(f"{'=' * 60}")

    key_features = ['开仓时加速度', '开仓时量能比率', '开仓时DXY燃料',
                    '平仓时张力', '平仓时加速度', '平仓时量能比率', '平仓时DXY燃料']

    for feature in key_features:
        try:
            fpr, tpr, thresholds = roc_curve(df_dir['是好机会'], df_dir[feature])
            youden = tpr - fpr
            optimal_idx = np.argmax(youden)
            optimal_threshold = thresholds[optimal_idx]
            optimal_youden = youden[optimal_idx]

            if optimal_youden > 0.2:  # 只显示有判别能力的
                print(f"\n{feature}:")
                print(f"  最优阈值: {optimal_threshold:.4f}")
                print(f"  Youden指数: {optimal_youden:.4f}")

                above = df_dir[df_dir[feature] >= optimal_threshold]
                below = df_dir[df_dir[feature] < optimal_threshold]
                if len(above) > 0 and len(below) > 0:
                    print(f"  阈值以上好机会率: {len(above[above['是好机会']==True])/len(above)*100:.1f}%")
                    print(f"  阈值以下好机会率: {len(below[below['是好机会']==True])/len(below)*100:.1f}%")
        except:
            pass

# ==================== 分析SHORT和LONG ====================
print("\n" + "=" * 80)
print("开始系统性分析...")
print("=" * 80)

# 分析SHORT震荡信号
df_short = df_complete[df_complete['方向'] == 'short'].copy()
analyze_complete_evolution(df_short, "SHORT信号", ['HIGH_OSCILLATION'])

# 分析LONG震荡信号
df_long = df_complete[df_complete['方向'] == 'long'].copy()
analyze_complete_evolution(df_long, "LONG信号", ['LOW_OSCILLATION'])

print("\n" + "=" * 80)
print("[OK] 系统性分析完成")
print("=" * 80)

# 保存完整数据
df_complete.to_csv('complete_evolutionary_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: complete_evolutionary_analysis.csv")
