# -*- coding: utf-8 -*-
"""
系统性黄金信号分析 - 统计学数学家方法
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
print("系统性黄金信号分析 - 统计学数学家方法")
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

# ==================== 定义分析函数 ====================
def analyze_signal_evolution(df, df_full, direction_name, signal_types):
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

    # 1. 分析信号出现前的市场状态（前5个周期）
    print(f"\n{'=' * 60}")
    print("【1. 信号出现前的市场状态（前5周期）")
    print(f"{'=' * 60}")

    pre_signal_tension = []
    pre_signal_accel = []
    pre_signal_volume = []
    is_good_opportunity = []

    for idx, signal in df_dir.iterrows():
        signal_time = pd.to_datetime(signal['时间'])
        signal_idx_list = df_full[df_full['timestamp'] == signal_time].index

        if len(signal_idx_list) == 0 or signal_idx_list[0] < 5:
            continue

        signal_idx = signal_idx_list[0]

        # 信号前5个周期的平均特征
        pre_tension = df_full.loc[signal_idx-5:signal_idx, 'tension'].mean()
        pre_accel = df_full.loc[signal_idx-5:signal_idx, 'acceleration'].mean()
        pre_volume = df_full.loc[signal_idx-5:signal_idx, 'volume_ratio'].mean()

        pre_signal_tension.append(pre_tension)
        pre_signal_accel.append(pre_accel)
        pre_signal_volume.append(pre_volume)
        is_good_opportunity.append(signal['是好机会'])

    df_pre = pd.DataFrame({
        '好机会': is_good_opportunity,
        '张力': pre_signal_tension,
        '加速度': pre_signal_accel,
        '量能比率': pre_signal_volume
    })

    good_pre = df_pre[df_pre['好机会'] == True]
    bad_pre = df_pre[df_pre['好机会'] == False]

    print(f"信号前5周期平均:")
    print(f"  张力 - 好机会: {good_pre['张力'].mean():.4f}, 坏机会: {bad_pre['张力'].mean():.4f}")
    print(f"  加速度 - 好机会: {good_pre['加速度'].mean():.6f}, 坏机会: {bad_pre['加速度'].mean():.6f}")
    print(f"  量能 - 好机会: {good_pre['量能比率'].mean():.4f}, 坏机会: {bad_pre['量能比率'].mean():.4f}")

    # 统计检验
    for col in ['张力', '加速度', '量能比率']:
        t_stat, p_val = stats.ttest_ind(good_pre[col], bad_pre[col], nan_policy='omit')
        print(f"  {col} t检验: t={t_stat:.3f}, p={p_val:.4f}")

    # 2. 从信号到最优开仓点的演变
    print(f"\n{'=' * 60}")
    print("【2. 从信号到最优开仓的演变过程】")
    print(f"{'=' * 60}")

    # 等待周期的分布
    print(f"等待最优开仓的周期分布:")
    for wait in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        count = len(df_dir[df_dir['最优开仓周期'] == wait])
        if count > 0:
            good_count = len(df_dir[(df_dir['最优开仓周期'] == wait) & (df_dir['是好机会'] == True)])
            print(f"  等待{wait}周期: {count}个, 好机会率 {good_count/count*100:.1f}%")

    # 3. 黄金开仓点的数值特征
    print(f"\n{'=' * 60}")
    print("【3. 黄金开仓点的数值特征】")
    print(f"{'=' * 60}")

    # 分析不同等待周期下的价格优势
    for wait in [0, 1, 2, 3, 4, 5]:
        df_wait = df_dir[df_dir['最优开仓周期'] == wait]
        if len(df_wait) > 0:
            good_wait = df_wait[df_wait['是好机会'] == True]
            bad_wait = df_wait[df_wait['是好机会'] == False]
            print(f"\n等待{wait}周期 (n={len(df_wait)}):")
            print(f"  价格优势 - 好机会: {good_wait['价格优势%'].mean():.3f}%, 坏机会: {bad_wait['价格优势%'].mean():.3f}%")
            print(f"  平均盈亏 - 好机会: {good_wait['最优平仓盈亏%'].mean():.2f}%, 坏机会: {bad_wait['最优平仓盈亏%'].mean():.2f}%")

    # 4. 黄金开仓点的共性（所有好机会）
    print(f"\n{'=' * 60}")
    print("【4. 黄金开仓点的数值共性（所有好机会）】")
    print(f"{'=' * 60}")

    print(f"好机会的开仓特征:")
    print(f"  张力 - 均值: {good['开仓时张力'].mean():.4f}, 标准差: {good['开仓时张力'].std():.4f}")
    print(f"  加速度 - 均值: {good['开仓时加速度'].mean():.6f}, 标准差: {good['开仓时加速度'].std():.6f}")
    print(f"  量能比率 - 均值: {good['开仓时量能比率'].mean():.4f}, 标准差: {good['开仓时量能比率'].std():.4f}")
    print(f"  DXY燃料 - 均值: {good['开仓时DXY燃料'].mean():.4f}, 标准差: {good['开仓时DXY燃料'].std():.4f}")

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
    print(f"  张力 - 均值: {good['平仓时张力'].mean():.4f}, 标准差: {good['平仓时张力'].std():.4f}")
    print(f"  加速度 - 均值: {good['平仓时加速度'].mean():.6f}, 标准差: {good['平仓时加速度'].std():.6f}")
    print(f"  量能比率 - 均值: {good['平仓时量能比率'].mean():.4f}, 标准差: {good['平仓时量能比率'].std():.4f}")
    print(f"  DXY燃料 - 均值: {good['平仓时DXY燃料'].mean():.4f}, 标准差: {good['平仓时DXY燃料'].std():.4f}")

    # 统计检验
    print(f"\n平仓时特征的统计检验:")
    for col in ['平仓时张力', '平仓时加速度', '平仓时量能比率', '平仓时DXY燃料']:
        t_stat, p_val = stats.ttest_ind(good[col], bad[col], nan_policy='omit')

        # Cohen's d
        n1, n2 = len(good), len(bad)
        var1, var2 = good[col].var(), bad[col].var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good[col].mean() - bad[col].mean()) / pooled_std if pooled_std > 0 else 0

        effect = "***" if abs(cohens_d) > 1.2 else "**" if abs(cohens_d) > 0.8 else "*" if abs(cohens_d) > 0.5 else ""

        print(f"  {col}: d={cohens_d:.3f}, p={p_val:.4f} {effect}")

    # 7. 找最优阈值（Youden指数）
    print(f"\n{'=' * 60}")
    print("【7. Youden指数最优阈值】")
    print(f"{'=' * 60}")

    # 平仓时量能比率
    try:
        fpr, tpr, thresholds = roc_curve(df_dir['是好机会'], df_dir['平仓时量能比率'])
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        optimal_threshold = thresholds[optimal_idx]
        optimal_youden = youden[optimal_idx]

        print(f"平仓时量能比率:")
        print(f"  最优阈值: {optimal_threshold:.4f}")
        print(f"  Youden指数: {optimal_youden:.4f}")

        above = df_dir[df_dir['平仓时量能比率'] >= optimal_threshold]
        below = df_dir[df_dir['平仓时量能比率'] < optimal_threshold]
        print(f"  阈值以上好机会率: {len(above[above['是好机会']==True])/len(above)*100:.1f}%")
        print(f"  阈值以下好机会率: {len(below[below['是好机会']==True])/len(below)*100:.1f}%")
    except:
        pass

# ==================== 分析SHORT和LONG ====================
print("\n" + "=" * 80)
print("开始系统性分析...")
print("=" * 80)

# 需要先读取step2_3的数据获取最优开仓/平仓信息
df_analysis = pd.read_csv('step2_3_correct_golden_analysis.csv', encoding='utf-8-sig')
df_analysis['信号时间'] = pd.to_datetime(df_analysis['信号时间'])

# 合并数据（step1用'时间'，step2_3用'信号时间'）
df_signals['时间'] = pd.to_datetime(df_signals['时间'])
df_merged = df_signals.merge(df_analysis, left_on='时间', right_on='信号时间', how='inner', suffixes=('', '_y'))

print(f"合并后数据: {len(df_merged)}个")

# 分析SHORT震荡信号
df_short = df_merged[df_merged['方向'] == 'short'].copy()
analyze_signal_evolution(df_short, df_full, "SHORT信号", ['HIGH_OSCILLATION'])

# 分析LONG震荡信号
df_long = df_merged[df_merged['方向'] == 'long'].copy()
analyze_signal_evolution(df_long, df_full, "LONG信号", ['LOW_OSCILLATION'])

print("\n" + "=" * 80)
print("[OK] 系统性分析完成")
print("=" * 80)
