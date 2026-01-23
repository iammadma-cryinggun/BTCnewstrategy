# -*- coding: utf-8 -*-
"""
完整的黄金信号参数分析 - 找最优解
=====================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("完整黄金信号参数分析 - 找最优解")
print("=" * 80)

# ==================== 读取数据 ====================
df_signals = pd.read_csv('step1_entry_signals_with_dxy.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']
df_full['tension_accel_ratio'] = np.abs(df_full['tension'] / df_full['acceleration'].replace(0, np.nan))

print(f"\n处理全部信号: {len(df_signals)}个")

# ==================== 提取所有信号的特征 ====================
print("\n正在提取特征...")

all_features = []

for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_price = signal['收盘价']
    signal_type = signal['信号类型']
    signal_tension = signal['张力']
    signal_accel = signal['加速度']

    if signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        direction = 'short'
    else:
        direction = 'long'

    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx_list) == 0:
        continue
    signal_idx = signal_idx_list[0]

    # 找最优平仓点
    best_pnl = -999
    best_exit_period = 0
    for period in range(1, 31):
        future_idx = signal_idx + period
        if future_idx >= len(df_full):
            break
        future_price = df_full.loc[future_idx, 'close']
        if direction == 'short':
            pnl = (signal_price - future_price) / signal_price * 100
        else:
            pnl = (future_price - signal_price) / signal_price * 100
        if pnl > best_pnl:
            best_pnl = pnl
            best_exit_period = period

    exit_idx = signal_idx + best_exit_period

    all_features.append({
        '信号类型': signal_type,
        '方向': direction,
        '盈亏%': best_pnl,
        # 开仓时特征
        '开仓时张力': signal_tension,
        '开仓时加速度': signal_accel,
        '开仓时量能比率': signal['量能比率'],
        '开仓时DXY燃料': signal.get('DXY燃料', 0),
        '开仓时张力加速度比': np.abs(signal_tension / signal_accel) if signal_accel != 0 else 0,
        # 平仓时特征
        '平仓时张力': df_full.loc[exit_idx, 'tension'],
        '平仓时加速度': df_full.loc[exit_idx, 'acceleration'],
        '平仓时量能比率': df_full.loc[exit_idx, 'volume_ratio'],
        '平仓时DXY燃料': df_full.loc[exit_idx, 'dxy_fuel'],
        '平仓时张力加速度比': df_full.loc[exit_idx, 'tension_accel_ratio'],
    })

df_all = pd.DataFrame(all_features)
print(f"提取完成: {len(df_all)}个信号")

# ==================== 分析函数 ====================
def analyze_feature_optimal(df, feature_name, direction_name):
    """分析某个特征的最优阈值"""
    print(f"\n{'=' * 80}")
    print(f"{direction_name} - {feature_name} 最优阈值分析")
    print(f"{'=' * 80}")

    # 盈亏中位数作为分界
    pnl_median = df['盈亏%'].median()
    df['是好机会'] = df['盈亏%'] >= pnl_median

    good = df[df['是好机会'] == True]
    bad = df[df['是好机会'] == False]

    # 数值分布
    print(f"\n【数值分布】")
    print(f"最小值: {df[feature_name].min():.4f}")
    print(f"25%分位: {df[feature_name].quantile(0.25):.4f}")
    print(f"50%分位: {df[feature_name].quantile(0.50):.4f}")
    print(f"75%分位: {df[feature_name].quantile(0.75):.4f}")
    print(f"最大值: {df[feature_name].max():.4f}")

    # 好机会 vs 坏机会的分布
    print(f"\n【好机会 vs 坏机会】")
    print(f"好机会 (n={len(good)}): {good[feature_name].mean():.4f} ± {good[feature_name].std():.4f}")
    print(f"坏机会 (n={len(bad)}): {bad[feature_name].mean():.4f} ± {bad[feature_name].std():.4f}")

    # t检验
    t_stat, p_val = stats.ttest_ind(good[feature_name], bad[feature_name], equal_var=False, nan_policy='omit')
    n1, n2 = len(good), len(bad)
    var1, var2 = good[feature_name].var(), bad[feature_name].var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (good[feature_name].mean() - bad[feature_name].mean()) / pooled_std if pooled_std > 0 else 0

    print(f"\n【统计检验】")
    print(f"t统计量: {t_stat:.4f}, p值: {p_val:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")

    if abs(cohens_d) > 1.2:
        effect = "超大效应"
    elif abs(cohens_d) > 0.8:
        effect = "大效应"
    elif abs(cohens_d) > 0.5:
        effect = "中等效应"
    elif abs(cohens_d) > 0.2:
        effect = "小效应"
    else:
        effect = "微小效应"
    print(f"效应量: {effect}")

    # Youden指数找最优阈值
    try:
        fpr, tpr, thresholds = roc_curve(df['是好机会'], df[feature_name])
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        optimal_youden = youden_index[optimal_idx]

        print(f"\n【Youden指数最优阈值】")
        print(f"最优阈值: {optimal_threshold:.4f}")
        print(f"Youden指数: {optimal_youden:.4f}")

        if optimal_youden > 0.8:
            discrim = "优秀"
        elif optimal_youden > 0.5:
            discrim = "良好"
        elif optimal_youden > 0.2:
            discrim = "一般"
        else:
            discrim = "差"
        print(f"判别能力: {discrim}")

        # 在这个阈值下的表现
        above_threshold = df[df[feature_name] >= optimal_threshold]
        below_threshold = df[df[feature_name] < optimal_threshold]

        print(f"\n阈值以上 (n={len(above_threshold)}): 平均盈亏 {above_threshold['盈亏%'].mean():.2f}%")
        print(f"阈值以下 (n={len(below_threshold)}): 平均盈亏 {below_threshold['盈亏%'].mean():.2f}%")

    except Exception as e:
        print(f"\n无法计算Youden指数: {e}")

    return optimal_threshold if 'optimal_threshold' in locals() else None

# ==================== 按信号类型和方向分组分析 ====================
print("\n" + "=" * 80)
print("按信号类型分组分析")
print("=" * 80)

# SHORT信号
df_short = df_all[df_all['方向'] == 'short'].copy()
print(f"\nSHORT信号总数: {len(df_short)}个")

# 按信号类型细分
for sig_type in ['HIGH_OSCILLATION', 'BULLISH_SINGULARITY']:
    df_sig = df_short[df_short['信号类型'] == sig_type]
    if len(df_sig) > 0:
        print(f"\n  {sig_type}: {len(df_sig)}个")

        # 分析开仓时DXY燃料
        if '开仓时DXY燃料' in df_sig.columns:
            analyze_feature_optimal(df_sig, '开仓时DXY燃料', f"{sig_type} (SHORT)")

        # 分析平仓时量能比率
        if '平仓时量能比率' in df_sig.columns:
            analyze_feature_optimal(df_sig, '平仓时量能比率', f"{sig_type} (SHORT)")

# LONG信号
df_long = df_all[df_all['方向'] == 'long'].copy()
print(f"\nLONG信号总数: {len(df_long)}个")

# 按信号类型细分
for sig_type in ['LOW_OSCILLATION', 'BEARISH_SINGULARITY']:
    df_sig = df_long[df_long['信号类型'] == sig_type]
    if len(df_sig) > 0:
        print(f"\n  {sig_type}: {len(df_sig)}个")

        # 分析开仓时加速度
        if '开仓时加速度' in df_sig.columns:
            analyze_feature_optimal(df_sig, '开仓时加速度', f"{sig_type} (LONG)")

        # 分析平仓时量能比率
        if '平仓时量能比率' in df_sig.columns:
            analyze_feature_optimal(df_sig, '平仓时量能比率', f"{sig_type} (LONG)")

print("\n" + "=" * 80)
print("[OK] 分析完成")
print("=" * 80)

# 保存完整数据
df_all.to_csv('complete_features_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: complete_features_analysis.csv")
