# -*- coding: utf-8 -*-
"""
黄金信号可视化 - 多空不对称策略
基于深度数据挖掘的物理规律
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

print("="*100)
print("黄金信号系统 - 多空不对称策略")
print("="*100)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条4小时K线")

# 识别极值点
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

df['高低点'] = ''
for i in local_max_indices:
    df.loc[i, '高低点'] = '高点'
for i in local_min_indices:
    df.loc[i, '高低点'] = '低点'

# 定义信号模式
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# ==============================================================================
# 黄金信号检测（基于多空不对称物理规律）
# ==============================================================================

print("\n" + "="*100)
print("黄金信号检测标准")
print("="*100)

print("""
【多头模型：接住坠落的飞刀】
- 加速度: ≤ -0.20（极高负值，暴跌）
- 张力: ≥ 0.80（极高正向，极限拉伸）
- 价格vsEMA: ≤ -1.5%（深度跌破）
- 量能: > 1.0（放量，恐慌盘）

【空头模型：推倒力竭的墙】
- 加速度: ≥ 0.03 且 ≤ 0.15（微弱正值，滞涨）
- 张力: ≤ -0.40（中等负向，压制）
- 量能: ≤ 0.70（显著萎缩，无人接盘）
""")

# 检测黄金做多信号
df['黄金做多'] = False
golden_long_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.20) &
    (df['张力'] >= 0.80) &
    (df['价格vsEMA%'] <= -1.5) &
    (df['量能比率'] > 1.0) &
    (df['高低点'] == '低点')
)
df.loc[golden_long_conditions, '黄金做多'] = True

# 检测黄金做空信号
df['黄金做空'] = False
golden_short_conditions = (
    (df['信号模式'] == 'SHORT_MODE') &
    (df['加速度'] >= 0.03) &
    (df['加速度'] <= 0.15) &
    (df['张力'] <= -0.40) &
    (df['量能比率'] <= 0.70) &
    (df['高低点'] == '高点')
)
df.loc[golden_short_conditions, '黄金做空'] = True

# 统计结果
golden_long_count = df['黄金做多'].sum()
golden_short_count = df['黄金做空'].sum()

print(f"\n检测结果:")
print(f"  黄金做多信号: {golden_long_count} 个")
print(f"  黄金做空信号: {golden_short_count} 个")
print(f"  总计: {golden_long_count + golden_short_count} 个")

# ==============================================================================
# 详细展示黄金信号
# ==============================================================================

if golden_long_count > 0:
    print("\n" + "="*100)
    print("【黄金做多信号详情】接住坠落的飞刀")
    print("="*100)

    long_signals = df[df['黄金做多']].copy()

    print(f"\n{'时间':<20} {'价格':<12} {'张力':<10} {'加速度':<12} {'乖离率':<12} {'量能':<10}")
    print("-" * 100)

    for idx, row in long_signals.iterrows():
        print(f"{str(row['时间'])[:18]:<20} "
              f"${row['收盘价']:>10.2f} "
              f"{row['张力']:>8.3f} "
              f"{row['加速度']:>10.4f} "
              f"{row['价格vsEMA%']:>10.2f}% "
              f"{row['量能比率']:>8.2f}")

if golden_short_count > 0:
    print("\n" + "="*100)
    print("【黄金做空信号详情】推倒力竭的墙")
    print("="*100)

    short_signals = df[df['黄金做空']].copy()

    print(f"\n{'时间':<20} {'价格':<12} {'张力':<10} {'加速度':<12} {'乖离率':<12} {'量能':<10}")
    print("-" * 100)

    for idx, row in short_signals.iterrows():
        print(f"{str(row['时间'])[:18]:<20} "
              f"${row['收盘价']:>10.2f} "
              f"{row['张力']:>8.3f} "
              f"{row['加速度']:>10.4f} "
              f"{row['价格vsEMA%']:>10.2f}% "
              f"{row['量能比率']:>8.2f}")

# ==============================================================================
# 对比：普通信号 vs 黄金信号
# ==============================================================================

print("\n" + "="*100)
print("信号质量对比")
print("="*100)

# 普通做多信号（BEARISH模式）
normal_long = df[df['信号模式'] == 'LONG_MODE']
print(f"\n普通做多信号（BEARISH模式）: {len(normal_long)} 个")

if len(normal_long) > 0:
    print(f"  平均张力: {normal_long['张力'].mean():.3f}")
    print(f"  平均加速度: {normal_long['加速度'].mean():.4f}")
    print(f"  平均乖离率: {normal_long['价格vsEMA%'].mean():.2f}%")
    print(f"  平均量能: {normal_long['量能比率'].mean():.2f}")

print(f"\n黄金做多信号（严格筛选）: {golden_long_count} 个")

if golden_long_count > 0:
    print(f"  平均张力: {long_signals['张力'].mean():.3f}")
    print(f"  平均加速度: {long_signals['加速度'].mean():.4f}")
    print(f"  平均乖离率: {long_signals['价格vsEMA%'].mean():.2f}%")
    print(f"  平均量能: {long_signals['量能比率'].mean():.2f}")

# 普通做空信号（BULLISH模式）
normal_short = df[df['信号模式'] == 'SHORT_MODE']
print(f"\n普通做空信号（BULLISH模式）: {len(normal_short)} 个")

if len(normal_short) > 0:
    print(f"  平均张力: {normal_short['张力'].mean():.3f}")
    print(f"  平均加速度: {normal_short['加速度'].mean():.4f}")
    print(f"  平均乖离率: {normal_short['价格vsEMA%'].mean():.2f}%")
    print(f"  平均量能: {normal_short['量能比率'].mean():.2f}")

print(f"\n黄金做空信号（严格筛选）: {golden_short_count} 个")

if golden_short_count > 0:
    print(f"  平均张力: {short_signals['张力'].mean():.3f}")
    print(f"  平均加速度: {short_signals['加速度'].mean():.4f}")
    print(f"  平均乖离率: {short_signals['价格vsEMA%'].mean():.2f}%")
    print(f"  平均量能: {short_signals['量能比率'].mean():.2f}")

# ==============================================================================
# 过滤率分析
# ==============================================================================

print("\n" + "="*100)
print("过滤效果分析")
print("="*100)

long_filter_rate = (1 - golden_long_count / len(normal_long)) * 100 if len(normal_long) > 0 else 0
short_filter_rate = (1 - golden_short_count / len(normal_short)) * 100 if len(normal_short) > 0 else 0

print(f"\n做多信号过滤: {len(normal_long)} → {golden_long_count} (过滤掉 {long_filter_rate:.1f}%)")
print(f"做空信号过滤: {len(normal_short)} → {golden_short_count} (过滤掉 {short_filter_rate:.1f}%)")

print(f"\n总信号: {len(df)}")
print(f"  普通交易信号: {len(normal_long) + len(normal_short)} ({(len(normal_long) + len(normal_short))/len(df)*100:.1f}%)")
print(f"  黄金交易信号: {golden_long_count + golden_short_count} ({(golden_long_count + golden_short_count)/len(df)*100:.1f}%)")

print("\n" + "="*100)
print("分析完成")
print("="*100)

# ==============================================================================
# 可视化（可选）
# ==============================================================================

try:
    import matplotlib.dates as mdates

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # 子图1: 价格 + 黄金信号
    ax1.plot(df['时间'], df['收盘价'], 'k-', linewidth=1, alpha=0.6, label='收盘价')

    # 标记黄金做多信号
    if golden_long_count > 0:
        long_signals = df[df['黄金做多']]
        ax1.scatter(long_signals['时间'], long_signals['收盘价'],
                   color='red', s=200, marker='^', zorder=5,
                   label=f'黄金做多 ({golden_long_count})', edgecolors='darkred', linewidths=2)

    # 标记黄金做空信号
    if golden_short_count > 0:
        short_signals = df[df['黄金做空']]
        ax1.scatter(short_signals['时间'], short_signals['收盘价'],
                   color='blue', s=200, marker='v', zorder=5,
                   label=f'黄金做空 ({golden_short_count})', edgecolors='darkblue', linewidths=2)

    ax1.set_ylabel('价格 (USD)', fontsize=12)
    ax1.set_title('BTC 4H - 黄金信号（多空不对称策略）', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2: 张力 + 加速度
    ax2.plot(df['时间'], df['张力'], 'r-', linewidth=1.5, alpha=0.7, label='张力')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # 双Y轴显示加速度
    ax2_accel = ax2.twinx()
    ax2_accel.plot(df['时间'], df['加速度'], 'b-', linewidth=1.5, alpha=0.5, label='加速度')
    ax2_accel.axhline(y=0, color='b', linestyle='--', linewidth=0.5)

    ax2.set_ylabel('张力', fontsize=12, color='red')
    ax2_accel.set_ylabel('加速度', fontsize=12, color='blue')

    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_accel.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax2.grid(True, alpha=0.3)

    # 格式化X轴日期
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('黄金信号可视化_多空不对称策略.png', dpi=150, bbox_inches='tight')
    print("\n可视化图表已保存: 黄金信号可视化_多空不对称策略.png")

except Exception as e:
    print(f"\n可视化生成失败: {e}")
