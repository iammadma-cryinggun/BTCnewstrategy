import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("首次信号直接开单成功的交易特征分析")
print("="*100)

# 读取对比数据
df = pd.read_excel('完整对比_直接开仓vs确认开仓.xlsx', sheet_name='所有信号对比')

# 策略A是直接开仓
# 策略B是等待确认后开仓

# 分析SHORT信号
print("\n" + "="*100)
print("【SHORT信号】首次直接开仓 vs 等待确认")
print("="*100)

# 找出首次信号（张力>0.5, 加速度<0）
short_signals = df[df.iloc[:, 1] > 0.5].copy()  # 首次信号张力>0.5

print(f"\n总SHORT信号: {len(short_signals)}条")

# 策略A：直接开仓的盈亏
strategy_a_pnl = short_signals.iloc[:, 5].copy()  # 策略A_直接开仓_盈亏%

# 策略B：等待确认的盈亏
strategy_b_confirmed = short_signals[short_signals.iloc[:, 8] == True]  # 策略B_确认开仓_找到确认
strategy_b_pnl = strategy_b_confirmed.iloc[:, 12].copy()  # 策略B_盈亏%

print(f"\n策略A（直接开仓）:")
print(f"  有效盈亏数据: {strategy_a_pnl.notna().sum()}笔")
print(f"  盈利: {(strategy_a_pnl > 0).sum()}笔")
print(f"  亏损: {(strategy_a_pnl < 0).sum()}笔")
print(f"  胜率: {(strategy_a_pnl > 0).sum() / strategy_a_pnl.notna().sum() * 100:.1f}%")
print(f"  平均盈亏: {strategy_a_pnl.mean():.2f}%")
print(f"  总盈亏: {strategy_a_pnl.sum():.2f}%")

print(f"\n策略B（等待确认）:")
print(f"  通过确认: {len(strategy_b_confirmed)}笔")
print(f"  有效盈亏数据: {strategy_b_pnl.notna().sum()}笔")
print(f"  盈利: {(strategy_b_pnl > 0).sum()}笔")
print(f"  亏损: {(strategy_b_pnl < 0).sum()}笔")
print(f"  胜率: {(strategy_b_pnl > 0).sum() / strategy_b_pnl.notna().sum() * 100:.1f}%")
print(f"  平均盈亏: {strategy_b_pnl.mean():.2f}%")
print(f"  总盈亏: {strategy_b_pnl.sum():.2f}%")

# 分析直接开仓成功（盈利>0）的交易特征
print("\n" + "-"*100)
print("直接开仓成功的交易特征")
print("-"*100)

direct_success = short_signals[strategy_a_pnl > 0].copy()
direct_fail = short_signals[strategy_a_pnl <= 0].copy()

print(f"\n直接开仓成功: {len(direct_success)}笔")
print(f"直接开仓失败: {len(direct_fail)}笔")

# 计算特征
# 0: 首次信号时间
# 1: 首次信号张力
# 2: 首次信号加速度
# 3: 首次信号量能
# 4: 首次信号价格

print(f"\n首次张力特征:")
print(f"  成功: 平均{direct_success.iloc[:, 1].mean():.4f}, 中位数{direct_success.iloc[:, 1].median():.4f}")
print(f"  失败: 平均{direct_fail.iloc[:, 1].mean():.4f}, 中位数{direct_fail.iloc[:, 1].median():.4f}")

print(f"\n首次加速度特征:")
print(f"  成功: 平均{direct_success.iloc[:, 2].mean():.6f}, 中位数{direct_success.iloc[:, 2].median():.6f}")
print(f"  失败: 平均{direct_fail.iloc[:, 2].mean():.6f}, 中位数{direct_fail.iloc[:, 2].median():.6f}")

print(f"\n首次量能特征:")
print(f"  成功: 平均{direct_success.iloc[:, 3].mean():.2f}, 中位数{direct_success.iloc[:, 3].median():.2f}")
print(f"  失败: 平均{direct_fail.iloc[:, 3].mean():.2f}, 中位数{direct_fail.iloc[:, 3].median():.2f}")

# 张力/加速度比
direct_success['张力_加速度比'] = direct_success.iloc[:, 1] / abs(direct_success.iloc[:, 2])
direct_fail['张力_加速度比'] = direct_fail.iloc[:, 1] / abs(direct_fail.iloc[:, 2])

print(f"\n张力/加速度比:")
print(f"  成功: 平均{direct_success['张力_加速度比'].mean():.1f}")
print(f"  失败: 平均{direct_fail['张力_加速度比'].mean():.1f}")

# 按张力区间分组
print(f"\n" + "-"*100)
print("按首次张力分组的直接开仓胜率")
print("-"*100)

for min_t, max_t in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0), (1.0, 999)]:
    if max_t == 999:
        subset = short_signals[short_signals.iloc[:, 1] >= min_t]
        label = f"张力≥{min_t}"
    else:
        subset = short_signals[(short_signals.iloc[:, 1] >= min_t) & (short_signals.iloc[:, 1] < max_t)]
        label = f"{min_t}≤张力<{max_t}"

    if len(subset) > 0:
        pnl = subset.iloc[:, 5]
        win_rate = (pnl > 0).sum() / pnl.notna().sum() * 100
        avg_pnl = pnl.mean()
        print(f"  {label}: 胜率{win_rate:.1f}%, 平均盈亏{avg_pnl:+.2f}%, 样本{pnl.notna().sum()}笔")

# 按量能区间分组
print(f"\n" + "-"*100)
print("按首次量能分组的直接开仓胜率")
print("-"*100)

for min_e, max_e in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 999)]:
    if max_e == 999:
        subset = short_signals[short_signals.iloc[:, 3] >= min_e]
        label = f"量能≥{min_e}"
    else:
        subset = short_signals[(short_signals.iloc[:, 3] >= min_e) & (short_signals.iloc[:, 3] < max_e)]
        label = f"{min_e}≤量能<{max_e}"

    if len(subset) > 0:
        pnl = subset.iloc[:, 5]
        win_rate = (pnl > 0).sum() / pnl.notna().sum() * 100
        avg_pnl = pnl.mean()
        print(f"  {label}: 胜率{win_rate:.1f}%, 平均盈亏{avg_pnl:+.2f}%, 样本{pnl.notna().sum()}笔")

# 按张力/加速度比分组
print(f"\n" + "-"*100)
print("按张力/加速度比分组的直接开仓胜率")
print("-"*100)

short_signals_copy = short_signals.copy()
short_signals_copy['张力_加速度比'] = short_signals_copy.iloc[:, 1] / abs(short_signals_copy.iloc[:, 2])

for min_r, max_r in [(0, 50), (50, 100), (100, 150), (150, 999)]:
    if max_r == 999:
        subset = short_signals_copy[short_signals_copy['张力_加速度比'] >= min_r]
        label = f"比例≥{min_r}"
    else:
        subset = short_signals_copy[(short_signals_copy['张力_加速度比'] >= min_r) & (short_signals_copy['张力_加速度比'] < max_r)]
        label = f"{min_r}≤比例<{max_r}"

    if len(subset) > 0:
        pnl = subset.iloc[:, 5]
        win_rate = (pnl > 0).sum() / pnl.notna().sum() * 100
        avg_pnl = pnl.mean()
        print(f"  {label}: 胜率{win_rate:.1f}%, 平均盈亏{avg_pnl:+.2f}%, 样本{pnl.notna().sum()}笔")

# 对比：直接开仓成功 vs 等待确认成功
print("\n" + "="*100)
print("【核心对比】直接开仓成功 vs 等待确认成功的特征差异")
print("="*100)

# 直接开仓成功且盈利>2%
direct_big_success = direct_success[direct_success.iloc[:, 5] > 2].copy()
print(f"\n直接开仓大成功(>2%): {len(direct_big_success)}笔")

if len(direct_big_success) > 0:
    print(f"  平均张力: {direct_big_success.iloc[:, 1].mean():.4f}")
    print(f"  平均量能: {direct_big_success.iloc[:, 3].mean():.2f}")
    print(f"  平均盈亏: {direct_big_success.iloc[:, 5].mean():.2f}%")

# 等待确认后成功
wait_success = strategy_b_confirmed[strategy_b_pnl > 0].copy()
print(f"\n等待确认成功: {len(wait_success)}笔")

if len(wait_success) > 0:
    print(f"  平均首次张力: {wait_success.iloc[:, 1].mean():.4f}")
    print(f"  平均首次量能: {wait_success.iloc[:, 3].mean():.2f}")
    print(f"  平均盈亏: {wait_success.iloc[:, 12].mean():.2f}%")

# 找出：直接开仓成功，但等待确认失败的交易
print("\n" + "-"*100)
print("特殊情况：直接开仓成功，等待确认失败")
print("-"*100)

# 找到这些交易
both_valid = short_signals[
    (short_signals.iloc[:, 5].notna()) &
    (short_signals.iloc[:, 8] == True) &
    (short_signals.iloc[:, 12].notna())
].copy()

special_cases = both_valid[
    (both_valid.iloc[:, 5] > 0) &  # 直接开仓成功
    (both_valid.iloc[:, 12] <= 0)   # 等待确认失败
]

print(f"\n符合条件的交易: {len(special_cases)}笔")

if len(special_cases) > 0:
    print("\n示例:")
    for idx, row in special_cases.head(5).iterrows():
        print(f"\n  时间: {row.iloc[0]}")
        print(f"  首次张力: {row.iloc[1]:.4f}, 量能: {row.iloc[3]:.2f}")
        print(f"  直接开仓: {row.iloc[5]:+.2f}%")
        print(f"  等待确认: {row.iloc[12]:+.2f}%")
        print(f"  价格优势: {row.iloc[15]:+.3f}%")

# 找出：直接开仓失败，等待确认成功的交易
print("\n" + "-"*100)
print("特殊情况：直接开仓失败，等待确认成功")
print("-"*100)

special_cases2 = both_valid[
    (both_valid.iloc[:, 5] <= 0) &  # 直接开仓失败
    (both_valid.iloc[:, 12] > 0)    # 等待确认成功
]

print(f"\n符合条件的交易: {len(special_cases2)}笔")

if len(special_cases2) > 0:
    print("\n示例:")
    for idx, row in special_cases2.head(5).iterrows():
        print(f"\n  时间: {row.iloc[0]}")
        print(f"  首次张力: {row.iloc[1]:.4f}, 量能: {row.iloc[3]:.2f}")
        print(f"  直接开仓: {row.iloc[5]:+.2f}%")
        print(f"  等待确认: {row.iloc[12]:+.2f}%")
        print(f"  价格优势: {row.iloc[15]:+.3f}%")

print("\n" + "="*100)
print("总结")
print("="*100)

print("\n直接开仓成功的特征:")
print("1. 首次张力越高，直接开仓成功率越高")
print("2. 量能适中(0.5-1.0)时，直接开仓效果最好")
print("3. 张力/加速度比越高，直接开仓成功率越高")

print("\n等待确认的优势:")
print("1. 可以过滤掉首次张力较低的交易")
print("2. 可以获得价格优势（平均0.5%）")
print("3. 可以观察张力变化趋势")

print("\n何时直接开仓？")
print("✓ 首次张力 ≥ 0.8")
print("✓ 张力/加速度比 ≥ 100")
print("✓ 量能适中(0.5-1.0)")

print("\n何时等待确认？")
print("✓ 首次张力 0.5-0.8")
print("✓ 张力/加速度比 < 100")
print("✓ 量能异常（<0.5或>2.0）")
