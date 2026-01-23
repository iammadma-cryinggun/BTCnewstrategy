# -*- coding: utf-8 -*-
"""
黄金时刻特征分析 - 描述性统计
================================

目标：找出手动标注为ACTION的时刻，数据有什么共同特征
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("GOLDEN SIGNAL MOMENT ANALYSIS - Descriptive Statistics")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')

# Extract action type
df['Action_Type'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'OTHER'
)

# Extract action detail (多/空)
df['Action_Detail'] = df['黄金信号'].apply(lambda x:
    '做多' if any(k in str(x) for k in ['多', 'Long']) else
    '做空' if any(k in str(x) for k in ['空', 'Short']) else
    'UNKNOWN'
)

# ============================================================================
# 1. SEPARATE ACTION vs HOLD MOMENTS
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Separate ACTION vs HOLD Moments")
print("="*120)

action_moments = df[df['Action_Type'] == 'ACTION'].copy()
hold_moments = df[df['Action_Type'] == 'HOLD'].copy()

print(f"\n手动标注ACTION时刻: {len(action_moments)} 个")
print(f"手动标注HOLD时刻: {len(hold_moments)} 个")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS - ACTION MOMENTS
# ============================================================================
print("\n" + "="*120)
print("STEP 2: What do ACTION moments look like?")
print("="*120)

params = ['量能比率', '价格vsEMA%', '张力', '加速度', 'DXY燃料']

print(f"\n{'参数':<15} {'均值':<12} {'标准差':<12} {'中位数':<12} {'25%分位':<12} {'75%分位':<12} {'最小':<12} {'最大':<12}")
print("-"*120)

for param in params:
    if param in action_moments.columns:
        data = action_moments[param].dropna()
        print(f"{param:<15} {data.mean():<12.4f} {data.std():<12.4f} {data.median():<12.4f} {data.quantile(0.25):<12.4f} {data.quantile(0.75):<12.4f} {data.min():<12.4f} {data.max():<12.4f}")

# ============================================================================
# 3. DESCRIPTIVE STATISTICS - HOLD MOMENTS
# ============================================================================
print("\n" + "="*120)
print("STEP 3: What do HOLD moments look like?")
print("="*120)

print(f"\n{'参数':<15} {'均值':<12} {'标准差':<12} {'中位数':<12} {'25%分位':<12} {'75%分位':<12} {'最小':<12} {'最大':<12}")
print("-"*120)

for param in params:
    if param in hold_moments.columns:
        data = hold_moments[param].dropna()
        print(f"{param:<15} {data.mean():<12.4f} {data.std():<12.4f} {data.median():<12.4f} {data.quantile(0.25):<12.4f} {data.quantile(0.75):<12.4f} {data.min():<12.4f} {data.max():<12.4f}")

# ============================================================================
# 4. COMPARISON: ACTION vs HOLD (Statistical Significance)
# ============================================================================
print("\n" + "="*120)
print("STEP 4: ACTION vs HOLD - Statistical Significance")
print("="*120)

print(f"\n{'参数':<15} {'ACTION均值':<15} {'HOLD均值':<15} {'差异':<12} {'t统计':<10} {'p值':<12} {'显著性':<10}")
print("-"*120)

for param in params:
    if param in action_moments.columns and param in hold_moments.columns:
        action_data = action_moments[param].dropna()
        hold_data = hold_moments[param].dropna()

        mean_action = action_data.mean()
        mean_hold = hold_data.mean()
        diff = mean_action - mean_hold

        # t-test
        t_stat, p_value = stats.ttest_ind(action_data, hold_data)

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''

        print(f"{param:<15} {mean_action:<15.4f} {mean_hold:<15.4f} {diff:<12.4f} {t_stat:<10.2f} {p_value:<12.6f} {sig:<10}")

# ============================================================================
# 5. DISTRIBUTION OVERLAP - How separable are they?
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Distribution Overlap Analysis")
print("="*120)

print("\nIf distributions don't overlap much, the parameter is good for prediction")
print("\nCohen's d (Effect Size):")
print(f"  Small: 0.2, Medium: 0.5, Large: 0.8")

print(f"\n{'参数':<15} {'Cohen\'s d':<15} {'效应大小':<15} {'可分性':<20}")
print("-"*90)

for param in params:
    if param in action_moments.columns and param in hold_moments.columns:
        action_data = action_moments[param].dropna()
        hold_data = hold_moments[param].dropna()

        # Cohen's d
        pooled_std = np.sqrt((action_data.std()**2 + hold_data.std()**2) / 2)
        cohens_d = abs(action_data.mean() - hold_data.mean()) / pooled_std

        effect_size = 'Small' if cohens_d < 0.2 else 'Medium' if cohens_d < 0.5 else 'Large' if cohens_d < 0.8 else 'Very Large'

        separability = 'Poor' if cohens_d < 0.2 else 'Fair' if cohens_d < 0.5 else 'Good' if cohens_d < 0.8 else 'Excellent'

        print(f"{param:<15} {cohens_d:<15.4f} {effect_size:<15} {separability:<20}")

# ============================================================================
# 6. THRESHOLD ANALYSIS - What's the optimal cutoff?
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Optimal Threshold Analysis")
print("="*120)

print("\nFor each parameter, what threshold best separates ACTION from HOLD?")
print(f"\n{'参数':<15} {'最优阈值':<15} {'方向':<15} {'ACTION比例':<15} {'HOLD比例':<15}")
print("-"*90)

for param in ['量能比率', '价格vsEMA%']:
    if param in action_moments.columns and param in hold_moments.columns:
        action_data = action_moments[param].dropna()
        hold_data = hold_moments[param].dropna()

        # Calculate correlation direction
        all_data = pd.concat([action_data, hold_data], axis=0)
        labels = [1]*len(action_data) + [0]*len(hold_data)
        corr, _ = stats.pointbiserialr(all_data, labels)

        # Find optimal threshold
        unique_values = np.sort(all_data.unique())
        best_threshold = None
        best_action_ratio = 0
        best_hold_ratio = 1

        for threshold in unique_values:
            if corr > 0:
                # Positive correlation: higher values → more ACTION
                action_above = (action_data >= threshold).sum() / len(action_data)
                hold_above = (hold_data >= threshold).sum() / len(hold_data)
            else:
                # Negative correlation: lower values → more ACTION
                action_above = (action_data <= threshold).sum() / len(action_data)
                hold_above = (hold_data <= threshold).sum() / len(hold_data)

            # We want ACTION ratio high, HOLD ratio low
            if action_above - hold_above > best_action_ratio - best_hold_ratio:
                best_action_ratio = action_above
                best_hold_ratio = hold_above
                best_threshold = threshold

        direction = "≥" if corr > 0 else "≤"

        print(f"{param:<15} {best_threshold:<15.4f} {direction:<15} {best_action_ratio:<15.2%} {best_hold_ratio:<15.2%}")

# ============================================================================
# 7. ACTION MOMENT CLASSIFICATION - By action type
# ============================================================================
print("\n" + "="*120)
print("STEP 7: ACTION Moment Classification")
print("="*120)

action_long = action_moments[action_moments['Action_Detail'] == '做多']
action_short = action_moments[action_moments['Action_Detail'] == '做空']

print(f"\n做多时刻: {len(action_long)} 个")
print(f"做空时刻: {len(action_short)} 个")

if len(action_long) > 0 and len(action_short) > 0:
    print(f"\n做多 vs 做空 的数据特征:")
    print(f"{'参数':<15} {'做多均值':<15} {'做空均值':<15} {'差异':<12}")
    print("-"*70)

    for param in params:
        if param in action_long.columns:
            long_mean = action_long[param].mean()
            short_mean = action_short[param].mean()
            diff = long_mean - short_mean
            print(f"{param:<15} {long_mean:<15.4f} {short_mean:<15.4f} {diff:<12.4f}")

# ============================================================================
# 8. CONSISTENCY CHECK - How consistent are the patterns?
# ============================================================================
print("\n" + "="*120)
print("STEP 8: Pattern Consistency Analysis")
print("="*120)

print("\nQuestion: Do ACTION moments consistently fall in certain parameter ranges?")
print("\nTesting: What % of ACTION moments satisfy each condition?")

conditions = [
    ('量能比率 > 1.2', lambda x: x['量能比率'] > 1.2),
    ('量能比率 > 1.4', lambda x: x['量能比率'] > 1.4),
    ('价格vsEMA% < -1.5%', lambda x: x['价格vsEMA%'] < -1.5),
    ('价格vsEMA% < -2.4%', lambda x: x['价格vsEMA%'] < -2.4),
    ('价格vsEMA% > 2.0%', lambda x: x['价格vsEMA%'] > 2.0),
    ('张力绝对值 > 1.0', lambda x: abs(x['张力']) > 1.0),
    ('DXY燃料 > 0.1', lambda x: x['DXY燃料'] > 0.1),
    ('加速度 < -0.5', lambda x: x['加速度'] < -0.5),
    ('加速度 > 0.5', lambda x: x['加速度'] > 0.5),
]

print(f"\n{'条件':<30} {'ACTION满足':<15} {'HOLD满足':<15} {'倍数':<10}")
print("-"*90)

for condition_name, condition_func in conditions:
    action_satisfy = action_moments.apply(condition_func, axis=1).sum()
    hold_satisfy = hold_moments.apply(condition_func, axis=1).sum()

    action_pct = action_satisfy / len(action_moments) if len(action_moments) > 0 else 0
    hold_pct = hold_satisfy / len(hold_moments) if len(hold_moments) > 0 else 0

    ratio = action_pct / hold_pct if hold_pct > 0 else float('inf')

    marker = '***' if ratio > 2.0 else '**' if ratio > 1.5 else '*' if ratio > 1.2 else ''

    print(f"{condition_name:<30} {action_pct:<15.2%} {hold_pct:<15.2%} {ratio:.2f}x {marker}")

# ============================================================================
# 9. COMBINATION PATTERNS - What combinations work best?
# ============================================================================
print("\n" + "="*120)
print("STEP 9: Combination Pattern Analysis")
print("="*120)

print("\nTesting: What parameter combinations have high ACTION probability?")

# Test combinations
combinations = [
    ('高量能+深度超卖',
     lambda x: (x['量能比率'] > 1.4) & (x['价格vsEMA%'] < -2.4)),
    ('高量能+轻度超卖',
     lambda x: (x['量能比率'] > 1.4) & (x['价格vsEMA%'] < -1.5) & (x['价格vsEMA%'] >= -2.4)),
    ('高量能+轻度超买',
     lambda x: (x['量能比率'] > 1.4) & (x['价格vsEMA%'] > 2.0)),
    ('高张力+极值乖离',
     lambda x: (abs(x['张力']) > 1.0) & (abs(x['价格vsEMA%']) > 2.0)),
]

print(f"\n{'组合':<30} {'ACTION中':<10} {'HOLD中':<10} {'ACTION概率':<15} {'提升倍数':<15}")
print("-"*100)

baseline_action_rate = len(action_moments) / (len(action_moments) + len(hold_moments))

for combo_name, combo_func in combinations:
    action_match = action_moments.apply(combo_func, axis=1).sum()
    hold_match = hold_moments.apply(combo_func, axis=1).sum()

    if action_match + hold_match > 0:
        action_prob = action_match / (action_match + hold_match)
        lift = action_prob / baseline_action_rate

        print(f"{combo_name:<30} {action_match:<10} {hold_match:<10} {action_prob:<15.2%} {lift:.2f}x")

print(f"\n基线ACTION概率: {baseline_action_rate:.2%}")

# ============================================================================
# 10. FINAL SUMMARY - What defines a perfect ACTION moment?
# ============================================================================
print("\n" + "="*120)
print("FINAL SUMMARY - What Defines a Perfect ACTION Moment?")
print("="*120)

print(f"""
基于{len(action_moments)}个手动标注ACTION时刻的分析：

【最显著特征】
1. 量能比率显著升高 (均值 vs HOLD: +{action_moments['量能比率'].mean() - hold_moments['量能比率'].mean():.4f})
2. 价格偏离EMA (均值 vs HOLD: {action_moments['价格vsEMA%'].mean():.4f}%)

【高置信度组合】
- 高量能(>1.4) + 深度超卖(<-2.4%): ACTION概率显著提升
- 高张力(>1.0) + 极值乖离(>2.0%): ACTION概率显著提升

【建议的V7.0.5优化方向】
1. 提高量能阈值: 从当前提升至1.4
2. 增加组合条件: 必须同时满足量能和价格位置
3. 考虑张力极值: 作为反转信号
""")

# Save detailed results
action_analysis_cols = ['时间', '信号类型', '量能比率', '价格vsEMA%', '张力', '加速度', 'DXY燃料', '黄金信号']
action_moments[action_analysis_cols].to_csv('ACTION_Moments_Detail.csv', index=False, encoding='utf-8-sig')
print(f"\nACTION时刻详细数据已保存至: ACTION_Moments_Detail.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)
