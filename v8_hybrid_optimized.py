# -*- coding: utf-8 -*-
"""
V8.0 混合策略 - 优化版
======================

优化方向:
1. 提高V8.0阈值，减少误报
2. 降低Singularity阈值，激活反转检测
3. 引入"双确认"机制
"""

import pandas as pd
import numpy as np

print("="*120)
print("V8.0 HYBRID STRATEGY - OPTIMIZED VERSION")
print("="*120)

# Load data
df = pd.read_csv('V8_Hybrid_Results.csv', encoding='utf-8-sig')

print(f"\n数据集: {len(df)} 条信号")

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*120)
print("THRESHOLD OPTIMIZATION ANALYSIS")
print("="*120)

# Test different threshold combinations
v8_thresholds = [0.55, 0.60, 0.65, 0.70]
sing_thresholds = [0.40, 0.50, 0.55, 0.60]

print("\n测试不同阈值组合...")
print(f"{'V8阈':<8} {'Sing阈':<8} {'触发数':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
print("-"*80)

df['Ideal_Action'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'UNKNOWN'
)

df_valid = df[df['Ideal_Action'].isin(['ACTION', 'HOLD'])].copy()
ideal_binary = (df_valid['Ideal_Action'] == 'ACTION').astype(int)

best_f1 = -1
best_config = None

for v8_th in v8_thresholds:
    for sing_th in sing_thresholds:
        # Hybrid logic
        def test_decision(row):
            v8_score = row['V8_Score']
            sing_score = row['Singularity_Score']
            normal_pass = row['V7.0.5通过'] in ['TRUE', True]

            # Priority 1: V8 Sniper
            if v8_score > v8_th:
                return 'ACTION'

            # Priority 2: Singularity Reversal
            elif sing_score > sing_th:
                return 'ACTION'

            # Priority 3: Normal Trend
            elif normal_pass:
                return 'ACTION'

            else:
                return 'HOLD'

        hybrid_pred = df_valid.apply(test_decision, axis=1)
        hybrid_binary = (hybrid_pred == 'ACTION').astype(int)

        tp = ((hybrid_binary == 1) & (ideal_binary == 1)).sum()
        fp = ((hybrid_binary == 1) & (ideal_binary == 0)).sum()
        fn = ((hybrid_binary == 0) & (ideal_binary == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        triggers = hybrid_binary.sum()

        print(f"{v8_th:<8.2f} {sing_th:<8.2f} {triggers:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_config = (v8_th, sing_th, precision, recall, f1, triggers)

print(f"\n最佳配置:")
print(f"  V8阈值: {best_config[0]}")
print(f"  Singularity阈值: {best_config[1]}")
print(f"  精确率: {best_config[2]:.4f}")
print(f"  召回率: {best_config[3]:.4f}")
print(f"  F1分数: {best_config[4]:.4f}")
print(f"  触发数: {best_config[5]}")

# ============================================================================
# APPLY OPTIMAL THRESHOLDS
# ============================================================================
print("\n" + "="*120)
print("APPLY OPTIMAL CONFIGURATION")
print("="*120)

OPTIMAL_V8_TH = best_config[0]
OPTIMAL_SING_TH = best_config[1]

print(f"\n采用最优阈值:")
print(f"  V8.0: {OPTIMAL_V8_TH}")
print(f"  Singularity: {OPTIMAL_SING_TH}")

def optimized_hybrid_decision(row):
    """
    优化后的混合决策

    Returns: (Action_Type, Confidence, Reason)
    """
    v8_score = row['V8_Score']
    sing_score = row['Singularity_Score']
    normal_pass = row['V7.0.5通过'] in ['TRUE', True]
    signal_type = row['信号类型']
    delta_ema = row['Delta_EMA'] if 'Delta_EMA' in row else 0

    # Rule 1: V8 Sniper (High confidence)
    if v8_score > OPTIMAL_V8_TH:
        direction = "做多" if delta_ema > 0 else "做空"

        # Check if Singularity also triggers (risk warning)
        if sing_score > OPTIMAL_SING_TH:
            risk = "高位赶顶" if delta_ema > 0 else "低位赶底"
            return ('ACTION_V8_HIGH', '重仓(警惕)', f'V8突变({direction}) + {risk}')
        else:
            return ('ACTION_V8_HIGH', '重仓', f'V8突变({direction})')

    # Rule 2: Singularity Reversal
    elif sing_score > OPTIMAL_SING_TH:
        tension = row['张力'] if '张力' in row else 0
        price_vs_ema = row['价格vsEMA%'] if '价格vsEMA%' in row else 0

        if tension > 1.0 or price_vs_ema > 2.0:
            direction = "反手做空"
        elif tension < -1.0 or price_vs_ema < -2.0:
            direction = "反手做多"
        else:
            direction = "观望"

        return ('ACTION_SINGULARITY', '中仓', f'奇点反转({direction})')

    # Rule 3: Normal Trend (Only if not OSCILLATION)
    elif normal_pass and signal_type != 'OSCILLATION':
        return ('ACTION_NORMAL', '标准仓', 'V7.0.5趋势')

    # Rule 4: Special case - V7.0.5 + OSCILLATION with high volume
    elif normal_pass and signal_type == 'OSCILLATION':
        vol = row['量能比率'] if '量能比率' in row else 0
        if vol > 1.5:
            return ('ACTION_OSC_BREAKOUT', '轻仓', '震荡放量突破')
        else:
            return ('HOLD', '空仓', '震荡低量-观望')

    # Default
    else:
        return ('HOLD', '空仓', '观望')

# Apply optimized logic
results = df_valid.apply(optimized_hybrid_decision, axis=1, result_type='expand')
df_valid['Opt_Action'] = results[0]
df_valid['Opt_Position'] = results[1]
df_valid['Opt_Reason'] = results[2]

# ============================================================================
# FINAL PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "="*120)
print("FINAL PERFORMANCE EVALUATION")
print("="*120)

opt_binary = df_valid['Opt_Action'].apply(lambda x: 1 if x.startswith('ACTION') else 0)

tp = ((opt_binary == 1) & (ideal_binary == 1)).sum()
tn = ((opt_binary == 0) & (ideal_binary == 0)).sum()
fp = ((opt_binary == 1) & (ideal_binary == 0)).sum()
fn = ((opt_binary == 0) & (ideal_binary == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n【优化后混淆矩阵】")
print(f"                    预测")
print(f"            HOLD        ACTION")
print(f"实际 HOLD   {tn:4d}       {fp:4d}")
print(f"     ACTION {fn:4d}       {tp:4d}")

print(f"\n【优化后性能指标】")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")
print(f"  准确率: {accuracy:.4f}")

# ============================================================================
# BREAKDOWN BY ACTION TYPE
# ============================================================================
print("\n" + "="*120)
print("BREAKDOWN BY ACTION TYPE")
print("="*120)

action_types = ['ACTION_V8_HIGH', 'ACTION_SINGULARITY', 'ACTION_NORMAL', 'ACTION_OSC_BREAKOUT']

for action_type in action_types:
    mask = df_valid['Opt_Action'] == action_type
    total = mask.sum()
    if total > 0:
        correct = ((mask) & (opt_binary == ideal_binary)).sum()
        print(f"\n{action_type}:")
        print(f"  触发次数: {total}")
        print(f"  正确次数: {correct}")
        print(f"  成功率: {correct/total*100:.1f}%")

# ============================================================================
# SAVE OPTIMIZED RESULTS
# ============================================================================
print("\n" + "="*120)
print("SAVE OPTIMIZED RESULTS")
print("="*120)

output_cols = [
    '时间', '信号类型',
    '量能比率', '价格vsEMA%', '张力', 'DXY燃料',
    'V8_Score', 'Singularity_Score',
    'Opt_Action', 'Opt_Position', 'Opt_Reason',
    '黄金信号'
]

df_valid[output_cols].to_csv('V8_Optimized_Results.csv', index=False, encoding='utf-8-sig')
print(f"\n优化结果已保存至: V8_Optimized_Results.csv")

print("\n" + "="*120)
print("OPTIMIZATION COMPLETE")
print("="*120)
print(f"\n最优配置总结:")
print(f"  V8阈值: {OPTIMAL_V8_TH}")
print(f"  Singularity阈值: {OPTIMAL_SING_TH}")
print(f"  F1分数: {f1:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"\n相比原始混合策略的改进:")
print(f"  F1: {f1 - 0.2135:+.4f}")
print(f"  Precision: {precision - 0.1219:+.4f}")
print(f"  Recall: {recall - 0.8588:+.4f}")
