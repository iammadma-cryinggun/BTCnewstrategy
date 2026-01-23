# -*- coding: utf-8 -*-
"""
V8.0 混合协同策略 - 海陆空三军完整体系
========================================

系统架构:
1. V8 特种部队 (狙击手) - 捕捉爆发启动点
2. Singularity 消防队 (反转专家) - 捕捉极值反转
3. Normal 常规步兵 (趋势跟随) - 稳健持仓

核心原则: 互补关系，不是替代关系
"""

import pandas as pd
import numpy as np

print("="*120)
print("V8.0 HYBRID STRATEGY - Complete System (海陆空三军)")
print("="*120)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条信号")
print(f"时间范围: {df['时间'].min()} 至 {df['时间'].max()}")

# ============================================================================
# 1. FEATURE ENGINEERING - Calculate V8.0 Deltas
# ============================================================================
print("\n" + "="*120)
print("STEP 1: Calculate Dynamic Deltas (V8.0 Engine)")
print("="*120)

# Calculate previous values
df['Last_Vol'] = df['量能比率'].shift(1)
df['Last_EMA_Pct'] = df['价格vsEMA%'].shift(1)

# Calculate Deltas
df['Delta_Vol'] = (df['量能比率'] - df['Last_Vol']) / df['Last_Vol'].replace(0, np.nan)
df['Delta_EMA'] = df['价格vsEMA%'] - df['Last_EMA_Pct']
df['Delta_EMA_Abs'] = df['Delta_EMA'].abs()

# Clean
df['Delta_Vol'] = df['Delta_Vol'].replace([np.inf, -np.inf], 0).fillna(0)
df['Delta_EMA'] = df['Delta_EMA'].replace([np.inf, -np.inf], 0).fillna(0)
df['Delta_EMA_Abs'] = df['Delta_EMA_Abs'].replace([np.inf, -np.inf], 0).fillna(0)

print("Delta计算完成")

# ============================================================================
# 2. V8.0 SCORING - 特种部队评分
# ============================================================================
print("\n" + "="*120)
print("STEP 2: V8.0 Scoring (特种部队 - 爆发检测)")
print("="*120)

def normalize_clipped(series, cap):
    return series.abs().clip(upper=cap) / cap

# V8.0 Score Components
score_ema_delta = normalize_clipped(df['Delta_EMA_Abs'], cap=1.0) * 0.5  # 50% weight
score_vol_delta = normalize_clipped(df['Delta_Vol'], cap=0.5) * 0.3     # 30% weight
score_base_vol = normalize_clipped(df['量能比率'], cap=1.5) * 0.2       # 20% weight

df['V8_Score'] = score_ema_delta + score_vol_delta + score_base_vol

print(f"V8_Score 分布:")
print(f"  均值: {df['V8_Score'].mean():.4f}")
print(f"  标准差: {df['V8_Score'].std():.4f}")
print(f"  最大值: {df['V8_Score'].max():.4f}")

# ============================================================================
# 3. SINGULARITY SCORING - 消防队评分
# ============================================================================
print("\n" + "="*120)
print("STEP 3: Singularity Scoring (消防队 - 极值反转检测)")
print("="*120)

# 奇点信号特征: 高张力 + 极端乖离 + DXY燃料
df['Tension_Abs'] = df['张力'].abs()
df['PricevsEMA_Abs'] = df['价格vsEMA%'].abs()

# Singularity Score Components
score_tension = normalize_clipped(df['Tension_Abs'], cap=2.0) * 0.4      # 40% weight
score_extremity = normalize_clipped(df['PricevsEMA_Abs'], cap=3.0) * 0.3 # 30% weight
score_dxy = normalize_clipped(df['DXY燃料'].abs(), cap=1.0) * 0.2        # 20% weight
score_vol_base = normalize_clipped(df['量能比率'], cap=1.5) * 0.1        # 10% weight

df['Singularity_Score'] = (
    score_tension + score_extremity + score_dxy + score_vol_base
)

print(f"Singularity_Score 分布:")
print(f"  均值: {df['Singularity_Score'].mean():.4f}")
print(f"  标准差: {df['Singularity_Score'].std():.4f}")
print(f"  最大值: {df['Singularity_Score'].max():.4f}")

# ============================================================================
# 4. NORMAL SIGNAL SCORING - 常规步兵评分
# ============================================================================
print("\n" + "="*120)
print("STEP 4: Normal Signal Scoring (常规步兵 - 趋势跟随)")
print("="*120)

# 普通信号特征: 稳定的趋势排列 + 适中量能
df['V705_Pass_Binary'] = df['V7.0.5通过'].isin(['TRUE', True]).astype(int)

# V7.0.5已经是一个综合评分，我们直接使用
df['Normal_Score'] = df['V705_Pass_Binary']

print(f"Normal_Score (V7.0.5):")
print(f"  通过信号数: {df['V705_Pass_Binary'].sum()}")
print(f"  通过率: {df['V705_Pass_Binary'].sum()/len(df)*100:.1f}%")

# ============================================================================
# 5. HYBRID DECISION LOGIC - 混合决策系统
# ============================================================================
print("\n" + "="*120)
print("STEP 5: Hybrid Decision Logic (混合决策系统)")
print("="*120)

print("""
【决策规则 - 宪法级优先级】

1. V8特种部队 (Score > 0.55) → 检测到爆发突变
   - 优先级最高，必须立即响应
   - 操作: 重仓追单

2. Singularity消防队 (Score > 0.60) → 检测到极值反转
   - 优先级第二，准备反转
   - 操作: 左侧摸顶/抄底

3. Normal常规步兵 (Pass = TRUE) → 趋势确认
   - 优先级第三，稳健入场
   - 操作: 标准仓位

4. 冲突处理:
   - V8说多 + Singularity说空 → 信V8短期，但准备随时跑路
   - V8没反应 + Singularity说空 → 信Singularity，左侧抄底
   - V8说多 + Normal说震荡 → 信V8，复活机会
""")

def hybrid_decision(row):
    """
    混合决策函数

    Returns: (Action_Type, Position_Size, Reason)
    """
    v8_score = row['V8_Score']
    sing_score = row['Singularity_Score']
    normal_pass = row['V705_Pass_Binary']
    signal_type = row['信号类型']

    # 触发阈值
    V8_THRESHOLD = 0.55
    SING_THRESHOLD = 0.60

    # Rule 1: V8特种部队触发
    if v8_score > V8_THRESHOLD:
        # 检测方向: 根据Delta_EMA的方向
        delta_ema = row['Delta_EMA']
        if delta_ema > 0:
            direction = "做多"
        elif delta_ema < 0:
            direction = "做空"
        else:
            direction = "观望"

        # 如果Singularity也触发，提示风险
        if sing_score > SING_THRESHOLD:
            risk_level = "高位赶顶" if delta_ema > 0 else "低位赶底"
            return ('ACTION_V8_SNIPER', '重仓(警惕反转)', f'V8突变检测({direction}) + {risk_level}')
        else:
            return ('ACTION_V8_SNIPER', '重仓', f'V8突变检测({direction})')

    # Rule 2: Singularity消防队触发
    elif sing_score > SING_THRESHOLD:
        # 判断是顶还是底
        tension = row['张力']
        price_vs_ema = row['价格vsEMA%']

        if tension > 1.0 or price_vs_ema > 2.0:
            direction = "反手做空"
        elif tension < -1.0 or price_vs_ema < -2.0:
            direction = "反手做多"
        else:
            direction = "观望"

        return ('ACTION_SINGULARITY', '中仓', f'奇点反转检测({direction})')

    # Rule 3: Normal常规步兵
    elif normal_pass == 1 and signal_type != 'OSCILLATION':
        return ('ACTION_NORMAL', '标准仓', 'V7.0.5趋势跟随')

    # Default: HOLD
    else:
        return ('HOLD', '空仓', '观望')

# Apply hybrid decision
results = df.apply(hybrid_decision, axis=1, result_type='expand')
df['Hybrid_Action'] = results[0]
df['Position_Size'] = results[1]
df['Decision_Reason'] = results[2]

# ============================================================================
# 6. STATISTICAL ANALYSIS - 统计分析
# ============================================================================
print("\n" + "="*120)
print("STEP 6: Statistical Analysis (统计分析)")
print("="*120)

# Extract ideal actions
df['Ideal_Action'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'UNKNOWN'
)

df_valid = df[df['Ideal_Action'].isin(['ACTION', 'HOLD'])].copy()

# Binary encoding for hybrid
hybrid_binary = df_valid['Hybrid_Action'].apply(lambda x: 1 if x.startswith('ACTION') else 0)
ideal_binary = (df_valid['Ideal_Action'] == 'ACTION').astype(int)

# Confusion Matrix
tp = ((hybrid_binary == 1) & (ideal_binary == 1)).sum()
tn = ((hybrid_binary == 0) & (ideal_binary == 0)).sum()
fp = ((hybrid_binary == 1) & (ideal_binary == 0)).sum()
fn = ((hybrid_binary == 0) & (ideal_binary == 1)).sum()

print("\n【混合策略混淆矩阵】")
print(f"                      混合策略预测")
print(f"              HOLD              ACTION")
print(f"实际 HOLD   {tn:4d} (正确)      {fp:4d} (误报)")
print(f"     ACTION {fn:4d} (漏报)      {tp:4d} (正确)")

# Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\n【混合策略性能指标】")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")
print(f"  准确率: {accuracy:.4f}")

# ============================================================================
# 7. BREAKDOWN BY ARMY TYPE - 各军种战果分析
# ============================================================================
print("\n" + "="*120)
print("STEP 7: Performance Breakdown by Army Type (各军种战果)")
print("="*120)

# V8 Sniper performance
v8_mask = df_valid['Hybrid_Action'] == 'ACTION_V8_SNIPER'
v8_correct = ((v8_mask) & (hybrid_binary == ideal_binary)).sum()
v8_total = v8_mask.sum()

print(f"\n【V8特种部队】")
print(f"  触发次数: {v8_total}")
print(f"  正确次数: {v8_correct}")
print(f"  成功率: {v8_correct/v8_total*100:.1f}%" if v8_total > 0 else "  成功率: N/A")

# Singularity performance
sing_mask = df_valid['Hybrid_Action'] == 'ACTION_SINGULARITY'
sing_correct = ((sing_mask) & (hybrid_binary == ideal_binary)).sum()
sing_total = sing_mask.sum()

print(f"\n【Singularity消防队】")
print(f"  触发次数: {sing_total}")
print(f"  正确次数: {sing_correct}")
print(f"  成功率: {sing_correct/sing_total*100:.1f}%" if sing_total > 0 else "  成功率: N/A")

# Normal performance
normal_mask = df_valid['Hybrid_Action'] == 'ACTION_NORMAL'
normal_correct = ((normal_mask) & (hybrid_binary == ideal_binary)).sum()
normal_total = normal_mask.sum()

print(f"\n【Normal常规步兵】")
print(f"  触发次数: {normal_total}")
print(f"  正确次数: {normal_correct}")
print(f"  成功率: {normal_correct/normal_total*100:.1f}%" if normal_total > 0 else "  成功率: N/A")

# ============================================================================
# 8. COMPARISON WITH PURE STRATEGIES
# ============================================================================
print("\n" + "="*120)
print("STEP 8: Comparison - Hybrid vs Pure Strategies")
print("="*120)

# Pure V8.0 (threshold 0.50)
pure_v8_binary = (df_valid['V8_Score'] >= 0.50).astype(int)
tp_v8 = ((pure_v8_binary == 1) & (ideal_binary == 1)).sum()
fp_v8 = ((pure_v8_binary == 1) & (ideal_binary == 0)).sum()
fn_v8 = ((pure_v8_binary == 0) & (ideal_binary == 1)).sum()
tn_v8 = ((pure_v8_binary == 0) & (ideal_binary == 0)).sum()
precision_v8 = tp_v8 / (tp_v8 + fp_v8) if (tp_v8 + fp_v8) > 0 else 0
recall_v8 = tp_v8 / (tp_v8 + fn_v8) if (tp_v8 + fn_v8) > 0 else 0
f1_v8 = 2 * precision_v8 * recall_v8 / (precision_v8 + recall_v8) if (precision_v8 + recall_v8) > 0 else 0

# Pure V7.0.5
pure_v7_binary = df_valid['V705_Pass_Binary']
tp_v7 = ((pure_v7_binary == 1) & (ideal_binary == 1)).sum()
fp_v7 = ((pure_v7_binary == 1) & (ideal_binary == 0)).sum()
fn_v7 = ((pure_v7_binary == 0) & (ideal_binary == 1)).sum()
tn_v7 = ((pure_v7_binary == 0) & (ideal_binary == 0)).sum()
precision_v7 = tp_v7 / (tp_v7 + fp_v7) if (tp_v7 + fp_v7) > 0 else 0
recall_v7 = tp_v7 / (tp_v7 + fn_v7) if (tp_v7 + fn_v7) > 0 else 0
f1_v7 = 2 * precision_v7 * recall_v7 / (precision_v7 + recall_v7) if (precision_v7 + recall_v7) > 0 else 0

print(f"\n{'策略':<20} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'符合率':<12}")
print("-"*80)
print(f"{'Pure V8.0':<20} {precision_v8:<12.4f} {recall_v8:<12.4f} {f1_v8:<12.4f} {(tp_v8+tn_v8)/len(df_valid):.4f}")
print(f"{'Pure V7.0.5':<20} {precision_v7:<12.4f} {recall_v7:<12.4f} {f1_v7:<12.4f} {(tp_v7+tn_v7)/len(df_valid):.4f}")
print(f"{'Hybrid (V8+V7)':<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:.4f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "="*120)
print("STEP 9: Save Results")
print("="*120)

output_cols = [
    '时间', '信号类型',
    '量能比率', '价格vsEMA%', '张力', 'DXY燃料',
    'Delta_Vol', 'Delta_EMA_Abs',
    'V8_Score', 'Singularity_Score', 'V7.0.5通过',
    'Hybrid_Action', 'Position_Size', 'Decision_Reason',
    '黄金信号'
]

df[output_cols].to_csv('V8_Hybrid_Results.csv', index=False, encoding='utf-8-sig')
print(f"\n完整结果已保存至: V8_Hybrid_Results.csv")

print("\n" + "="*120)
print("V8.0 HYBRID STRATEGY - FINAL SUMMARY")
print("="*120)
print(f"\n最终混合策略性能:")
print(f"  F1分数: {f1:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  准确率: {accuracy:.4f}")
print(f"\n各军种贡献:")
print(f"  V8特种部队触发: {v8_total}次")
print(f"  Singularity消防队触发: {sing_total}次")
print(f"  Normal常规步兵触发: {normal_total}次")
