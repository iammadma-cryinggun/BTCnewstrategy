import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("统计学全面分析：好机会 vs 差机会的判别特征")
print("="*100)

# 读取数据
df = pd.read_excel('2025年6-12月_完整标注结果_SHORT+LONG.xlsx', sheet_name='所有标注')

def comprehensive_analysis(direction):
    print("\n" + "="*100)
    print(f"【{direction}信号】统计学分析报告")
    print("="*100)

    data = df[df['交易方向'] == direction].copy()
    good = data[data['是好机会'] == '是'].copy()
    bad = data[data['是好机会'] == '否'].copy()

    # 转换目标变量
    data['target'] = (data['是好机会'] == '是').astype(int)

    print(f"\n样本量:")
    print(f"  总数: {len(data)}笔")
    print(f"  好机会: {len(good)}笔 ({len(good)/len(data)*100:.1f}%)")
    print(f"  差机会: {len(bad)}笔 ({len(bad)/len(data)*100:.1f}%)")

    # ========== 第一部分：描述性统计 ==========
    print("\n" + "-"*100)
    print("一、描述性统计")
    print("-"*100)

    # 1.1 连续变量统计
    if direction == 'SHORT':
        good['张力_加速度比'] = good['首次张力'] / abs(good['首次加速度'])
        bad['张力_加速度比'] = bad['首次张力'] / abs(bad['首次加速度'])
        data['张力_加速度比'] = data['首次张力'] / abs(data['首次加速度'])
    else:
        good['张力_加速度比'] = abs(good['首次张力']) / good['首次加速度']
        bad['张力_加速度比'] = abs(bad['首次张力']) / bad['首次加速度']
        data['张力_加速度比'] = abs(data['首次张力']) / data['首次加速度']

    continuous_vars = ['首次张力', '首次加速度', '首次能量', '张力_加速度比', '张力变化%', '价格优势%', '等待周期']

    for var in continuous_vars:
        if var in ['首次张力', '首次加速度']:
            if direction == 'SHORT' and var == '首次张力':
                good_vals = good[var]
                bad_vals = bad[var]
            elif direction == 'LONG' and var == '首次张力':
                good_vals = abs(good[var])
                bad_vals = abs(bad[var])
            elif direction == 'SHORT' and var == '首次加速度':
                good_vals = abs(good[var])
                bad_vals = abs(bad[var])
            else:
                good_vals = good[var]
                bad_vals = bad[var]
        else:
            good_vals = good[var]
            bad_vals = bad[var]

        print(f"\n{var}:")
        print(f"  好机会: 均值={good_vals.mean():.4f}, 中位数={good_vals.median():.4f}, 标准差={good_vals.std():.4f}")
        print(f"         Q1={good_vals.quantile(0.25):.4f}, Q3={good_vals.quantile(0.75):.4f}")
        print(f"  差机会: 均值={bad_vals.mean():.4f}, 中位数={bad_vals.median():.4f}, 标准差={bad_vals.std():.4f}")
        print(f"         Q1={bad_vals.quantile(0.25):.4f}, Q3={bad_vals.quantile(0.75):.4f}")

        # t检验
        t_stat, p_value = stats.ttest_ind(good_vals, bad_vals)
        print(f"  t检验: t={t_stat:.4f}, p={p_value:.4f} {'***显著***' if p_value < 0.05 else ''}")

    # ========== 第二部分：相关性分析 ==========
    print("\n" + "-"*100)
    print("二、相关性分析（与是否好机会的相关性）")
    print("-"*100)

    # 计算各变量与目标变量的相关系数
    correlations = []
    for var in ['首次张力', '首次加速度', '首次能量', '张力_加速度比', '张力变化%', '价格优势%', '等待周期']:
        if var == '首次张力' and direction == 'LONG':
            corr = data['target'].corr(abs(data[var]))
        elif var == '首次加速度' and direction == 'SHORT':
            corr = data['target'].corr(abs(data[var]))
        else:
            corr = data['target'].corr(data[var])
        correlations.append({'变量': var, '相关系数': corr})

    corr_df = pd.DataFrame(correlations).sort_values('相关系数', key=abs, ascending=False)
    for _, row in corr_df.iterrows():
        significance = '***' if abs(row['相关系数']) > 0.3 else '**' if abs(row['相关系数']) > 0.1 else '*'
        print(f"  {row['变量']}: {row['相关系数']:.4f} {significance}")

    # ========== 第三部分：判别分析 ==========
    print("\n" + "-"*100)
    print("三、判别特征分析")
    print("-"*100)

    # 3.1 计算Cohen's d (效应量)
    print("\nCohen's d (效应量):")
    for var in ['首次张力', '首次能量', '张力_加速度比', '张力变化%', '价格优势%']:
        if var == '首次张力' and direction == 'LONG':
            good_vals = abs(good[var])
            bad_vals = abs(bad[var])
        elif var == '首次张力' and direction == 'SHORT':
            good_vals = good[var]
            bad_vals = bad[var]
        else:
            good_vals = good[var]
            bad_vals = bad[var]

        # Cohen's d
        pooled_std = np.sqrt(((len(good_vals)-1)*good_vals.std()**2 + (len(bad_vals)-1)*bad_vals.std()**2) / (len(good_vals)+len(bad_vals)-2))
        cohens_d = (good_vals.mean() - bad_vals.mean()) / pooled_std

        effect_size = '超大' if abs(cohens_d) > 0.8 else '大' if abs(cohens_d) > 0.5 else '中等' if abs(cohens_d) > 0.2 else '小'
        print(f"  {var}: d={cohens_d:.4f} ({effect_size}效应)")

    # 3.2 最优阈值分析（Youden指数）
    print("\n最优判别阈值:")
    for var in ['首次张力', '首次能量', '张力_加速度比', '张力变化%', '价格优势%']:
        if var == '首次张力' and direction == 'LONG':
            vals = abs(data[var])
        elif var == '首次张力' and direction == 'SHORT':
            vals = data[var]
        else:
            vals = data[var]

        # 尝试不同阈值，找最优
        best_threshold = None
        best_youden = -1

        for threshold in np.linspace(vals.min(), vals.max(), 100):
            tp = ((vals >= threshold) & (data['target'] == 1)).sum()
            fp = ((vals >= threshold) & (data['target'] == 0)).sum()
            tn = ((vals < threshold) & (data['target'] == 0)).sum()
            fn = ((vals < threshold) & (data['target'] == 1)).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = sensitivity + specificity - 1

            if youden > best_youden:
                best_youden = youden
                best_threshold = threshold

        print(f"  {var}: ≥{best_threshold:.4f} (Youden指数={best_youden:.4f})")

    # ========== 第四部分：多变量分析 ==========
    print("\n" + "-"*100)
    print("四、多变量特征重要性（随机森林）")
    print("-"*100)

    # 准备特征
    features_data = pd.DataFrame({
        '张力': abs(data['首次张力']) if direction == 'LONG' else data['首次张力'],
        '加速度': abs(data['首次加速度']),
        '能量': data['首次能量'],
        '张力_加速度比': data['张力_加速度比'],
        '等待周期': data['等待周期']
    })

    target = data['target']

    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features_data, target)

    # 特征重要性
    importance_df = pd.DataFrame({
        '特征': features_data.columns,
        '重要性': rf.feature_importances_
    }).sort_values('重要性', ascending=False)

    for _, row in importance_df.iterrows():
        print(f"  {row['特征']}: {row['重要性']:.4f} ({row['重要性']*100:.1f}%)")

    # ========== 第五部分：概率分布 ==========
    print("\n" + "-"*100)
    print("五、好机会概率分布")
    print("-"*100)

    # 5.1 按张力区间
    if direction == 'SHORT':
        tension_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0), (1.0, 999)]
    else:
        tension_ranges = [(-999, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5)]

    print("\n按首次张力分组的好机会概率:")
    for min_t, max_t in tension_ranges:
        if max_t == 999:
            subset = data[data['首次张力'] >= min_t]
            label = f"张力≥{min_t}"
        elif min_t == -999:
            subset = data[data['首次张力'] < max_t]
            label = f"张力<{max_t}"
        else:
            subset = data[(data['首次张力'] >= min_t) & (data['首次张力'] < max_t)]
            label = f"{min_t}≤张力<{max_t}"

        if len(subset) > 0:
            prob = (subset['target'] == 1).sum() / len(subset) * 100
            ci = 1.96 * np.sqrt(prob/100 * (1-prob/100) / len(subset)) * 100  # 95%置信区间
            print(f"  {label}: {prob:.1f}% ±{ci:.1f}% (n={len(subset)})")

    # 5.2 按能量区间
    energy_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 999)]

    print("\n按首次能量分组的好机会概率:")
    for min_e, max_e in energy_ranges:
        if max_e == 999:
            subset = data[data['首次能量'] >= min_e]
            label = f"能量≥{min_e}"
        else:
            subset = data[(data['首次能量'] >= min_e) & (data['首次能量'] < max_e)]
            label = f"{min_e}≤能量<{max_e}"

        if len(subset) > 0:
            prob = (subset['target'] == 1).sum() / len(subset) * 100
            ci = 1.96 * np.sqrt(prob/100 * (1-prob/100) / len(subset)) * 100
            print(f"  {label}: {prob:.1f}% ±{ci:.1f}% (n={len(subset)})")

    # 5.3 按等待周期
    print("\n按等待周期的好机会概率:")
    for wait in range(1, 8):
        subset = data[data['等待周期'] == wait]
        if len(subset) > 0:
            prob = (subset['target'] == 1).sum() / len(subset) * 100
            ci = 1.96 * np.sqrt(prob/100 * (1-prob/100) / len(subset)) * 100
            print(f"  等待{wait}周期: {prob:.1f}% ±{ci:.1f}% (n={len(subset)})")

    # 5.4 按张力_加速度比
    ratio_ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 999)]

    print("\n按张力/加速度比分组的好机会概率:")
    for min_r, max_r in ratio_ranges:
        if max_r == 999:
            subset = data[data['张力_加速度比'] >= min_r]
            label = f"比例≥{min_r}"
        else:
            subset = data[(data['张力_加速度比'] >= min_r) & (data['张力_加速度比'] < max_r)]
            label = f"{min_r}≤比例<{max_r}"

        if len(subset) > 0:
            prob = (subset['target'] == 1).sum() / len(subset) * 100
            ci = 1.96 * np.sqrt(prob/100 * (1-prob/100) / len(subset)) * 100
            print(f"  {label}: {prob:.1f}% ±{ci:.1f}% (n={len(subset)})")

    # ========== 第六部分：综合判别模型 ==========
    print("\n" + "-"*100)
    print("六、综合判别模型")
    print("-"*100)

    # 基于统计显著性构建判别函数
    # 找出最优组合
    best_combinations = []

    # 组合1：张力 + 能量 + 等待周期
    if direction == 'SHORT':
        # 张力适中(0.5-0.7) + 能量(1.0-2.0) + 等待(4-6)
        subset1 = data[
            (data['首次张力'] >= 0.5) & (data['首次张力'] < 0.7) &
            (data['首次能量'] >= 1.0) & (data['首次能量'] < 2.0) &
            (data['等待周期'] >= 4) & (data['等待周期'] <= 6)
        ]
    else:
        # 张力<-0.7 + 能量≥1.0 + 等待(4-6)
        subset1 = data[
            (data['首次张力'] < -0.7) &
            (data['首次能量'] >= 1.0) &
            (data['等待周期'] >= 4) & (data['等待周期'] <= 6)
        ]

    if len(subset1) > 0:
        prob1 = (subset1['target'] == 1).sum() / len(subset1) * 100
        best_combinations.append({
            '组合': '张力适中+能量适中+等待4-6周期',
            '样本数': len(subset1),
            '好机会率': prob1
        })

    # 组合2：高比例(>100) + 等待(4-6)
    subset2 = data[
        (data['张力_加速度比'] >= 100) &
        (data['等待周期'] >= 4) & (data['等待周期'] <= 6)
    ]

    if len(subset2) > 0:
        prob2 = (subset2['target'] == 1).sum() / len(subset2) * 100
        best_combinations.append({
            '组合': '高比例+等待4-6周期',
            '样本数': len(subset2),
            '好机会率': prob2
        })

    # 组合3：张力大幅变化(>5%) + 等待(4-6)
    subset3 = data[
        (data['张力变化%'] >= 5) &
        (data['等待周期'] >= 4) & (data['等待周期'] <= 6)
    ]

    if len(subset3) > 0:
        prob3 = (subset3['target'] == 1).sum() / len(subset3) * 100
        best_combinations.append({
            '组合': '张力大幅变化+等待4-6周期',
            '样本数': len(subset3),
            '好机会率': prob3
        })

    print("\n最优组合策略:")
    best_combinations.sort(key=lambda x: x['好机会率'], reverse=True)
    for combo in best_combinations:
        print(f"  {combo['组合']}:")
        print(f"    样本数: {combo['样本数']}")
        print(f"    好机会率: {combo['好机会率']:.1f}%")

    return data, good, bad

# 分析SHORT和LONG
short_data, short_good, short_bad = comprehensive_analysis('SHORT')
long_data, long_good, long_bad = comprehensive_analysis('LONG')

# ========== 第七部分：SHORT vs LONG对比 ==========
print("\n" + "="*100)
print("【SHORT vs LONG】对比分析")
print("="*100)

print("\n好机会率:")
print(f"  SHORT: {len(short_good)/len(short_data)*100:.1f}%")
print(f"  LONG:  {len(long_good)/len(long_data)*100:.1f}%")
print(f"  差异:  {len(long_good)/len(long_data)*100 - len(short_good)/len(short_data)*100:+.1f}个百分点")

print("\n平均等待周期:")
print(f"  SHORT好机会: {short_good['等待周期'].mean():.1f}")
print(f"  LONG好机会:  {long_good['等待周期'].mean():.1f}")

print("\n平均张力变化:")
print(f"  SHORT好机会: {short_good['张力变化%'].mean():.2f}%")
print(f"  LONG好机会:  {long_good['张力变化%'].mean():.2f}%")

print("\n平均价格优势:")
print(f"  SHORT好机会: {short_good['价格优势%'].mean():.3f}%")
print(f"  LONG好机会:  {long_good['价格优势%'].mean():.3f}%")

# 保存详细统计
print("\n" + "="*100)
print("保存统计分析结果...")
print("="*100)

output_file = '统计学全面分析报告.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    short_data.to_excel(writer, sheet_name='SHORT全部', index=False)
    short_good.to_excel(writer, sheet_name='SHORT好机会', index=False)
    short_bad.to_excel(writer, sheet_name='SHORT差机会', index=False)

    long_data.to_excel(writer, sheet_name='LONG全部', index=False)
    long_good.to_excel(writer, sheet_name='LONG好机会', index=False)
    long_bad.to_excel(writer, sheet_name='LONG差机会', index=False)

print(f"\n已保存到: {output_file}")

print("\n" + "="*100)
print("统计学分析完成！")
print("="*100)
