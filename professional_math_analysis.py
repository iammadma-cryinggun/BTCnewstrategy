# -*- coding: utf-8 -*-
"""
Professional Mathematical Analysis
==================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("MATHEMATICAL ANALYSIS - PROFESSIONAL STATISTICAL FRAMEWORK")
print("="*100)

# Load data
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')

# Classification
df['Action_Type'] = df['黄金信号'].apply(lambda x:
    'ACTION' if any(k in str(x) for k in ['平', '反', '开']) else
    'HOLD' if any(k in str(x) for k in ['继续持', '持仓']) else 'OTHER'
)

df_analysis = df[df['Action_Type'].isin(['ACTION', 'HOLD'])].copy()
df_analysis['Target'] = (df_analysis['Action_Type'] == 'ACTION').astype(int)

# Parameters
params = ['张力', '加速度', 'DXY燃料', '量能比率', '价格vsEMA%']
params = [p for p in params if p in df_analysis.columns]

X = df_analysis[params].values
y = df_analysis['Target'].values

print(f"\nSample Size: n={len(df_analysis)}")
print(f"ACTION cases: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"HOLD cases: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

# Section 1: Descriptive Statistics
print("\n" + "="*100)
print("SECTION 1: DESCRIPTIVE STATISTICS WITH 95% CONFIDENCE INTERVALS")
print("="*100)

for param in params:
    data_action = df_analysis[df_analysis['Action_Type']=='ACTION'][param].dropna()
    data_hold = df_analysis[df_analysis['Action_Type']=='HOLD'][param].dropna()

    print(f"\n{param}:")
    print("-"*80)

    n1, n2 = len(data_action), len(data_hold)
    mean1, mean2 = data_action.mean(), data_hold.mean()
    std1, std2 = data_action.std(), data_hold.std()

    # 95% CI
    ci1 = stats.t.interval(0.95, n1-1, loc=mean1, scale=std1/np.sqrt(n1))
    ci2 = stats.t.interval(0.95, n2-1, loc=mean2, scale=std2/np.sqrt(n2))

    # Cohen's d
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohens_d = abs(mean1 - mean2) / pooled_std

    print(f"ACTION:   n={n1}, mean={mean1:.6f}, 95% CI=[{ci1[0]:.6f}, {ci1[1]:.6f}]")
    print(f"HOLD:     n={n2}, mean={mean2:.6f}, 95% CI=[{ci2[0]:.6f}, {ci2[1]:.6f}]")
    print(f"Cohen's d: {cohens_d:.4f} ({'Small' if cohens_d<0.2 else 'Medium' if cohens_d<0.5 else 'Large'})")

    # Normality test
    if n1 >= 3 and n1 <= 5000:
        stat1, p1 = stats.shapiro(data_action[:min(5000, len(data_action))])
        print(f"Normality ACTION: W={stat1:.4f}, p={p1:.4f} {'[Normal]' if p1 > 0.05 else '[Not Normal]'}")

# Section 2: Mann-Whitney U Test
print("\n" + "="*100)
print("SECTION 2: MANN-WHITNEY U TEST (Non-parametric)")
print("="*100)

print(f"\n{'Parameter':>15s} | {'U-Statistic':>15s} | {'p-value':>12s} | {'Sig':>6s} | {'Effect r':>10s}")
print("-"*100)

for param in params:
    data_action = df_analysis[df_analysis['Action_Type']=='ACTION'][param].dropna()
    data_hold = df_analysis[df_analysis['Action_Type']=='HOLD'][param].dropna()

    u_stat, p_value = mannwhitneyu(data_action, data_hold, alternative='two-sided')

    n1, n2 = len(data_action), len(data_hold)
    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0
    effect_r = abs(z_score) / np.sqrt(n1 + n2)

    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''

    print(f"{param:>15s} | {u_stat:>15.2f} | {p_value:>12.6f} | {sig:>6s} | {effect_r:>10.4f}")

# Section 3: Correlation Analysis
print("\n" + "="*100)
print("SECTION 3: POINT-BISERIAL CORRELATION")
print("="*100)

correlations = {}
for param in params:
    corr, p_value = stats.pointbiserialr(df_analysis[param].dropna(),
                                         df_analysis.loc[df_analysis[param].dropna().index, 'Target'])
    correlations[param] = {'corr': corr, 'p': p_value}

sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]['corr']), reverse=True)

print(f"\n{'Parameter':>15s} | {'Correlation':>15s} | {'p-value':>12s} | {'Sig':>6s}")
print("-"*100)

for param, stats_dict in sorted_params:
    corr = stats_dict['corr']
    p_val = stats_dict['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"{param:>15s} | {corr:>15.6f} | {p_val:>12.6f} | {sig:>6s}")

# Section 4: Random Forest
print("\n" + "="*100)
print("SECTION 4: RANDOM FOREST FEATURE IMPORTANCE")
print("="*100)

X_clean = df_analysis[params].dropna()
y_clean = df_analysis.loc[X_clean.index, 'Target']

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_clean, y_clean)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\nFeature Importance Ranking:")
print("-"*80)
for i, idx in enumerate(indices):
    param_name = params[idx]
    importance = importances[idx]
    print(f"{i+1}. {param_name}: {importance:.4f} ({importance*100:.2f}%)")

cv_scores = cross_val_score(rf, X_clean, y_clean, cv=5, scoring='roc_auc')
print(f"\nModel Performance (5-fold CV):")
print(f"  Mean AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Section 5: PCA
print("\n" + "="*100)
print("SECTION 5: PRINCIPAL COMPONENT ANALYSIS")
print("="*100)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

pca = PCA()
pca.fit(X_scaled)

print(f"\nVariance Explained:")
print("-"*80)
cum_var = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cum_var += var
    print(f"PC{i+1}: {var*100:>6.2f}% | Cumulative: {cum_var:>6.2f}%")

# Section 6: Optimal Thresholds
print("\n" + "="*100)
print("SECTION 6: OPTIMAL DECISION THRESHOLDS")
print("="*100)

for param in ['量能比率', '价格vsEMA%']:
    data = df_analysis[param].dropna()
    target = df_analysis.loc[data.index, 'Target']

    unique_values = np.sort(data.unique())
    best_threshold = None
    best_f1 = -1

    for threshold in unique_values:
        corr = correlations[param]['corr']

        if corr > 0:
            predicted = (data >= threshold).astype(int)
        else:
            predicted = (data <= threshold).astype(int)

        tp = ((predicted == 1) & (target == 1)).sum()
        fp = ((predicted == 1) & (target == 0)).sum()
        fn = ((predicted == 0) & (target == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n{param}:")
    print(f"  Optimal threshold: {best_threshold:.6f}")
    print(f"  Best F1-Score: {best_f1:.4f}")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
