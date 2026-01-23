# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("统计学归纳分析 - ACTION vs HOLD")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_完整合并.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

print(f"\n数据集: {len(df)} 条信号")
print(f"ACTION: {len(df[df['黄金信号']=='ACTION'])} 条")
print(f"HOLD: {len(df[df['黄金信号']=='HOLD'])} 条")

# 创建目标变量
df['target'] = (df['黄金信号'] == 'ACTION').astype(int)

action_df = df[df['黄金信号'] == 'ACTION']
hold_df = df[df['黄金信号'] == 'HOLD']

features = ['量能比率', '价格vsEMA%', '张力', '加速度', 'DXY燃料']

# ============================================================================
# STEP 1: 描述性统计
# ============================================================================
print("\n" + "="*120)
print("STEP 1: 描述性统计")
print("="*120)

print(f"\n{'参数':<15} {'ACTION均值':<12} {'ACTION标准差':<14} {'HOLD均值':<12} {'HOLD标准差':<14} {'差异':<12}")
print("-" * 100)

for feat in features:
    if feat in df.columns:
        action_values = action_df[feat].dropna()
        hold_values = hold_df[feat].dropna()
        
        action_mean = action_values.mean()
        action_std = action_values.std()
        hold_mean = hold_values.mean()
        hold_std = hold_values.std()
        diff = action_mean - hold_mean
        
        print(f"{feat:<15} {action_mean:<12.4f} {action_std:<14.4f} {hold_mean:<12.4f} {hold_std:<14.4f} {diff:+<12.4f}")

# ============================================================================
# STEP 2: 统计显著性检验
# ============================================================================
print("\n" + "="*120)
print("STEP 2: 统计显著性检验 (独立样本t检验)")
print("="*120)

print(f"\n{'参数':<15} {'t统计量':<12} {'p值':<12} {'Cohen d':<12} {'效应量':<15} {'显著性':<10}")
print("-" * 100)

significant_features = []

for feat in features:
    if feat in df.columns:
        action_values = action_df[feat].dropna()
        hold_values = hold_df[feat].dropna()
        
        t_stat, p_value = stats.ttest_ind(action_values, hold_values)
        
        pooled_std = np.sqrt((action_values.std()**2 + hold_values.std()**2) / 2)
        cohens_d = (action_values.mean() - hold_values.mean()) / pooled_std if pooled_std > 0 else 0
        
        if abs(cohens_d) < 0.2:
            effect = "微小"
        elif abs(cohens_d) < 0.5:
            effect = "小"
        elif abs(cohens_d) < 0.8:
            effect = "中等"
        else:
            effect = "大"
        
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        if p_value < 0.05:
            significant_features.append(feat)
        
        print(f"{feat:<15} {t_stat:<12.4f} {p_value:<12.4f} {cohens_d:<12.4f} {effect:<15} {sig:<10}")

print(f"\n显著差异的参数 (p<0.05): {len(significant_features)} 个")
for feat in significant_features:
    print(f"  - {feat}")

# ============================================================================
# STEP 3: 相关性分析
# ============================================================================
print("\n" + "="*120)
print("STEP 3: 与ACTION的相关性分析")
print("="*120)

print(f"\n{'参数':<15} {'相关系数':<12} {'p值':<12} {'相关性强度':<15}")
print("-" * 70)

correlations = {}
for feat in features:
    if feat in df.columns:
        valid_data = df[[feat, 'target']].dropna()
        if len(valid_data) > 10:
            corr, p_value = stats.pointbiserialr(valid_data['target'], valid_data[feat])
            correlations[feat] = abs(corr)
            
            if abs(corr) < 0.1:
                strength = "极弱"
            elif abs(corr) < 0.3:
                strength = "弱"
            elif abs(corr) < 0.5:
                strength = "中等"
            elif abs(corr) < 0.7:
                strength = "强"
            else:
                strength = "极强"
            
            print(f"{feat:<15} {corr:<12.4f} {p_value:<12.4f} {strength:<15}")

# ============================================================================
# STEP 4: 最优阈值分析
# ============================================================================
print("\n" + "="*120)
print("STEP 4: 最优阈值分析")
print("="*120)

threshold_results = []

for feat in features:
    if feat in df.columns:
        valid_data = df[[feat, 'target']].dropna()
        
        if len(valid_data) > 50:
            X = valid_data[[feat]].values
            y = valid_data['target'].values
            
            fpr, tpr, thresholds = roc_curve(y, X)
            roc_auc = auc(fpr, tpr)
            
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            threshold_results.append({
                'feature': feat,
                'auc': roc_auc,
                'optimal_threshold': optimal_threshold,
                'tpr_at_opt': tpr[optimal_idx],
                'fpr_at_opt': fpr[optimal_idx]
            })

threshold_results.sort(key=lambda x: x['auc'], reverse=True)

print(f"\n{'参数':<15} {'AUC':<10} {'最优阈值':<15} {'真阳性率':<15} {'假阳性率':<15}")
print("-" * 90)

for res in threshold_results:
    print(f"{res['feature']:<15} {res['auc']:<10.4f} {res['optimal_threshold']:<15.4f} "
          f"{res['tpr_at_opt']:<15.4f} {res['fpr_at_opt']:<15.4f}")

# ============================================================================
# STEP 5: 机器学习预测
# ============================================================================
print("\n" + "="*120)
print("STEP 5: 机器学习预测模型")
print("="*120)

model_features = [f for f in features if f in df.columns]
model_data = df[model_features + ['target']].dropna()

X = model_data[model_features].values
y = model_data['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\n训练集: {len(y_train)} 样本")
print(f"测试集: {len(y_test)} 样本")

# 逻辑回归
print("\n【逻辑回归】")
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

y_prob = lr.predict_proba(X_test)[:, 1]
print(f"准确率: {lr.score(X_test, y_test):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

print(f"\n特征系数:")
for feat, coef in zip(model_features, lr.coef_[0]):
    print(f"  {feat:<15}: {coef:>8.4f}")

# 随机森林
print("\n【随机森林】")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"准确率: {rf.score(X_test, y_test):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

print(f"\n特征重要性:")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices):
    print(f"  {i+1}. {model_features[idx]:<15}: {importances[idx]:.4f}")

# 交叉验证
print("\n【交叉验证 (5-fold)】")
lr_scores = cross_val_score(LogisticRegression(random_state=42), X_scaled, y, cv=5, scoring='roc_auc')
print(f"逻辑回归 AUC: {lr_scores.mean():.4f} (+/- {lr_scores.std() * 2:.4f})")

print("\n" + "="*120)
print("统计分析完成")
print("="*120)
