# -*- coding: utf-8 -*-
"""
V7.0.5黄金信号完整统计学分析
====================================

基于V7.0.5逻辑的完整统计分析：
1. 信号生成：验证5逻辑（含DXY燃料）
2. 入场过滤：V7.0.5过滤器
3. 出场策略：固定止盈止损+5%/-2.5%

统计学方法：
- p值显著性检验（t检验，p<0.05）
- Cohen's d效应量（>0.5为中等效应，>0.8为大效应）
- Youden Index（最优判别阈值）
- 95%置信区间
- 随机森林特征重要性

参考：C:\Users\Martin\Desktop\● 完美！统计学分析完成！让我以数学家的角度总结核心规律：.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V7.0.5黄金信号完整统计学分析 - 专业数学家方法")
print("=" * 80)

# ==================== 读取数据 ====================
df = pd.read_csv('step1_entry_signals.csv', encoding='utf-8')

# 重命名列（中文列名）
df.columns = ['时间', '价格', '信号类型', '张力', '加速度', '量能', '置信度',
              'EMA偏离%', 'V705通过', '过滤原因']

# 转换数据类型
df['时间'] = pd.to_datetime(df['时间'])
df['价格'] = df['价格'].astype(float)
df['张力'] = df['张力'].astype(float)
df['加速度'] = df['加速度'].astype(float)
df['量能'] = df['量能'].astype(float)
df['置信度'] = df['置信度'].astype(float)
df['EMA偏离%'] = df['EMA偏离%'].astype(float)

print(f"\n数据范围：{df['时间'].min()} 至 {df['时间'].max()}")
print(f"总信号数：{len(df)}个")
print(f"通过V7.0.5：{len(df[df['V705通过']==True])}个")

# 按信号类型统计
for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    sig_data = df[df['信号类型'] == sig_type]
    print(f"  {sig_type}: {len(sig_data)}个")

print("\n" + "=" * 80)
print("开始黄金信号统计学分析...")
print("=" * 80)

# 后续分析将在下一个脚本中完成
print("\n✓ 数据准备完成")
print("⚠️ 注意：由于V7.0.5使用固定止盈止损，不需要复杂的平仓策略分析")
print("   V7.0.5的黄金信号主要关注：入场时机（首次开仓 vs 等待确认）")
