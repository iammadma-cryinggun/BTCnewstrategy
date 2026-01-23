# -*- coding: utf-8 -*-
"""
分析每个开仓信号的特征和最终结果
===============================

目标：
1. 列出所有开仓交易
2. 显示开仓时的特征（张力、加速度、量能）
3. 显示最终盈亏
4. 找出好交易和坏交易的特征差异
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("所有开仓交易的详细分析")
print("=" * 100)

# 读取数据
df = pd.read_excel('trading_execution_log.xlsx', engine='openpyxl')

# 提取所有交易
trades = []
current_trade = None

for idx, row in df.iterrows():
    if row['交易状态'] == '开仓':
        # 新交易开始
        current_trade = {
            '开仓索引': idx,
            '开仓时间': row['timestamp'],
            '信号类型': row['信号类型'],
            '方向': row['持仓方向'],
            '开仓价': row['close'],
            '开仓张力': row['tension'],
            '开仓加速度': row['acceleration'],
            '开仓量能': row['volume_ratio']
        }
    elif current_trade is not None and row['交易状态'] == '平仓':
        # 交易结束
        current_trade['平仓索引'] = idx
        current_trade['平仓时间'] = row['timestamp']
        current_trade['平仓价'] = row['close']
        current_trade['盈亏%'] = row['盈亏%']
        current_trade['平仓原因'] = row['备注']
        current_trade['持仓周期'] = idx - current_trade['开仓索引']
        trades.append(current_trade)
        current_trade = None

df_trades = pd.DataFrame(trades)

print(f"\n总交易数: {len(df_trades)}笔")
print(f"\n盈利交易: {len(df_trades[df_trades['盈亏%'] > 0])}笔")
print(f"亏损交易: {len(df_trades[df_trades['盈亏%'] < 0])}笔")
print(f"平均盈亏: {df_trades['盈亏%'].mean():.2f}%")

# ==================== 显示所有交易 ====================
print("\n" + "=" * 100)
print("所有交易明细")
print("=" * 100)

for idx, trade in df_trades.iterrows():
    profit_mark = "[WIN]" if trade['盈亏%'] > 0 else "[LOSS]"
    print(f"\n交易 #{idx+1} {profit_mark}")
    print(f"  开仓时间: {trade['开仓时间']}")
    print(f"  信号类型: {trade['信号类型']}")
    print(f"  方向: {trade['方向'].upper()}")
    print(f"  开仓价: {trade['开仓价']:.2f}")
    print(f"  开仓张力: {trade['开仓张力']:.4f}")
    print(f"  开仓加速度: {trade['开仓加速度']:.4f}")
    print(f"  开仓量能: {trade['开仓量能']:.2f}")
    print(f"  持仓周期: {trade['持仓周期']}周期")
    print(f"  平仓时间: {trade['平仓时间']}")
    print(f"  平仓价: {trade['平仓价']:.2f}")
    print(f"  盈亏: {trade['盈亏%']:.2f}%")
    print(f"  平仓原因: {trade['平仓原因']}")

# ==================== 统计分析：好交易 vs 坏交易 ====================
print("\n" + "=" * 100)
print("统计分析：好交易 vs 坏交易（开仓时特征）")
print("=" * 100)

pnl_median = df_trades['盈亏%'].median()
df_trades['是好交易'] = df_trades['盈亏%'] >= pnl_median

good_trades = df_trades[df_trades['是好交易'] == True]
bad_trades = df_trades[df_trades['是好交易'] == False]

print(f"\n好交易（盈亏>={pnl_median:.2f}%）: {len(good_trades)}笔, 平均盈亏 {good_trades['盈亏%'].mean():.2f}%")
print(f"坏交易（盈亏<{pnl_median:.2f}%）: {len(bad_trades)}笔, 平均盈亏 {bad_trades['盈亏%'].mean():.2f}%")

# 按方向分析
for direction in ['long', 'short']:
    df_dir = df_trades[df_trades['方向'] == direction]

    if len(df_dir) == 0:
        continue

    pnl_median_dir = df_dir['盈亏%'].median()
    df_dir['是好交易'] = df_dir['盈亏%'] >= pnl_median_dir

    good_dir = df_dir[df_dir['是好交易'] == True]
    bad_dir = df_dir[df_dir['是好交易'] == False]

    print(f"\n{'=' * 100}")
    print(f"{direction.upper()}交易 - 开仓时特征分析")
    print(f"{'=' * 100}")
    print(f"好交易: {len(good_dir)}笔, 平均盈亏 {good_dir['盈亏%'].mean():.2f}%")
    print(f"坏交易: {len(bad_dir)}笔, 平均盈亏 {bad_dir['盈亏%'].mean():.2f}%")

    print(f"\n{'特征':<15} {'好交易均值':<15} {'坏交易均值':<15} {'Cohen''s d':<12} {'p值':<12} {'显著性'}")
    print(f"{'-' * 100}")

    for col in ['开仓张力', '开仓加速度', '开仓量能']:
        good_mean = good_dir[col].mean()
        bad_mean = bad_dir[col].mean()
        t_stat, p_val = stats.ttest_ind(good_dir[col], bad_dir[col], nan_policy='omit')

        n1, n2 = len(good_dir), len(bad_dir)
        var1, var2 = good_dir[col].var(), bad_dir[col].var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (good_dir[col].mean() - bad_dir[col].mean()) / pooled_std if pooled_std > 0 else 0

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        effect = '超大' if abs(cohens_d) > 1.2 else '大' if abs(cohens_d) > 0.8 else '中等' if abs(cohens_d) > 0.5 else '小'

        print(f"{col:<15} {good_mean:<15.4f} {bad_mean:<15.4f} {cohens_d:<12.3f} {p_val:<12.4f} {sig} {effect}")

        # 如果p<0.1，显示更多细节
        if p_val < 0.1:
            print(f"    → 好交易{col} {'更高' if good_mean > bad_mean else '更低'}: {good_mean:.4f} vs {bad_mean:.4f}")

# ==================== 保存到CSV ====================
df_trades.to_csv('all_trades_detail.csv', index=False, encoding='utf-8-sig')
print(f"\n[OK] 已保存: all_trades_detail.csv")

print("\n" + "=" * 100)
print("[完成] 所有开仓交易分析完成！")
print("=" * 100)
