import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("重新检查：真正触发平仓条件的交易")
print("="*100)

# 读取对比结果
df = pd.read_excel('完整对比_直接开仓vs确认开仓.xlsx', sheet_name='所有信号对比')

print(f"\n总信号数: {len(df)}")

# 检查策略A的平仓原因
strategy_a_exited = df[df['策略A_平仓原因'] != '超时'].copy()
strategy_a_timeout = df[df['策略A_平仓原因'] == '超时'].copy()

print(f"\n策略A（直接开仓）:")
print(f"  触发平仓条件: {len(strategy_a_exited)}笔")
print(f"  超时未平仓: {len(strategy_a_timeout)}笔")

if len(strategy_a_exited) > 0:
    print(f"\n  触发平仓的交易表现:")
    print(f"    总盈亏: {strategy_a_exited['策略A_直接开仓_盈亏%'].sum():.2f}%")
    print(f"    胜率: {(strategy_a_exited['策略A_直接开仓_盈亏%'] > 0).sum() / len(strategy_a_exited) * 100:.2f}%")
    print(f"    平均盈亏: {strategy_a_exited['策略A_直接开仓_盈亏%'].mean():.2f}%")
    print(f"    平均持仓: {strategy_a_exited['策略A_持仓周期'].mean():.1f}周期")

if len(strategy_a_timeout) > 0:
    print(f"\n  超时的交易表现:")
    print(f"    盈亏: {strategy_a_timeout['策略A_直接开仓_盈亏%'].sum():.2f}%")
    print(f"    胜率: {(strategy_a_timeout['策略A_直接开仓_盈亏%'] > 0).sum() / len(strategy_a_timeout) * 100:.2f}%")
    print(f"    平均盈亏: {strategy_a_timeout['策略A_直接开仓_盈亏%'].mean():.2f}%")

# 检查策略B的平仓原因
strategy_b_normal = df[df['策略B_平仓原因'] == '正常平仓'].copy()
strategy_b_timeout = df[df['策略B_平仓原因'] == '超时'].copy()

print(f"\n策略B（确认开仓）:")
print(f"  通过确认: {(df['策略B_确认开仓_找到确认'] == True).sum()}个")
print(f"  触发平仓条件: {len(strategy_b_normal)}笔")
print(f"  超时未平仓: {len(strategy_b_timeout)}笔")

if len(strategy_b_normal) > 0:
    print(f"\n  触发平仓的交易表现:")
    print(f"    总盈亏: {strategy_b_normal['策略B_盈亏%'].sum():.2f}%")
    print(f"    胜率: {(strategy_b_normal['策略B_盈亏%'] > 0).sum() / len(strategy_b_normal) * 100:.2f}%")
    print(f"    平均盈亏: {strategy_b_normal['策略B_盈亏%'].mean():.2f}%")
    print(f"    平均持仓: {strategy_b_normal['策略B_总持仓周期'].mean():.1f}周期")

if len(strategy_b_timeout) > 0:
    print(f"\n  超时的交易表现:")
    print(f"    盈亏: {strategy_b_timeout['策略B_盈亏%'].sum():.2f}%")
    print(f"    胜率: {(strategy_b_timeout['策略B_盈亏%'] > 0).sum() / len(strategy_b_timeout) * 100:.2f}%")
    print(f"    平均盈亏: {strategy_b_timeout['策略B_盈亏%'].mean():.2f}%")

print("\n" + "="*100)
print("关键发现")
print("="*100)
print(f"\n问题：平仓条件太严格！")
print(f"  策略A: 只有{len(strategy_a_exited)}/{len(df)} = {len(strategy_a_exited)/len(df)*100:.1f}%触发平仓")
print(f"  策略B: 只有{len(strategy_b_normal)}/{(df['策略B_确认开仓_找到确认'] == True).sum()} = {len(strategy_b_normal)/(df['策略B_确认开仓_找到确认'] == True).sum()*100:.1f}%触发平仓")

print(f"\n大部分交易都是'超时'，我用第10周期的临时盈亏来统计，这是错误的！")
print(f"真实的盈亏应该只看那些真正触发平仓条件的交易")

# 只统计真正触发平仓的交易
print("\n" + "="*100)
print("修正后的统计（只看真正平仓的交易）")
print("="*100)

if len(strategy_a_exited) > 0:
    print(f"\n策略A - 真正平仓的交易 ({len(strategy_a_exited)}笔):")
    print(f"  总盈亏: {strategy_a_exited['策略A_直接开仓_盈亏%'].sum():.2f}%")
    print(f"  胜率: {(strategy_a_exited['策略A_直接开仓_盈亏%'] > 0).sum() / len(strategy_a_exited) * 100:.2f}%")
    print(f"  平均盈亏: {strategy_a_exited['策略A_直接开仓_盈亏%'].mean():.2f}%")
    print(f"  最大盈利: {strategy_a_exited['策略A_直接开仓_盈亏%'].max():.2f}%")
    print(f"  最大亏损: {strategy_a_exited['策略A_直接开仓_盈亏%'].min():.2f}%")

if len(strategy_b_normal) > 0:
    print(f"\n策略B - 真正平仓的交易 ({len(strategy_b_normal)}笔):")
    print(f"  总盈亏: {strategy_b_normal['策略B_盈亏%'].sum():.2f}%")
    print(f"  胜率: {(strategy_b_normal['策略B_盈亏%'] > 0).sum() / len(strategy_b_normal) * 100:.2f}%")
    print(f"  平均盈亏: {strategy_b_normal['策略B_盈亏%'].mean():.2f}%")
    print(f"  最大盈利: {strategy_b_normal['策略B_盈亏%'].max():.2f}%")
    print(f"  最大亏损: {strategy_b_normal['策略B_盈亏%'].min():.2f}%")
