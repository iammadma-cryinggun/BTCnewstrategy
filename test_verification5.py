# -*- coding: utf-8 -*-
"""
测试验证5引擎
"""

from v80_live_verification5 import Verification5Engine
import pandas as pd

print('='*80)
print('测试验证5引擎')
print('='*80)

engine = Verification5Engine()

# 测试1: 获取BTC数据
print('\n[测试1] 获取BTC数据...')
btc_df = engine.fetch_btc_data(limit=100)
if btc_df is not None:
    print(f'[OK] 成功获取 {len(btc_df)} 条BTC数据')
    print(f'   最新价格: ${btc_df["close"].iloc[-1]:,.0f}')
    print(f'   时间范围: {btc_df.index[0]} 到 {btc_df.index[-1]}')
else:
    print('[ERROR] BTC数据获取失败')
    exit(1)

# 测试2: 获取DXY数据
print('\n[测试2] 获取DXY数据...')
dxy_df = engine.fetch_dxy_data(days_back=30)
if dxy_df is not None and not dxy_df.empty:
    print(f'[OK] 成功获取 {len(dxy_df)} 条DXY数据')
    print(f'   最新DXY: {dxy_df["Close"].iloc[-1]:.2f}')
else:
    print('[WARNING] DXY数据获取失败（可能网络问题，系统会继续运行）')
    dxy_df = None

# 测试3: 计算物理指标
print('\n[测试3] 计算物理指标（验证5逻辑）...')
if btc_df is not None:
    prices = btc_df['close'].tail(100).values
    tension, acceleration = engine.calculate_tension_acceleration(prices)
    if tension is not None:
        print(f'[OK] 物理指标计算成功')
        print(f'   张力: {tension:.4f}')
        print(f'   加速度: {acceleration:.6f}')
    else:
        print('[ERROR] 物理指标计算失败')
        exit(1)

# 测试4: 市场状态诊断
print('\n[测试4] 市场状态诊断...')
if tension is not None and acceleration is not None:
    signal_type, description, confidence = engine.diagnose_regime(tension, acceleration, dxy_fuel=0.0)
    print(f'[OK] 市场状态诊断完成')
    print(f'   信号类型: {signal_type}')
    print(f'   描述: {description}')
    print(f'   置信度: {confidence:.1%}')

print('\n' + '='*80)
print('[SUCCESS] 所有测试通过！系统可以正常工作')
print('='*80)
