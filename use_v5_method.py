# -*- coding: utf-8 -*-
"""
完全按照验证5的方法计算张力和加速度
===================================
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("完全按照验证5的方法计算")
print("=" * 80)

# 读取数据
df = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

prices = df['close'].values

# 验证5的计算方法
d_prices = detrend(prices)
coeffs = fft(d_prices)
coeffs[8:] = 0
filtered = ifft(coeffs).real

analytic = hilbert(filtered)
tension = np.imag(analytic)

# 标准化（使用全部数据）
if len(tension) > 1 and np.std(tension) > 0:
    tension_normalized = (tension - np.mean(tension)) / np.std(tension)
else:
    tension_normalized = tension

# 加速度：张力的二阶差分
acceleration = np.zeros_like(tension_normalized)
for i in range(2, len(tension_normalized)):
    velocity = tension_normalized[i] - tension_normalized[i-1]
    prev_velocity = tension_normalized[i-1] - tension_normalized[i-2]
    acceleration[i] = velocity - prev_velocity

print(f"\n计算结果（验证5方法）:")
print(f"  张力范围: [{tension_normalized.min():.4f}, {tension_normalized.max():.4f}]")
print(f"  加速度范围: [{acceleration.min():.6f}, {acceleration.max():.6f}]")
print(f"  加速度标准差: {np.std(acceleration):.6f}")

# 保存正确计算的数据
df_metrics = pd.DataFrame({
    'timestamp': df['timestamp'],
    'close': df['close'].values,
    'volume': df['volume'].values,
    'tension': tension_normalized,
    'acceleration': acceleration
})

df_metrics.to_csv('step1_full_data_v5_method.csv', index=False, encoding='utf-8-sig')

# 测试不同阈值
print(f"\n测试不同ACCEL_THRESHOLD的奇点信号数量:")

for accel_thresh in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    bearish = sum((tension_normalized > 0.35) & (acceleration < -accel_thresh))
    bullish = sum((tension_normalized < -0.35) & (acceleration > accel_thresh))
    total = bearish + bullish
    pct = total / len(tension_normalized) * 100
    print(f"  ACCEL={accel_thresh}: 奇点看空={bearish:3d}, 奇点看涨={bullish:2d}, 总计={total:3d} ({pct:5.2f}%)")

print(f"\n已保存: step1_full_data_v5_method.csv")
print("请检查这个数据是否正确")
