# -*- coding: utf-8 -*-
"""
完全按照验证5的方法逐点计算张力和加速度
========================================

关键：每个时间点用前100个点计算，不是一次性计算全部！
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("完全按照验证5的方法逐点计算")
print("=" * 80)

# 读取数据
df = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

def calculate_tension_acceleration_v5(prices):
    """
    完全按照验证5Line 204-242的方法
    计算张力和加速度（速度的导数）
    """
    if len(prices) < 60:
        return None, None

    try:
        prices_array = np.array(prices, dtype=np.float64)
        d_prices = detrend(prices_array)

        # FFT滤波（保留更多动态）
        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real

        # Hilbert变换
        analytic = hilbert(filtered)
        tension = np.imag(analytic)  # 瞬时相位（张力）

        # 标准化
        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension

        # 计算加速度（张力的二阶差分）
        # 只计算最后一个点！
        current_tension = norm_tension[-1]
        prev_tension = norm_tension[-2] if len(norm_tension) > 1 else current_tension
        prev2_tension = norm_tension[-3] if len(norm_tension) > 2 else prev_tension

        # 速度 = 张力的一阶差分
        velocity = current_tension - prev_tension

        # 加速度 = 速度的一阶差分（张力的二阶差分）
        acceleration = velocity - (prev_tension - prev2_tension)

        return float(current_tension), float(acceleration)

    except:
        return None, None

# 逐点计算（从第100个点开始）
print("\n正在逐点计算张力和加速度...")

results = []
for i in range(100, len(df)):
    timestamp = df.loc[i, 'timestamp']
    close_price = df.loc[i, 'close']

    # 获取前100个点
    window_start = max(0, i - 99)
    window_end = i + 1
    prices_window = df.loc[window_start:window_end, 'close'].values

    # 计算张力和加速度
    tension, acceleration = calculate_tension_acceleration_v5(prices_window)

    if tension is not None and acceleration is not None:
        results.append({
            'timestamp': timestamp,
            'close': close_price,
            'tension': tension,
            'acceleration': acceleration,
            'volume': df.loc[i, 'volume']
        })

    if len(results) % 100 == 0:
        print(f"  进度: {len(results)}个数据点")

df_result = pd.DataFrame(results)

print(f"\n计算完成: {len(df_result)}个数据点")
print(f"张力范围: [{df_result['tension'].min():.4f}, {df_result['tension'].max():.4f}]")
print(f"加速度范围: [{df_result['acceleration'].min():.6f}, {df_result['acceleration'].max():.6f}]")
print(f"加速度标准差: {df_result['acceleration'].std():.6f}")

# 测试不同阈值下的奇点信号数量
print(f"\n测试不同ACCEL_THRESHOLD的奇点信号数量:")

TENSION_THRESHOLD = 0.35

for accel_thresh in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    bearish = sum((df_result['tension'] > TENSION_THRESHOLD) & (df_result['acceleration'] < -accel_thresh))
    bullish = sum((df_result['tension'] < -TENSION_THRESHOLD) & (df_result['acceleration'] > accel_thresh))
    total = bearish + bullish
    pct = total / len(df_result) * 100
    print(f"  ACCEL={accel_thresh}: 奇点看空={bearish:3d}, 奇点看涨={bullish:2d}, 总计={total:3d} ({pct:5.2f}%)")

# 保存结果
df_result.to_csv('step1_full_data_v5_rolling.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: step1_full_data_v5_rolling.csv")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
