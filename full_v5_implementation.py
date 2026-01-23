# -*- coding: utf-8 -*-
"""
完整实现验证5的全部计算步骤
=============================

包括：
1. FFT低通滤波
2. Hilbert变换
3. 标准化（每个窗口单独标准化）
4. 加速度计算（二阶差分）
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("完整实现验证5的全部计算步骤")
print("=" * 80)

# 读取数据
df = pd.read_csv('step1_full_data_with_dxy.csv', encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\n原始数据: {len(df)}条")

# ==================== 完整计算流程 ====================
print("\n正在逐点计算（每个窗口100个点）...")

results = []

# 从第100个点开始计算（需要前100个点作为窗口）
for i in range(100, len(df)):
    if i % 200 == 0:
        print(f"  进度: {i}/{len(df)}")

    timestamp = df.loc[i, 'timestamp']
    close_price = df.loc[i, 'close']
    volume = df.loc[i, 'volume']

    # 获取前100个点的窗口
    window_start = i - 99  # 包含当前点，共100个点
    window_end = i + 1
    prices_window = df.loc[window_start:window_end, 'close'].values

    # ========== 步骤1: 去趋势 ==========
    try:
        prices_array = np.array(prices_window, dtype=np.float64)
        d_prices = detrend(prices_array)

        # ========== 步骤2: FFT低通滤波 ==========
        # 保留前8个系数，其余置0
        coeffs = fft(d_prices)
        coeffs[8:] = 0  # 低通滤波，只保留低频成分
        filtered = ifft(coeffs).real

        # ========== 步骤3: Hilbert变换 ==========
        # 得到解析信号，虚部就是瞬时相位
        analytic = hilbert(filtered)
        tension = np.imag(analytic)  # 瞬时相位（张力）

        # ========== 步骤4: 标准化（关键！）==========
        # 对这100个点的张力进行标准化
        if len(tension) > 1 and np.std(tension) > 0:
            tension_mean = np.mean(tension)
            tension_std = np.std(tension)
            norm_tension = (tension - tension_mean) / tension_std
        else:
            norm_tension = tension

        # ========== 步骤5: 计算加速度（二阶差分）==========
        # 只计算最后一个点（当前点）的加速度
        if len(norm_tension) >= 3:
            current_tension = norm_tension[-1]
            prev_tension = norm_tension[-2]
            prev2_tension = norm_tension[-3]

            # 速度 = 张力的一阶差分
            velocity_current = current_tension - prev_tension
            velocity_prev = prev_tension - prev2_tension

            # 加速度 = 速度的一阶差分（张力的二阶差分）
            acceleration = velocity_current - velocity_prev

            results.append({
                'timestamp': timestamp,
                'close': close_price,
                'volume': volume,
                'tension': float(current_tension),
                'acceleration': float(acceleration)
            })
        else:
            # 数据不足，跳过
            continue

    except Exception as e:
        # 出错跳过
        continue

df_result = pd.DataFrame(results)

print(f"\n计算完成: {len(df_result)}个数据点")
print(f"\n统计结果:")
print(f"  张力范围: [{df_result['tension'].min():.4f}, {df_result['tension'].max():.4f}]")
print(f"  张力均值: {df_result['tension'].mean():.4f}")
print(f"  张力标准差: {df_result['tension'].std():.4f}")
print(f"  加速度范围: [{df_result['acceleration'].min():.6f}, {df_result['acceleration'].max():.6f}]")
print(f"  加速度均值: {df_result['acceleration'].mean():.6f}")
print(f"  加速度标准差: {df_result['acceleration'].std():.6f}")

# ==================== 测试验证5的阈值 ====================
print(f"\n" + "=" * 80)
print("使用验证5的阈值测试信号数量")
print("=" * 80)

TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02
OSCILLATION_BAND = 0.5

# 统计各类信号数量
bearish_sig = sum((df_result['tension'] > TENSION_THRESHOLD) & (df_result['acceleration'] < -ACCEL_THRESHOLD))
bullish_sig = sum((df_result['tension'] < -TENSION_THRESHOLD) & (df_result['acceleration'] > ACCEL_THRESHOLD))
oscillation = sum((np.abs(df_result['tension']) < OSCILLATION_BAND) & (np.abs(df_result['acceleration']) < 0.02))
high_osc = sum((df_result['tension'] > 0.3) & (np.abs(df_result['acceleration']) < 0.01))
low_osc = sum((df_result['tension'] < -0.3) & (np.abs(df_result['acceleration']) < 0.01))

print(f"\n验证5的信号类型统计:")
print(f"  BEARISH_SINGULARITY（奇点看空）: {bearish_sig}个 ({bearish_sig/len(df_result)*100:.2f}%)")
print(f"  BULLISH_SINGULARITY（奇点看涨）: {bullish_sig}个 ({bullish_sig/len(df_result)*100:.2f}%)")
print(f"  OSCILLATION（平衡震荡）: {oscillation}个 ({oscillation/len(df_result)*100:.2f}%)")
print(f"  HIGH_OSCILLATION（高位震荡）: {high_osc}个 ({high_osc/len(df_result)*100:.2f}%)")
print(f"  LOW_OSCILLATION（低位震荡）: {low_osc}个 ({low_osc/len(df_result)*100:.2f}%)")

# 测试不同阈值
print(f"\n不同ACCEL_THRESHOLD下的奇点信号总数:")
for accel_thresh in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
    bearish = sum((df_result['tension'] > TENSION_THRESHOLD) & (df_result['acceleration'] < -accel_thresh))
    bullish = sum((df_result['tension'] < -TENSION_THRESHOLD) & (df_result['acceleration'] > accel_thresh))
    total = bearish + bullish
    pct = total / len(df_result) * 100
    print(f"  ACCEL={accel_thresh}: 奇点总计={total:3d} ({pct:5.2f}%) [看空={bearish:3d}, 看涨={bullish:2d}]")

# 保存结果
df_result.to_csv('step1_full_data_v5_complete.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存: step1_full_data_v5_complete.csv")

print("\n" + "=" * 80)
print("完整计算完成！")
print("=" * 80)
