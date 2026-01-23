# -*- coding: utf-8 -*-
"""
第一步：收集半年数据并找到所有开仓信号
- 使用V7.0.7验证5逻辑（正确信号方向）
- 记录所有4H收盘价和信号
- 区分：原始信号 vs V7.0.5过滤后的开仓信号
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("第一步：收集半年数据和开仓信号")
print("=" * 80)
print()

# ==================== 配置 ====================
# 半年时间范围
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=180)

print(f"时间范围: {START_DATE.strftime('%Y-%m-%d')} 到 {END_DATE.strftime('%Y-%m-%d')}")
print()

# ==================== 获取BTC 4H数据 ====================
print("正在获取BTC 4H数据...")

def fetch_btc_4h_data(start_date, end_date):
    """从Binance获取BTC 4H数据"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        all_data = []

        current_start = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        while current_start < end_timestamp:
            params = {
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'startTime': current_start,
                'endTime': end_timestamp,
                'limit': 1000
            }

            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            current_start = data[-1][6] + 1

            if len(data) < 1000:
                break

        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        print(f"[ERROR] 获取数据失败: {e}")
        return None

df = fetch_btc_4h_data(START_DATE, END_DATE)

if df is None or len(df) < 100:
    print("[ERROR] 数据获取失败")
    exit(1)

print(f"[OK] 获取数据: {len(df)}条")
print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
print()

# ==================== 计算物理指标（验证5逻辑） ====================
print("正在计算物理指标...")

from scipy.signal import detrend, hilbert
from scipy.fft import fft, ifft

def calculate_physics_metrics(df):
    """计算物理指标：张力、加速度"""
    prices = df['close'].values

    # FFT + Hilbert
    d_prices = detrend(prices)
    coeffs = fft(d_prices)
    coeffs[8:] = 0
    filtered = ifft(coeffs).real

    analytic = hilbert(filtered)
    tension = np.imag(analytic)

    # 标准化
    if len(tension) > 1 and np.std(tension) > 0:
        tension_normalized = (tension - np.mean(tension)) / np.std(tension)
    else:
        tension_normalized = tension

    # 加速度
    acceleration = np.zeros_like(tension_normalized)
    for i in range(2, len(tension_normalized)):
        velocity = tension_normalized[i] - tension_normalized[i-1]
        prev_velocity = tension_normalized[i-1] - tension_normalized[i-2]
        acceleration[i] = velocity - prev_velocity

    return tension_normalized, acceleration

# 计算指标
tension, acceleration = calculate_physics_metrics(df)

# 添加到DataFrame
df_metrics = pd.DataFrame({
    'close': df['close'].values,
    'volume': df['volume'].values,
    'tension': tension,
    'acceleration': acceleration
}, index=df.index)

print(f"[OK] 计算完成: {len(df_metrics)}条")
print()

# ==================== 检测所有信号（验证5逻辑） ====================
print("正在检测信号...")

def detect_signal(tension, acceleration):
    """
    验证5逻辑：检测信号

    返回: (signal_type, confidence, description)
    """
    TENSION_THRESHOLD = 0.35
    ACCEL_THRESHOLD = 0.02

    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        return 'BEARISH_SINGULARITY', 0.7, f"奇点看空(T={tension:.2f}≥{TENSION_THRESHOLD})"

    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        return 'BULLISH_SINGULARITY', 0.6, f"奇点看涨(T={tension:.2f}≤-{TENSION_THRESHOLD})"

    elif tension > 0.3 and abs(acceleration) < 0.01:
        return 'HIGH_OSCILLATION', 0.6, f"高位震荡(T={tension:.2f}>0.3)"

    elif tension < -0.3 and abs(acceleration) < 0.01:
        return 'LOW_OSCILLATION', 0.6, f"低位震荡(T={tension:.2f}<-0.3)"

    return None, 0.0, "无信号"

# 收集所有信号
all_signals = []

for i in range(60, len(df_metrics)):
    current_time = df_metrics.index[i]
    current_price = df_metrics['close'].iloc[i]
    current_tension = df_metrics['tension'].iloc[i]
    current_accel = df_metrics['acceleration'].iloc[i]
    current_volume = df_metrics['volume'].iloc[i]

    # 计算量能比率
    avg_volume = df_metrics['volume'].iloc[i-20:i].mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    # 计算EMA偏离
    ema = pd.Series(df_metrics['close'].iloc[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
    price_vs_ema = (current_price - ema) / ema * 100

    # 检测信号
    signal_type, confidence, description = detect_signal(current_tension, current_accel)

    if signal_type is None:
        continue

    # V7.0.5过滤
    should_pass = True
    filter_reason = "通过"

    if signal_type == 'BULLISH_SINGULARITY':
        if volume_ratio > 0.95:
            should_pass = False
            filter_reason = f"量能放大({volume_ratio:.2f})"
        elif price_vs_ema > 0.05:
            should_pass = False
            filter_reason = f"主升浪(偏离{price_vs_ema:.1f}%)"

    # 记录信号
    all_signals.append({
        '时间': current_time,
        '收盘价': current_price,
        '信号类型': signal_type,
        '置信度': confidence,
        '描述': description,
        '张力': current_tension,
        '加速度': current_accel,
        '量能比率': volume_ratio,
        'EMA偏离%': price_vs_ema,
        'V705通过': should_pass,
        '过滤原因': filter_reason if not should_pass else ""
    })

print(f"[OK] 检测到信号: {len(all_signals)}个")
print()

# ==================== 统计 ====================
print("=" * 80)
print("信号统计")
print("=" * 80)

# 原始信号统计
for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
    total = sum(1 for s in all_signals if s['信号类型'] == sig_type)
    passed = sum(1 for s in all_signals if s['信号类型'] == sig_type and s['V705通过'])
    if total > 0:
        print(f"\n{sig_type}:")
        print(f"  原始信号: {total}个")
        print(f"  V705通过: {passed}个 ({passed/total*100:.1f}%)")
        print(f"  被过滤: {total-passed}个 ({(total-passed)/total*100:.1f}%)")

# 总体统计
total_signals = len(all_signals)
passed_signals = sum(1 for s in all_signals if s['V705通过'])
filtered_signals = total_signals - passed_signals

print(f"\n总计:")
print(f"  原始信号: {total_signals}个")
print(f"  V705通过（开仓信号）: {passed_signals}个 ({passed_signals/total_signals*100:.1f}%)")
print(f"  被过滤（普通信号）: {filtered_signals}个 ({filtered_signals/total_signals*100:.1f}%)")

# ==================== 保存数据 ====================
print("\n" + "=" * 80)
print("保存数据...")
print()

# 保存所有信号
df_signals = pd.DataFrame(all_signals)
df_signals.to_csv('step1_all_signals.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 所有信号已保存到: step1_all_signals.csv")

# 保存开仓信号
df_entry_signals = df_signals[df_signals['V705通过'] == True]
df_entry_signals.to_csv('step1_entry_signals.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 开仓信号已保存到: step1_entry_signals.csv ({len(df_entry_signals)}个)")

# 保存完整数据（包含指标）
df_metrics.to_csv('step1_full_data.csv', encoding='utf-8-sig')
print(f"[OK] 完整数据已保存到: step1_full_data.csv")

print("\n" + "=" * 80)
print("第一步完成！")
print("=" * 80)
print()
print("下一步：分析这些开仓信号的最优开仓和平仓时机")
