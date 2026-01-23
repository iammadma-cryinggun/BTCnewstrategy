# -*- coding: utf-8 -*-
"""
完全从头计算 - 严格按照验证5.py逻辑
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("完全从头计算 - 严格按照验证5.py逻辑")
print("=" * 100)

# 配置
START_DATE = '2025-08-10'
END_DATE = '2026-01-20'
TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02
OSCILLATION_BAND = 0.5

print(f"\n时间范围: {START_DATE} 至 {END_DATE}")
print(f"参数: TENSION={TENSION_THRESHOLD}, ACCEL={ACCEL_THRESHOLD}\n")

# 1. 获取BTC OHLC数据
print("[步骤1] 获取BTC 4H OHLC数据...")
url = "https://api.binance.com/api/v3/klines"
params = {
    'symbol': 'BTCUSDT',
    'interval': '4h',
    'startTime': int(pd.Timestamp(START_DATE).timestamp() * 1000),
    'endTime': int(pd.Timestamp(END_DATE).timestamp() * 1000),
    'limit': 1000
}

resp = requests.get(url, params=params, timeout=15)
data = resp.json()

df = pd.DataFrame(data, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

df = df[['open', 'high', 'low', 'close', 'volume']]

print(f"  获取数据: {len(df)}条")
print(f"  时间范围: {df.index[0]} 至 {df.index[-1]}")

# 2. 获取DXY数据
print("\n[步骤2] 获取DXY数据...")
try:
    resp = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS", timeout=15)
    if resp.status_code == 200:
        dxy_df = pd.read_csv(StringIO(resp.text))
        dxy_df['observation_date'] = pd.to_datetime(dxy_df['observation_date'])
        dxy_df.set_index('observation_date', inplace=True)
        dxy_df.rename(columns={'DTWEXBGS': 'Close'}, inplace=True)
        dxy_df = dxy_df.dropna()
        dxy_df['Close'] = pd.to_numeric(dxy_df['Close'], errors='coerce')
        mask = (dxy_df.index >= START_DATE) & (dxy_df.index <= END_DATE)
        dxy_df = dxy_df[mask]
        print(f"  DXY数据: {len(dxy_df)}条")
    else:
        dxy_df = None
        print("  WARNING: DXY获取失败")
except Exception as e:
    dxy_df = None
    print(f"  WARNING: DXY获取失败 - {e}")

# 3. 计算物理指标（验证5逻辑）
print("\n[步骤3] 计算张力和加速度（验证5.py第204-242行）...")

def calculate_tension_acceleration(prices):
    if len(prices) < 60:
        return None, None

    try:
        prices_array = np.array(prices, dtype=np.float64)
        d_prices = detrend(prices_array)

        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real

        analytic = hilbert(filtered)
        tension = np.imag(analytic)

        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension

        # 计算加速度（张力的二阶差分）
        current_tension = norm_tension[-1]
        prev_tension = norm_tension[-2] if len(norm_tension) > 1 else current_tension
        prev2_tension = norm_tension[-3] if len(norm_tension) > 2 else prev_tension

        velocity = current_tension - prev_tension
        acceleration = velocity - (prev_tension - prev2_tension)

        return float(current_tension), float(acceleration)
    except:
        return None, None

# 为每个K线计算
print("  正在计算...")
tensions = []
accelerations = []
dxy_fuels = []

for i in range(len(df)):
    if i >= 100:
        window_prices = df['close'].iloc[i-100:i].values
    else:
        window_prices = df['close'].iloc[:i+1].values

    if len(window_prices) >= 60:
        tension, accel = calculate_tension_acceleration(window_prices)
        tensions.append(tension if tension is not None else np.nan)
        accelerations.append(accel if accel is not None else np.nan)
    else:
        tensions.append(np.nan)
        accelerations.append(np.nan)

    # DXY燃料
    if dxy_df is not None and not dxy_df.empty:
        try:
            current_date = df.index[i]
            mask = dxy_df.index <= current_date
            recent = dxy_df[mask].tail(5)
            if len(recent) >= 3:
                closes = recent['Close'].values.astype(float)
                change_1 = (closes[-1] - closes[-2]) / closes[-2]
                change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1
                accel_dxy = change_1 - change_2
                fuel = -accel_dxy * 100
                dxy_fuels.append(float(fuel))
            else:
                dxy_fuels.append(0.0)
        except:
            dxy_fuels.append(0.0)
    else:
        dxy_fuels.append(0.0)

df['tension'] = tensions
df['acceleration'] = accelerations
df['dxy_fuel'] = dxy_fuels

print(f"  计算完成")
print(f"  张力范围: [{df['tension'].min():.4f}, {df['tension'].max():.4f}]")
print(f"  加速度范围: [{df['acceleration'].min():.6f}, {df['acceleration'].max():.6f}]")

# 4. 检测信号
print("\n[步骤4] 检测信号（验证5.py第273-314行）...")

def diagnose_regime(tension, acceleration, dxy_fuel=0.0):
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        if dxy_fuel > 0.1:
            return "BEARISH_SINGULARITY", "强奇点看空", 0.9
        else:
            return "BEARISH_SINGULARITY", "奇点看空", 0.7

    if tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        if dxy_fuel > 0.2:
            return "BULLISH_SINGULARITY", "超强奇点看涨", 0.95
        elif dxy_fuel > 0:
            return "BULLISH_SINGULARITY", "强奇点看涨", 0.8
        else:
            return "BULLISH_SINGULARITY", "奇点看涨", 0.6

    if abs(tension) < OSCILLATION_BAND and abs(acceleration) < 0.02:
        return "OSCILLATION", "系统平衡", 0.8

    if tension > 0.3 and abs(acceleration) < 0.01:
        return "HIGH_OSCILLATION", "高位震荡", 0.6

    if tension < -0.3 and abs(acceleration) < 0.01:
        return "LOW_OSCILLATION", "低位震荡", 0.6

    if tension > 0 and acceleration > 0:
        return "TRANSITION_UP", "向上过渡", 0.4
    elif tension < 0 and acceleration < 0:
        return "TRANSITION_DOWN", "向下过渡", 0.4

    return None, 0.0, "无信号"

all_signals = []
for idx, row in df.iterrows():
    if pd.isna(row['tension']) or pd.isna(row['acceleration']):
        continue

    signal_type, description, confidence = diagnose_regime(
        row['tension'], row['acceleration'], row['dxy_fuel']
    )

    if signal_type is not None and confidence >= 0.6:
        all_signals.append({
            '时间': idx,
            '开盘价': row['open'],
            '最高价': row['high'],
            '最低价': row['low'],
            '收盘价': row['close'],
            '成交量': row['volume'],
            '信号类型': signal_type,
            '置信度': confidence,
            '描述': description,
            '张力': row['tension'],
            '加速度': row['acceleration'],
            'DXY燃料': row['dxy_fuel']
        })

df_signals = pd.DataFrame(all_signals)

print(f"  检测到信号: {len(df_signals)}个")
print(f"\n  信号类型分布:")
print(df_signals['信号类型'].value_counts())

# 5. 保存
print("\n[步骤5] 保存数据...")
df.to_csv('BTC_4H_完整数据_验证5逻辑.csv', encoding='utf-8-sig')
print(f"  [OK] K线数据: BTC_4H_完整数据_验证5逻辑.csv")

df_signals.to_csv('普通信号_验证5逻辑_完整计算.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] 普通信号: 普通信号_验证5逻辑_完整计算.csv ({len(df_signals)}个)")

print("\n" + "=" * 100)
print("[完成] 数据计算完成")
print("=" * 100)
