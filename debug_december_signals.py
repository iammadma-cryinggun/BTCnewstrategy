# -*- coding: utf-8 -*-
"""
调试12月信号检测问题
"""

import pandas as pd
import numpy as np
from scipy.signal import detrend, hilbert
from scipy.fft import fft, ifft
import requests
from datetime import datetime

# 获取完整数据
start_ts = int(datetime(2025, 11, 20).timestamp() * 1000)  # 从11月20日开始，确保有足够的历史数据
end_ts = int(datetime(2026, 1, 20).timestamp() * 1000)

url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '4h',
    'startTime': start_ts,
    'endTime': end_ts,
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

print(f"K线数据: {len(df)}条")
print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
print("")

# 信号阈值
TENSION_THRESHOLD = 0.35
ACCEL_THRESHOLD = 0.02
CONF_THRESHOLD = 0.6

signals = []

# 检查每个4H K线
for i in range(60, len(df)):
    current_time = df.index[i]

    # 只关注12月1日-12月20日
    if current_time < pd.Timestamp('2025-12-01') or current_time > pd.Timestamp('2025-12-20'):
        continue

    current_price = df.iloc[i]['close']
    df_history = df.iloc[:i+1].copy()

    # 计算物理指标
    prices = df_history['close'].values
    d_prices = detrend(prices)
    coeffs = fft(d_prices)
    coeffs[8:] = 0
    filtered = ifft(coeffs).real
    analytic = hilbert(filtered)
    tension = np.imag(analytic)

    if len(tension) > 1 and np.std(tension) > 0:
        tension_normalized = (tension - np.mean(tension)) / np.std(tension)
    else:
        tension_normalized = tension

    acceleration = np.zeros_like(tension_normalized)
    for idx in range(2, len(tension_normalized)):
        current_tension = tension_normalized[idx]
        prev_tension = tension_normalized[idx-1]
        prev2_tension = tension_normalized[idx-2]
        velocity = current_tension - prev_tension
        acceleration[idx] = velocity - (prev_tension - prev2_tension)

    tension_val = tension_normalized[-1]
    accel_val = acceleration[-1]

    # 诊断信号（使用v707_trader_main.py的逻辑）
    signal_type = None
    confidence = 0.0
    description = ""

    if tension_val > TENSION_THRESHOLD and accel_val < -ACCEL_THRESHOLD:
        confidence = 0.7
        description = f"奇点看空(T={tension_val:.2f}>={TENSION_THRESHOLD})"
        signal_type = 'BEARISH_SINGULARITY'

    elif tension_val < -TENSION_THRESHOLD and accel_val > ACCEL_THRESHOLD:
        confidence = 0.6
        description = f"奇点看涨(T={tension_val:.2f}<={-TENSION_THRESHOLD})"
        signal_type = 'BULLISH_SINGULARITY'

    elif abs(tension_val) < 0.5 and abs(accel_val) < ACCEL_THRESHOLD:
        confidence = 0.8
        signal_type = 'OSCILLATION'
        description = f"系统平衡震荡(|T|={abs(tension_val):.2f}<0.5)"

    elif tension_val > 0.3 and abs(accel_val) < 0.01:
        confidence = 0.6
        signal_type = 'HIGH_OSCILLATION'
        description = f"高位震荡(T={tension_val:.2f}>0.3)"

    elif tension_val < -0.3 and abs(accel_val) < 0.01:
        confidence = 0.6
        signal_type = 'LOW_OSCILLATION'
        description = f"低位震荡(T={tension_val:.2f}<-0.3)"

    # 只记录置信度>=0.6的信号
    if signal_type is not None and confidence >= CONF_THRESHOLD:
        # 计算V7.0.5过滤器需要的参数
        avg_volume = df_history['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df_history.iloc[-1]['volume'] / avg_volume

        ema = pd.Series(prices).ewm(span=20, adjust=False).mean().iloc[-1]
        price_vs_ema = (current_price - ema) / ema

        # V7.0.5过滤
        should_pass = True
        filter_reason = "通过V7.0.5"

        if signal_type == 'BULLISH_SINGULARITY':
            if volume_ratio > 0.95:
                should_pass = False
                filter_reason = f"量能放大({volume_ratio:.2f})"
            elif price_vs_ema > 0.05:
                should_pass = False
                filter_reason = f"主升浪(偏离{price_vs_ema*100:.1f}%)"

        elif signal_type == 'OSCILLATION':
            should_pass = False
            filter_reason = "系统平衡震荡，无明显趋势"

        signals.append({
            'time': current_time,
            'price': current_price,
            'signal_type': signal_type,
            'confidence': confidence,
            'description': description,
            'tension': tension_val,
            'acceleration': accel_val,
            'volume_ratio': volume_ratio,
            'price_vs_ema': price_vs_ema * 100,
            'should_pass': should_pass,
            'filter_reason': filter_reason
        })

print(f"找到{len(signals)}个信号（置信度>=0.6）")
print("")

# 按日期分组
df_signals = pd.DataFrame(signals)
if len(df_signals) > 0:
    df_signals['date'] = pd.to_datetime(df_signals['time']).dt.date
    daily_counts = df_signals.groupby('date').size()

    print("按日期统计:")
    for date, count in daily_counts.items():
        print(f"  {date}: {count}个")
    print("")

    # 显示前30个信号
    print("前30个信号:")
    print("时间                  | 信号类型                   | 置信度 | 张力   | 加速度 | 量能比 | EMA偏离 | 通过过滤器")
    print("-" * 110)

    for i, sig in df_signals.head(30).iterrows():
        status = "✓" if sig['should_pass'] else "✗"
        print(f"{sig['time']} | {sig['signal_type']:25s} | {sig['confidence']:.1f}    | {sig['tension']:6.3f} | {sig['acceleration']:6.3f} | {sig['volume_ratio']:6.2f} | {sig['price_vs_ema']:7.2f}% | {status} {sig['filter_reason']}")

    # 统计
    print("")
    print("=== 总体统计 ===")
    print(f"总信号数: {len(df_signals)}")
    print(f"通过过滤器: {len(df_signals[df_signals['should_pass'] == True])}")
    print(f"被过滤: {len(df_signals[df_signals['should_pass'] == False])}")
    print("")
    print("=== 信号类型分布 ===")
    for sig_type, count in df_signals['signal_type'].value_counts().items():
        print(f"  {sig_type}: {count}个")
else:
    print("没有找到任何信号！")
