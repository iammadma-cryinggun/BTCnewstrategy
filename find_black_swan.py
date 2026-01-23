# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 识别极值点
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

df['高低点'] = ''
for i in local_max_indices:
    df.loc[i, '高低点'] = '高点'
for i in local_min_indices:
    df.loc[i, '高低点'] = '低点'

def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

print('='*100)
print('黑天鹅信号搜索（调整后参数）')
print('='*100)
print()
print('条件:')
print('  加速度 <= -0.20')
print('  张力 >= 0.75  (放宽)')
print('  价格vsEMA% <= -1.5%')
print('  量能比率 > 1.0')
print('  高低点 == 低点')
print()

black_swan_conditions = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.20) &
    (df['张力'] >= 0.75) &
    (df['价格vsEMA%'] <= -1.5) &
    (df['量能比率'] > 1.0) &
    (df['高低点'] == '低点')
)

signals = df[black_swan_conditions]

print(f'找到信号: {len(signals)} 个')
print()

if len(signals) > 0:
    print(f"{'时间':<20} {'收盘价':<12} {'加速度':<12} {'张力':<10} {'乖离率':<12} {'量能':<10}")
    print('-' * 100)

    for idx, row in signals.iterrows():
        print(f"{str(row['时间'])[:18]:<20} "
              f"${row['收盘价']:>10.2f} "
              f"{row['加速度']:>10.4f} "
              f"{row['张力']:>8.3f} "
              f"{row['价格vsEMA%']:>10.2f}% "
              f"{row['量能比率']:>8.2f}")

        # 检查第二层防御
        if idx + 1 < len(df):
            next_high = df.loc[idx + 1, '最高价']
            signal_high = row['最高价']
            print(f"    第二层防御: 当前高点 ${signal_high:.2f} -> 下一根高点 ${next_high:.2f}")
            if next_high > signal_high:
                print(f'    结果: [OK] 确认！')

                # 模拟入场并计算后续表现
                entry_price = df.loc[idx + 1, '收盘价']
                entry_time = df.loc[idx + 1, '时间']

                # 计算未来10个周期的收益
                max_pnl = 0
                max_pnl_bar = 0
                final_pnl = 0

                for i in range(1, min(11, len(df) - idx - 1)):
                    future_close = df.loc[idx + 1 + i, '收盘价']
                    pnl = (future_close - entry_price) / entry_price * 100

                    if pnl > max_pnl:
                        max_pnl = pnl
                        max_pnl_bar = i

                    if i == 10:
                        final_pnl = pnl

                print(f'    入场: ${entry_price:.2f} ({entry_time})')
                print(f'    未来10周期最大收益: +{max_pnl:.2f}% (第{max_pnl_bar}周期)')
                print(f'    第10周期收益: {final_pnl:+.2f}%')

            else:
                print(f'    结果: [X] 拒绝！')
        print()
