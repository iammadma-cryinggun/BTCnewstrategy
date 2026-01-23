# -*- coding: utf-8 -*-
"""
分析黑天鹅信号的利用情况
"""
import pandas as pd

df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 计算下影线
df['下影线'] = df.apply(lambda row: (row['收盘价'] - row['最低价']) / row['收盘价']
                        if row['收盘价'] > row['最低价'] else 0, axis=1)

# 信号模式
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# 黑天鹅识别
df['是黑天鹅'] = (
    (df['信号模式'] == 'LONG_MODE') &
    (df['加速度'] <= -0.15) &
    (df['张力'] >= 0.60) &
    (df['下影线'] < 0.35) &
    (df['量能比率'] > 1.0)
)

print('所有黑天鹅信号：')
print('='*100)
bs_signals = df[df['是黑天鹅']][['时间', '收盘价', '加速度', '张力', '下影线', '量能比率']]

for idx, row in bs_signals.iterrows():
    print(f"{row['时间']} | 收盘价:${row['收盘价']:,.2f} | "
          f"加速度:{row['加速度']:.4f} | 张力:{row['张力']:.3f} | "
          f"下影线:{row['下影线']:.3f} | 量能:{row['量能比率']:.2f}")

# 读取交易记录
print("\n" + "="*100)
print("黑天鹅交易记录：")
print('='*100)

trades = pd.read_csv('混合策略V2_回测结果.csv', encoding='utf-8-sig')
bs_trades = trades[trades['is_black_swan']]

print(f"黑天鹅交易数: {len(bs_trades)}")
for idx, row in bs_trades.iterrows():
    print(f"\n入场: {row['entry_time']} @ ${row['entry_price']:,.2f}")
    print(f"离场: {row['exit_time']} @ ${row['exit_price']:,.2f}")
    print(f"盈亏: {row['pnl_pct']:+.2f}% (${row['pnl_usd']:+,.2f})")
    print(f"原因: {row['exit_reason']}")
    print(f"持仓: {row['hold_hours']:.0f}小时")
