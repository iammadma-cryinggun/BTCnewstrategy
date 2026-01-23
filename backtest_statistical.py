# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

print("="*120)
print("基于统计学的策略回测验证")
print("="*120)

# 加载数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"\n数据集: {len(df)} 条4小时K线")

# 识别局部极值点
order = 2
local_max_indices = argrelextrema(df['收盘价'].values, np.greater, order=order)[0]
local_min_indices = argrelextrema(df['收盘价'].values, np.less, order=order)[0]

peak_valley_labels = []
for i in range(len(df)):
    if i in local_max_indices:
        peak_valley_labels.append('高点')
    elif i in local_min_indices:
        peak_valley_labels.append('低点')
    else:
        peak_valley_labels.append('')

df['高低点'] = peak_valley_labels

# 手动修正
rows = df[df['时间'].dt.strftime('%Y-%m-%d %H:%M') == '2025-08-22 04:00']
if len(rows) > 0:
    idx = rows.index[0]
    if df.loc[idx, '高低点'] == '':
        df.loc[idx, '高低点'] = '高点'

# 定义信号模式
def get_signal_mode(signal_type):
    if signal_type in ['BEARISH_SINGULARITY', 'LOW_OSCILLATION']:
        return 'LONG_MODE'
    elif signal_type in ['BULLISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return 'SHORT_MODE'
    else:
        return 'NO_TRADE'

df['信号模式'] = df['信号类型'].apply(get_signal_mode)

# 回测函数
def run_backtest(filter_conditions, strategy_name):
    INITIAL_CAPITAL = 10000
    POSITION_SIZE_PCT = 0.03
    STOP_LOSS_PCT = 0.02
    COMMISSION_PCT = 0.0005

    trades = []
    capital = INITIAL_CAPITAL
    current_position = 'NONE'
    entry_price = None
    entry_idx = None
    position_size = 0
    prev_signal_mode = None

    for i in range(len(df)):
        signal_type = df.loc[i, '信号类型']
        signal_mode = df.loc[i, '信号模式']
        current_close = df.loc[i, '收盘价']
        is_peak = (df.loc[i, '高低点'] == '高点')
        is_valley = (df.loc[i, '高低点'] == '低点')

        tension = df.loc[i, '张力']
        acceleration = df.loc[i, '加速度']
        volume_ratio = df.loc[i, '量能比率']

        # 应用过滤条件
        skip_trade = False
        if filter_conditions:
            if 'tension' in filter_conditions:
                t_min = filter_conditions['tension'].get('min', -999)
                t_max = filter_conditions['tension'].get('max', 999)
                if not (t_min <= tension <= t_max):
                    skip_trade = True

            if 'acceleration' in filter_conditions:
                a_min = filter_conditions['acceleration'].get('min', -999)
                a_max = filter_conditions['acceleration'].get('max', 999)
                if not (a_min <= acceleration <= a_max):
                    skip_trade = True

            if 'volume_ratio' in filter_conditions:
                v_min = filter_conditions['volume_ratio'].get('min', 0)
                v_max = filter_conditions['volume_ratio'].get('max', 999)
                if not (v_min <= volume_ratio <= v_max):
                    skip_trade = True

        is_new_signal = (prev_signal_mode != signal_mode) and (signal_mode != 'NO_TRADE')

        # 平仓逻辑
        if current_position == 'LONG':
            unrealized_pnl = (current_close - entry_price) / entry_price

            if unrealized_pnl <= -STOP_LOSS_PCT:
                pnl = (current_close - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl
                trades.append({'type': 'LONG', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': current_close, 'pnl_pct': (current_close - entry_price) / entry_price * 100, 'pnl_usd': pnl, 'exit_reason': 'stop', 'hold_bars': i - entry_idx})
                current_position = 'NONE'
                entry_price = None
                position_size = 0
            elif is_peak or (signal_mode in ['SHORT_MODE', 'NO_TRADE']):
                pnl = (current_close - entry_price) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl
                reason = 'peak' if is_peak else 'switch'
                trades.append({'type': 'LONG', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': current_close, 'pnl_pct': (current_close - entry_price) / entry_price * 100, 'pnl_usd': pnl, 'exit_reason': reason, 'hold_bars': i - entry_idx})
                current_position = 'NONE'
                entry_price = None
                position_size = 0

        elif current_position == 'SHORT':
            unrealized_pnl = (entry_price - current_close) / entry_price

            if unrealized_pnl <= -STOP_LOSS_PCT:
                pnl = (entry_price - current_close) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl
                trades.append({'type': 'SHORT', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': current_close, 'pnl_pct': (entry_price - current_close) / entry_price * 100, 'pnl_usd': pnl, 'exit_reason': 'stop', 'hold_bars': i - entry_idx})
                current_position = 'NONE'
                entry_price = None
                position_size = 0
            elif is_valley or (signal_mode in ['LONG_MODE', 'NO_TRADE']):
                pnl = (entry_price - current_close) / entry_price * position_size
                pnl -= position_size * COMMISSION_PCT
                capital += pnl
                reason = 'valley' if is_valley else 'switch'
                trades.append({'type': 'SHORT', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': current_close, 'pnl_pct': (entry_price - current_close) / entry_price * 100, 'pnl_usd': pnl, 'exit_reason': reason, 'hold_bars': i - entry_idx})
                current_position = 'NONE'
                entry_price = None
                position_size = 0

        # 开仓逻辑
        if current_position == 'NONE' and not skip_trade:
            if signal_mode == 'LONG_MODE':
                if is_new_signal or is_valley:
                    current_position = 'LONG'
                    entry_price = current_close
                    entry_idx = i
                    position_size = capital * POSITION_SIZE_PCT
            elif signal_mode == 'SHORT_MODE':
                if is_new_signal or is_peak:
                    current_position = 'SHORT'
                    entry_price = current_close
                    entry_idx = i
                    position_size = capital * POSITION_SIZE_PCT

        prev_signal_mode = signal_mode

    # 计算结果
    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {'name': strategy_name, 'trades': 0, 'return': 0, 'win_rate': 0, 'dd': 0}

    total_pnl = trades_df['pnl_usd'].sum()
    final_capital = capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    win_rate = len(winning_trades) / len(trades_df) * 100

    # 最大回撤
    capital_curve = [INITIAL_CAPITAL] + trades_df['pnl_usd'].cumsum().add(INITIAL_CAPITAL).tolist()
    peak = capital_curve[0]
    max_drawdown = 0
    for cap in capital_curve:
        if cap > peak:
            peak = cap
        drawdown = (peak - cap) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    long_count = len(trades_df[trades_df['type'] == 'LONG'])
    short_count = len(trades_df[trades_df['type'] == 'SHORT'])

    return {
        'name': strategy_name,
        'trades': len(trades_df),
        'return': total_return,
        'win_rate': win_rate,
        'avg_win': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
        'dd': max_drawdown,
        'long': long_count,
        'short': short_count,
        'trades_df': trades_df
    }

# 测试不同的过滤条件
print("\n" + "="*120)
print("测试不同过滤条件组合")
print("="*120)

strategies = [
    ('1-无过滤', None),
    ('2-张力<0-量能>1.0', {'tension': {'min': -999, 'max': 0}, 'volume_ratio': {'min': 1.0, 'max': 999}}),
    ('3-张力<0-加速度>0', {'tension': {'min': -999, 'max': 0}, 'acceleration': {'min': 0, 'max': 999}}),
    ('4-只张力<0', {'tension': {'min': -999, 'max': 0}}),
    ('5-只加速度>0', {'acceleration': {'min': 0, 'max': 999}}),
    ('6-只量能>1.0', {'volume_ratio': {'min': 1.0, 'max': 999}}),
    ('7-加速度>0.02', {'acceleration': {'min': 0.02, 'max': 999}}),
]

results = []
for name, conditions in strategies:
    result = run_backtest(conditions, name)
    results.append(result)

# 对比结果
print("\n" + "="*120)
print("回测结果对比")
print("="*120)

print(f"\n{'策略':<20} {'交易数':<8} {'收益率':<12} {'胜率':<10} {'回撤':<10} {'做多':<8} {'做空':<8}")
print("-" * 100)

results_sorted = sorted(results, key=lambda x: x['return'], reverse=True)

for res in results_sorted:
    print(f"{res['name']:<20} {res['trades']:<8} {res['return']:+<12.2f}% {res['win_rate']:<10.1f}% {res['dd']:<10.2f}% {res['long']:<8} {res['short']:<8}")

# 最优策略
best = results_sorted[0]
print("\n" + "="*120)
print(f"最优策略: {best['name']}")
print("="*120)
print(f"收益率: {best['return']:+.2f}%")
print(f"交易次数: {best['trades']}")
print(f"胜率: {best['win_rate']:.1f}%")
print(f"平均盈利: {best['avg_win']:+.2f}%")
print(f"平均亏损: {best['avg_loss']:+.2f}%")
print(f"最大回撤: {best['dd']:.2f}%")
print(f"做多/做空: {best['long']}/{best['short']}")

print("\n" + "="*120)
print("回测验证完成")
print("="*120)
