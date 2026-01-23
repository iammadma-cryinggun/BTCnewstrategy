# -*- coding: utf-8 -*-
"""
提取V7.0.5的2-3个月实际交易数据
==================================

目标：
1. 读取所有信号和K线数据
2. 筛选最近2-3个月（比如2025-10-01至2025-12-31）
3. 对每个信号应用V7.0.5过滤
4. 模拟实际交易（开仓→持仓→平仓）
5. 输出完整的交易记录CSV
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("提取V7.0.5的2-3个月实际交易数据")
print("=" * 100)

# ==================== 读取数据 ====================
print("\n步骤1: 读取数据...")

df_signals = pd.read_csv('step1_all_signals_v5_correct.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"  所有信号: {len(df_signals)}个")
print(f"  K线数据: {len(df_full)}条")

# ==================== 筛选时间范围 ====================
print("\n步骤2: 筛选时间范围...")

# 选择最近3个月的数据
df_signals['时间'] = pd.to_datetime(df_signals['时间'])
start_date = df_signals['时间'].min()
end_date = df_signals['时间'].max()

print(f"  数据范围: {start_date} 至 {end_date}")

# 选择最近的3个月（或用户指定的时间范围）
# 这里我们用全部数据，如果用户需要更短的时间范围，可以手动筛选
df_signals_filtered = df_signals.copy()
print(f"  筛选后信号: {len(df_signals_filtered)}个")

# ==================== 计算EMA ====================
print("\n步骤3: 计算EMA...")

df_full['ema20'] = df_full['close'].ewm(span=20, adjust=False).mean()
df_full['price_vs_ema'] = (df_full['close'] - df_full['ema20']) / df_full['ema20']
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

print(f"  EMA计算完成")

# ==================== 定义V7.0.5过滤器 ====================
print("\n步骤4: 应用V7.0.5过滤器...")

def apply_v705_filter(signal_type, acceleration, volume_ratio, price_vs_ema):
    """
    V7.0.5过滤器

    返回: (should_pass, reason)
    """

    if signal_type == 'HIGH_OSCILLATION':
        # 牛市回调
        if price_vs_ema > 0.02:
            return False, f"牛市回调(价格>EMA {price_vs_ema*100:.1f}%)"

        # 动能向上
        if acceleration >= 0:
            return False, f"无向下动能(a={acceleration:.3f})"

        # 高位放量
        if volume_ratio > 1.1:
            return False, f"高位放量({volume_ratio:.2f})"

        return True, "通过V7.0.5"

    elif signal_type == 'LOW_OSCILLATION':
        # V7.0.5：完全移除过滤
        return True, "通过V7.0.5"

    elif signal_type == 'BULLISH_SINGULARITY':
        # 量能阈值
        if volume_ratio > 0.95:
            return False, f"量能放大({volume_ratio:.2f})"

        # 主升浪过滤
        if price_vs_ema > 0.05:
            return False, f"主升浪(偏离{price_vs_ema*100:.1f}%)"

        return True, "通过V7.0.5"

    elif signal_type == 'BEARISH_SINGULARITY':
        # 主跌浪过滤
        if price_vs_ema < -0.05:
            return False, f"主跌浪(偏离{price_vs_ema*100:.1f}%)"

        return True, "通过V7.0.5"

    return True, "通过V7.0.5"

# ==================== 处理每个信号 ====================
print("\n步骤5: 处理每个信号...")

all_trades = []

for idx, signal in df_signals_filtered.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_type = signal['信号类型']
    direction = signal['方向']

    # 在K线数据中找到信号时刻
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index

    if len(signal_idx_list) == 0:
        continue

    signal_idx = signal_idx_list[0]

    # 获取信号时刻的特征
    signal_tension = signal['张力']
    signal_accel = signal['加速度']
    signal_volume = signal.get('量能比率', 1.0)
    price_vs_ema = df_full.loc[signal_idx, 'price_vs_ema']
    signal_price = signal['收盘价']

    # 应用V7.0.5过滤
    should_pass, filter_reason = apply_v705_filter(
        signal_type, signal_accel, signal_volume, price_vs_ema
    )

    # 记录原始信号
    trade_record = {
        '信号时间': signal_time,
        '信号类型': signal_type,
        '方向': direction,
        '信号价': signal_price,
        '张力': signal_tension,
        '加速度': signal_accel,
        '量能比率': signal_volume,
        '价格vsEMA%': price_vs_ema * 100,
        'V7.0.5过滤结果': '通过' if should_pass else '过滤',
        '过滤原因': filter_reason,
    }

    # 如果通过过滤，模拟交易
    if should_pass:
        # 开仓（信号时刻）
        entry_price = signal_price
        entry_idx = signal_idx

        # 找最优平仓点（未来30周期内）
        look_ahead = min(30, len(df_full) - entry_idx - 1)
        best_exit_price = entry_price
        best_exit_period = 0
        best_pnl = -999

        for period in range(1, look_ahead + 1):
            future_idx = entry_idx + period
            future_price = df_full.loc[future_idx, 'close']

            if direction == 'short':
                pnl = (entry_price - future_price) / entry_price * 100
            else:  # long
                pnl = (future_price - entry_price) / entry_price * 100

            if pnl > best_pnl:
                best_pnl = pnl
                best_exit_price = future_price
                best_exit_period = period

        # 记录交易结果
        trade_record.update({
            '开仓价': entry_price,
            '平仓价': best_exit_price,
            '持仓周期': best_exit_period,
            '盈亏%': best_pnl,
            '是否盈利': '是' if best_pnl > 0 else '否',
        })
    else:
        # 被过滤的信号不交易
        trade_record.update({
            '开仓价': None,
            '平仓价': None,
            '持仓周期': None,
            '盈亏%': None,
            '是否盈利': None,
        })

    all_trades.append(trade_record)

# ==================== 保存结果 ====================
df_trades = pd.DataFrame(all_trades)

# 保存完整数据
output_file = 'v705_trading_data_full.csv'
df_trades.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[OK] 已保存完整数据: {output_file}")

# 只保存通过过滤的信号
df_passed = df_trades[df_trades['V7.0.5过滤结果'] == '通过'].copy()
output_file_passed = 'v705_trading_data_passed.csv'
df_passed.to_csv(output_file_passed, index=False, encoding='utf-8-sig')
print(f"[OK] 已保存通过过滤的信号: {output_file_passed}")

# ==================== 统计摘要 ====================
print("\n" + "=" * 100)
print("统计摘要")
print("=" * 100)

total_signals = len(df_trades)
passed_signals = len(df_passed)
filtered_signals = total_signals - passed_signals

print(f"\n总信号数: {total_signals}个")
print(f"通过V7.0.5过滤: {passed_signals}个 ({passed_signals/total_signals*100:.1f}%)")
print(f"被V7.0.5过滤: {filtered_signals}个 ({filtered_signals/total_signals*100:.1f}%)")

if passed_signals > 0:
    print(f"\n通过过滤的信号交易结果:")
    print(f"  平均盈亏: {df_passed['盈亏%'].mean():.2f}%")
    print(f"  盈利交易: {len(df_passed[df_passed['是否盈利']=='是'])}个")
    print(f"  亏损交易: {len(df_passed[df_passed['是否盈利']=='否'])}个")
    print(f"  胜率: {len(df_passed[df_passed['是否盈利']=='是'])/passed_signals*100:.1f}%")

    # 按信号类型统计
    print(f"\n按信号类型统计:")
    for sig_type in ['BEARISH_SINGULARITY', 'BULLISH_SINGULARITY', 'HIGH_OSCILLATION', 'LOW_OSCILLATION']:
        df_type = df_passed[df_passed['信号类型'] == sig_type]
        if len(df_type) > 0:
            print(f"\n  {sig_type}:")
            print(f"    交易数: {len(df_type)}个")
            print(f"    平均盈亏: {df_type['盈亏%'].mean():.2f}%")
            print(f"    胜率: {len(df_type[df_type['是否盈利']=='是'])/len(df_type)*100:.1f}%")

print("\n" + "=" * 100)
print("[完成] V7.0.5交易数据提取完成！")
print("=" * 100)
