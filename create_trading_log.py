# -*- coding: utf-8 -*-
"""
创建实际交易执行日志（一行行标注）
==================================

目标：
1. 把每个K线周期都标注清楚
2. 显示信号出现、开仓、持仓、平仓的全过程
3. 生成Excel文件，可以一行行查看交易执行
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("创建实际交易执行日志")
print("=" * 100)

# 读取数据
df_signals = pd.read_csv('step1_all_signals_v5_correct.csv', encoding='utf-8-sig')
df_full = pd.read_csv('step1_full_data_v5_complete.csv', encoding='utf-8-sig')
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 计算额外特征
df_full['avg_volume_20'] = df_full['volume'].rolling(20).mean()
df_full['volume_ratio'] = df_full['volume'] / df_full['avg_volume_20']

print(f"\nK线数据: {len(df_full)}条")
print(f"信号数据: {len(df_signals)}个")

# ==================== 创建交易日志 ====================
print("\n正在创建交易日志...")

# 创建交易日志DataFrame
trading_log = df_full.copy()
trading_log['信号类型'] = None
trading_log['交易状态'] = '观察中'  # 观察中, 持仓中, 已平仓
trading_log['持仓方向'] = None  # long, short
trading_log['开仓价'] = None
trading_log['当前价'] = trading_log['close']
trading_log['盈亏%'] = 0.0
trading_log['备注'] = None

# 标注所有信号位置
signal_positions = {}
for idx, signal in df_signals.iterrows():
    signal_time = pd.to_datetime(signal['时间'])
    signal_idx_list = df_full[df_full['timestamp'] == signal_time].index
    if len(signal_idx_list) > 0:
        signal_idx = signal_idx_list[0]
        signal_positions[signal_idx] = {
            'type': signal['信号类型'],
            'direction': signal['方向'],
            'tension': signal['张力'],
            'accel': signal['加速度'],
            'volume_ratio': signal.get('量能比率', 1.0)
        }

        # 标注信号
        trading_log.loc[signal_idx, '信号类型'] = signal['信号类型']

# 执行交易逻辑
current_position = None  # {'entry_idx': int, 'direction': 'long/short', 'entry_price': float, 'signal_type': str}

for idx in range(len(trading_log)):
    if idx in signal_positions and current_position is None:
        # 有新信号且当前无持仓 -> 开仓
        signal_info = signal_positions[idx]
        current_position = {
            'entry_idx': idx,
            'direction': signal_info['direction'],
            'entry_price': trading_log.loc[idx, 'close'],
            'signal_type': signal_info['type']
        }
        trading_log.loc[idx, '交易状态'] = '开仓'
        trading_log.loc[idx, '持仓方向'] = signal_info['direction']
        trading_log.loc[idx, '开仓价'] = trading_log.loc[idx, 'close']
        trading_log.loc[idx, '备注'] = f"{signal_info['type']}信号开仓"

    elif current_position is not None:
        # 当前有持仓
        entry_idx = current_position['entry_idx']
        direction = current_position['direction']
        entry_price = current_position['entry_price']
        current_price = trading_log.loc[idx, 'close']

        # 计算盈亏
        if direction == 'short':
            pnl_pct = (entry_price - current_price) / entry_price * 100
        else:  # long
            pnl_pct = (current_price - entry_price) / entry_price * 100

        trading_log.loc[idx, '交易状态'] = '持仓中'
        trading_log.loc[idx, '持仓方向'] = direction
        trading_log.loc[idx, '开仓价'] = entry_price
        trading_log.loc[idx, '盈亏%'] = pnl_pct

        # 检查是否需要平仓（持仓30周期或达到条件）
        holding_periods = idx - entry_idx

        # 平仓条件：
        # 1. 持仓超过30周期
        # 2. SHORT: 盈利>2% 且 (张力<0 或 加速度>-0.05 或 量能>1.27)
        # 3. LONG: 盈利>2% 且 (张力>-0.01 或 量能>1.42) 或 亏损<-5%

        should_close = False
        close_reason = ""

        if holding_periods >= 30:
            should_close = True
            close_reason = "持仓满30周期"
        elif direction == 'short' and pnl_pct > 2:
            tension = trading_log.loc[idx, 'tension']
            accel = trading_log.loc[idx, 'acceleration']
            volume = trading_log.loc[idx, 'volume_ratio']

            if tension < 0:
                should_close = True
                close_reason = f"盈利{pnl_pct:.2f}%,张力{tension:.3f}<0"
            elif accel > -0.05:
                should_close = True
                close_reason = f"盈利{pnl_pct:.2f}%,加速度{accel:.3f}>-0.05"
            elif volume > 1.27:
                should_close = True
                close_reason = f"盈利{pnl_pct:.2f}%,量能{volume:.2f}>1.27"

        elif direction == 'long':
            if pnl_pct < -5:
                should_close = True
                close_reason = f"止损:亏损{pnl_pct:.2f}%<-5%"
            elif pnl_pct > 2:
                tension = trading_log.loc[idx, 'tension']
                volume = trading_log.loc[idx, 'volume_ratio']

                if tension > -0.01:
                    should_close = True
                    close_reason = f"盈利{pnl_pct:.2f}%,张力{tension:.3f}>-0.01"
                elif volume > 1.42:
                    should_close = True
                    close_reason = f"盈利{pnl_pct:.2f}%,量能{volume:.2f}>1.42"

        if should_close:
            trading_log.loc[idx, '交易状态'] = '平仓'
            trading_log.loc[idx, '备注'] = close_reason
            current_position = None

# 保存交易日志
output_file = 'trading_execution_log.xlsx'
trading_log.to_excel(output_file, index=False, engine='openpyxl')
print(f"\n[OK] 已保存: {output_file}")

# ==================== 统计交易结果 ====================
print("\n" + "=" * 100)
print("交易统计")
print("=" * 100)

# 找出所有开仓和平仓
entries = trading_log[trading_log['交易状态'] == '开仓']
exits = trading_log[trading_log['交易状态'] == '平仓']

print(f"\n总开仓次数: {len(entries)}次")
print(f"总平仓次数: {len(exits)}次")

if len(exits) > 0:
    print(f"\n平仓盈亏分布:")
    print(f"  平均盈亏: {exits['盈亏%'].mean():.2f}%")
    print(f"  最大盈利: {exits['盈亏%'].max():.2f}%")
    print(f"  最大亏损: {exits['盈亏%'].min():.2f}%")
    print(f"  盈利次数: {len(exits[exits['盈亏%'] > 0])}次")
    print(f"  亏损次数: {len(exits[exits['盈亏%'] < 0])}次")

    print(f"\n盈亏比:")
    avg_win = exits[exits['盈亏%'] > 0]['盈亏%'].mean() if len(exits[exits['盈亏%'] > 0]) > 0 else 0
    avg_loss = exits[exits['盈亏%'] < 0]['盈亏%'].mean() if len(exits[exits['盈亏%'] < 0]) > 0 else 0
    print(f"  平均盈利: {avg_win:.2f}%")
    print(f"  平均亏损: {avg_loss:.2f}%")
    if avg_loss != 0:
        print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}")

    # 按方向统计
    print(f"\n按方向统计:")
    for direction in ['long', 'short']:
        exits_dir = exits[exits['持仓方向'] == direction]
        if len(exits_dir) > 0:
            print(f"\n  {direction.upper()}交易:")
            print(f"    次数: {len(exits_dir)}次")
            print(f"    平均盈亏: {exits_dir['盈亏%'].mean():.2f}%")
            print(f"    盈利次数: {len(exits_dir[exits_dir['盈亏%'] > 0])}次")
            print(f"    亏损次数: {len(exits_dir[exits_dir['盈亏%'] < 0])}次")

# ==================== 展示前50行示例 ====================
print("\n" + "=" * 100)
print("前50行交易日志示例")
print("=" * 100)

display_cols = ['timestamp', 'close', '信号类型', '交易状态', '持仓方向', '开仓价', '盈亏%', '备注']
print(trading_log[display_cols].head(50).to_string(index=False))

print("\n" + "=" * 100)
print("[完成] 交易日志已生成，请打开Excel文件查看完整记录")
print("=" * 100)
