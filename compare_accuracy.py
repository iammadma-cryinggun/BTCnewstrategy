import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("策略准确性对比：我们总结的规则 vs V7.0.7系统")
print("2025年6月-12月")
print("="*100)

# 1. 读取信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

signals_2025 = signals_df[
    (signals_df['时间'] >= '2025-06-01') &
    (signals_df['时间'] <= '2025-12-31')
].copy()

print(f"\n信号数据: {len(signals_2025)}条")

# 2. 回测我们总结的规则
print("\n" + "="*100)
print("回测我们总结的规则...")
print("="*100)

our_trades = []
position = None
entry_signal_index = None
entry_time = None
entry_price = None
entry_tension = None

for idx, row in signals_2025.iterrows():
    signal_time = row['时间']
    price = row['收盘价']
    tension = row['张力']
    acceleration = row['加速度']

    # 检查开仓信号
    if position is None:
        # SHORT开仓条件
        if tension > 0.5 and acceleration < 0:
            entry_signal_index = idx
            entry_tension = tension
            position = 'short_pending'
            continue

        # LONG开仓条件
        elif tension < -0.5 and acceleration > 0:
            entry_signal_index = idx
            entry_tension = tension
            position = 'long_pending'
            continue

    # 等待确认
    elif position == 'short_pending':
        periods_waited = idx - entry_signal_index
        if periods_waited == 1:
            if tension > 0.5 and acceleration < 0:
                position = 'short'
                entry_time = signal_time
                entry_price = price
                print(f"\nSHORT开仓: {signal_time}, 价格={price:.2f}, 张力={tension:.4f}")
            else:
                position = None
                entry_signal_index = None

    elif position == 'long_pending':
        periods_waited = idx - entry_signal_index
        if periods_waited == 1:
            if tension < -0.5:
                position = 'long'
                entry_time = signal_time
                entry_price = price
                print(f"\nLONG开仓: {signal_time}, 价格={price:.2f}, 张力={tension:.4f}")
            else:
                position = None
                entry_signal_index = None

    # 检查平仓
    elif position == 'short':
        if entry_tension > 0:
            tension_change = (entry_tension - tension) / entry_tension

            # 平仓条件：张力下降40%或加速度转正
            if tension_change > 0.4 or (acceleration > 0):
                pnl = (entry_price - price) / entry_price * 100
                hold_periods = idx - entry_signal_index - 1  # 减去等待的1个周期

                our_trades.append({
                    '方向': 'SHORT',
                    '开仓时间': entry_time,
                    '开仓价': entry_price,
                    '开仓张力': entry_tension,
                    '平仓时间': signal_time,
                    '平仓价': price,
                    '平仓张力': tension,
                    '盈亏%': pnl,
                    '持仓周期': hold_periods,
                    '出场原因': '张力下降40%' if tension_change > 0.4 else '加速度转正'
                })

                print(f"SHORT平仓: {signal_time}, 价格={price:.2f}, 盈亏={pnl:+.2f}%, 持仓={hold_periods}周期")

                position = None
                entry_signal_index = None
                entry_tension = None
                entry_time = None
                entry_price = None

    elif position == 'long':
        # 平仓条件：张力转正
        if tension > 0:
            pnl = (price - entry_price) / entry_price * 100
            hold_periods = idx - entry_signal_index - 1

            our_trades.append({
                '方向': 'LONG',
                '开仓时间': entry_time,
                '开仓价': entry_price,
                '开仓张力': entry_tension,
                '平仓时间': signal_time,
                '平仓价': price,
                '平仓张力': tension,
                '盈亏%': pnl,
                '持仓周期': hold_periods,
                '出场原因': '张力转正'
            })

            print(f"LONG平仓: {signal_time}, 价格={price:.2f}, 盈亏={pnl:+.2f}%, 持仓={hold_periods}周期")

            position = None
            entry_signal_index = None
            entry_tension = None
            entry_time = None
            entry_price = None

# 3. 读取V7.0.7系统交易
all_trades = pd.read_csv('backtest_results_2024_2025.csv')
all_trades['入场时间'] = pd.to_datetime(all_trades['入场时间'])

v707_trades_2025 = all_trades[
    (all_trades['入场时间'] >= '2025-06-01') &
    (all_trades['入场时间'] <= '2025-12-31')
]

print(f"\n我们总结的规则: {len(our_trades)}笔")
print(f"V7.0.7系统: {len(v707_trades_2025)}笔")

# 4. 统计对比
our_trades_df = pd.DataFrame(our_trades)

if len(our_trades_df) > 0:
    our_long = our_trades_df[our_trades_df['方向'] == 'LONG']
    our_short = our_trades_df[our_trades_df['方向'] == 'SHORT']

    our_total_pnl = our_trades_df['盈亏%'].sum()
    our_win_rate = (our_trades_df['盈亏%'] > 0).sum() / len(our_trades_df) * 100
    our_avg_pnl = our_trades_df['盈亏%'].mean()
    our_max_win = our_trades_df['盈亏%'].max()
    our_max_loss = our_trades_df['盈亏%'].min()

    print("\n" + "="*100)
    print("我们总结的规则统计")
    print("="*100)
    print(f"总交易: {len(our_trades_df)}笔")
    print(f"总盈亏: {our_total_pnl:+.2f}%")
    print(f"胜率: {our_win_rate:.2f}%")
    print(f"平均盈亏: {our_avg_pnl:+.2f}%")
    print(f"最大盈利: {our_max_win:+.2f}%")
    print(f"最大亏损: {our_max_loss:+.2f}%")

    if len(our_long) > 0:
        print(f"\nLONG: {len(our_long)}笔, 盈亏{our_long['盈亏%'].sum():+.2f}%, 胜率{(our_long['盈亏%']>0).sum()/len(our_long)*100:.2f}%")

    if len(our_short) > 0:
        print(f"SHORT: {len(our_short)}笔, 盈亏{our_short['盈亏%'].sum():+.2f}%, 胜率{(our_short['盈亏%']>0).sum()/len(our_short)*100:.2f}%")

# V7.0.7统计
v707_long = v707_trades_2025[v707_trades_2025['方向'] == 'LONG']
v707_short = v707_trades_2025[v707_trades_2025['方向'] == 'SHORT']

v707_total_pnl = v707_trades_2025['盈亏%'].sum()
v707_win_rate = (v707_trades_2025['盈亏%'] > 0).sum() / len(v707_trades_2025) * 100
v707_avg_pnl = v707_trades_2025['盈亏%'].mean()

print("\n" + "="*100)
print("V7.0.7系统统计")
print("="*100)
print(f"总交易: {len(v707_trades_2025)}笔")
print(f"总盈亏: {v707_total_pnl:+.2f}%")
print(f"胜率: {v707_win_rate:.2f}%")
print(f"平均盈亏: {v707_avg_pnl:+.2f}%")
print(f"\nLONG: {len(v707_long)}笔, 盈亏{v707_long['盈亏%'].sum():+.2f}%, 胜率{(v707_long['盈亏%']>0).sum()/len(v707_long)*100:.2f}%")
print(f"SHORT: {len(v707_short)}笔, 盈亏{v707_short['盈亏%'].sum():+.2f}%, 胜率{(v707_short['盈亏%']>0).sum()/len(v707_short)*100:.2f}%")

# 5. 创建对比Excel
with pd.ExcelWriter('准确性对比_2025年6-12月.xlsx', engine='openpyxl') as writer:

    # Sheet1: 我们总结的规则交易明细
    if len(our_trades_df) > 0:
        our_trades_df.to_excel(writer, sheet_name='我们规则_交易明细', index=False)

    # Sheet2: V7.0.7系统交易明细
    v707_trades_2025.to_excel(writer, sheet_name='V707系统_交易明细', index=False)

    # Sheet3: 核心对比
    comparison_data = []

    if len(our_trades_df) > 0:
        comparison_data.append({
            '策略': '我们总结的规则',
            '参数': '张力>±0.5, 加速度条件',
            '总交易数': len(our_trades_df),
            'LONG数': len(our_long),
            'SHORT数': len(our_short),
            '总盈亏%': our_total_pnl,
            '胜率%': our_win_rate,
            '平均盈亏%': our_avg_pnl,
            '最大盈利%': our_max_win,
            '最大亏损%': our_max_loss,
            '盈亏比': abs(our_trades_df[our_trades_df['盈亏%']>0]['盈亏%'].mean() / our_trades_df[our_trades_df['盈亏%']<0]['盈亏%'].mean()) if len(our_trades_df[our_trades_df['盈亏%']<0]) > 0 else 0,
        })

    comparison_data.append({
        '策略': 'V7.0.7系统',
        '参数': '张力>±0.35, V7.0.5过滤器',
        '总交易数': len(v707_trades_2025),
        'LONG数': len(v707_long),
        'SHORT数': len(v707_short),
        '总盈亏%': v707_total_pnl,
        '胜率%': v707_win_rate,
        '平均盈亏%': v707_avg_pnl,
        '最大盈利%': v707_trades_2025['盈亏%'].max(),
        '最大亏损%': v707_trades_2025['盈亏%'].min(),
        '盈亏比': abs(v707_trades_2025[v707_trades_2025['盈亏%']>0]['盈亏%'].mean() / v707_trades_2025[v707_trades_2025['盈亏%']<0]['盈亏%'].mean()) if len(v707_trades_2025[v707_trades_2025['盈亏%']<0]) > 0 else 0,
    })

    pd.DataFrame(comparison_data).to_excel(writer, sheet_name='核心对比', index=False)

    # Sheet4: 结论分析
    if len(our_trades_df) > 0:
        # 计算月均收益
        our_monthly_avg = our_total_pnl / 7  # 6-12月=7个月
        v707_monthly_avg = v707_total_pnl / 7

        conclusion = pd.DataFrame([
            ['准确性对比结论', ''],
            ['', ''],
            ['总收益', ''],
            ['我们总结的规则', f'{our_total_pnl:+.2f}%'],
            ['V7.0.7系统', f'{v707_total_pnl:+.2f}%'],
            ['差异', f'{v707_total_pnl - our_total_pnl:+.2f}%'],
            ['', ''],
            ['月均收益', ''],
            ['我们总结的规则', f'{our_monthly_avg:+.2f}%'],
            ['V7.0.7系统', f'{v707_monthly_avg:+.2f}%'],
            ['', ''],
            ['胜率', ''],
            ['我们总结的规则', f'{our_win_rate:.2f}%'],
            ['V7.0.7系统', f'{v707_win_rate:.2f}%'],
            ['差异', f'{v707_win_rate - our_win_rate:+.2f}个百分点'],
            ['', ''],
            ['交易频率', ''],
            ['我们总结的规则', f'{len(our_trades_df)}笔 (每月{len(our_trades_df)/7:.1f}笔)'],
            ['V7.0.7系统', f'{len(v707_trades_2025)}笔 (每月{len(v707_trades_2025)/7:.1f}笔)'],
            ['', ''],
            ['关键结论', ''],
            ['哪个更准确?', 'V7.0.7系统' if v707_total_pnl > our_total_pnl else '我们总结的规则'],
            ['原因?', 'V7.0.7参数更宽松，捕捉更多机会' if v707_total_pnl > our_total_pnl else '我们的参数更严格，质量更高'],
        ])
        conclusion.to_excel(writer, sheet_name='结论分析', index=False, header=False)

print("\n" + "="*100)
print("Excel文件已生成: 准确性对比_2025年6-12月.xlsx")
print("="*100)

if len(our_trades_df) > 0:
    print("\n最终结论:")
    if v707_total_pnl > our_total_pnl:
        print(f"  ✅ V7.0.7系统更准确（收益高{v707_total_pnl - our_total_pnl:.2f}%）")
    else:
        print(f"  ✅ 我们总结的规则更准确（收益高{our_total_pnl - v707_total_pnl:.2f}%）")

print("\n" + "="*100)
