# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('带信号标记_关键行_V2.csv', encoding='utf-8-sig')

print('='*120)
print('V2版本策略回测分析（修复后）')
print('='*120)

# 追踪交易
trades = []
entry_price = None
entry_time = None
entry_type = None

for idx, row in df.iterrows():
    action = row['信号动作']

    # 开仓
    if pd.notna(action) and '开多' in action:
        entry_price = row['入场价']
        entry_time = row['时间']
        entry_type = action

    # 平仓
    elif pd.notna(action) and '平仓' in action and entry_price is not None:
        exit_price = row['收盘价']
        exit_reason = action.split('(')[1].split(')')[0] if '(' in action else action

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_time': entry_time,
            'exit_time': row['时间'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'type': entry_type,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct
        })

        entry_price = None
        entry_time = None
        entry_type = None

print(f'\n总交易数: {len(trades)}\n')

print('='*120)
print(f'{"序号":<4} {"入场时间":<20} {"出场时间":<20} {"类型":<20} {"入场价":>10} {"出场价":>10} {"盈亏%":>10} {"原因":<15}')
print('='*120)

total_pnl = 0
win_count = 0
loss_count = 0
hold_periods = []

for i, trade in enumerate(trades, 1):
    pnl_pct = trade['pnl_pct']
    total_pnl += pnl_pct

    if pnl_pct > 0:
        win_count += 1
    else:
        loss_count += 1

    # 计算持仓时间
    entry_t = pd.to_datetime(trade['entry_time'])
    exit_t = pd.to_datetime(trade['exit_time'])
    hold_hours = (exit_t - entry_t).total_seconds() / 3600
    hold_periods.append(hold_hours)

    print(f'{i:<4} {str(trade["entry_time"]):<20} {str(trade["exit_time"]):<20} {trade["type"]:<20} '
          f'{trade["entry_price"]:>10.2f} {trade["exit_price"]:>10.2f} {pnl_pct:>+9.2f}% {trade["exit_reason"]:<15}')

print('='*120)
print('\n总体统计')
print('='*120)
print(f'总交易次数: {len(trades)}')
print(f'盈利交易: {win_count}')
print(f'亏损交易: {loss_count}')

if len(trades) > 0:
    win_rate = win_count / len(trades) * 100
    avg_pnl = total_pnl / len(trades)

    print(f'胜率: {win_rate:.1f}%')
    print(f'平均每笔盈亏: {avg_pnl:+.2f}%')
    print(f'总盈亏: {total_pnl:+.2f}%')

    if hold_periods:
        avg_hold = sum(hold_periods) / len(hold_periods)
        print(f'平均持仓时间: {avg_hold:.1f}小时 ({avg_hold/4:.1f}个4小时周期)')

    # 盈亏比
    if win_count > 0 and loss_count > 0:
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['pnl_pct'] <= 0]

        avg_win = sum([t['pnl_pct'] for t in winning_trades]) / len(winning_trades)
        avg_loss = sum([t['pnl_pct'] for t in losing_trades]) / len(losing_trades)

        profit_factor = abs(avg_win * win_count / (avg_loss * loss_count))
        print(f'平均盈利: {avg_win:+.2f}%')
        print(f'平均亏损: {avg_loss:+.2f}%')
        print(f'盈亏比: {profit_factor:.2f}')

# 按平仓原因分类
print('\n' + '='*120)
print('按平仓原因分类')
print('='*120)

exit_reasons = {}
for trade in trades:
    reason = trade['exit_reason']
    if reason not in exit_reasons:
        exit_reasons[reason] = []
    exit_reasons[reason].append(trade['pnl_pct'])

for reason, pnls in exit_reasons.items():
    avg_pnl = sum(pnls) / len(pnls)
    win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100
    print(f'{reason:<20} 数量:{len(pnls):>2}  平均盈亏:{avg_pnl:>+7.2f}%  胜率:{win_rate:>5.1f}%')
