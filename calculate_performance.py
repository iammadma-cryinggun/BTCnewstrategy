# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('带信号标记_关键行.csv', encoding='utf-8-sig')

print('='*100)
print('完整策略回测统计')
print('='*100)

# 追踪每笔交易
trades = []
entry_price = None
entry_time = None
entry_type = None
position_open = False

for idx, row in df.iterrows():
    action = row['信号动作']

    # 开仓
    if pd.notna(action) and '开多' in action:
        if position_open:
            # 前一笔未平仓，强制平仓
            trades.append({
                'entry_time': entry_time,
                'exit_time': row['时间'],
                'entry_price': entry_price,
                'exit_price': row['收盘价'],
                'type': entry_type,
                'exit_reason': '信号冲突'
            })

        entry_price = row['入场价']
        entry_time = row['时间']
        entry_type = action
        position_open = True

    # 平仓
    elif pd.notna(action) and '平仓' in action and position_open:
        exit_reason = action.split('(')[1].split(')')[0] if '(' in action else action

        # 从持仓行获取入场价
        if idx > 0:
            prev_row = df.iloc[idx - 1]
            if pd.notna(prev_row['入场价']) and prev_row['入场价'] > 0:
                entry_price = prev_row['入场价']

        trades.append({
            'entry_time': entry_time,
            'exit_time': row['时间'],
            'entry_price': entry_price,
            'exit_price': row['收盘价'],
            'type': entry_type,
            'exit_reason': exit_reason
        })

        position_open = False
        entry_price = None
        entry_time = None
        entry_type = None

# 计算统计
print(f'\n总交易数: {len(trades)}\n')

print('='*100)
print(f'{"序号":<4} {"入场时间":<20} {"出场时间":<20} {"类型":<20} {"入场价":>10} {"出场价":>10} {"盈亏%":>10} {"原因":<15}')
print('='*100)

total_pnl_pct = 0
win_count = 0
loss_count = 0
hold_periods = []

for i, trade in enumerate(trades, 1):
    entry_p = trade['entry_price']
    exit_p = trade['exit_price']

    if pd.notna(entry_p) and entry_p > 0:
        pnl_pct = (exit_p - entry_p) / entry_p * 100
        total_pnl_pct += pnl_pct

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
              f'{entry_p:>10.2f} {exit_p:>10.2f} {pnl_pct:>+9.2f}% {trade["exit_reason"]:<15}')

print('='*100)
print('\n总体统计')
print('='*100)
print(f'总交易次数: {len(trades)}')
print(f'盈利交易: {win_count}')
print(f'亏损交易: {loss_count}')

if len(trades) > 0:
    win_rate = win_count / len(trades) * 100
    avg_pnl = total_pnl_pct / len(trades)

    print(f'胜率: {win_rate:.1f}%')
    print(f'平均每笔盈亏: {avg_pnl:+.2f}%')
    print(f'总盈亏: {total_pnl_pct:+.2f}%')

    if hold_periods:
        avg_hold = sum(hold_periods) / len(hold_periods)
        print(f'平均持仓时间: {avg_hold:.1f}小时 ({avg_hold/4:.1f}个4小时周期)')

    # 盈亏比
    if win_count > 0 and loss_count > 0:
        winning_trades = [t for t in trades if pd.notna(t['entry_price']) and t['entry_price'] > 0 and (t['exit_price'] - t['entry_price']) / t['entry_price'] > 0]
        losing_trades = [t for t in trades if pd.notna(t['entry_price']) and t['entry_price'] > 0 and (t['exit_price'] - t['entry_price']) / t['entry_price'] <= 0]

        if winning_trades and losing_trades:
            avg_win = sum([(t['exit_price'] - t['entry_price']) / t['entry_price'] * 100 for t in winning_trades]) / len(winning_trades)
            avg_loss = sum([(t['exit_price'] - t['entry_price']) / t['entry_price'] * 100 for t in losing_trades]) / len(losing_trades)

            profit_factor = abs(avg_win * win_count / (avg_loss * loss_count))
            print(f'平均盈利: {avg_win:+.2f}%')
            print(f'平均亏损: {avg_loss:+.2f}%')
            print(f'盈亏比: {profit_factor:.2f}')

# 按平仓原因分类
print('\n' + '='*100)
print('按平仓原因分类')
print('='*100)

exit_reasons = {}
for trade in trades:
    reason = trade['exit_reason']
    if reason not in exit_reasons:
        exit_reasons[reason] = []
    pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
    exit_reasons[reason].append(pnl)

for reason, pnls in exit_reasons.items():
    avg_pnl = sum(pnls) / len(pnls)
    win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100
    print(f'{reason:<20} 数量:{len(pnls):>2}  平均盈亏:{avg_pnl:>+7.2f}%  胜率:{win_rate:>5.1f}%')
