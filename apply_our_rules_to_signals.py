import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("将我们总结的策略规则标注到2025年6-12月信号中")
print("基于你提供的5笔交易（12月-1月）总结的规律")
print("="*100)

# 1. 读取2025年6-12月的信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

# 筛选6-12月的数据
signals_2025 = signals_df[
    (signals_df['时间'] >= '2025-06-01') &
    (signals_df['时间'] <= '2025-12-31')
].copy()

print(f"\n2025年6-12月信号数据: {len(signals_2025)}条")
print(f"时间范围: {signals_2025['时间'].min()} 至 {signals_2025['时间'].max()}")

# 2. 根据我们总结的规则标注信号
# 我们从5笔交易中总结的规律：
# SHORT: 张力>0.5, 加速度<0, 等待1周期开仓
# LONG: 张力<-0.5, 加速度>0, 等待1周期开仓

annotated_signals = []
position = None  # None, 'long', 'short'
entry_signal_index = None
entry_time = None
entry_tension = None

for idx, row in signals_2025.iterrows():
    signal_time = row['时间']
    tension = row['张力']
    acceleration = row['加速度']
    signal_type = row['信号类型']
    confidence = row['置信度']

    annotation = {
        '索引': idx,
        '时间': signal_time,
        '收盘价': row['收盘价'],
        '张力': tension,
        '加速度': acceleration,
        '量能比率': row['量能比率'],
        'EMA偏离%': row['EMA偏离%'],
        '信号类型': signal_type,
        '置信度': confidence,
        '标注': None,
        '标注说明': None
    }

    # 检查是否触发首次信号
    if position is None:
        # SHORT信号触发条件（基于我们总结的）
        if tension > 0.5 and acceleration < 0:
            annotation['标注'] = '【首次信号】SHORT'
            annotation['标注说明'] = f'张力={tension:.3f}>0.5, 加速度={acceleration:.3f}<0，等待1周期开仓'
            entry_signal_index = idx
            entry_tension = tension
            position = 'short_pending'

        # LONG信号触发条件（基于我们总结的）
        elif tension < -0.5 and acceleration > 0:
            annotation['标注'] = '【首次信号】LONG'
            annotation['标注说明'] = f'张力={tension:.3f}<-0.5, 加速度={acceleration:.3f}>0，等待1周期开仓'
            entry_signal_index = idx
            entry_tension = tension
            position = 'long_pending'

    # 等待确认后开仓
    elif position == 'short_pending':
        # SHORT开仓确认：1周期后
        periods_waited = idx - entry_signal_index
        if periods_waited == 1:
            if tension > 0.5 and acceleration < 0:
                annotation['标注'] = '⭐【最佳开仓】SHORT'
                annotation['标注说明'] = f'张力={tension:.3f}, 加速度={acceleration:.3f}'
                position = 'short'
                entry_time = signal_time
            else:
                position = None  # 条件不满足，取消
                entry_signal_index = None

    elif position == 'long_pending':
        # LONG开仓确认：1周期后
        periods_waited = idx - entry_signal_index
        if periods_waited == 1:
            if tension < -0.5:
                annotation['标注'] = '⭐【最佳开仓】LONG'
                annotation['标注说明'] = f'张力={tension:.3f}'
                position = 'long'
                entry_time = signal_time
            else:
                position = None  # 条件不满足，取消
                entry_signal_index = None

    # 持仓中，检查平仓条件
    elif position == 'short':
        # SHORT平仓条件：张力下降40%或加速度转正
        if entry_tension > 0:
            tension_change = (entry_tension - tension) / entry_tension
            if tension_change > 0.4 or (acceleration > 0):
                annotation['标注'] = '⭐⭐⭐【最佳平仓】SHORT'
                annotation['标注说明'] = f'张力变化={tension_change*100:.1f}%, 加速度={acceleration:.3f}'
                position = None
                entry_signal_index = None
                entry_tension = None
                entry_time = None

    elif position == 'long':
        # LONG平仓条件：张力转正或张力变化>100%
        if tension > 0:
            annotation['标注'] = '⭐⭐⭐【最佳平仓】LONG'
            annotation['标注说明'] = f'张力转正={tension:.3f}'
            position = None
            entry_signal_index = None
            entry_tension = None
            entry_time = None

    annotated_signals.append(annotation)

# 创建标注后的DataFrame
annotated_df = pd.DataFrame(annotated_signals)

# 统计标注结果
first_signals_count = len(annotated_df[annotated_df['标注'].str.contains('首次信号', na=False)])
best_entry_count = len(annotated_df[annotated_df['标注'].str.contains('最佳开仓', na=False)])
best_exit_count = len(annotated_df[annotated_df['标注'].str.contains('最佳平仓', na=False)])

print(f"\n标注结果:")
print(f"  首次信号: {first_signals_count}个")
print(f"  最佳开仓: {best_entry_count}个")
print(f"  最佳平仓: {best_exit_count}个")

# 3. 读取V7.0.7系统实际交易，用于对比
all_trades = pd.read_csv('backtest_results_2024_2025.csv')
all_trades['入场时间'] = pd.to_datetime(all_trades['入场时间'])

v707_trades_2025 = all_trades[
    (all_trades['入场时间'] >= '2025-06-01') &
    (all_trades['入场时间'] <= '2025-12-31')
]

print(f"\nV7.0.7系统实际交易: {len(v707_trades_2025)}笔")

# 4. 创建Excel对比
with pd.ExcelWriter('策略标注对比_2025年6-12月.xlsx', engine='openpyxl') as writer:

    # Sheet1: 标注后的全部信号
    annotated_df.to_excel(writer, sheet_name='标注信号_全部', index=False)

    # Sheet2: 只显示有标注的关键信号
    key_signals = annotated_df[annotated_df['标注'].notna()].copy()
    key_signals.to_excel(writer, sheet_name='关键信号_标注', index=False)

    # Sheet3: 按交易分组
    trade_groups = []
    current_trade = []
    trade_id = 0

    for idx, row in annotated_df.iterrows():
        if pd.notna(row['标注']) and '首次信号' in row['标注']:
            # 新交易开始
            if len(current_trade) > 0:
                trade_id += 1
                trade_groups.extend(current_trade)
            current_trade = [row]

        elif pd.notna(row['标注']):
            # 关键点，添加到当前交易
            current_trade.append(row)

    # 添加最后一个交易
    if len(current_trade) > 0:
        trade_groups.extend(current_trade)

    if len(trade_groups) > 0:
        trades_df = pd.DataFrame(trade_groups)
        trades_df.to_excel(writer, sheet_name='按交易分组', index=False)

    # Sheet4: V7.0.7系统实际交易
    v707_trades_2025.to_excel(writer, sheet_name='V707系统_实际交易', index=False)

    # Sheet5: 我们的规则说明
    our_rules = pd.DataFrame([
        ['我们从5笔交易总结的规则', ''],
        ['', ''],
        ['SHORT规则', ''],
        ['1. 首次信号', '张力>0.5, 加速度<0'],
        ['2. 最佳开仓', '等待1周期后，张力仍>0.5, 加速度<0'],
        ['3. 最佳平仓', '张力下降40% 或 加速度转正'],
        ['', ''],
        ['LONG规则', ''],
        ['1. 首次信号', '张力<-0.5, 加速度>0'],
        ['2. 最佳开仓', '等待1周期后，张力仍<-0.5'],
        ['3. 最佳平仓', '张力转正(>0)'],
        ['', ''],
        ['2025年6-12月应用结果', ''],
        ['首次信号数', f'{first_signals_count}个'],
        ['最佳开仓数', f'{best_entry_count}个'],
        ['最佳平仓数', f'{best_exit_count}个'],
        ['', ''],
        ['V7.0.7系统同期表现', ''],
        ['实际交易数', f'{len(v707_trades_2025)}笔'],
        ['总盈亏', f'{v707_trades_2025["盈亏%"].sum():.2f}%'],
        ['胜率', f'{(v707_trades_2025["盈亏%"]>0).sum()/len(v707_trades_2025)*100:.1f}%'],
    ])
    our_rules.to_excel(writer, sheet_name='我们的规则说明', index=False, header=False)

    # Sheet6: 对比分析
    comparison = pd.DataFrame([
        ['对比项', '我们总结的规则', 'V7.0.7系统', '差异'],
        ['', '', '', ''],
        ['信号逻辑', '顺势（张力>0.5做空）', '反向（张力<-0.35做空）', '⚠️ 完全相反'],
        ['', '', '', ''],
        ['SHORT触发条件', '张力>0.5, 加速度<0', 'BULLISH(T<-0.35)或HIGH_OSC(T>0.3)', '阈值不同'],
        ['', '', '', ''],
        ['LONG触发条件', '张力<-0.5, 加速度>0', 'BEARISH(T>0.35)或LOW_OSC(T<-0.3)', '阈值不同'],
        ['', '', '', ''],
        ['开仓确认', '等待1周期', '通过V7.0.5过滤器', '过滤方式不同'],
        ['', '', '', ''],
        ['2025年6-12月交易数', f'{best_entry_count}笔', f'{len(v707_trades_2025)}笔', f'V707多{len(v707_trades_2025)-best_entry_count}笔'],
        ['', '', '', ''],
        ['数据来源', '基于12月-1月5笔交易', '基于完整信号计算', '样本量差异'],
    ])
    comparison.to_excel(writer, sheet_name='对比分析', index=False, header=False)

print("\n" + "="*100)
print("Excel文件已生成: 策略标注对比_2025年6-12月.xlsx")
print("="*100)

print("\n包含以下Sheet:")
print("  1. 标注信号_全部 - 所有信号，包含我们的标注")
print("  2. 关键信号_标注 - 只显示有标注的关键信号")
print("  3. 按交易分组 - 按交易流程分组显示")
print("  4. V707系统_实际交易 - V7.0.7系统的实际交易")
print("  5. 我们的规则说明 - 我们从5笔交易总结的规则")
print("  6. 对比分析 - 我们总结的规则 vs V7.0.7系统")

print("\n" + "="*100)
print("核心发现")
print("="*100)
print(f"\n我们总结的规则在2025年6-12月:")
print(f"  触发首次信号: {first_signals_count}次")
print(f"  实际开仓: {best_entry_count}笔")
print(f"  实际平仓: {best_exit_count}笔")

print(f"\nV7.0.7系统在2025年6-12月:")
print(f"  实际交易: {len(v707_trades_2025)}笔")

if best_entry_count > 0:
    ratio = len(v707_trades_2025) / best_entry_count
    print(f"\n交易数量对比: V7.0.7是我们的 {ratio:.1f} 倍")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
