import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("创建完整对比分析Excel：2025年6-12月")
print("  - 原始信号 vs V7.0.7系统 vs 我们总结的规则")
print("="*100)

# 1. 读取2025年6-12月的信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])
print(f"\n原始信号数据: {len(signals_df)}条")
print(f"时间范围: {signals_df['时间'].min()} 至 {signals_df['时间'].max()}")

# 2. 读取V7.0.7系统回测结果，筛选2025年6-12月
all_trades = pd.read_csv('backtest_results_2024_2025.csv')
all_trades['入场时间'] = pd.to_datetime(all_trades['入场时间'])
all_trades['出场时间'] = pd.to_datetime(all_trades['出场时间'])

v707_trades_2025 = all_trades[
    (all_trades['入场时间'] >= '2025-06-01') &
    (all_trades['入场时间'] <= '2025-12-31')
]

print(f"\nV7.0.7系统2025年6-12月实际交易: {len(v707_trades_2025)}笔")
print(f"时间范围: {v707_trades_2025['入场时间'].min()} 至 {v707_trades_2025['入场时间'].max()}")

# 3. 读取我们总结的策略回测结果
our_strategy_df = pd.read_csv('策略回测结果_2025_6-12月_正确版.csv', encoding='utf-8-sig')
our_strategy_df['开仓时间'] = pd.to_datetime(our_strategy_df['开仓时间'])
our_strategy_df['平仓时间'] = pd.to_datetime(our_strategy_df['平仓时间'])

# 只筛选6-12月的
our_strategy_2025 = our_strategy_df[
    (our_strategy_df['开仓时间'] >= '2025-06-01') &
    (our_strategy_df['开仓时间'] <= '2025-12-31')
]

print(f"\n我们总结的策略回测交易: {len(our_strategy_2025)}笔")
print(f"时间范围: {our_strategy_2025['开仓时间'].min()} 至 {our_strategy_2025['开仓时间'].max()}")

# 4. 创建对比Excel
with pd.ExcelWriter('完整对比分析_2025年6-12月.xlsx', engine='openpyxl') as writer:

    # Sheet1: 所有原始信号
    signals_df.to_excel(writer, sheet_name='原始信号_全部', index=False)

    # Sheet2: V7.0.7系统实际交易
    v707_trades_2025.to_excel(writer, sheet_name='V707系统_实际交易', index=False)

    # Sheet3: 我们总结的策略交易
    our_strategy_2025.to_excel(writer, sheet_name='我们总结的策略', index=False)

    # Sheet4: 核心对比 - 三个维度的统计
    comparison_data = []

    # === V7.0.7系统统计 ===
    v707_long = v707_trades_2025[v707_trades_2025['方向'] == 'LONG']
    v707_short = v707_trades_2025[v707_trades_2025['方向'] == 'SHORT']

    comparison_data.append({
        '策略': 'V7.0.7系统',
        '总交易数': len(v707_trades_2025),
        'LONG数': len(v707_long),
        'SHORT数': len(v707_short),
        '总盈亏%': v707_trades_2025['盈亏%'].sum(),
        '胜率%': (v707_trades_2025['盈亏%'] > 0).sum() / len(v707_trades_2025) * 100 if len(v707_trades_2025) > 0 else 0,
        'LONG盈亏%': v707_long['盈亏%'].sum() if len(v707_long) > 0 else 0,
        'SHORT盈亏%': v707_short['盈亏%'].sum() if len(v707_short) > 0 else 0,
        '平均盈亏%': v707_trades_2025['盈亏%'].mean() if len(v707_trades_2025) > 0 else 0,
    })

    # === 我们总结的策略统计 ===
    our_long = our_strategy_2025[our_strategy_2025['方向'] == 'LONG']
    our_short = our_strategy_2025[our_strategy_2025['方向'] == 'SHORT']

    comparison_data.append({
        '策略': '我们总结的规则',
        '总交易数': len(our_strategy_2025),
        'LONG数': len(our_long),
        'SHORT数': len(our_short),
        '总盈亏%': our_strategy_2025['盈亏%'].sum(),
        '胜率%': (our_strategy_2025['盈亏%'] > 0).sum() / len(our_strategy_2025) * 100 if len(our_strategy_2025) > 0 else 0,
        'LONG盈亏%': our_long['盈亏%'].sum() if len(our_long) > 0 else 0,
        'SHORT盈亏%': our_short['盈亏%'].sum() if len(our_short) > 0 else 0,
        '平均盈亏%': our_strategy_2025['盈亏%'].mean() if len(our_strategy_2025) > 0 else 0,
    })

    pd.DataFrame(comparison_data).to_excel(writer, sheet_name='核心对比', index=False)

    # Sheet5: 我们总结的规则详细说明
    our_rules_summary = pd.DataFrame([
        ['我们的总结规则', ''],
        ['', ''],
        ['SHORT开仓条件', ''],
        ['- 张力', '> 0.5'],
        ['- 加速度', '< 0'],
        ['- 量能比率', '< 1.0'],
        ['- 等待确认', '1-2个4H周期'],
        ['', ''],
        ['LONG开仓条件', ''],
        ['- 张力', '< -0.5'],
        ['- 加速度', '> 0'],
        ['- 等待确认', '1个4H周期'],
        ['', ''],
        ['SHORT平仓条件', ''],
        ['- 张力变化', '> 40%'],
        ['- 加速度', '> 0 或接近0'],
        ['- 量能比率', '> 1.0'],
        ['', ''],
        ['LONG平仓条件', ''],
        ['- 张力', '> 0'],
        ['- 张力变化', '> 100%'],
        ['- 加速度', '> 0'],
        ['', ''],
        ['回测结果', ''],
        ['- 总交易', f'{len(our_strategy_2025)}笔'],
        ['- 总盈亏', f'{our_strategy_2025["盈亏%"].sum():.2f}%'],
        ['- 胜率', f'{(our_strategy_2025["盈亏%"]>0).sum()/len(our_strategy_2025)*100:.1f}%' if len(our_strategy_2025) > 0 else '0%'],
    ])
    our_rules_summary.to_excel(writer, sheet_name='我们的规则说明', index=False, header=False)

    # Sheet6: V7.0.7真实规则说明
    v707_rules_summary = pd.DataFrame([
        ['V7.0.7系统真实规则', ''],
        ['', ''],
        ['核心特性：反向交易', ''],
        ['', ''],
        ['信号类型与交易方向', ''],
        ['- BEARISH_SINGULARITY（看空奇点）', '→ 做LONG'],
        ['- BULLISH_SINGULARITY（看涨奇点）', '→ 做SHORT'],
        ['- HIGH_OSCILLATION（高位震荡）', '→ 做SHORT'],
        ['- LOW_OSCILLATION（低位震荡）', '→ 做LONG'],
        ['- OSCILLATION（平衡震荡）', '→ 不交易'],
        ['', ''],
        ['信号触发条件', ''],
        ['BEARISH_SINGULARITY', 'T > 0.35, a < -0.02, 置信度0.7'],
        ['BULLISH_SINGULARITY', 'T < -0.35, a > 0.02, 置信度0.6'],
        ['HIGH_OSCILLATION', 'T > 0.3, |a| < 0.01, 置信度0.6'],
        ['LOW_OSCILLATION', 'T < -0.3, |a| < 0.01, 置信度0.6'],
        ['', ''],
        ['V7.0.5过滤器', ''],
        ['HIGH_OSC→SHORT', '价格vs EMA≤2%, a<0, 量能≤1.1'],
        ['LOW_OSC→LONG', '直接通过'],
        ['BULLISH→SHORT', '量能≤0.95, EMA偏离≤5%'],
        ['BEARISH→LONG', 'EMA偏离≥-5%'],
        ['', ''],
        ['出场策略', ''],
        ['- 止盈', '±5%'],
        ['- 止损', '±2.5%'],
        ['- 最大持仓', '42个4H周期（7天）'],
        ['', ''],
        ['回测结果', ''],
        ['- 总交易', f'{len(v707_trades_2025)}笔'],
        ['- 总盈亏', f'{v707_trades_2025["盈亏%"].sum():.2f}%'],
        ['- 胜率', f'{(v707_trades_2025["盈亏%"]>0).sum()/len(v707_trades_2025)*100:.1f}%' if len(v707_trades_2025) > 0 else '0%'],
    ])
    v707_rules_summary.to_excel(writer, sheet_name='V707规则说明', index=False, header=False)

    # Sheet7: 关键差异对比
    differences_summary = pd.DataFrame([
        ['关键差异', 'V7.0.7系统', '我们总结的规则', '影响'],
        ['', '', '', ''],
        ['交易逻辑', '反向交易（信号名称≠交易方向）', '顺势交易（信号名称=交易方向）', '⚠️ 关键错误'],
        ['', '', '', ''],
        ['SHORT信号', 'BULLISH_SINGULARITY (T<-0.35)', '张力>0.5', '完全相反'],
        ['SHORT信号', 'HIGH_OSCILLATION (T>0.3)', '张力>0.5', '部分重叠'],
        ['', '', '', ''],
        ['LONG信号', 'BEARISH_SINGULARITY (T>0.35)', '张力<-0.5', '完全相反'],
        ['LONG信号', 'LOW_OSCILLATION (T<-0.3)', '张力<-0.5', '部分重叠'],
        ['', '', '', ''],
        ['过滤器', 'V7.0.5多重过滤', '简单量能过滤', 'V707更严格'],
        ['', '', '', ''],
        ['出场策略', '固定止盈止损（±5%/±2.5%）', '张力变化40%平仓', '策略不同'],
        ['', '', '', ''],
        ['交易数量', f'{len(v707_trades_2025)}笔', f'{len(our_strategy_2025)}笔', 'V707交易更多'],
        ['', '', '', ''],
        ['总盈亏', f'{v707_trades_2025["盈亏%"].sum():.2f}%', f'{our_strategy_2025["盈亏%"].sum():.2f}%', 'V707收益更高'],
        ['胜率', f'{(v707_trades_2025["盈亏%"]>0).sum()/len(v707_trades_2025)*100:.1f}%', f'{(our_strategy_2025["盈亏%"]>0).sum()/len(our_strategy_2025)*100:.1f}%' if len(our_strategy_2025) > 0 else '0%', 'V707胜率接近'],
    ])
    differences_summary.to_excel(writer, sheet_name='关键差异对比', index=False, header=False)

print("\n" + "="*100)
print("Excel文件已生成: 完整对比分析_2025年6-12月.xlsx")
print("="*100)

print("\n包含以下Sheet:")
print("  1. 原始信号_全部 - 1,345条原始信号数据")
print("  2. V707系统_实际交易 - V7.0.7系统的185笔实际交易")
print("  3. 我们总结的策略 - 基于我们总结规则的8笔交易")
print("  4. 核心对比 - 两个策略的统计对比")
print("  5. 我们的规则说明 - 我们总结的规则详细说明")
print("  6. V707规则说明 - V7.0.7系统真实规则说明")
print("  7. 关键差异对比 - 两个策略的关键差异")

# 输出关键对比
print("\n" + "="*100)
print("关键对比分析")
print("="*100)

print(f"\n【V7.0.7系统】")
print(f"  总交易: {len(v707_trades_2025)}笔")
print(f"  总盈亏: {v707_trades_2025['盈亏%'].sum():.2f}%")
print(f"  胜率: {(v707_trades_2025['盈亏%']>0).sum()/len(v707_trades_2025)*100:.2f}%")
print(f"  LONG: {len(v707_long)}笔, 盈亏{v707_long['盈亏%'].sum():.2f}%")
print(f"  SHORT: {len(v707_short)}笔, 盈亏{v707_short['盈亏%'].sum():.2f}%")

print(f"\n【我们总结的策略】")
print(f"  总交易: {len(our_strategy_2025)}笔")
print(f"  总盈亏: {our_strategy_2025['盈亏%'].sum():.2f}%")
if len(our_strategy_2025) > 0:
    print(f"  胜率: {(our_strategy_2025['盈亏%']>0).sum()/len(our_strategy_2025)*100:.2f}%")
if len(our_long) > 0:
    print(f"  LONG: {len(our_long)}笔, 盈亏{our_long['盈亏%'].sum():.2f}%")
else:
    print(f"  LONG: 0笔")
if len(our_short) > 0:
    print(f"  SHORT: {len(our_short)}笔, 盈亏{our_short['盈亏%'].sum():.2f}%")
else:
    print(f"  SHORT: 0笔")

print(f"\n【核心差异】")
print(f"  V7.0.7: 反向交易（看涨信号→做空，看空信号→做多）")
print(f"  我们: 顺势交易（看涨信号→做多，看空信号→做空）⚠️")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
