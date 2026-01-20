import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("对比分析：首次信号开仓 vs 等待确认后开仓")
print("（好机会 vs 所有通过确认的机会）")
print("="*100)

# 读取分析结果
df = pd.read_excel('按完整规律分析_2025年6-12月.xlsx', sheet_name='所有机会')

# 分析所有通过确认的机会
analysis = []

for idx, row in df.iterrows():
    first_price = row['首次信号价格']
    entry_price = row['开仓价格']
    first_tension = row['首次信号张力']
    entry_tension = row['开仓张力']
    wait_periods = row['等待周期']

    # 对于SHORT交易，价格越低越好
    # 如果entry_price < first_price，说明等待后价格更优
    price_advantage = (first_price - entry_price) / first_price * 100

    # 张力变化
    tension_change = (entry_tension - first_tension) / first_tension * 100

    analysis.append({
        '首次信号时间': row['首次信号时间'],
        '首次信号张力': first_tension,
        '首次信号价格': first_price,
        '等待周期': wait_periods,
        '开仓时间': row['最佳开仓时间'],
        '开仓张力': entry_tension,
        '开仓价格': entry_price,
        '价格优势%': price_advantage,
        '张力变化%': tension_change,
        '盈亏%': row['盈亏%'] if pd.notna(row['盈亏%']) else None,
        '是好机会': row['是好机会'],
        '平仓时间': row['平仓时间']
    })

analysis_df = pd.DataFrame(analysis)

# 统计所有通过确认的
confirmed = analysis_df[analysis_df['平仓时间'].notna()].copy()

print(f"\n已平仓交易: {len(confirmed)}笔")

# 对比价格优势
print("\n" + "="*100)
print("价格优势分析")
print("="*100)

# 计算价格优势统计
price_adv = confirmed['价格优势%']

print(f"\n所有{len(confirmed)}笔已平仓交易:")
print(f"  平均价格优势: {price_adv.mean():+.2f}%")
print(f"  中位数价格优势: {price_adv.median():+.2f}%")
print(f"  最大优势: {price_adv.max():+.2f}%")
print(f"  最大劣势: {price_adv.min():+.2f}%")
print(f"  有优势的交易: {(price_adv > 0).sum()}笔 ({(price_adv > 0).sum()/len(confirmed)*100:.1f}%)")
print(f"  有劣势的交易: {(price_adv < 0).sum()}笔 ({(price_adv < 0).sum()/len(confirmed)*100:.1f}%)")

# 好机会
good_trades = confirmed[confirmed['是好机会'] == '是']
if len(good_trades) > 0:
    good_price_adv = good_trades['价格优势%']

    print(f"\n好机会（盈利>2%）: {len(good_trades)}笔")
    print(f"  平均价格优势: {good_price_adv.mean():+.2f}%")
    print(f"  中位数价格优势: {good_price_adv.median():+.2f}%")
    print(f"  最大优势: {good_price_adv.max():+.2f}%")
    print(f"  最大劣势: {good_price_adv.min():+.2f}%")

# 差机会
bad_trades = confirmed[confirmed['是好机会'] == '否']
if len(bad_trades) > 0:
    bad_price_adv = bad_trades['价格优势%']

    print(f"\n差机会（盈利≤2%）: {len(bad_trades)}笔")
    print(f"  平均价格优势: {bad_price_adv.mean():+.2f}%")
    print(f"  中位数价格优势: {bad_price_adv.median():+.2f}%")
    print(f"  最大优势: {bad_price_adv.max():+.2f}%")
    print(f"  最大劣势: {bad_price_adv.min():+.2f}%")

# 详细展示每笔交易
print("\n" + "="*100)
print("详细对比")
print("="*100)

for idx, row in confirmed.iterrows():
    is_good = row['是好机会'] == '是'
    marker = '***' if is_good else '   '

    print(f"\n{marker} 交易 {idx+1}")
    print(f"   首次信号: {row['首次信号时间']}")
    print(f"     张力: {row['首次信号张力']:.4f}, 价格: {row['首次信号价格']:.2f}")
    print(f"   等待{row['等待周期']:.0f}个周期后:")
    print(f"     开仓: {row['开仓时间']}")
    print(f"     张力: {row['开仓张力']:.4f} (变化{row['张力变化%']:+.2f}%)")
    print(f"     价格: {row['开仓价格']:.2f}")
    print(f"   价格优势: {row['价格优势%']:+.2f}% ({'更优' if row['价格优势%'] > 0 else '更差' if row['价格优势%'] < 0 else '持平'})")
    print(f"   最终盈亏: {row['盈亏%']:+.2f}% ({'好机会' if is_good else '差机会'})")

# 保存到Excel
with pd.ExcelWriter('开仓价格优势对比.xlsx', engine='openpyxl') as writer:
    confirmed.to_excel(writer, sheet_name='所有已平仓交易', index=False)

    if len(good_trades) > 0:
        good_trades.to_excel(writer, sheet_name='好机会', index=False)

    if len(bad_trades) > 0:
        bad_trades.to_excel(writer, sheet_name='差机会', index=False)

    # 统计汇总
    summary = pd.DataFrame([
        ['价格优势分析', ''],
        ['', ''],
        ['所有交易', ''],
        ['交易数', len(confirmed)],
        ['平均价格优势', f'{price_adv.mean():+.2f}%'],
        ['有优势的', f'{(price_adv > 0).sum()}笔 ({(price_adv > 0).sum()/len(confirmed)*100:.1f}%)'],
        ['有劣势的', f'{(price_adv < 0).sum()}笔 ({(price_adv < 0).sum()/len(confirmed)*100:.1f}%)'],
        ['', ''],
        ['好机会', ''],
        ['交易数', len(good_trades)],
        ['平均价格优势', f'{good_price_adv.mean():+.2f}%' if len(good_trades) > 0 else 'N/A'],
        ['', ''],
        ['差机会', ''],
        ['交易数', len(bad_trades)],
        ['平均价格优势', f'{bad_price_adv.mean():+.2f}%' if len(bad_trades) > 0 else 'N/A'],
        ['', ''],
        ['结论', ''],
        ['等待确认策略', '平均价格' + ('更优' if price_adv.mean() > 0 else '更差' if price_adv.mean() < 0 else '持平')],
        ['优势幅度', f'{abs(price_adv.mean()):.2f}%'],
    ])
    summary.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

print("\n" + "="*100)
print("分析完成！已保存到: 开仓价格优势对比.xlsx")
print("="*100)

# 结论
print(f"\n核心结论:")
if price_adv.mean() > 0:
    print(f"  等待确认后开仓，平均价格优势为 {price_adv.mean():+.2f}%")
    print(f"  这说明等待1-2个周期确认是有价值的！")
elif price_adv.mean() < 0:
    print(f"  等待确认后开仓，平均价格劣势为 {price_adv.mean():+.2f}%")
    print(f"  这说明等待确认反而损失了价格优势！")
else:
    print(f"  等待确认后开仓，价格基本持平")
    print(f"  说明等待确认对价格影响不大")
