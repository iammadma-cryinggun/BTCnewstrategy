# -*- coding: utf-8 -*-
import pandas as pd

df_trades = pd.read_csv('最优策略_实战测试.csv', encoding='utf-8-sig')

print("="*100)
print("问题诊断：胜率高但收益低")
print("="*100)

# 基本统计
winning = df_trades[df_trades['pnl_pct'] > 0]
losing = df_trades[df_trades['pnl_pct'] <= 0]

print(f"\n交易统计:")
print(f"  总交易数: {len(df_trades)}")
print(f"  盈利交易: {len(winning)} (70%)")
print(f"  亏损交易: {len(losing)} (30%)")

print(f"\n盈亏分布:")
print(f"  盈利交易总盈亏: +${winning['pnl_usd'].sum():.2f}")
print(f"  盈利交易平均: {winning['pnl_pct'].mean():+.2f}%")
print(f"  盈利交易最大: {winning['pnl_pct'].max():+.2f}%")

print(f"\n  亏损交易总盈亏: ${losing['pnl_usd'].sum():.2f}")
print(f"  亏损交易平均: {losing['pnl_pct'].mean():+.2f}%")
print(f"  亏损交易最大: {losing['pnl_pct'].min():+.2f}%")

print(f"\n净收益: ${winning['pnl_usd'].sum():.2f} + ${losing['pnl_usd'].sum():.2f} = ${df_trades['pnl_usd'].sum():.2f}")

print("\n" + "="*100)
print("为什么胜率高但收益低？")
print("="*100)

print("\n问题1: 仓位太小")
INITIAL = 10000
POSITION = 0.03
print(f"  初始资金: ${INITIAL:,.2f}")
print(f"  每笔仓位: {POSITION*100}% = ${INITIAL * POSITION:,.2f}")
print(f"  平均每笔盈利: ${winning['pnl_usd'].sum() / len(winning):.2f}")
print(f"  平均每笔亏损: ${losing['pnl_usd'].sum() / len(losing):.2f}")

print("\n问题2: 平均盈利幅度小")
print(f"  盈利交易平均盈利: {winning['pnl_pct'].mean():+.2f}%")
print(f"  亏损交易平均亏损: {losing['pnl_pct'].mean():+.2f}%")
print(f"  这意味着大部分交易都是微利")

print("\n问题3: 很多零盈亏交易")
zero_pnl = df_trades[df_trades['pnl_pct'] == 0]
print(f"  零盈亏交易: {len(zero_pnl)} 笔")
print(f"  这些交易占用资金但不产生收益")

print("\n问题4: 持仓时间短")
print(f"  盈利交易平均持仓: {winning['hold_hours'].mean():.1f}小时")
print(f"  亏损交易平均持仓: {losing['hold_hours'].mean():.1f}小时")
print(f"  短持仓限制了利润空间")

print("\n" + "="*100)
print("验证：如果用100%仓位会怎样？")
print("="*100)

total_pnl = df_trades['pnl_usd'].sum()
theoretical_pnl = total_pnl / POSITION
theoretical_return = theoretical_pnl / INITIAL * 100

print(f"\n当前3%仓位:")
print(f"  总收益: ${total_pnl:+.2f}")
print(f"  收益率: {total_pnl / INITIAL * 100:+.2f}%")

print(f"\n如果是100%仓位（理论）:")
print(f"  总收益: ${theoretical_pnl:+.2f}")
print(f"  收益率: {theoretical_return:+.2f}%")

print(f"\n但100%仓位风险巨大！")
print(f"  最大单笔亏损: {df_trades['pnl_pct'].min():.2f}%")
print(f"  如果100%仓位，最大亏损: ${INITIAL * df_trades['pnl_pct'].min() / 100:+,.2f}")

print("\n" + "="*100)
print("结论")
print("="*100)

print("""
胜率高（70%）但收益低（1.59%）的原因：

1. 仓位保守：每笔只有3%
   - 虽然安全，但限制了绝对收益
   - 这是风险管理的结果

2. 盈利幅度小：平均每笔盈利仅1.17%
   - 大部分交易是微利
   - 没有捕捉到大趋势

3. 持仓时间短：平均只有几小时
   - 错过了更大的利润空间
   - 极值点触发后就平仓

4. 零盈亏交易多：占用资金但不产生收益
   - 信号切换导致的快速平仓

策略本质：
- 这是一个"剥头皮"类型的策略
- 追求高胜率、低回撤
- 牺牲了收益率

如果想要更高收益：
1. 增加仓位（但风险会剧增）
2. 放宽止损（但会增加大亏）
3. 延长持仓（但会增加回撤）

这就是风险与收益的平衡！
""")
