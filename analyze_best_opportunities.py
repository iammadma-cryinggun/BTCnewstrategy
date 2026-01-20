import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 读取分析结果
df = pd.read_excel('按完整规律分析_2025年6-12月.xlsx', sheet_name='所有机会')

print("="*100)
print("详细分析所有符合规律的机会")
print("="*100)

# 所有通过确认的
confirmed = df[df['平仓时间'].notna()].copy()
print(f"\n已平仓的交易: {len(confirmed)}笔")

# 显示每一笔
for idx, row in confirmed.iterrows():
    print(f"\n{'='*80}")
    print(f"交易 #{idx+1}")
    print(f"{'='*80}")
    print(f"首次信号: {row['首次信号时间']}")
    print(f"  张力: {row['首次信号张力']:.4f}, 加速度: {row['首次信号加速度']:.4f}, 量能: {row['首次信号量能']:.2f}")
    print(f"\n等待{row['等待周期']:.0f}个周期后开仓:")
    print(f"开仓: {row['最佳开仓时间']}")
    print(f"  张力: {row['开仓张力']:.4f} (变化{row['张力变化%']:+.2f}%)")
    print(f"  加速度: {row['开仓加速度']:.4f}")
    print(f"  量能: {row['开仓量能']:.2f}")
    print(f"  价格: {row['开仓价格']:.2f}")
    print(f"\n平仓: {row['平仓时间']}")
    print(f"  持仓{row['持仓周期']:.0f}个周期")
    print(f"  价格: {row['平仓价格']:.2f}")
    print(f"  盈亏: {row['盈亏%']:+.2f}%")
    print(f"  评价: {'好机会' if row['盈亏%'] > 2 else '普通机会'}")

# 统计好机会和差机会的特征
good = confirmed[confirmed['盈亏%'] > 2]
bad = confirmed[confirmed['盈亏%'] <= 2]

print(f"\n{'='*100}")
print(f"好机会特征分析（盈利>2%）")
print(f"{'='*100}")
if len(good) > 0:
    print(f"首次信号张力: 平均{good['首次信号张力'].mean():.4f}, 范围[{good['首次信号张力'].min():.4f}, {good['首次信号张力'].max():.4f}]")
    print(f"开仓张力: 平均{good['开仓张力'].mean():.4f}, 范围[{good['开仓张力'].min():.4f}, {good['开仓张力'].max():.4f}]")
    print(f"张力变化: 平均{good['张力变化%'].mean():+.2f}%")
    print(f"持仓周期: 平均{good['持仓周期'].mean():.1f}个")

print(f"\n{'='*100}")
print(f"差机会特征分析（盈利≤2%）")
print(f"{'='*100}")
if len(bad) > 0:
    print(f"首次信号张力: 平均{bad['首次信号张力'].mean():.4f}, 范围[{bad['首次信号张力'].min():.4f}, {bad['首次信号张力'].max():.4f}]")
    print(f"开仓张力: 平均{bad['开仓张力'].mean():.4f}, 范围[{bad['开仓张力'].min():.4f}, {bad['开仓张力'].max():.4f}]")
    print(f"张力变化: 平均{bad['张力变化%'].mean():+.2f}%")
    print(f"持仓周期: 平均{bad['持仓周期'].mean():.1f}个")

# 未平仓的
un_closed = df[df['平仓时间'].isna()].copy()
print(f"\n{'='*100}")
print(f"未平仓的交易: {len(un_closed)}笔")
print(f"{'='*100}")
if len(un_closed) > 0:
    print("前10个:")
    for idx, row in un_closed.head(10).iterrows():
        print(f"  {row['首次信号时间']} → {row['最佳开仓时间']} (等待{row['等待周期']:.0f}周期)")
        print(f"    张力: {row['首次信号张力']:.4f} → {row['开仓张力']:.4f} (变化{row['张力变化%']:+.2f}%)")
