import pandas as pd

df = pd.read_csv('backtest_results_2024_2025.csv')

print("="*80)
print("V7.0.7 回测结果分析（2024-2025）")
print("="*80)

print(f"\n总交易数: {len(df)}")

print("\n按方向统计:")
print(df['方向'].value_counts())

print("\n按出场原因统计:")
print(df['出场原因'].value_counts())

print("\n总体盈亏:")
pnl = df['盈亏%']
print(f"  总盈亏: {pnl.sum():.2f}%")
print(f"  胜率: {(pnl>0).sum()/len(df)*100:.2f}%")
print(f"  平均盈亏: {pnl.mean():.2f}%")
print(f"  最大盈利: {pnl.max():.2f}%")
print(f"  最大亏损: {pnl.min():.2f}%")

print("\n各方向盈亏:")
for direction in ['LONG', 'SHORT']:
    dir_df = df[df['方向']==direction]
    if len(dir_df) > 0:
        win_rate = (dir_df['盈亏%']>0).sum()/len(dir_df)*100
        print(f"  {direction}: {len(dir_df)}笔, 盈亏{dir_df['盈亏%'].sum():.2f}%, 胜率{win_rate:.2f}%")

# 只看2025年6-12月的数据
df['入场时间'] = pd.to_datetime(df['入场时间'])
df_2025_6_12 = df[(df['入场时间'] >= '2025-06-01') & (df['入场时间'] <= '2025-12-31')]

print(f"\n\n{'='*80}")
print("2025年6-12月数据")
print("="*80)
print(f"总交易数: {len(df_2025_6_12)}")

if len(df_2025_6_12) > 0:
    print("\n按方向统计:")
    print(df_2025_6_12['方向'].value_counts())

    print("\n按出场原因统计:")
    print(df_2025_6_12['出场原因'].value_counts())

    print("\n总体盈亏:")
    pnl_2025 = df_2025_6_12['盈亏%']
    print(f"  总盈亏: {pnl_2025.sum():.2f}%")
    print(f"  胜率: {(pnl_2025>0).sum()/len(df_2025_6_12)*100:.2f}%")
    print(f"  平均盈亏: {pnl_2025.mean():.2f}%")
    print(f"  最大盈利: {pnl_2025.max():.2f}%")
    print(f"  最大亏损: {pnl_2025.min():.2f}%")

    print("\n各方向盈亏:")
    for direction in ['LONG', 'SHORT']:
        dir_df = df_2025_6_12[df_2025_6_12['方向']==direction]
        if len(dir_df) > 0:
            win_rate = (dir_df['盈亏%']>0).sum()/len(dir_df)*100
            print(f"  {direction}: {len(dir_df)}笔, 盈亏{dir_df['盈亏%'].sum():.2f}%, 胜率{win_rate:.2f}%")

print("\n" + "="*80)
