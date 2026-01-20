import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("检查2025年6-12月符合我们条件的信号数量")
print("="*100)

# 1. 读取信号数据
signals_df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
signals_df['时间'] = pd.to_datetime(signals_df['时间'])

signals_2025 = signals_df[
    (signals_df['时间'] >= '2025-06-01') &
    (signals_df['时间'] <= '2025-12-31')
].copy()

print(f"\n总信号数: {len(signals_2025)}条")

# 2. 统计符合SHORT条件的信号（张力>0.5, 加速度<0）
short_signals = signals_2025[
    (signals_2025['张力'] > 0.5) &
    (signals_2025['加速度'] < 0)
].copy()

print(f"\n符合SHORT条件的信号（张力>0.5, 加速度<0）:")
print(f"  数量: {len(short_signals)}条")
print(f"  占比: {len(short_signals)/len(signals_2025)*100:.2f}%")

if len(short_signals) > 0:
    print(f"\n  张力范围: {short_signals['张力'].min():.3f} ~ {short_signals['张力'].max():.3f}")
    print(f"  加速度范围: {short_signals['加速度'].min():.3f} ~ {short_signals['加速度'].max():.3f}")
    print(f"\n  前10个信号:")
    for idx, row in short_signals.head(10).iterrows():
        print(f"    {row['时间']}: 张力={row['张力']:.4f}, 加速度={row['加速度']:.4f}, 价格={row['收盘价']:.2f}")

# 3. 统计符合LONG条件的信号（张力<-0.5, 加速度>0）
long_signals = signals_2025[
    (signals_2025['张力'] < -0.5) &
    (signals_2025['加速度'] > 0)
].copy()

print(f"\n符合LONG条件的信号（张力<-0.5, 加速度>0）:")
print(f"  数量: {len(long_signals)}条")
print(f"  占比: {len(long_signals)/len(signals_2025)*100:.2f}%")

if len(long_signals) > 0:
    print(f"\n  张力范围: {long_signals['张力'].min():.3f} ~ {long_signals['张力'].max():.3f}")
    print(f"  加速度范围: {long_signals['加速度'].min():.3f} ~ {long_signals['加速度'].max():.3f}")
    print(f"\n  前10个信号:")
    for idx, row in long_signals.head(10).iterrows():
        print(f"    {row['时间']}: 张力={row['张力']:.4f}, 加速度={row['加速度']:.4f}, 价格={row['收盘价']:.2f}")

# 4. 统计所有符合开仓条件的信号
all_entry_signals = pd.concat([short_signals, long_signals])
print(f"\n所有符合开仓条件的信号:")
print(f"  总数: {len(all_entry_signals)}条")
print(f"  SHORT: {len(short_signals)}条")
print(f"  LONG: {len(long_signals)}条")

# 5. 按月统计
short_signals['月份'] = short_signals['时间'].dt.to_period('M')
long_signals['月份'] = long_signals['时间'].dt.to_period('M')

print(f"\n按月统计SHORT信号:")
for month, group in short_signals.groupby('月份'):
    print(f"  {month}: {len(group)}条")

print(f"\n按月统计LONG信号:")
for month, group in long_signals.groupby('月份'):
    print(f"  {month}: {len(group)}条")

# 6. 保存符合条件的信号到Excel
with pd.ExcelWriter('符合条件的信号分析.xlsx', engine='openpyxl') as writer:
    short_signals.to_excel(writer, sheet_name='SHORT信号_全部', index=False)
    long_signals.to_excel(writer, sheet_name='LONG信号_全部', index=False)

    # 统计汇总
    summary = pd.DataFrame([
        ['数据周期', '2025-06-01 至 2025-12-31'],
        ['总信号数', len(signals_2025)],
        ['', ''],
        ['SHORT条件', '张力>0.5, 加速度<0'],
        ['SHORT信号数', len(short_signals)],
        ['SHORT占比', f'{len(short_signals)/len(signals_2025)*100:.2f}%'],
        ['', ''],
        ['LONG条件', '张力<-0.5, 加速度>0'],
        ['LONG信号数', len(long_signals)],
        ['LONG占比', f'{len(long_signals)/len(signals_2025)*100:.2f}%'],
        ['', ''],
        ['总计', len(all_entry_signals)],
        ['', ''],
        ['关键发现', ''],
        ['如果每个信号都开仓', f'应有{len(all_entry_signals)}笔交易'],
        ['但我们只有16笔', '说明逻辑有问题！'],
    ])
    summary.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

print("\n" + "="*100)
print("详细数据已保存到: 符合条件的信号分析.xlsx")
print("="*100)
