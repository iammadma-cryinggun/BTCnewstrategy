import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("简单直接的价格优势对比")
print("="*100)

# 读取数据
df = pd.read_excel('完整对比_直接开仓vs确认开仓.xlsx', sheet_name='所有信号对比')

# 只看通过确认的
confirmed = df[df['策略B_确认开仓_找到确认'] == True].copy()

print(f"\n通过确认的交易: {len(confirmed)}个")

# 计算价格优势
# 对于SHORT：首次信号价格 - 开仓价格 = 价格优势（正数=更优）
# 对于SHORT：开仓价格更低更好
price_advantage = (confirmed['首次信号价格'] - confirmed['策略B_开仓价格']) / confirmed['首次信号价格'] * 100

confirmed = confirmed.copy()
confirmed['价格优势%'] = price_advantage

print(f"\n价格优势统计:")
print(f"  平均: {price_advantage.mean():+.3f}%")
print(f"  中位数: {price_advantage.median():+.3f}%")
print(f"  最大优势: {price_advantage.max():+.3f}%")
print(f"  最大劣势: {price_advantage.min():+.3f}%")
print(f"  有优势: {(price_advantage > 0).sum()}个 ({(price_advantage > 0).sum()/len(confirmed)*100:.1f}%)")
print(f"  有劣势: {(price_advantage < 0).sum()}个 ({(price_advantage < 0).sum()/len(confirmed)*100:.1f}%)")
print(f"  持平: {(price_advantage == 0).sum()}个 ({(price_advantage == 0).sum()/len(confirmed)*100:.1f}%)")

# 按价格优势排序
confirmed_sorted = confirmed.sort_values('价格优势%', ascending=False)

print(f"\n" + "="*100)
print(f"价格优势排名 Top 20")
print("="*100)

for idx, row in confirmed_sorted.head(20).iterrows():
    print(f"\n{row['首次信号时间']}")
    print(f"  首次信号: {row['首次信号价格']:.2f} (张力={row['首次信号张力']:.4f})")
    print(f"  等待{row['策略B_等待周期']:.0f}周期后: {row['策略B_开仓价格']:.2f} (张力={row['策略B_开仓张力']:.4f})")
    print(f"  价格优势: {row['价格优势%']:+.3f}% ({'更优' if row['价格优势%'] > 0 else '更差' if row['价格优势%'] < 0 else '持平'})")

# 按价格劣势排序
print(f"\n" + "="*100)
print(f"价格劣势排名 Top 20 (等待后反而更差)")
print("="*100)

confirmed_worst = confirmed.sort_values('价格优势%', ascending=True)

for idx, row in confirmed_worst.head(20).iterrows():
    print(f"\n{row['首次信号时间']}")
    print(f"  首次信号: {row['首次信号价格']:.2f} (张力={row['首次信号张力']:.4f})")
    print(f"  等待{row['策略B_等待周期']:.0f}周期后: {row['策略B_开仓价格']:.2f} (张力={row['策略B_开仓张力']:.4f})")
    print(f"  价格优势: {row['价格优势%']:+.3f}% ({'更优' if row['价格优势%'] > 0 else '更差' if row['价格优势%'] < 0 else '持平'})")

# 分析价格优势与张力的关系
print(f"\n" + "="*100)
print("价格优势与张力变化的关系")
print("="*100)

confirmed['张力变化%'] = (confirmed['策略B_开仓张力'] - confirmed['首次信号张力']) / confirmed['首次信号张力'] * 100

# 按价格优势分组
excellent = confirmed[confirmed['价格优势%'] > 0.5]
good = confirmed[(confirmed['价格优势%'] > 0) & (confirmed['价格优势%'] <= 0.5)]
bad = confirmed[confirmed['价格优势%'] < 0]

print(f"\n价格优势 > 0.5%: {len(excellent)}个")
if len(excellent) > 0:
    print(f"  平均首次张力: {excellent['首次信号张力'].mean():.4f}")
    print(f"  平均开仓张力: {excellent['策略B_开仓张力'].mean():.4f}")
    print(f"  平均张力变化: {excellent['张力变化%'].mean():+.2f}%")

print(f"\n价格优势 0~0.5%: {len(good)}个")
if len(good) > 0:
    print(f"  平均首次张力: {good['首次信号张力'].mean():.4f}")
    print(f"  平均开仓张力: {good['策略B_开仓张力'].mean():.4f}")
    print(f"  平均张力变化: {good['张力变化%'].mean():+.2f}%")

print(f"\n价格劣势 < 0%: {len(bad)}个")
if len(bad) > 0:
    print(f"  平均首次张力: {bad['首次信号张力'].mean():.4f}")
    print(f"  平均开仓张力: {bad['策略B_开仓张力'].mean():.4f}")
    print(f"  平均张力变化: {bad['张力变化%'].mean():+.2f}%")

print(f"\n" + "="*100)
print("结论")
print("="*100)

print(f"\n等待确认后开仓:")
print(f"  平均价格优势: {price_advantage.mean():+.3f}%")
print(f"  有价格优势的占: {(price_advantage > 0).sum()/len(confirmed)*100:.1f}%")
print(f"  有价格劣势的占: {(price_advantage < 0).sum()/len(confirmed)*100:.1f}%")

if price_advantage.mean() > 0:
    print(f"\n  ✅ 等待确认后，平均价格优势{price_advantage.mean():.3f}%")
    print(f"  → 对于SHORT交易，开仓价格平均更低，更有利！")
elif price_advantage.mean() < 0:
    print(f"\n  ❌ 等待确认后，平均价格劣势{abs(price_advantage.mean()):.3f}%")
    print(f"  → 对于SHORT交易，开仓价格平均更高，更不利！")
else:
    print(f"\n  ➖ 等待确认后，价格几乎持平")
    print(f"  → 等待确认对价格影响不大")
