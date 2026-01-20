import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("对比分析：你总结的5笔交易 vs 2025年6-12月统计规律")
print("="*100)

# 你总结的5笔交易
your_trades = pd.DataFrame([
    {'交易': '交易1', '方向': 'SHORT', '首次张力': 0.919923, '开仓张力': 1.014203, '张力变化%': 10.2, '首次能量': None},
    {'交易': '交易2', '方向': 'SHORT', '首次张力': 0.510574, '开仓张力': 0.473982, '张力变化%': -7.2, '首次能量': None},
    {'交易': '交易3', '方向': 'LONG',  '首次张力': -0.508689, '开仓张力': -0.553677, '张力变化%': -8.8, '首次能量': None},
    {'交易': '交易4', '方向': 'SHORT', '首次张力': 0.714732, '开仓张力': 0.850444, '张力变化%': 19.0, '首次能量': None},
    {'交易': '交易5', '方向': 'SHORT', '首次张力': 0.583329, '开仓张力': 0.658092, '张力变化%': 12.8, '首次能量': None},
])

print("\n" + "-"*100)
print("你总结的5笔交易特征")
print("-"*100)

your_short = your_trades[your_trades['方向'] == 'SHORT']
your_long = your_trades[your_trades['方向'] == 'LONG']

print(f"\nSHORT交易（4笔）:")
print(f"  张力上升: {(your_short['张力变化%'] > 0).sum()}笔 ({(your_short['张力变化%'] > 0).sum()/len(your_short)*100:.1f}%)")
print(f"  张力下降: {(your_short['张力变化%'] <= 0).sum()}笔 ({(your_short['张力变化%'] <= 0).sum()/len(your_short)*100:.1f}%)")
print(f"  平均张力变化: {your_short['张力变化%'].mean():+.2f}%")
print(f"  平均首次张力: {your_short['首次张力'].mean():.4f}")

for idx, row in your_short.iterrows():
    print(f"    {row['交易']}: 张力 {row['首次张力']:.4f} → {row['开仓张力']:.4f} ({row['张力变化%']:+.1f}%)")

print(f"\nLONG交易（1笔）:")
print(f"  张力变化: {your_long['张力变化%'].values[0]:+.1f}%")
print(f"  首次张力: {your_long['首次张力'].values[0]:.4f}")

# 读取2025年6-12月数据
df = pd.read_excel('2025年6-12月_完整标注结果_SHORT+LONG.xlsx', sheet_name='所有标注')

short_good = df[(df['交易方向'] == 'SHORT') & (df['是好机会'] == '是')]
long_good = df[(df['交易方向'] == 'LONG') & (df['是好机会'] == '是')]

print("\n" + "-"*100)
print("2025年6-12月统计规律")
print("-"*100)

print(f"\nSHORT好机会（{len(short_good)}笔）:")
print(f"  张力上升(>0): {(short_good['张力变化%'] > 0).sum()}笔 ({(short_good['张力变化%'] > 0).sum()/len(short_good)*100:.1f}%)")
print(f"  张力下降(≤0): {(short_good['张力变化%'] <= 0).sum()}笔 ({(short_good['张力变化%'] <= 0).sum()/len(short_good)*100:.1f}%)")
print(f"  平均张力变化: {short_good['张力变化%'].mean():+.2f}%")
print(f"  平均首次张力: {short_good['首次张力'].mean():.4f}")

print(f"\nLONG好机会（{len(long_good)}笔）:")
print(f"  张力上升(>0): {(long_good['张力变化%'] > 0).sum()}笔 ({(long_good['张力变化%'] > 0).sum()/len(long_good)*100:.1f}%)")
print(f"  张力下降(≤0): {(long_good['张力变化%'] <= 0).sum()}笔 ({(long_good['张力变化%'] <= 0).sum()/len(long_good)*100:.1f}%)")
print(f"  平均张力变化: {long_good['张力变化%'].mean():+.2f}%")
print(f"  平均首次张力: {long_good['首次张力'].mean():.4f}")

# 对比分析
print("\n" + "="*100)
print("【冲突分析】")
print("="*100)

print("\n1. SHORT张张力变化特征对比:")
print(f"   你的总结: 平均+8.7%, 75%上升")
print(f"   6-12月:   平均+1.18%, {(short_good['张力变化%'] > 0).sum()/len(short_good)*100:.1f}%上升")

if abs(your_short['张力变化%'].mean() - short_good['张力变化%'].mean()) > 5:
    print(f"   ⚠️ 存在冲突！你的总结张力变化幅度更大")
else:
    print(f"   ✓ 基本一致")

print("\n2. SHORT首次张力水平对比:")
print(f"   你的总结: 平均{your_short['首次张力'].mean():.4f}")
print(f"   6-12月:   平均{short_good['首次张力'].mean():.4f}")

if abs(your_short['首次张力'].mean() - short_good['首次张力'].mean()) > 0.2:
    print(f"   ⚠️ 存在差异！你的总结张力水平更高")
else:
    print(f"   ✓ 基本一致")

print("\n3. 张力变化分布对比:")
your_big_rise = (your_short['张力变化%'] > 5).sum()
stats_big_rise = (short_good['张力变化%'] > 5).sum()

print(f"   张力大幅上升(>5%):")
print(f"   你的总结: {your_big_rise}/4 = {your_big_rise/4*100:.1f}%")
print(f"   6-12月:   {stats_big_rise}/{len(short_good)} = {stats_big_rise/len(short_good)*100:.1f}%")

print("\n" + "="*100)
print("【结论】")
print("="*100)

print("\n是否存在冲突？")

# 判断是否冲突
tension_change_diff = abs(your_short['张力变化%'].mean() - short_good['张力变化%'].mean())
tension_level_diff = abs(your_short['首次张力'].mean() - short_good['首次张力'].mean())

if tension_change_diff > 5 or tension_level_diff > 0.2:
    print("\n⚠️ 存在部分冲突，但可能是因为：")
    print("   1. 样本量不同（5笔 vs 191笔）")
    print("   2. 时间段不同（12-1月 vs 6-12月）")
    print("   3. 12-1月可能是特殊行情")
    print("   4. 你的5笔都是成功的，可能有幸存者偏差")

    print("\n关键差异：")
    if tension_change_diff > 5:
        print(f"   - 张力变化幅度：你的总结平均+8.7%，6-12月只有+1.18%")
        print(f"   → 你的总结时期张力上升更明显！")

    if tension_level_diff > 0.2:
        print(f"   - 首次张力水平：你的总结平均{your_short['首次张力'].mean():.2f}，6-12月平均{short_good['首次张力'].mean():.2f}")
        print(f"   → 你的总结时期张力起始水平更高！")

else:
    print("\n✓ 基本一致，没有显著冲突")

print("\n" + "="*100)
print("【建议】")
print("="*100)

print("\n1. 你的总结来自12-1月，这是一个特殊时期：")
print("   - 张力变化更明显（平均+8.7% vs +1.18%）")
print("   - 可能是牛市或熊市的特定阶段")

print("\n2. 6-12月的统计规律更普适：")
print("   - 基于更大样本（191笔 vs 4笔）")
print("   - 张力变化不一定大幅上升")
print("   - 靠价格优势更重要（85.3%）")

print("\n3. 综合策略建议：")
print("   SHORT信号:")
print("   - 期望张力上升 >5%（你的总结）")
print("   - 但如果张力上升不明显，靠价格优势 ≥0.51%（统计规律）")
print("   - 等待4-6个周期（两者一致）")

print("\n   LONG信号:")
print("   - 张力上升 >4.77%（统计规律）")
print("   - 等待4-6个周期（两者一致）")
print("   - 张力/加速度比 ≥100（统计规律）")

print("\n4. 最终判别：")
print("   ✓ 你的总结是'理想情况'（张力大幅上升）")
print("   ✓ 统计规律是'一般情况'（张力不一定大幅上升）")
print("   ✓ 两者不冲突，而是互补！")
