import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("按完整策略重新分析：张力变化趋势的影响")
print("="*100)

# 读取数据
df = pd.read_excel('完整对比_直接开仓vs确认开仓.xlsx', sheet_name='所有信号对比', engine='openpyxl')

# 只看通过确认的（第9列：策略B_确认开仓_找到确认）
confirmed = df[df.iloc[:, 9] == True].copy()

print(f"\n通过确认的交易: {len(confirmed)}个")

# 列映射（基于索引）
# 0: 首次信号时间
# 1: 首次信号张力
# 2: 首次信号加速度
# 3: 首次信号量能
# 4: 首次信号价格
# 5: 策略A_直接开仓_盈亏%
# 6: 策略A_持仓周期
# 7: 策略A_平仓原因
# 8: 策略B_确认开仓_找到确认
# 9: 策略B_等待周期
# 10: 策略B_开仓价格
# 11: 策略B_开仓张力
# 12: 策略B_盈亏%
# 13: 策略B_总持仓周期
# 14: 策略B_平仓原因
# 15: 价格优势%

# 计算张力变化
confirmed['张力变化%'] = (confirmed.iloc[:, 11] - confirmed.iloc[:, 1]) / confirmed.iloc[:, 1] * 100

# 计算价格优势（对于SHORT：首次价格 - 开仓价格，正数=更优）
confirmed['价格优势%'] = (confirmed.iloc[:, 4] - confirmed.iloc[:, 10]) / confirmed.iloc[:, 4] * 100

print(f"\n张力变化分布:")
print(f"  平均: {confirmed['张力变化%'].mean():+.2f}%")
print(f"  范围: [{confirmed['张力变化%'].min():+.2f}%, {confirmed['张力变化%'].max():+.2f}%]")

# 按张力变化分组
tension_rising = confirmed[confirmed['张力变化%'] > 0]  # 张力上升
tension_falling = confirmed[confirmed['张力变化%'] <= 0]  # 张力下降或持平

print(f"\n" + "="*100)
print("按张力变化分组分析")
print("="*100)

print(f"\n张力上升的 ({len(tension_rising)}个):")
print(f"  平均张力变化: {tension_rising['张力变化%'].mean():+.2f}%")
print(f"  平均价格优势: {tension_rising['价格优势%'].mean():+.3f}%")
print(f"  有价格优势: {(tension_rising['价格优势%'] > 0).sum()}个 ({(tension_rising['价格优势%'] > 0).sum()/len(tension_rising)*100:.1f}%)")

print(f"\n张力下降的 ({len(tension_falling)}个):")
print(f"  平均张力变化: {tension_falling['张力变化%'].mean():+.2f}%")
print(f"  平均价格优势: {tension_falling['价格优势%'].mean():+.3f}%")
print(f"  有价格优势: {(tension_falling['价格优势%'] > 0).sum()}个 ({(tension_falling['价格优势%'] > 0).sum()/len(tension_falling)*100:.1f}%)")

# 分析：张力大幅上升 vs 小幅上升
print(f"\n" + "="*100)
print("关键发现：张力变化与价格优势的关系")
print("="*100)

# 张力大幅上升 vs 小幅上升
big_rise = confirmed[confirmed['张力变化%'] > 5]  # 张力上升>5%
small_rise = confirmed[(confirmed['张力变化%'] > 0) & (confirmed['张力变化%'] <= 5)]
big_fall = confirmed[confirmed['张力变化%'] < -5]  # 张力下降>5%
small_fall = confirmed[(confirmed['张力变化%'] >= -5) & (confirmed['张力变化%'] <= 0)]

print(f"\n张力大幅上升 (>5%): {len(big_rise)}个")
if len(big_rise) > 0:
    print(f"  平均张力变化: {big_rise['张力变化%'].mean():+.2f}%")
    print(f"  平均价格优势: {big_rise['价格优势%'].mean():+.3f}%")
    print(f"  有价格优势: {(big_rise['价格优势%'] > 0).sum()}/{len(big_rise)} = {(big_rise['价格优势%'] > 0).sum()/len(big_rise)*100:.1f}%")

print(f"\n张力小幅上升 (0~5%): {len(small_rise)}个")
if len(small_rise) > 0:
    print(f"  平均张力变化: {small_rise['张力变化%'].mean():+.2f}%")
    print(f"  平均价格优势: {small_rise['价格优势%'].mean():+.3f}%")
    print(f"  有价格优势: {(small_rise['价格优势%'] > 0).sum()}/{len(small_rise)} = {(small_rise['价格优势%'] > 0).sum()/len(small_rise)*100:.1f}%")

print(f"\n张力小幅下降 (0~-5%): {len(small_fall)}个")
if len(small_fall) > 0:
    print(f"  平均张力变化: {small_fall['张力变化%'].mean():+.2f}%")
    print(f"  平均价格优势: {small_fall['价格优势%'].mean():+.3f}%")
    print(f"  有价格优势: {(small_fall['价格优势%'] > 0).sum()}/{len(small_fall)} = {(small_fall['价格优势%'] > 0).sum()/len(small_fall)*100:.1f}%")

print(f"\n张力大幅下降 (<-5%): {len(big_fall)}个")
if len(big_fall) > 0:
    print(f"  平均张力变化: {big_fall['张力变化%'].mean():+.2f}%")
    print(f"  平均价格优势: {big_fall['价格优势%'].mean():+.3f}%")
    print(f"  有价格优势: {(big_fall['价格优势%'] > 0).sum()}/{len(big_fall)} = {(big_fall['价格优势%'] > 0).sum()/len(big_fall)*100:.1f}%")

# 分析：你总结的5笔交易的张力变化特点
print(f"\n" + "="*100)
print("你总结的5笔交易的张力变化")
print("="*100)

your_trades = pd.DataFrame([
    {'交易': '交易1', '首次张力': 0.919923, '开仓张力': 1.014203, '变化%': 10.2},
    {'交易': '交易2', '首次张力': 0.510574, '开仓张力': 0.473982, '变化%': -7.2},
    {'交易': '交易3', '首次张力': -0.508689, '开仓张力': -0.553677, '变化%': -8.8},  # LONG
    {'交易': '交易4', '首次张力': 0.714732, '开仓张力': 0.850444, '变化%': 19.0},
    {'交易': '交易5', '首次张力': 0.583329, '开仓张力': 0.658092, '变化%': 12.8},
])

print("\nSHORT交易（4笔）:")
short_your = your_trades[your_trades['交易'].str.contains('交易[1-5]')]
for idx, row in your_trades.iterrows():
    print(f"  {row['交易']}: 张力 {row['首次张力']:.4f} → {row['开仓张力']:.4f} ({row['变化%']:+.1f}%)")

print(f"\nSHORT交易张力变化:")
print(f"  上升: 3笔（交易1, 4, 5）")
print(f"  下降: 1笔（交易2）")
print(f"  平均: {(10.2 + 19.0 + 12.8 - 7.2)/4:+.1f}%")

# 对比2025年6-12月的数据
print(f"\n" + "="*100)
print("2025年6-12月 vs 你总结的交易")
print("="*100)

print(f"\n你总结的4笔SHORT交易:")
print(f"  张力上升: 3笔 (75%)")
print(f"  张力下降: 1笔 (25%)")
print(f"  平均变化: +8.7%")

print(f"\n2025年6-12月通过确认的{len(confirmed)}笔交易:")
print(f"  张力上升: {len(tension_rising)}笔 ({len(tension_rising)/len(confirmed)*100:.1f}%)")
print(f"  张力下降: {len(tension_falling)}笔 ({len(tension_falling)/len(confirmed)*100:.1f}%)")
print(f"  平均变化: {confirmed['张力变化%'].mean():+.2f}%")

print(f"\n关键差异:")
print(f"  你的交易: 张力倾向于上升 (+8.7%)")
print(f"  6-12月: 张力倾向于{('上升' if confirmed['张力变化%'].mean() > 0 else '下降')} ({confirmed['张力变化%'].mean():+.2f}%)")
if confirmed['张力变化%'].mean() < 8.7:
    print(f"  这可能就是为什么6-12月的好机会少！")
