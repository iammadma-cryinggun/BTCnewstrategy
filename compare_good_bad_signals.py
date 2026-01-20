import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("对比好机会和差机会的参数差异")
print("找出更精确的过滤条件")
print("="*100)

# 读取分析结果
short_analysis = pd.read_excel('所有SHORT信号分析.xlsx', sheet_name='所有SHORT信号分析')

# 筛选通过确认的交易
confirmed = short_analysis[short_analysis['确认通过'] == '是'].copy()

good_trades = confirmed[confirmed['是好机会'] == '是']
bad_trades = confirmed[confirmed['是好机会'] == '否']

print(f"\n好机会: {len(good_trades)}笔")
print(f"差机会: {len(bad_trades)}笔")

# 对比参数
print("\n" + "="*100)
print("参数对比分析")
print("="*100)

params_to_compare = [
    '开仓张力',
    '开仓加速度',
    '开仓量能',
    '首次信号张力',
    '首次信号加速度',
]

for param in params_to_compare:
    if param in good_trades.columns and param in bad_trades.columns:
        good_avg = good_trades[param].mean()
        bad_avg = bad_trades[param].mean()
        good_min = good_trades[param].min()
        good_max = good_trades[param].max()
        bad_min = bad_trades[param].min()
        bad_max = bad_trades[param].max()

        print(f"\n{param}:")
        print(f"  好机会: 平均={good_avg:.4f}, 范围=[{good_min:.4f}, {good_max:.4f}]")
        print(f"  差机会: 平均={bad_avg:.4f}, 范围=[{bad_min:.4f}, {bad_max:.4f}]")
        print(f"  差异: {good_avg - bad_avg:+.4f}")

# 分析盈亏分布
print("\n" + "="*100)
print("盈亏分布分析")
print("="*100)

print(f"\n好机会盈亏分布:")
print(f"  平均: {good_trades['盈亏%'].mean():.2f}%")
print(f"  中位数: {good_trades['盈亏%'].median():.2f}%")
print(f"  最大: {good_trades['盈亏%'].max():.2f}%")
print(f"  最小: {good_trades['盈亏%'].min():.2f}%")

print(f"\n差机会盈亏分布:")
print(f"  平均: {bad_trades['盈亏%'].mean():.2f}%")
print(f"  中位数: {bad_trades['盈亏%'].median():.2f}%")
print(f"  最大: {bad_trades['盈亏%'].max():.2f}%")
print(f"  最小: {bad_trades['盈亏%'].min():.2f}%")

# 寻找最佳张力阈值
print("\n" + "="*100)
print("寻找最佳张力阈值")
print("="*100)

tension_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

for threshold in tension_thresholds:
    filtered = confirmed[confirmed['开仓张力'] >= threshold]
    if len(filtered) > 0:
        good_count = (filtered['是好机会'] == '是').sum()
        total_count = len(filtered)
        good_ratio = good_count / total_count * 100
        avg_pnl = filtered['盈亏%'].mean()

        print(f"\n张力 >= {threshold}:")
        print(f"  交易数: {total_count}")
        print(f"  好机会占比: {good_ratio:.1f}%")
        print(f"  平均盈亏: {avg_pnl:.2f}%")

# 寻找最佳加速度阈值
print("\n" + "="*100)
print("寻找最佳加速度阈值")
print("="*100)

accel_thresholds = [-0.002, -0.003, -0.004, -0.005, -0.006, -0.007, -0.008, -0.009, -0.01]

for threshold in accel_thresholds:
    filtered = confirmed[confirmed['开仓加速度'] <= threshold]
    if len(filtered) > 0:
        good_count = (filtered['是好机会'] == '是').sum()
        total_count = len(filtered)
        good_ratio = good_count / total_count * 100
        avg_pnl = filtered['盈亏%'].mean()

        print(f"\n加速度 <= {threshold}:")
        print(f"  交易数: {total_count}")
        print(f"  好机会占比: {good_ratio:.1f}%")
        print(f"  平均盈亏: {avg_pnl:.2f}%")

# 组合条件测试
print("\n" + "="*100)
print("组合条件测试")
print("="*100)

# 测试: 张力>0.6 且 加速度<-0.005
test1 = confirmed[
    (confirmed['开仓张力'] > 0.6) &
    (confirmed['开仓加速度'] < -0.005)
]
if len(test1) > 0:
    good_count = (test1['是好机会'] == '是').sum()
    print(f"\n条件1: 张力>0.6 且 加速度<-0.005")
    print(f"  交易数: {len(test1)}")
    print(f"  好机会: {good_count}笔 ({good_count/len(test1)*100:.1f}%)")
    print(f"  平均盈亏: {test1['盈亏%'].mean():.2f}%")

# 测试: 张力>0.65 且 加速度<-0.006
test2 = confirmed[
    (confirmed['开仓张力'] > 0.65) &
    (confirmed['开仓加速度'] < -0.006)
]
if len(test2) > 0:
    good_count = (test2['是好机会'] == '是').sum()
    print(f"\n条件2: 张力>0.65 且 加速度<-0.006")
    print(f"  交易数: {len(test2)}")
    print(f"  好机会: {good_count}笔 ({good_count/len(test2)*100:.1f}%)")
    print(f"  平均盈亏: {test2['盈亏%'].mean():.2f}%")

# 测试: 张力>0.7 且 加速度<-0.007
test3 = confirmed[
    (confirmed['开仓张力'] > 0.7) &
    (confirmed['开仓加速度'] < -0.007)
]
if len(test3) > 0:
    good_count = (test3['是好机会'] == '是').sum()
    print(f"\n条件3: 张力>0.7 且 加速度<-0.007")
    print(f"  交易数: {len(test3)}")
    print(f"  好机会: {good_count}笔 ({good_count/len(test3)*100:.1f}%)")
    print(f"  平均盈亏: {test3['盈亏%'].mean():.2f}%")

# 保存详细对比到Excel
with pd.ExcelWriter('好机会vs差机会_参数对比.xlsx', engine='openpyxl') as writer:
    good_trades.to_excel(writer, sheet_name='好机会_全部', index=False)
    bad_trades.to_excel(writer, sheet_name='差机会_全部', index=False)

    # 参数对比表
    comparison_data = []
    for param in params_to_compare:
        if param in good_trades.columns and param in bad_trades.columns:
            comparison_data.append({
                '参数': param,
                '好机会_平均': good_trades[param].mean(),
                '好机会_最小': good_trades[param].min(),
                '好机会_最大': good_trades[param].max(),
                '差机会_平均': bad_trades[param].mean(),
                '差机会_最小': bad_trades[param].min(),
                '差机会_最大': bad_trades[param].max(),
                '差异': good_trades[param].mean() - bad_trades[param].mean(),
            })

    pd.DataFrame(comparison_data).to_excel(writer, sheet_name='参数对比', index=False)

print("\n" + "="*100)
print("详细对比已保存到: 好机会vs差机会_参数对比.xlsx")
print("="*100)
