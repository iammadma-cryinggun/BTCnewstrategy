import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("深入分析：好机会的特征规律")
print("="*100)

# 读取已标注的数据
df = pd.read_excel('2025年6-12月_完整标注结果_SHORT+LONG.xlsx', sheet_name='所有标注')

print(f"\n总交易数: {len(df)}笔")
print(f"  - SHORT: {(df['交易方向']=='SHORT').sum()}笔")
print(f"  - LONG: {(df['交易方向']=='LONG').sum()}笔")

# 分别分析SHORT和LONG
for direction in ['SHORT', 'LONG']:
    print("\n" + "="*100)
    print(f"【{direction}信号】深度特征分析")
    print("="*100)

    data = df[df['交易方向'] == direction].copy()
    good = data[data['是好机会'] == '是']
    bad = data[data['是好机会'] == '否']

    print(f"\n好机会: {len(good)}笔")
    print(f"差机会: {len(bad)}笔")

    # 1. 张力/加速度比例关系
    print(f"\n" + "-"*100)
    print("1. 张力与加速度的比例关系")
    print("-"*100)

    if direction == 'SHORT':
        # SHORT: 加速度是负数，取绝对值
        good['张力_加速度比'] = good['首次张力'] / abs(good['首次加速度'])
        bad['张力_加速度比'] = bad['首次张力'] / abs(bad['首次加速度'])
    else:
        # LONG: 都是正数
        good['张力_加速度比'] = abs(good['首次张力']) / good['首次加速度']
        bad['张力_加速度比'] = abs(bad['首次张力']) / bad['首次加速度']

    print(f"好机会张力/加速度比:")
    print(f"  平均: {good['张力_加速度比'].mean():.2f}")
    print(f"  中位数: {good['张力_加速度比'].median():.2f}")
    print(f"  范围: [{good['张力_加速度比'].min():.2f}, {good['张力_加速度比'].max():.2f}]")

    print(f"\n差机会张力/加速度比:")
    print(f"  平均: {bad['张力_加速度比'].mean():.2f}")
    print(f"  中位数: {bad['张力_加速度比'].median():.2f}")
    print(f"  范围: [{bad['张力_加速度比'].min():.2f}, {bad['张力_加速度比'].max():.2f}]")

    # 按比例区间分组
    print(f"\n按张力/加速度比分组:")
    for threshold in [50, 100, 150, 200]:
        good_ratio = good[good['张力_加速度比'] >= threshold]
        bad_ratio = bad[bad['张力_加速度比'] >= threshold]
        if len(good_ratio) > 0 or len(bad_ratio) > 0:
            good_rate = len(good_ratio) / len(good) * 100 if len(good) > 0 else 0
            bad_rate = len(bad_ratio) / len(bad) * 100 if len(bad) > 0 else 0
            print(f"  比例≥{threshold}: 好机会{len(good_ratio)}笔({good_rate:.1f}%) vs 差机会{len(bad_ratio)}笔({bad_rate:.1f}%)")

    # 2. 首次信号张力分布
    print(f"\n" + "-"*100)
    print("2. 首次信号张力分布")
    print("-"*100)

    print(f"好机会首次张力:")
    print(f"  平均: {good['首次张力'].mean():.4f}")
    print(f"  中位数: {good['首次张力'].median():.4f}")
    print(f"  范围: [{good['首次张力'].min():.4f}, {good['首次张力'].max():.4f}]")

    print(f"\n差机会首次张力:")
    print(f"  平均: {bad['首次张力'].mean():.4f}")
    print(f"  中位数: {bad['首次张力'].median():.4f}")
    print(f"  范围: [{bad['首次张力'].min():.4f}, {bad['首次张力'].max():.4f}]")

    # 按张力区间分组
    print(f"\n按首次张力分组:")
    if direction == 'SHORT':
        ranges = [(0.5, 0.7), (0.7, 1.0), (1.0, 999)]
    else:
        ranges = [(-999, -0.7), (-0.7, -0.5), (-0.5, 0)]

    for min_t, max_t in ranges:
        if max_t == 999:
            good_range = good[good['首次张力'] >= min_t]
            bad_range = bad[bad['首次张力'] >= min_t]
            label = f"张力≥{min_t}"
        elif min_t == -999:
            good_range = good[good['首次张力'] < max_t]
            bad_range = bad[bad['首次张力'] < max_t]
            label = f"张力<{max_t}"
        else:
            good_range = good[(good['首次张力'] >= min_t) & (good['首次张力'] < max_t)]
            bad_range = bad[(bad['首次张力'] >= min_t) & (bad['首次张力'] < max_t)]
            label = f"{min_t}≤张力<{max_t}"

        good_in_range = len(good_range)
        bad_in_range = len(bad_range)
        total_in_range = good_in_range + bad_in_range

        if total_in_range > 0:
            good_rate = good_in_range / total_in_range * 100
            print(f"  {label}: 好机会{good_in_range}笔 vs 差机会{bad_in_range}笔 → 好机会率{good_rate:.1f}%")

    # 3. 首次信号加速度分布
    print(f"\n" + "-"*100)
    print("3. 首次信号加速度分布")
    print("-"*100)

    print(f"好机会首次加速度:")
    print(f"  平均: {good['首次加速度'].mean():.6f}")
    print(f"  中位数: {good['首次加速度'].median():.6f}")
    print(f"  绝对值平均: {abs(good['首次加速度']).mean():.6f}")

    print(f"\n差机会首次加速度:")
    print(f"  平均: {bad['首次加速度'].mean():.6f}")
    print(f"  中位数: {bad['首次加速度'].median():.6f}")
    print(f"  绝对值平均: {abs(bad['首次加速度']).mean():.6f}")

    # 4. 能量比率分布
    print(f"\n" + "-"*100)
    print("4. 能量比率分布")
    print("-"*100)

    print(f"好机会首次能量:")
    print(f"  平均: {good['首次能量'].mean():.2f}")
    print(f"  中位数: {good['首次能量'].median():.2f}")

    print(f"\n差机会首次能量:")
    print(f"  平均: {bad['首次能量'].mean():.2f}")
    print(f"  中位数: {bad['首次能量'].median():.2f}")

    # 按能量比率分组
    print(f"\n按首次能量分组:")
    energy_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 999)]
    for min_e, max_e in energy_ranges:
        if max_e == 999:
            good_range = good[good['首次能量'] >= min_e]
            bad_range = bad[bad['首次能量'] >= min_e]
            label = f"能量≥{min_e}"
        else:
            good_range = good[(good['首次能量'] >= min_e) & (good['首次能量'] < max_e)]
            bad_range = bad[(bad['首次能量'] >= min_e) & (bad['首次能量'] < max_e)]
            label = f"{min_e}≤能量<{max_e}"

        good_in_range = len(good_range)
        bad_in_range = len(bad_range)
        total_in_range = good_in_range + bad_in_range

        if total_in_range > 0:
            good_rate = good_in_range / total_in_range * 100
            print(f"  {label}: 好机会{good_in_range}笔 vs 差机会{bad_in_range}笔 → 好机会率{good_rate:.1f}%")

    # 5. 等待周期分析
    print(f"\n" + "-"*100)
    print("5. 等待周期分析")
    print("-"*100)

    print(f"好机会等待周期:")
    print(f"  平均: {good['等待周期'].mean():.1f}")
    print(f"  中位数: {good['等待周期'].median():.1f}")

    print(f"\n差机会等待周期:")
    print(f"  平均: {bad['等待周期'].mean():.1f}")
    print(f"  中位数: {bad['等待周期'].median():.1f}")

    # 按等待周期分组
    print(f"\n按等待周期分组:")
    for wait in [1, 2, 3, 4, 5, 6, 7]:
        good_wait = good[good['等待周期'] == wait]
        bad_wait = bad[bad['等待周期'] == wait]
        total_wait = len(good_wait) + len(bad_wait)

        if total_wait > 0:
            good_rate = len(good_wait) / total_wait * 100
            print(f"  等待{wait}周期: 好机会{len(good_wait)}笔 vs 差机会{len(bad_wait)}笔 → 好机会率{good_rate:.1f}%")

    # 6. 张力变化与价格优势的关系
    print(f"\n" + "-"*100)
    print("6. 张力变化与价格优势的关系")
    print("-"*100)

    # 好机会中，主要是靠张力还是靠价格优势？
    good_by_tension = good[good['张力变化%'] > 5]
    good_by_price = good[good['价格优势%'] > 0.5]
    good_both = good[(good['张力变化%'] > 5) & (good['价格优势%'] > 0.5)]

    print(f"好机会分类:")
    print(f"  主要靠张力(>5%): {len(good_by_tension)}笔 ({len(good_by_tension)/len(good)*100:.1f}%)")
    print(f"  主要靠价格优势(>0.5%): {len(good_by_price)}笔 ({len(good_by_price)/len(good)*100:.1f}%)")
    print(f"  两者兼具: {len(good_both)}笔 ({len(good_both)/len(good)*100:.1f}%)")

    # 7. 综合评分模型
    print(f"\n" + "-"*100)
    print("7. 综合特征总结")
    print("-"*100)

    print(f"\n好机会典型特征:")
    print(f"  张力/加速度比: {good['张力_加速度比'].mean():.1f} vs {bad['张力_加速度比'].mean():.1f} (差)")
    if good['张力_加速度比'].mean() > bad['张力_加速度比'].mean():
        print(f"  → 好机会的比例更高{' ✓' if good['张力_加速度比'].mean() > bad['张力_加速度比'].mean() * 1.2 else ''}")

    print(f"  首次张力: {good['首次张力'].mean():.4f} vs {bad['首次张力'].mean():.4f} (差)")
    if direction == 'SHORT':
        if good['首次张力'].mean() > bad['首次张力'].mean():
            print(f"  → 好机会的张力更高{' ✓' if good['首次张力'].mean() > bad['首次张力'].mean() * 1.1 else ''}")
    else:
        if abs(good['首次张力'].mean()) > abs(bad['首次张力'].mean()):
            print(f"  → 好机会的张力绝对值更大{' ✓' if abs(good['首次张力'].mean()) > abs(bad['首次张力'].mean()) * 1.1 else ''}")

    print(f"  首次能量: {good['首次能量'].mean():.2f} vs {bad['首次能量'].mean():.2f} (差)")
    if good['首次能量'].mean() < bad['首次能量'].mean():
        print(f"  → 好机会的能量更低（缩量）{' ✓' if bad['首次能量'].mean() > good['首次能量'].mean() * 1.2 else ''}")

# 保存特征分析到Excel
print(f"\n" + "="*100)
print("保存特征分析结果...")
print("="*100)

output_file = '特征分析_好机会vs差机会.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # SHORT
    short_data = df[df['交易方向'] == 'SHORT'].copy()
    short_data['张力_加速度比'] = short_data['首次张力'] / abs(short_data['首次加速度'])
    short_data.to_excel(writer, sheet_name='SHORT特征', index=False)

    short_good = short_data[short_data['是好机会'] == '是']
    short_bad = short_data[short_data['是好机会'] == '否']
    short_good.to_excel(writer, sheet_name='SHORT好机会', index=False)
    short_bad.to_excel(writer, sheet_name='SHORT差机会', index=False)

    # LONG
    long_data = df[df['交易方向'] == 'LONG'].copy()
    long_data['张力_加速度比'] = abs(long_data['首次张力']) / long_data['首次加速度']
    long_data.to_excel(writer, sheet_name='LONG特征', index=False)

    long_good = long_data[long_data['是好机会'] == '是']
    long_bad = long_data[long_data['是好机会'] == '否']
    long_good.to_excel(writer, sheet_name='LONG好机会', index=False)
    long_bad.to_excel(writer, sheet_name='LONG差机会', index=False)

print(f"\n已保存到: {output_file}")

print("\n" + "="*100)
print("分析完成！")
print("="*100)
