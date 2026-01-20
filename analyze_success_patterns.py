import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)

print("=" * 100)
print("BTC 4小时信号成功开单规律分析")
print("=" * 100)

# 读取信号日志
df = pd.read_csv('btc_4h_signals_matching_log.csv', encoding='utf-8-sig')

print(f"\n总信号数: {len(df)}")
print(f"数据列: {df.columns.tolist()}")

# 统计开单情况
opened = df[df['是否开单'] == '是']
not_opened = df[df['是否开单'] == '否']

print(f"\n" + "=" * 100)
print("开单情况统计")
print("=" * 100)
print(f"开单数: {len(opened)} ({len(opened)/len(df)*100:.2f}%)")
print(f"未开单数: {len(not_opened)} ({len(not_opened)/len(df)*100:.2f}%)")

# 分析开单信号的特征
print(f"\n" + "=" * 100)
print("成功开单信号特征分析")
print("=" * 100)

if len(opened) > 0:
    print(f"\n1. 信号类型分布:")
    print(opened['信号类型'].value_counts().to_string())

    print(f"\n2. 交易方向分布:")
    print(opened['交易方向'].value_counts().to_string())

    print(f"\n3. 置信度分布:")
    print(opened['置信度'].value_counts().to_string())

    print(f"\n4. 通过过滤器情况:")
    print(opened['通过V705过滤器'].value_counts().to_string())

    # 数值特征统计
    print(f"\n5. 张力统计:")
    print(f"   最小值: {opened['张力'].min():.4f}")
    print(f"   最大值: {opened['张力'].max():.4f}")
    print(f"   平均值: {opened['张力'].mean():.4f}")
    print(f"   中位数: {opened['张力'].median():.4f}")
    print(f"   标准差: {opened['张力'].std():.4f}")

    print(f"\n6. 加速度统计:")
    print(f"   最小值: {opened['加速度'].min():.6f}")
    print(f"   最大值: {opened['加速度'].max():.6f}")
    print(f"   平均值: {opened['加速度'].mean():.6f}")
    print(f"   中位数: {opened['加速度'].median():.6f}")
    print(f"   标准差: {opened['加速度'].std():.6f}")

    print(f"\n7. 量能比率统计:")
    print(f"   最小值: {opened['量能比率'].min():.4f}")
    print(f"   最大值: {opened['量能比率'].max():.4f}")
    print(f"   平均值: {opened['量能比率'].mean():.4f}")
    print(f"   中位数: {opened['量能比率'].median():.4f}")
    print(f"   标准差: {opened['量能比率'].std():.4f}")

    print(f"\n8. EMA偏离%统计:")
    print(f"   最小值: {opened['EMA偏离%'].min():.4f}")
    print(f"   最大值: {opened['EMA偏离%'].max():.4f}")
    print(f"   平均值: {opened['EMA偏离%'].mean():.4f}")
    print(f"   中位数: {opened['EMA偏离%'].median():.4f}")
    print(f"   标准差: {opened['EMA偏离%'].std():.4f}")

    # 按交易方向分组分析
    print(f"\n" + "=" * 100)
    print("按交易方向分组分析")
    print("=" * 100)

    for direction in opened['交易方向'].unique():
        subset = opened[opened['交易方向'] == direction]
        print(f"\n【{direction.upper()}】信号 (共{len(subset)}个):")
        print(f"  张力: 平均={subset['张力'].mean():.4f}, 范围=[{subset['张力'].min():.4f}, {subset['张力'].max():.4f}]")
        print(f"  加速度: 平均={subset['加速度'].mean():.6f}, 范围=[{subset['加速度'].min():.6f}, {subset['加速度'].max():.6f}]")
        print(f"  量能比率: 平均={subset['量能比率'].mean():.4f}, 范围=[{subset['量能比率'].min():.4f}, {subset['量能比率'].max():.4f}]")
        print(f"  EMA偏离%: 平均={subset['EMA偏离%'].mean():.4f}, 范围=[{subset['EMA偏离%'].min():.4f}, {subset['EMA偏离%'].max():.4f}]")

    # 张力区间分析
    print(f"\n" + "=" * 100)
    print("张力区间分析")
    print("=" * 100)

    opened_copy = opened.copy()
    opened_copy.loc[:, '张力区间'] = pd.cut(opened_copy['张力'],
                                             bins=[-1, -0.5, -0.35, 0, 0.35, 0.5, 1],
                                             labels=['<-0.5', '-0.5~-0.35', '-0.35~0', '0~0.35', '0.35~0.5', '>0.5'])

    tension_analysis = opened_copy.groupby(['张力区间', '交易方向']).agg({
        '时间': 'count',
        '张力': 'mean',
        '加速度': 'mean',
        '量能比率': 'mean'
    }).round(4)
    tension_analysis.columns = ['数量', '平均张力', '平均加速度', '平均量能']
    print(tension_analysis.to_string())

    # 未开单信号的特征分析
    print(f"\n" + "=" * 100)
    print("未开单信号分析")
    print("=" * 100)

    print(f"\n1. 过滤原因统计:")
    filter_reasons = not_opened['过滤原因'].value_counts()
    print(filter_reasons.to_string())

    print(f"\n2. 未开单信号的张力统计:")
    print(f"   最小值: {not_opened['张力'].min():.4f}")
    print(f"   最大值: {not_opened['张力'].max():.4f}")
    print(f"   平均值: {not_opened['张力'].mean():.4f}")
    print(f"   中位数: {not_opened['张力'].median():.4f}")

    print(f"\n3. 未开单信号的量能比率统计:")
    print(f"   最小值: {not_opened['量能比率'].min():.4f}")
    print(f"   最大值: {not_opened['量能比率'].max():.4f}")
    print(f"   平均值: {not_opened['量能比率'].mean():.4f}")
    print(f"   中位数: {not_opened['量能比率'].median():.4f}")

    # 成功开单的详细列表
    print(f"\n" + "=" * 100)
    print("成功开单信号详细列表")
    print("=" * 100)

    detail_cols = ['时间', '收盘价', '张力', '加速度', '量能比率', 'EMA偏离%',
                   '信号类型', '置信度', '交易方向', '通过V705过滤器']
    opened_detail = opened[detail_cols].copy()
    opened_detail['时间'] = pd.to_datetime(opened_detail['时间']).dt.strftime('%Y-%m-%d %H:%M')
    print(opened_detail.to_string(index=False))

    # 生成总结
    print(f"\n" + "=" * 100)
    print("关键发现总结")
    print("=" * 100)

    if len(opened) > 0:
        print(f"\n1. 开单条件:")
        print(f"   - 张力范围: {opened['张力'].min():.4f} ~ {opened['张力'].max():.4f}")
        print(f"   - 张力主要集中在: {opened['张力'].quantile(0.25):.4f} ~ {opened['张力'].quantile(0.75):.4f} (中间50%)")
        print(f"   - 量能比率范围: {opened['量能比率'].min():.4f} ~ {opened['量能比率'].max():.4f}")
        print(f"   - 量能比率主要集中在: {opened['量能比率'].quantile(0.25):.4f} ~ {opened['量能比率'].quantile(0.75):.4f} (中间50%)")

        print(f"\n2. 信号类型偏好:")
        signal_counts = opened['信号类型'].value_counts()
        for sig, count in signal_counts.items():
            print(f"   - {sig}: {count}个 ({count/len(opened)*100:.1f}%)")

        print(f"\n3. 交易方向偏好:")
        direction_counts = opened['交易方向'].value_counts()
        for dir, count in direction_counts.items():
            print(f"   - {dir}: {count}个 ({count/len(opened)*100:.1f}%)")

        print(f"\n4. 过滤器通过率: {len(opened[opened['通过V705过滤器']=='TRUE'])/len(opened)*100:.1f}%")

print(f"\n" + "=" * 100)
print("分析完成！")
print("=" * 100)

# 保存结果
opened.to_csv('成功开单信号_详细分析.csv', index=False, encoding='utf-8-sig')
print(f"\n成功开单信号已保存到: 成功开单信号_详细分析.csv")
