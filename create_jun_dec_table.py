import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("创建2025年6-12月完整信号表格（手动标注用）")
print("="*100)

# 读取信号数据
df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 筛选6-12月
df_2025 = df[
    (df['时间'] >= '2025-06-01') &
    (df['时间'] <= '2025-12-31')
].copy().reset_index(drop=True)

print(f"\n总信号数: {len(df_2025)}条")

# 按照原始格式添加列
# 原始列: 时间, 开盘价, 收盘价, 最高价, 最低价, 成交量, 能量比率, EMA偏离%, 张力, 加速度, 信号类型, 置信度, 信号方向, 交易方向, 通过V705过滤器, 拒绝原因, 是否开单

# 添加空列用于手动标注
df_2025['标注'] = ''
df_2025['是好机会'] = ''
df_2025['备注'] = ''

# CSV只有这些列: 时间, 收盘价, 能量比率, EMA偏离%, 张力, 加速度, 信号类型, 置信度, 信号方向, 交易方向
# 直接使用所有列
output_df = df_2025.copy()

# 保存到Excel
output_file = '2025年6-12月_完整信号表_手动标注.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    output_df.to_excel(writer, sheet_name='Sheet1', index=False)

    # 添加第二个sheet：只显示首次信号（张力>0.5, 加速度<0的SHORT）
    first_signals = output_df[
        (output_df.iloc[:, 5] > 0.5) &  # 张力 > 0.5
        (output_df.iloc[:, 6] < 0)      # 加速度 < 0
    ].copy().reset_index(drop=True)

    first_signals.to_excel(writer, sheet_name='首次信号_SHORT', index=False)

    # 添加第三个sheet：只显示LONG首次信号
    long_signals = output_df[
        (output_df.iloc[:, 5] < -0.5) &  # 张力 < -0.5
        (output_df.iloc[:, 6] > 0)       # 加速度 > 0
    ].copy().reset_index(drop=True)

    long_signals.to_excel(writer, sheet_name='首次信号_LONG', index=False)

print(f"\n已保存到: {output_file}")
print(f"  - Sheet1: 所有{len(output_df)}条信号")
print(f"  - 首次信号_SHORT: {len(first_signals)}条 (张力>0.5, 加速度<0)")
print(f"  - 首次信号_LONG: {len(long_signals)}条 (张力<-0.5, 加速度>0)")

# 显示前10条SHORT信号示例（使用列索引避免编码问题）
print("\n" + "="*100)
print("SHORT首次信号示例（前10条）")
print("="*100)

for idx, row in first_signals.head(10).iterrows():
    time_val = row.iloc[0]  # 时间
    price = row.iloc[1]  # 收盘价
    volume = row.iloc[2]  # 成交量
    energy_ratio = row.iloc[3]  # 能量比率
    ema_dev = row.iloc[4]  # EMA偏离%
    tension = row.iloc[5]  # 张力
    accel = row.iloc[6]  # 加速度
    signal_type = row.iloc[7]  # 信号类型
    confidence = row.iloc[8]  # 置信度
    direction = row.iloc[9]  # 信号方向

    print(f"\n{time_val}")
    print(f"  价格: {price:.2f}")
    print(f"  张力: {tension:.4f}, 加速度: {accel:.4f}")
    print(f"  能量比率: {energy_ratio:.2f}, EMA偏离: {ema_dev:.2f}%")
    print(f"  信号类型: {signal_type}, 方向: {direction}")

print("\n" + "="*100)
print("提示：请在Excel中手动标注好机会")
print("="*100)
print("  - '标注'列: 标记这是第几次确认")
print("  - '是好机会'列: 填'是'或'否'")
print("  - '备注'列: 记录张力变化、价格优势等观察")
