import pandas as pd
df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig')
df['时间'] = pd.to_datetime(df['时间'])

# 筛选6-12月
df_2025 = df[
    (df['时间'] >= '2025-06-01') &
    (df['时间'] <= '2025-12-31')
].copy().reset_index(drop=True)

print(f"Total rows: {len(df_2025)}")
print(f"\nColumn indices:")
for i, col in enumerate(df_2025.columns):
    print(f"  {i}: {col}")

print(f"\nFirst row:")
row = df_2025.iloc[0]
for i, col in enumerate(df_2025.columns):
    print(f"  {i} ({col}): {row.iloc[i]}")

# 筛选SHORT信号（张力>0.5, 加速度<0）
print(f"\n\nFiltering: tension > 0.5 and accel < 0")
tension_col = df_2025.iloc[:, 4]  # 张力
accel_col = df_2025.iloc[:, 5]    # 加速度

short_mask = (tension_col > 0.5) & (accel_col < 0)
print(f"  SHORT signals: {short_mask.sum()}")

print(f"\nFirst 3 SHORT signals:")
short_df = df_2025[short_mask].head(3)
for idx, row in short_df.iterrows():
    print(f"\n  Row {idx}:")
    print(f"    时间: {row.iloc[0]}")
    print(f"    收盘价: {row.iloc[1]:.2f}")
    print(f"    能量比率: {row.iloc[2]:.2f}")
    print(f"    EMA偏离%: {row.iloc[3]:.2f}")
    print(f"    张力: {row.iloc[4]:.4f}")
    print(f"    加速度: {row.iloc[5]:.4f}")
    print(f"    置信度: {row.iloc[6]:.4f}")
    print(f"    信号类型: {row.iloc[7]}")
    print(f"    方向: {row.iloc[9]}")
