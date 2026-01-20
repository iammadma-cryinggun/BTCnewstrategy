import pandas as pd
df = pd.read_csv('信号数据_2025_6-12月_正确版.csv', encoding='utf-8-sig', nrows=3)
print('Column count:', len(df.columns))
print('\nFirst row values:')
for i, val in enumerate(df.iloc[0]):
    print(f'  {i}: {val}')
