# generate_signals_price_action.py
import pandas as pd
import numpy as np

print("="*60)
print("基于价格走势生成黄金信号")
print("="*60)

# 读取数据
df = pd.read_csv('最终数据_普通信号_完整含DXY_OHLC.csv', encoding='utf-8')
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)

print(f"数据行数: {len(df)}")
print(f"时间范围: {df['时间'].iloc[0]} 到 {df['时间'].iloc[-1]}")

# 分析价格走势
def analyze_price_trend(df, lookback=3):
    """分析价格走势趋势"""
    signals = []
    current_position = None
    entry_price = None
    entry_time = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['收盘价']
        
        # 如果是开始，没有足够的历史数据
        if i < lookback:
            signals.append('')
            continue
        
        # 分析最近的价格走势
        recent_prices = df['收盘价'].iloc[max(0, i-lookback):i+1].values
        recent_high = np.max(recent_prices[:-1])  # 不包括当前价格
        recent_low = np.min(recent_prices[:-1])
        current_price = recent_prices[-1]
        
        # 获取信号信息
        signal_type = row['信号类型']
        v705_pass = row['V7.0.5通过']
        
        # 确定新信号
        new_signal = ''
        
        # 如果有特殊黄金信号，优先使用
        if pd.notna(row['黄金信号']) and '黄金' in str(row['黄金信号']):
            new_signal = row['黄金信号']
        # 如果过滤未通过，观望
        elif v705_pass == False:
            new_signal = '观望(震荡不开仓)'
        else:
            # 根据当前仓位决定
            if current_position is None:
                # 无仓位，判断是否开仓
                if signal_type == 'BEARISH_SINGULARITY':
                    # 看空奇点，做多
                    new_signal = '开多'
                    current_position = '多'
                    entry_price = price
                    entry_time = row['时间']
                elif signal_type == 'BULLISH_SINGULARITY':
                    # 看涨奇点，做空
                    new_signal = '开空'
                    current_position = '空'
                    entry_price = price
                    entry_time = row['时间']
            
            elif current_position == '多':
                # 持有多头
                # 检查是否应该平仓
                should_close = False
                
                # 规则1: 价格从开仓位上涨一定幅度后开始回落
                if entry_price is not None:
                    profit_pct = (price - entry_price) / entry_price * 100
                    
                    # 如果已经盈利，且价格开始从近期高点回落
                    if profit_pct > 0.5:  # 盈利超过0.5%
                        # 当前价格低于近期高点一定比例
                        price_from_high = (recent_high - price) / recent_high * 100
                        if price_from_high > 0.3:  # 从高点回落0.3%以上
                            should_close = True
                
                # 规则2: 出现明确的反向信号
                if signal_type == 'BULLISH_SINGULARITY':
                    should_close = True
                
                if should_close:
                    new_signal = '平多'
                    current_position = None
                    entry_price = None
                else:
                    new_signal = '继续持多'
            
            elif current_position == '空':
                # 持有空头
                # 检查是否应该平仓
                should_close = False
                
                # 规则1: 价格从开仓位下跌一定幅度后开始反弹
                if entry_price is not None:
                    profit_pct = (entry_price - price) / entry_price * 100
                    
                    # 如果已经盈利，且价格开始从近期低点反弹
                    if profit_pct > 0.5:  # 盈利超过0.5%
                        # 当前价格高于近期低点一定比例
                        price_from_low = (price - recent_low) / recent_low * 100
                        if price_from_low > 0.3:  # 从低点反弹0.3%以上
                            should_close = True
                
                # 规则2: 出现明确的反向信号
                if signal_type == 'BEARISH_SINGULARITY':
                    should_close = True
                
                if should_close:
                    new_signal = '平空'
                    current_position = None
                    entry_price = None
                else:
                    new_signal = '继续持空'
        
        signals.append(new_signal)
    
    return signals

# 生成新信号
print("基于价格走势生成信号...")
new_signals = analyze_price_trend(df, lookback=4)
df['黄金信号_价格走势'] = new_signals

# 专门分析8月19-22日
print("\n" + "="*60)
print("8月19-22日详细分析")
print("="*60)

aug_data = df[(df['时间'] >= '2025-08-19 16:00') & (df['时间'] <= '2025-08-22 12:00')].copy()

print("时间线分析:")
print("时间               收盘价      信号类型           原信号             新信号")
print("-" * 90)

for idx, row in aug_data.iterrows():
    time_str = row['时间'].strftime('%m-%d %H:%M')
    price = f"{row['收盘价']:,.0f}"
    signal_type = row['信号类型'][:20] if pd.notna(row['信号类型']) else ''
    old_signal = row['黄金信号'] if pd.notna(row['黄金信号']) else ''
    new_signal = row['黄金信号_价格走势'] if pd.notna(row['黄金信号_价格走势']) else ''
    
    print(f"{time_str:12} {price:>10} {signal_type:20} {old_signal:15} {new_signal:15}")

# 分析具体的交易决策
print("\n" + "="*60)
print("交易决策分析")
print("="*60)

print("""
关键点分析:

1. 8月19日 20:00:
   - 价格: 112,873
   - 信号: BEARISH_SINGULARITY (看空奇点)
   - 决策: 做多 ✓
   - 理由: 反向策略，价格可能反弹

2. 多头持仓分析:
   - 8月20日 0:00: 价格113,526 (+653点) → 继续持多
   - 8月20日 4:00: 价格113,490 (-36点) → 继续持多  
   - 8月20日 8:00: 价格113,667 (+177点) → 继续持多
   - 8月20日 12:00: 价格113,354 (-313点) → 继续持多
   - 8月20日 16:00: 价格114,277 (+923点) → 继续持多
   
   累计盈利: 114,277 - 112,873 = 1,404点

3. 平多决策点:
   - 8月20日 20:00: 价格114,271 (-6点)
      * 从高点114,277轻微回落
      * 信号: BEARISH_SINGULARITY (应继续持多)
      * 但价格开始下跌趋势
   
   - 8月21日 0:00: 价格113,955 (-316点)
      * 明显下跌
      * 应该考虑平多

4. 做空机会:
   - 从8月20日20:00开始明显下跌趋势
   - 8月21日20:00: 价格112,500
   - 8月22日8:00: 价格112,320 (低点)
   - 然后开始反弹
""")

# 统计信号分布
print("\n" + "="*60)
print("信号分布统计")
print("="*60)

print("原黄金信号分布:")
print(df['黄金信号'].value_counts())

print("\n新黄金信号分布:")
print(df['黄金信号_价格走势'].value_counts())

# 计算交易次数
def count_trades(signals):
    """统计交易次数"""
    trades = 0
    for s in signals:
        if s in ['开多', '开空', '多平/反空', '空平/反多']:
            trades += 1
    return trades

old_trades = count_trades(df['黄金信号'].fillna('').tolist())
new_trades = count_trades(df['黄金信号_价格走势'].fillna('').tolist())

print(f"\n交易次数对比:")
print(f"原信号: {old_trades} 次")
print(f"新信号: {new_trades} 次")

# 保存结果
output_file = '基于价格走势的黄金信号.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: {output_file}")

# 生成交易规则总结
print("\n" + "="*60)
print("总结：基于价格走势的交易规则")
print("="*60)

print("""
1. 开仓规则：
   - BEARISH_SINGULARITY → 做多（反向）
   - BULLISH_SINGULARITY → 做空（反向）
   - 只在V7.0.5通过时执行

2. 平仓规则：
   A) 价格从极端位置回调时：
      - 做多：价格上涨0.5%+后，从高点回落0.3%+
      - 做空：价格下跌0.5%+后，从低点反弹0.3%+
   
   B) 出现反向信号时：
      - 持多时出现BULLISH_SINGULARITY → 平多
      - 持空时出现BEARISH_SINGULARITY → 平空

3. 反手规则（可选）：
   - 平多后如果看跌，可反手做空
   - 平空后如果看涨，可反手做多

4. 实际应用示例（8月19-22日）：
   - 8/19 20:00：BEARISH_SINGULARITY → 开多
   - 持多到8/20 20:00或8/21 0:00 → 平多
   - 观察到下跌趋势 → 可考虑做空
   - 8/22 8:00到达低点112,320 → 平空或观望
""")

# 提取示例供参考
print("\n" + "="*60)
print("示例信号序列")
print("="*60)

example_seq = df[(df['时间'] >= '2025-08-19 16:00') & 
                 (df['时间'] <= '2025-08-22 12:00')]

print("建议的信号序列:")
suggested_signals = [
    ('2025-08-19 20:00', '开多', 'BEARISH_SINGULARITY，反向做多'),
    ('2025-08-20 00:00', '继续持多', '价格上涨'),
    ('2025-08-20 04:00', '继续持多', '价格整理'),
    ('2025-08-20 08:00', '继续持多', '价格上涨'),
    ('2025-08-20 12:00', '继续持多', '价格回调但趋势未变'),
    ('2025-08-20 16:00', '继续持多', '创新高'),
    ('2025-08-20 20:00', '平多', '从高点回落，开始下跌趋势'),
    ('2025-08-21 00:00', '开空', '确认下跌趋势'),
    ('2025-08-21 04:00', '继续持空', '继续下跌'),
    ('2025-08-21 08:00', '继续持空', '继续下跌'),
    ('2025-08-21 12:00', '继续持空', '继续下跌'),
    ('2025-08-21 16:00', '继续持空', '继续下跌'),
    ('2025-08-21 20:00', '继续持空', '到达低点区域'),
    ('2025-08-22 00:00', '平空', 'BEARISH_SINGULARITY信号，准备做多'),
    ('2025-08-22 04:00', '开多', '价格开始反弹'),
    ('2025-08-22 08:00', '继续持多', '确认反弹'),
]

for time_str, signal, reason in suggested_signals:
    print(f"{time_str}: {signal:10} - {reason}")

print("\n" + "="*60)
print("完成！")
print("="*60)