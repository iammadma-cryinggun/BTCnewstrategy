import pandas as pd
import itertools

# 读取数据
try:
    df = pd.read_csv('最终数据_标注黄金信号_后验最优.csv')
except FileNotFoundError:
    print("❌ 找不到文件")
    exit()

# === 🛠️ 暴力搜索参数池 ===
accel_options = [-0.12, -0.15, -0.20]       # 入场：越来越严
tension_entry_options = [0.5, 0.6, 0.7]     # 入场：越来越严
tension_exit_options = [0.4, 0.2, 0.05]     # 离场：0.4=快跑, 0.05=贪婪(归零才跑)
stop_loss_options = [0.02]                  # 止损：固定 2% (保命底线)

results = []

print(f"=== 💰 利润最大化：全参数暴力回测 ===")
print(f"正在扫描 27 种策略组合... 请稍候...")
print("-" * 60)

# 暴力循环所有组合
for acc in accel_options:
    for t_in in tension_entry_options:
        for t_out in tension_exit_options:
            
            # 初始化回测变量
            balance = 10000.0
            trade_count = 0
            wins = 0
            current_pos = None
            
            # 开始遍历历史数据
            for i in range(len(df) - 1):
                row = df.iloc[i]
                next_row = df.iloc[i+1]
                
                # --- 空仓 ---
                if current_pos is None:
                    # 入场判断
                    if row['加速度'] <= acc and row['张力'] >= t_in:
                        entry_price = row['最高价'] * (1 + 0.0001)
                        stop_price = entry_price * (1 - 0.02) # 固定2%止损
                        
                        # 检查次日成交
                        if next_row['最高价'] > entry_price:
                            # 检查是否秒杀
                            if next_row['最低价'] < stop_price:
                                balance *= 0.98
                                trade_count += 1
                            else:
                                current_pos = {
                                    'entry_price': entry_price, 
                                    'stop_price': stop_price
                                }

                # --- 持仓 ---
                else:
                    # 1. 止损检查
                    if row['最低价'] < current_pos['stop_price']:
                        balance *= 0.98
                        trade_count += 1
                        current_pos = None
                    
                    # 2. 止盈检查 (使用当前的 t_out 参数)
                    elif row['张力'] < t_out:
                        exit_price = row['收盘价']
                        pnl = (exit_price - current_pos['entry_price']) / current_pos['entry_price']
                        balance *= (1 + pnl)
                        trade_count += 1
                        if pnl > 0: wins += 1
                        current_pos = None
            
            # 记录这一组参数的结果
            roi = (balance - 10000) / 10000 * 100
            win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
            
            results.append({
                'Accel': acc,
                'In': t_in,
                'Out': t_out,
                'Trades': trade_count,
                'WinRate': win_rate,
                'ROI': roi
            })

# === 📊 结果分析 ===
# 按收益率(ROI)从高到低排序
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='ROI', ascending=False)

print(f"{'排名':<4} {'Accel':<8} {'In':<6} {'Out':<6} {'交易数':<8} {'胜率':<8} {'总收益(ROI)':<10}")
print("-" * 65)

for i in range(min(10, len(results_df))):
    res = results_df.iloc[i]
    print(f"#{i+1:<3} {res['Accel']:<8} {res['In']:<6} {res['Out']:<6} {int(res['Trades']):<8} {res['WinRate']:<7.1f}% {res['ROI']:>8.2f}%")

print("-" * 65)
print("💡 提示：'Out' 越小，代表拿得越久(越贪婪)。")