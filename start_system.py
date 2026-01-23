# -*- coding: utf-8 -*-
"""
一键启动脚本 - 同时启动数据收集和实盘交易
"""

import subprocess
import sys
from pathlib import Path

print("="*100)
print("V8.0 + 简化微观结构交易系统 - 一键启动")
print("="*100)

print("\n[启动中...]")
print("  1. 数据收集系统 (后台)")
print("  2. 实盘交易引擎 (前台)")

# 启动数据收集 (非交互模式)
print("\n[1/2] 启动数据收集...")

collector_script = Path(__file__).parent / "auto_collector.py"

# 创建自动收集脚本
collector_code = '''# -*- coding: utf-8 -*-
import time
import schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

class SimplifiedMicrostructureMonitor:
    def __init__(self):
        self.price_history = []
        self.volume_history = []

    def analyze(self, price, volume):
        self.price_history.append(price)
        self.volume_history.append(volume)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
            self.volume_history.pop(0)

        if len(self.price_history) < 3:
            return 0, 0.5, False, False

        # 波动率
        returns = pd.Series(self.price_history).pct_change().dropna()
        vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # 流动性
        liquidity = 0.5

        # 风险检测
        crash_risk = vol > 0.8
        squeeze = vol < 0.3

        return vol, liquidity, crash_risk, squeeze

monitor = SimplifiedMicrostructureMonitor()
output_file = "realtime_microstructure_data.csv"

def collect_data():
    try:
        # 获取数据
        response = requests.get('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT', timeout=10)
        data = response.json()

        price = float(data['lastPrice'])
        volume = float(data['volume'])

        # 分析
        vol, liquidity, crash, squeeze = monitor.analyze(price, volume)

        # 保存
        row = {
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '价格': price,
            '成交量': volume,
            '波动率': f"{vol:.4f}",
            '流动性': f"{liquidity:.2f}",
            '闪崩风险': 'YES' if crash else 'No',
            '暴涨设置': 'YES' if squeeze else 'No'
        }

        df = pd.DataFrame([row])
        df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False, encoding='utf-8-sig')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collected: ${price:,.0f} | Vol: {vol:.2%} | Crash: {crash} | Squeeze: {squeeze}")

    except Exception as e:
        print(f"[ERROR] {e}")

# 立即执行一次
collect_data()

# 每4小时执行
schedule.every(4).hours.do(collect_data)

print("[数据收集系统已启动] 每4小时自动收集")
print("按 Ctrl+C 停止\\n")

try:
    while True:
        schedule.run_pending()
        time.sleep(60)
except KeyboardInterrupt:
    print("\\n[数据收集已停止]")
'''

with open(collector_script, 'w', encoding='utf-8') as f:
    f.write(collector_code)

# 启动数据收集 (后台)
collector_process = subprocess.Popen(
    [sys.executable, str(collector_script)],
    cwd=Path(__file__).parent,
    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
)

print(f"  → PID: {collector_process.pid}")

# 等待2秒确保数据收集启动
time.sleep(2)

# 启动实盘交易 (前台)
print("\n[2/2] 启动实盘交易引擎...")
print("  → 交易日志将保存到: live_trading_log.csv")
print("  → 按 Ctrl+C 停止交易\n")

try:
    # 导入并运行实盘引擎
    from v80_microstructure_live import LiveTradingEngine

    engine = LiveTradingEngine(
        account_balance=10000,
        risk_per_trade=0.02
    )

    engine.run(interval_seconds=300)  # 每5分钟检查

except KeyboardInterrupt:
    print("\n\n[停止] 系统已停止")

    # 停止数据收集
    if collector_process.poll() is None:
        collector_process.terminate()
        print("[数据收集已停止]")

    print("\n[数据文件]")
    print("  - realtime_microstructure_data.csv (市场数据)")
    print("  - live_trading_log.csv (交易日志)")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\n[数据收集仍在后台运行]")
    print(f"  PID: {collector_process.pid}")
    print("  如需停止，请手动关闭该进程")
