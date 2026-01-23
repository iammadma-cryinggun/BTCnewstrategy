# -*- coding: utf-8 -*-
"""
数据收集系统 - 基于验证5逻辑

每4小时自动收集:
1. BTC 4小时K线数据
2. DXY美元指数数据
3. 物理指标 (张力、加速度)
4. 市场状态诊断

保存到: realtime_data_验证5.csv
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import requests
from io import StringIO
from datetime import datetime, timedelta
import schedule
import time
import warnings
warnings.filterwarnings('ignore')


class Verification5Collector:
    """验证5数据收集器"""

    def __init__(self, output_file='realtime_data_验证5.csv'):
        self.output_file = output_file
        self.window_size = 100

        # 验证5参数
        self.TENSION_THRESHOLD = 0.35
        self.ACCEL_THRESHOLD = 0.02

        # 数据缓存
        self.price_history = []

    def fetch_btc_data(self):
        """获取BTC 4小时数据"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {'symbol': 'BTCUSDT', 'interval': '4h', 'limit': 1000}
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            print(f"[ERROR] BTC数据获取失败: {e}")
            return None

    def fetch_dxy_data(self):
        """获取DXY数据"""
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
            response = requests.get(url, timeout=15)

            if response.status_code != 200:
                return None

            dxy_df = pd.read_csv(StringIO(response.text))
            dxy_df['observation_date'] = pd.to_datetime(dxy_df['observation_date'])
            dxy_df.set_index('observation_date', inplace=True)
            dxy_df.rename(columns={'DTWEXBGS': 'Close'}, inplace=True)
            dxy_df = dxy_df.dropna()
            dxy_df['Close'] = pd.to_numeric(dxy_df['Close'], errors='coerce')

            cutoff_date = datetime.now() - timedelta(days=30)
            dxy_df = dxy_df[dxy_df.index >= cutoff_date]

            return dxy_df

        except Exception as e:
            print(f"[WARNING] DXY数据获取失败: {e}")
            return None

    def calculate_tension_acceleration(self, prices):
        """计算张力和加速度（验证5逻辑）"""
        if len(prices) < 3:
            return None, None

        try:
            d_prices = detrend(prices)
            coeffs = fft(d_prices)
            coeffs[8:] = 0
            filtered = ifft(coeffs).real
            analytic = hilbert(filtered)
            tension = np.imag(analytic)

            if len(tension) > 1 and np.std(tension) > 0:
                norm_tension = (tension - np.mean(tension)) / np.std(tension)
            else:
                norm_tension = tension

            if len(norm_tension) >= 3:
                current_tension = norm_tension[-1]
                prev_tension = norm_tension[-2]
                prev2_tension = norm_tension[-3]
                velocity = current_tension - prev_tension
                acceleration = velocity - (prev_tension - prev2_tension)
            else:
                acceleration = 0.0

            return float(norm_tension[-1]), float(acceleration)

        except Exception as e:
            print(f"[ERROR] 物理指标计算失败: {e}")
            return None, None

    def calculate_dxy_fuel(self, dxy_df, current_date):
        """计算DXY燃料"""
        if dxy_df is None or dxy_df.empty:
            return 0.0

        try:
            mask = dxy_df.index <= current_date
            recent = dxy_df[mask].tail(5)

            if len(recent) < 3:
                return 0.0

            closes = recent['Close'].values.astype(float)
            change_1 = (closes[-1] - closes[-2]) / closes[-2]
            change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1
            acceleration = change_1 - change_2
            fuel = -acceleration * 100

            return float(fuel)

        except:
            return 0.0

    def diagnose_regime(self, tension, acceleration, dxy_fuel=0.0):
        """诊断市场状态"""
        if tension > self.TENSION_THRESHOLD and acceleration < -self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.1:
                return "BEARISH_SINGULARITY", "强奇点看空", 0.9
            else:
                return "BEARISH_SINGULARITY", "奇点看空", 0.7

        if tension < -self.TENSION_THRESHOLD and acceleration > self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.2:
                return "BULLISH_SINGULARITY", "超强奇点看涨", 0.95
            elif dxy_fuel > 0:
                return "BULLISH_SINGULARITY", "强奇点看涨", 0.8
            else:
                return "BULLISH_SINGULARITY", "奇点看涨", 0.6

        if abs(tension) < 0.5 and abs(acceleration) < 0.02:
            return "OSCILLATION", "震荡", 0.8

        if tension > 0.3 and abs(acceleration) < 0.01:
            return "HIGH_OSCILLATION", "高位震荡", 0.6

        if tension < -0.3 and abs(acceleration) < 0.01:
            return "LOW_OSCILLATION", "低位震荡", 0.6

        return "TRANSITION", "过渡", 0.3

    def collect_once(self):
        """收集一次数据"""
        print(f"\n{'='*80}")
        print(f"数据收集 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # 获取数据
        btc_df = self.fetch_btc_data()
        if btc_df is None:
            print("[ERROR] 数据收集失败")
            return

        dxy_df = self.fetch_dxy_data()

        # 计算物理指标
        prices = btc_df['close'].tail(100).values
        tension, acceleration = self.calculate_tension_acceleration(prices)

        if tension is None:
            print("[ERROR] 物理指标计算失败")
            return

        # 计算DXY燃料
        current_date = datetime.now()
        dxy_fuel = self.calculate_dxy_fuel(dxy_df, current_date)

        # 诊断市场状态
        signal_type, description, confidence = self.diagnose_regime(tension, acceleration, dxy_fuel)

        # 准备数据
        data = {
            '时间': current_date.strftime('%Y-%m-%d %H:%M:%S'),
            'BTC价格': btc_df['close'].iloc[-1],
            'BTC成交量': btc_df['volume'].iloc[-1],
            '张力': tension,
            '加速度': acceleration,
            'DXY燃料': dxy_fuel,
            '信号类型': signal_type,
            '描述': description,
            '置信度': confidence
        }

        # 保存到CSV
        df_new = pd.DataFrame([data])

        try:
            df_existing = pd.read_csv(self.output_file, encoding='utf-8-sig')
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_combined = df_new

        df_combined.to_csv(self.output_file, index=False, encoding='utf-8-sig')

        print(f"\n[数据收集完成]")
        print(f"  BTC价格: ${data['BTC价格']:,.0f}")
        print(f"  张力: {data['张力']:.4f}")
        print(f"  加速度: {data['加速度']:.6f}")
        print(f"  DXY燃料: {data['DXY燃料']:.2f}")
        print(f"  信号类型: {data['信号类型']}")
        print(f"  描述: {data['描述']}")
        print(f"  置信度: {data['置信度']:.1%}")
        print(f"  已保存到: {self.output_file}")

    def run(self):
        """运行数据收集"""
        print(f"\n{'='*80}")
        print(f"验证5数据收集系统")
        print(f"{'='*80}")
        print(f"\n[配置]")
        print(f"  数据源: Binance (BTC) + FRED (DXY)")
        print(f"  收集频率: 每4小时")
        print(f"  输出文件: {self.output_file}")

        print(f"\n[启动]")
        print(f"  系统将每4小时自动收集数据")
        print(f"  首次收集立即执行\n")

        # 立即执行一次
        self.collect_once()

        # 定时执行
        schedule.every(4).hours.do(self.collect_once)

        print(f"\n[运行中]")
        print(f"  下次收集: {schedule.next_run().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  按 Ctrl+C 停止\n")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print(f"\n\n[停止] 数据收集已停止")
            print(f"[数据文件] {self.output_file}")


if __name__ == "__main__":
    collector = Verification5Collector()
    collector.run()
