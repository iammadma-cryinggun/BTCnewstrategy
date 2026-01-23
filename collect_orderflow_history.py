# -*- coding: utf-8 -*-
"""
收集历史订单流数据用于回测
支持Binance Futures历史数据下载
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderFlowHistoryCollector:
    """订单流历史数据收集器"""

    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.symbol = "BTCUSDT"

    def fetch_historical_trades(self, start_time, end_time, limit=1000):
        """
        获取历史成交数据

        Args:
            start_time: 开始时间（datetime）
            end_time: 结束时间（datetime）
            limit: 每次请求数量（最大1000）

        Returns:
            DataFrame: 历史成交数据
        """
        url = f"{self.base_url}/fapi/v1/aggTrades"

        all_trades = []
        current_time = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)

        while current_time < end_timestamp:
            params = {
                'symbol': self.symbol,
                'startTime': current_time,
                'endTime': end_timestamp,
                'limit': limit
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_trades.extend(data)

                # 更新时间戳（最后一笔交易的时间 + 1ms）
                current_time = data[-1]['T'] + 1

                logger.info(f"已获取 {len(data)} 笔成交，总计 {len(all_trades)} 笔")

                # 避免请求过快
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"获取数据失败: {e}")
                break

        if not all_trades:
            return None

        # 转换为DataFrame
        df = pd.DataFrame(all_trades)
        df['price'] = df['p'].astype(float)
        df['qty'] = df['q'].astype(float)
        df['quote_qty'] = df['price'] * df['qty']
        df['time'] = pd.to_datetime(df['T'], unit='ms')  # 转换为datetime
        df['is_buyer_maker'] = df['m']  # True=主动卖出, False=主动买入

        logger.info(f"总共获取 {len(df)} 笔历史成交数据")
        return df[['time', 'price', 'qty', 'quote_qty', 'is_buyer_maker']]  # 确保time列在前面

    def calculate_historical_cvd(self, trades_df, window='5min'):
        """
        从历史成交数据计算CVD

        Args:
            trades_df: 历史成交数据
            window: 时间窗口

        Returns:
            DataFrame: CVD数据
        """
        if trades_df is None or trades_df.empty:
            return None

        logger.info(f"计算历史CVD（窗口: {window}）...")

        trades_df.set_index('time', inplace=True)

        # 按时间窗口聚合
        buy_volume = trades_df[~trades_df['is_buyer_maker']]['quote_qty'].resample(window).sum()
        sell_volume = trades_df[trades_df['is_buyer_maker']]['quote_qty'].resample(window).sum()

        delta = buy_volume - sell_volume
        cvd = delta.cumsum()

        total_volume = buy_volume + sell_volume
        buy_ratio = buy_volume / total_volume

        result = pd.DataFrame({
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'cvd': cvd,
            'cvd_change': delta,
            'buy_ratio': buy_ratio
        }).dropna()

        # 判断趋势
        result['trend'] = result.apply(
            lambda x: 'bullish' if x['cvd_change'] > 0 and x['buy_ratio'] > 0.6 else
                      ('bearish' if x['cvd_change'] < 0 and x['buy_ratio'] < 0.4 else 'neutral'),
            axis=1
        )

        logger.info(f"计算完成: {len(result)} 个时间点")
        return result

    def detect_whale_trades(self, trades_df, threshold_usd=1000000):
        """
        检测历史大单交易

        Args:
            trades_df: 历史成交数据
            threshold_usd: 门槛金额（USD）

        Returns:
            DataFrame: 大单交易
        """
        if trades_df is None or trades_df.empty:
            return None

        logger.info(f"检测鲸鱼交易（门槛: ${threshold_usd:,.0f}）...")

        whale_trades = trades_df[trades_df['quote_qty'] >= threshold_usd].copy()
        whale_trades['side'] = whale_trades['is_buyer_maker'].apply(
            lambda x: 'SELL' if x else 'BUY'
        )

        logger.info(f"检测到 {len(whale_trades)} 笔鲸鱼交易")
        return whale_trades

    def collect_period(self, start_date, end_date, output_dir='./historical_data'):
        """
        收集指定时间段的数据

        Args:
            start_date: 开始日期（字符串: '2024-01-01'）
            end_date: 结束日期（字符串: '2024-12-31'）
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        start_time = pd.to_datetime(start_date)
        end_time = pd.to_datetime(end_date)

        logger.info("=" * 70)
        logger.info("开始收集历史订单流数据")
        logger.info("=" * 70)
        logger.info(f"时间范围: {start_date} 至 {end_date}")
        logger.info(f"交易对: {self.symbol}")

        # 1. 获取历史成交
        trades_df = self.fetch_historical_trades(start_time, end_time)

        if trades_df is None:
            logger.error("获取历史数据失败")
            return

        # 2. 先保存原始成交数据（在CVD计算之前，避免time列丢失）
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_file = f"{output_dir}/trades_{start_date}_{end_date}.csv"

        trades_df.to_csv(trades_file, index=False)
        logger.info(f"✅ 原始成交数据已保存: {trades_file}")
        logger.info(f"   文件大小: {os.path.getsize(trades_file) / 1024 / 1024:.2f} MB")
        logger.info(f"   时间范围: {trades_df['time'].min()} 至 {trades_df['time'].max()}")
        logger.info(f"   数据行数: {len(trades_df):,} 笔")

        # 3. 计算CVD（使用副本，避免修改原始数据）
        cvd_df = self.calculate_historical_cvd(trades_df.copy(), window='5min')

        # 4. 检测大单
        whale_df = self.detect_whale_trades(trades_df, threshold_usd=1000000)

        # 5. 保存CVD和鲸鱼数据
        # 保存CVD数据
        if cvd_df is not None:
            cvd_file = f"{output_dir}/cvd_{start_date}_{end_date}.csv"
            cvd_df.to_csv(cvd_file)  # 保存索引（时间）
            logger.info(f"✅ CVD数据已保存: {cvd_file}")
            logger.info(f"   时间点数量: {len(cvd_df)}")

        # 保存大单数据
        if whale_df is not None:
            whale_file = f"{output_dir}/whale_trades_{start_date}_{end_date}.csv"
            whale_df.to_csv(whale_file, index=False)
            logger.info(f"✅ 鲸鱼交易数据已保存: {whale_file}")
            logger.info(f"   鲸鱼交易数量: {len(whale_df)}")

        logger.info("=" * 70)
        logger.info("数据收集完成！")
        logger.info("=" * 70)

        return {
            'trades': trades_df,
            'cvd': cvd_df,
            'whale': whale_df
        }


# 使用示例
if __name__ == "__main__":
    collector = OrderFlowHistoryCollector()

    # 收集最近7天的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"\n收集时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"预计数据量: 约 {7 * 24} 小时 = 168个5分钟CVD数据点")
    print(f"预计成交数据: 约 {7 * 24 * 60 * 60} 笔（估算）")
    print("开始收集...\n")

    data = collector.collect_period(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    if data and data['cvd'] is not None:
        print("\n" + "=" * 70)
        print("CVD数据预览:")
        print("=" * 70)
        print(data['cvd'].tail(10))
        print("\n✅ 数据收集完成！现在可以运行回测:")
        print("   python backtest_v81_orderflow.py")
