# -*- coding: utf-8 -*-
"""
改进版订单流历史数据收集器
- 分批收集（每天独立）
- 自动重试机制
- SSL错误处理
- 进度保存
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


class RobustOrderFlowCollector:
    """鲁棒的订单流历史数据收集器"""

    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.symbol = "BTCUSDT"
        self.max_retries = 3
        self.retry_delay = 5

    def fetch_historical_trades(self, start_time, end_time, limit=1000):
        """
        获取历史成交数据（带重试机制）

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

        retry_count = 0

        while current_time < end_timestamp:
            params = {
                'symbol': self.symbol,
                'startTime': current_time,
                'endTime': end_timestamp,
                'limit': limit
            }

            try:
                # 使用更长的超时时间
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data:
                    logger.info(f"  无更多数据，停止")
                    break

                all_trades.extend(data)

                # 更新时间戳（最后一笔交易的时间 + 1ms）
                current_time = data[-1]['T'] + 1

                # 心跳日志（包含时间戳）
                current_time_str = datetime.now().strftime('%H:%M:%S')
                logger.info(f"  [{current_time_str}] ♥ 已获取 {len(data)} 笔，总计 {len(all_trades)} 笔")

                # 重置重试计数
                retry_count = 0

                # 避免请求过快
                time.sleep(0.2)

            except requests.exceptions.SSLError as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    logger.warning(f"  SSL错误，第{retry_count}次重试... ({str(e)[:50]})")
                    time.sleep(self.retry_delay * retry_count)
                else:
                    logger.error(f"  SSL错误，重试{self.max_retries}次后失败")
                    # 返回已收集的数据
                    if all_trades:
                        logger.info(f"  返回已收集的 {len(all_trades)} 笔数据")
                    return all_trades if all_trades else None

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    logger.warning(f"  请求错误，第{retry_count}次重试... ({str(e)[:50]})")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"  请求错误，重试失败: {e}")
                    return None

            except Exception as e:
                logger.error(f"  未知错误: {e}")
                return None

        if not all_trades:
            return None

        # 转换为DataFrame
        df = pd.DataFrame(all_trades)
        df['price'] = df['p'].astype(float)
        df['qty'] = df['q'].astype(float)
        df['quote_qty'] = df['price'] * df['qty']
        df['time'] = pd.to_datetime(df['T'], unit='ms')
        df['is_buyer_maker'] = df['m']

        return df[['time', 'price', 'qty', 'quote_qty', 'is_buyer_maker']]

    def calculate_historical_cvd(self, trades_df, window='5min'):
        """从历史成交数据计算CVD"""
        if trades_df is None or trades_df.empty:
            return None

        logger.info(f"  计算CVD（窗口: {window}）...")

        trades_df_copy = trades_df.copy()
        trades_df_copy.set_index('time', inplace=True)

        # 按时间窗口聚合
        buy_volume = trades_df_copy[~trades_df_copy['is_buyer_maker']]['quote_qty'].resample(window).sum()
        sell_volume = trades_df_copy[trades_df_copy['is_buyer_maker']]['quote_qty'].resample(window).sum()

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

        logger.info(f"  计算完成: {len(result)} 个时间点")
        return result

    def detect_whale_trades(self, trades_df, threshold_usd=1000000):
        """检测历史大单交易"""
        if trades_df is None or trades_df.empty:
            return None

        whale_trades = trades_df[trades_df['quote_qty'] >= threshold_usd].copy()
        whale_trades['side'] = whale_trades['is_buyer_maker'].apply(
            lambda x: 'SELL' if x else 'BUY'
        )

        return whale_trades

    def collect_single_day(self, date_str, output_dir='./historical_data'):
        """
        收集单天的数据

        Args:
            date_str: 日期字符串（'2024-01-01'）
            output_dir: 输出目录
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"收集日期: {date_str}")
        logger.info(f"{'='*70}")

        start_time = pd.to_datetime(date_str)
        end_time = start_time + timedelta(days=1)

        # 1. 获取历史成交
        logger.info(f"时间范围: {start_time} 至 {end_time}")
        trades_df = self.fetch_historical_trades(start_time, end_time)

        if trades_df is None or trades_df.empty:
            logger.error(f"  ❌ {date_str} 数据获取失败")
            return None

        logger.info(f"  ✓ 成交数据: {len(trades_df)} 笔")

        # 2. 保存原始成交数据
        os.makedirs(output_dir, exist_ok=True)
        trades_file = f"{output_dir}/trades_{date_str}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"  ✓ 已保存: {trades_file}")

        # 3. 计算CVD
        cvd_df = self.calculate_historical_cvd(trades_df.copy(), window='5min')

        if cvd_df is not None:
            cvd_file = f"{output_dir}/cvd_{date_str}.csv"
            cvd_df.to_csv(cvd_file)
            logger.info(f"  ✓ 已保存: {cvd_file}")

        # 4. 检测大单
        whale_df = self.detect_whale_trades(trades_df, threshold_usd=1000000)

        if whale_df is not None:
            whale_file = f"{output_dir}/whale_trades_{date_str}.csv"
            whale_df.to_csv(whale_file, index=False)
            logger.info(f"  ✓ 已保存: {whale_file}")

        logger.info(f"  ✅ {date_str} 收集完成")

        return {
            'trades': trades_df,
            'cvd': cvd_df,
            'whale': whale_df
        }

    def collect_multiple_days(self, start_date, end_date, output_dir='./historical_data'):
        """
        收集多天数据（分批收集）

        Args:
            start_date: 开始日期（'2024-01-01'）
            end_date: 结束日期（'2024-01-07'）
            output_dir: 输出目录
        """
        logger.info("=" * 70)
        logger.info("开始批量收集历史订单流数据")
        logger.info("=" * 70)
        logger.info(f"日期范围: {start_date} 至 {end_date}")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        logger.info(f"总共 {len(dates)} 天")

        results = []
        failed_dates = []

        for i, date_str in enumerate(dates, 1):
            logger.info(f"\n进度: [{i}/{len(dates)}]")

            result = self.collect_single_day(date_str, output_dir)

            if result is None:
                failed_dates.append(date_str)
            else:
                results.append(result)

        # 汇总
        logger.info("\n" + "=" * 70)
        logger.info("收集完成汇总")
        logger.info("=" * 70)
        logger.info(f"成功: {len(results)}/{len(dates)} 天")
        logger.info(f"失败: {len(failed_dates)} 天")

        if failed_dates:
            logger.warning(f"失败日期: {', '.join(failed_dates)}")

        # 合并所有数据
        if results:
            logger.info("\n合并数据...")

            all_trades = pd.concat([r['trades'] for r in results], ignore_index=True)
            logger.info(f"  合并成交数据: {len(all_trades)} 笔")

            # 保存合并文件
            merged_trades_file = f"{output_dir}/trades_{start_date}_{end_date}.csv"
            all_trades.to_csv(merged_trades_file, index=False)
            logger.info(f"  ✓ 已保存: {merged_trades_file}")

            if results[0]['cvd'] is not None:
                all_cvd = pd.concat([r['cvd'] for r in results if r['cvd'] is not None])
                logger.info(f"  合并CVD数据: {len(all_cvd)} 个时间点")

                merged_cvd_file = f"{output_dir}/cvd_{start_date}_{end_date}.csv"
                all_cvd.to_csv(merged_cvd_file)
                logger.info(f"  ✓ 已保存: {merged_cvd_file}")

            if results[0]['whale'] is not None:
                all_whale = pd.concat([r['whale'] for r in results if r['whale'] is not None], ignore_index=True)
                logger.info(f"  合并鲸鱼交易: {len(all_whale)} 笔")

                merged_whale_file = f"{output_dir}/whale_trades_{start_date}_{end_date}.csv"
                all_whale.to_csv(merged_whale_file, index=False)
                logger.info(f"  ✓ 已保存: {merged_whale_file}")

            logger.info("\n✅ 所有数据收集完成！")

        return results


# 使用示例
if __name__ == "__main__":
    collector = RobustOrderFlowCollector()

    # 收集最近2天的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)

    print(f"\n收集时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"预计: 2天 = 48小时 = 约576个5分钟CVD数据点")
    print("开始收集...\n")

    data = collector.collect_multiple_days(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    if data:
        print("\n" + "=" * 70)
        print("✅ 数据收集成功！现在可以运行回测:")
        print("   python backtest_v81_orderflow.py")
        print("=" * 70)
