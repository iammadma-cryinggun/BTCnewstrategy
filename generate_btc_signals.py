# -*- coding: utf-8 -*-
"""
生成BTC 4小时信号数据表（2025-12-01 至 2026-01-19）
包含：普通信号和开单信号（通过V7.0.5过滤器）
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入V7.0.7系统组件
from v707_trader_main import (
    V707TraderConfig,
    DataFetcher,
    PhysicsSignalCalculator,
    V705EntryFilter
)


def fetch_btc_klines(start_date, end_date, interval='4h'):
    """
    从Binance获取BTC 4小时K线数据

    参数:
        start_date: 开始日期 '2025-12-01'
        end_date: 结束日期 '2026-01-19'
        interval: K线周期 '4h'

    返回:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        # ⭐ 关键修复：从更早的时间开始获取数据，确保有足够的历史数据计算FFT
        # 物理指标需要至少60条数据，所以从start_date之前10天开始
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=10)
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

        start_ts = int(start_datetime.timestamp() * 1000)
        end_ts = int(end_datetime.timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        all_data = []

        current_ts = start_ts
        while current_ts < end_ts:
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'startTime': current_ts,
                'endTime': end_ts,
                'limit': 1000
            }

            logger.info(f"获取数据: {datetime.fromtimestamp(current_ts/1000)}")
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            if not data:
                break

            all_data.extend(data)

            # 更新时间戳
            current_ts = data[-1][6] + 1  # 使用最后一条的close_time + 1

            if len(data) < 1000:
                break

        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 转换数据类型
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        logger.info(f"获取数据完成: {len(df)}条")
        logger.info(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

        return df

    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return None


def generate_signals_table(df):
    """
    生成信号表（普通信号 + 开单信号）

    返回:
        DataFrame: 包含所有信号和开单标记
    """
    config = V707TraderConfig()
    calculator = PhysicsSignalCalculator(config)
    filter = V705EntryFilter(config)

    signals = []

    logger.info("开始计算信号...")
    logger.info("=" * 70)

    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df.iloc[i]['close']

        # 需要至少60条数据才能计算物理指标
        if i < 60:
            continue

        # 获取历史数据
        df_history = df.iloc[:i+1].copy()

        try:
            # 计算物理指标
            df_metrics = calculator.calculate_physics_metrics(df_history)
            if df_metrics is None:
                continue

            latest = df_metrics.iloc[-1]
            tension = latest['tension']
            acceleration = latest['acceleration']

            # 计算量能比率
            avg_volume = df_metrics['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1.0

            # 计算EMA偏离
            prices = df_metrics['close'].values
            ema = filter.calculate_ema(prices, period=20)
            price_vs_ema = (current_price - ema) / ema if ema > 0 else 0

            # 诊断信号
            signal_type, confidence, description = calculator.diagnose_regime(
                tension, acceleration
            )

            if signal_type is None:
                continue

            # 应用V7.0.5过滤器
            should_pass, filter_reason = filter.apply_filter(
                signal_type, acceleration, volume_ratio, price_vs_ema, df_metrics
            )

            # 确定交易方向
            direction_map = {
                'BEARISH_SINGULARITY': 'long',
                'LOW_OSCILLATION': 'long',
                'BULLISH_SINGULARITY': 'short',
                'HIGH_OSCILLATION': 'short',
                'OSCILLATION': 'none'
            }
            direction = direction_map.get(signal_type, 'none')

            # 记录信号
            signal_record = {
                '时间': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                '开盘价': df.iloc[i]['open'],
                '收盘价': current_price,
                '最高价': df.iloc[i]['high'],
                '最低价': df.iloc[i]['low'],
                '交易量': latest['volume'],
                '量能比率': volume_ratio,
                'EMA偏离%': price_vs_ema * 100,
                '张力': tension,
                '加速度': acceleration,
                '信号类型': signal_type,
                '置信度': confidence,
                '信号描述': description,
                '交易方向': direction,
                '通过V705过滤器': should_pass,
                '过滤原因': filter_reason,
                '是否开单': '是' if should_pass else '否'
            }

            signals.append(signal_record)

            # 输出到日志
            if should_pass:
                logger.info(f"[开单信号] {current_time} | {signal_type} | {direction} | ${current_price:.2f} | {filter_reason}")
            else:
                logger.info(f"[普通信号] {current_time} | {signal_type} | ${current_price:.2f} | {filter_reason}")

        except Exception as e:
            logger.error(f"计算信号失败 (时间: {current_time}): {e}")
            continue

    logger.info("=" * 70)
    logger.info(f"信号生成完成: 共{len(signals)}个信号")

    return pd.DataFrame(signals)


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("BTC 4小时信号生成工具（2025-12-01 至 2026-01-19）")
    logger.info("=" * 70)

    # 1. 获取数据
    df = fetch_btc_klines('2025-12-01', '2026-01-20', interval='4h')

    if df is None:
        logger.error("获取数据失败，退出")
        return

    # 2. 生成信号
    df_signals = generate_signals_table(df)

    # 3. 保存到CSV
    output_file = 'btc_4h_signals_20251201_20260119.csv'
    df_signals.to_csv(output_file, index=False, encoding='utf-8-sig')

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"数据已保存到: {output_file}")
    logger.info(f"总信号数: {len(df_signals)}")
    logger.info(f"开单信号: {len(df_signals[df_signals['是否开单'] == '是'])}")
    logger.info(f"过滤信号: {len(df_signals[df_signals['是否开单'] == '否'])}")
    logger.info("=" * 70)

    # 4. 打印统计
    logger.info("\n信号类型统计:")
    signal_stats = df_signals.groupby('信号类型').size()
    for signal_type, count in signal_stats.items():
        logger.info(f"  {signal_type}: {count}个")

    logger.info("\n开单信号统计:")
    trade_signals = df_signals[df_signals['是否开单'] == '是']
    if len(trade_signals) > 0:
        trade_stats = trade_signals.groupby('信号类型').size()
        for signal_type, count in trade_stats.items():
            logger.info(f"  {signal_type}: {count}个")
    else:
        logger.info("  无开单信号")


if __name__ == "__main__":
    main()
