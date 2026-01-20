# -*- coding: utf-8 -*-
"""
生成与实际运行系统完全一致的信号数据
使用滚动300条数据（与实际系统相同）
"""

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


def fetch_historical_4h_data(target_date):
    """
    获取指定日期的历史4H数据（模拟当时可获取的数据）

    参数:
        target_date: 目标日期（datetime对象），返回该日期收盘时可获取的数据
    """
    try:
        # 获取比目标日期多300条的数据，确保能模拟当时的滚动窗口
        # 4H周期，300条 = 1200小时 = 50天
        start_datetime = target_date - timedelta(days=55)
        end_datetime = target_date + timedelta(hours=4)

        start_ts = int(start_datetime.timestamp() * 1000)
        end_ts = int(end_datetime.timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '4h',
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if not data:
            return None

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 转换为北京时间（UTC+8）- 与实际系统一致
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=8)
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # 只返回目标日期之前的数据（不包括目标日期之后的）
        df = df[df.index <= target_date]

        # 确保返回最近300条（如果有的话）
        if len(df) > 300:
            df = df.iloc[-300:]

        return df

    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return None


def generate_signals_for_period(start_date, end_date):
    """
    为指定时间段生成信号（每个4H周期独立计算）

    使用与实际系统完全相同的逻辑：
    1. 对每个时间点，获取当时的最近300条数据
    2. 使用标准的diagnose_regime函数
    3. 应用V7.0.5过滤器
    """
    config = V707TraderConfig()
    calculator = PhysicsSignalCalculator(config)
    filter = V705EntryFilter(config)

    signals = []

    logger.info("开始计算信号...")
    logger.info("=" * 70)

    # 生成所有4H时间点（从start_date 00:00 到 end_date 20:00）
    current_time = datetime.strptime(start_date, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
    end_time = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=20, minute=0, second=0)

    while current_time <= end_time:
        # 跳过非4H整点时间
        if current_time.hour % 4 != 0:
            current_time += timedelta(hours=1)
            continue

        logger.info(f"处理时间: {current_time}")

        # 获取当时的300条数据
        df_300 = fetch_historical_4h_data(current_time)

        if df_300 is None or len(df_300) < 60:
            logger.warning(f"数据不足，跳过: {current_time}")
            current_time += timedelta(hours=4)
            continue

        try:
            # 计算物理指标（与实际系统相同）
            df_metrics = calculator.calculate_physics_metrics(df_300)
            if df_metrics is None:
                logger.warning(f"物理指标计算失败，跳过: {current_time}")
                current_time += timedelta(hours=4)
                continue

            # 获取当前K线的指标
            if current_time not in df_metrics.index:
                # 找最接近的时间
                closest_idx = df_metrics.index.get_indexer([current_time], method='nearest')[0]
                if closest_idx == -1:
                    logger.warning(f"找不到对应数据，跳过: {current_time}")
                    current_time += timedelta(hours=4)
                    continue
                latest = df_metrics.iloc[closest_idx]
            else:
                latest = df_metrics.loc[current_time]

            tension = latest['tension']
            acceleration = latest['acceleration']
            current_price = latest['close']

            # 计算量能比率
            avg_volume = df_metrics['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1.0

            # 计算EMA偏离
            prices = df_metrics['close'].values
            ema = filter.calculate_ema(prices, period=20)
            price_vs_ema = (current_price - ema) / ema if ema > 0 else 0

            # ⭐ 使用与实际系统完全相同的诊断函数
            signal_type, confidence, description = calculator.diagnose_regime(
                tension, acceleration
            )

            # ⭐ 只记录有效信号（置信度>=0.6）
            if signal_type is None:
                logger.info(f"  无有效信号（置信度不足）| T={tension:.3f} | A={acceleration:.3f}")
                current_time += timedelta(hours=4)
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
                '收盘价': current_price,
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

            status = "开单" if should_pass else "过滤"
            logger.info(f"  [{status}] {signal_type} | T={tension:.3f} | A={acceleration:.3f} | {description}")

        except Exception as e:
            logger.error(f"计算信号失败 (时间: {current_time}): {e}")
            import traceback
            traceback.print_exc()

        current_time += timedelta(hours=4)

    logger.info("=" * 70)
    logger.info(f"信号生成完成: 共{len(signals)}个信号")

    return pd.DataFrame(signals)


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("BTC 4小时信号生成工具（与实际系统一致）")
    logger.info("=" * 70)

    # 1. 生成信号
    df_signals = generate_signals_for_period('2025-12-01', '2026-01-19')

    # 2. 保存到CSV
    output_file = 'btc_4h_signals_matching_log.csv'
    df_signals.to_csv(output_file, index=False, encoding='utf-8-sig')

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"数据已保存到: {output_file}")
    logger.info(f"总信号数: {len(df_signals)}")
    logger.info(f"开单信号: {len(df_signals[df_signals['是否开单'] == '是'])}")
    logger.info(f"过滤信号: {len(df_signals[df_signals['是否开单'] == '否'])}")
    logger.info("=" * 70)

    # 3. 打印统计
    logger.info("\n信号类型统计:")
    signal_stats = df_signals['信号类型'].value_counts()
    for signal_type, count in signal_stats.items():
        logger.info(f"  {signal_type}: {count}个")

    logger.info("\n开单信号统计:")
    trade_signals = df_signals[df_signals['是否开单'] == '是']
    if len(trade_signals) > 0:
        trade_stats = trade_signals['信号类型'].value_counts()
        for signal_type, count in trade_stats.items():
            logger.info(f"  {signal_type}: {count}个")
    else:
        logger.info("  无开单信号")


if __name__ == "__main__":
    main()
