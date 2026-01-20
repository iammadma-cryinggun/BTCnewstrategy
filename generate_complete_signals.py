# -*- coding: utf-8 -*-
"""
生成BTC 4小时信号数据表（完整版）
包含所有信号，不管是否通过过滤器或置信度高低
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


def fetch_btc_klines(start_date, end_date, interval='4h'):
    """从Binance获取BTC 4小时K线数据"""
    try:
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

            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            current_ts = data[-1][6] + 1

            if len(data) < 1000:
                break

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

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


def diagnose_regime_all(tension, acceleration):
    """
    诊断市场状态（确保总是返回有效信号类型）
    返回信号类型，不管置信度如何
    """
    signal_type = None
    confidence = 0.0
    description = "无信号"

    if tension > 0.35 and acceleration < -0.02:
        confidence = 0.7
        description = f"奇点看空(T={tension:.2f}>=0.35)"
        signal_type = 'BEARISH_SINGULARITY'

    elif tension < -0.35 and acceleration > 0.02:
        confidence = 0.6
        description = f"奇点看涨(T={tension:.2f}<=-0.35)"
        signal_type = 'BULLISH_SINGULARITY'

    elif abs(tension) < 0.5 and abs(acceleration) < 0.02:
        confidence = 0.8
        signal_type = 'OSCILLATION'
        description = f"系统平衡震荡"

    elif tension > 0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        signal_type = 'HIGH_OSCILLATION'
        description = f"高位震荡(T={tension:.2f}>0.3)"

    elif tension < -0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        signal_type = 'LOW_OSCILLATION'
        description = f"低位震荡(T={tension:.2f}<-0.3)"

    # ⭐ 修正：如果不符合任何正常条件，根据张力值强制归类
    if signal_type is None:
        if tension > 0:
            # 多头市场
            if tension > 0.5:
                signal_type = 'BEARISH_SINGULARITY'
                confidence = 0.5
                description = f"奇点看空(低置信度T={tension:.2f})"
            else:
                signal_type = 'HIGH_OSCILLATION'
                confidence = 0.5
                description = f"高位震荡(快速变化T={tension:.2f})"
        else:
            # 空头市场
            if tension < -0.5:
                signal_type = 'BULLISH_SINGULARITY'
                confidence = 0.5
                description = f"奇点看涨(低置信度T={tension:.2f})"
            else:
                signal_type = 'OSCILLATION'
                confidence = 0.4
                description = f"系统平衡震荡(T={tension:.2f})"

    return signal_type, confidence, description


def generate_signals_table(df):
    """生成完整信号表（包含所有信号）"""
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

        # 只关注12月1日之后的信号
        if current_time < pd.Timestamp('2025-12-01'):
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

            # ⭐ 使用新的诊断函数，包含所有信号
            signal_type, confidence, description = diagnose_regime_all(
                tension, acceleration
            )

            # ⭐ 记录所有信号，包括低置信度的
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
                '信号类型': signal_type if signal_type else '无信号',
                '置信度': confidence,
                '信号描述': description,
            }

            # 确定交易方向
            direction_map = {
                'BEARISH_SINGULARITY': 'long',
                'LOW_OSCILLATION': 'long',
                'BULLISH_SINGULARITY': 'short',
                'HIGH_OSCILLATION': 'short',
                'OSCILLATION': 'none',
                '无信号': 'none'
            }
            direction = direction_map.get(signal_type, 'none')
            signal_record['交易方向'] = direction

            # ⭐ 对所有信号应用V7.0.5过滤器（获取具体过滤原因）
            should_pass, filter_reason = filter.apply_filter(
                signal_type, acceleration, volume_ratio, price_vs_ema, df_metrics
            )
            signal_record['通过V705过滤器'] = should_pass
            signal_record['过滤原因'] = filter_reason

            # ⭐ 只有置信度>=0.6且通过过滤器的信号才开单
            if confidence >= 0.6 and should_pass:
                signal_record['是否开单'] = '是'
            else:
                signal_record['是否开单'] = '否'
                # 更新过滤原因，如果是低置信度则标注
                if confidence < 0.6:
                    signal_record['过滤原因'] = f"置信度不足({confidence:.1f}) - {filter_reason}"

            signals.append(signal_record)

        except Exception as e:
            logger.error(f"计算信号失败 (时间: {current_time}): {e}")
            continue

    logger.info("=" * 70)
    logger.info(f"信号生成完成: 共{len(signals)}个信号")

    return pd.DataFrame(signals)


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("BTC 4小时信号生成工具（完整版 - 包含所有信号）")
    logger.info("=" * 70)

    # 1. 获取数据
    df = fetch_btc_klines('2025-12-01', '2026-01-19', '4h')

    if df is None:
        logger.error("获取数据失败，退出")
        return

    # 2. 生成信号
    df_signals = generate_signals_table(df)

    # 3. 保存到CSV
    output_file = 'btc_4h_signals_complete_20251201_20260119.csv'
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

    logger.info(f"\n时间范围: {df_signals['时间'].min()} 至 {df_signals['时间'].max()}")


if __name__ == "__main__":
    main()
