# -*- coding: utf-8 -*-
"""
回测总结的最佳开仓/平仓策略
时间范围：2025年6月-12月
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入V7.0.7系统组件
try:
    from v707_trader_main import (
        V707TraderConfig,
        DataFetcher,
        PhysicsSignalCalculator,
        V705EntryFilter
    )
    USE_SYSTEM = True
    logger.info("成功导入V7.0.7系统组件")
except ImportError as e:
    USE_SYSTEM = False
    logger.warning(f"无法导入系统组件: {e}，将使用简化版本")


def fetch_btc_4h_data(start_date, end_date):
    """从Binance获取BTC 4小时K线数据"""
    try:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

        start_ts = int(start_datetime.timestamp() * 1000)
        end_ts = int(end_datetime.timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        all_data = []

        current_ts = start_ts
        while current_ts < end_ts:
            params = {
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'startTime': current_ts,
                'endTime': end_ts,
                'limit': 1000
            }

            logger.info(f"获取数据: {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')}")
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


def calculate_signals(df):
    """计算信号（张力、加速度等）- 使用简化版计算"""
    logger.info("使用简化版计算信号")

    signals = []

    # 需要至少300条历史数据来计算
    window = 300

    for i in range(window, len(df)):
        current_data = df.iloc[i-window:i].copy()

        # 简化版计算
        tension, acceleration = calculate_simple_metrics(current_data)
        volume_ratio = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-20:-1].mean()
        ema_deviation = (current_data['close'].iloc[-1] - current_data['close'].mean()) / current_data['close'].mean() * 100

        signal_type, confidence, description = diagnose_regime_all(tension, acceleration)

        signal = {
            '时间': current_data.index[-1],
            '收盘价': current_data['close'].iloc[-1],
            '交易量': current_data['volume'].iloc[-1],
            '量能比率': volume_ratio,
            'EMA偏离%': ema_deviation,
            '张力': tension,
            '加速度': acceleration,
            '信号类型': signal_type,
            '置信度': confidence,
            '信号描述': description,
        }

        signals.append(signal)

    logger.info(f"计算完成: {len(signals)}条信号")
    return pd.DataFrame(signals)


def calculate_simple_metrics(df):
    """简化版计算张力和加速度"""
    closes = df['close'].values

    # EMA
    ema = pd.Series(closes).ewm(span=50, adjust=False).mean()

    # 张力：(价格 - EMA) / EMA
    tension = (closes[-1] - ema.iloc[-1]) / ema.iloc[-1]

    # 加速度：最近的价格变化率变化
    if len(closes) >= 10:
        recent_changes = np.diff(closes[-10:])
        acceleration = (recent_changes[-1] - recent_changes[0]) / closes[-1] if len(recent_changes) >= 2 else 0
    else:
        acceleration = 0

    return tension, acceleration


def diagnose_regime_all(tension, acceleration):
    """诊断市场状态"""
    signal_type = 'OSCILLATION'
    confidence = 0.4
    description = f"系统平衡震荡(T={tension:.2f})"

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
        description = f"系统平衡震荡(|T|={abs(tension):.2f}<0.5)"
        signal_type = 'OSCILLATION'

    elif tension > 0.5:
        confidence = 0.6
        description = f"高位震荡(T={tension:.2f}>0.3)"
        signal_type = 'HIGH_OSCILLATION'

    elif tension < -0.5:
        confidence = 0.6
        description = f"低位震荡(T={tension:.2f}<-0.3)"
        signal_type = 'LOW_OSCILLATION'

    return signal_type, confidence, description


def apply_strategy(df):
    """应用总结的策略进行回测"""
    trades = []
    position = None  # None, 'long', 'short'
    entry_price = None
    entry_time = None
    entry_tension = None
    signal_start_time = None
    signal_start_tension = None
    wait_count = 0

    for i, row in df.iterrows():
        tension = row['张力']
        acceleration = row['加速度']
        volume_ratio = row['量能比率']
        price = row['收盘价']

        # 检查信号触发
        if position is None:
            # SHORT信号触发条件
            if tension > 0.5 and acceleration < 0:
                signal_start_time = i
                signal_start_tension = tension
                wait_count = 0

            # LONG信号触发条件
            elif tension < -0.5 and abs(acceleration) < 0.0001:
                signal_start_time = i
                signal_start_tension = tension
                wait_count = 0

            # 等待开仓确认
            if signal_start_time is not None:
                wait_count += 1

                # SHORT开仓确认
                if signal_start_tension > 0.5 and acceleration < 0:
                    if tension > 0.5 and acceleration < 0 and volume_ratio < 1.0:
                        if wait_count >= 1 and wait_count <= 2:  # 1-2个周期
                            position = 'short'
                            entry_price = price
                            entry_time = i
                            entry_tension = tension
                            signal_start_time = None
                            logger.info(f"SHORT开仓: {i}, 价格={price:.2f}, 张力={tension:.6f}")

                # LONG开仓确认
                elif signal_start_tension < -0.5:
                    if tension < -0.55 and acceleration > 0:
                        if wait_count == 1:  # 1个周期
                            position = 'long'
                            entry_price = price
                            entry_time = i
                            entry_tension = tension
                            signal_start_time = None
                            logger.info(f"LONG开仓: {i}, 价格={price:.2f}, 张力={tension:.6f}")

        # 持仓中 - 检查平仓条件
        elif position == 'short':
            tension_change = (entry_tension - tension) / entry_tension if entry_tension != 0 else 0

            # SHORT平仓条件
            if tension_change > 0.4 and (acceleration > 0 or abs(acceleration) < 0.001):
                if volume_ratio > 1.0:
                    pnl = (entry_price - price) / entry_price * 100
                    trades.append({
                        '方向': 'SHORT',
                        '开仓时间': entry_time,
                        '开仓价': entry_price,
                        '开仓张力': entry_tension,
                        '平仓时间': i,
                        '平仓价': price,
                        '平仓张力': tension,
                        '盈亏%': pnl,
                        '持仓周期': (i - entry_time).total_seconds() / 3600 / 4
                    })
                    position = None
                    entry_price = None
                    entry_time = None
                    entry_tension = None
                    logger.info(f"SHORT平仓: {i}, 价格={price:.2f}, 盈亏={pnl:.2f}%")

        elif position == 'long':
            tension_change = (tension - entry_tension) / abs(entry_tension) if entry_tension != 0 else 0

            # LONG平仓条件
            if tension > 0 and tension_change > 1.0 and acceleration > 0:
                pnl = (price - entry_price) / entry_price * 100
                trades.append({
                    '方向': 'LONG',
                    '开仓时间': entry_time,
                    '开仓价': entry_price,
                    '开仓张力': entry_tension,
                    '平仓时间': i,
                    '平仓价': price,
                    '平仓张力': tension,
                    '盈亏%': pnl,
                    '持仓周期': (tension - entry_time).total_seconds() / 3600 / 4
                })
                position = None
                entry_price = None
                entry_time = None
                entry_tension = None
                logger.info(f"LONG平仓: {i}, 价格={price:.2f}, 盈亏={pnl:.2f}%")

    return pd.DataFrame(trades)


def main():
    logger.info("="*80)
    logger.info("开始回测最佳开仓/平仓策略")
    logger.info("时间范围: 2025年6月 - 2025年12月")
    logger.info("="*80)

    # 1. 获取数据
    logger.info("\n步骤1: 获取BTC 4H数据")
    df = fetch_btc_4h_data('2025-06-01', '2025-12-31')

    if df is None:
        logger.error("获取数据失败，退出")
        return

    # 2. 计算信号
    logger.info("\n步骤2: 计算信号")
    signals_df = calculate_signals(df)

    if len(signals_df) == 0:
        logger.error("计算信号失败，退出")
        return

    logger.info(f"生成信号: {len(signals_df)}条")
    logger.info(f"信号时间范围: {signals_df['时间'].min()} 至 {signals_df['时间'].max()}")

    # 3. 应用策略回测
    logger.info("\n步骤3: 应用策略回测")
    trades_df = apply_strategy(signals_df)

    # 4. 分析结果
    logger.info("\n步骤4: 分析回测结果")
    logger.info("="*80)

    if len(trades_df) > 0:
        logger.info(f"\n总交易数: {len(trades_df)}")

        for direction in ['LONG', 'SHORT']:
            dir_trades = trades_df[trades_df['方向'] == direction]
            if len(dir_trades) > 0:
                logger.info(f"\n{direction}交易:")
                logger.info(f"  数量: {len(dir_trades)}")
                logger.info(f"  胜率: {(dir_trades['盈亏%'] > 0).sum() / len(dir_trades) * 100:.2f}%")
                logger.info(f"  平均盈亏: {dir_trades['盈亏%'].mean():.2f}%")
                logger.info(f"  总盈亏: {dir_trades['盈亏%'].sum():.2f}%")
                logger.info(f"  最大盈利: {dir_trades['盈亏%'].max():.2f}%")
                logger.info(f"  最大亏损: {dir_trades['盈亏%'].min():.2f}%")
                logger.info(f"  平均持仓: {dir_trades['持仓周期'].mean():.1f}周期")

        logger.info(f"\n所有交易:")
        logger.info(f"  总盈亏: {trades_df['盈亏%'].sum():.2f}%")
        logger.info(f"  胜率: {(trades_df['盈亏%'] > 0).sum() / len(trades_df) * 100:.2f}%")
        logger.info(f"  盈亏比: {abs(trades_df[trades_df['盈亏%']>0]['盈亏%'].mean() / trades_df[trades_df['盈亏%']<0]['盈亏%'].mean()):.2f}")

        # 保存详细交易记录
        trades_df.to_csv('策略回测结果_2025_6-12月.csv', index=False, encoding='utf-8-sig')
        logger.info(f"\n详细交易记录已保存到: 策略回测结果_2025_6-12月.csv")

        # 显示每笔交易
        logger.info(f"\n详细交易记录:")
        for idx, trade in trades_df.iterrows():
            logger.info(f"  {trade['方向']} | "
                       f"开仓:{trade['开仓时间'].strftime('%Y-%m-%d %H:%M')} "
                       f"@{trade['开仓价']:.2f} (T={trade['开仓张力']:.4f}) | "
                       f"平仓:{trade['平仓时间'].strftime('%Y-%m-%d %H:%M')} "
                       f"@{trade['平仓价']:.2f} (T={trade['平仓张力']:.4f}) | "
                       f"盈亏:{trade['盈亏%']:+.2f}% "
                       f"({trade['持仓周期']:.0f}周期)")

    else:
        logger.warning("未产生任何交易！可能需要调整策略参数")

    # 保存信号数据
    signals_df.to_csv('信号数据_2025_6-12月.csv', index=False, encoding='utf-8-sig')
    logger.info(f"\n信号数据已保存到: 信号数据_2025_6-12月.csv")

    logger.info("\n" + "="*80)
    logger.info("回测完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()
