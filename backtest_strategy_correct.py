# -*- coding: utf-8 -*-
"""
回测总结的最佳开仓/平仓策略（使用正确的FFT计算）
时间范围：2025年6月-12月
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from scipy.fft import fft, ifft
from scipy.signal import detrend, hilbert


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


def calculate_physics_metrics(df):
    """使用FFT和希尔伯特变换计算物理指标"""
    if len(df) < 60:
        return None

    try:
        prices = df['close'].values

        # 去趋势
        d_prices = detrend(prices)

        # FFT滤波（保留前8个频率分量）
        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real

        # 希尔伯特变换
        analytic = hilbert(filtered)
        tension = np.imag(analytic)

        # 标准化
        if len(tension) > 1 and np.std(tension) > 0:
            tension_normalized = (tension - np.mean(tension)) / np.std(tension)
        else:
            tension_normalized = tension

        # 计算加速度
        acceleration = np.zeros_like(tension_normalized)
        for i in range(2, len(tension_normalized)):
            current_tension = tension_normalized[i]
            prev_tension = tension_normalized[i-1]
            prev2_tension = tension_normalized[i-2]

            velocity = current_tension - prev_tension
            acceleration[i] = velocity - (prev_tension - prev2_tension)

        return {
            'tension': tension_normalized[-1],
            'acceleration': acceleration[-1]
        }

    except Exception as e:
        logger.error(f"物理指标计算失败: {e}")
        return None


def diagnose_regime(tension, acceleration):
    """诊断市场状态"""
    TENSION_THRESHOLD = 0.35
    ACCEL_THRESHOLD = 0.02
    OSCILLATION_BAND = 0.5

    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        confidence = 0.7
        description = f"奇点看空(T={tension:.2f}>={TENSION_THRESHOLD})"
        signal_type = 'BEARISH_SINGULARITY'

    elif tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        confidence = 0.6
        description = f"奇点看涨(T={tension:.2f}<=-{TENSION_THRESHOLD})"
        signal_type = 'BULLISH_SINGULARITY'

    elif abs(tension) < OSCILLATION_BAND and abs(acceleration) < ACCEL_THRESHOLD:
        confidence = 0.8
        signal_type = 'OSCILLATION'
        description = f"系统平衡震荡(|T|={abs(tension):.2f}<{OSCILLATION_BAND})"

    elif tension > 0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        signal_type = 'HIGH_OSCILLATION'
        description = f"高位震荡(T={tension:.2f}>0.3)"

    elif tension < -0.3 and abs(acceleration) < 0.01:
        confidence = 0.6
        signal_type = 'LOW_OSCILLATION'
        description = f"低位震荡(T={tension:.2f}<-0.3)"

    else:
        signal_type = 'OSCILLATION'
        confidence = 0.4
        description = f"其他状态(T={tension:.2f})"

    return signal_type, confidence, description


def calculate_signals(df):
    """计算所有信号"""
    logger.info("使用FFT+希尔伯特变换计算信号")

    signals = []
    window = 300

    for i in range(window, len(df)):
        current_data = df.iloc[i-window:i].copy()

        # 计算物理指标
        metrics = calculate_physics_metrics(current_data)
        if metrics is None:
            continue

        tension = metrics['tension']
        acceleration = metrics['acceleration']

        # 计算量能比率
        volume_ratio = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-20:-1].mean()

        # 计算EMA偏离
        ema = pd.Series(current_data['close'].values).ewm(span=20, adjust=False).mean().iloc[-1]
        ema_deviation = (current_data['close'].iloc[-1] - ema) / ema * 100

        # 诊断市场状态
        signal_type, confidence, description = diagnose_regime(tension, acceleration)

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

        if len(signals) % 100 == 0:
            logger.info(f"已处理: {len(signals)}条信号")

    logger.info(f"计算完成: {len(signals)}条信号")
    return pd.DataFrame(signals)


def apply_strategy(df):
    """应用总结的策略进行回测"""
    trades = []
    position = None
    entry_price = None
    entry_idx = None
    entry_tension = None
    signal_start_idx = None
    signal_start_tension = None
    wait_count = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        tension = row['张力']
        acceleration = row['加速度']
        volume_ratio = row['量能比率']
        price = row['收盘价']

        # 检查信号触发
        if position is None:
            # SHORT信号触发条件
            if tension > 0.5 and acceleration < 0:
                signal_start_idx = idx
                signal_start_tension = tension
                wait_count = 0

            # LONG信号触发条件
            elif tension < -0.5 and abs(acceleration) < 0.0001:
                signal_start_idx = idx
                signal_start_tension = tension
                wait_count = 0

            # 等待开仓确认
            if signal_start_idx is not None:
                wait_count += 1

                # SHORT开仓确认
                if signal_start_tension > 0.5 and acceleration < 0:
                    if tension > 0.5 and acceleration < 0 and volume_ratio < 1.0:
                        if wait_count >= 1 and wait_count <= 2:
                            position = 'short'
                            entry_price = price
                            entry_idx = idx
                            entry_tension = tension
                            signal_start_idx = None
                            logger.info(f"SHORT开仓: {row['时间']}, 价格={price:.2f}, 张力={tension:.4f}")

                # LONG开仓确认
                elif signal_start_tension < -0.5:
                    if tension < -0.55 and acceleration > 0:
                        if wait_count == 1:
                            position = 'long'
                            entry_price = price
                            entry_idx = idx
                            entry_tension = tension
                            signal_start_idx = None
                            logger.info(f"LONG开仓: {row['时间']}, 价格={price:.2f}, 张力={tension:.4f}")

        # 持仓中 - 检查平仓条件
        elif position == 'short':
            tension_change = (entry_tension - tension) / entry_tension if entry_tension != 0 else 0

            # SHORT平仓条件
            if tension_change > 0.4 and (acceleration > 0 or abs(acceleration) < 0.001):
                if volume_ratio > 1.0:
                    pnl = (entry_price - price) / entry_price * 100
                    trades.append({
                        '方向': 'SHORT',
                        '开仓时间': df.iloc[entry_idx]['时间'],
                        '开仓价': entry_price,
                        '开仓张力': entry_tension,
                        '平仓时间': row['时间'],
                        '平仓价': price,
                        '平仓张力': tension,
                        '盈亏%': pnl,
                        '持仓周期': idx - entry_idx
                    })
                    position = None
                    entry_price = None
                    entry_idx = None
                    entry_tension = None
                    logger.info(f"SHORT平仓: {row['时间']}, 价格={price:.2f}, 盈亏={pnl:.2f}%")

        elif position == 'long':
            tension_change = (tension - entry_tension) / abs(entry_tension) if entry_tension != 0 else 0

            # LONG平仓条件
            if tension > 0 and tension_change > 1.0 and acceleration > 0:
                pnl = (price - entry_price) / entry_price * 100
                trades.append({
                    '方向': 'LONG',
                    '开仓时间': df.iloc[entry_idx]['时间'],
                    '开仓价': entry_price,
                    '开仓张力': entry_tension,
                    '平仓时间': row['时间'],
                    '平仓价': price,
                    '平仓张力': tension,
                    '盈亏%': pnl,
                    '持仓周期': idx - entry_idx
                })
                position = None
                entry_price = None
                entry_idx = None
                entry_tension = None
                logger.info(f"LONG平仓: {row['时间']}, 价格={price:.2f}, 盈亏={pnl:.2f}%")

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

    # 保存信号数据
    signals_df.to_csv('信号数据_2025_6-12月_正确版.csv', index=False, encoding='utf-8-sig')
    logger.info(f"信号数据已保存到: 信号数据_2025_6-12月_正确版.csv")

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

        if len(trades_df[trades_df['盈亏%']>0]) > 0 and len(trades_df[trades_df['盈亏%']<0]) > 0:
            logger.info(f"  盈亏比: {abs(trades_df[trades_df['盈亏%']>0]['盈亏%'].mean() / trades_df[trades_df['盈亏%']<0]['盈亏%'].mean()):.2f}")

        # 保存详细交易记录
        trades_df.to_csv('策略回测结果_2025_6-12月_正确版.csv', index=False, encoding='utf-8-sig')
        logger.info(f"\n详细交易记录已保存到: 策略回测结果_2025_6-12月_正确版.csv")

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

    logger.info("\n" + "="*80)
    logger.info("回测完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()
