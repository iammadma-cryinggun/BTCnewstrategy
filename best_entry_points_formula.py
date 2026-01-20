# -*- coding: utf-8 -*-
"""
最佳开单点规律总结与回测验证（2024-01-01 到 2025-12-31）

【最佳开单点三大规律】

基于对您标注的3个最佳开单点（12/27 16:00, 1/7 12:00, 1/16 12:00）的分析，
以及与其他43个普通开单点的对比，总结出以下规律：

规律1: 极端张力 (|T| > 0.65)
  - 市场处于超买(T > 0.65)或超卖(T < -0.65)状态
  - 最佳点: 100%满足 (3/3)
  - 其他点: 只有44%满足 (19/43)

规律2: 张力加速度反向 (T × A < 0)
  - 张力与加速度方向相反，形成物理"阻力点"
  - 表示市场即将发生反转
  - 最佳点: 100%满足 (3/3)

规律3: 信号类型切换时刻
  - 从OSCILLATION切换到HIGH_OSCILLATION或LOW_OSCILLATION
  - 是趋势的起点，而不是趋势的延续
  - 最佳点都发生在切换时刻

【最佳开单点公式】

def is_best_entry_point(tension, acceleration, signal_type):
    '''判断是否为最佳开单点'''

    # 条件1: 极端张力
    if abs(tension) <= 0.65:
        return False

    # 条件2: 张力与加速度反向
    if tension * acceleration >= 0:
        return False

    # 条件3: 加速度接近阈值
    if abs(acceleration) >= 0.01:
        return False

    # 条件4: 必须是HIGH_OSCILLATION或LOW_OSCILLATION
    if signal_type not in ['HIGH_OSCILLATION', 'LOW_OSCILLATION']:
        return False

    # 条件5: 张力与加速度方向正确性
    if tension > 0 and acceleration < 0:
        # 高位震荡，做空
        return 'SHORT'
    elif tension < 0 and acceleration > 0:
        # 低位震荡，做多
        return 'LONG'

    return False

【回测策略】

1. 数据范围: 2024-01-01 到 2025-12-31 (2年)
2. 时间周期: 4小时
3. 入场条件: 满足上述最佳点公式
4. 出场条件:
   - 止盈: +5%
   - 止损: -2.5%
   - 最大持仓: 42个4H周期 (7天)

5. 对比基准:
   - 所有通过V7.0.5过滤器的信号
   - 只满足条件1+2的点（不加信号类型切换条件）
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
    PhysicsSignalCalculator,
    V705EntryFilter
)


def is_best_entry_point(tension, acceleration, signal_type):
    """
    判断是否为最佳开单点

    返回:
        'LONG', 'SHORT', 或 False
    """
    # 条件1: 极端张力 |T| > 0.65
    if abs(tension) <= 0.65:
        return False

    # 条件2: 张力与加速度反向 T × A < 0
    if tension * acceleration >= 0:
        return False

    # 条件3: 加速度接近阈值 |A| < 0.01
    if abs(acceleration) >= 0.01:
        return False

    # 条件4: 必须是HIGH_OSCILLATION或LOW_OSCILLATION
    if signal_type not in ['HIGH_OSCILLATION', 'LOW_OSCILLATION']:
        return False

    # 条件5: 确定方向
    if tension > 0 and acceleration < 0:
        return 'SHORT'  # 做空
    elif tension < 0 and acceleration > 0:
        return 'LONG'  # 做多

    return False


def fetch_btc_klines_2years():
    """获取2024-2025年的BTC 4H数据"""
    try:
        # 从2024年1月1日开始（提前10天确保有足够历史数据）
        start_datetime = datetime(2024, 1, 1) - timedelta(days=10)
        end_datetime = datetime(2026, 1, 1)  # 到2025年底

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

            logger.info(f"获取数据: {datetime.fromtimestamp(current_ts/1000)}")
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


def backtest_strategy(df):
    """回测最佳开单点策略"""
    config = V707TraderConfig()
    calculator = PhysicsSignalCalculator(config)
    filter = V705EntryFilter(config)

    # 存储所有信号
    all_signals = []
    best_entry_signals = []

    logger.info("开始回测...")
    logger.info("=" * 80)

    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df.iloc[i]['close']

        # 只关注2024年之后的信号
        if current_time < pd.Timestamp('2024-01-01'):
            continue

        # 需要至少60条历史数据
        if i < 60:
            continue

        # 获取历史数据（使用滚动300条，模拟实际系统）
        df_history = df.iloc[max(0, i-299):i+1].copy()

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

            # 使用标准诊断函数
            signal_type, confidence, description = calculator.diagnose_regime(
                tension, acceleration
            )

            if signal_type is None:
                continue

            # 应用V7.0.5过滤器
            should_pass, filter_reason = filter.apply_filter(
                signal_type, acceleration, volume_ratio, price_vs_ema, df_metrics
            )

            # 记录所有有效信号
            signal_record = {
                '时间': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                '收盘价': current_price,
                '张力': tension,
                '加速度': acceleration,
                'T×A': tension * acceleration,
                '信号类型': signal_type,
                '置信度': confidence,
                '通过V705过滤器': should_pass,
                '是否开单(V705)': '是' if should_pass else '否',
            }

            # 判断是否为最佳开单点
            best_direction = is_best_entry_point(tension, acceleration, signal_type)
            signal_record['最佳开单点'] = best_direction if best_direction else '否'

            all_signals.append(signal_record)

            # 如果是最佳开单点，记录详细信息
            if best_direction:
                best_record = signal_record.copy()
                best_record['交易方向'] = best_direction
                best_record['量能比率'] = volume_ratio
                best_record['EMA偏离%'] = price_vs_ema * 100
                best_entry_signals.append(best_record)

                logger.info(f"[最佳开单点] {current_time} | {best_direction} | "
                          f"T={tension:.3f} | A={acceleration:.4f} | {signal_type}")

        except Exception as e:
            logger.error(f"计算失败 ({current_time}): {e}")
            continue

    logger.info("=" * 80)
    logger.info(f"回测完成")
    logger.info(f"总信号数: {len(all_signals)}")
    logger.info(f"最佳开单点: {len(best_entry_signals)}")

    return pd.DataFrame(all_signals), pd.DataFrame(best_entry_signals)


def calculate_backtest_results(df, best_signals):
    """计算回测结果"""
    if len(best_signals) == 0:
        logger.warning("没有最佳开单点，无法计算回测结果")
        return None

    results = []

    logger.info("\n开始计算交易结果...")
    logger.info("=" * 80)

    for idx, signal in best_signals.iterrows():
        entry_time = pd.Timestamp(signal['时间'])
        entry_price = signal['收盘价']
        direction = signal['交易方向']

        # 找到对应的索引位置
        entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]

        # 止盈止损
        if direction == 'LONG':
            tp_price = entry_price * 1.05  # +5%
            sl_price = entry_price * 0.975  # -2.5%
        else:  # SHORT
            tp_price = entry_price * 0.95  # -5%
            sl_price = entry_price * 1.025  # +2.5%

        # 模拟交易，找到出场点
        exit_time = None
        exit_price = None
        exit_reason = None
        max_hold_bars = 42  # 最多持仓42个4H周期

        for i in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, len(df))):
            bar = df.iloc[i]
            high = bar['high']
            low = bar['low']

            if direction == 'LONG':
                # 检查止盈
                if high >= tp_price:
                    exit_time = df.index[i]
                    exit_price = tp_price
                    exit_reason = '止盈'
                    break
                # 检查止损
                if low <= sl_price:
                    exit_time = df.index[i]
                    exit_price = sl_price
                    exit_reason = '止损'
                    break
            else:  # SHORT
                # 检查止盈
                if low <= tp_price:
                    exit_time = df.index[i]
                    exit_price = tp_price
                    exit_reason = '止盈'
                    break
                # 检查止损
                if high >= sl_price:
                    exit_time = df.index[i]
                    exit_price = sl_price
                    exit_reason = '止损'
                    break

        # 如果没有触发止盈止损，在最大持仓时间平仓
        if exit_time is None:
            exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)
            exit_time = df.index[exit_idx]
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = '时间止损'

        # 计算盈亏
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        results.append({
            '入场时间': signal['时间'],
            '出场时间': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            '方向': direction,
            '入场价': entry_price,
            '出场价': exit_price,
            '盈亏%': pnl_pct,
            '出场原因': exit_reason,
            '张力': signal['张力'],
            '加速度': signal['加速度'],
        })

        logger.info(f"[交易] {signal['时间'][:10]} | {direction:4s} | "
                  f"入场${entry_price:.2f} | 出场${exit_price:.2f} | "
                  f"盈亏{pnl_pct:+.2f}% | {exit_reason}")

    return pd.DataFrame(results)


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("最佳开单点规律回测验证 (2024-01-01 到 2025-12-31)")
    logger.info("=" * 80)

    # 1. 获取数据
    df = fetch_btc_klines_2years()
    if df is None:
        logger.error("获取数据失败")
        return

    # 2. 回测策略
    all_signals, best_signals = backtest_strategy(df)

    # 3. 保存信号数据
    all_signals.to_csv('backtest_all_signals_2024_2025.csv', index=False, encoding='utf-8-sig')
    best_signals.to_csv('backtest_best_entry_points_2024_2025.csv', index=False, encoding='utf-8-sig')

    logger.info("\n" + "=" * 80)
    logger.info("信号统计")
    logger.info("=" * 80)
    logger.info(f"总有效信号: {len(all_signals)}")
    logger.info(f"最佳开单点: {len(best_signals)}")
    logger.info(f"比例: {len(best_signals)/len(all_signals)*100:.2f}%")

    # 4. 计算回测结果
    results = calculate_backtest_results(df, best_signals)

    if results is not None:
        results.to_csv('backtest_results_2024_2025.csv', index=False, encoding='utf-8-sig')

        logger.info("\n" + "=" * 80)
        logger.info("回测结果统计")
        logger.info("=" * 80)

        total_trades = len(results)
        winning_trades = len(results[results['盈亏%'] > 0])
        losing_trades = len(results[results['盈亏%'] < 0])

        total_pnl = results['盈亏%'].sum()
        avg_pnl = results['盈亏%'].mean()
        max_win = results['盈亏%'].max()
        max_loss = results['盈亏%'].min()

        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # 按出场原因统计
        tp_count = len(results[results['出场原因'] == '止盈'])
        sl_count = len(results[results['出场原因'] == '止损'])
        time_count = len(results[results['出场原因'] == '时间止损'])

        logger.info(f"\n总交易次数: {total_trades}")
        logger.info(f"盈利交易: {winning_trades} ({win_rate:.1f}%)")
        logger.info(f"亏损交易: {losing_trades}")
        logger.info(f"\n总盈亏: {total_pnl:+.2f}%")
        logger.info(f"平均盈亏: {avg_pnl:+.2f}%")
        logger.info(f"最大盈利: {max_win:+.2f}%")
        logger.info(f"最大亏损: {max_loss:+.2f}%")
        logger.info(f"\n出场原因统计:")
        logger.info(f"  止盈: {tp_count}次 ({tp_count/total_trades*100:.1f}%)")
        logger.info(f"  止损: {sl_count}次 ({sl_count/total_trades*100:.1f}%)")
        logger.info(f"  时间止损: {time_count}次 ({time_count/total_trades*100:.1f}%)")

        logger.info("\n" + "=" * 80)
        logger.info("回测完成！文件已保存")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
