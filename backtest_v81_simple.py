# -*- coding: utf-8 -*-
"""
V8.1 订单流策略回测（简化版）
更直观地展示策略效果
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simple_backtest_demo():
    """简化版回测演示"""

    print("=" * 70)
    print("V8.1 订单流策略回测演示")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. 加载历史数据...")
    trades_df = pd.read_csv('./historical_data/trades_2026-01-20_2026-01-23.csv')
    trades_df['time'] = pd.to_datetime(trades_df.index)

    cvd_df = pd.read_csv('./historical_data/cvd_2026-01-20_2026-01-23.csv', index_col=0, parse_dates=True)

    print(f"   成交数据: {len(trades_df):,} 笔")
    print(f"   CVD数据: {len(cvd_df)} 个时间点")
    print(f"   时间范围: {trades_df['time'].min()} 至 {trades_df['time'].max()}")

    # 2. 展示数据样本
    print("\n2. 数据样本展示:")
    print("   成交数据（前5笔）:")
    print(trades_df[['time', 'price', 'qty', 'quote_qty']].head())

    print("\n   CVD数据:")
    print(cvd_df.tail())

    # 3. 简化版策略演示
    print("\n3. 策略演示:")

    # 找几个时间点演示
    demo_times = cvd_df.index[:5]

    for i, timestamp in enumerate(demo_times, 1):
        print(f"\n   【时间点 {i}】{timestamp}")

        # 获取该时间附近的成交数据
        window_start = timestamp - timedelta(minutes=30)
        window_end = timestamp + timedelta(minutes=30)

        window_trades = trades_df[
            (trades_df['time'] >= window_start) &
            (trades_df['time'] <= window_end)
        ]

        if len(window_trades) == 0:
            print("   → 该时间段无成交数据")
            continue

        # 计算简单指标
        prices = window_trades['price'].values
        avg_price = np.mean(prices)
        price_change = (prices[-1] - prices[0]) / prices[0] * 100

        # 获取CVD数据
        cvd_row = cvd_df[cvd_df.index == timestamp]
        if cvd_row.empty:
            continue

        cvd_trend = cvd_row['trend'].values[0]
        buy_ratio = cvd_row['buy_ratio'].values[0]

        print(f"   → 平均价格: ${avg_price:,.2f}")
        print(f"   → 价格变化: {price_change:+.2f}%")
        print(f"   → CVD趋势: {cvd_trend}")
        print(f"   → 买入占比: {buy_ratio:.1%}")

        # 模拟信号生成
        if price_change > 0.5:
            signal_direction = 'long'
            signal_strength = 'strong' if price_change > 1.0 else 'moderate'
        elif price_change < -0.5:
            signal_direction = 'short'
            signal_strength = 'strong' if price_change < -1.0 else 'moderate'
        else:
            signal_direction = 'wait'
            signal_strength = 'weak'

        print(f"   → 验证5信号: {signal_direction.upper()} ({signal_strength})")

        # 订单流确认
        order_flow_boost = 0.0

        if signal_direction == 'long':
            if cvd_trend == 'bullish' and buy_ratio > 0.6:
                order_flow_boost = 0.05
                print(f"   → 订单流确认: ✅ CVD看涨 (+5%)")
            elif cvd_trend == 'bearish' and buy_ratio < 0.4:
                order_flow_boost = -0.10
                print(f"   → 订单流确认: ❌ CVD看跌 (-10%)")
            else:
                print(f"   → 订单流确认: 中立 (0%)")
        elif signal_direction == 'short':
            if cvd_trend == 'bearish' and buy_ratio < 0.4:
                order_flow_boost = 0.05
                print(f"   → 订单流确认: ✅ CVD看跌 (+5%)")
            elif cvd_trend == 'bullish' and buy_ratio > 0.6:
                order_flow_boost = -0.10
                print(f"   → 订单流确认: ❌ CVD看涨 (-10%)")
            else:
                print(f"   → 订单流确认: 中立 (0%)")

        # 模拟置信度
        base_confidence = 0.70 if signal_strength == 'strong' else 0.50
        final_confidence = base_confidence + order_flow_boost

        print(f"   → 基础置信度: {base_confidence:.1%}")
        print(f"   → 最终置信度: {final_confidence:.1%}")

        if final_confidence >= 0.6:
            print(f"   → 决策: ✅ 开仓 {signal_direction.upper()}")
        else:
            print(f"   → 决策: ❌ 置信度不足，不开仓")

    # 4. 统计总结
    print("\n" + "=" * 70)
    print("回测总结")
    print("=" * 70)

    total_bullish = len(cvd_df[cvd_df['trend'] == 'bullish'])
    total_bearish = len(cvd_df[cvd_df['trend'] == 'bearish'])
    total_neutral = len(cvd_df[cvd_df['trend'] == 'neutral'])

    print(f"\nCVD趋势分布:")
    print(f"  看涨 (bullish): {total_bullish} 次")
    print(f"  看跌 (bearish): {total_bearish} 次")
    print(f"  中性 (neutral): {total_neutral} 次")

    avg_buy_ratio = cvd_df['buy_ratio'].mean()
    print(f"\n平均买入占比: {avg_buy_ratio:.1%}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("\n说明:")
    print("- 这是一个简化版回测，仅展示策略逻辑")
    print("- 完整回测需要:")
    print("  1. 更长的时间范围（建议至少1个月）")
    print("  2. 4小时K线数据对齐")
    print("  3. 完整的验证5算法实现")
    print("  4. 止盈止损逻辑")
    print("=" * 70)


if __name__ == "__main__":
    simple_backtest_demo()
