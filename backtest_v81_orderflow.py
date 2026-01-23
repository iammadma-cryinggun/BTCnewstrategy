# -*- coding: utf-8 -*-
"""
V8.1 订单流策略回测（使用历史数据）
回测验证5 + CVD + 鲸鱼交易的组合策略
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V81OrderFlowBacktest:
    """V8.1订单流策略回测"""

    def __init__(self, trades_file, cvd_file):
        """
        Args:
            trades_file: 历史成交数据CSV
            cvd_file: CVD数据CSV
        """
        logger.info("加载历史数据...")

        # 加载成交数据
        self.trades_df = pd.read_csv(trades_file)
        # 转换time列
        self.trades_df['time'] = pd.to_datetime(self.trades_df['time'])

        # 加载CVD数据（索引是时间）
        self.cvd_df = pd.read_csv(cvd_file, index_col=0, parse_dates=True)

        logger.info(f"成交数据: {len(self.trades_df)} 笔")
        logger.info(f"  时间范围: {self.trades_df['time'].min()} 至 {self.trades_df['time'].max()}")
        logger.info(f"CVD数据: {len(self.cvd_df)} 个时间点")
        logger.info(f"  时间范围: {self.cvd_df.index.min()} 至 {self.cvd_df.index.max()}")

    def calculate_verification5(self, prices):
        """
        计算验证5指标

        Args:
            prices: 价格数组

        Returns:
            tension, acceleration
        """
        if len(prices) < 10:
            return None, None

        # 简化版验证5
        # 实际应该使用完整的算法
        recent = prices[-10:]

        # 张力 = (当前价格 - 最低价) / (最高价 - 最低价)
        min_price = np.min(recent)
        max_price = np.max(recent)
        current_price = prices[-1]

        if max_price == min_price:
            tension = 0.5
        else:
            tension = (current_price - min_price) / (max_price - min_price)

        # 加速度 = 价格变化的变化率
        if len(prices) >= 20:
            change1 = prices[-5] - prices[-10]
            change2 = prices[-1] - prices[-5]
            acceleration = change2 - change1
        else:
            acceleration = 0

        return tension, acceleration

    def classify_market_state(self, tension, acceleration):
        """
        市场状态分类（简化版）

        Returns:
            signal_type, base_confidence, direction
        """
        # 简化版信号分类
        if tension < 0.3 and acceleration > 0:
            return "LOW_OSCILLATION", 0.70, "long"
        elif tension > 0.7 and acceleration < 0:
            return "HIGH_OSCILLATION", 0.70, "short"
        elif acceleration > 0.01:
            return "BULLISH_TREND", 0.60, "long"
        elif acceleration < -0.01:
            return "BEARISH_TREND", 0.60, "short"
        else:
            return "TRANSITION", 0.30, "wait"

    def run_backtest(self, confidence_threshold=0.6):
        """
        运行回测

        Args:
            confidence_threshold: 置信度门槛
        """
        logger.info("=" * 70)
        logger.info("开始回测 V8.1 订单流策略")
        logger.info("=" * 70)

        trades = []
        total_signals = 0
        filtered_signals = 0
        taken_trades = 0

        # 按天分组
        self.trades_df['date'] = self.trades_df['time'].dt.date
        self.trades_df['hour'] = self.trades_df['time'].dt.hour

        # 每4小时检查一次（0, 4, 8, 12, 16, 20）
        check_hours = [0, 4, 8, 12, 16, 20]

        for date, group in self.trades_df.groupby('date'):
            for hour in check_hours:
                # 筛选该时间段的数据
                mask = (group['hour'] == hour)
                hour_data = group[mask]

                if len(hour_data) < 10:
                    continue

                # 获取该时间点的价格序列
                prices = hour_data['price'].values

                # 1. 计算验证5信号
                tension, acceleration = self.calculate_verification5(prices)
                if tension is None:
                    continue

                signal_type, base_confidence, direction = self.classify_market_state(
                    tension, acceleration
                )

                total_signals += 1

                # 2. 获取订单流数据
                timestamp = pd.Timestamp(date) + timedelta(hours=hour)

                # 找到最近的CVD数据
                cvd_data = self.cvd_df[self.cvd_df.index <= timestamp].tail(1)

                if cvd_data.empty:
                    continue

                cvd_trend = cvd_data['trend'].values[0]
                buy_ratio = cvd_data['buy_ratio'].values[0]

                # 3. 订单流调整
                order_flow_boost = 0.0

                if direction == 'long':
                    if cvd_trend == 'bullish' and buy_ratio > 0.6:
                        order_flow_boost += 0.05
                    elif cvd_trend == 'bearish' and buy_ratio < 0.4:
                        order_flow_boost -= 0.10
                elif direction == 'short':
                    if cvd_trend == 'bearish' and buy_ratio < 0.4:
                        order_flow_boost += 0.05
                    elif cvd_trend == 'bullish' and buy_ratio > 0.6:
                        order_flow_boost -= 0.10

                # 4. 最终置信度
                final_confidence = base_confidence + order_flow_boost
                final_confidence = max(0, min(final_confidence, 1.0))

                # 5. 置信度过滤
                if final_confidence < confidence_threshold:
                    filtered_signals += 1
                    continue

                # 6. 开仓
                if direction == 'wait':
                    continue

                entry_price = prices[-1]
                entry_time = timestamp

                # 计算止盈止损
                if direction == 'long':
                    stop_loss = entry_price * 0.97
                    take_profit = entry_price * 1.10
                else:
                    stop_loss = entry_price * 1.03
                    take_profit = entry_price * 0.90

                # 记录交易
                trade = {
                    'time': entry_time,
                    'signal_type': signal_type,
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'base_confidence': base_confidence,
                    'order_flow_boost': order_flow_boost,
                    'final_confidence': final_confidence,
                    'tension': tension,
                    'acceleration': acceleration,
                    'cvd_trend': cvd_trend,
                    'buy_ratio': buy_ratio
                }

                trades.append(trade)
                taken_trades += 1

                logger.info(f"开仓: {direction.upper()} @ ${entry_price:,.2f} | 置信度: {final_confidence:.2f}")

        # 保存结果
        results_df = pd.DataFrame(trades)

        if not results_df.empty:
            output_file = f"backtest_v81_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"✅ 回测结果已保存: {output_file}")

        # 统计
        logger.info("=" * 70)
        logger.info("回测统计")
        logger.info("=" * 70)
        logger.info(f"总信号数: {total_signals}")
        logger.info(f"过滤信号数: {filtered_signals}")
        logger.info(f"开仓次数: {taken_trades}")
        logger.info(f"信号过滤率: {filtered_signals/total_signals*100:.1f}%")

        if not results_df.empty:
            logger.info(f"\n交易分布:")
            logger.info(f"  做多: {len(results_df[results_df['direction']=='long'])} 次")
            logger.info(f"  做空: {len(results_df[results_df['direction']=='short'])} 次")

            logger.info(f"\n平均置信度:")
            logger.info(f"  基础: {results_df['base_confidence'].mean():.2f}")
            logger.info(f"  订单流调整: {results_df['order_flow_boost'].mean():.3f}")
            logger.info(f"  最终: {results_df['final_confidence'].mean():.2f}")

        logger.info("=" * 70)

        return results_df


# 使用示例
if __name__ == "__main__":
    import os

    # 检查历史数据是否存在
    historical_dir = './historical_data'

    if not os.path.exists(historical_dir):
        print("错误: 历史数据目录不存在！")
        print("请先运行: python collect_orderflow_history.py")
        sys.exit(1)

    # 查找最新的数据文件
    trades_files = [f for f in os.listdir(historical_dir) if f.startswith('trades_')]
    cvd_files = [f for f in os.listdir(historical_dir) if f.startswith('cvd_')]

    if not trades_files or not cvd_files:
        print("错误: 未找到历史数据文件！")
        print("请先运行: python collect_orderflow_history.py")
        sys.exit(1)

    trades_file = f"{historical_dir}/{trades_files[-1]}"
    cvd_file = f"{historical_dir}/{cvd_files[-1]}"

    print(f"使用数据文件:")
    print(f"  成交数据: {trades_file}")
    print(f"  CVD数据: {cvd_file}")

    # 运行回测
    backtest = V81OrderFlowBacktest(trades_file, cvd_file)
    results = backtest.run_backtest(confidence_threshold=0.6)

    if results is not None and not results.empty:
        print("\n" + "=" * 70)
        print("交易信号预览:")
        print("=" * 70)
        print(results[['time', 'direction', 'entry_price', 'final_confidence', 'cvd_trend']].head(10))
