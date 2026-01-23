# -*- coding: utf-8 -*-
"""
简化版微观结构监控系统

使用现有价格数据计算proxy指标，无需昂贵的历史期权数据

核心思路:
- 用已知的指标proxy期权Greeks
- 用价格/成交量数据proxy订单流
- 实现从现在开始实时积累数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimplifiedMarketMetrics:
    """简化的市场指标"""
    timestamp: datetime
    price: float
    volatility: float  # Proxy for IV
    volume_ratio: float  # 量能比率
    price_acceleration: float  # 价格加速度
    volume_surge: bool  # 放量突破
    volatility_spike: bool  # 波动率飙升
    liquidity_score: float  # 流动性评分 (0-1)


class SimplifiedMicrostructureMonitor:
    """简化版微观结构监控器"""

    def __init__(self, lookback_periods: int = 20):
        self.lookback = lookback_periods
        self.price_history: List[float] = []
        self.volume_history: List[float] = []

    def calculate_volatility(self, prices: List[float]) -> float:
        """
        计算历史波动率 (Proxy for IV)

        高波动率 → 可能是IV高 → 做空机会
        低波动率 → 可能是IV低 → 做多机会
        """
        if len(prices) < self.lookback:
            return 0.0

        returns = pd.Series(prices).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率

        return volatility

    def calculate_volume_surge(self, current_volume: float) -> bool:
        """
        检测量能激增 (Proxy for Order Flow Pressure)

        当前量能 > 平均量能 * 1.5 = 放量
        """
        if len(self.volume_history) < self.lookback:
            return False

        avg_volume = np.mean(self.volume_history[-self.lookback:])
        is_surge = current_volume > avg_volume * 1.5

        return is_surge

    def calculate_liquidity_score(self, price_change: float, volume: float) -> float:
        """
        计算流动性评分 (Proxy for Order Book Depth)

        高流动性:
        - 成交量大
        - 价格波动小

        低流动性:
        - 成交量小
        - 价格波动大 (闪崩风险)
        """
        # 标准化
        volume_score = min(volume / 1000000, 1.0)  # 假设100万为正常成交量

        # 价格波动率 (越低越好)
        volatility_score = max(1 - abs(price_change) / 0.05, 0)  # 5%为阈值

        liquidity = (volume_score + volatility_score) / 2

        return liquidity

    def detect_crash_risk(self, metrics: SimplifiedMarketMetrics) -> Tuple[bool, float, List[str]]:
        """
        检测闪崩风险 (Proxy for Negative Gamma Trap)

        触发条件:
        1. 高波动率 (> 80%年化)
        2. 大幅下跌 (加速度 < -0.2)
        3. 低流动性 (< 0.3)
        """
        reasons = []
        confidence = 0.0

        # 条件1: 波动率飙升
        if metrics.volatility > 0.8:  # 80%年化波动率
            reasons.append(f"Volatility spike: {metrics.volatility:.1%}")
            confidence += 0.4

        # 条件2: 价格急跌
        if metrics.price_acceleration < -0.2:
            reasons.append(f"Price crash: accel={metrics.price_acceleration:.3f}")
            confidence += 0.4

        # 条件3: 流动性枯竭
        if metrics.liquidity_score < 0.3:
            reasons.append(f"Liquidity dry: score={metrics.liquidity_score:.2f}")
            confidence += 0.2

        is_crash_risk = confidence >= 0.5

        return is_crash_risk, confidence, reasons

    def detect_squeeze_setup(self, metrics: SimplifiedMarketMetrics) -> Tuple[bool, float, List[str]]:
        """
        检测暴涨设置 (Proxy for Vanna Squeeze)

        触发条件:
        1. 低波动率 (< 30%年化) → IV便宜
        2. 放量滞涨 → 吸筹
        3. 流动性充裕 (> 0.7)
        """
        reasons = []
        confidence = 0.0

        # 条件1: 低波动率
        if metrics.volatility < 0.3:  # 30%年化波动率
            reasons.append(f"Low volatility: {metrics.volatility:.1%} (IV可能便宜)")
            confidence += 0.3

        # 条件2: 放量滞涨
        if metrics.volume_surge and abs(metrics.price_acceleration) < 0.1:
            reasons.append("Volume surge without price move (absorption)")
            confidence += 0.4

        # 条件3: 高流动性
        if metrics.liquidity_score > 0.7:
            reasons.append(f"High liquidity: score={metrics.liquidity_score:.2f}")
            confidence += 0.3

        is_squeeze = confidence >= 0.6

        return is_squeeze, confidence, reasons

    def analyze_row(self, row: pd.Series) -> Dict:
        """
        分析单行数据，返回简化版微观结构判断
        """
        timestamp = pd.to_datetime(row['时间'])
        price = row['收盘价']
        volume = row.get('成交量', row.get('量能比率', 1.0) * 1000000)

        # 更新历史
        self.price_history.append(price)
        self.volume_history.append(volume)

        # 保持历史长度
        if len(self.price_history) > 100:
            self.price_history.pop(0)
            self.volume_history.pop(0)

        # 计算指标
        volatility = self.calculate_volatility(self.price_history)

        # 计算价格加速度
        if len(self.price_history) >= 3:
            price_change_1 = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            price_change_2 = (self.price_history[-2] - self.price_history[-3]) / self.price_history[-3]
            acceleration = price_change_1 - price_change_2
        else:
            acceleration = 0

        volume_surge = self.calculate_volume_surge(volume)
        liquidity_score = self.calculate_liquidity_score(acceleration, volume)

        # 构建指标对象
        metrics = SimplifiedMarketMetrics(
            timestamp=timestamp,
            price=price,
            volatility=volatility,
            volume_ratio=row.get('量能比率', 1.0),
            price_acceleration=acceleration,
            volume_surge=volume_surge,
            volatility_spike=volatility > 0.6,
            liquidity_score=liquidity_score
        )

        # 检测市场状态
        is_crash_risk, crash_confidence, crash_reasons = self.detect_crash_risk(metrics)
        is_squeeze, squeeze_confidence, squeeze_reasons = self.detect_squeeze_setup(metrics)

        return {
            'timestamp': timestamp,
            'volatility': volatility,
            'liquidity_score': liquidity_score,
            'is_crash_risk': is_crash_risk,
            'crash_confidence': crash_confidence,
            'crash_reasons': crash_reasons,
            'is_squeeze': is_squeeze,
            'squeeze_confidence': squeeze_confidence,
            'squeeze_reasons': squeeze_reasons,
            'recommendation': self._make_recommendation(is_crash_risk, is_squeeze, metrics)
        }

    def _make_recommendation(self, is_crash_risk: bool, is_squeeze: bool, metrics: SimplifiedMarketMetrics) -> str:
        """生成交易建议"""
        if is_crash_risk:
            return "【警告】高闪崩风险 - 减仓或观望"
        elif is_squeeze:
            return "【机会】低波动吸筹 - 准备突破"
        elif metrics.liquidity_score < 0.4:
            return "【注意】流动性不足 - 谨慎交易"
        else:
            return "【正常】市场平稳 - 按策略交易"


class RealTimeDataCollector:
    """
    实时数据收集器

    从现在开始积累数据，为未来回测做准备
    """

    def __init__(self, output_file: str = "realtime_microstructure_data.csv"):
        self.output_file = output_file
        self.data_buffer: List[Dict] = []
        self.monitor = SimplifiedMicrostructureMonitor()

    def add_data_point(self, timestamp: datetime, price: float, volume: float, signal_action: str = ""):
        """
        添加新数据点

        可以由V8.0策略定时调用
        """
        row = pd.Series({
            '时间': timestamp,
            '收盘价': price,
            '成交量': volume,
            '信号动作': signal_action
        })

        # 分析微观结构
        analysis = self.monitor.analyze_row(row)

        # 保存数据
        data_point = {
            '时间': timestamp,
            '价格': price,
            '成交量': volume,
            '波动率': analysis['volatility'],
            '流动性评分': analysis['liquidity_score'],
            '闪崩风险': analysis['is_crash_risk'],
            '闪崩置信度': analysis['crash_confidence'],
            '暴涨设置': analysis['is_squeeze'],
            '暴涨置信度': analysis['squeeze_confidence'],
            '建议': analysis['recommendation'],
            '信号动作': signal_action
        }

        self.data_buffer.append(data_point)

        # 每100条保存一次
        if len(self.data_buffer) >= 100:
            self.save()

    def save(self):
        """保存数据到CSV"""
        df = pd.DataFrame(self.data_buffer)

        # 追加模式保存
        if pd.api.types.is_file_like(self.output_file):
            df.to_csv(self.output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(self.output_file, index=False, encoding='utf-8-sig')

        print(f"[{datetime.now()}] Saved {len(df)} records to {self.output_file}")
        self.data_buffer = []


def analyze_historical_data():
    """
    分析现有历史数据，生成简化版微观结构标记
    """
    print("="*100)
    print("简化版微观结构分析 - 历史数据")
    print("="*100)

    # 读取现有数据
    df = pd.read_csv('带信号标记_完整数据_修复版.csv', encoding='utf-8-sig')
    df['时间'] = pd.to_datetime(df['时间'])

    print(f"\n[分析数据] {len(df)} rows")

    monitor = SimplifiedMicrostructureMonitor()

    results = []

    for i, row in df.iterrows():
        if i < 20:  # 前20行用于初始化
            monitor.price_history.append(row['收盘价'])
            monitor.volume_history.append(row.get('成交量', 1000000))
            continue

        analysis = monitor.analyze_row(row)

        results.append({
            '时间': analysis['timestamp'],
            '收盘价': row['收盘价'],
            '波动率': f"{analysis['volatility']:.1%}",
            '流动性': f"{analysis['liquidity_score']:.2f}",
            '闪崩风险': 'YES' if analysis['is_crash_risk'] else 'No',
            '暴涨设置': 'YES' if analysis['is_squeeze'] else 'No',
            '建议': analysis['recommendation'],
            '原始信号': row.get('信号动作', '')
        })

    # 保存结果
    result_df = pd.DataFrame(results)

    print(f"\n[统计结果]")
    crash_count = sum(1 for r in results if r['闪崩风险'] == 'YES')
    squeeze_count = sum(1 for r in results if r['暴涨设置'] == 'YES')

    print(f"  闪崩风险时期: {crash_count} ({crash_count/len(results)*100:.1f}%)")
    print(f"  暴涨设置时期: {squeeze_count} ({squeeze_count/len(results)*100:.1f}%)")

    # 显示示例
    print(f"\n[闪崩风险示例]")
    crash_examples = [r for r in results if r['闪崩风险'] == 'YES'][:5]
    for ex in crash_examples:
        print(f"  {ex['时间']}: {ex['建议']}")

    print(f"\n[暴涨设置示例]")
    squeeze_examples = [r for r in results if r['暴涨设置'] == 'YES'][:5]
    for ex in squeeze_examples:
        print(f"  {ex['时间']}: {ex['建议']}")

    # 保存完整结果
    result_df.to_csv('简化微观结构分析结果.csv', index=False, encoding='utf-8-sig')
    print(f"\n[保存] 简化微观结构分析结果.csv")

    return result_df


if __name__ == "__main__":
    print("="*100)
    print("简化版微观结构监控系统")
    print("="*100)

    print("\n[说明]")
    print("由于历史期权和订单流数据难以收集，我们使用现有数据计算proxy指标:")
    print("  1. 波动率 (Proxy for IV)")
    print("  2. 成交量激增 (Proxy for Order Flow)")
    print("  3. 价格加速度 (Proxy for Gamma Exposure)")
    print("  4. 流动性评分 (Proxy for Order Book Depth)")

    print("\n[功能]")
    print("  1. 分析历史数据 - 标记高风险/机会时期")
    print("  2. 实时监控 - 从现在开始积累数据")
    print("  3. 回测对比 - 简化版 vs 原始策略")

    print("\n" + "="*100)
    print("选择:")
    print("="*100)
    print("1. 分析现有历史数据")
    print("2. 启动实时监控 (从现在开始收集)")
    print("3. 运行简化版回测")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        result_df = analyze_historical_data()

    elif choice == "2":
        print("\n[启动实时监控]")
        print("从现在开始收集数据，为未来回测做准备...")
        print("按 Ctrl+C 停止")

        collector = RealTimeDataCollector()

        # 模拟数据添加 (实际应该从V8.0策略获取)
        try:
            while True:
                # 这里应该从实际数据源获取
                # 示例:
                # collector.add_data_point(datetime.now(), current_price, current_volume, signal_action)

                import time
                time.sleep(3600)  # 每小时一次

        except KeyboardInterrupt:
            print("\n[停止]")
            collector.save()

    elif choice == "3":
        print("\n[运行简化版回测]")
        print("基于简化指标进行回测对比...")

        # TODO: 实现简化版回测
        print("请先运行选项1生成分析结果")

    else:
        print("无效选择")

    print("\n" + "="*100)
    print("总结:")
    print("="*100)
    print("虽然无法获取历史期权数据，但我们可以:")
    print("  1. 使用现有价格数据计算proxy指标")
    print("  2. 从现在开始实时积累真实数据")
    print("  3. 逐步验证和优化策略")
    print("\n3-6个月后，我们就有自己的历史数据可以用了！")
