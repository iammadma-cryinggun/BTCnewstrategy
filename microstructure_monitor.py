# -*- coding: utf-8 -*-
"""
期权微观结构实时监控引擎

整合:
1. Greeks数据 (宏观脆弱性)
2. OrderFlow数据 (微观催化剂)
3. LCR流动性覆盖率
4. 负Gamma闪崩检测
5. Vanna挤压暴涨检测

输出:
    MarketRegime: 市场状态 + 交易建议 + 置信度
"""

import time
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from options_microstructure import (
    GreeksDataSource, OptionsAnalyzer, OptionsGreeks,
    OrderFlowMetrics, calculate_lcr
)
from orderflow_monitor import OrderFlowMonitor, Trade, OrderBookSnapshot
from gamma_trap_strategy import NegativeGammaTrapDetector, CrashSignal
from vanna_squeeze_strategy import VannaSqueezeDetector, BullishSignal


class MarketRegimeType(Enum):
    """市场状态类型"""
    NEGATIVE_GAMMA_TRAP = "负Gamma陷阱"  # 闪崩风险
    VANNA_SQUEEZE = "Vanna挤压"  # 暴涨机会
    NEUTRAL = "中性市场"  # 观望
    HIGH_VOLATILITY = "高波动"  # 谨慎


@dataclass
class MarketRegime:
    """市场状态综合判断"""
    timestamp: datetime
    regime: MarketRegimeType
    fragility: float  # 脆弱性指数 0-1 (1=极度脆弱)
    catalyst_score: float  # 催化剂得分 0-1 (1=强催化剂)
    lcr: float  # 流动性覆盖率
    gex: float  # Gamma暴露
    iv_percentile: float  # IV历史分位数

    # 信号
    crash_signal: Optional[CrashSignal] = None
    bullish_signal: Optional[BullishSignal] = None

    # 推荐
    recommendation: str = ""
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)


class MicrostructureMonitor:
    """期权微观结构实时监控引擎"""

    def __init__(self,
                 symbol: str = "BTCUSDT",
                 greeks_update_interval: int = 60,  # Greeks数据更新间隔(秒)
                 orderflow_window: int = 300):  # 订单流窗口(秒)

        self.symbol = symbol
        self.greeks_update_interval = greeks_update_interval
        self.orderflow_window = orderflow_window

        # 初始化模块
        self.greeks_source = GreeksDataSource()
        self.greeks_analyzer = OptionsAnalyzer(self.greeks_source)
        self.orderflow_monitor = OrderFlowMonitor(symbol)

        # 检测器
        self.gamma_trap_detector = NegativeGammaTrapDetector()
        self.vanna_squeeze_detector = VannaSqueezeDetector()

        # 状态
        self.current_greeks: Optional[OptionsGreeks] = None
        self.iv_percentile: float = 50.0
        self.is_running = False

        print(f"[初始化] 期权微观结构监控引擎 - {symbol}")

    async def update_greeks_data(self):
        """更新Greeks数据 (每分钟)"""
        try:
            self.current_greeks = self.greeks_analyzer.analyze_greeks("BTC")
            self.iv_percentile = self.greeks_analyzer.calculate_iv_percentile(
                self.current_greeks.atm_iv,
                lookback_days=30
            )

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Greeks数据已更新 "
                  f"| GEX: ${self.current_greeks.gex/1e8:.2f}亿 "
                  f"| IV: {self.current_greeks.atm_iv:.2f}% ({self.iv_percentile:.1f}分位)")

        except Exception as e:
            print(f"[ERROR] Greeks数据更新失败: {e}")

    def on_trade(self, trade: Trade):
        """处理成交数据"""
        self.orderflow_monitor.on_trade(trade)

    def on_orderbook_update(self, snapshot: OrderBookSnapshot):
        """处理订单簿更新"""
        self.orderflow_monitor.on_orderbook_update(snapshot)

    def calculate_lcr(self) -> float:
        """
        计算流动性覆盖率

        LCR = OrderBook_Quantity / |GEX × Price_Change_1%|
        """
        if not self.current_greeks:
            return float('inf')

        orderflow = self.orderflow_monitor.get_metrics()
        hedging_need = abs(self.current_greeks.gex) * 0.01  # 1%价格变动

        lcr = calculate_lcr(orderflow.bid_quantity_1pct, hedging_need)

        return lcr

    def calculate_fragility_index(self) -> float:
        """
        计算市场脆弱性指数 (0-1)

        脆弱性由以下因素决定:
        1. 负GEX程度
        2. LCR (流动性覆盖率)
        3. Skew斜率
        4. VPIN (有毒流量)

        1.0 = 极度脆弱 (即将闪崩)
        0.0 = 稳定
        """
        if not self.current_greeks:
            return 0.0

        orderflow = self.orderflow_monitor.get_metrics()

        fragility_score = 0.0

        # 因素1: 负GEX (权重 40%)
        gex_normalized = min(abs(self.current_greeks.gex) / 1e9, 1.0)  # 10亿USD = 满分
        if self.current_greeks.gex < 0:
            fragility_score += 0.4 * gex_normalized

        # 因素2: LCR倒数 (权重 30%)
        lcr = self.calculate_lcr()
        if lcr < 1.0:
            lcr_score = 1.0 - lcr
        elif lcr < 1.5:
            lcr_score = 0.3
        else:
            lcr_score = 0.0
        fragility_score += 0.3 * lcr_score

        # 因素3: Skew斜率 (权重 15%)
        skew_score = min(self.current_greeks.skew_slope / 10.0, 1.0)
        fragility_score += 0.15 * skew_score

        # 因素4: VPIN (权重 15%)
        vpin_score = min(orderflow.vpin / 0.6, 1.0)
        fragility_score += 0.15 * vpin_score

        return min(fragility_score, 1.0)

    def calculate_catalyst_score(self) -> float:
        """
        计算催化剂得分 (0-1)

        催化剂由以下因素决定:
        1. CVD背离 (吸筹)
        2. 买压/卖压比率
        3. IV分位数 (均值回归空间)
        4. Call Skew

        1.0 = 强催化剂 (即将突破)
        0.0 = 无催化剂
        """
        if not self.current_greeks:
            return 0.0

        orderflow = self.orderflow_monitor.get_metrics()
        catalyst_score = 0.0

        # 因素1: CVD背离 (权重 35%)
        is_diverging, div_type = self.orderflow_monitor.detect_divergence()
        if div_type == "ABSORPTION":
            catalyst_score += 0.35

        # 因素2: 买压/卖压比率 (权重 25%)
        pressure_ratio = orderflow.buy_pressure / max(orderflow.sell_pressure, 1)
        pressure_score = min(pressure_ratio / 3.0, 1.0)  # 3倍买压 = 满分
        catalyst_score += 0.25 * pressure_score

        # 因素3: IV分位数 (反向, 权重 20%)
        iv_score = (100 - self.iv_percentile) / 100
        catalyst_score += 0.20 * iv_score

        # 因素4: Call Skew (权重 20%)
        skew_score = min(self.current_greeks.call_skew / 6.0, 1.0)
        catalyst_score += 0.20 * skew_score

        return min(catalyst_score, 1.0)

    def analyze_market_regime(self) -> MarketRegime:
        """
        综合分析市场状态

        双层过滤系统:
        第一层: Greeks宏观脆弱性
        第二层: OrderFlow微观催化剂
        """
        if not self.current_greeks:
            return MarketRegime(
                timestamp=datetime.now(),
                regime=MarketRegimeType.NEUTRAL,
                fragility=0.0,
                catalyst_score=0.0,
                lcr=float('inf'),
                gex=0,
                iv_percentile=50.0,
                recommendation="等待数据..."
            )

        # 获取订单流指标
        orderflow = self.orderflow_monitor.get_metrics()

        # 执行检测
        crash_signal = self.gamma_trap_detector.detect(
            greeks=self.current_greeks,
            orderflow=orderflow,
            current_price=self.current_greeks.spot_price,
            price_change_1pct=self.current_greeks.spot_price * 0.01
        )

        bullish_signal = self.vanna_squeeze_detector.detect(
            greeks=self.current_greeks,
            orderflow=orderflow,
            iv_percentile=self.iv_percentile
        )

        # 计算综合指标
        fragility = self.calculate_fragility_index()
        catalyst = self.calculate_catalyst_score()
        lcr = self.calculate_lcr()

        # 决策逻辑
        reasoning = []
        regime = MarketRegimeType.NEUTRAL
        recommendation = "观望"
        confidence = 0.0

        # 优先级1: 负Gamma陷阱 (最高风险)
        if crash_signal.is_crash_imminent:
            regime = MarketRegimeType.NEGATIVE_GAMMA_TRAP
            recommendation = crash_signal.recommendation
            confidence = crash_signal.confidence
            reasoning = crash_signal.trigger_reasons

        # 优先级2: Vanna挤压 (暴涨机会)
        elif bullish_signal.is_bullish_setup:
            regime = MarketRegimeType.VANNA_SQUEEZE
            recommendation = bullish_signal.recommendation
            confidence = bullish_signal.confidence
            reasoning = bullish_signal.trigger_reasons

        # 优先级3: 高波动
        elif fragility > 0.6 or catalyst > 0.6:
            regime = MarketRegimeType.HIGH_VOLATILITY
            recommendation = "谨慎交易, 缩小仓位"
            confidence = max(fragility, catalyst)
            reasoning.append(f"脆弱性={fragility:.2f}, 催化剂={catalyst:.2f}")

        # 默认: 中性
        else:
            regime = MarketRegimeType.NEUTRAL
            recommendation = "无明确信号, 保持观望"
            confidence = 0.0

        market_regime = MarketRegime(
            timestamp=datetime.now(),
            regime=regime,
            fragility=fragility,
            catalyst_score=catalyst,
            lcr=lcr,
            gex=self.current_greeks.gex,
            iv_percentile=self.iv_percentile,
            crash_signal=crash_signal,
            bullish_signal=bullish_signal,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning
        )

        return market_regime

    def format_regime_report(self, regime: MarketRegime) -> str:
        """格式化市场状态报告"""
        lines = []
        lines.append("="*100)
        lines.append(f"期权微观结构监控报告 - {self.symbol}")
        lines.append("="*100)

        lines.append(f"\n【时间】 {regime.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"【市场状态】 {regime.regime.value}")
        lines.append(f"【置信度】 {regime.confidence:.1%}")

        lines.append(f"\n{'='*100}")
        lines.append("核心指标")
        lines.append(f"{'='*100}")

        lines.append(f"  脆弱性指数: {regime.fragility:.2f} / 1.0 ({'极度脆弱' if regime.fragility > 0.7 else '脆弱' if regime.fragility > 0.4 else '稳定'})")
        lines.append(f"  催化剂得分: {regime.catalyst_score:.2f} / 1.0 ({'强催化剂' if regime.catalyst_score > 0.7 else '有催化剂' if regime.catalyst_score > 0.4 else '无催化剂'})")
        lines.append(f"  LCR: {regime.lcr:.2f} ({'危险' if regime.lcr < 1.0 else '警戒' if regime.lcr < 1.5 else '安全'})")
        lines.append(f"  GEX: ${regime.gex/1e8:.2f}亿")
        lines.append(f"  IV分位数: {regime.iv_percentile:.1f}%")

        lines.append(f"\n{'='*100}")
        lines.append("交易建议")
        lines.append(f"{'='*100}")
        lines.append(f"  {regime.recommendation}")

        if len(regime.reasoning) > 0:
            lines.append(f"\n{'='*100}")
            lines.append("触发原因")
            lines.append(f"{'='*100}")
            for i, reason in enumerate(regime.reasoning, 1):
                lines.append(f"  [{i}] {reason}")

        # 特殊报告
        if regime.crash_signal and regime.crash_signal.is_crash_imminent:
            lines.append(f"\n{self.gamma_trap_detector.explain(regime.crash_signal)}")

        if regime.bullish_signal and regime.bullish_signal.is_bullish_setup:
            lines.append(f"\n{self.vanna_squeeze_detector.explain(regime.bullish_signal)}")

        lines.append("\n" + "="*100)

        return "\n".join(lines)

    async def run_monitoring_loop(self, duration_seconds: int = 3600):
        """
        运行监控循环

        模拟实时数据流
        """
        print(f"\n[启动] 实时监控循环 - 持续{duration_seconds}秒")
        print("="*100)

        self.is_running = True
        start_time = datetime.now()
        last_greeks_update = datetime.now() - timedelta(seconds=self.greeks_update_interval)

        while self.is_running:
            now = datetime.now()

            # 检查是否更新Greeks数据
            if (now - last_greeks_update).total_seconds() >= self.greeks_update_interval:
                await self.update_greeks_data()
                last_greeks_update = now

                # 分析市场状态
                regime = self.analyze_market_regime()

                # 打印报告
                if regime.confidence > 0.3:  # 只打印有意义的信号
                    print(f"\n{self.format_regime_report(regime)}")

            # 检查是否超时
            if (now - start_time).total_seconds() >= duration_seconds:
                print(f"\n[完成] 监控循环结束")
                break

            # 等待下一次循环
            await asyncio.sleep(5)

    def stop(self):
        """停止监控"""
        self.is_running = False


# ============================================================================
# 整合到现有V8.0系统的接口
# ============================================================================

def integrate_with_v80_dynamic_strategy():
    """
    将期权微观结构监控整合到V8.0动态策略中

    V8.0策略 (变化率检测) + 期权微观结构 (脆弱性+催化剂) = 终极系统
    """

    print("\n" + "="*100)
    print("V8.0动态策略 + 期权微观结构 = 终极交易系统")
    print("="*100)

    print("\n[系统架构]")
    print("  第一层: V8.0突变检测器 (测速仪)")
    print("    - EMA突变 (50%权重)")
    print("    - 量能突变 (30%权重)")
    print("    - 基础量能 (20%权重)")
    print("    → 输出: V8_Score (触发信号)")

    print("\n  第二层: 期权微观结构过滤器 (地图+仪表盘)")
    print("    - Greeks数据 (宏观脆弱性)")
    print("    - OrderFlow数据 (微观催化剂)")
    print("    - LCR流动性覆盖率")
    print("    → 输出: MarketRegime (过滤噪音)")

    print("\n[决策逻辑]")
    print("  IF V8_Score >= 0.7 AND MarketRegime == 'VANNA_SQUEEZE':")
    print("    → 做多 (强信号)")
    print("  ELIF V8_Score >= 0.7 AND MarketRegime == 'NEGATIVE_GAMMA_TRAP':")
    print("    → 跳过 (闪崩风险)")
    print("  ELIF V8_Score >= 0.7 AND MarketRegime == 'NEUTRAL':")
    print("    → 正常交易 (中信号)")
    print("  ELSE:")
    print("    → 观望")

    print("\n[优势]")
    print("  1. V8.0提供时机 (何时进场)")
    print("  2. 期权微观结构提供方向 (做多/做空/观望)")
    print("  3. 避免在\"危险\"区域交易")
    print("  4. 只在\"安全+强催化剂\"时重仓")

    print("\n" + "="*100)


# ============================================================================
# 测试代码
# ============================================================================

async def test_microstructure_monitor():
    """测试监控引擎"""
    print("="*100)
    print("期权微观结构监控引擎 - 测试")
    print("="*100)

    # 创建监控器
    monitor = MicrostructureMonitor(
        symbol="BTCUSDT",
        greeks_update_interval=60,
        orderflow_window=300
    )

    # 初始化Greeks数据
    await monitor.update_greeks_data()

    # 模拟一些交易数据
    now = datetime.now()
    for i in range(200):
        trade = Trade(
            timestamp=now + timedelta(seconds=i),
            price=95000 + np.random.randn() * 50,
            quantity=np.random.uniform(0.1, 3.0),
            is_buyer_maker=np.random.choice([True, False]),
            side='buy' if np.random.random() > 0.5 else 'sell'
        )
        monitor.on_trade(trade)

    # 模拟订单簿快照
    snapshot = OrderBookSnapshot(
        timestamp=now + timedelta(seconds=200),
        bids=[],
        asks=[],
        bid_quantity_1pct=15000000,
        bid_quantity_2pct=30000000,
        ask_quantity_1pct=15000000,
        ask_quantity_2pct=30000000
    )
    monitor.on_orderbook_update(snapshot)

    # 分析市场状态
    regime = monitor.analyze_market_regime()

    # 打印报告
    report = monitor.format_regime_report(regime)
    print(report)

    return regime


if __name__ == "__main__":
    # 测试监控引擎
    asyncio.run(test_microstructure_monitor())

    # 整合说明
    integrate_with_v80_dynamic_strategy()
