# -*- coding: utf-8 -*-
"""
Vanna挤压暴涨检测模型 (Vanna Squeeze & Absorption)

核心逻辑:
1. 宏观条件: 低IV + Call Skew上升 (IV均值回归 + Vanna效应)
2. 微观触发: CVD背离 + 被动吸筹 + 冰山订单
3. 交易指令: 建立多头头寸

机制:
    IV下降 → Call Delta增加 → 做市商必须买入期货对冲 → 推升价格
    价格上涨 → 进一步压低IV → 形成正反馈循环 (Gamma Squeeze)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from options_microstructure import OptionsGreeks
from orderflow_monitor import OrderFlowMetrics


@dataclass
class BullishSignal:
    """暴涨信号数据结构"""
    timestamp: datetime
    is_bullish_setup: bool  # 是否形成暴涨设置
    confidence: float  # 置信度 0-1
    iv_percentile: float  # IV历史分位数
    call_skew: float  # Call Skew
    cvd_divergence: bool  # CVD背离
    absorption_detected: bool  # 吸筹检测
    call_oi_above: float  # 上方Call持仓量
    trigger_reasons: list  # 触发原因列表
    recommendation: str  # 交易建议


class VannaSqueezeDetector:
    """Vanna挤压检测器"""

    def __init__(self,
                 iv_percentile_threshold: float = 30.0,  # IV < 30分位 (便宜)
                 call_skew_threshold: float = 3.0,  # Call Skew > 3.0
                 absorption_threshold: float = 0.5):  # CVD背离阈值

        self.iv_percentile_threshold = iv_percentile_threshold
        self.call_skew_threshold = call_skew_threshold
        self.absorption_threshold = absorption_threshold

    def detect(self,
               greeks: OptionsGreeks,
               orderflow: OrderFlowMetrics,
               iv_percentile: float) -> BullishSignal:

        """
        综合检测Vanna挤压暴涨信号

        参数:
            greeks: 期权Greeks数据
            orderflow: 订单流指标
            iv_percentile: IV历史分位数

        返回:
            BullishSignal: 暴涨信号对象
        """
        trigger_reasons = []
        confidence_score = 0.0

        # ===== 条件1: IV均值回归空间 =====
        iv_condition = iv_percentile < self.iv_percentile_threshold
        if iv_condition:
            trigger_reasons.append(f"IV分位数={iv_percentile:.1f}% (低位, 有均值回归空间)")
            confidence_score += 0.25

        # ===== 条件2: Call Skew上升 =====
        call_skew_condition = greeks.call_skew > self.call_skew_threshold
        if call_skew_condition:
            trigger_reasons.append(f"Call Skew={greeks.call_skew:.2f} (看涨情绪升温)")
            confidence_score += 0.2

        # ===== 条件3: 上方巨量Call (Gamma Squeeze潜力) =====
        has_call_wall = greeks.call_oi_above > 500000000  # >5亿USD
        if has_call_wall:
            trigger_reasons.append(f"上方Call持仓=${greeks.call_oi_above/1e8:.2f}亿 (突破将触发Gamma Squeeze)")
            confidence_score += 0.15

        # ===== 条件4: CVD背离 (吸筹) =====
        cvd_divergence = (
            orderflow.price_trend < 0 and  # 价格下跌
            orderflow.cvd_trend > self.absorption_threshold  # CVD上升
        )
        if cvd_divergence:
            trigger_reasons.append(f"吸筹背离: 价格下跌(趋势={orderflow.price_trend:.2f}) + CVD上升(趋势={orderflow.cvd_trend:.2f})")
            confidence_score += 0.25

        # ===== 条件5: 买压 > 卖压 =====
        buying_pressure = orderflow.buy_pressure > orderflow.sell_pressure * 1.5
        if buying_pressure:
            trigger_reasons.append(f"买压=${orderflow.buy_pressure/1e6:.1f}M > 卖压=${orderflow.sell_pressure/1e6:.1f}M")
            confidence_score += 0.15

        # ===== 综合判断 =====
        is_bullish_setup = (
            iv_condition and  # 必须满足低IV
            call_skew_condition and  # 且满足Call Skew上升
            (cvd_divergence or buying_pressure)  # 且满足吸筹或买压
        )

        # 生成交易建议
        if is_bullish_setup:
            if confidence_score >= 0.7:
                recommendation = "【强信号】建立多头头寸 (Long Call或现货)"
            elif confidence_score >= 0.5:
                recommendation = "【中等信号】轻仓做多 + 突破确认"
            else:
                recommendation = "【观察】等待更多确认信号"
        else:
            recommendation = "【观望】无暴涨设置"

        signal = BullishSignal(
            timestamp=datetime.now(),
            is_bullish_setup=is_bullish_setup,
            confidence=min(confidence_score, 1.0),
            iv_percentile=iv_percentile,
            call_skew=greeks.call_skew,
            cvd_divergence=cvd_divergence,
            absorption_detected=cvd_divergence,
            call_oi_above=greeks.call_oi_above,
            trigger_reasons=trigger_reasons,
            recommendation=recommendation
        )

        return signal

    def explain(self, signal: BullishSignal) -> str:
        """
        生成信号解释报告
        """
        report = []
        report.append("="*100)
        report.append("Vanna挤压暴涨检测报告")
        report.append("="*100)

        report.append(f"\n【时间】 {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"【状态】 {'暴涨设置' if signal.is_bullish_setup else '无信号'}")
        report.append(f"【置信度】 {signal.confidence:.1%}")

        report.append(f"\n{'='*100}")
        report.append("核心指标")
        report.append(f"{'='*100}")

        report.append(f"  IV分位数: {signal.iv_percentile:.1f}%")
        report.append(f"  Call Skew: {signal.call_skew:.2f}")
        report.append(f"  上方Call持仓: ${signal.call_oi_above/1e8:.2f}亿")
        report.append(f"  吸筹背离: {'是' if signal.cvd_divergence else '否'}")

        report.append(f"\n{'='*100}")
        report.append("触发原因")
        report.append(f"{'='*100}")

        if len(signal.trigger_reasons) == 0:
            report.append("  无触发条件")
        else:
            for i, reason in enumerate(signal.trigger_reasons, 1):
                report.append(f"  [{i}] {reason}")

        report.append(f"\n{'='*100}")
        report.append("交易建议")
        report.append(f"{'='*100}")
        report.append(f"  {signal.recommendation}")

        # 机制解释
        if signal.is_bullish_setup:
            report.append(f"\n{'='*100}")
            report.append("机制解释")
            report.append(f"{'='*100}")

            report.append("\n当前形成【Vanna挤压 + 吸筹】设置:")

            report.append("\n[第一阶段: IV均值回归 + Vanna效应]")
            report.append("  1. IV处于低位 → 期权便宜 → 交易员买入Call")
            report.append("  2. 价格上涨 → IV下降 → Call Delta增加 (Vanna效应)")
            report.append("  3. 做市商必须买入期货对冲Call → 推升价格")

            report.append("\n[第二阶段: Gamma Squeeze正反馈]")
            if signal.call_oi_above > 500000000:
                report.append(f"  4. 上方有${signal.call_oi_above/1e8:.2f}亿Call持仓")
                report.append("  5. 价格突破行权价 → 做市商Gamma变正 → 必须追涨买入")
                report.append("  6. 形成正反馈循环 → 暴涨")

            if signal.cvd_divergence:
                report.append("\n[吸筹确认]")
                report.append("  - 价格下跌但CVD上升 → 巨鲸在用限价单承接")
                report.append("  - 散户恐慌卖出 → 聪明钱被动吸筹")
                report.append("  - 流动性正在积累, 等待突破")

        report.append("\n" + "="*100)

        return "\n".join(report)


def test_bullish_detection():
    """测试Vanna挤压暴涨检测"""
    print("="*100)
    print("Vanna挤压暴涨检测模型 - 测试")
    print("="*100)

    from options_microstructure import OptionsGreeks
    from orderflow_monitor import OrderFlowMetrics

    # 创建检测器
    detector = VannaSqueezeDetector()

    # 模拟低IV + 吸筹市场数据
    greeks = OptionsGreeks(
        timestamp=datetime.now(),
        spot_price=95000,
        gex=200000000,  # 正GEX (市场稳定)
        vanna=300000000,
        charm=100000000,
        atm_iv=28.5,  # 低IV
        iv_term_structure={'1D': 29.0, '1W': 28.5, '1M': 28.0},
        skew_slope=2.5,
        call_skew=4.2,  # Call Skew上升
        put_skew=6.8,
        max_pain=92000,
        call_oi_above=920000000,  # 上方巨量Call
        put_oi_below=450000000
    )

    orderflow = OrderFlowMetrics(
        timestamp=datetime.now(),
        cvd=3000000,
        cvd_trend=150000,  # CVD上升
        price_trend=-30,  # 价格下跌 (背离)
        bid_quantity_1pct=20000000,
        bid_quantity_2pct=40000000,
        ask_quantity_1pct=18000000,
        ask_quantity_2pct=38000000,
        liquidity_drop_rate=0,
        sell_pressure=2500000,
        buy_pressure=6000000,  # 买压 > 卖压
        vpin=0.18,  # 低VPIN (噪音交易)
        toxic_flow_ratio=0.12  # 低有毒流量
    )

    # IV分位数 20% (便宜)
    iv_percentile = 20.0

    # 执行检测
    signal = detector.detect(
        greeks=greeks,
        orderflow=orderflow,
        iv_percentile=iv_percentile
    )

    # 生成报告
    report = detector.explain(signal)
    print(report)

    return signal


if __name__ == "__main__":
    test_bullish_detection()
