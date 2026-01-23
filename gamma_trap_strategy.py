# -*- coding: utf-8 -*-
"""
负Gamma闪崩检测模型 (Negative Gamma Trap)

核心逻辑:
1. 宏观条件: 市场处于负Gamma区域 (GEX < 0)
2. 微观触发: 撤单效应 + 有毒流量
3. 交易指令: 清仓或买入Put

公式:
    LCR = OrderBook_Quantity / |GEX × Price_Change|

    当 LCR < 1.0 时, 必然发生滑点和闪崩
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from options_microstructure import OptionsGreeks, calculate_lcr
from orderflow_monitor import OrderFlowMetrics


@dataclass
class CrashSignal:
    """闪崩信号数据结构"""
    timestamp: datetime
    is_crash_imminent: bool  # 是否即将闪崩
    confidence: float  # 置信度 0-1
    lcr: float  # 流动性覆盖率
    gex: float  # Gamma暴露
    skew_slope: float  # Skew斜率
    liquidity_drop_rate: float  # 流动性撤单速度
    toxic_flow_ratio: float  # 有毒流量占比
    sell_pressure: float  # 卖压
    trigger_reasons: list  # 触发原因列表
    recommendation: str  # 交易建议


class NegativeGammaTrapDetector:
    """负Gamma陷阱检测器"""

    def __init__(self,
                 gex_threshold: float = -100000000,  # GEX < -1亿USD
                 lcr_threshold: float = 1.0,  # LCR < 1.0
                 skew_slope_threshold: float = 5.0,  # Skew斜率 > 5.0
                 vpin_threshold: float = 0.4,  # VPIN > 0.4
                 toxic_flow_threshold: float = 0.3):  # 有毒流量 > 30%

        self.gex_threshold = gex_threshold
        self.lcr_threshold = lcr_threshold
        self.skew_slope_threshold = skew_slope_threshold
        self.vpin_threshold = vpin_threshold
        self.toxic_flow_threshold = toxic_flow_threshold

    def detect(self,
               greeks: OptionsGreeks,
               orderflow: OrderFlowMetrics,
               current_price: float,
               price_change_1pct: float) -> CrashSignal:

        """
        综合检测负Gamma闪崩信号

        参数:
            greeks: 期权Greeks数据
            orderflow: 订单流指标
            current_price: 当前价格
            price_change_1pct: 1%价格变动对应的美元金额

        返回:
            CrashSignal: 闪崩信号对象
        """
        trigger_reasons = []
        confidence_score = 0.0

        # ===== 条件1: 宏观脆弱性 - 负Gamma区域 =====
        gex_condition = greeks.gex < self.gex_threshold
        if gex_condition:
            trigger_reasons.append(f"GEX={greeks.gex/1e8:.2f}亿 (负Gamma陷阱)")
            confidence_score += 0.3

        # ===== 条件2: Skew斜率 - 聪明钱买保护 =====
        skew_condition = greeks.skew_slope > self.skew_slope_threshold
        if skew_condition:
            trigger_reasons.append(f"Skew斜率={greeks.skew_slope:.2f} (短期Put极其昂贵)")
            confidence_score += 0.2

        # ===== 条件3: 流动性覆盖率 =====
        hedging_need = abs(greeks.gex) * 0.01  # 假设1%价格变动
        lcr = calculate_lcr(orderflow.bid_quantity_1pct, hedging_need)
        lcr_condition = lcr < self.lcr_threshold

        if lcr_condition:
            trigger_reasons.append(f"LCR={lcr:.2f} < 1.0 (对冲需求>挂单厚度)")
            confidence_score += 0.25

        # ===== 条件4: 撤单效应 =====
        liquidity_pulling = orderflow.liquidity_drop_rate < -50000  # 每秒撤单>5万USD
        if liquidity_pulling:
            trigger_reasons.append(f"撤单速度={orderflow.liquidity_drop_rate/1000:.1f}K USD/秒 (流动性逃离)")
            confidence_score += 0.15

        # ===== 条件5: 有毒流量 =====
        toxic_condition = orderflow.toxic_flow_ratio > self.toxic_flow_threshold
        if toxic_condition:
            trigger_reasons.append(f"有毒流量={orderflow.toxic_flow_ratio:.1%} (大单主动卖出)")
            confidence_score += 0.1

        # ===== 条件6: VPIN =====
        vpin_condition = orderflow.vpin > self.vpin_threshold
        if vpin_condition:
            trigger_reasons.append(f"VPIN={orderflow.vpin:.3f} (知情交易活跃)")

        # ===== 综合判断 =====
        is_crash_imminent = (
            gex_condition and  # 必须满足负Gamma
            (lcr_condition or toxic_condition)  # 且满足流动性危机或有毒流量
        )

        # 生成交易建议
        if is_crash_imminent:
            if confidence_score >= 0.7:
                recommendation = "【紧急】立刻清仓 + 买入Put"
            elif confidence_score >= 0.5:
                recommendation = "【警告】减仓 + 准备Put"
            else:
                recommendation = "【注意】警惕回调风险"
        else:
            recommendation = "【安全】无闪崩风险"

        signal = CrashSignal(
            timestamp=datetime.now(),
            is_crash_imminent=is_crash_imminent,
            confidence=min(confidence_score, 1.0),
            lcr=lcr,
            gex=greeks.gex,
            skew_slope=greeks.skew_slope,
            liquidity_drop_rate=orderflow.liquidity_drop_rate,
            toxic_flow_ratio=orderflow.toxic_flow_ratio,
            sell_pressure=orderflow.sell_pressure,
            trigger_reasons=trigger_reasons,
            recommendation=recommendation
        )

        return signal

    def explain(self, signal: CrashSignal) -> str:
        """
        生成信号解释报告
        """
        report = []
        report.append("="*100)
        report.append("负Gamma闪崩检测报告")
        report.append("="*100)

        report.append(f"\n【时间】 {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"【状态】 {'闪崩预警' if signal.is_crash_imminent else '安全'}")
        report.append(f"【置信度】 {signal.confidence:.1%}")

        report.append(f"\n{'='*100}")
        report.append("核心指标")
        report.append(f"{'='*100}")

        report.append(f"  GEX: ${signal.gex/1e8:.2f}亿")
        report.append(f"  LCR: {signal.lcr:.2f} ({'危险' if signal.lcr < 1.0 else '安全' if signal.lcr > 1.5 else '警戒'})")
        report.append(f"  Skew斜率: {signal.skew_slope:.2f}")
        report.append(f"  有毒流量: {signal.toxic_flow_ratio:.1%}")

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
        if signal.is_crash_imminent:
            report.append(f"\n{'='*100}")
            report.append("机制解释")
            report.append(f"{'='*100}")

            report.append("\n当前处于【负Gamma陷阱】: ")
            report.append("  1. 做市商持有大量负Gamma头寸")
            report.append("  2. 价格下跌 → 做市商Delta变长 → 必须卖出期货对冲")
            report.append("  3. 做市商的卖出进一步压低价格 → 触发更多对冲卖出")
            report.append("  4. 形成负反馈循环 → 流动性蒸发 → 闪崩")

            if signal.lcr < 1.0:
                report.append("\n流动性覆盖率不足:")
                report.append(f"  - 对冲需求: ${abs(signal.gex)*0.01/1e6:.1f}M (1%价格变动)")
                report.append(f"  - 订单簿深度: ${signal.lcr*abs(signal.gex)*0.01/1e6:.1f}M")
                report.append(f"  - 做市商无法在不造成大幅滑点的情况下完成对冲")

            if signal.toxic_flow_ratio > 0.3:
                report.append("\nToxic Flow Dominant:")
                report.append(f"  - Large sell orders ratio: {signal.toxic_flow_ratio:.1%}")
                report.append(f"  - Smart money exiting or shorting")
                report.append(f"  - Retail orders being consumed, liquidity evaporating")

        report.append("\n" + "="*100)

        return "\n".join(report)


def test_crash_detection():
    """测试负Gamma闪崩检测"""
    print("="*100)
    print("负Gamma闪崩检测模型 - 测试")
    print("="*100)

    from options_microstructure import OptionsGreeks
    from orderflow_monitor import OrderFlowMetrics

    # 创建检测器
    detector = NegativeGammaTrapDetector()

    # 模拟极端市场数据
    greeks = OptionsGreeks(
        timestamp=datetime.now(),
        spot_price=95000,
        gex=-800000000,  # -8亿USD (严重负Gamma)
        vanna=200000000,
        charm=-150000000,
        atm_iv=37.75,
        iv_term_structure={'1D': 45.0, '1W': 38.0, '1M': 37.0},
        skew_slope=8.5,  # 极高斜率
        call_skew=2.3,
        put_skew=15.0,
        max_pain=92000,
        call_oi_above=850000000,
        put_oi_below=620000000
    )

    orderflow = OrderFlowMetrics(
        timestamp=datetime.now(),
        cvd=-5000000,
        cvd_trend=-100000,
        price_trend=-50,
        bid_quantity_1pct=8000000,  # 仅800万USD深度
        bid_quantity_2pct=15000000,
        ask_quantity_1pct=12000000,
        ask_quantity_2pct=25000000,
        liquidity_drop_rate=-150000,  # 快速撤单
        sell_pressure=8000000,  # 高卖压
        buy_pressure=2000000,
        vpin=0.52,  # 有毒流量
        toxic_flow_ratio=0.45  # 45%有毒流量
    )

    # 执行检测
    signal = detector.detect(
        greeks=greeks,
        orderflow=orderflow,
        current_price=95000,
        price_change_1pct=950
    )

    # 生成报告
    report = detector.explain(signal)
    print(report)

    return signal


if __name__ == "__main__":
    test_crash_detection()
