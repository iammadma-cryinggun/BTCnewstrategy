# -*- coding: utf-8 -*-
"""
期权微观结构分析模块
结合 Greeks.live 宏观数据与 CryExc 微观订单流

核心公式:
    LCR (Liquidity Coverage Ratio) = OrderBook_Quantity / Hedging_Need

    当 LCR < 1.0 时, 做市商对冲需求超过市场挂单厚度 → 必然发生滑点
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptionsGreeks:
    """期权希腊字母数据结构"""
    timestamp: datetime
    spot_price: float
    gex: float  # Gamma Exposure (正值=市场稳定, 负值=负Gamma陷阱)
    vanna: float  # Vanna Exposure
    charm: float  # Charm Exposure
    atm_iv: float  # ATM隐含波动率
    iv_term_structure: Dict[str, float]  # {'1D': iv, '1W': iv, ...}
    skew_slope: float  # Skew期限结构斜率
    call_skew: float  # Call端Skew
    put_skew: float  # Put端Skew
    max_pain: float  # 最大痛点
    call_oi_above: float  # 上方Call持仓量
    put_oi_below: float  # 下方Put持仓量


@dataclass
class OrderFlowMetrics:
    """订单流指标数据结构"""
    timestamp: datetime
    cvd: float  # Cumulative Volume Delta
    cvd_trend: float  # CVD趋势
    price_trend: float  # 价格趋势
    bid_quantity_1pct: float  # 1%挂单深度
    bid_quantity_2pct: float  # 2%挂单深度
    ask_quantity_1pct: float
    ask_quantity_2pct: float
    liquidity_drop_rate: float  # 流动性撤单速度
    sell_pressure: float  # 卖压强度
    buy_pressure: float  # 买压强度
    vpin: float  # VPIN (Volume-Synchronized Probability of Informed Trading)
    toxic_flow_ratio: float  # 有毒流量占比


@dataclass
class MarketRegime:
    """市场状态分类"""
    regime: str  # 'NEGATIVE_GAMMA_TRAP', 'VANNA_SQUEEZE', 'NEUTRAL'
    fragility: float  # 脆弱性指数 (0-1)
    catalyst_score: float  # 催化剂得分 (0-1)
    lcr: float  # 流动性覆盖率
    recommendation: str  # 交易建议
    confidence: float  # 置信度


class GreeksDataSource:
    """Greeks.live 数据源"""

    def __init__(self, proxy: Optional[dict] = None):
        self.base_url = "https://api.greeks.live/v1"
        self.proxy = proxy
        self.session = requests.Session()
        if proxy:
            self.session.proxies = proxy

    def get_options_chain(self, symbol: str = "BTC") -> Dict:
        """
        获取期权链数据
        返回完整的Greeks数据
        """
        try:
            # 注意: 这是模拟数据结构, 实际需要根据Greeks.live API文档调整
            url = f"{self.base_url}/options-chain/{symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Greeks API调用失败: {e}")
            return self._get_mock_greeks_data(symbol)

    def get_iv_term_structure(self, symbol: str = "BTC") -> Dict[str, float]:
        """获取IV期限结构"""
        try:
            url = f"{self.base_url}/iv-term-structure/{symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                '1D': data.get('iv_1d', 0),
                '1W': data.get('iv_1w', 0),
                '1M': data.get('iv_1m', 0),
                '2M': data.get('iv_2m', 0),
                '3M': data.get('iv_3m', 0),
                '6M': data.get('iv_6m', 0),
                '1Y': data.get('iv_1y', 0),
            }
        except:
            # 返回默认值
            return {
                '1D': 37.03, '1W': 36.5, '1M': 37.75,
                '2M': 38.2, '3M': 38.8, '6M': 40.1, '1Y': 42.5
            }

    def _get_mock_greeks_data(self, symbol: str) -> Dict:
        """生成模拟Greeks数据 (开发测试用)"""
        return {
            'symbol': symbol,
            'spot': 95000,
            'gex': -500000000,  # 负GEX - 5亿USD的负Gamma暴露
            'vanna': 200000000,
            'charm': -150000000,
            'atm_iv': 37.75,
            'iv_term_structure': {
                '1D': 37.03, '1W': 36.5, '1M': 37.75,
                '2M': 38.2, '3M': 38.8, '6M': 40.1, '1Y': 42.5
            },
            'skew_slope': 8.5,  # 短期Put极其昂贵
            'call_skew': 2.3,
            'put_skew': 10.8,
            'max_pain': 92000,
            'call_oi_above': 850000000,
            'put_oi_below': 620000000
        }


class OptionsAnalyzer:
    """期权数据解析器"""

    def __init__(self, data_source: GreeksDataSource):
        self.data_source = data_source
        self.historical_greeks: List[OptionsGreeks] = []

    def calculate_gex(self, options_chain: Dict) -> float:
        """
        计算Gamma Exposure (GEX)

        GEX = Total_Gamma * Spot_Price^2 * 100

        负GEX (< -1亿): 做市商需要低买高卖对冲 → 市场稳定
        正GEX (> +1亿): 做市商需要追涨杀跌对冲 → 市场不稳定
        """
        # 简化计算
        total_gamma = options_chain.get('total_gamma', 0)
        spot = options_chain.get('spot', 95000)

        gex = total_gamma * (spot ** 2) * 100
        return gex

    def calculate_skew_slope(self, iv_term_structure: Dict[str, float]) -> float:
        """
        计算Skew期限结构斜率

        Slope = (Put_Skew_1W - Call_Skew_1W) - (Put_Skew_3M - Call_Skew_3M)

        高正值 → 短期Put极其昂贵 → 聪明钱在买保护
        """
        iv_1d = iv_term_structure.get('1D', 0)
        iv_1m = iv_term_structure.get('1M', 0)
        iv_3m = iv_term_structure.get('3M', 0)

        # 计算近端与远端的IV差
        near_term_spread = iv_1d - iv_1m
        long_term_spread = iv_1m - iv_3m

        skew_slope = near_term_spread - long_term_spread
        return skew_slope

    def detect_risk_structure(self, iv_term_structure: Dict[str, float]) -> str:
        """
        识别风险结构类型

        返回: 'EVENT_RISK' (事件型风险) 或 'STRUCTURAL_RISK' (结构性风险)
        """
        iv_1d = iv_term_structure.get('1D', 0)
        iv_1m = iv_term_structure.get('1M', 0)

        # 近端IV显著高于中期 → 事件型风险
        if iv_1d > iv_1m * 1.05:
            return 'EVENT_RISK'
        else:
            return 'STRUCTURAL_RISK'

    def analyze_greeks(self, symbol: str = "BTC") -> OptionsGreeks:
        """
        完整的Greeks分析
        返回 OptionsGreeks 数据对象
        """
        raw_data = self.data_source.get_options_chain(symbol)
        iv_structure = self.data_source.get_iv_term_structure(symbol)

        gex = self.calculate_gex(raw_data)
        skew_slope = self.calculate_skew_slope(iv_structure)

        greeks = OptionsGreeks(
            timestamp=datetime.now(),
            spot_price=raw_data.get('spot', 95000),
            gex=gex,
            vanna=raw_data.get('vanna', 0),
            charm=raw_data.get('charm', 0),
            atm_iv=iv_structure.get('1M', 0),
            iv_term_structure=iv_structure,
            skew_slope=skew_slope,
            call_skew=raw_data.get('call_skew', 0),
            put_skew=raw_data.get('put_skew', 0),
            max_pain=raw_data.get('max_pain', 0),
            call_oi_above=raw_data.get('call_oi_above', 0),
            put_oi_below=raw_data.get('put_oi_below', 0)
        )

        self.historical_greeks.append(greeks)

        return greeks

    def calculate_iv_percentile(self, iv: float, lookback_days: int = 30) -> float:
        """
        计算当前IV的历史分位数

        返回 0-100 之间的值, 表示IV在历史中的位置
        """
        if len(self.historical_greeks) < lookback_days:
            return 50.0  # 数据不足, 返回中性值

        recent_ivs = [g.atm_iv for g in self.historical_greeks[-lookback_days:]]

        percentile = (sum(iv <= x for x in recent_ivs) / len(recent_ivs)) * 100
        return percentile


def calculate_lcr(orderbook_quantity: float, hedging_need: float) -> float:
    """
    计算流动性覆盖率 (Liquidity Coverage Ratio)

    LCR = OrderBook_Quantity / Hedging_Need

    LCR < 1.0: 对冲需求 > 挂单厚度 → 必然发生滑点
    LCR > 1.5: 流动性充足
    """
    if hedging_need == 0:
        return float('inf')

    lcr = orderbook_quantity / abs(hedging_need)
    return lcr


if __name__ == "__main__":
    # 测试代码
    print("="*100)
    print("期权微观结构分析模块 - 测试")
    print("="*100)

    data_source = GreeksDataSource()
    analyzer = OptionsAnalyzer(data_source)

    greeks = analyzer.analyze_greeks("BTC")

    print(f"\n[Options Greeks Data]")
    print(f"  Spot Price: ${greeks.spot_price:,.0f}")
    print(f"  GEX: ${greeks.gex:,.0f} ({'Negative Gamma Trap' if greeks.gex < 0 else 'Stable Zone'})")
    print(f"  ATM IV: {greeks.atm_iv:.2f}%")
    print(f"  Skew Slope: {greeks.skew_slope:.2f}")
    print(f"  Risk Structure: {analyzer.detect_risk_structure(greeks.iv_term_structure)}")

    print(f"\n[IV Term Structure]")
    for term, iv in greeks.iv_term_structure.items():
        print(f"  {term}: {iv:.2f}%")

    # Calculate LCR example
    orderbook_qty = 15000000  # 15M USD orderbook
    hedging_need = abs(greeks.gex) * 0.01  # 1% price change hedging need
    lcr = calculate_lcr(orderbook_qty, hedging_need)

    print(f"\n[Liquidity Coverage Ratio]")
    print(f"  Orderbook Depth: ${orderbook_qty:,.0f}")
    print(f"  Hedging Need: ${hedging_need:,.0f}")
    print(f"  LCR: {lcr:.2f} ({'DANGER' if lcr < 1.0 else 'SAFE' if lcr > 1.5 else 'WARNING'})")

    print("\n"+ "="*100)
