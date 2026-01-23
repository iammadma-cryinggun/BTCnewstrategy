# -*- coding: utf-8 -*-
"""
订单流监控模块 (CryExc Backend)

实时监控:
1. 订单簿深度变化
2. CVD (Cumulative Volume Delta)
3. VPIN (Volume-Synchronized Probability of Informed Trading)
4. 有毒流量 (Toxic Flow)
5. 撤单效应 (Liquidity Pulling)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """单笔成交数据"""
    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool  # True=卖方主动(吃单), False=买方主动(挂单成交)
    side: str  # 'buy' or 'sell'


@dataclass
class OrderBookLevel:
    """订单簿档位"""
    price: float
    quantity: float


@dataclass
class OrderBookSnapshot:
    """Orderbook snapshot"""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    bid_quantity_1pct: float  # Bid quantity within 1% price range
    bid_quantity_2pct: float  # Bid quantity within 2% price range
    ask_quantity_1pct: float
    ask_quantity_2pct: float


@dataclass
class OrderFlowMetrics:
    """Order flow metrics data structure"""
    timestamp: datetime
    cvd: float  # Cumulative Volume Delta
    cvd_trend: float  # CVD trend
    price_trend: float  # Price trend
    bid_quantity_1pct: float
    bid_quantity_2pct: float
    ask_quantity_1pct: float
    ask_quantity_2pct: float
    liquidity_drop_rate: float  # Liquidity pulling rate
    sell_pressure: float
    buy_pressure: float
    vpin: float  # Volume-Synchronized PIN
    toxic_flow_ratio: float  # Toxic flow ratio


class OrderFlowMonitor:
    """订单流实时监控器"""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.trades_buffer: deque[Trade] = deque(maxlen=1000)
        self.orderbook_history: List[OrderBookSnapshot] = []
        self.cvd_history: List[Tuple[datetime, float]] = []

        # VPIN计算参数
        self.vpin_window = 100  # 交易笔数窗口
        self.buckets = []  # VPIN分桶
        self.current_bucket_vol = 0
        self.bucket_buying_vol = 0
        self.bucket_selling_vol = 0
        self.target_bucket_volume = 1000000  # 每桶目标成交量

    def on_trade(self, trade: Trade):
        """
        处理新成交数据
        """
        self.trades_buffer.append(trade)

        # 更新CVD
        cvd_delta = trade.quantity if not trade.is_buyer_maker else -trade.quantity
        latest_cvd = self.cvds[-1][1] if self.cvds else 0
        new_cvd = latest_cvd + cvd_delta
        self.cvds.append((trade.timestamp, new_cvd))

        # 更新VPIN分桶
        self._update_vpin_buckets(trade)

    def on_orderbook_update(self, snapshot: OrderBookSnapshot):
        """
        处理订单簿更新
        """
        self.orderbook_history.append(snapshot)

        # 保持历史记录在合理范围
        if len(self.orderbook_history) > 100:
            self.orderbook_history.pop(0)

    def calculate_cvd(self, window_minutes: int = 60) -> float:
        """
        计算CVD (Cumulative Volume Delta)

        CVD = 主动买入量 - 主动卖出量

        正值: 买方强势
        负值: 卖方强势
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        cvd = 0
        for trade in self.trades_buffer:
            if trade.timestamp < cutoff_time:
                continue

            # 主动买入(挂单成交) +, 主动卖出(吃单) -
            cvd += trade.quantity if not trade.is_buyer_maker else -trade.quantity

        return cvd

    def calculate_cvds(self, window_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """
        返回CVD时间序列(用于趋势计算)
        """
        return self.cvds

    def calculate_cvd_trend(self, window_minutes: int = 15) -> float:
        """
        计算CVD趋势

        返回线性回归斜率
        正值: 买方力量增强
        负值: 卖方力量增强
        """
        cvds = self.calculate_cvds(window_minutes)

        if len(cvds) < 10:
            return 0.0

        # 提取时间和CVD值
        times = [(t - cvds[0][0]).total_seconds() for t, _ in cvds]
        values = [v for _, v in cvds]

        # 线性回归
        times_array = np.array(times)
        values_array = np.array(values)

        if np.std(times_array) == 0:
            return 0.0

        slope = np.cov(times_array, values_array)[0, 1] / np.var(times_array)

        return slope

    def calculate_price_trend(self, window_minutes: int = 15) -> float:
        """
        计算价格趋势

        返回线性回归斜率
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        prices = []
        times = []

        for trade in self.trades_buffer:
            if trade.timestamp < cutoff_time:
                continue

            prices.append(trade.price)
            times.append(trade.timestamp)

        if len(prices) < 10:
            return 0.0

        # 线性回归
        times_array = np.array([(t - times[0]).total_seconds() for t in times])
        prices_array = np.array(prices)

        if np.std(times_array) == 0:
            return 0.0

        slope = np.cov(times_array, prices_array)[0, 1] / np.var(times_array)

        return slope

    def _update_vpin_buckets(self, trade: Trade):
        """
        更新VPIN分桶计算

        VPIN = |买量 - 卖量| / 总量

        高VPIN (>0.4) → 有毒流量(知情交易活跃)
        低VPIN (<0.2) → 噪音交易(散户博弈)
        """
        # 更新当前桶
        self.current_bucket_vol += trade.quantity

        if not trade.is_buyer_maker:
            # 主动买入(挂单成交)
            self.bucket_buying_vol += trade.quantity
        else:
            # 主动卖出(吃单)
            self.bucket_selling_vol += trade.quantity

        # 检查是否完成一桶
        if self.current_bucket_vol >= self.target_bucket_volume:
            # 计算VPIN
            vpin = abs(self.bucket_buying_vol - self.bucket_selling_vol) / self.current_bucket_vol

            self.buckets.append(vpin)

            # 重置
            self.current_bucket_vol = 0
            self.bucket_buying_vol = 0
            self.bucket_selling_vol = 0

            # 保持窗口
            if len(self.buckets) > self.vpin_window:
                self.buckets.pop(0)

    def calculate_vpin(self) -> float:
        """
        计算当前VPIN值
        """
        if len(self.buckets) < 10:
            return 0.0

        return np.mean(self.buckets)

    def detect_liquidity_pulling(self) -> Tuple[bool, float]:
        """
        检测撤单效应

        流动性撤单 = bidQuantity下降速度 > 价格下降速度

        返回: (是否发生撤单, 撤单速度)
        """
        if len(self.orderbook_history) < 2:
            return False, 0.0

        latest = self.orderbook_history[-1]
        previous = self.orderbook_history[-2]

        # bidQuantity变化
        bid_qty_change = latest.bid_quantity_1pct - previous.bid_quantity_1pct

        # 时间间隔
        time_delta = (latest.timestamp - previous.timestamp).total_seconds()

        if time_delta == 0:
            return False, 0.0

        # 撤单速度 (USD/秒)
        pulling_rate = bid_qty_change / time_delta

        # 负值表示撤单
        is_pulling = pulling_rate < -50000  # 阈值: 每秒撤单超过5万USD

        return is_pulling, pulling_rate

    def calculate_toxic_flow_ratio(self, window_seconds: int = 60) -> float:
        """
        计算有毒流量占比

        有毒流量 = 主动卖出大单 (isBuyerMaker=True 且 quantity > 平均值*2)

        返回 0-1 之间的值
        """
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)

        toxic_trades = 0
        total_trades = 0

        # 计算平均成交量
        quantities = [t.quantity for t in self.trades_buffer if t.timestamp > cutoff_time]
        if len(quantities) == 0:
            return 0.0

        avg_qty = np.mean(quantities)
        large_threshold = avg_qty * 2

        for trade in self.trades_buffer:
            if trade.timestamp < cutoff_time:
                continue

            total_trades += 1

            # 大单主动卖出 = 有毒流量
            if trade.is_buyer_maker and trade.quantity > large_threshold:
                toxic_trades += 1

        if total_trades == 0:
            return 0.0

        return toxic_trades / total_trades

    def calculate_sell_pressure(self, window_seconds: int = 60) -> float:
        """
        计算卖压强度

        卖压 = Σ(主动卖出的量)
        """
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)

        sell_pressure = 0
        for trade in self.trades_buffer:
            if trade.timestamp < cutoff_time:
                continue

            if trade.is_buyer_maker:  # 主动卖出
                sell_pressure += trade.quantity

        return sell_pressure

    def calculate_buy_pressure(self, window_seconds: int = 60) -> float:
        """
        计算买压强度

        买压 = Σ(主动买入的量)
        """
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)

        buy_pressure = 0
        for trade in self.trades_buffer:
            if trade.timestamp < cutoff_time:
                continue

            if not trade.is_buyer_maker:  # 主动买入
                buy_pressure += trade.quantity

        return buy_pressure

    def detect_divergence(self, window_minutes: int = 15) -> Tuple[bool, str]:
        """
        检测背离

        价格下跌 + CVD上升 = 吸筹背离 (Bullish Divergence)
        价格上涨 + CVD下降 = 派发背离 (Bearish Divergence)

        返回: (是否背离, 背离类型)
        """
        price_trend = self.calculate_price_trend(window_minutes)
        cvd_trend = self.calculate_cvd_trend(window_minutes)

        # 价格下跌 + CVD上升 = 吸筹
        if price_trend < 0 and cvd_trend > 0:
            return True, "ABSORPTION"  # 吸筹背离

        # 价格上涨 + CVD下降 = 派发
        elif price_trend > 0 and cvd_trend < 0:
            return True, "DISTRIBUTION"  # 派发背离

        else:
            return False, "NONE"

    def get_metrics(self) -> 'OrderFlowMetrics':
        """
        获取当前所有订单流指标
        """
        # 获取最新订单簿快照
        latest_snapshot = self.orderbook_history[-1] if self.orderbook_history else None

        # 检测撤单
        is_pulling, pulling_rate = self.detect_liquidity_pulling()

        metrics = OrderFlowMetrics(
            timestamp=datetime.now(),
            cvd=self.calculate_cvd(),
            cvd_trend=self.calculate_cvd_trend(),
            price_trend=self.calculate_price_trend(),
            bid_quantity_1pct=latest_snapshot.bid_quantity_1pct if latest_snapshot else 0,
            bid_quantity_2pct=latest_snapshot.bid_quantity_2pct if latest_snapshot else 0,
            ask_quantity_1pct=latest_snapshot.ask_quantity_1pct if latest_snapshot else 0,
            ask_quantity_2pct=latest_snapshot.ask_quantity_2pct if latest_snapshot else 0,
            liquidity_drop_rate=pulling_rate if is_pulling else 0,
            sell_pressure=self.calculate_sell_pressure(),
            buy_pressure=self.calculate_buy_pressure(),
            vpin=self.calculate_vpin(),
            toxic_flow_ratio=self.calculate_toxic_flow_ratio()
        )

        return metrics

    @property
    def cvds(self) -> List[Tuple[datetime, float]]:
        """CVD time series"""
        return self.cvd_history


if __name__ == "__main__":
    # 测试代码
    print("="*100)
    print("订单流监控模块 - 测试")
    print("="*100)

    monitor = OrderFlowMonitor("BTCUSDT")

    # 模拟一些交易数据
    now = datetime.now()
    for i in range(100):
        trade = Trade(
            timestamp=now + timedelta(seconds=i),
            price=95000 + np.random.randn() * 100,
            quantity=np.random.uniform(0.1, 2.0),
            is_buyer_maker=np.random.choice([True, False]),
            side='buy' if np.random.random() > 0.5 else 'sell'
        )
        monitor.on_trade(trade)

    # 模拟订单簿快照
    snapshot = OrderBookSnapshot(
        timestamp=now + timedelta(seconds=100),
        bids=[],
        asks=[],
        bid_quantity_1pct=15000000,
        bid_quantity_2pct=30000000,
        ask_quantity_1pct=15000000,
        ask_quantity_2pct=30000000
    )
    monitor.on_orderbook_update(snapshot)

    metrics = monitor.get_metrics()

    print(f"\n[订单流指标]")
    print(f"  CVD: {metrics.cvd:,.2f}")
    print(f"  CVD趋势: {metrics.cvd_trend:.2f}")
    print(f"  价格趋势: {metrics.price_trend:.4f}")
    print(f"  买压: {metrics.buy_pressure:,.2f}")
    print(f"  卖压: {metrics.sell_pressure:,.2f}")
    print(f"  VPIN: {metrics.vpin:.3f} ({'有毒流量' if metrics.vpin > 0.4 else '噪音交易' if metrics.vpin < 0.2 else '混合'})")
    print(f"  有毒流量占比: {metrics.toxic_flow_ratio:.1%}")

    # 检测背离
    is_diverging, div_type = monitor.detect_divergence()
    print(f"\n[背离检测]")
    print(f"  背离: {'是' if is_diverging else '否'}")
    print(f"  类型: {div_type}")

    print("\n" + "="*100)
