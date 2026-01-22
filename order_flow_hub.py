# -*- coding: utf-8 -*-
"""
订单流数据获取模块
从Binance Futures获取实时订单流数据
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OrderFlowHub:
    """订单流数据中心"""

    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.symbol = "BTCUSDT"
        self.session = requests.Session()

        # 数据缓存
        self.orderbook_cache = None
        self.orderbook_timestamp = None
        self.cv_data = []  # CVD累积数据

        logger.info("订单流数据中心初始化完成")

    def get_orderbook(self, depth: int = 20) -> Optional[Dict]:
        """
        获取订单簿深度数据

        Args:
            depth: 深度档位（5, 10, 20等）

        Returns:
            {
                'bids': [[price, qty], ...],  # 买盘
                'asks': [[price, qty], ...],  # 卖盘
                'timestamp': datetime
            }
        """
        try:
            url = f"{self.base_url}/fapi/v1/depth"
            params = {
                'symbol': self.symbol,
                'limit': depth
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 解析数据
            bids = [[float(p), float(q)] for p, q in data['bids']]
            asks = [[float(p), float(q)] for p, q in data['asks']]

            result = {
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }

            # 缓存
            self.orderbook_cache = result
            self.orderbook_timestamp = datetime.now()

            logger.info(f"获取订单簿数据成功: 买盘{len(bids)}档, 卖盘{len(asks)}档")
            return result

        except Exception as e:
            logger.error(f"获取订单簿数据失败: {e}")
            return None

    def get_recent_trades(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取最近成交数据

        Args:
            limit: 获取数量（最大1000）

        Returns:
            DataFrame with columns:
            - price: 成交价格
            - qty: 成交数量
            - time: 成交时间
            - is_buyer_maker: 是否买方挂单（False=主动买入，True=主动卖出）
        """
        try:
            url = f"{self.base_url}/fapi/v1/aggTrades"
            params = {
                'symbol': self.symbol,
                'limit': limit
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 转换为DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                return None

            df['price'] = df['p'].astype(float)
            df['qty'] = df['q'].astype(float)
            df['time'] = pd.to_datetime(df['T'], unit='ms')
            df['is_buyer_maker'] = df['m']  # True=主动卖出, False=主动买入

            # 计算成交额
            df['quote_qty'] = df['price'] * df['qty']

            logger.info(f"获取最近成交数据: {len(df)}笔")
            return df[['price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']]

        except Exception as e:
            logger.error(f"获取成交数据失败: {e}")
            return None

    def get_liquidations(self, limit: int = 100) -> Optional[List[Dict]]:
        """
        获取清算事件

        Args:
            limit: 获取数量（最大100）

        Returns:
            [
                {
                    'side': 'SELL' (多头被清算) or 'BUY' (空头被清算),
                    'price': 清算价格,
                    'qty': 清算数量,
                    'time': 清算时间
                },
                ...
            ]
        """
        try:
            url = f"{self.base_url}/fapi/v1/allForceOrders"
            params = {
                'symbol': self.symbol,
                'limit': limit
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            liquidations = []
            for item in data:
                liquidations.append({
                    'side': item['side'],  # SELL=多被清算, BUY=空被清算
                    'price': float(item['price']),
                    'qty': float(item['amount']),
                    'time': pd.to_datetime(item['time'], unit='ms')
                })

            logger.info(f"获取清算事件: {len(liquidations)}条")
            return liquidations

        except Exception as e:
            logger.error(f"获取清算事件失败: {e}")
            return None

    def calculate_cvd(self, trades_df: pd.DataFrame, window: str = '5min') -> Dict:
        """
        计算CVD（Cumulative Volume Delta）

        CVD = (主动买入量 - 主动卖出量) 的累积值

        Args:
            trades_df: 成交数据DataFrame
            window: 时间窗口（'1min', '5min', '15min'等）

        Returns:
            {
                'current_cvd': 当前CVD值,
                'cvd_change': CVD变化量,
                'buy_volume': 主动买入量,
                'sell_volume': 主动卖出量,
                'buy_ratio': 买入占比,
                'trend': 'bullish' | 'bearish' | 'neutral'
            }
        """
        if trades_df is None or trades_df.empty:
            return {}

        try:
            # 按时间窗口聚合
            trades_df.set_index('time', inplace=True)

            # 主动买入 = is_buyer_maker == False
            # 主动卖出 = is_buyer_maker == True
            buy_trades = trades_df[~trades_df['is_buyer_maker']]['quote_qty'].resample(window).sum()
            sell_trades = trades_df[trades_df['is_buyer_maker']]['quote_qty'].resample(window).sum()

            # 计算Delta
            delta = buy_trades - sell_trades

            # 累积CVD
            cvd = delta.cumsum()

            if cvd.empty:
                return {}

            current_cvd = cvd.iloc[-1]
            cvd_change = delta.iloc[-1] if len(delta) > 0 else 0
            buy_volume = buy_trades.iloc[-1] if len(buy_trades) > 0 else 0
            sell_volume = sell_trades.iloc[-1] if len(sell_trades) > 0 else 0
            total_volume = buy_volume + sell_volume
            buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5

            # 判断趋势
            if cvd_change > 0 and buy_ratio > 0.6:
                trend = 'bullish'
            elif cvd_change < 0 and buy_ratio < 0.4:
                trend = 'bearish'
            else:
                trend = 'neutral'

            result = {
                'current_cvd': current_cvd,
                'cvd_change': cvd_change,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'total_volume': total_volume,
                'buy_ratio': buy_ratio,
                'trend': trend
            }

            logger.info(f"CVD: {current_cvd:,.0f} ({cvd_change:+,.0f}), 买入占比: {buy_ratio:.1%}, 趋势: {trend}")
            return result

        except Exception as e:
            logger.error(f"计算CVD失败: {e}")
            return {}

    def identify_order_walls(self, orderbook: Dict, threshold_pct: float = 0.5) -> Dict:
        """
        从订单簿中识别真正的订单墙（大额挂单墙）

        Args:
            orderbook: 订单簿数据
            threshold_pct: 门槛百分比（占总成交量的百分比）

        Returns:
            {
                'support_walls': [{'price': 价格, 'qty': 数量, 'distance': 距离当前价%}, ...],
                'resistance_walls': [{'price': 价格, 'qty': 数量, 'distance': 距离当前价%}, ...],
                'current_price': 当前价格
            }
        """
        if orderbook is None:
            return {}

        try:
            bids = orderbook['bids']  # 买盘（支撑）
            asks = orderbook['asks']  # 卖盘（阻力）

            # 计算当前中间价
            if bids and asks:
                current_price = (bids[0][0] + asks[0][0]) / 2
            else:
                return {}

            # 计算平均挂单量
            all_qty = [qty for _, qty in bids + asks]
            avg_qty = np.mean(all_qty) if all_qty else 0
            threshold = avg_qty * (1 + threshold_pct)  # 门槛 = 平均值 * (1 + 50%)

            # 识别支撑墙（买盘）
            support_walls = []
            for price, qty in bids:
                if qty >= threshold:
                    distance = (price - current_price) / current_price
                    # 只关注当前价下方的支撑
                    if distance < 0:
                        support_walls.append({
                            'price': price,
                            'qty': qty,
                            'distance': distance
                        })

            # 识别阻力墙（卖盘）
            resistance_walls = []
            for price, qty in asks:
                if qty >= threshold:
                    distance = (price - current_price) / current_price
                    # 只关注当前价上方的阻力
                    if distance > 0:
                        resistance_walls.append({
                            'price': price,
                            'qty': qty,
                            'distance': distance
                        })

            # 按距离排序
            support_walls.sort(key=lambda x: abs(x['distance']))
            resistance_walls.sort(key=lambda x: abs(x['distance']))

            # 取最近的3个墙
            support_walls = support_walls[:3]
            resistance_walls = resistance_walls[:3]

            result = {
                'support_walls': support_walls,
                'resistance_walls': resistance_walls,
                'current_price': current_price
            }

            logger.info(f"识别订单墙: 支撑{len(support_walls)}个, 阻力{len(resistance_walls)}个")
            return result

        except Exception as e:
            logger.error(f"识别订单墙失败: {e}")
            return {}

    def detect_whale_trades(self, trades_df: pd.DataFrame, threshold_usd: float = 1000000) -> List[Dict]:
        """
        检测大单交易（鲸鱼交易）

        Args:
            trades_df: 成交数据
            threshold_usd: 门槛金额（USD）

        Returns:
            [
                {
                    'price': 价格,
                    'qty': 数量,
                    'value': 成交额(USD),
                    'side': 'BUY' | 'SELL',
                    'time': 时间
                },
                ...
            ]
        """
        if trades_df is None or trades_df.empty:
            return []

        try:
            # 筛选大额交易
            large_trades = trades_df[trades_df['quote_qty'] >= threshold_usd].copy()

            if large_trades.empty:
                return []

            # 添加方向
            large_trades['side'] = large_trades['is_buyer_maker'].apply(
                lambda x: 'SELL' if x else 'BUY'
            )

            # 格式化输出
            whale_trades = []
            for _, row in large_trades.iterrows():
                whale_trades.append({
                    'price': row['price'],
                    'qty': row['qty'],
                    'value': row['quote_qty'],
                    'side': row['side'],
                    'time': row['time']
                })

            logger.info(f"检测到鲸鱼交易: {len(whale_trades)}笔 (>${threshold_usd:,.0f})")
            return whale_trades

        except Exception as e:
            logger.error(f"检测大单失败: {e}")
            return []

    def get_order_flow_summary(self) -> Dict:
        """
        获取订单流综合分析

        Returns:
            {
                'orderbook': 订单簿数据,
                'cvd': CVD分析,
                'order_walls': 订单墙,
                'whale_trades': 鲸鱼交易,
                'liquidations': 清算事件
            }
        """
        summary = {}

        # 1. 获取订单簿
        logger.info("正在获取订单簿数据...")
        orderbook = self.get_orderbook(depth=20)
        summary['orderbook'] = orderbook

        # 2. 获取成交数据
        logger.info("正在获取成交数据...")
        trades_df = self.get_recent_trades(limit=1000)

        # 3. 计算CVD
        if trades_df is not None:
            cvd = self.calculate_cvd(trades_df, window='5min')
            summary['cvd'] = cvd

            # 4. 检测鲸鱼交易
            whale_trades = self.detect_whale_trades(trades_df, threshold_usd=1000000)
            summary['whale_trades'] = whale_trades

        # 5. 识别订单墙
        if orderbook:
            walls = self.identify_order_walls(orderbook, threshold_pct=0.5)
            summary['order_walls'] = walls

        # 6. 获取清算事件
        logger.info("正在获取清算事件...")
        liquidations = self.get_liquidations(limit=100)
        summary['liquidations'] = liquidations

        return summary


# 测试代码
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    hub = OrderFlowHub()

    print("=" * 70)
    print("订单流数据测试")
    print("=" * 70)

    # 获取综合数据
    summary = hub.get_order_flow_summary()

    print("\n" + "=" * 70)
    print("订单流分析结果")
    print("=" * 70)

    # 当前价格
    if summary.get('order_walls'):
        current_price = summary['order_walls']['current_price']
        print(f"\n当前价格: ${current_price:,.0f}")

    # 订单墙
    if summary.get('order_walls'):
        walls = summary['order_walls']
        print(f"\n📐 订单墙（订单簿）:")

        if walls['support_walls']:
            print(f"  支撑墙:")
            for wall in walls['support_walls'][:3]:
                print(f"    ${wall['price']:,.0f} ({wall['distance']:+.2%}) - {wall['qty']:,.2f} BTC")

        if walls['resistance_walls']:
            print(f"  阻力墙:")
            for wall in walls['resistance_walls'][:3]:
                print(f"    ${wall['price']:,.0f} ({wall['distance']:+.2%}) - {wall['qty']:,.2f} BTC")

    # CVD
    if summary.get('cvd'):
        cvd = summary['cvd']
        print(f"\n📊 CVD分析:")
        print(f"  CVD值: {cvd['current_cvd']:,.0f} USD")
        print(f"  变化: {cvd['cvd_change']:+,.0f} USD")
        print(f"  买入占比: {cvd['buy_ratio']:.1%}")
        print(f"  趋势: {cvd['trend']}")

    # 鲸鱼交易
    if summary.get('whale_trades'):
        whale = summary['whale_trades']
        print(f"\n🐋 鲸鱼交易 (${len(whale)}笔):")
        for trade in whale[:5]:
            print(f"  {trade['side']} ${trade['value']:,.0f} @ ${trade['price']:,.0f}")

    # 清算事件
    if summary.get('liquidations'):
        liq = summary['liquidations']
        print(f"\n💥 清算事件 ({len(liq)}条):")
        long_liq = [x for x in liq if x['side'] == 'SELL']
        short_liq = [x for x in liq if x['side'] == 'BUY']
        print(f"  多头清算: {len(long_liq)}笔")
        print(f"  空头清算: {len(short_liq)}笔")

    print("\n" + "=" * 70)
