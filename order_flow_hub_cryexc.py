# -*- coding: utf-8 -*-
"""
订单流数据获取模块 - CryExc WebSocket版本
通过CryExc后端获取实时订单流数据（替代REST API轮询）
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import websockets
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CryExcOrderFlowClient:
    """CryExc WebSocket客户端"""

    def __init__(self, uri: str = "ws://127.0.0.1:8000/ws"):
        self.uri = uri
        self.websocket = None
        self.connected = False
        self.subscriptions = {}

        # 数据缓存
        self.trade_cache = []
        self.orderbook_cache = None
        self.cvd_cache = []
        self.liquidation_cache = []

        logger.info(f"CryExc客户端初始化: {uri}")

    async def connect(self):
        """连接到CryExc后端"""
        try:
            logger.info(f"正在连接到CryExc后端: {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            logger.info("✓ CryExc连接成功")

            # 启动消息监听
            asyncio.create_task(self._listen_messages())

        except Exception as e:
            logger.error(f"CryExc连接失败: {e}")
            self.connected = False

    async def _listen_messages(self):
        """监听WebSocket消息"""
        try:
            async for message in self.websocket:
                data = json.loads(message)

                msg_type = data.get('type')

                if msg_type == 'trade':
                    self.trade_cache.append(data['data'])
                    # 限制缓存大小
                    if len(self.trade_cache) > 10000:
                        self.trade_cache = self.trade_cache[-5000:]

                elif msg_type == 'orderbook':
                    self.orderbook_cache = data['data']

                elif msg_type == 'cvd_historical':
                    self.cvd_cache.append(data['data'])

                elif msg_type == 'cvd':
                    self.cvd_cache.append(data['data'])

                elif msg_type == 'liquidation':
                    self.liquidation_cache.append(data['data'])

                elif msg_type == 'error':
                    logger.error(f"CryExc错误: {data.get('error')}")

                elif msg_type in ['subscribed', 'unsubscribed']:
                    logger.info(f"CryExc: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("CryExc连接已关闭")
            self.connected = False
        except Exception as e:
            logger.error(f"CryExc消息监听错误: {e}")

    async def subscribe_trade(self, symbol: str = "BTCUSDT", min_notional: float = 50000):
        """订阅成交数据"""
        message = {
            "type": "stream_subscribe",
            "stream": "trade",
            "config": {
                "symbol": symbol,
                "minNotional": min_notional
            }
        }

        await self.websocket.send(json.dumps(message))
        self.subscriptions['trade'] = True
        logger.info(f"✓ 已订阅trade流: {symbol}, 最小{min_notional}")

    async def subscribe_orderbook(self, symbol: str = "BTCUSDT", depth: int = 20):
        """订阅订单簿"""
        message = {
            "type": "stream_subscribe",
            "stream": "orderbook",
            "config": {
                "symbol": symbol,
                "depth": depth
            }
        }

        await self.websocket.send(json.dumps(message))
        self.subscriptions['orderbook'] = True
        logger.info(f"✓ 已订阅orderbook流: {symbol}, 深度{depth}")

    async def subscribe_cvd(self, symbol: str = "BTCUSDT", window: str = "5min"):
        """订阅CVD数据"""
        message = {
            "type": "stream_subscribe",
            "stream": "cvd",
            "config": {
                "symbol": symbol,
                "window": window
            }
        }

        await self.websocket.send(json.dumps(message))
        self.subscriptions['cvd'] = True
        logger.info(f"✓ 已订阅cvd流: {symbol}, 窗口{window}")

    async def subscribe_liquidation(self, symbol: str = "BTCUSDT"):
        """订阅清算数据"""
        message = {
            "type": "stream_subscribe",
            "stream": "liquidation",
            "config": {
                "symbol": symbol
            }
        }

        await self.websocket.send(json.dumps(message))
        self.subscriptions['liquidation'] = True
        logger.info(f"✓ 已订阅liquidation流: {symbol}")

    def get_recent_trades(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        从缓存获取最近成交数据

        Returns:
            DataFrame with columns:
            - price: 成交价格
            - qty: 成交数量
            - quote_qty: 成交额(USD)
            - time: 成交时间
            - is_buyer_maker: 是否买方挂单
        """
        if not self.trade_cache:
            return None

        # 取最近的N笔
        recent_trades = self.trade_cache[-limit:]

        # 转换为DataFrame
        data = []
        for trade in recent_trades:
            data.append({
                'price': trade['price'],
                'qty': trade['qty'],
                'quote_qty': trade['quoteQty'],
                'time': pd.to_datetime(trade['timestamp'], unit='ms'),
                'is_buyer_maker': trade['isBuyerMaker']
            })

        df = pd.DataFrame(data)

        logger.info(f"从CryExc获取成交数据: {len(df)}笔")
        return df

    def get_orderbook(self) -> Optional[Dict]:
        """获取缓存的订单簿数据"""
        if not self.orderbook_cache:
            return None

        return {
            'bids': self.orderbook_cache['bids'],
            'asks': self.orderbook_cache['asks'],
            'timestamp': pd.to_datetime(self.orderbook_cache['timestamp'], unit='ms')
        }

    def get_cvd_data(self) -> Optional[pd.DataFrame]:
        """获取缓存的CVD数据"""
        if not self.cvd_cache:
            return None

        # 转换为DataFrame
        data = []
        for cvd_point in self.cvd_cache[-100:]:  # 最近100个点
            data.append({
                'timestamp': pd.to_datetime(cvd_point['timestamp'], unit='ms'),
                'cvd': cvd_point['cvd'],
                'delta': cvd_point['delta'],
                'buy_volume': cvd_point['buyVolume'],
                'sell_volume': cvd_point['sellVolume']
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        logger.info(f"从CryExc获取CVD数据: {len(df)}个点")
        return df

    def get_liquidations(self) -> Optional[List[Dict]]:
        """获取缓存的清算数据"""
        if not self.liquidation_cache:
            return []

        liquidations = []
        for liq in self.liquidation_cache[-50:]:  # 最近50条
            liquidations.append({
                'side': liq['side'],
                'price': liq['price'],
                'qty': liq['qty'],
                'time': pd.to_datetime(liq['time'], unit='ms')
            })

        return liquidations

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("CryExc连接已关闭")


class OrderFlowHubCryExc:
    """订单流数据中心（CryExc版本）"""

    def __init__(self, use_cryexc: bool = True):
        self.use_cryexc = use_cryexc

        if use_cryexc:
            self.cryexc_client = CryExcOrderFlowClient()
            # 同步接口（为了兼容现有代码）
            self.client = None
        else:
            # 回退到REST API
            from order_flow_hub import OrderFlowHub
            self.client = OrderFlowHub()
            self.cryexc_client = None

        logger.info(f"订单流数据中心初始化完成 (CryExc: {use_cryexc})")

    async def initialize_async(self):
        """异步初始化（如果使用CryExc）"""
        if self.use_cryexc:
            await self.cryexc_client.connect()

            # 检查连接是否成功
            if not self.cryexc_client.connected:
                logger.error("CryExc连接失败，无法订阅数据流")
                return

            # 订阅所有数据流
            await self.cryexc_client.subscribe_trade()
            await self.cryexc_client.subscribe_orderbook()
            await self.cryexc_client.subscribe_cvd()
            await self.cryexc_client.subscribe_liquidation()

            # 等待数据填充
            await asyncio.sleep(2)

    def get_orderbook(self, depth: int = 20) -> Optional[Dict]:
        """获取订单簿数据"""
        if self.use_cryexc:
            return self.cryexc_client.get_orderbook()
        else:
            return self.client.get_orderbook(depth)

    def get_recent_trades(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """获取成交数据"""
        if self.use_cryexc:
            return self.cryexc_client.get_recent_trades(limit)
        else:
            return self.client.get_recent_trades(limit)

    def get_liquidations(self, limit: int = 100) -> Optional[List[Dict]]:
        """获取清算数据"""
        if self.use_cryexc:
            return self.cryexc_client.get_liquidations()
        else:
            return self.client.get_liquidations(limit)

    def calculate_cvd(self, trades_df: pd.DataFrame, window: str = '5min') -> Dict:
        """计算CVD"""
        # 如果使用CryExc，直接从CryExc获取CVD数据
        if self.use_cryexc:
            cvd_df = self.cryexc_client.get_cvd_data()

            if cvd_df is None or cvd_df.empty:
                return {}

            latest = cvd_df.iloc[-1]

            result = {
                'current_cvd': latest['cvd'],
                'cvd_change': latest['delta'],
                'buy_volume': latest['buy_volume'],
                'sell_volume': latest['sell_volume'],
                'buy_ratio': latest['buy_volume'] / (latest['buy_volume'] + latest['sell_volume']),
                'trend': 'bullish' if latest['delta'] > 0 else 'bearish' if latest['delta'] < 0 else 'neutral'
            }

            logger.info(f"从CryExc获取CVD: {result}")
            return result
        else:
            # 使用原有的计算方法
            return self.client.calculate_cvd(trades_df, window)

    def identify_order_walls(self, orderbook: Dict, threshold_pct: float = 0.5) -> Dict:
        """识别订单墙"""
        if self.use_cryexc:
            # CryExc的订单簿格式相同，直接使用原逻辑
            pass

        # 复用原有逻辑
        from order_flow_hub import OrderFlowHub
        temp_hub = OrderFlowHub()
        return temp_hub.identify_order_walls(orderbook, threshold_pct)

    def detect_whale_trades(self, trades_df: pd.DataFrame, threshold_usd: float = 1000000) -> List[Dict]:
        """检测大单交易"""
        if trades_df is None or trades_df.empty:
            return []

        whale_trades = trades_df[trades_df['quote_qty'] >= threshold_usd].copy()

        if whale_trades.empty:
            return []

        whale_trades['side'] = whale_trades['is_buyer_maker'].apply(
            lambda x: 'SELL' if x else 'BUY'
        )

        result = []
        for _, row in whale_trades.iterrows():
            result.append({
                'price': row['price'],
                'qty': row['qty'],
                'value': row['quote_qty'],
                'side': row['side'],
                'time': row['time']
            })

        logger.info(f"检测到鲸鱼交易: {len(result)}笔 (>${threshold_usd:,.0f})")
        return result

    async def get_order_flow_summary_async(self) -> Dict:
        """异步获取订单流综合分析"""
        summary = {}

        # 1. 订单簿
        if self.use_cryexc:
            # 等待数据填充
            await asyncio.sleep(1)

        logger.info("正在获取订单簿数据...")
        orderbook = self.get_orderbook()
        summary['orderbook'] = orderbook

        # 2. 成交数据
        logger.info("正在获取成交数据...")
        trades_df = self.get_recent_trades(limit=10000)

        # 3. CVD
        if trades_df is not None:
            cvd = self.calculate_cvd(trades_df, window='15min')
            summary['cvd'] = cvd

            # 4. 鲸鱼交易
            whale_trades = self.detect_whale_trades(trades_df, threshold_usd=1000000)
            summary['whale_trades'] = whale_trades

            # 5. 数据信息
            time_span = trades_df['time'].max() - trades_df['time'].min()
            summary['data_info'] = {
                'trade_count': len(trades_df),
                'time_span': time_span,
                'time_span_minutes': time_span.total_seconds() / 60
            }

        # 6. 订单墙
        if orderbook:
            walls = self.identify_order_walls(orderbook, threshold_pct=0.5)
            summary['order_walls'] = walls

        # 7. 清算数据
        logger.info("正在获取清算事件...")
        liquidations = self.get_liquidations(limit=100)
        summary['liquidations'] = liquidations

        return summary

    def get_order_flow_summary(self) -> Dict:
        """同步接口（兼容现有代码）"""
        if self.use_cryexc:
            # 如果使用CryExc，需要先调用initialize_async
            logger.warning("CryExc模式需要先调用initialize_async()，使用空数据")
            return {}
        else:
            # 使用原有逻辑
            return self.client.get_order_flow_summary(use_extended_data=True)


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_cryexc():
        """测试CryExc客户端"""
        print("=" * 70)
        print("CryExc订单流客户端测试")
        print("=" * 70)

        # 创建CryExc客户端
        hub = OrderFlowHubCryExc(use_cryexc=True)

        # 初始化
        print("\n[步骤1] 连接到CryExc后端...")
        await hub.initialize_async()

        # 等待数据填充
        print("\n[步骤2] 等待数据填充（3秒）...")
        await asyncio.sleep(3)

        # 获取数据
        print("\n[步骤3] 获取订单流综合分析...")
        summary = await hub.get_order_flow_summary_async()

        # 显示结果
        print("\n" + "=" * 70)
        print("订单流分析结果")
        print("=" * 70)

        if summary.get('order_walls'):
            walls = summary['order_walls']
            current_price = walls['current_price']
            print(f"\n当前价格: ${current_price:,.0f}")

        if summary.get('cvd'):
            cvd = summary['cvd']
            print(f"\n📊 CVD分析:")
            print(f"  CVD值: {cvd['current_cvd']:,.0f} USD")
            print(f"  变化: {cvd['cvd_change']:+,.0f} USD")
            print(f"  买入占比: {cvd['buy_ratio']:.1%}")
            print(f"  趋势: {cvd['trend']}")

        if summary.get('whale_trades'):
            whale = summary['whale_trades']
            print(f"\n🐋 鲸鱼交易 ({len(whale)}笔):")
            for trade in whale[:5]:
                print(f"  {trade['side']} ${trade['value']:,.0f} @ ${trade['price']:,.2f}")

        if summary.get('data_info'):
            info = summary['data_info']
            print(f"\n📊 数据信息:")
            print(f"  数据量: {info['trade_count']:,}笔")
            print(f"  时间跨度: {info['time_span']}")
            print(f"  时间跨度: {info['time_span_minutes']:.1f}分钟")

        print("\n" + "=" * 70)

        # 关闭连接
        await hub.cryexc_client.close()

    # 运行测试
    asyncio.run(test_cryexc())
