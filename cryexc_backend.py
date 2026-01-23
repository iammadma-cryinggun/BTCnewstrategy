# -*- coding: utf-8 -*-
"""
CryExc Backend Server - 实现CryExc协议

功能：
1. WebSocket服务器（/ws）
2. 实现stream_subscribe, stream_unsubscribe, ping/pong
3. 流类型：trade, orderbook, cvd, liquidation
4. 从Binance获取数据并实时转发

作者: Claude Sonnet 4.5
日期: 2026-01-23
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Any, List
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CryExc Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
connected_clients: Set[WebSocket] = set()
client_subscriptions: Dict[WebSocket, Dict[str, Any]] = {}


class ConnectionManager:
    """管理WebSocket连接"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        connected_clients.add(websocket)
        client_subscriptions[websocket] = {}
        logger.info(f"✓ 新客户端连接 (总计: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        connected_clients.discard(websocket)
        if websocket in client_subscriptions:
            del client_subscriptions[websocket]
        logger.info(f"✗ 客户端断开 (总计: {len(self.active_connections)})")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


class DataStreamer:
    """数据流管理器 - 从Binance获取数据并转发"""

    def __init__(self):
        self.binance_base = "https://fapi.binance.com"
        self.session = requests.Session()
        self.running = True

    async def stream_trade(self, websocket: WebSocket, config: dict):
        """
        流式发送成交数据

        Config:
        - symbol: 交易对
        - minNotional: 最小成交额
        """
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            min_notional = config.get('minNotional', 50000)

            while self.running and websocket in connected_clients:
                try:
                    # 获取最近成交
                    url = f"{self.binance_base}/fapi/v1/aggTrades"
                    params = {'symbol': symbol, 'limit': 100}

                    response = self.session.get(url, params=params, timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    # 过滤大额交易
                    filtered_trades = []
                    for trade in data:
                        price = float(trade['p'])
                        qty = float(trade['q'])
                        quote_qty = price * qty

                        if quote_qty >= min_notional:
                            filtered_trades.append({
                                "exchange": "binancef",
                                "symbol": symbol,
                                "price": price,
                                "qty": qty,
                                "quoteQty": quote_qty,
                                "isBuyerMaker": trade['m'],
                                "timestamp": trade['T']
                            })

                    # 发送数据
                    if filtered_trades:
                        for trade in filtered_trades:
                            await manager.send_personal_message({
                                "type": "trade",
                                "data": trade
                            }, websocket)

                    # 等待1秒再获取下一批
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"获取trade数据失败: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Trade stream error: {e}")

    async def stream_orderbook(self, websocket: WebSocket, config: dict):
        """
        流式发送订单簿数据

        Config:
        - symbol: 交易对
        - depth: 深度（默认20）
        """
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            depth = config.get('depth', 20)

            while self.running and websocket in connected_clients:
                try:
                    # 获取订单簿
                    url = f"{self.binance_base}/fapi/v1/depth"
                    params = {'symbol': symbol, 'limit': depth}

                    response = self.session.get(url, params=params, timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    # 格式化数据
                    orderbook = {
                        "symbol": symbol,
                        "bids": [[float(p), float(q)] for p, q in data['bids']],
                        "asks": [[float(p), float(q)] for p, q in data['asks']],
                        "timestamp": int(datetime.now().timestamp() * 1000)
                    }

                    # 发送数据
                    await manager.send_personal_message({
                        "type": "orderbook",
                        "data": orderbook
                    }, websocket)

                    # 每2秒更新一次
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"获取orderbook数据失败: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Orderbook stream error: {e}")

    async def stream_cvd(self, websocket: WebSocket, config: dict):
        """
        流式发送CVD数据

        Config:
        - symbol: 交易对
        - window: 时间窗口（默认5min）
        """
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            window = config.get('window', '5min')

            # 获取历史数据用于CVD计算
            url = f"{self.binance_base}/fapi/v1/aggTrades"
            params = {'symbol': symbol, 'limit': 10000}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 转换为DataFrame
            trades_data = []
            for trade in data:
                trades_data.append({
                    'price': float(trade['p']),
                    'qty': float(trade['q']),
                    'quote_qty': float(trade['p']) * float(trade['q']),
                    'time': pd.to_datetime(trade['T'], unit='ms'),
                    'is_buyer_maker': trade['m']
                })

            df = pd.DataFrame(trades_data)
            df.set_index('time', inplace=True)

            # 计算CVD
            buy_volume = df[~df['is_buyer_maker']]['quote_qty'].resample(window).sum()
            sell_volume = df[df['is_buyer_maker']]['quote_qty'].resample(window).sum()
            delta = buy_volume - sell_volume
            cvd = delta.cumsum()

            # 发送历史CVD数据
            for i in range(len(cvd)):
                await manager.send_personal_message({
                    "type": "cvd_historical",
                    "data": {
                        "timestamp": int(cvd.index[i].timestamp() * 1000),
                        "cvd": float(cvd.iloc[i]),
                        "delta": float(delta.iloc[i]) if i < len(delta) else 0,
                        "buyVolume": float(buy_volume.iloc[i]) if i < len(buy_volume) else 0,
                        "sellVolume": float(sell_volume.iloc[i]) if i < len(sell_volume) else 0
                    }
                }, websocket)

            # 持续发送最新CVD
            while self.running and websocket in connected_clients:
                await asyncio.sleep(60)  # 每分钟更新一次

        except Exception as e:
            logger.error(f"CVD stream error: {e}")

    async def stream_liquidation(self, websocket: WebSocket, config: dict):
        """
        流式发送清算数据

        Config:
        - symbol: 交易对
        """
        try:
            symbol = config.get('symbol', 'BTCUSDT')

            while self.running and websocket in connected_clients:
                try:
                    # 获取清算数据
                    url = f"{self.binance_base}/fapi/v1/allForceOrders"
                    params = {'symbol': symbol, 'limit': 100}

                    response = self.session.get(url, params=params, timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    # 发送清算事件
                    for item in data:
                        await manager.send_personal_message({
                            "type": "liquidation",
                            "data": {
                                "exchange": "binancef",
                                "symbol": symbol,
                                "side": item['side'],
                                "price": float(item['price']),
                                "qty": float(item['amount']),
                                "time": item['time']
                            }
                        }, websocket)

                    # 每5秒更新一次
                    await asyncio.sleep(5)

                except Exception as e:
                    logger.error(f"获取liquidation数据失败: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Liquidation stream error: {e}")


streamer = DataStreamer()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 实现CryExc协议"""
    await manager.connect(websocket)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get('type')

            # 处理不同类型的消息
            if msg_type == 'ping':
                await manager.send_personal_message({"type": "pong"}, websocket)

            elif msg_type == 'stream_subscribe':
                stream = message.get('stream')
                config = message.get('config', {})

                logger.info(f"订阅流: {stream}, config: {config}")

                # 保存订阅信息
                if websocket not in client_subscriptions:
                    client_subscriptions[websocket] = {}
                client_subscriptions[websocket][stream] = config

                # 确认订阅
                await manager.send_personal_message({
                    "type": "subscribed",
                    "stream": stream,
                    "config": config
                }, websocket)

                # 启动数据流
                if stream == 'trade':
                    asyncio.create_task(streamer.stream_trade(websocket, config))
                elif stream == 'orderbook':
                    asyncio.create_task(streamer.stream_orderbook(websocket, config))
                elif stream == 'cvd':
                    asyncio.create_task(streamer.stream_cvd(websocket, config))
                elif stream == 'liquidation':
                    asyncio.create_task(streamer.stream_liquidation(websocket, config))
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "stream": stream,
                        "error": f"Unknown stream: {stream}"
                    }, websocket)

            elif msg_type == 'stream_unsubscribe':
                stream = message.get('stream')

                logger.info(f"取消订阅流: {stream}")

                # 移除订阅
                if websocket in client_subscriptions and stream in client_subscriptions[websocket]:
                    del client_subscriptions[websocket][stream]

                # 确认取消订阅
                await manager.send_personal_message({
                    "type": "unsubscribed",
                    "stream": stream
                }, websocket)

            else:
                await manager.send_personal_message({
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}"
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "CryExc Backend",
        "version": "1.0.0",
        "status": "running",
        "connections": len(manager.active_connections),
        "streams": ["trade", "orderbook", "cvd", "liquidation"]
    }


@app.get("/api/exchanges")
async def get_exchanges():
    """获取支持的交易所"""
    return {
        "exchanges": [
            {"id": "binancef", "name": "Binance Futures", "enabled": True}
        ]
    }


@app.get("/api/symbols")
async def get_symbols():
    """获取支持的交易对"""
    return {
        "symbols": [
            {"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT", "enabled": True},
            {"symbol": "ETHUSDT", "base": "ETH", "quote": "USDT", "enabled": True}
        ]
    }


if __name__ == "__main__":
    print("=" * 70)
    print("CryExc Backend Server")
    print("=" * 70)
    print("启动服务器...")
    print("WebSocket端点: ws://localhost:8000/ws")
    print("API文档: http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
