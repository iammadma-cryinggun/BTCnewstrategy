# -*- coding: utf-8 -*-
"""
CryExc后端测试客户端

测试WebSocket连接和数据流
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import json
import websockets
from datetime import datetime


async def test_cryexc_backend():
    """测试CryExc后端"""

    uri = "ws://127.0.0.1:8000/ws"

    print("=" * 70)
    print("CryExc Backend 测试客户端")
    print("=" * 70)
    print(f"连接到: {uri}")
    print()

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ 连接成功！")
            print()

            # 测试1: Ping
            print("[测试1] 发送ping...")
            await websocket.send(json.dumps({"type": "ping"}))
            response = await websocket.recv()
            data = json.loads(response)
            print(f"  收到: {data}")
            assert data['type'] == 'pong', "Ping失败"
            print("  ✓ Ping测试通过")
            print()

            # 测试2: 订阅trade流
            print("[测试2] 订阅trade流...")
            await websocket.send(json.dumps({
                "type": "stream_subscribe",
                "stream": "trade",
                "config": {
                    "symbol": "BTCUSDT",
                    "minNotional": 100000  # 只显示大于$100K的交易
                }
            }))

            # 等待确认
            response = await websocket.recv()
            data = json.loads(response)
            print(f"  收到: {data}")
            assert data['type'] == 'subscribed', "订阅失败"
            print("  ✓ Trade流订阅成功")
            print()

            # 测试3: 接收trade数据（5秒）
            print("[测试3] 接收trade数据（5秒）...")
            end_time = asyncio.get_event_loop().time() + 5
            trade_count = 0

            while asyncio.get_event_loop().time() < end_time:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)

                    if data.get('type') == 'trade':
                        trade = data['data']
                        trade_count += 1
                        side = "BUY" if not trade['isBuyerMaker'] else "SELL"
                        print(f"  [{trade_count}] {side} ${trade['quote_qty']:,.0f} @ ${trade['price']:,.2f}")

                except asyncio.TimeoutError:
                    break

            print(f"  ✓ 共接收到 {trade_count} 笔大额交易")
            print()

            # 测试4: 订阅orderbook流
            print("[测试4] 订阅orderbook流...")
            await websocket.send(json.dumps({
                "type": "stream_subscribe",
                "stream": "orderbook",
                "config": {
                    "symbol": "BTCUSDT",
                    "depth": 10
                }
            }))

            response = await websocket.recv()
            data = json.loads(response)
            print(f"  收到: {data}")
            assert data['type'] == 'subscribed', "订阅失败"
            print("  ✓ Orderbook流订阅成功")
            print()

            # 接收orderbook数据
            print("[测试5] 接收orderbook数据（1次）...")
            response = await websocket.recv()
            data = json.loads(response)
            assert data['type'] == 'orderbook', "数据类型错误"

            orderbook = data['data']
            print(f"  买盘前3档:")
            for bid in orderbook['bids'][:3]:
                print(f"    ${bid[0]:,.2f} - {bid[1]:.4f} BTC")
            print(f"  卖盘前3档:")
            for ask in orderbook['asks'][:3]:
                print(f"    ${ask[0]:,.2f} - {ask[1]:.4f} BTC")
            print()

            # 测试6: 取消订阅trade流
            print("[测试6] 取消订阅trade流...")
            await websocket.send(json.dumps({
                "type": "stream_unsubscribe",
                "stream": "trade"
            }))

            response = await websocket.recv()
            data = json.loads(response)
            print(f"  收到: {data}")
            assert data['type'] == 'unsubscribed', "取消订阅失败"
            print("  ✓ Trade流取消订阅成功")
            print()

            print("=" * 70)
            print("✓ 所有测试通过！")
            print("=" * 70)

    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket连接失败: {e}")
        print("\n请确保CryExc后端正在运行:")
        print("  python cryexc_backend.py")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_cryexc_backend())
