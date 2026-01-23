# CryExc Backend Server

基于CryExc协议的WebSocket后端服务器，用于实时加密货币交易数据。

## 功能

### 实现的数据流

| 流类型 | 说明 | 状态 |
|--------|------|------|
| `trade` | 实时成交数据（支持过滤大额交易） | ✓ |
| `orderbook` | 订单簿深度数据 | ✓ |
| `cvd` | 累积成交量偏差（CVD） | ✓ |
| `liquidation` | 清算事件 | ✓ |

### CryExc协议兼容性

- ✓ `ping` / `pong` 心跳
- ✓ `stream_subscribe` 订阅流
- ✓ `stream_unsubscribe` 取消订阅
- ✓ `subscribed` / `unsubscribed` 确认消息
- ✓ `error` 错误消息

## 安装依赖

```bash
pip install fastapi uvicorn websockets requests pandas
```

## 使用方法

### 启动服务器

```bash
python cryexc_backend.py
```

服务器将在 `ws://127.0.0.1:8000/ws` 启动。

**注意**: Windows用户请使用 `127.0.0.1` 而不是 `localhost`，因为localhost在Windows上可能导致WebSocket连接失败。

### 测试连接

**方法1：使用测试客户端**
```bash
python test_cryexc_client.py
```

**方法2：使用在线WebSocket工具**
1. 访问 http://www.websocket-test.com/
2. 连接到 `ws://localhost:8000/ws`
3. 发送消息：
```json
{
  "type": "stream_subscribe",
  "stream": "trade",
  "config": {
    "symbol": "BTCUSDT",
    "minNotional": 100000
  }
}
```

### API端点

| 端点 | 说明 |
|------|------|
| `GET /` | 服务器状态 |
| `GET /api/exchanges` | 支持的交易所 |
| `GET /api/symbols` | 支持的交易对 |
| `WS /ws` | WebSocket连接 |

## 订阅示例

### 订阅成交数据

```json
{
  "type": "stream_subscribe",
  "stream": "trade",
  "config": {
    "symbol": "BTCUSDT",
    "minNotional": 50000
  }
}
```

**返回数据示例：**
```json
{
  "type": "trade",
  "data": {
    "exchange": "binancef",
    "symbol": "BTCUSDT",
    "price": 89500.50,
    "qty": 1.5,
    "quoteQty": 134250.75,
    "isBuyerMaker": false,
    "timestamp": 1704067200000
  }
}
```

### 订阅订单簿

```json
{
  "type": "stream_subscribe",
  "stream": "orderbook",
  "config": {
    "symbol": "BTCUSDT",
    "depth": 20
  }
}
```

### 订阅CVD

```json
{
  "type": "stream_subscribe",
  "stream": "cvd",
  "config": {
    "symbol": "BTCUSDT",
    "window": "5min"
  }
}
```

### 订阅清算数据

```json
{
  "type": "stream_subscribe",
  "stream": "liquidation",
  "config": {
    "symbol": "BTCUSDT"
  }
}
```

## 与前端集成

### CryExc前端

1. 打开CryExc前端
2. 点击连接设置
3. 选择"Self-Hosted Backend"
4. 输入：`ws://localhost:8000`
5. 点击连接

### 自定义前端

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  // 订阅trade流
  ws.send(JSON.stringify({
    type: 'stream_subscribe',
    stream: 'trade',
    config: {
      symbol: 'BTCUSDT',
      minNotional: 100000
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'trade') {
    console.log('Trade:', data.data);
  }
};
```

## 性能

- 单连接处理：异步
- 数据更新频率：
  - trade: 1秒
  - orderbook: 2秒
  - cvd: 60秒
  - liquidation: 5秒

## 数据来源

所有数据来自Binance期货API：
- 成交数据：`/fapi/v1/aggTrades`
- 订单簿：`/fapi/v1/depth`
- 清算：`/fapi/v1/allForceOrders`

## 注意事项

1. 这是基础实现，仅支持Binance期货
2. CVD计算基于历史10000笔成交
3. 建议在生产环境使用专用WebSocket服务器（如node.js）
4. 可以扩展支持多交易所聚合

## 后续改进

- [ ] 支持历史数据查询
- [ ] 添加数据缓存
- [ ] 支持多交易所聚合
- [ ] 实现更多流类型（footprint, DOM等）
- [ ] 添加认证机制

## 许可

MIT License
