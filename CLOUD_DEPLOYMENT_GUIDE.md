# V8.0 云端部署指南

**版本**: v5.0 Cloud
**日期**: 2026-01-22
**状态**: ✅ 生产就绪

---

## 📋 部署前准备

### 1. Telegram Bot 创建

1. **与 BotFather 对话**
   ```
   打开 Telegram，搜索 @BotFather
   发送 /newbot
   按提示设置 bot 名称和用户名
   ```

2. **获取 Token**
   ```
   BotFather 会返回类似这样的 token:
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

3. **获取 Chat ID**
   ```
   方式1（推荐）:
   在 Telegram 搜索 @userinfobot
   发送任意消息，它会返回你的 Chat ID

   方式2（手动）:
   访问: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   找到 "chat":{"id":数字} 部分
   ```

---

## 🚀 方式1: Zeabur 部署（推荐）

### 步骤1: 推送代码到 GitHub

```bash
cd D:\机器人\btc_4hour_alert
git init
git add v80_cloud_telegram.py requirements_cloud.txt
git commit -m "V8.0 Cloud Trading System with Telegram"
git remote add origin https://github.com/你的用户名/btc-trading-bot.git
git push -u origin main
```

### 步骤2: 在 Zeabur 创建项目

1. 访问 https://zeabur.com
2. 点击 "New Project"
3. 选择 "Deploy from GitHub"
4. 选择你的仓库

### 步骤3: 配置环境变量

在 Zeabur 项目设置中添加以下环境变量：

| 变量名 | 值 | 说明 | 必填 |
|-------|-----|------|------|
| `TELEGRAM_TOKEN` | 你的Bot Token | Telegram Bot认证 | ✅ |
| `TELEGRAM_CHAT_ID` | 你的Chat ID | 接收通知的聊天ID | ✅ |
| `ACCOUNT_BALANCE` | 10000 | 账户余额(USD) | ❌(默认10000) |
| `RISK_PER_TRADE` | 0.02 | 单笔风险(2%) | ❌(默认0.02) |
| `CHECK_INTERVAL` | 300 | 检查间隔(秒) | ❌(默认300) |

### 步骤4: 启动部署

1. Zeabur 会自动检测 Python 项目
2. 确认使用 `requirements_cloud.txt` 作为依赖
3. 点击 "Deploy"
4. 等待部署完成（约2-3分钟）

### 步骤5: 验证运行

在你的 Telegram 中发送：
```
/start
```

Bot 应该回复：
```
🎯 V8.0 云端交易系统已启动

✅ 系统正常运行
📊 使用 /status 查看状态
📈 使用 /signals 查看信号
📋 使用 /trades 查看交易历史
❓ 使用 /help 查看所有命令
```

---

## 🐳 方式2: Docker 部署

### 步骤1: 创建 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY v80_cloud_telegram.py .

CMD ["python", "v80_cloud_telegram.py"]
```

### 步骤2: 构建和运行

```bash
# 构建镜像
docker build -t btc-trading-v80 .

# 运行容器
docker run -d \
  -e TELEGRAM_TOKEN="你的token" \
  -e TELEGRAM_CHAT_ID="你的chat_id" \
  -e ACCOUNT_BALANCE="10000" \
  -e RISK_PER_TRADE="0.02" \
  -e CHECK_INTERVAL="300" \
  --name btc-bot \
  btc-trading-v80
```

---

## 💻 方式3: Replit 部署

### 步骤1: 创建 Repl

1. 访问 https://replit.com
2. 点击 "+ Create Repl"
3. 选择 "Python" 模板
4. 命名为 "btc-trading-v80"

### 步骤2: 上传代码

1. 将 `v80_cloud_telegram.py` 上传到 Repl
2. 将 `requirements_cloud.txt` 上传到 Repl

### 步骤3: 安装依赖

在 Repl 终端运行：
```bash
pip install -r requirements_cloud.txt
```

### 步骤4: 配置环境变量

在 Repl 左侧点击 "Secrets"，添加：
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
- ACCOUNT_BALANCE
- RISK_PER_TRADE
- CHECK_INTERVAL

### 步骤5: 运行

在 Repl 终端运行：
```bash
python v80_cloud_telegram.py
```

点击 "Keep Running" 确保持续运行。

---

## 📱 Telegram 命令说明

### 基础命令

| 命令 | 说明 |
|------|------|
| `/start` | 启动 Bot，显示欢迎信息 |
| `/status` | 查看当前状态（持仓、余额、统计） |
| `/signals` | 查看最近的交易信号（最多20条） |
| `/trades` | 查看交易历史（所有已完成交易） |
| `/clear` | 手动平仓（危险操作，需确认） |
| `/config` | 查看当前配置参数 |
| `/help` | 显示帮助信息 |

### 通知类型

系统会自动发送以下通知：

1. **新信号通知**
   ```
   🎯 新交易信号

   类型: BEARISH_SINGULARITY
   动作: LONG (反向抄底)
   置信度: 90%
   价格: $95,000

   张力: 0.42
   加速度: -0.03
   DXY燃料: 0.15

   理由: 强奇点看空 (宏观失速)
   ```

2. **开仓通知**
   ```
   ✅ 开仓成功

   方向: LONG
   价格: $95,000
   仓位: $6,667

   止损: $92,150 (-3%)
   止盈: $104,500 (+10%)

   信号: BEARISH_SINGULARITY
   置信度: 90%
   ```

3. **平仓通知**
   ```
   ✅ 平仓成功

   方向: LONG
   开仓价: $95,000
   平仓价: $104,500
   盈亏: +10.00%

   利润: +$667
   余额: $10,667
   原因: 止盈 (10.00%)
   ```

4. **错误通知**
   ```
   ⚠️ 系统错误

   类型: 数据获取失败
   详情: BTC API连接超时
   时间: 2026-01-22 14:30:00
   ```

---

## 🔍 监控和日志

### Zeabur 日志

1. 在 Zeabur 项目页面
2. 点击 "Logs" 标签
3. 实时查看系统日志

### 日志级别

```
[INFO]  - 正常运行信息
[WARNING] - 警告（如DXY数据获取失败）
[ERROR] - 错误（如API连接失败）
```

### 关键日志示例

```
2026-01-22 14:30:00 [INFO] [配置] 从环境变量加载配置
2026-01-22 14:30:00 [INFO] [Telegram] Telegram Bot已启动
2026-01-22 14:30:05 [INFO] [数据] BTC: $95,000 | DXY: 103.2
2026-01-22 14:30:05 [INFO] [信号] BEARISH_SINGULARITY | 置信度: 90%
2026-01-22 14:30:05 [INFO] [开仓] LONG $95,000 仓位=$6,667
2026-01-22 14:30:06 [INFO] [Telegram] 开仓通知已发送
```

---

## ⚠️ 注意事项

### 1. 数据源

- **BTC数据**: 来自 Binance API（无需认证）
- **DXY数据**: 来自 FRED API（免费，无需认证）
- **更新频率**: 每5分钟检查一次

### 2. 风险控制

- **止损**: -3%（自动触发）
- **止盈**: +10%（自动触发）
- **单笔风险**: 账户的2%
- **最大仓位**: 账户的66.7%

### 3. 信号过滤

- 只交易置信度 >= 60% 的信号
- 低于50% 置信度会自动平仓

### 4. 网络稳定性

- 需要稳定的网络连接
- API连接失败会自动重试
- 连续失败会发送错误通知

### 5. 时区

- 系统使用 UTC 时间
- 所有时间戳显示为系统本地时间

---

## 🛠️ 故障排除

### 问题1: Bot 不响应命令

**检查**:
```bash
# 确认 Token 和 Chat ID 正确
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe

# 确认能发送消息
curl "https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage?chat_id=<YOUR_CHAT_ID>&text=Test"
```

**解决**:
- 检查环境变量是否正确设置
- 重新部署项目

### 问题2: 系统无法启动

**检查日志**:
```
[ERROR] BTC数据获取失败
[ERROR] DXY数据获取失败
```

**解决**:
- 检查网络连接
- 确认 API 可访问
- 增加超时时间（修改代码中的 timeout 参数）

### 问题3: 信号延迟

**原因**: 5分钟检查间隔可能错过快速行情

**解决**:
- 修改环境变量 `CHECK_INTERVAL=60`（每1分钟检查）
- 注意：更频繁的检查会增加 API 调用

### 问题4: 内存不足

**原因**: 云平台免费版内存限制

**解决**:
- 定期清理历史数据（代码已自动处理）
- 升级到付费版本

---

## 📊 性能优化建议

### 1. 数据缓存

当前实现：
- 价格历史: 最多100个点
- 信号历史: 最多100条
- 交易历史: 无限制（建议定期归档）

### 2. API 调用

当前频率：
- BTC API: 每5分钟1次
- DXY API: 每5分钟1次
- Telegram API: 按需调用

### 3. 数据库存储（可选扩展）

如需持久化存储，可以添加：
```python
import sqlite3

# 创建数据库
conn = sqlite3.connect('trading_history.db')
cursor = conn.cursor()

# 保存交易
cursor.execute('''
    INSERT INTO trades VALUES (?, ?, ?, ?, ?)
''', (time, action, price, pnl, reason))
```

---

## 🔄 更新和升级

### 更新代码

1. 修改本地代码
2. 推送到 GitHub
3. Zeabur 会自动检测并重新部署

### 修改配置

1. 在 Zeabur 项目设置中修改环境变量
2. 重启项目

### 查看版本

代码中版本信息：
```python
# 版本: v5.0 Cloud
# 日期: 2026-01-22
```

---

## ✅ 部署检查清单

部署前确认：

- [ ] 已创建 Telegram Bot
- [ ] 已获取 Token 和 Chat ID
- [ ] 代码已推送到 GitHub
- [ ] 环境变量已配置
- [ ] requirements_cloud.txt 已上传
- [ ] 已阅读并理解风险控制参数
- [ ] 已测试 Telegram 命令

部署后验证：

- [ ] `/start` 命令有响应
- [ ] `/status` 显示正常状态
- [ ] 收到系统启动通知
- [ ] 日志显示数据正常获取
- [ ] 信号计算正常（等待第一个信号）

---

## 📞 支持

如遇问题：

1. 检查日志文件
2. 查看本文档的故障排除部分
3. 确认所有环境变量已正确设置
4. 验证网络连接正常

---

**祝交易顺利！** 🎯
