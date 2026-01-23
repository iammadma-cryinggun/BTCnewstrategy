# Telegram Bot 409错误清理指南

## 问题描述
```
Error code: 409. Conflict: terminated by other getUpdates request;
make sure that only one bot instance is running
```

## 原因
同一个Bot Token在多个地方同时运行

---

## 🔧 解决方案

### 方案1：检查并清理Zeabur服务（推荐）

#### 步骤1：登录Zeabur
```bash
zeabur auth login
```

#### 步骤2：检查项目和服务器
```bash
# 列出所有项目
zeabur project list

# 列出所有服务
zeabur service list

# 查看服务状态
zeabur service logs <service-name>
```

#### 步骤3：停止或删除重复服务
```bash
# 停止服务（保留配置）
zeabur service stop <service-name>

# 或者删除服务（彻底删除）
zeabur service delete <service-name>
```

#### 步骤4：重启正确服务
```bash
# 启动服务
zeabur service start <service-name>

# 或重新部署
cd C:\Users\Martin\Downloads\机器人\btc_4hour_alert
zeabur deploy
```

---

### 方案2：创建新的Bot Token（最彻底）

#### 步骤1：在Telegram中创建新Bot

1. 在Telegram中搜索 **@BotFather**
2. 发送命令：`/newbot`
3. 按提示设置bot名称和用户名
4. **保存新的Token**（格式：`123456789:ABCdefGHIjklMNOpqrsTUVwxyz`）

#### 步骤2：停止所有使用旧Token的服务

```bash
# 停止Zeabur服务
zeabur service stop <your-service-name>

# 或者如果本地运行，按Ctrl+C停止
```

#### 步骤3：更新配置文件

```bash
# 1. 创建新的.env文件
cd C:\Users\Martin\Downloads\机器人\btc_4hour_alert
cp .env.example .env

# 2. 编辑.env，替换TELEGRAM_TOKEN
# TELEGRAM_TOKEN=你的新Token
# TELEGRAM_CHAT_ID=你的ChatID（保持不变）

# 3. 编辑.env.example（更新示例）
```

#### 步骤4：更新远程环境变量

在Zeabur控制台：
1. 进入项目 → 服务 → Variables
2. 更新 `TELEGRAM_TOKEN` 为新值
3. 保存并重启服务

#### 步骤5：测试新Bot

```python
python -c "
from v708_golden_module import V708TelegramNotifier
import os
notifier = V708TelegramNotifier(
    token='你的新Token',
    chat_id='838429342',
    enabled=True
)
notifier.send('✅ 新Bot测试成功！', priority='high')
"
```

---

### 方案3：等待自动清理（临时方案）

Telegram API会在约5-10分钟后自动清理失效连接。

**操作**：
1. 停止所有bot实例（Zeabur + 本地）
2. 等待10分钟
3. 重新启动一个实例

---

## 🎯 推荐操作流程

### 立即执行（5分钟）

1. **停止Zeabur服务**
   ```bash
   zeabur service list  # 找到服务名
   zeabur service stop btc-4hour-alert
   ```

2. **等待2分钟**（让Telegram API释放连接）

3. **重新启动**
   ```bash
   zeabur service start btc-4hour-alert
   ```

4. **检查日志**
   ```bash
   zeabur service logs btc-4hour-alert --tail 50
   ```

### 如果问题 persists → 使用方案2（创建新Token）

---

## 📋 检查清单

- [ ] 确认只有一处服务在运行
- [ ] 更新所有配置文件中的Token
- [ ] 在Zeabur控制台更新环境变量
- [ ] 测试通知功能
- [ ] 监控日志确认没有409错误

---

## 🔍 如何确认问题解决？

运行以下命令检查日志：
```bash
zeabur service logs <service-name> --tail 100 | grep -E "409|Conflict|Telegram"
```

**正常情况**：不应该看到任何409错误

**如果还有问题**：说明还有其他地方在使用同一个Token，建议创建新Token（方案2）
