# V8.0 BTC交易系统 - 生产环境

## 📋 文件清单

```
production/
├── main_v80.py              # 主程序（V8.0验证5系统）
├── order_flow_hub.py        # 订单流数据获取
├── collect_orderflow_robust.py  # 历史数据收集
├── Procfile                  # Zeabur部署配置
├── requirements.txt          # Python依赖
└── README.md                # 项目说明
```

## 🚀 快速启动

### 方式1：本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 配置Telegram Token（可选）
# 编辑main_v80.py，修改第42-43行的token

# 运行主程序
python main_v80.py
```

### 方式2：Zeabur云端部署

1. **准备GitHub**
   - 确保代码已推送到GitHub
   - Zeabur会自动检测到Procfile

2. **创建Zeabur服务**
   - 登录 https://zeabur.com
   - 点击"New Project"
   - 选择GitHub仓库
   - 选择服务类型（Predeployed Docker）

3. **配置环境变量**（可选）
   ```
   TELEGRAM_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

4. **部署**
   - 点击"Deploy"
   - 等待服务启动

5. **查看日志**
   - 点击服务 → "Logs"
   - 应该看到每小时心跳日志

### 方式3：收集历史数据

```bash
# 收集2天历史数据（用于回测）
python collect_orderflow_robust.py
```

## 📊 系统功能

### main_v80.py - 主程序

**特性**：
- ✅ V8.0验证5逻辑（FFT + Hilbert + 二阶差分）
- ✅ 每4小时检查交易信号
- ✅ 每小时检查持仓状态
- ✅ Telegram实时通知
- ✅ 每小时心跳日志
- ✅ 自动状态保存

**交易规则**：
- 止损：-3%
- 止盈：+10%
- 风险：每次2%
- 信号置信度阈值：60%

### order_flow_hub.py - 数据获取

**功能**：
- 📊 获取订单簿深度（买盘/卖盘）
- 📈 获取成交数据（最近1000笔）
- 💰 计算CVD（累积成交量偏差）
- 🐋 检测大单交易（>$1M）
- 🧱 识别订单墙（支撑/阻力）

**改进点**：
- 扩展数据：10,000笔 vs 1,000笔
- CVD窗口：15min vs 5min
- 更准确的趋势判断

## 📱 Telegram命令

部署后，可通过Telegram控制：

```
/start - 显示帮助信息
/status - 查看当前持仓
/signals - 查看最近信号
/trades - 查看交易历史
/clear - 手动平仓（谨慎使用）
```

## ⚠️ 注意事项

1. **首次运行**：会立即检查信号并发送通知
2. **状态保存**：自动保存到`v80_state.pkl`
3. **日志文件**：保存到`v80_cloud.log`
4. **时区**：使用北京时间（UTC+8）

## 🔧 故障排除

### 问题1：Zeabur容器不断重启

**原因**：可能是依赖缺失

**解决**：检查requirements.txt是否包含所有依赖

### 问题2：没有心跳日志

**原因**：系统还在启动中

**解决**：等待1小时，心跳日志每小时出现一次

### 问题3：Telegram通知失败

**原因**：Token配置错误

**解决**：
1. 检查Token是否正确
2. 检查Chat ID是否正确
3. 运行测试脚本验证

## 📞 支持

如有问题，请查看项目根目录的README.md
