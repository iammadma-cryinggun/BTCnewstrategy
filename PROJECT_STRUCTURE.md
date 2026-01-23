# 项目文件结构说明

## 📁 production/ - 生产环境核心文件

这些是系统运行**必需**的文件：

### 主程序
- `main_v80.py` - V8.0验证5智能预警系统（主程序）
  - 每4小时检查交易信号
  - 每小时检查持仓
  - Telegram通知
  - 每小时心跳日志

### 数据获取
- `order_flow_hub.py` - 订单流数据获取（直接从Binance API）
  - 获取订单簿深度
  - 获取成交数据
  - 计算CVD（累积成交量偏差）
  - 检测大单交易
  - 识别订单墙

### 历史数据收集
- `collect_orderflow_robust.py` - 历史数据收集脚本
  - 分批收集多天数据
  - SSL错误自动重试
  - 带心跳日志

### 部署配置
- `Procfile` - Zeabur部署配置
  - 指定启动命令：`python main_v80.py`
- `requirements.txt` - Python依赖列表
  - pandas, numpy, requests, telebot, scipy等

### 文档
- `README.md` - 项目说明文档

---

## 📁 docs/ - 文档和说明

项目相关的文档文件

---

## 📁 tests/ - 测试文件

测试和验证脚本

---

## 📁 data/ - 数据文件

历史数据和结果文件

---

## 📁 logs/ - 日志文件

运行日志和输出

---

## 📁 根目录 - 其他文件

- 其他版本的main文件（main.py, main_v708.py等）
- 回测脚本
- 分析工具
- 临时文件

---

## 🚀 快速开始

### 本地运行
```bash
cd production
python main_v80.py
```

### Zeabur部署
1. 推送代码到GitHub
2. Zeabur选择此仓库
3. 配置环境变量（TELEGRAM_TOKEN等）
4. 部署

### 收集历史数据
```bash
python collect_orderflow_robust.py
```

---

## 📊 文件优先级

**必需**（生产环境）：
- ✅ production/main_v80.py
- ✅ production/order_flow_hub.py
- ✅ production/Procfile
- ✅ production/requirements.txt

**可选**（开发和测试）：
- docs/, tests/, data/ 文件夹内容
- 根目录的其他脚本和工具
