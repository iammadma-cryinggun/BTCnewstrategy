# -*- coding: utf-8 -*-
"""
Telegram连接测试脚本
"""
import os
import sys

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# 导入配置
from v707_trader_main import V707TraderConfig
from v707_trader_part2 import TelegramNotifier

print("=" * 70)
print("Telegram连接测试")
print("=" * 70)

# 创建配置
config = V707TraderConfig()

print(f"\n[配置] Token: {config.telegram_token[:20]}...")
print(f"[配置] Chat ID: {config.telegram_chat_id}")
print(f"[配置] Enabled: {config.telegram_enabled}")

# 创建通知器
notifier = TelegramNotifier(config)

# 测试发送消息
print("\n[测试] 发送测试消息...")
test_message = """
🧪 *V7.0.7 Telegram测试*

这是一条测试消息，用于验证Telegram连接是否正常。

如果您看到这条消息，说明：
✅ Token配置正确
✅ Chat ID配置正确
✅ 网络连接正常
✅ V7.0.7系统就绪

⏰ 测试时间: 请检查
"""

notifier.send_message(test_message)

print("\n[测试] 消息已发送，请检查Telegram是否收到")
print("\n如果未收到，请检查：")
print("1. Token是否正确（应以850518开头）")
print("2. Chat ID是否正确（应为838429342）")
print("3. 网络是否可以访问Telegram API")
print("4. .env文件是否已配置")
