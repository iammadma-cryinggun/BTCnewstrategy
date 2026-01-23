# -*- coding: utf-8 -*-
"""
发送Telegram系统启动欢迎消息
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import telebot
from datetime import datetime

TELEGRAM_TOKEN = '8505180201:AAGOSkhXHRu77OlRMu0PZCbKtYMEr1tRGAk'
TELEGRAM_CHAT_ID = '838429342'

print("=" * 70)
print("发送系统启动欢迎消息")
print("=" * 70)

try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    # 构建欢迎消息
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    welcome_message = f"""
🔔 V8.0云端交易系统启动

━━━━━━━━━━━━━━━━━━━━━
⏰ 启动时间: {current_time}
🤖 Bot: @cryinggunbtc4h_bot
━━━━━━━━━━━━━━━━━━━━━

✅ 验证5逻辑已激活
✅ Telegram通知已激活
✅ 订单流增强已启用

📊 可用命令：
/status - 查看持仓状态
/signals - 查看最近信号
/trades - 查看交易历史
/clear - 手动平仓
/help - 显示帮助

━━━━━━━━━━━━━━━━━━━━━
系统就绪，开始监控市场...
"""

    print(f"发送到Chat ID: {TELEGRAM_CHAT_ID}")
    result = bot.send_message(TELEGRAM_CHAT_ID, welcome_message, timeout=15)
    print(f"✓ 欢迎消息发送成功!")
    print(f"✓ 消息ID: {result.message_id}")
    print(f"\n您应该收到一条系统启动通知！")

except Exception as e:
    print(f"✗ 发送失败: {e}")

print("=" * 70)
