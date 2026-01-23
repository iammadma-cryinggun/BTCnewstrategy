# -*- coding: utf-8 -*-
"""
测试Telegram API连接
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import telebot
import requests
from datetime import datetime

# Token和Chat ID
TELEGRAM_TOKEN = '8505180201:AAGOSkhXHRu77OlRMu0PZCbKtYMEr1tRGAk'
TELEGRAM_CHAT_ID = '838429342'

print("=" * 70)
print("Telegram API 连接测试")
print("=" * 70)

# 测试1: 检查网络连接
print("\n[测试1] 检查Telegram API连接...")
try:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
    print(f"  请求: {url}")
    response = requests.get(url, timeout=10)
    print(f"  状态码: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        if data['ok']:
            bot_info = data['result']
            print(f"  ✓ Bot信息: @{bot_info.get('username', 'N/A')}")
            print(f"  ✓ Bot名称: {bot_info.get('first_name', 'N/A')}")
        else:
            print(f"  ✗ API返回错误: {data}")
    else:
        print(f"  ✗ HTTP错误: {response.status_code}")

except requests.exceptions.Timeout:
    print("  ✗ 连接超时 - 无法连接到Telegram服务器")
    print("  可能原因:")
    print("    1. 网络防火墙阻止")
    print("    2. 需要VPN/代理")
    print("    3. Telegram服务被屏蔽")
except Exception as e:
    print(f"  ✗ 连接错误: {e}")

# 测试2: 发送测试消息
print("\n[测试2] 发送测试消息...")
try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    message = f"""
🤖 Telegram测试消息

时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Token: {TELEGRAM_TOKEN[:20]}...
Chat ID: {TELEGRAM_CHAT_ID}

这是一条测试消息，如果您收到此消息，说明Telegram配置正确！
"""

    print(f"  发送到Chat ID: {TELEGRAM_CHAT_ID}")
    result = bot.send_message(TELEGRAM_CHAT_ID, message, timeout=15)
    print(f"  ✓ 消息发送成功!")
    print(f"  ✓ 消息ID: {result.message_id}")

except telebot.apihelper.ApiTelegramException as e:
    print(f"  ✗ Telegram API错误: {e}")
    if "bot was blocked by the user" in str(e):
        print("  原因: Bot被用户阻止")
    elif "chat not found" in str(e):
        print("  原因: Chat ID不存在")
    elif "user is deactivated" in str(e):
        print("  原因: 用户账号已停用")
except Exception as e:
    print(f"  ✗ 发送失败: {e}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
