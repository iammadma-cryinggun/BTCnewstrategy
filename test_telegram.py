# -*- coding: utf-8 -*-
"""
Telegram推送诊断脚本
测试Telegram Bot是否正常工作
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import telebot
from main_v80 import V80Config

print("=" * 70)
print("Telegram推送诊断")
print("=" * 70)

# 加载配置
config = V80Config()

print(f"\n1. Telegram启用状态: {config.telegram_enabled}")

if not config.telegram_enabled:
    print("   ❌ Telegram未启用！")
    print("   请在config.json中设置telegram_enabled为true")
    sys.exit(1)

print(f"2. Bot Token: {config.telegram_token[:10]}...{config.telegram_token[-10:]}")
print(f"3. Chat ID: {config.telegram_chat_id}")

# 测试Bot连接
print("\n4. 测试Bot连接...")
try:
    bot = telebot.TeleBot(config.telegram_token)
    bot_info = bot.get_me()
    print(f"   ✅ Bot连接成功!")
    print(f"   Bot名称: @{bot_info.username}")
    print(f"   Bot ID: {bot_info.id}")
except Exception as e:
    print(f"   ❌ Bot连接失败: {e}")
    sys.exit(1)

# 测试发送消息
print("\n5. 测试发送消息...")
test_message = """
🧪 Telegram测试消息

如果你看到这条消息，说明Telegram推送正常！

时间: {time}
""".format(time=__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

try:
    result = bot.send_message(config.telegram_chat_id, test_message)
    print(f"   ✅ 消息发送成功!")
    print(f"   消息ID: {result.message_id}")
except Exception as e:
    print(f"   ❌ 消息发送失败: {e}")
    print(f"\n可能的原因:")
    print(f"   1. Chat ID错误（当前: {config.telegram_chat_id}）")
    print(f"   2. Bot没有被添加到群组")
    print(f"   3. Bot没有发送消息权限")
    print(f"   4. 网络连接问题")

    # 提供获取正确Chat ID的方法
    print("\n💡 如何获取正确的Chat ID?")
    print("   1. 给Bot发送一条任意消息")
    api_url = f"https://api.telegram.org/bot{config.telegram_token}/getUpdates"
    print(f"   2. 访问: {api_url}")
    print("   3. 在返回的JSON中找到'chat':{'id':数字}")
    print("   4. 复制这个数字作为Chat ID")

    sys.exit(1)

print("\n" + "=" * 70)
print("✅ Telegram推送诊断完成 - 一切正常!")
print("=" * 70)
