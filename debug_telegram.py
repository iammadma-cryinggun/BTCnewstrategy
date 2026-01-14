# -*- coding: utf-8 -*-
"""
Telegram详细调试脚本
"""
import os
import sys
import requests

# 测试配置
TOKEN = "8505180201:AAGOSkhXHRu77OlRMu0PZCbKtYMEr1tRGAk"
CHAT_ID = "838429342"

print("=" * 70)
print("Telegram详细调试")
print("=" * 70)

# 1. 测试Token
print(f"\n[步骤1] 测试Token配置")
print(f"Token: {TOKEN[:20]}...")
print(f"Chat ID: {CHAT_ID}")

# 2. 测试getMe API
print(f"\n[步骤2] 测试getMe API（验证Token）")
url = f"https://api.telegram.org/bot{TOKEN}/getMe"
try:
    resp = requests.get(url, timeout=10)
    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"响应: {result}")

    if result.get('ok'):
        bot_info = result.get('result', {})
        print(f"✅ Token有效")
        print(f"   Bot名称: {bot_info.get('first_name')}")
        print(f"   Bot用户名: @{bot_info.get('username')}")
    else:
        print(f"❌ Token无效")
        sys.exit(1)
except Exception as e:
    print(f"❌ 请求失败: {e}")
    sys.exit(1)

# 3. 测试getUpdates API
print(f"\n[步骤3] 测试getUpdates API")
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
try:
    resp = requests.get(url, timeout=10)
    print(f"状态码: {resp.status_code}")
    result = resp.json()

    if result.get('ok'):
        updates = result.get('result', [])
        print(f"✅ 获取到 {len(updates)} 条更新")

        # 检查是否有来自Chat ID的消息
        for update in updates:
            if 'message' in update:
                chat = update['message'].get('chat', {})
                chat_id = chat.get('id')
                print(f"   Chat ID: {chat_id}, 类型: {chat.get('type')}")

                if str(chat_id) == CHAT_ID:
                    print(f"   ✅ 找到目标Chat ID的消息")
                    if 'text' in update['message']:
                        print(f"   最新消息: {update['message']['text']}")
    else:
        print(f"❌ 获取更新失败: {result}")
except Exception as e:
    print(f"❌ 请求失败: {e}")

# 4. 删除webhook
print(f"\n[步骤4] 删除webhook")
url = f"https://api.telegram.org/bot{TOKEN}/deleteWebhook"
try:
    resp = requests.post(url, timeout=10)
    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"响应: {result}")
    if result.get('ok'):
        print(f"✅ Webhook已删除")
except Exception as e:
    print(f"❌ 删除webhook失败: {e}")

# 5. 测试发送纯文本消息
print(f"\n[步骤5] 测试发送纯文本消息（无parse_mode）")
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
data = {
    'chat_id': CHAT_ID,
    'text': '🧪 测试消息1 - 纯文本\n时间: 2026-01-15 12:00:00',
    'disable_web_page_preview': True
}

try:
    resp = requests.post(url, json=data, timeout=10)
    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"响应: {result}")

    if resp.status_code == 200:
        print(f"✅ 纯文本消息发送成功")
    else:
        print(f"❌ 发送失败: {result}")
except Exception as e:
    print(f"❌ 请求失败: {e}")

# 6. 测试发送简单消息（无emoji）
print(f"\n[步骤6] 测试发送简单文本（无emoji）")
data = {
    'chat_id': CHAT_ID,
    'text': 'Test message 2 - Simple text without emoji',
}

try:
    resp = requests.post(url, json=data, timeout=10)
    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"响应: {result}")

    if resp.status_code == 200:
        print(f"✅ 简单文本发送成功")
    else:
        print(f"❌ 发送失败: {result}")
except Exception as e:
    print(f"❌ 请求失败: {e}")

print("\n" + "=" * 70)
print("调试完成")
print("=" * 70)
