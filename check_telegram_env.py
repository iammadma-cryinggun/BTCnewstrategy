# -*- coding: utf-8 -*-
"""
检查Telegram环境配置
"""
import sys
import os

print("=" * 70)
print("Telegram Environment Check")
print("=" * 70)

# Check 1: Python version
print("\n[Check 1] Python Version:")
print(f"  Version: {sys.version}")
print(f"  Executable: {sys.executable}")

# Check 2: pyTelegramBotAPI installation
print("\n[Check 2] pyTelegramBotAPI Installation:")
try:
    import telebot
    print(f"  [OK] telebot module found")
    print(f"  Version: {telebot.__version__ if hasattr(telebot, '__version__') else 'Unknown'}")
    print(f"  File: {telebot.__file__}")
except ImportError as e:
    print(f"  [FAIL] telebot module not found: {e}")
    print("  Please run: pip install pyTelegramBotAPI")
    exit(1)

# Check 3: Environment variables
print("\n[Check 3] Environment Variables:")
token = os.getenv('TELEGRAM_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')
enabled = os.getenv('TELEGRAM_ENABLED')

print(f"  TELEGRAM_TOKEN: {'{0:.20}...'.format(token) if token else '[NOT SET]'}")
print(f"  TELEGRAM_CHAT_ID: {chat_id if chat_id else '[NOT SET]'}")
print(f"  TELEGRAM_ENABLED: {enabled if enabled else '[NOT SET]'}")

if not token:
    print("  [WARN] TELEGRAM_TOKEN not set in environment")
if not chat_id:
    print("  [WARN] TELEGRAM_CHAT_ID not set in environment")

# Check 4: Test bot initialization
print("\n[Check 4] Bot Initialization:")
if token:
    try:
        bot = telebot.TeleBot(token)
        print("  [OK] Bot initialized successfully")
    except Exception as e:
        print(f"  [FAIL] Bot initialization failed: {e}")
else:
    print("  [SKIP] No token available")

# Check 5: Test message sending
print("\n[Check 5] Test Message Sending:")
if token and chat_id:
    try:
        bot = telebot.TeleBot(token)
        bot.send_message(chat_id, "[TEST] Environment check - OK")
        print("  [OK] Test message sent successfully")
    except Exception as e:
        print(f"  [FAIL] Test message failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  [SKIP] Token or chat_id not available")

print("\n" + "=" * 70)
print("Check Complete")
print("=" * 70)
