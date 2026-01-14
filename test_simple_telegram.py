# -*- coding: utf-8 -*-
"""
简单Telegram测试 - 逐步测试
"""
import os
import telebot

# 配置
TOKEN = "8505180201:AAGOSkhXHRu77OlRMu0PZCbKtYMEr1tRGAk"
CHAT_ID = "838429342"

print("=" * 70)
print("Telegram Simple Test")
print("=" * 70)

# Test 1: Initialize bot
print("\n[Test 1] Initialize TeleBot...")
try:
    bot = telebot.TeleBot(TOKEN)
    print("[OK] TeleBot initialized successfully")
except Exception as e:
    print(f"[FAIL] TeleBot init failed: {e}")
    exit(1)

# Test 2: Send simple English text
print("\n[Test 2] Send simple English text...")
try:
    bot.send_message(CHAT_ID, "Test message 1 - Simple English")
    print("[OK] Simple English text sent successfully")
except Exception as e:
    print(f"[FAIL] Simple English text failed: {e}")

# Test 3: Send text with newlines
print("\n[Test 3] Send text with newlines...")
try:
    bot.send_message(CHAT_ID, "Line 1\nLine 2\nLine 3")
    print("[OK] Text with newlines sent successfully")
except Exception as e:
    print(f"[FAIL] Text with newlines failed: {e}")

# Test 4: Send text with emoji
print("\n[Test 4] Send text with emoji...")
try:
    bot.send_message(CHAT_ID, "Test with emoji\nSuccess\nError")
    print("[OK] Text with emoji sent successfully")
except Exception as e:
    print(f"[FAIL] Text with emoji failed: {e}")

# Test 5: Send Chinese text
print("\n[Test 5] Send Chinese text...")
try:
    bot.send_message(CHAT_ID, "Test Chinese message")
    print("[OK] Chinese text sent successfully")
except Exception as e:
    print(f"[FAIL] Chinese text failed: {e}")

# Test 6: Send full format message (simulate actual notification)
print("\n[Test 6] Send full format message...")
try:
    message = """V7.0.7 New Signal

Type: TEST
Confidence: 0.95
Description: Test message
Price: $100000.00
Tension: 0.500
Acceleration: 0.100

Time: 2026-01-15 12:00:00 (Beijing)
"""
    bot.send_message(CHAT_ID, message)
    print("[OK] Full format message sent successfully")
except Exception as e:
    print(f"[FAIL] Full format message failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Try integer chat_id
print("\n[Test 7] Use integer chat_id...")
try:
    bot.send_message(int(CHAT_ID), "Test with integer chat_id")
    print("[OK] Integer chat_id sent successfully")
except Exception as e:
    print(f"[FAIL] Integer chat_id failed: {e}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
