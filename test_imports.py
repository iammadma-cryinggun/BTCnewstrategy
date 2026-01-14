# -*- coding: utf-8 -*-
"""
Test all imports for main_v707.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("Testing Imports for main_v707.py")
print("=" * 70)

# Test 1: Import from v707_trader_main
print("\n[Test 1] Import from v707_trader_main...")
try:
    from v707_trader_main import (
        V707TraderConfig,
        DataFetcher,
        PhysicsSignalCalculator,
        V705EntryFilter,
        V707ZigZagExitManager
    )
    print("[OK] All imports from v707_trader_main successful")
except ImportError as e:
    print(f"[FAIL] Import from v707_trader_main failed: {e}")
    sys.exit(1)

# Test 2: Import TelegramNotifier from v707_trader_part2
print("\n[Test 2] Import TelegramNotifier from v707_trader_part2...")
try:
    from v707_trader_part2 import TelegramNotifier
    print("[OK] TelegramNotifier imported successfully")
except ImportError as e:
    print(f"[FAIL] TelegramNotifier import failed: {e}")
    sys.exit(1)

# Test 3: Import start_telegram_listener
print("\n[Test 3] Import start_telegram_listener...")
try:
    from v707_telegram_handler import start_telegram_listener
    print("[OK] start_telegram_listener imported successfully")
except ImportError as e:
    print(f"[FAIL] start_telegram_listener import failed: {e}")
    sys.exit(1)

# Test 4: Import V707TradingEngine
print("\n[Test 4] Import V707TradingEngine...")
try:
    from main_v707 import V707TradingEngine
    print("[OK] V707TradingEngine imported successfully")
except ImportError as e:
    print(f"[FAIL] V707TradingEngine import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("All imports successful!")
print("=" * 70)
