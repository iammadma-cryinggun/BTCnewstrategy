# -*- coding: utf-8 -*-
"""
测试实时V8.1系统
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from main_v81_orderflow import V81OrderFlowEnhanced
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("V8.1 实时系统测试")
print("=" * 70)

# 创建引擎实例
engine = V81OrderFlowEnhanced()

print("\n[步骤1] 获取订单流数据...")
order_flow_success = engine.fetch_order_flow_data()
print(f"结果: {'✓ 成功' if order_flow_success else '✗ 失败'}")

print("\n[步骤2] 获取期权数据...")
options_success = engine.fetch_options_data(force=True)
print(f"结果: {'✓ 成功' if options_success else '✗ 失败'}")

print("\n[步骤3] 执行完整的信号检查...")
print("-" * 70)

# 执行信号检查
engine.check_signals_enhanced()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

print("\n[总结]")
print(f"订单流数据: {'✓ 正常' if order_flow_success else '✗ 异常'}")
print(f"期权数据: {'✓ 正常' if options_success else '✗ 异常'}")
print(f"信号检查: 已执行（请查看上方日志）")
print(f"Telegram通知: 请查看上方日志或检查Telegram消息")
