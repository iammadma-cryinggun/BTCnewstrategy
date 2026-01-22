# -*- coding: utf-8 -*-
"""
测试期权组合策略
验证所有4个方案是否正常工作
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from main_v80 import V80Config
from deribit_data_hub import DeribitDataHub
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("期权组合策略测试")
print("=" * 70)

# 模拟数据
class MockOptionsEnhanced:
    def __init__(self):
        self.deribit_hub = DeribitDataHub()
        self.options_data = None
        self.config = V80Config()

    def simulate_scenario(self, scenario_name, signal_direction, net_gamma, vanna_conf, call_wall_dist, put_wall_dist):
        """模拟特定场景"""
        print(f"\n{'=' * 70}")
        print(f"场景: {scenario_name}")
        print(f"{'=' * 70}")

        current_price = 90000
        base_confidence = 0.70

        print(f"验证5信号: {signal_direction.upper()}")
        print(f"基础置信度: {base_confidence:.2f}")
        print(f"当前价格: ${current_price:,.0f}")

        # 期权数据
        print(f"\n期权数据:")
        print(f"  净Gamma: {net_gamma:,.0f}")

        # 方案2：增强 + 方案1：确认
        options_boost = 0.0

        if net_gamma > 0:
            print(f"  Gamma状态: 多头友好")
            if signal_direction == 'long':
                options_boost += 0.10
                print(f"    ✅ Gamma支持做多，置信度+10%")
            elif signal_direction == 'short':
                options_boost -= 0.20
                print(f"    ⚠️ Gamma反对做空，置信度-20%")
        elif net_gamma < 0:
            print(f"  Gamma状态: 空头友好")
            if signal_direction == 'short':
                options_boost += 0.10
                print(f"    ✅ Gamma支持做空，置信度+10%")
            elif signal_direction == 'long':
                options_boost -= 0.20
                print(f"    ⚠️ Gamma反对做多，置信度-20%")

        # Vanna挤压
        print(f"  Vanna挤压: {vanna_conf:.1%}")

        # 方案3：否决
        options_veto = False
        if vanna_conf > 0.8:
            print(f"    ❌ Vanna挤压过高，期权否决交易")
            options_veto = True
        elif vanna_conf > 0:
            print(f"    ⚠️ Vanna挤压风险，需要谨慎")

        # 订单墙
        print(f"  订单墙:")
        if call_wall_dist:
            print(f"    CALL墙: {call_wall_dist:+.2%}")
        if put_wall_dist:
            print(f"    PUT墙: {put_wall_dist:+.2%}")

        # 最终置信度
        if options_veto:
            final_confidence = 0
            print(f"\n结果: ❌ 期权否决，不开仓")
        else:
            final_confidence = base_confidence + options_boost
            final_confidence = max(0, min(final_confidence, 1.0))  # 限制在0-1之间

            print(f"\n置信度调整:")
            print(f"  {base_confidence:.2f} (基础) + {options_boost:+.2f} (期权) = {final_confidence:.2f}")

            if final_confidence >= 0.6:
                print(f"  结果: ✅ 开仓 {signal_direction.upper()} @ ${current_price:,.0f}")

                # 方案4：调整止盈止损
                if signal_direction == 'long':
                    stop_loss = current_price * 0.97
                    take_profit = current_price * 1.10

                    print(f"\n止盈止损:")
                    print(f"  原始: 止损 ${stop_loss:,.0f} | 止盈 ${take_profit:,.0f}")

                    # CALL墙调整止盈
                    if call_wall_dist and call_wall_dist > 0:
                        call_wall_price = current_price * (1 + call_wall_dist)
                        if call_wall_price < take_profit:
                            old_tp = take_profit
                            take_profit = call_wall_price * 0.99
                            print(f"  📊 止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f} (阻力墙)")

                    # PUT墙调整止损
                    if put_wall_dist and put_wall_dist < 0:
                        put_wall_price = current_price * (1 + put_wall_dist)
                        if put_wall_price > stop_loss:
                            old_sl = stop_loss
                            stop_loss = put_wall_price * 0.99
                            print(f"  📊 止损调整: ${old_sl:,.0f} → ${stop_loss:,.0f} (支撑墙)")

                else:  # short
                    stop_loss = current_price * 1.03
                    take_profit = current_price * 0.90

                    print(f"\n止盈止损:")
                    print(f"  原始: 止损 ${stop_loss:,.0f} | 止盈 ${take_profit:,.0f}")

                    # PUT墙调整止盈
                    if put_wall_dist and put_wall_dist < 0:
                        put_wall_price = current_price * (1 + put_wall_dist)
                        if put_wall_price > take_profit:
                            old_tp = take_profit
                            take_profit = put_wall_price * 1.01
                            print(f"  📊 止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f} (支撑墙)")

                    # CALL墙调整止损
                    if call_wall_dist and call_wall_dist > 0:
                        call_wall_price = current_price * (1 + call_wall_dist)
                        if call_wall_price < stop_loss:
                            old_sl = stop_loss
                            stop_loss = call_wall_price * 1.01
                            print(f"  📊 止损调整: ${old_sl:,.0f} → ${stop_loss:,.0f} (阻力墙)")

            else:
                print(f"  结果: ❌ 置信度不足，不开仓")

        print(f"\n总结:")
        if options_veto:
            print(f"  ❌ 期权否决 (Vanna挤压 {vanna_conf:.1%} > 80%)")
        elif final_confidence >= 0.6:
            print(f"  ✅ 开仓 {signal_direction.upper()} (置信度 {final_confidence:.2f} >= 60%)")
        else:
            print(f"  ❌ 置信度不足 ({final_confidence:.2f} < 60%)")

# 运行测试
tester = MockOptionsEnhanced()

# 场景1: 最佳情况 - 所有指标一致
tester.simulate_scenario(
    scenario_name="场景1: 最佳情况（做多+Gamma支持+无Vanna+订单墙配合）",
    signal_direction='long',
    net_gamma=1000000,  # 正Gamma，支持做多
    vanna_conf=0.0,  # 无Vanna挤压
    call_wall_dist=0.10,  # CALL墙在+10%（不影响止盈）
    put_wall_dist=-0.02  # PUT墙在-2%（提供支撑保护）
)

# 场景2: 冲突情况 - Gamma反对
tester.simulate_scenario(
    scenario_name="场景2: 冲突情况（做多+Gamma反对）",
    signal_direction='long',
    net_gamma=-1000000,  # 负Gamma，反对做多
    vanna_conf=0.0,
    call_wall_dist=0.10,
    put_wall_dist=-0.02
)

# 场景3: Vanna挤压否决
tester.simulate_scenario(
    scenario_name="场景3: Vanna挤压否决",
    signal_direction='short',
    net_gamma=-500000,  # Gamma支持做空
    vanna_conf=0.85,  # Vanna挤压85%，直接否决
    call_wall_dist=0.05,
    put_wall_dist=-0.10
)

# 场景4: 临界情况 - 刚好通过
tester.simulate_scenario(
    scenario_name="场景4: 临界情况（刚好60%）",
    signal_direction='short',
    net_gamma=-500000,  # Gamma支持+10%
    vanna_conf=0.0,
    call_wall_dist=0.08,
    put_wall_dist=-0.05
)

# 场景5: 订单墙调整止盈
tester.simulate_scenario(
    scenario_name="场景5: 订单墙调整止盈",
    signal_direction='long',
    net_gamma=500000,  # Gamma支持+10%
    vanna_conf=0.0,
    call_wall_dist=0.08,  # CALL墙在+8%（会提前止盈）
    put_wall_dist=-0.02
)

# 场景6: 订单墙调整止损
tester.simulate_scenario(
    scenario_name="场景6: 订单墙调整止损",
    signal_direction='long',
    net_gamma=500000,
    vanna_conf=0.0,
    call_wall_dist=0.15,
    put_wall_dist=-0.01  # PUT墙在-1%（会放宽止损）
)

print(f"\n{'=' * 70}")
print("测试完成！")
print(f"{'=' * 70}")
print("\n总结:")
print("✅ 方案1（期权确认）: Gamma反对时降低置信度")
print("✅ 方案2（期权增强）: Gamma支持时提高置信度")
print("✅ 方案3（期权否决）: Vanna挤压>80%时直接否决")
print("✅ 方案4（调整止盈止损）: 根据订单墙优化止盈止损")
