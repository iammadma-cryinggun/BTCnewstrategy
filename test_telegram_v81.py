# -*- coding: utf-8 -*-
"""
测试V8.1 Telegram消息格式
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import telebot
from main_v80 import V80Config

config = V80Config()

# 模拟V8.1消息格式
def test_v81_message_format():
    """测试V8.1消息格式"""

    # 模拟数据
    signal_type = "LOW_OSCILLATION"
    enhanced_description = "低位震荡 | 期权: 支撑墙$90,000"
    final_confidence = 0.80
    base_confidence = 0.70
    options_boost = 0.10
    order_flow_boost = 0.0
    current_price = 90418.50
    tension = -0.502
    acceleration = 0.010
    dxy_fuel = 0.15
    direction = "LONG"
    stop_loss = 87706.95
    take_profit = 99460.35

    # 构建消息（使用修复后的格式）
    lines = []
    lines.append("🎯 V8.1 新交易信号（订单流增强版）")
    lines.append("")
    lines.append(f"📊 类型: {signal_type}")
    lines.append(f"📈 描述: {enhanced_description}")
    lines.append(f"🎯 置信度: {final_confidence:.1%} (基础: {base_confidence:.1%} + 期权: {options_boost:+.1%} + 订单流: {order_flow_boost:+.1%})")
    lines.append("")
    lines.append(f"💰 价格: ${current_price:,.2f}")
    lines.append(f"📊 张力: {tension:.3f} | 加速度: {acceleration:.3f} | DXY: {dxy_fuel:.3f}")

    # 期权数据
    options_success = True
    gamma_exposure = {'net_gamma_exposure': 1500000}
    max_pain = 90000

    if options_success:
        lines.append("")
        lines.append("📐 期权数据:")
        lines.append(f"  净Gamma: {gamma_exposure['net_gamma_exposure']:.0f}")
        lines.append(f"  最大痛点: ${max_pain:,.0f}")

    # 订单流数据
    order_flow_success = True
    cvd = {'trend': 'bullish', 'buy_ratio': 0.829}

    if order_flow_success:
        lines.append("")
        lines.append("📊 订单流数据:")
        lines.append(f"  CVD趋势: {cvd['trend']}")
        lines.append(f"  买入占比: {cvd['buy_ratio']:.1%}")

    lines.append("")
    lines.append(f"🚀 方向: {direction.upper()}")
    lines.append(f"💵 入场: ${current_price:,.2f}")
    lines.append(f"🛑 止损: ${stop_loss:,.2f} ({(stop_loss/current_price - 1)*100:+.2f}%)")
    lines.append(f"🎯 止盈: ${take_profit:,.2f} ({(take_profit/current_price - 1)*100:+.2f}%)")
    lines.append(f"📈 盈亏比: {(abs(take_profit - current_price) / abs(stop_loss - current_price)):.2f}")

    message = "\n".join(lines)

    print("=" * 70)
    print("V8.1 Telegram消息格式测试")
    print("=" * 70)
    print("\n生成的消息:")
    print("-" * 70)
    print(message)
    print("-" * 70)

    # 检查消息长度
    print(f"\n消息长度: {len(message)} 字符")
    if len(message) > 4096:
        print("   ⚠️ 警告: 消息超过4096字符限制!")
    else:
        print("   ✅ 消息长度正常")

    # 发送测试
    print("\n发送测试消息...")
    try:
        bot = telebot.TeleBot(config.telegram_token)
        result = bot.send_message(config.telegram_chat_id, message)
        print(f"   ✅ 发送成功! 消息ID: {result.message_id}")
        print("\n请检查Telegram是否收到格式正确的消息")
    except Exception as e:
        print(f"   ❌ 发送失败: {e}")

    print("=" * 70)


if __name__ == "__main__":
    test_v81_message_format()
