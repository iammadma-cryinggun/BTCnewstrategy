# -*- coding: utf-8 -*-
"""
V7.0.7 主程序 - 第二部分
包含：Telegram通知、主循环、命令处理
"""

import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ⭐ 北京时间（UTC+8）
BEIJING_TZ_OFFSET = timedelta(hours=8)


def get_beijing_time():
    """获取当前北京时间"""
    return datetime.utcnow() + BEIJING_TZ_OFFSET


# ==================== [Telegram通知模块] ====================
class TelegramNotifier:
    """Telegram通知和交互模块"""

    def __init__(self, config):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.telegram_token}"

    def send_message(self, message, parse_mode='Markdown'):
        """发送Telegram消息"""
        if not self.config.telegram_enabled:
            return

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': parse_mode
            }

            resp = requests.post(url, json=data, timeout=10)

            if resp.status_code == 200:
                logger.info(f"[Telegram] 消息已发送")
            else:
                logger.warning(f"[Telegram] 发送失败: HTTP {resp.status_code}")

        except Exception as e:
            logger.error(f"[Telegram] 发送消息异常: {e}")

    def notify_signal(self, signal_type, confidence, description, price, tension, acceleration):
        """通知新信号"""
        # ⭐ 使用北京时间
        now_beijing = get_beijing_time()
        message = f"""
🎯 *V7.0.7新信号*

📊 *信号类型*: {signal_type}
📈 *置信度*: {confidence:.2f}
💡 *描述*: {description}
💰 *当前价格*: ${price:.2f}
📐 *张力*: {tension:.3f}
🚀 *加速度*: {acceleration:.3f}

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""
        self.send_message(message)

    def notify_entry(self, direction, price, signal_type, confidence, tp, sl):
        """通知开仓"""
        # ⭐ 使用北京时间
        now_beijing = get_beijing_time()
        emoji = "📈" if direction == 'long' else "📉"
        message = f"""
{emoji} *V7.0.7开仓*

📍 *方向*: {direction.upper()}
💰 *入场价*: ${price:.2f}
🎯 *信号*: {signal_type}
📊 *置信度*: {confidence:.2f}
🎯 *止盈*: ${tp:.2f}
🛑 *止损*: ${sl:.2f}

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""
        self.send_message(message)

    def notify_exit(self, direction, entry_price, exit_price, pnl_pct, reason):
        """通知平仓"""
        # ⭐ 使用北京时间
        now_beijing = get_beijing_time()
        emoji = "✅" if pnl_pct > 0 else "❌"
        message = f"""
{emoji} *V7.0.7平仓*

📍 *方向*: {direction.upper()}
💰 *入场*: ${entry_price:.2f}
💵 *出场*: ${exit_price:.2f}
📊 *盈亏*: {pnl_pct:+.2f}%
🎯 *原因*: {reason}

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""
        self.send_message(message)

    def notify_status(self):
        """通知系统状态"""
        # ⭐ 使用北京时间
        now_beijing = get_beijing_time()
        if self.config.has_position:
            # 计算持仓时长（假设entry_time也是北京时间）
            if self.config.entry_time:
                hold_delta = now_beijing - self.config.entry_time
                hold_hours = hold_delta.total_seconds() / 3600
                hold_time_str = f"{hold_hours:.1f}小时"
            else:
                hold_time_str = "未知"

            message = f"""
📊 *V7.0.7持仓状态*

📍 *方向*: {self.config.position_type.upper()}
💰 *入场价*: ${self.config.entry_price:.2f}
🎯 *止盈*: ${self.config.take_profit_price:.2f}
🛑 *止损*: ${self.config.stop_loss_price:.2f}
⏱ *持仓时长*: {hold_time_str}
📊 *入场置信度*: {self.config.entry_confidence:.2f}

📈 *总交易*: {self.config.total_trades}
✅ *盈利*: {self.config.winning_trades}
❌ *亏损*: {self.config.losing_trades}
💵 *总盈亏*: {self.config.total_pnl:.2f}%

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""
        else:
            message = f"""
📊 *V7.0.7系统状态*

⭕ *当前状态*: 空仓
📈 *总交易*: {self.config.total_trades}
✅ *盈利*: {self.config.winning_trades}
❌ *亏损*: {self.config.losing_trades}
💵 *总盈亏*: {self.config.total_pnl:.2f}%

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""
        self.send_message(message)
