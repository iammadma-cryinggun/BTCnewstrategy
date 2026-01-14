# -*- coding: utf-8 -*-
"""
V7.0.7 Telegram命令处理器 - 完整交互支持
===========================================

⭐ 使用telebot库（参考SOL系统实现）

支持的命令：
- /start : 启动机器人并显示帮助
- /status : 查看当前持仓状态
- /signals : 查看最近的信号历史
- /trades : 查看交易历史
- /clear : 手动平仓（⚠️ 谨慎使用）
- /config : 查看当前配置
- /help : 显示帮助信息
"""

import telebot
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import threading

logger = logging.getLogger(__name__)

# ⭐ 北京时间（UTC+8）
BEIJING_TZ_OFFSET = timedelta(hours=8)


def get_beijing_time():
    """获取当前北京时间"""
    return datetime.utcnow() + BEIJING_TZ_OFFSET


class TelegramCommandHandler:
    """Telegram命令处理器（⭐ 使用telebot库）"""

    def __init__(self, config, trading_engine):
        self.config = config
        self.engine = trading_engine
        self.token = config.telegram_token
        self.chat_id = config.telegram_chat_id
        self.enabled = config.telegram_enabled

        # ⭐ 使用telebot库（参考SOL系统）
        if self.enabled and self.token:
            try:
                self.bot = telebot.TeleBot(self.token)
                logger.info("[Telegram] CommandHandler TeleBot初始化成功")
                self._register_handlers()
            except Exception as e:
                logger.error(f"[Telegram] CommandHandler TeleBot初始化失败: {e}")
                self.bot = None
                self.enabled = False
        else:
            self.bot = None

    def _register_handlers(self):
        """注册Telegram消息处理器"""
        if not self.bot:
            return

        # 导入types
        from telebot import types

        @self.bot.message_handler(commands=['start', 'help'])
        def send_help(message):
            if message.chat.id != int(self.chat_id):
                return
            help_text = """
🤖 V7.0.7交易系统 - 交互式控制

可用命令：
/status - 查看当前持仓状态
/signals - 查看最近6个信号
/trades - 查看交易历史
/clear - 手动平仓
/config - 查看系统配置

V7.0.7特性：
- V7.0.5入场过滤器（量能/趋势/动能）
- V7.0.7 ZigZag动态止盈止损
- 完美过滤1月13-14日错误信号
            """
            try:
                self.bot.reply_to(message, help_text)
            except Exception as e:
                logger.error(f"[Telegram] 发送帮助失败: {e}")

        @self.bot.message_handler(commands=['status'])
        def send_status(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                now_beijing = get_beijing_time()
                if self.config.has_position:
                    hold_time = 0
                    if self.config.entry_time:
                        hold_time = (now_beijing - self.config.entry_time).total_seconds() / 3600

                    current_price = 0
                    try:
                        df = self.engine.fetcher.fetch_btc_data(interval='4h', limit=5)
                        if df is not None:
                            current_price = df.iloc[-1]['close']
                    except:
                        pass

                    if current_price > 0:
                        if self.config.position_type == 'long':
                            current_pnl_pct = (current_price - self.config.entry_price) / self.config.entry_price * 100
                        else:
                            current_pnl_pct = (self.config.entry_price - current_price) / self.config.entry_price * 100
                    else:
                        current_pnl_pct = 0.0

                    pnl_emoji = "🟢" if current_pnl_pct > 0 else "🔴"

                    status_text = f"""📊 V7.0.7持仓状态

📍 方向: {'📈 做多' if self.config.position_type == 'long' else '📉 做空'}
💰 入场价: ${self.config.entry_price:.2f}
💵 当前价: ${current_price:.2f}
{pnl_emoji} 盈亏: {current_pnl_pct:+.2f}%
⏱ 持仓时长: {hold_time:.1f}小时
📊 入场置信度: {self.config.entry_confidence:.2f}

📈 总交易: {self.config.total_trades}
✅ 盈利: {self.config.winning_trades}
❌ 亏损: {self.config.losing_trades}
💵 总盈亏: {self.config.total_pnl:.2f}%
"""
                else:
                    status_text = f"""📊 V7.0.7系统状态

⭕ 当前状态: 空仓
📈 总交易: {self.config.total_trades}
✅ 盈利: {self.config.winning_trades}
❌ 亏损: {self.config.losing_trades}
💵 总盈亏: {self.config.total_pnl:.2f}%

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""

                self.bot.reply_to(message, status_text)
            except Exception as e:
                logger.error(f"[Telegram] 发送状态失败: {e}")

        @self.bot.message_handler(commands=['signals'])
        def send_signals(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                if not self.config.signal_history or len(self.config.signal_history) == 0:
                    message_text = "📡 信号历史\n\n暂无信号记录"
                else:
                    recent_signals = self.config.signal_history[-6:]
                    message_text = "📡 最近6个信号\n\n"

                    for i, signal in enumerate(reversed(recent_signals), 1):
                        time_str = signal.get('time', 'N/A')
                        sig_type = signal.get('type', 'N/A')
                        price = signal.get('price', 0)
                        conf = signal.get('confidence', 0)
                        desc = signal.get('description', '')
                        traded = signal.get('traded', True)
                        filtered = signal.get('filtered', False)
                        filter_reason = signal.get('filter_reason', '')

                        if traded:
                            status_emoji = "✅"
                            status_text = "已交易"
                        elif filtered:
                            status_emoji = "🚫"
                            status_text = f"被过滤: {filter_reason}"
                        else:
                            status_emoji = "⏳"
                            status_text = "等待处理"

                        message_text += f"{i}. {sig_type}\n"
                        message_text += f"   {status_emoji} 状态: {status_text}\n"
                        message_text += f"   🕐 时间: {time_str}\n"
                        message_text += f"   💰 价格: ${price:.2f}\n"
                        message_text += f"   📊 置信度: {conf:.2f}\n\n"

                self.bot.reply_to(message, message_text)
            except Exception as e:
                logger.error(f"[Telegram] 发送信号历史失败: {e}")

        @self.bot.message_handler(commands=['trades'])
        def send_trades(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                if not self.config.position_history or len(self.config.position_history) == 0:
                    message_text = "📝 交易历史\n\n暂无交易记录"
                else:
                    recent_trades = self.config.position_history[-5:]
                    message_text = "📝 最近交易历史\n\n"

                    for i, trade in enumerate(reversed(recent_trades), 1):
                        entry_time = trade.get('entry_time', 'N/A')
                        direction = trade.get('direction', 'N/A')
                        entry_price = trade.get('entry_price', 0)
                        exit_price = trade.get('exit_price', 0)
                        pnl_pct = trade.get('pnl_pct', 0)
                        reason = trade.get('reason', 'N/A')

                        direction_emoji = "📈" if direction == 'long' else "📉"
                        pnl_emoji = "🎉" if pnl_pct > 0 else "🛑"

                        message_text += f"{i}. {direction_emoji} {direction.upper()}\n"
                        message_text += f"   入场: {entry_time}\n"
                        message_text += f"   价格: ${entry_price:.2f} → ${exit_price:.2f}\n"
                        message_text += f"   盈亏: {pnl_emoji} {pnl_pct:+.2f}%\n"
                        message_text += f"   原因: {reason}\n\n"

                self.bot.reply_to(message, message_text)
            except Exception as e:
                logger.error(f"[Telegram] 发送交易历史失败: {e}")

        @self.bot.message_handler(commands=['clear'])
        def handle_clear(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                if not self.config.has_position:
                    self.bot.reply_to(message, "❌ 当前无持仓，无需平仓")
                    return

                # 获取当前价格
                df = self.engine.fetcher.fetch_btc_data(interval='4h', limit=5)
                if df is not None:
                    current_price = df.iloc[-1]['close']

                    # 计算当前盈亏
                    if self.config.position_type == 'long':
                        pnl_pct = (current_price - self.config.entry_price) / self.config.entry_price * 100
                    else:
                        pnl_pct = (self.config.entry_price - current_price) / self.config.entry_price * 100

                    # 执行平仓
                    direction_emoji = "📈" if self.config.position_type == 'long' else "📉"
                    pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"

                    # 记录交易历史
                    trade_record = {
                        'entry_time': self.config.entry_time.strftime('%Y-%m-%d %H:%M:%S') if self.config.entry_time else 'N/A',
                        'direction': self.config.position_type,
                        'entry_price': self.config.entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'reason': '手动平仓(/clear命令)',
                        'signal_type': self.config.entry_signal_type,
                        'confidence': self.config.entry_confidence,
                        'take_profit': self.config.take_profit_price,
                        'stop_loss': self.config.stop_loss_price
                    }
                    self.config.position_history.append(trade_record)

                    # 只保留最近20笔交易
                    if len(self.config.position_history) > 20:
                        self.config.position_history = self.config.position_history[-20:]

                    # 更新统计
                    self.config.total_trades += 1
                    if pnl_pct > 0:
                        self.config.winning_trades += 1
                    else:
                        self.config.losing_trades += 1
                    self.config.total_pnl += pnl_pct

                    # 保存状态
                    self.config.save_state()

                    # 发送平仓通知
                    now_beijing = get_beijing_time()
                    message_text = f"""✅ V7.0.7手动平仓成功

{direction_emoji} {self.config.position_type.upper()}
💰 开仓价: ${self.config.entry_price:.2f}
💵 出场价: ${current_price:.2f}
{pnl_emoji} 盈亏: {pnl_pct:+.2f}%
⚠️ 原因: 手动平仓(/clear命令)

⏰ {now_beijing.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)
"""

                    # ⭐ 清除持仓状态
                    self.config.has_position = False
                    self.config.position_type = None
                    self.config.entry_price = None
                    self.config.entry_time = None
                    self.config.take_profit_price = None
                    self.config.stop_loss_price = None

                    # 保存状态
                    self.config.save_state()

                    logger.warning(f"[命令] 用户手动平仓: {self.config.position_type.upper()} @ ${current_price:.2f}, 盈亏: {pnl_pct:+.2f}%")

                    self.bot.reply_to(message, message_text)
                else:
                    self.bot.reply_to(message, "❌ 获取当前价格失败，无法平仓")
            except Exception as e:
                logger.error(f"[Telegram] 手动平仓失败: {e}")
                self.bot.reply_to(message, f"❌ 手动平仓失败: {str(e)}")

        @self.bot.message_handler(commands=['config'])
        def send_config(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                config_text = f"""⚙️ V7.0.7系统配置

V7.0.5过滤器参数:
- BULLISH量能阈值: {self.config.BULLISH_VOLUME_THRESHOLD}
- HIGH_OSC EMA阈值: {self.config.HIGH_OSC_EMA_THRESHOLD*100:.0f}%
- HIGH_OSC量能阈值: {self.config.HIGH_OSC_VOLUME_THRESHOLD}
- BEARISH EMA阈值: {self.config.BEARISH_EMA_THRESHOLD*100:.0f}%

V7.0.7 ZigZag参数:
- ZigZag深度: {self.config.ZIGZAG_DEPTH}
- ZigZag偏差: {self.config.ZIGZAG_DEVIATION}%
- 最大持仓周期: {self.config.MAX_HOLD_PERIODS}周期（7天）

交易参数:
- 基础仓位: {self.config.BASE_POSITION_SIZE*100:.1f}%

运行配置:
- 信号检测: 北京时间4小时K线收盘
- 持仓检查: 每1小时
- Telegram通知: {'✅' if self.enabled else '❌'}
"""
                self.bot.reply_to(message, config_text)
            except Exception as e:
                logger.error(f"[Telegram] 发送配置失败: {e}")

        logger.info("[Telegram] 消息处理器已注册")


def start_telegram_listener(config, trading_engine):
    """启动Telegram监听器（独立线程）- ⭐ 使用telebot库"""

    handler = TelegramCommandHandler(config, trading_engine)

    if not handler.enabled or not handler.bot:
        logger.warning("[Telegram] 未启用或初始化失败，跳过监听器启动")
        return

    logger.info("[Telegram] 启动命令监听器...")
    logger.info(f"[Telegram] telegram_enabled={config.telegram_enabled}")
    logger.info(f"[Telegram] chat_id={config.telegram_chat_id}")

    # ⭐ 使用telebot的polling模式（参考SOL系统）
    while True:
        try:
            logger.info("[Telegram] 轮询启动...")
            handler.bot.polling(non_stop=False, interval=1, timeout=60, long_polling_timeout=20)
        except Exception as e:
            logger.error(f"[Telegram] 轮询异常: {e}", exc_info=True)
            logger.info("[Telegram] 5秒后重新启动...")
            import time
            time.sleep(5)


if __name__ == "__main__":
    # 测试代码
    class TestConfig:
        telegram_token = "8505180201:AAGOSkhXHRu77OlRMu0PZCbKtYMEr1tRGAk"
        telegram_chat_id = "838429342"
        telegram_enabled = True
        has_position = False
        signal_history = []
        position_history = []
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        BULLISH_VOLUME_THRESHOLD = 0.95
        HIGH_OSC_EMA_THRESHOLD = 0.02
        HIGH_OSC_VOLUME_THRESHOLD = 1.1
        BEARISH_EMA_THRESHOLD = -0.05
        ZIGZAG_DEPTH = 12
        ZIGZAG_DEVIATION = 5
        MAX_HOLD_PERIODS = 42
        BASE_POSITION_SIZE = 0.50

    class TestEngine:
        def __init__(self):
            self.fetcher = None

        def fetch_btc_data(self, interval='4h', limit=5):
            return None

    config = TestConfig()
    engine = TestEngine()

    print("测试Telegram连接...")
    handler = TelegramCommandHandler(config, engine)

    if handler.bot:
        print("✅ TeleBot初始化成功")
        print("开始监听命令...（按Ctrl+C停止）")
        start_telegram_listener(config, engine)
    else:
        print("❌ TeleBot初始化失败")
