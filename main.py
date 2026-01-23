# -*- coding: utf-8 -*-
"""
================================================================================
V8.0 验证5智能预警系统 - 主程序
================================================================================
基于验证5逻辑（FFT + Hilbert + 二阶差分）

使用方法：
1. 配置.env文件中的TELEGRAM_TOKEN和TELEGRAM_CHAT_ID
2. 运行：python main_v80.py

特点：
- 每4小时检查信号（北京时间 0:00, 4:00, 8:00, 12:00, 16:00, 20:00）
- 每小时检查持仓
- Telegram实时通知
- V8.0反向策略（系统看空我做多，系统看涨我做空）

================================================================================
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import requests
from io import StringIO
import telebot
import pickle

# ==================== 配置 ====================

class V80Config:
    """V8.0配置管理"""

    def __init__(self):
        # Telegram配置
        self.telegram_token = os.environ.get('TELEGRAM_TOKEN', '8505180201:AAGOSkhXHRu77OlRMu0PZCbKtYMEr1tRGAk')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '838429342')
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)

        # 验证5参数
        self.TENSION_THRESHOLD = 0.35
        self.ACCEL_THRESHOLD = 0.02
        self.OSCILLATION_BAND = 0.5
        self.CONFIDENCE_THRESHOLD = 0.6

        # 风险控制
        self.STOP_LOSS_PCT = 0.03
        self.TAKE_PROFIT_PCT = 0.10
        self.RISK_PER_TRADE = 0.02
        self.ACCOUNT_BALANCE = 10000

        # 持仓状态
        self.has_position = False
        self.position_type = None  # 'long' or 'short'
        self.entry_price = None
        self.entry_time = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.entry_signal_type = None
        self.entry_confidence = None

        # 历史记录
        self.signal_history = []
        self.position_history = []

        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # 状态文件
        self.state_file = 'v80_state.pkl'

    def save_state(self):
        """保存状态到文件"""
        state = {
            'has_position': self.has_position,
            'position_type': self.position_type,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'entry_signal_type': self.entry_signal_type,
            'entry_confidence': self.entry_confidence,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'signal_history': self.signal_history,
            'position_history': self.position_history
        }
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logging.error(f"保存状态失败: {e}")

    def load_state(self):
        """从文件加载状态"""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)

            self.has_position = state.get('has_position', False)
            self.position_type = state.get('position_type')
            self.entry_price = state.get('entry_price')
            self.entry_time = state.get('entry_time')
            self.stop_loss_price = state.get('stop_loss_price')
            self.take_profit_price = state.get('take_profit_price')
            self.entry_signal_type = state.get('entry_signal_type')
            self.entry_confidence = state.get('entry_confidence')
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.losing_trades = state.get('losing_trades', 0)
            self.total_pnl = state.get('total_pnl', 0.0)
            self.signal_history = state.get('signal_history', [])
            self.position_history = state.get('position_history', [])

        except Exception as e:
            logging.error(f"加载状态失败: {e}")


# ==================== 数据获取 ====================

class DataFetcher:
    """数据获取器"""

    def __init__(self):
        self.binance_url = "https://api.binance.com/api/v3/klines"

    def fetch_btc_data(self, interval='4h', limit=300):
        """获取BTC K线数据"""
        try:
            params = {'symbol': 'BTCUSDT', 'interval': interval, 'limit': limit}
            response = requests.get(self.binance_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            logging.error(f"[ERROR] BTC数据获取失败: {e}")
            return None

    def fetch_dxy_data(self, days_back=30):
        """获取DXY美元指数数据"""
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
            response = requests.get(url, timeout=15)

            if response.status_code != 200:
                return None

            dxy_df = pd.read_csv(StringIO(response.text))
            dxy_df['observation_date'] = pd.to_datetime(dxy_df['observation_date'])
            dxy_df.set_index('observation_date', inplace=True)
            dxy_df.rename(columns={'DTWEXBGS': 'Close'}, inplace=True)
            dxy_df = dxy_df.dropna()
            dxy_df['Close'] = pd.to_numeric(dxy_df['Close'], errors='coerce')

            cutoff_date = datetime.now() - timedelta(days=days_back)
            dxy_df = dxy_df[dxy_df.index >= cutoff_date]

            return dxy_df

        except Exception as e:
            logging.warning(f"[WARNING] DXY数据获取失败: {e}")
            return None


# ==================== 验证5计算引擎 ====================

def calculate_tension_acceleration_verification5(prices):
    """计算张力和加速度（验证5逻辑）"""
    if len(prices) < 3:
        return None, None

    try:
        # 步骤1：去趋势
        d_prices = detrend(prices)

        # 步骤2：FFT滤波（保留前8个频率分量）
        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real

        # 步骤3：Hilbert变换
        analytic = hilbert(filtered)
        tension = np.imag(analytic)

        # 步骤4：标准化
        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension

        # 步骤5：计算加速度（二阶差分）
        if len(norm_tension) >= 3:
            current_tension = norm_tension[-1]
            prev_tension = norm_tension[-2]
            prev2_tension = norm_tension[-3]

            velocity = current_tension - prev_tension
            acceleration = velocity - (prev_tension - prev2_tension)
        else:
            acceleration = 0.0

        return float(norm_tension[-1]), float(acceleration)

    except Exception as e:
        logging.error(f"[ERROR] 物理指标计算失败: {e}")
        return None, None


def calculate_dxy_fuel(dxy_history):
    """计算DXY燃料"""
    if len(dxy_history) < 3:
        return 0.0

    try:
        closes = np.array(dxy_history)
        change_1 = (closes[-1] - closes[-2]) / closes[-2]
        change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1

        acceleration = change_1 - change_2
        fuel = -acceleration * 100

        return float(fuel)

    except Exception as e:
        logging.error(f"[ERROR] DXY燃料计算失败: {e}")
        return 0.0


def classify_market_state(tension, acceleration, dxy_fuel=0.0):
    """分类市场状态（验证5逻辑）"""

    TENSION_THRESHOLD = 0.35
    ACCEL_THRESHOLD = 0.02
    OSCILLATION_BAND = 0.5

    # 1. BEARISH_SINGULARITY
    if tension > TENSION_THRESHOLD and acceleration < -ACCEL_THRESHOLD:
        if dxy_fuel > 0.1:
            return "BEARISH_SINGULARITY", "强奇点看空 (宏观失速)", 0.9
        else:
            return "BEARISH_SINGULARITY", "奇点看空 (动力失速)", 0.7

    # 2. BULLISH_SINGULARITY
    if tension < -TENSION_THRESHOLD and acceleration > ACCEL_THRESHOLD:
        if dxy_fuel > 0.2:
            return "BULLISH_SINGULARITY", "超强奇点看涨 (燃料爆炸)", 0.95
        elif dxy_fuel > 0:
            return "BULLISH_SINGULARITY", "强奇点看涨 (动力回归)", 0.8
        else:
            return "BULLISH_SINGULARITY", "奇点看涨 (弹性释放)", 0.6

    # 3. OSCILLATION
    if abs(tension) < OSCILLATION_BAND and abs(acceleration) < ACCEL_THRESHOLD:
        return "OSCILLATION", "系统平衡 (震荡收敛)", 0.8

    # 4. HIGH_OSCILLATION
    if tension > 0.3 and abs(acceleration) < 0.01:
        return "HIGH_OSCILLATION", "高位震荡 (风险积聚)", 0.6

    # 5. LOW_OSCILLATION
    if tension < -0.3 and abs(acceleration) < 0.01:
        return "LOW_OSCILLATION", "低位震荡 (机会积聚)", 0.6

    # 6. TRANSITION
    if tension > 0 and acceleration > 0:
        return "TRANSITION_UP", "向上过渡 (蓄力)", 0.4
    elif tension < 0 and acceleration < 0:
        return "TRANSITION_DOWN", "向下过渡 (泄力)", 0.4

    return "TRANSITION", "体制切换中", 0.3


# ==================== Telegram通知 ====================

class TelegramNotifier:
    """Telegram通知系统"""

    def __init__(self, config):
        self.config = config
        self.enabled = config.telegram_enabled

        if self.enabled:
            try:
                self.bot = telebot.TeleBot(config.telegram_token)
                logging.info("[Telegram] Bot初始化成功")
            except Exception as e:
                logging.error(f"[Telegram] Bot初始化失败: {e}")
                self.enabled = False
                self.bot = None
        else:
            self.bot = None

    def send_message(self, message):
        """发送消息"""
        if not self.enabled or not self.bot:
            return

        try:
            self.bot.send_message(self.config.telegram_chat_id, message)
            logging.info("[Telegram] 消息已发送")
        except Exception as e:
            logging.error(f"[Telegram] 发送消息失败: {e}")

    def notify_signal(self, signal_type, confidence, description, price, tension, acceleration, dxy_fuel=0.0):
        """发送信号通知"""
        message = f"""🎯 V8.0 新交易信号

📊 类型: {signal_type}
📈 描述: {description}
🎯 置信度: {confidence:.1%}
💰 价格: ${price:,.0f}

📐 张力: {tension:.3f}
📐 加速度: {acceleration:.3f}
⛽ DXY燃料: {dxy_fuel:.3f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)

    def notify_entry(self, position_type, entry_price, stop_loss, take_profit, signal_type, confidence):
        """发送开仓通知"""
        direction_emoji = "📈 做多" if position_type == 'long' else "📉 做空"

        message = f"""✅ V8.0 开仓成功

{direction_emoji}
💰 入场价: ${entry_price:,.2f}
🎯 止盈: ${take_profit:,.2f} (+{(take_profit/entry_price - 1)*100:.1f}%)
🛑 止损: ${stop_loss:,.2f} ({(stop_loss/entry_price - 1)*100:.1f}%)

📊 信号: {signal_type}
🎯 置信度: {confidence:.1%}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)

    def notify_exit(self, position_type, entry_price, exit_price, pnl_pct, reason):
        """发送平仓通知"""
        direction_emoji = "📈 做多" if position_type == 'long' else "📉 做空"
        pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"

        message = f"""✅ V8.0 平仓成功

{direction_emoji}
💰 入场价: ${entry_price:,.2f}
💰 平仓价: ${exit_price:,.2f}
{pnl_emoji} 盈亏: {pnl_pct:+.2f}%

📊 原因: {reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)

    def notify_status(self):
        """发送状态通知"""
        if self.config.has_position:
            hold_time = (datetime.now() - self.config.entry_time).total_seconds() / 3600 if self.config.entry_time else 0

            # 获取当前价格
            try:
                df = DataFetcher().fetch_btc_data(interval='4h', limit=5)
                current_price = df.iloc[-1]['close'] if df is not None else 0
            except:
                current_price = 0

            if current_price > 0:
                if self.config.position_type == 'long':
                    current_pnl_pct = (current_price - self.config.entry_price) / self.config.entry_price * 100
                else:
                    current_pnl_pct = (self.config.entry_price - current_price) / self.config.entry_price * 100
            else:
                current_pnl_pct = 0.0

            pnl_emoji = "🟢" if current_pnl_pct > 0 else "🔴"

            tp_pct = (self.config.take_profit_price - self.config.entry_price) / self.config.entry_price * 100 if self.config.take_profit_price else 0
            sl_pct = (self.config.stop_loss_price - self.config.entry_price) / self.config.entry_price * 100 if self.config.stop_loss_price else 0

            message = f"""📊 V8.0持仓状态

📍 方向: {'📈 做多' if self.config.position_type == 'long' else '📉 做空'}
💰 入场价: ${self.config.entry_price:,.2f}
💵 当前价: ${current_price:,.2f}
{pnl_emoji} 盈亏: {current_pnl_pct:+.2f}%
🎯 止盈: ${self.config.take_profit_price:,.2f} ({tp_pct:+.2f}%)
🛑 止损: ${self.config.stop_loss_price:,.2f} ({sl_pct:+.2f}%)
⏱ 持仓时长: {hold_time:.1f}小时
📊 入场置信度: {self.config.entry_confidence:.2f}

📈 总交易: {self.config.total_trades}
✅ 盈利: {self.config.winning_trades}
❌ 亏损: {self.config.losing_trades}
💵 总盈亏: {self.config.total_pnl:.2f}%
"""
        else:
            message = f"""📊 V8.0系统状态

⭕ 当前状态: 空仓
📈 总交易: {self.config.total_trades}
✅ 盈利: {self.config.winning_trades}
❌ 亏损: {self.config.losing_trades}
💵 总盈亏: {self.config.total_pnl:.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        self.send_message(message)


# ==================== Telegram命令处理 ====================

class TelegramHandler:
    """Telegram命令处理器"""

    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        self.notifier = engine.notifier

        if config.telegram_enabled and config.telegram_token:
            try:
                self.bot = telebot.TeleBot(config.telegram_token)
                self._register_handlers()
                logging.info("[Telegram] Handler初始化成功")
            except Exception as e:
                logging.error(f"[Telegram] Handler初始化失败: {e}")
                self.bot = None
        else:
            self.bot = None

    def _register_handlers(self):
        """注册命令处理器"""

        @self.bot.message_handler(commands=['start', 'help'])
        def send_help(message):
            if message.chat.id != int(self.config.telegram_chat_id):
                return
            help_text = """🤖 V8.0验证5预警系统

可用命令：
/status - 查看当前状态
/signals - 查看最近信号
/trades - 查看交易历史
/clear - 手动平仓

V8.0特性：
- 验证5逻辑（FFT + Hilbert + 二阶差分）
- V8.0反向策略
- DXY燃料增强
- 风险控制（止损3%，止盈10%）
"""
            try:
                self.bot.reply_to(message, help_text)
            except Exception as e:
                logging.error(f"[Telegram] 发送帮助失败: {e}")

        @self.bot.message_handler(commands=['status'])
        def send_status(message):
            if message.chat.id != int(self.config.telegram_chat_id):
                return
            self.notifier.notify_status()

        @self.bot.message_handler(commands=['signals'])
        def send_signals(message):
            if message.chat.id != int(self.config.telegram_chat_id):
                return

            if not self.config.signal_history:
                self.bot.reply_to(message, "暂无信号历史")
                return

            signals_text = "📊 最近信号\n\n"
            for sig in self.config.signal_history[-10:]:
                traded_mark = "✅" if sig.get('traded', False) else "❌"
                filtered_mark = "🚫" if sig.get('filtered', False) else ""
                signals_text += f"{traded_mark} {sig['type']} | {sig['confidence']:.1%} | {sig['description']}\n"
                signals_text += f"   💰 ${sig['price']:,.0f} | 📐 {sig['tension']:.3f} | {sig['acceleration']:.3f} {filtered_mark}\n\n"

            try:
                self.bot.reply_to(message, signals_text)
            except Exception as e:
                logging.error(f"[Telegram] 发送信号历史失败: {e}")

        @self.bot.message_handler(commands=['trades'])
        def send_trades(message):
            if message.chat.id != int(self.config.telegram_chat_id):
                return

            if not self.config.position_history:
                self.bot.reply_to(message, "暂无交易历史")
                return

            trades_text = "📋 交易历史\n\n"
            for trade in self.config.position_history[-10:]:
                pnl_emoji = "🟢" if trade['pnl_pct'] > 0 else "🔴"
                trades_text += f"{pnl_emoji} {trade['direction'].upper()} @ ${trade['entry_price']:,.2f}\n"
                trades_text += f"   平仓: ${trade['exit_price']:,.2f} | {trade['pnl_pct']:+.2f}%\n"
                trades_text += f"   原因: {trade['reason']}\n\n"

            try:
                self.bot.reply_to(message, trades_text)
            except Exception as e:
                logging.error(f"[Telegram] 发送交易历史失败: {e}")

        @self.bot.message_handler(commands=['clear'])
        def manual_close(message):
            if message.chat.id != int(self.config.telegram_chat_id):
                return

            if not self.config.has_position:
                self.bot.reply_to(message, "当前无持仓")
                return

            # 获取当前价格
            try:
                df = self.engine.fetcher.fetch_btc_data(interval='4h', limit=5)
                current_price = df.iloc[-1]['close'] if df is not None else 0
            except:
                current_price = self.config.entry_price

            # 手动平仓
            self.engine.close_position("手动平仓", current_price)

            try:
                self.bot.reply_to(message, f"✅ 已手动平仓 @ ${current_price:,.2f}")
            except Exception as e:
                logging.error(f"[Telegram] 发送平仓确认失败: {e}")

    def run_polling(self):
        """运行轮询"""
        while True:
            try:
                logging.info("[Telegram] Polling启动...")
                self.bot.polling(non_stop=False, interval=1, timeout=60, long_polling_timeout=20)
            except Exception as e:
                logging.error(f"[Telegram] Polling异常: {e}")
                logging.info("[Telegram] 5秒后重新启动...")
                time.sleep(5)


# ==================== 交易引擎 ====================

class V80TradingEngine:
    """V8.0交易引擎"""

    def __init__(self):
        self.config = V80Config()
        self.fetcher = DataFetcher()
        self.notifier = TelegramNotifier(self.config)

        # 加载状态
        self.config.load_state()

        # V8.0反向策略映射
        self.strategy_map = {
            'BEARISH_SINGULARITY': ('long', '反向抄底'),
            'BULLISH_SINGULARITY': ('short', '反向逃顶'),
            'LOW_OSCILLATION': ('long', '低位做多'),
            'HIGH_OSCILLATION': ('short', '高位做空'),
            'OSCILLATION': ('wait', '震荡观望'),
            'TRANSITION_UP': ('wait', '向上过渡'),
            'TRANSITION_DOWN': ('wait', '向下过渡'),
            'TRANSITION': ('wait', '体制切换'),
        }

    def check_signals(self):
        """检查交易信号（每4小时）"""
        try:
            logging.info("=" * 70)
            logging.info("开始检查信号...")

            # 获取4H数据
            df_4h = self.fetcher.fetch_btc_data(interval='4h', limit=300)
            if df_4h is None:
                logging.error("获取4H数据失败")
                return

            logging.info(f"4H K线数据: {len(df_4h)}条")

            # 计算验证5指标
            prices = df_4h['close'].values
            tension, acceleration = calculate_tension_acceleration_verification5(prices)

            if tension is None:
                logging.error("验证5指标计算失败")
                return

            # 获取DXY数据
            dxy_df = self.fetcher.fetch_dxy_data(days_back=30)
            dxy_fuel = 0.0
            if dxy_df is not None and len(dxy_df) >= 3:
                dxy_history = dxy_df['Close'].tolist()
                dxy_fuel = calculate_dxy_fuel(dxy_history)

            # 分类市场状态
            signal_type, description, confidence = classify_market_state(
                tension, acceleration, dxy_fuel
            )

            current_price = df_4h.iloc[-1]['close']
            current_time = df_4h.index[-1]

            logging.info(f"检测到信号: {signal_type} | 置信度: {confidence:.2f} | {description}")
            logging.info(f"价格: ${current_price:.2f} | 张力: {tension:.3f} | 加速度: {acceleration:.3f} | DXY燃料: {dxy_fuel:.3f}")

            # 记录信号到历史
            signal_record = {
                'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'type': signal_type,
                'confidence': confidence,
                'description': description,
                'price': current_price,
                'tension': tension,
                'acceleration': acceleration,
                'dxy_fuel': dxy_fuel,
                'traded': False,
                'filtered': False
            }
            self.config.signal_history.append(signal_record)

            # 只保留最近20个信号
            if len(self.config.signal_history) > 20:
                self.config.signal_history = self.config.signal_history[-20:]

            # 发送信号通知（所有信号都发送）
            self.notifier.notify_signal(
                signal_type, confidence, description,
                current_price, tension, acceleration, dxy_fuel
            )

            # 置信度过滤
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                logging.info(f"置信度不足 ({confidence:.2f} < {self.config.CONFIDENCE_THRESHOLD})，跳过")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = f'置信度不足: {confidence:.2f}'
                self.config.save_state()
                return

            # 检查是否已有持仓
            if self.config.has_position:
                logging.info("已有持仓，忽略新信号")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = '已有持仓，忽略新信号'
                self.notifier.send_message(f"⏸️ 信号被忽略：已有持仓")
                self.config.save_state()
                return

            # 确定入场方向
            direction, reason = self.strategy_map.get(signal_type, ('wait', '未知状态'))

            if direction == 'wait':
                logging.info(f"观望状态: {signal_type}")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = f'观望状态: {signal_type}'
                self.config.save_state()
                return

            # 计算止盈止损
            if direction == 'long':
                stop_loss = current_price * 0.97  # -3%
                take_profit = current_price * 1.10  # +10%
            else:
                stop_loss = current_price * 1.03  # +3%
                take_profit = current_price * 0.90  # -10%

            # 开仓
            logging.info(f"[开仓] {direction.upper()} @ ${current_price:.2f}")
            logging.info(f"  止盈: ${take_profit:.2f} ({(take_profit/current_price - 1)*100:+.1f}%)")
            logging.info(f"  止损: ${stop_loss:.2f} ({(stop_loss/current_price - 1)*100:+.1f}%)")
            logging.info(f"  信号: {signal_type} | 置信度: {confidence:.2f}")

            # 更新状态
            self.config.has_position = True
            self.config.position_type = direction
            self.config.entry_price = current_price
            self.config.entry_time = datetime.now()
            self.config.stop_loss_price = stop_loss
            self.config.take_profit_price = take_profit
            self.config.entry_signal_type = signal_type
            self.config.entry_confidence = confidence

            # 标记已交易
            self.config.signal_history[-1]['traded'] = True
            self.config.signal_history[-1]['filtered'] = False

            # 发送开仓通知
            self.notifier.notify_entry(
                direction, current_price, stop_loss, take_profit,
                signal_type, confidence
            )

            # 保存状态
            self.config.save_state()

            logging.info("开仓成功！")

        except Exception as e:
            logging.error(f"检查信号异常: {e}", exc_info=True)

    def check_position(self):
        """检查持仓（每1小时）"""
        try:
            if not self.config.has_position:
                return

            logging.info("检查持仓...")

            # 获取当前价格
            df = self.fetcher.fetch_btc_data(interval='4h', limit=5)
            if df is None:
                logging.error("获取当前价格失败")
                return

            current_price = df.iloc[-1]['close']

            # 计算当前盈亏
            if self.config.position_type == 'long':
                pnl_pct = (current_price - self.config.entry_price) / self.config.entry_price
            else:
                pnl_pct = (self.config.entry_price - current_price) / self.config.entry_price

            # 计算持仓时长
            hold_time = (datetime.now() - self.config.entry_time).total_seconds() / 3600 if self.config.entry_time else 0

            logging.info(f"持仓: {self.config.position_type.upper()} @ ${self.config.entry_price:.2f}")
            logging.info(f"当前价格: ${current_price:.2f} | 盈亏: {pnl_pct*100:+.2f}% | 持仓时长: {hold_time:.1f}小时")

            # 检查平仓条件
            should_close = False
            close_reason = ""

            # 止损
            if pnl_pct < -self.config.STOP_LOSS_PCT:
                should_close = True
                close_reason = f"止损 ({pnl_pct*100:.2f}%)"

            # 止盈
            elif pnl_pct > self.config.TAKE_PROFIT_PCT:
                should_close = True
                close_reason = f"止盈 ({pnl_pct*100:.2f}%)"

            # 超时（7天 = 168小时）
            elif hold_time >= 168:
                should_close = True
                close_reason = f"超时 ({hold_time:.1f}小时)"

            if should_close:
                self.close_position(close_reason, current_price)

        except Exception as e:
            logging.error(f"检查持仓异常: {e}", exc_info=True)

    def close_position(self, reason, current_price):
        """平仓"""
        if not self.config.has_position:
            return

        # 计算盈亏
        if self.config.position_type == 'long':
            pnl_pct = (current_price - self.config.entry_price) / self.config.entry_price
        else:
            pnl_pct = (self.config.entry_price - current_price) / self.config.entry_price

        logging.info(f"[平仓] {self.config.position_type.upper()} @ ${current_price:.2f}")
        logging.info(f"  盈亏: {pnl_pct*100:+.2f}%")
        logging.info(f"  原因: {reason}")

        # 更新统计
        self.config.total_trades += 1
        if pnl_pct > 0:
            self.config.winning_trades += 1
        else:
            self.config.losing_trades += 1
        self.config.total_pnl += pnl_pct * 100

        # 记录交易历史
        trade_record = {
            'entry_time': self.config.entry_time.strftime('%Y-%m-%d %H:%M:%S') if self.config.entry_time else 'N/A',
            'direction': self.config.position_type,
            'entry_price': self.config.entry_price,
            'exit_price': current_price,
            'pnl_pct': pnl_pct * 100,
            'reason': reason,
            'signal_type': self.config.entry_signal_type,
            'confidence': self.config.entry_confidence,
            'take_profit': self.config.take_profit_price,
            'stop_loss': self.config.stop_loss_price
        }
        self.config.position_history.append(trade_record)

        # 只保留最近20笔交易
        if len(self.config.position_history) > 20:
            self.config.position_history = self.config.position_history[-20:]

        # 发送平仓通知
        self.notifier.notify_exit(
            self.config.position_type,
            self.config.entry_price,
            current_price,
            pnl_pct * 100,
            reason
        )

        # 重置状态
        self.config.has_position = False
        self.config.position_type = None
        self.config.entry_price = None
        self.config.entry_time = None
        self.config.stop_loss_price = None
        self.config.take_profit_price = None
        self.config.entry_signal_type = None
        self.config.entry_confidence = None

        # 保存状态
        self.config.save_state()

        logging.info("平仓成功！")

    def run(self):
        """主循环"""
        logging.info("=" * 70)
        logging.info("V8.0 验证5智能预警系统启动")
        logging.info("=" * 70)
        logging.info(f"Telegram Token: {self.config.telegram_token[:20]}...")
        logging.info(f"Telegram Chat ID: {self.config.telegram_chat_id}")
        logging.info(f"Telegram Enabled: {self.config.telegram_enabled}")
        logging.info("")

        # 启动时通知
        self.notifier.notify_status()

        # 启动Telegram Polling（后台线程）
        if self.config.telegram_enabled:
            telegram_handler = TelegramHandler(self.config, self)
            if telegram_handler.bot:
                import threading
                telegram_thread = threading.Thread(
                    target=telegram_handler.run_polling,
                    daemon=False
                )
                telegram_thread.start()
                logging.info("[系统] Telegram Polling已启动（后台线程）")
            else:
                logging.warning("[系统] Telegram未启用")

        # 定时任务
        logging.info("定时任务已设置：")
        logging.info("  - 信号检查: 北京时间 0:00, 4:00, 8:00, 12:00, 16:00, 20:00")
        logging.info("  - 持仓检查: 每1小时")
        logging.info("")

        # 主循环
        logging.info("进入主循环...")
        logging.info("=" * 70)

        last_signal_check_hour = None
        last_position_check_hour = None

        loop_count = 0
        heartbeat_interval = 3600  # 每小时打印一次心跳（3600秒）

        while True:
            try:
                loop_count += 1

                # 获取当前北京时间
                now_beijing = datetime.utcnow() + timedelta(hours=8)

                # 心跳日志（每小时一次）
                if loop_count % heartbeat_interval == 0:
                    current_time_str = now_beijing.strftime('%Y-%m-%d %H:%M:%S')
                    logging.info(f"♥ [{current_time_str}] 系统运行中 - 循环次数: {loop_count:,}")
                    logging.info(f"  当前持仓: {'有' if self.config.has_position else '无'}")
                    logging.info(f"  历史信号数: {len(self.config.signal_history)}")
                    logging.info(f"  历史交易数: {self.config.total_trades}")
                current_hour = now_beijing.hour
                current_minute = now_beijing.minute

                # 信号检查：北京时间4H K线收盘时间（0:00, 4:00, 8:00, 12:00, 16:00, 20:00）
                # 在收盘后5分钟内执行（0:00-0:05, 4:00-4:05, ...）
                if current_hour % 4 == 0 and current_minute < 5:
                    if last_signal_check_hour != current_hour:
                        logging.info(f"[定时] 触发信号检查（北京时间 {now_beijing.strftime('%H:%M')}）")
                        self.check_signals()
                        last_signal_check_hour = current_hour

                # 持仓检查：每1小时整点执行
                if current_minute < 1:
                    if last_position_check_hour != current_hour:
                        logging.info(f"[定时] 触发持仓检查（北京时间 {now_beijing.strftime('%H:%M')}）")
                        self.check_position()
                        last_position_check_hour = current_hour

                # 每秒检查一次
                time.sleep(1)

            except KeyboardInterrupt:
                logging.info("收到停止信号，正在退出...")
                break
            except Exception as e:
                logging.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(60)


# ==================== 主入口 ====================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('v80_cloud.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 创建并运行引擎
    engine = V80TradingEngine()
    engine.run()
