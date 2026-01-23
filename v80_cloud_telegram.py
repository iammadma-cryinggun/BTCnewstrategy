# -*- coding: utf-8 -*-
"""
V8.0 云端部署系统 - 验证5 + Telegram通知

功能:
1. 验证5逻辑（FFT + Hilbert + 二阶差分）
2. Telegram Bot通知（所有信号、开仓、平仓）
3. 远程控制（状态查询、手动平仓）
4. 云端运行支持（Zeabur/Replit）

版本: v5.0 Cloud
日期: 2026-01-22
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
import requests
from io import StringIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import os
import logging
import telebot
from telebot import types
import threading
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置管理 ====================

class Config:
    """系统配置"""

    # Telegram配置
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
    TELEGRAM_ENABLED = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

    # 账户配置
    ACCOUNT_BALANCE = float(os.environ.get('ACCOUNT_BALANCE', '10000'))
    RISK_PER_TRADE = float(os.environ.get('RISK_PER_TRADE', '0.02'))

    # 验证5参数
    TENSION_THRESHOLD = 0.35
    ACCEL_THRESHOLD = 0.02
    OSCILLATION_BAND = 0.5

    # 风险控制
    STOP_LOSS_PCT = 0.03
    TAKE_PROFIT_PCT = 0.10

    # 运行配置
    CHECK_INTERVAL = int(os.environ.get('CHECK_INTERVAL', '300'))  # 5分钟

    @classmethod
    def load_from_env(cls):
        """从环境变量加载配置"""
        logging.info("[配置] 从环境变量加载配置")
        return cls()

# ==================== 数据结构 ====================

@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: datetime
    action: str  # LONG, SHORT, WAIT
    position_size: float
    reason: str
    signal_type: str
    confidence: float
    metrics: Dict

@dataclass
class Position:
    """持仓信息"""
    entry_time: datetime
    entry_price: float
    size: float
    side: str
    stop_loss: float
    take_profit: float
    signal_type: str
    confidence: float
    reason: str

# ==================== Telegram Bot ====================

class TelegramNotifier:
    """Telegram通知系统"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

        if self.enabled:
            try:
                self.bot = telebot.TeleBot(token)
                logging.info(f"[Telegram] Bot初始化成功")
                self._test_connection()
            except Exception as e:
                logging.error(f"[Telegram] Bot初始化失败: {e}")
                self.enabled = False
        else:
            self.bot = None

    def _test_connection(self):
        """测试Telegram连接"""
        try:
            self.bot.send_message(self.chat_id, "🔔 V8.0系统启动\n\n验证5逻辑 + Telegram通知已激活")
            logging.info("[Telegram] 连接测试成功")
        except Exception as e:
            logging.error(f"[Telegram] 连接测试失败: {e}")
            self.enabled = False

    def send_signal(self, signal: TradingSignal):
        """发送新信号通知"""
        if not self.enabled:
            return

        try:
            emoji = "📈" if signal.action == 'LONG' else "📉" if signal.action == 'SHORT' else "⏸"
            action_text = "做多" if signal.action == 'LONG' else "做空" if signal.action == 'SHORT' else "观望"

            message = f"""
🎯 【验证5信号检测】

{emoji} 信号类型: {signal.signal_type}
📊 置信度: {signal.confidence:.1%}
💰 当前价格: ${signal.metrics.get('btc_price', 0):,.0f}
📈 张力: {signal.metrics.get('tension', 0):.4f}
📉 加速度: {signal.metrics.get('acceleration', 0):.6f}
🔋 DXY燃料: {signal.metrics.get('dxy_fuel', 0):.2f}

⚡ 决策: {action_text}
📝 理由: {signal.reason}
🕐 时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{'✅ 执行交易' if signal.action != 'WAIT' else '⏸ 观望'}
"""

            self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
            logging.info(f"[Telegram] 信号通知已发送: {signal.signal_type}")

        except Exception as e:
            logging.error(f"[Telegram] 发送信号失败: {e}")

    def send_open_position(self, position: Position):
        """发送开仓通知"""
        if not self.enabled:
            return

        try:
            emoji = "📈" if position.side == 'LONG' else "📉"
            side_text = "做多" if position.side == 'LONG' else "做空"

            message = f"""
✅ 【开仓执行】

{emoji} 方向: {side_text}
💰 入场价: ${position.entry_price:,.0f}
💵 仓位: ${position.size:,.0f}
🎯 信号: {position.signal_type}
📊 置信度: {position.confidence:.1%}

🛑 止损: ${position.stop_loss:,.0f}
🎯 止盈: ${position.take_profit:,.0f}

📝 理由: {position.reason}
🕐 时间: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
            logging.info(f"[Telegram] 开仓通知已发送")

        except Exception as e:
            logging.error(f"[Telegram] 发送开仓失败: {e}")

    def send_close_position(self, position: Position, pnl_ratio: float, pnl_amount: float, reason: str, balance: float):
        """发送平仓通知"""
        if not self.enabled:
            return

        try:
            pnl_emoji = "🟢" if pnl_ratio > 0 else "🔴"
            profit_loss = "盈利" if pnl_ratio > 0 else "亏损"

            message = f"""
❌ 【平仓执行】

{pnl_emoji} {profit_loss}: {pnl_ratio:+.2f} (${pnl_amount:+,.0f})
📝 理由: {reason}
💰 当前余额: ${balance:,.0f}

📊 交易详情:
   方向: {position.side}
   入场价: ${position.entry_price:,.0f}
   平仓价: ${position.side == 'LONG' and position.take_profit or position.stop_loss:,.0f}
   信号: {position.signal_type}
   时长: {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f}小时

🕐 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
            logging.info(f"[Telegram] 平仓通知已发送")

        except Exception as e:
            logging.error(f"[Telegram] 发送平仓失败: {e}")

    def send_status(self, position: Optional[Position], balance: float, total_trades: int, winning_trades: int, losing_trades: int, total_pnl: float):
        """发送状态通知"""
        if not self.enabled:
            return

        try:
            if position:
                # 获取当前价格
                current_price = 0
                try:
                    btc_fetcher = BTCDataFetcher()
                    df = btc_fetcher.fetch()
                    if df is not None:
                        current_price = df['close'].iloc[-1]
                except:
                    pass

                if current_price > 0:
                    if position.side == 'LONG':
                        pnl_ratio = (current_price - position.entry_price) / position.entry_price
                    else:
                        pnl_ratio = (position.entry_price - current_price) / position.entry_price

                    pnl_emoji = "🟢" if pnl_ratio > 0 else "🔴"

                    message = f"""
📊 【系统状态】

💼 当前持仓: {'📈 做多' if position.side == 'LONG' else '📉 做空'}
💰 入场价: ${position.entry_price:,.0f}
💵 当前价: ${current_price:,.0f}
{pnl_emoji} 盈亏: {pnl_ratio:+.2%}
⏱ 持仓时长: {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f}小时
🎯 信号: {position.signal_type}
📊 置信度: {position.confidence:.1%}
"""
                else:
                    message = f"""
📊 【系统状态】

⭕ 当前状态: 空仓

📈 总交易: {total_trades}
✅ 盈利: {winning_trades}
❌ 亏损: {losing_trades}
💵 总盈亏: {total_pnl:+.2f}%
💰 余额: ${balance:,.0f}
"""

            self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
            logging.info(f"[Telegram] 状态通知已发送")

        except Exception as e:
            logging.error(f"[Telegram] 发送状态失败: {e}")

    def send_error(self, error_msg: str):
        """发送错误通知"""
        if not self.enabled:
            return

        try:
            message = f"""
⚠️ 【系统错误】

{error_msg}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
            logging.warning(f"[Telegram] 错误通知已发送")

        except Exception as e:
            logging.error(f"[Telegram] 发送错误失败: {e}")

    def register_commands(self, engine):
        """注册Telegram命令"""
        if not self.enabled:
            return

        @self.bot.message_handler(commands=['start', 'help'])
        def send_help(message):
            if message.chat.id != int(self.chat_id):
                return

            help_text = """
🤖 V8.0交易系统 - 验证5逻辑 + Telegram通知

📋 可用命令：
/status - 查看当前持仓状态
/signals - 查看最近6个信号
/trades - 查看交易历史
/clear - 手动平仓（⚠️ 谨慎使用）
/config - 查看系统配置
/help - 显示此帮助

⭐ V8.0验证5特性：
- FFT滤波 + Hilbert变换
- 张力二阶差分计算加速度
- DXY燃料增强
- V8.0反向策略

🛡️ 风险控制：
- 止损: -3%
- 止盈: +10%
- 单笔风险: 2%
"""

            try:
                self.bot.reply_to(message, help_text)
            except Exception as e:
                logging.error(f"[Telegram] 发送帮助失败: {e}")

        @self.bot.message_handler(commands=['status'])
        def send_status(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                self.send_status(
                    engine.position,
                    engine.account_balance,
                    engine.total_trades,
                    engine.winning_trades,
                    engine.losing_trades,
                    engine.total_pnl
                )
            except Exception as e:
                logging.error(f"[Telegram] 发送状态失败: {e}")

        @self.bot.message_handler(commands=['clear'])
        def manual_close(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                if engine.position:
                    # 获取当前价格
                    current_price = 0
                    try:
                        btc_fetcher = BTCDataFetcher()
                        df = btc_fetcher.fetch()
                        if df is not None:
                            current_price = df['close'].iloc[-1]
                    except:
                        pass

                    if current_price > 0:
                        engine.close_position("手动平仓", current_price)
                        self.bot.reply_to(message, "✅ 已执行手动平仓")
                    else:
                        self.bot.reply_to(message, "❌ 无法获取当前价格")
                else:
                    self.bot.reply_to(message, "⭕ 当前无持仓")
            except Exception as e:
                logging.error(f"[Telegram] 手动平仓失败: {e}")

        @self.bot.message_handler(commands=['config'])
        def send_config(message):
            if message.chat.id != int(self.chat_id):
                return

            try:
                config_text = f"""
⚙️ 【系统配置】

💰 账户余额: ${Config.ACCOUNT_BALANCE:,.0f}
📊 单笔风险: {Config.RISK_PER_TRADE:.1%}
⏱ 检查间隔: {Config.CHECK_INTERVAL}秒

📐 验证5参数:
  张力阈值: {Config.TENSION_THRESHOLD}
  加速度阈值: {Config.ACCEL_THRESHOLD}
  震荡带: {Config.OSCILLATION_BAND}

🛡️ 风险控制:
  止损: {Config.STOP_LOSS_PCT:.1%}
  止盈: {Config.TAKE_PROFIT_PCT:.1%}

🔔 Telegram: {'✅ 已启用' if self.enabled else '❌ 未启用'}
"""

                self.bot.reply_to(message, config_text)
            except Exception as e:
                logging.error(f"[Telegram] 发送配置失败: {e}")

        # 启动轮询
        logging.info("[Telegram] 启动消息轮询")
        self.bot.polling(non_stop=True)

# ==================== 验证5引擎（复用模块化版本）====================

class BTCDataFetcher:
    """BTC数据获取器"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"

    def fetch(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """获取BTC 4小时K线数据"""
        try:
            params = {'symbol': 'BTCUSDT', 'interval': '4h', 'limit': limit}
            response = requests.get(self.base_url, params=params, timeout=15)
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

class DXYDataFetcher:
    """DXY数据获取器"""

    def fetch(self, days_back: int = 30) -> Optional[pd.DataFrame]:
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

def calculate_tension_acceleration_verification5(prices: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """计算张力和加速度（验证5逻辑）"""
    if len(prices) < 3:
        return None, None

    try:
        d_prices = detrend(prices)
        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real
        analytic = hilbert(filtered)
        tension = np.imag(analytic)

        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension

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

def calculate_dxy_fuel(dxy_history: List[float]) -> float:
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

class MarketStateClassifier:
    """市场状态分类器（验证5逻辑）"""

    def __init__(self):
        self.TENSION_THRESHOLD = Config.TENSION_THRESHOLD
        self.ACCEL_THRESHOLD = Config.ACCEL_THRESHOLD
        self.OSCILLATION_BAND = Config.OSCILLATION_BAND

    def classify(self, tension: float, acceleration: float, dxy_fuel: float = 0.0) -> Tuple[str, str, float]:
        """分类市场状态"""

        # 1. BEARISH_SINGULARITY
        if tension > self.TENSION_THRESHOLD and acceleration < -self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.1:
                return "BEARISH_SINGULARITY", "强奇点看空 (宏观失速)", 0.9
            else:
                return "BEARISH_SINGULARITY", "奇点看空 (动力失速)", 0.7

        # 2. BULLISH_SINGULARITY
        if tension < -self.TENSION_THRESHOLD and acceleration > self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.2:
                return "BULLISH_SINGULARITY", "超强奇点看涨 (燃料爆炸)", 0.95
            elif dxy_fuel > 0:
                return "BULLISH_SINGULARITY", "强奇点看涨 (动力回归)", 0.8
            else:
                return "BULLISH_SINGULARITY", "奇点看涨 (弹性释放)", 0.6

        # 3. OSCILLATION
        if abs(tension) < self.OSCILLATION_BAND and abs(acceleration) < self.ACCEL_THRESHOLD:
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

# ==================== 交易引擎 ====================

class CloudTradingEngine:
    """云端交易引擎（验证5 + Telegram）"""

    def __init__(self, config: Config):
        self.config = config

        # 数据获取
        self.btc_fetcher = BTCDataFetcher()
        self.dxy_fetcher = DXYDataFetcher()

        # 数据缓存
        self.price_history: List[float] = []
        self.dxy_history: List[float] = []

        # 验证5引擎
        self.classifier = MarketStateClassifier()

        # V8.0反向策略
        self.strategy_map = {
            'BEARISH_SINGULARITY': ('LONG', '反向抄底'),
            'BULLISH_SINGULARITY': ('SHORT', '反向逃顶'),
            'LOW_OSCILLATION': ('LONG', '低位做多'),
            'HIGH_OSCILLATION': ('SHORT', '高位做空'),
            'OSCILLATION': ('WAIT', '震荡观望'),
            'TRANSITION_UP': ('WAIT', '向上过渡'),
            'TRANSITION_DOWN': ('WAIT', '向下过渡'),
            'TRANSITION': ('WAIT', '体制切换'),
        }

        # 账户管理
        self.account_balance = self.config.ACCOUNT_BALANCE
        self.position: Optional[Position] = None

        # 交易统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # 信号历史
        self.signal_history: List[Dict] = []

        # 交易历史
        self.position_history: List[Dict] = []

        # Telegram通知
        self.telegram = TelegramNotifier(
            config.TELEGRAM_TOKEN,
            config.TELEGRAM_CHAT_ID
        )

        if self.telegram.enabled:
            self.telegram.register_commands(self)

    def analyze_and_trade(self):
        """分析和交易（主循环）"""
        try:
            # 第0层：数据获取
            btc_df = self.btc_fetcher.fetch()
            if btc_df is None:
                return

            dxy_df = self.dxy_fetcher.fetch()

            current_price = btc_df['close'].iloc[-1]
            current_volume = btc_df['volume'].iloc[-1]

            # 更新缓存
            self.price_history.append(current_price)
            if len(self.price_history) > 100:
                self.price_history.pop(0)

            if dxy_df is not None:
                dxy_latest = dxy_df['Close'].iloc[-1]
                self.dxy_history.append(dxy_latest)
                if len(self.dxy_history) > 10:
                    self.dxy_history.pop(0)

            if len(self.price_history) < 60:
                return

            # 第1层：物理指标计算（验证5）
            prices_array = np.array(self.price_history)
            tension, acceleration = calculate_tension_acceleration_verification5(prices_array)

            if tension is None:
                return

            dxy_fuel = calculate_dxy_fuel(self.dxy_history)

            # 第2层：市场状态诊断
            signal_type, description, confidence = self.classifier.classify(
                tension, acceleration, dxy_fuel
            )

            # 记录信号
            signal_entry = {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': signal_type,
                'description': description,
                'confidence': confidence,
                'price': current_price,
                'tension': tension,
                'acceleration': acceleration,
                'dxy_fuel': dxy_fuel,
                'traded': False,
                'filtered': False
            }

            # 只记录高置信度信号
            if confidence >= 0.6:
                self.signal_history.append(signal_entry)
                if len(self.signal_history) > 100:
                    self.signal_history.pop(0)

            # 置信度过滤
            if confidence < 0.6:
                return

            # 第3层：V8.0反向策略决策
            action, reason_base = self.strategy_map.get(signal_type, ('WAIT', '未知状态'))
            reason = f"{signal_type} → {reason_base}"

            if action == 'WAIT':
                return

            # 第4层：执行交易
            if self.position is None:
                # 计算仓位
                base_size = 1.0 + (confidence - 0.6) * 0.5

                # DXY燃料增强
                if abs(dxy_fuel) > 0.2:
                    base_size *= 1.2

                # 风险控制
                max_position = self.account_balance * (self.config.RISK_PER_TRADE / self.config.STOP_LOSS_PCT)
                position_value = min(self.account_balance * base_size, max_position)

                # 开仓
                self.open_position(action, current_price, position_value, signal_type, confidence, reason, tension, acceleration, dxy_fuel)

            else:
                # 检查持仓
                self.check_position(current_price, confidence)

        except Exception as e:
            logging.error(f"[ERROR] analyze_and_trade失败: {e}")
            self.telegram.send_error(f"分析失败: {str(e)}")

    def open_position(self, action: str, price: float, size: float, signal_type: str, confidence: float, reason: str, tension: float, acceleration: float, dxy_fuel: float):
        """开仓"""
        stop_loss = price * (0.97 if action == 'LONG' else 1.03)
        take_profit = price * (1.10 if action == 'LONG' else 0.90)

        self.position = Position(
            entry_time=datetime.now(),
            entry_price=price,
            size=size,
            side=action,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type=signal_type,
            confidence=confidence,
            reason=reason
        )

        # Telegram通知
        self.telegram.send_open_position(self.position)

        logging.info(f"[开仓] {action} ${price:,.0f} 仓位=${size:,.0f}")

    def check_position(self, current_price: float, current_confidence: float):
        """检查持仓"""
        if self.position is None:
            return

        # 计算盈亏
        if self.position.side == 'LONG':
            pnl_ratio = (current_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_ratio = (self.position.entry_price - current_price) / self.position.entry_price

        # 检查平仓条件
        should_close = False
        close_reason = ""

        # 止损
        if pnl_ratio < -self.config.STOP_LOSS_PCT:
            should_close = True
            close_reason = f"止损 ({pnl_ratio:.2%})"

        # 止盈
        elif pnl_ratio > self.config.TAKE_PROFIT_PCT:
            should_close = True
            close_reason = f"止盈 ({pnl_ratio:.2%})"

        # 信号消失
        elif current_confidence < 0.5:
            should_close = True
            close_reason = "信号消失"

        if should_close:
            self.close_position(close_reason, current_price)

    def close_position(self, reason: str, current_price: float):
        """平仓"""
        if self.position is None:
            return

        # 计算盈亏
        if self.position.side == 'LONG':
            pnl_ratio = (current_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_ratio = (self.position.entry_price - current_price) / self.position.entry_price

        pnl_amount = self.position.size * pnl_ratio
        self.account_balance += pnl_amount

        # 更新统计
        self.total_trades += 1
        if pnl_ratio > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_pnl += pnl_ratio

        # 记录历史
        self.position_history.append({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'CLOSE',
            'side': self.position.side,
            'entry_price': self.position.entry_price,
            'exit_price': current_price,
            'pnl_ratio': pnl_ratio,
            'pnl_amount': pnl_amount,
            'reason': reason
        })

        # Telegram通知
        self.telegram.send_close_position(
            self.position,
            pnl_ratio,
            pnl_amount,
            reason,
            self.account_balance
        )

        logging.info(f"[平仓] {reason} {pnl_ratio:+.2%} (${pnl_amount:+,.0f})")

        self.position = None

    def run(self):
        """运行主循环"""
        logging.info("="*80)
        logging.info("V8.0 云端交易系统 - 验证5 + Telegram通知")
        logging.info("="*80)
        logging.info(f"账户余额: ${self.account_balance:,.0f}")
        logging.info(f"单笔风险: {self.config.RISK_PER_TRADE:.1%}")
        logging.info(f"检查间隔: {self.config.CHECK_INTERVAL}秒")
        logging.info(f"Telegram: {'✅ 已启用' if self.telegram.enabled else '❌ 未启用'}")
        logging.info("系统启动...")
        logging.info("")

        loop_count = 0
        heartbeat_interval = 30  # 每30次循环打印一次心跳

        try:
            while True:
                loop_count += 1

                # 心跳日志
                if loop_count % heartbeat_interval == 0:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logging.info(f"♥ [{current_time}] 系统运行中 - 循环次数: {loop_count}")

                self.analyze_and_trade()
                time.sleep(self.config.CHECK_INTERVAL)

        except KeyboardInterrupt:
            logging.info("\n系统停止")
            logging.info(f"最终余额: ${self.account_balance:,.0f}")
            logging.info(f"总交易: {self.total_trades}")
            logging.info(f"总盈亏: {self.total_pnl:+.2%}")

# ==================== 主程序 ====================

def main():
    """主程序"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('v80_cloud.log'),
            logging.StreamHandler()
        ]
    )

    # 加载配置
    config = Config.load_from_env()

    # 创建引擎
    engine = CloudTradingEngine(config)

    # 运行
    engine.run()

if __name__ == "__main__":
    main()
