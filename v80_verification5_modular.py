# -*- coding: utf-8 -*-
"""
V8.0 完整模块化系统 - 基于验证5逻辑

包含所有9大模块，25个子模块
每个模块都是独立的类，方便维护和扩展

版本: v3.0 Modular
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
import warnings
warnings.filterwarnings('ignore')


# ==================== 模块1：数据获取层（第0层） ====================

class BTCDataFetcher:
    """1.1 BTC数据获取器"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = "BTCUSDT"
        self.interval = "4h"
        self.limit = 1000

    def fetch(self) -> Optional[pd.DataFrame]:
        """获取BTC 4小时K线数据"""
        try:
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'limit': self.limit
            }
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

            print(f"[BTC数据] 获取{len(df)}条，最新价格 ${df['close'].iloc[-1]:,.0f}")
            return df

        except Exception as e:
            print(f"[ERROR] BTC数据获取失败: {e}")
            return None


class DXYDataFetcher:
    """1.2 DXY数据获取器"""

    def __init__(self):
        self.base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        self.series_id = "DTWEXBGS"

    def fetch(self, days_back: int = 30) -> Optional[pd.DataFrame]:
        """获取DXY美元指数数据"""
        try:
            url = f"{self.base_url}?id={self.series_id}"
            response = requests.get(url, timeout=15)

            if response.status_code != 200:
                print(f"[WARNING] DXY数据获取失败: HTTP {response.status_code}")
                return None

            dxy_df = pd.read_csv(StringIO(response.text))
            dxy_df['observation_date'] = pd.to_datetime(dxy_df['observation_date'])
            dxy_df.set_index('observation_date', inplace=True)
            dxy_df.rename(columns={'DTWEXBGS': 'Close'}, inplace=True)
            dxy_df = dxy_df.dropna()
            dxy_df['Close'] = pd.to_numeric(dxy_df['Close'], errors='coerce')

            cutoff_date = datetime.now() - timedelta(days=days_back)
            dxy_df = dxy_df[dxy_df.index >= cutoff_date]

            print(f"[DXY数据] 获取{len(dxy_df)}条，最新DXY {dxy_df['Close'].iloc[-1]:.2f}")
            return dxy_df

        except Exception as e:
            print(f"[WARNING] DXY数据获取失败: {e}")
            return None


class DataCache:
    """1.3 数据缓存管理器"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.dxy_history: List[float] = []

    def update_price(self, price: float, volume: float):
        """更新价格和成交量历史"""
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > self.max_size:
            self.price_history.pop(0)
            self.volume_history.pop(0)

    def update_dxy(self, dxy: float):
        """更新DXY历史"""
        self.dxy_history.append(dxy)

        if len(self.dxy_history) > 10:
            self.dxy_history.pop(0)

    def get_prices_array(self) -> np.ndarray:
        """获取价格数组"""
        return np.array(self.price_history)

    def is_ready(self) -> bool:
        """检查是否有足够数据"""
        return len(self.price_history) >= 60


# ==================== 模块2：物理指标计算（第1层） ====================

def preprocess_prices(prices: np.ndarray) -> np.ndarray:
    """2.1 数据预处理：去趋势"""
    try:
        d_prices = detrend(prices, type='linear')
        return d_prices
    except Exception as e:
        print(f"[ERROR] 数据预处理失败: {e}")
        return prices


def fft_filter(prices: np.ndarray, n_components: int = 8) -> np.ndarray:
    """2.2 FFT滤波"""
    try:
        coeffs = fft(prices)
        coeffs[n_components:] = 0
        filtered = ifft(coeffs).real
        return filtered
    except Exception as e:
        print(f"[ERROR] FFT滤波失败: {e}")
        return prices


def hilbert_transform(filtered_prices: np.ndarray) -> np.ndarray:
    """2.3 Hilbert变换，提取张力"""
    try:
        analytic = hilbert(filtered_prices)
        tension = np.imag(analytic)
        return tension
    except Exception as e:
        print(f"[ERROR] Hilbert变换失败: {e}")
        return np.zeros_like(filtered_prices)


def normalize_tension(tension: np.ndarray) -> np.ndarray:
    """2.4 张力标准化"""
    try:
        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension
        return norm_tension
    except Exception as e:
        print(f"[ERROR] 张力标准化失败: {e}")
        return tension


def calculate_acceleration(norm_tension: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """2.5 计算加速度（张力的二阶差分）"""
    if len(norm_tension) < 3:
        return None, None

    try:
        current_tension = norm_tension[-1]
        prev_tension = norm_tension[-2]
        prev2_tension = norm_tension[-3]

        velocity = current_tension - prev_tension
        acceleration = velocity - (prev_tension - prev2_tension)

        return float(current_tension), float(acceleration)

    except Exception as e:
        print(f"[ERROR] 加速度计算失败: {e}")
        return None, None


def calculate_dxy_fuel(dxy_history: List[float]) -> float:
    """2.6 计算DXY燃料"""
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
        print(f"[ERROR] DXY燃料计算失败: {e}")
        return 0.0


class PhysicsIndicatorCalculator:
    """物理指标计算器（整合模块2的所有功能）"""

    def calculate_all(self, prices: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """计算所有物理指标"""
        try:
            # 2.1 预处理
            d_prices = preprocess_prices(prices)

            # 2.2 FFT滤波
            filtered = fft_filter(d_prices)

            # 2.3 Hilbert变换
            tension_raw = hilbert_transform(filtered)

            # 2.4 标准化
            tension_norm = normalize_tension(tension_raw)

            # 2.5 计算加速度
            tension, acceleration = calculate_acceleration(tension_norm)

            return tension, acceleration

        except Exception as e:
            print(f"[ERROR] 物理指标计算失败: {e}")
            return None, None


# ==================== 模块3：市场状态诊断（第2层） ====================

class MarketStateClassifier:
    """3.1 市场状态分类器"""

    def __init__(self):
        self.TENSION_THRESHOLD = 0.35
        self.ACCEL_THRESHOLD = 0.02
        self.OSCILLATION_BAND = 0.5

    def classify(self,
                tension: float,
                acceleration: float,
                dxy_fuel: float = 0.0) -> Tuple[str, str, float]:
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


def evaluate_confidence(signal_type: str,
                       base_confidence: float,
                       dxy_fuel: float) -> float:
    """3.2 评估置信度"""
    if abs(dxy_fuel) > 0.2:
        base_confidence = min(base_confidence + 0.1, 0.95)
    return base_confidence


def should_enhance_with_dxy(signal_type: str, dxy_fuel: float) -> bool:
    """3.3 判断是否需要DXY增强"""
    if signal_type in ['BULLISH_SINGULARITY', 'LOW_OSCILLATION']:
        return dxy_fuel > 0.2

    if signal_type in ['BEARISH_SINGULARITY', 'HIGH_OSCILLATION']:
        return dxy_fuel < -0.2

    return False


# ==================== 模块4：交易决策（第3层） ====================

class V8ReverseStrategy:
    """4.1 V8.0反向策略"""

    def __init__(self):
        self.strategy_map = {
            'BEARISH_SINGULARITY': ('LONG', '抄底'),
            'BULLISH_SINGULARITY': ('SHORT', '逃顶'),
            'LOW_OSCILLATION': ('LONG', '低位做多'),
            'HIGH_OSCILLATION': ('SHORT', '高位做空'),
            'OSCILLATION': ('WAIT', '震荡观望'),
            'TRANSITION_UP': ('WAIT', '向上过渡'),
            'TRANSITION_DOWN': ('WAIT', '向下过渡'),
            'TRANSITION': ('WAIT', '体制切换')
        }

    def get_action(self, signal_type: str) -> Tuple[str, str]:
        """获取交易动作"""
        action, reason_base = self.strategy_map.get(signal_type, ('WAIT', '未知状态'))
        reason = f"{signal_type} → {reason_base}"
        return action, reason


def calculate_base_position(confidence: float) -> float:
    """4.2 计算基础仓位"""
    base_size = 1.0 + (confidence - 0.6) * 0.5
    base_size = max(0.8, min(base_size, 1.5))
    return base_size


def apply_dxy_enhancement(base_size: float,
                         signal_type: str,
                         dxy_fuel: float) -> Tuple[float, str]:
    """4.3 应用DXY增强"""
    enhanced_size = base_size
    enhancement = ""

    if signal_type in ['BULLISH_SINGULARITY', 'LOW_OSCILLATION']:
        if dxy_fuel > 0.2:
            enhanced_size *= 1.2
            enhancement = f" + DXY燃料增强({dxy_fuel:.2f})"

    elif signal_type in ['BEARISH_SINGULARITY', 'HIGH_OSCILLATION']:
        if dxy_fuel < -0.2:
            enhanced_size *= 1.2
            enhancement = f" + DXY燃料增强({dxy_fuel:.2f})"

    return enhanced_size, enhancement


def apply_risk_control(strategy_size: float,
                      account_balance: float,
                      risk_per_trade: float,
                      stop_loss_pct: float = 0.03) -> float:
    """4.4 应用风险控制"""
    strategy_position = account_balance * strategy_size
    max_position_by_risk = account_balance * (risk_per_trade / stop_loss_pct)
    actual_position = min(strategy_position, max_position_by_risk)
    return actual_position


# ==================== 模块5：风险管理（第4层） ====================

def calculate_stop_loss(entry_price: float,
                       action: str,
                       stop_loss_pct: float = 0.03) -> float:
    """5.1 计算止损价格"""
    if action == 'LONG':
        return entry_price * (1 - stop_loss_pct)
    else:
        return entry_price * (1 + stop_loss_pct)


def calculate_take_profit(entry_price: float,
                         action: str,
                         take_profit_pct: float = 0.10) -> float:
    """5.2 计算止盈价格"""
    if action == 'LONG':
        return entry_price * (1 + take_profit_pct)
    else:
        return entry_price * (1 - take_profit_pct)


def check_signal_disappeared(current_confidence: float,
                            threshold: float = 0.5) -> bool:
    """5.3 检测信号是否消失"""
    return current_confidence < threshold


class PositionMonitor:
    """5.4 持仓监控器"""

    def __init__(self, position: Dict):
        self.position = position

    def check(self, current_price: float, current_confidence: float) -> Tuple[bool, str]:
        """检查持仓状态"""
        entry_price = self.position['entry_price']
        side = self.position['side']
        stop_loss = self.position['stop_loss']
        take_profit = self.position['take_profit']

        # 计算盈亏
        if side == 'LONG':
            pnl_ratio = (current_price - entry_price) / entry_price
        else:
            pnl_ratio = (entry_price - current_price) / entry_price

        # 1. 止损检查
        if pnl_ratio < -0.03:
            return True, f"止损 ({pnl_ratio:.2%})"

        # 2. 止盈检查
        if pnl_ratio > 0.10:
            return True, f"止盈 ({pnl_ratio:.2%})"

        # 3. 信号消失检查
        if current_confidence < 0.5:
            return True, "信号消失"

        # 4. 价格触发检查
        if side == 'LONG':
            if current_price <= stop_loss:
                return True, "触发止损价"
            if current_price >= take_profit:
                return True, "触发止盈价"
        else:
            if current_price >= stop_loss:
                return True, "触发止损价"
            if current_price <= take_profit:
                return True, "触发止盈价"

        return False, ""


# ==================== 模块6：订单执行 ====================

class OrderExecutor:
    """6.1 & 6.2 订单执行器"""

    def __init__(self):
        self.positions: List[Dict] = []

    def open_position(self,
                     signal: Dict,
                     account_balance: float,
                     risk_per_trade: float) -> Dict:
        """开仓"""
        action = signal['action']
        entry_price = signal['current_price']
        confidence = signal['confidence']

        # 计算仓位
        base_size = calculate_base_position(confidence)
        enhanced_size, enhancement = apply_dxy_enhancement(
            base_size,
            signal['signal_type'],
            signal.get('dxy_fuel', 0)
        )
        position_value = apply_risk_control(
            enhanced_size,
            account_balance,
            risk_per_trade
        )

        # 计算止损止盈
        stop_loss = calculate_stop_loss(entry_price, action, 0.03)
        take_profit = calculate_take_profit(entry_price, action, 0.10)

        # 创建持仓
        position = {
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'size': position_value,
            'side': action,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_type': signal['signal_type'],
            'confidence': confidence,
            'reason': signal['reason'] + enhancement
        }

        self.positions.append(position)

        return position

    def close_position(self,
                      position: Dict,
                      current_price: float,
                      reason: str) -> Tuple[float, str]:
        """平仓"""
        entry_price = position['entry_price']
        side = position['side']
        size = position['size']

        # 计算盈亏
        if side == 'LONG':
            pnl_ratio = (current_price - entry_price) / entry_price
        else:
            pnl_ratio = (entry_price - current_price) / entry_price

        pnl_amount = size * pnl_ratio

        # 从持仓列表中移除
        if position in self.positions:
            self.positions.remove(position)

        description = f"{reason}: {pnl_ratio:.2%}"
        return pnl_ratio, description


class AccountManager:
    """6.3 账户管理器"""

    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.total_pnl = 0.0

    def update_balance(self, pnl_amount: float):
        """更新余额"""
        self.balance += pnl_amount
        self.total_pnl += pnl_amount

    def get_performance(self) -> Dict:
        """获取性能统计"""
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_pnl': self.total_pnl,
            'total_return': total_return
        }


# ==================== 模块7：日志记录 ====================

class TradingLogger:
    """7.1 交易日志记录器"""

    def __init__(self, log_file: str = 'live_trading_验证5逻辑.csv'):
        self.log_file = log_file
        self.logs: List[Dict] = []

    def log_open(self, position: Dict):
        """记录开仓"""
        log_entry = {
            '时间': position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            '动作': 'OPEN',
            '信号类型': position['signal_type'],
            '置信度': f"{position['confidence']:.1%}",
            '方向': position['side'],
            '价格': f"${position['entry_price']:,.0f}",
            '仓位': f"${position['size']:,.0f}",
            '止损': f"${position['stop_loss']:,.0f}",
            '止盈': f"${position['take_profit']:,.0f}",
            '理由': position['reason']
        }

        self.logs.append(log_entry)
        self._save()

    def log_close(self,
                  close_time: datetime,
                  pnl_ratio: float,
                  pnl_amount: float,
                  balance: float,
                  reason: str):
        """记录平仓"""
        log_entry = {
            '时间': close_time.strftime('%Y-%m-%d %H:%M:%S'),
            '动作': 'CLOSE',
            '盈亏': f"{pnl_ratio:+.2%}",
            '金额': f"${pnl_amount:+,.0f}",
            '余额': f"${balance:,.0f}",
            '理由': reason
        }

        self.logs.append(log_entry)
        self._save()

    def _save(self):
        """保存到CSV"""
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_file, index=False, encoding='utf-8-sig')


class DataSaver:
    """7.3 数据保存器"""

    def __init__(self, data_file: str = 'realtime_data_验证5.csv'):
        self.data_file = data_file
        self.data: List[Dict] = []

    def save_market_data(self,
                        timestamp: datetime,
                        btc_price: float,
                        tension: float,
                        acceleration: float,
                        dxy_fuel: float,
                        signal_type: str,
                        confidence: float):
        """保存市场数据"""
        data_entry = {
            '时间': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'BTC价格': btc_price,
            '张力': tension,
            '加速度': acceleration,
            'DXY燃料': dxy_fuel,
            '信号类型': signal_type,
            '置信度': confidence
        }

        self.data.append(data_entry)

        # 保存到CSV
        df = pd.DataFrame(self.data)
        df.to_csv(self.data_file, index=False, encoding='utf-8-sig')


# ==================== 模块8：主循环控制 ====================

class StateManager:
    """8.2 状态管理器"""

    def __init__(self):
        self.current_position: Optional[Dict] = None
        self.last_signal: Optional[Dict] = None
        self.market_data: Optional[Dict] = None

    def update_position(self, position: Optional[Dict]):
        """更新持仓"""
        self.current_position = position

    def update_signal(self, signal: Optional[Dict]):
        """更新信号"""
        self.last_signal = signal

    def update_market_data(self, data: Optional[Dict]):
        """更新市场数据"""
        self.market_data = data

    def has_position(self) -> bool:
        """是否有持仓"""
        return self.current_position is not None


# ==================== 模块9：数据收集（独立进程） ====================

class DataCollector:
    """9.1 数据收集器（独立进程）"""

    def __init__(self, interval_hours: int = 4):
        self.interval_hours = interval_hours

        # 初始化模块
        self.btc_fetcher = BTCDataFetcher()
        self.physics_calculator = PhysicsIndicatorCalculator()
        self.data_saver = DataSaver('realtime_data_验证5.csv')

    def collect_once(self):
        """收集一次数据"""
        try:
            print(f"\n[{datetime.now()}] 开始收集数据...")

            # 获取BTC数据
            btc_df = self.btc_fetcher.fetch()
            if btc_df is None:
                return

            # 计算指标
            prices = btc_df['close'].tail(100).values
            tension, acceleration = self.physics_calculator.calculate_all(prices)

            # 保存数据
            self.data_saver.save_market_data(
                timestamp=datetime.now(),
                btc_price=btc_df['close'].iloc[-1],
                tension=tension if tension else 0,
                acceleration=acceleration if acceleration else 0,
                dxy_fuel=0,
                signal_type='N/A',
                confidence=0
            )

            print(f"[{datetime.now()}] 数据收集完成")

        except Exception as e:
            print(f"[ERROR] 数据收集失败: {e}")

    def start(self):
        """启动数据收集"""
        print(f"\n数据收集系统启动，每{self.interval_hours}小时收集一次")
        print("按 Ctrl+C 停止\n")

        # 立即执行一次
        self.collect_once()

        try:
            while True:
                time.sleep(self.interval_hours * 3600)
                self.collect_once()

        except KeyboardInterrupt:
            print("\n\n数据收集系统停止")


# ==================== 完整交易系统 ====================

class LiveTradingSystem:
    """完整实盘交易系统（整合所有模块）"""

    def __init__(self,
                 account_balance: float = 10000,
                 risk_per_trade: float = 0.02,
                 interval_seconds: int = 300):

        # 模块1：数据获取
        self.btc_fetcher = BTCDataFetcher()
        self.dxy_fetcher = DXYDataFetcher()
        self.data_cache = DataCache(max_size=100)

        # 模块2：物理指标计算
        self.physics_calculator = PhysicsIndicatorCalculator()

        # 模块3：市场状态诊断
        self.market_classifier = MarketStateClassifier()

        # 模块4：交易决策
        self.v8_strategy = V8ReverseStrategy()

        # 模块5：风险管理
        self.position_monitor = None

        # 模块6：订单执行
        self.order_executor = OrderExecutor()
        self.account_manager = AccountManager(account_balance)

        # 模块7：日志记录
        self.trading_logger = TradingLogger()
        self.data_saver = DataSaver()

        # 模块8：状态管理
        self.state_manager = StateManager()

        # 配置
        self.risk_per_trade = risk_per_trade
        self.interval_seconds = interval_seconds

    def check_and_trade(self):
        """检查并交易（主循环回调）"""
        try:
            print(f"\n{'='*80}")
            print(f"信号检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            # 第0层：数据获取
            print("\n[第0层] 数据获取...")
            btc_df = self.btc_fetcher.fetch()
            if btc_df is None:
                return

            dxy_df = self.dxy_fetcher.fetch()
            dxy_latest = dxy_df['Close'].iloc[-1] if dxy_df is not None else None

            current_price = btc_df['close'].iloc[-1]
            current_volume = btc_df['volume'].iloc[-1]

            # 更新缓存
            self.data_cache.update_price(current_price, current_volume)
            if dxy_latest:
                self.data_cache.update_dxy(dxy_latest)

            # 检查数据是否足够
            if not self.data_cache.is_ready():
                print("  数据不足，等待更多数据...")
                return

            # 第1层：物理指标计算
            print("\n[第1层] 物理指标计算...")
            prices = self.data_cache.get_prices_array()
            tension, acceleration = self.physics_calculator.calculate_all(prices)

            if tension is None:
                print("  指标计算失败")
                return

            dxy_fuel = calculate_dxy_fuel(self.data_cache.dxy_history)

            print(f"  张力: {tension:.4f}")
            print(f"  加速度: {acceleration:.6f}")
            print(f"  DXY燃料: {dxy_fuel:.2f}")

            # 第2层：市场状态诊断
            print("\n[第2层] 市场状态诊断...")
            signal_type, description, confidence = self.market_classifier.classify(
                tension, acceleration, dxy_fuel
            )

            print(f"  信号类型: {signal_type}")
            print(f"  描述: {description}")
            print(f"  置信度: {confidence:.1%}")

            # 保存市场数据
            self.data_saver.save_market_data(
                timestamp=datetime.now(),
                btc_price=current_price,
                tension=tension,
                acceleration=acceleration,
                dxy_fuel=dxy_fuel,
                signal_type=signal_type,
                confidence=confidence
            )

            # 置信度过滤
            if confidence < 0.6:
                print("\n  → 置信度不足，跳过")
                return

            # 第3层：交易决策
            print("\n[第3层] 交易决策...")
            action, reason = self.v8_strategy.get_action(signal_type)

            if action == 'WAIT':
                print(f"  → 观望：{reason}")
                return

            # 构造信号
            signal = {
                'action': action,
                'signal_type': signal_type,
                'confidence': confidence,
                'current_price': current_price,
                'tension': tension,
                'acceleration': acceleration,
                'dxy_fuel': dxy_fuel,
                'reason': reason
            }

            print(f"  动作: {action}")
            print(f"  理由: {reason}")

            # 第4层：风险管理 + 订单执行
            if not self.state_manager.has_position():
                # 开仓
                print("\n[第4层] 执行开仓...")
                position = self.order_executor.open_position(
                    signal,
                    self.account_manager.balance,
                    self.risk_per_trade
                )

                self.state_manager.update_position(position)
                self.trading_logger.log_open(position)

                print(f"\n  ✅ 开仓执行:")
                print(f"     方向: {position['side']}")
                print(f"     价格: ${position['entry_price']:,.0f}")
                print(f"     仓位: ${position['size']:,.0f}")
                print(f"     止损: ${position['stop_loss']:,.0f}")
                print(f"     止盈: ${position['take_profit']:,.0f}")
                print(f"     理由: {position['reason']}")

            else:
                # 有持仓，检查是否需要平仓
                print("\n[第4层] 检查持仓...")
                self._check_position(current_price, confidence)

        except Exception as e:
            print(f"[ERROR] check_and_trade失败: {e}")

    def _check_position(self, current_price: float, current_confidence: float):
        """检查持仓"""
        try:
            position = self.state_manager.current_position

            # 初始化监控器
            if self.position_monitor is None:
                self.position_monitor = PositionMonitor(position)

            # 检查
            should_close, reason = self.position_monitor.check(
                current_price, current_confidence
            )

            # 计算当前盈亏
            if position['side'] == 'LONG':
                pnl_ratio = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_ratio = (position['entry_price'] - current_price) / position['entry_price']

            print(f"  持仓: {position['side']} | PnL {pnl_ratio:+.2%}")

            if should_close:
                # 平仓
                print(f"\n  ⚠️  {reason}，平仓")

                pnl_ratio, description = self.order_executor.close_position(
                    position, current_price, reason
                )

                # 更新账户
                pnl_amount = position['size'] * pnl_ratio
                self.account_manager.update_balance(pnl_amount)

                # 记录日志
                self.trading_logger.log_close(
                    datetime.now(),
                    pnl_ratio,
                    pnl_amount,
                    self.account_manager.balance,
                    reason
                )

                print(f"  平仓: {description}")
                print(f"  余额: ${self.account_manager.balance:,.0f}")

                # 清除状态
                self.state_manager.update_position(None)
                self.position_monitor = None

        except Exception as e:
            print(f"[ERROR] 检查持仓失败: {e}")

    def run(self):
        """运行系统"""
        print("\n" + "="*80)
        print("V8.0 完整模块化系统 - 基于验证5逻辑")
        print("="*80)

        print(f"\n[配置]")
        print(f"  账户余额: ${self.account_manager.balance:,.0f}")
        print(f"  单笔风险: {self.risk_per_trade:.1%}")
        print(f"  检查间隔: {self.interval_seconds}秒")

        print(f"\n[模块清单]")
        print(f"  ✅ 模块1: 数据获取层 (BTC + DXY)")
        print(f"  ✅ 模块2: 物理指标计算 (FFT + Hilbert)")
        print(f"  ✅ 模块3: 市场状态诊断")
        print(f"  ✅ 模块4: 交易决策 (V8.0反向策略)")
        print(f"  ✅ 模块5: 风险管理")
        print(f"  ✅ 模块6: 订单执行")
        print(f"  ✅ 模块7: 日志记录")
        print(f"  ✅ 模块8: 状态管理")

        print(f"\n[启动]")
        print(f"  系统将每{self.interval_seconds//60}分钟检查一次")
        print(f"  按 Ctrl+C 停止\n")

        try:
            while True:
                self.check_and_trade()
                time.sleep(self.interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n[停止] 交易系统已停止")
            print(f"[最终余额] ${self.account_manager.balance:,.0f}")
            perf = self.account_manager.get_performance()
            print(f"[总回报] {perf['total_return']:+.2%}")
            print(f"[交易日志] live_trading_验证5逻辑.csv")


# ==================== 主程序 ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        # 数据收集模式
        collector = DataCollector(interval_hours=4)
        collector.start()
    else:
        # 实盘交易模式
        system = LiveTradingSystem(
            account_balance=10000,
            risk_per_trade=0.02,
            interval_seconds=300  # 5分钟
        )
        system.run()
