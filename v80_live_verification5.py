# -*- coding: utf-8 -*-
"""
V8.0 实盘交易系统 - 基于验证5逻辑

完全按照 step1_collect_验证5完全一致.py 的逻辑

数据源:
1. Binance API (BTC 4小时数据)
2. FRED API (DXY美元指数数据)

核心逻辑:
- FFT滤波 + Hilbert变换计算张力
- 加速度 = 张力的二阶差分
- DXY燃料 = -DXY加速度 * 100

作者: Claude Sonnet
日期: 2026-01-22
版本: v2.0 (基于验证5逻辑)
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert, detrend
from scipy.interpolate import interp1d
import requests
from io import StringIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构 ====================

@dataclass
class MarketData:
    """市场数据"""
    timestamp: datetime
    btc_price: float
    btc_volume: float
    btc_high: float
    btc_low: float
    dxy_close: Optional[float] = None


@dataclass
class PhysicsMetrics:
    """物理指标（验证5逻辑）"""
    tension: float  # 张力 (Hilbert变换后的虚部)
    acceleration: float  # 加速度 (张力的二阶差分)
    dxy_fuel: float  # DXY燃料


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: datetime
    action: str  # LONG, SHORT, SKIP, WAIT
    position_size: float
    reason: str
    signal_type: str
    confidence: float
    metrics: Dict


# ==================== 验证5逻辑引擎 ====================

class Verification5Engine:
    """
    验证5逻辑引擎

    完全按照 step1_collect_验证5完全一致.py 的实现
    """

    def __init__(self):
        self.window_size = 100  # 计算窗口

        # 验证5参数
        self.TENSION_THRESHOLD = 0.35
        self.ACCEL_THRESHOLD = 0.02
        self.OSCILLATION_BAND = 0.5

        # 数据缓存
        self.price_history: List[float] = []
        self.dxy_history: List[float] = []

    def fetch_btc_data(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取BTC 4小时数据（第0层-A）

        数据源: Binance API
        """
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            print(f"[第0层-A BTC数据] 获取成功: {len(df)}条")
            print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")

            return df

        except Exception as e:
            print(f"[ERROR] BTC数据获取失败: {e}")
            return None

    def fetch_dxy_data(self, days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        获取DXY美元指数数据（第0层-B）

        数据源: FRED (Federal Reserve Economic Data)
        """
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
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

            # 只取最近的数据
            cutoff_date = datetime.now() - timedelta(days=days_back)
            dxy_df = dxy_df[dxy_df.index >= cutoff_date]

            print(f"[第0层-B DXY数据] 获取成功: {len(dxy_df)}条")

            return dxy_df

        except Exception as e:
            print(f"[WARNING] DXY数据获取失败: {e}")
            return None

    def calculate_tension_acceleration(self, prices: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        计算张力和加速度（验证5逻辑 - 关键！）

        关键: 加速度 = 张力的二阶差分
        使用 FFT滤波 + Hilbert变换
        """
        if len(prices) < 3:
            return None, None

        try:
            # 去趋势
            d_prices = detrend(prices)

            # FFT滤波 (保留前8个频率分量)
            coeffs = fft(d_prices)
            coeffs[8:] = 0
            filtered = ifft(coeffs).real

            # Hilbert变换
            analytic = hilbert(filtered)
            tension = np.imag(analytic)

            # 标准化张力
            if len(tension) > 1 and np.std(tension) > 0:
                norm_tension = (tension - np.mean(tension)) / np.std(tension)
            else:
                norm_tension = tension

            # 【关键】计算加速度：张力的二阶差分
            if len(norm_tension) >= 3:
                current_tension = norm_tension[-1]
                prev_tension = norm_tension[-2]
                prev2_tension = norm_tension[-3]

                # 速度 = 张力的一阶差分
                velocity = current_tension - prev_tension

                # 加速度 = 速度的一阶差分（张力的二阶差分）
                acceleration = velocity - (prev_tension - prev2_tension)
            else:
                acceleration = 0.0

            return float(norm_tension[-1]), float(acceleration)

        except Exception as e:
            print(f"[ERROR] 物理指标计算失败: {e}")
            return None, None

    def calculate_dxy_fuel(self, dxy_df: Optional[pd.DataFrame], current_date: datetime) -> float:
        """
        计算DXY燃料（验证5逻辑）

        DXY燃料 = -DXY加速度 * 100
        """
        if dxy_df is None or dxy_df.empty:
            return 0.0

        try:
            mask = dxy_df.index <= current_date
            recent = dxy_df[mask].tail(5)

            if len(recent) < 3:
                return 0.0

            closes = recent['Close'].values.astype(float)

            change_1 = (closes[-1] - closes[-2]) / closes[-2]
            change_2 = (closes[-2] - closes[-3]) / closes[-3] if len(closes) >= 3 else change_1

            acceleration = change_1 - change_2
            fuel = -acceleration * 100

            return float(fuel)

        except:
            return 0.0

    def diagnose_regime(self, tension: float, acceleration: float, dxy_fuel: float = 0.0) -> Tuple[str, str, float]:
        """
        诊断市场状态（验证5逻辑）

        返回: (信号类型, 描述, 置信度)
        """
        # 1. 奇点看空
        if tension > self.TENSION_THRESHOLD and acceleration < -self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.1:
                return "BEARISH_SINGULARITY", "强奇点看空 (宏观失速)", 0.9
            else:
                return "BEARISH_SINGULARITY", "奇点看空 (动力失速)", 0.7

        # 2. 奇点看涨
        if tension < -self.TENSION_THRESHOLD and acceleration > self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.2:
                return "BULLISH_SINGULARITY", "超强奇点看涨 (燃料爆炸)", 0.95
            elif dxy_fuel > 0:
                return "BULLISH_SINGULARITY", "强奇点看涨 (动力回归)", 0.8
            else:
                return "BULLISH_SINGULARITY", "奇点看涨 (弹性释放)", 0.6

        # 3. 震荡
        if abs(tension) < self.OSCILLATION_BAND and abs(acceleration) < 0.02:
            return "OSCILLATION", "系统平衡 (震荡收敛)", 0.8

        # 4. 高位震荡
        if tension > 0.3 and abs(acceleration) < 0.01:
            return "HIGH_OSCILLATION", "高位震荡 (风险积聚)", 0.6

        # 5. 低位震荡
        if tension < -0.3 and abs(acceleration) < 0.01:
            return "LOW_OSCILLATION", "低位震荡 (机会积聚)", 0.6

        # 6. 过渡状态
        if tension > 0 and acceleration > 0:
            return "TRANSITION_UP", "向上过渡 (蓄力)", 0.4
        elif tension < 0 and acceleration < 0:
            return "TRANSITION_DOWN", "向下过渡 (泄力)", 0.4

        return "TRANSITION", "体制切换中", 0.3

    def analyze_current_state(self, dxy_df: Optional[pd.DataFrame]) -> Optional[Dict]:
        """
        分析当前市场状态（完整流程）
        """
        # 获取BTC数据
        btc_df = self.fetch_btc_data(limit=1000)
        if btc_df is None:
            return None

        # 获取DXY数据
        if dxy_df is None:
            dxy_df = self.fetch_dxy_data(days_back=30)

        # 更新价格历史
        current_price = btc_df['close'].iloc[-1]
        self.price_history.append(current_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)

        # 确保有足够数据计算物理指标
        if len(self.price_history) < 60:
            print(f"[第1层] 数据不足 (当前: {len(self.price_history)} < 60)")
            return None

        # 第1层: 计算物理指标（验证5逻辑）
        prices_array = np.array(self.price_history)
        tension, acceleration = self.calculate_tension_acceleration(prices_array)

        if tension is None or acceleration is None:
            return None

        # 计算DXY燃料
        current_date = datetime.now()
        dxy_fuel = self.calculate_dxy_fuel(dxy_df, current_date)

        # 第2层: 诊断市场状态
        signal_type, description, confidence = self.diagnose_regime(tension, acceleration, dxy_fuel)

        return {
            'timestamp': current_date,
            'btc_price': current_price,
            'tension': tension,
            'acceleration': acceleration,
            'dxy_fuel': dxy_fuel,
            'signal_type': signal_type,
            'description': description,
            'confidence': confidence
        }


# ==================== 实盘交易引擎 ====================

class LiveTradingEngine:
    """实盘交易引擎 - 基于验证5逻辑"""

    def __init__(self, account_balance: float = 10000, risk_per_trade: float = 0.02):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade

        # 初始化验证5引擎
        self.v5_engine = Verification5Engine()
        self.dxy_df = None  # DXY数据缓存

        # 持仓
        self.position: Optional[Dict] = None

        # 日志
        self.trade_log: List[Dict] = []

        print(f"[初始化] 基于验证5逻辑的实盘交易引擎")
        print(f"  账户余额: ${account_balance:,.0f}")
        print(f"  单笔风险: {risk_per_trade:.1%}")

    def check_signals(self) -> Optional[TradingSignal]:
        """检查交易信号"""
        print(f"\n{'='*100}")
        print(f"信号检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")

        # 第0层: 数据获取
        print(f"\n[第0层] 数据获取...")

        # 首次获取DXY数据（缓存）
        if self.dxy_df is None:
            self.dxy_df = self.v5_engine.fetch_dxy_data(days_back=30)

        # 分析当前状态
        analysis = self.v5_engine.analyze_current_state(self.dxy_df)
        if not analysis:
            print("  [跳过] 数据不足或获取失败")
            return None

        # 显示分析结果
        print(f"\n[第1层 物理指标]")
        print(f"  BTC价格: ${analysis['btc_price']:,.0f}")
        print(f"  张力: {analysis['tension']:.4f}")
        print(f"  加速度: {analysis['acceleration']:.6f}")
        print(f"  DXY燃料: {analysis['dxy_fuel']:.2f}")

        # 第2层: 市场状态诊断
        signal_type = analysis['signal_type']
        confidence = analysis['confidence']
        description = analysis['description']

        print(f"\n[第2层 市场状态]")
        print(f"  信号类型: {signal_type}")
        print(f"  描述: {description}")
        print(f"  置信度: {confidence:.1%}")

        # 只处理高置信度信号
        if confidence < 0.6:
            print(f"\n  → 置信度不足，跳过")
            return None

        # 第3层: 交易决策（V8.0反向策略）
        action, position_size, reason = self.make_decision_v8_reverse(signal_type, confidence, analysis)

        if action == 'WAIT':
            print(f"\n  → 观望")
            return None

        print(f"\n[第3层 交易决策]")
        print(f"  动作: {action}")
        print(f"  仓位: {position_size*100:.0f}%")
        print(f"  理由: {reason}")

        return TradingSignal(
            timestamp=datetime.now(),
            action=action,
            position_size=position_size,
            reason=reason,
            signal_type=signal_type,
            confidence=confidence,
            metrics=analysis
        )

    def make_decision_v8_reverse(self, signal_type: str, confidence: float, analysis: Dict) -> Tuple[str, float, str]:
        """
        V8.0反向策略决策

        逻辑: "系统看空我做多，系统看涨我做空"
        """
        base_size = 1.0
        action = 'WAIT'
        reason = ''

        # 根据信号类型决策
        if signal_type == 'BEARISH_SINGULARITY':
            action = 'LONG'
            base_size = 1.0 + (confidence - 0.6) * 0.5  # 0.6→1.0, 0.9→1.15
            reason = 'BEARISH_SINGULARITY → 反向做多 (抄底)'

        elif signal_type == 'BULLISH_SINGULARITY':
            action = 'SHORT'
            base_size = 1.0 + (confidence - 0.6) * 0.5
            reason = 'BULLISH_SINGULARITY → 反向做空 (逃顶)'

        elif signal_type == 'LOW_OSCILLATION':
            action = 'LONG'
            base_size = 1.0
            reason = 'LOW_OSCILLATION → 低位做多'

        elif signal_type == 'HIGH_OSCILLATION':
            action = 'SHORT'
            base_size = 1.0
            reason = 'HIGH_OSCILLATION → 高位做空'

        elif signal_type == 'OSCILLATION':
            action = 'WAIT'
            base_size = 0
            reason = 'OSCILLATION → 震荡观望'

        # DXY燃料增强
        dxy_fuel = analysis.get('dxy_fuel', 0)
        if dxy_fuel > 0.2 and action in ['LONG', 'SHORT']:
            base_size *= 1.2  # 增加20%仓位
            reason += f' + DXY燃料增强({dxy_fuel:.2f})'

        return action, base_size, reason

    def execute_trade(self, signal: TradingSignal):
        """执行交易"""
        if signal.action == 'WAIT':
            print(f"\n  → 观望，不交易")
            return

        # 计算仓位
        max_position_by_risk = self.account_balance * (self.risk_per_trade / 0.03)
        position_value = self.account_balance * signal.position_size
        actual_position = min(position_value, max_position_by_risk)

        # 开仓
        self.position = {
            'entry_time': datetime.now(),
            'entry_price': signal.metrics['btc_price'],
            'size': actual_position,
            'side': signal.action,
            'stop_loss': signal.metrics['btc_price'] * (0.97 if signal.action == 'LONG' else 1.03),
            'take_profit': signal.metrics['btc_price'] * (1.10 if signal.action == 'LONG' else 0.90),
            'signal_type': signal.signal_type,
            'confidence': signal.confidence
        }

        print(f"\n  ✅ 开仓执行:")
        print(f"     方向: {self.position['side']}")
        print(f"     价格: ${self.position['entry_price']:,.0f}")
        print(f"     仓位: ${self.position['size']:,.0f}")
        print(f"     止损: ${self.position['stop_loss']:,.0f}")
        print(f"     止盈: ${self.position['take_profit']:,.0f}")

        # 记录日志
        self.log_trade(signal, 'OPEN')

    def check_position(self):
        """检查持仓"""
        if self.position is None:
            return

        # 获取当前价格
        analysis = self.v5_engine.analyze_current_state(self.dxy_df)
        if not analysis:
            return

        current_price = analysis['btc_price']

        # 计算盈亏
        if self.position['side'] == 'LONG':
            pnl_ratio = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:
            pnl_ratio = (self.position['entry_price'] - current_price) / self.position['entry_price']

        # 检查平仓条件
        should_close = False
        close_reason = ""

        # 止损
        if pnl_ratio < -0.03:
            should_close = True
            close_reason = f"止损 ({pnl_ratio:.2%})"

        # 止盈
        elif pnl_ratio > 0.10:
            should_close = True
            close_reason = f"止盈 ({pnl_ratio:.2%})"

        # 信号消失
        elif analysis['confidence'] < 0.5:
            should_close = True
            close_reason = "信号消失"

        if should_close:
            self.close_position(close_reason, current_price)
        else:
            print(f"  持仓中: {self.position['side']} | PnL {pnl_ratio:+.2%}")

    def close_position(self, reason: str, current_price: float):
        """平仓"""
        if self.position is None:
            return

        # 计算盈亏
        if self.position['side'] == 'LONG':
            pnl_ratio = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:
            pnl_ratio = (self.position['entry_price'] - current_price) / self.position['entry_price']

        pnl_amount = self.position['size'] * pnl_ratio
        self.account_balance += pnl_amount

        print(f"\n  ❌ 平仓:")
        print(f"     理由: {reason}")
        print(f"     价格: ${current_price:,.0f}")
        print(f"     盈亏: {pnl_ratio:+.2%} (${pnl_amount:+,.0f})")
        print(f"     余额: ${self.account_balance:,.0f}")

        # 记录日志
        self.log_trade(None, 'CLOSE', pnl_ratio, reason)

        self.position = None

    def log_trade(self, signal: Optional[TradingSignal], action: str, pnl: float = 0, reason: str = ""):
        """记录日志"""
        log_entry = {
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '动作': action,
            '信号类型': signal.signal_type if signal else '',
            '置信度': f"{signal.confidence:.1%}" if signal else '',
            '方向': signal.action if signal else '',
            '理由': signal.reason if signal else reason,
            '盈亏': f"{pnl:.2%}",
            '余额': f"${self.account_balance:,.0f}"
        }

        self.trade_log.append(log_entry)

        # 保存到CSV
        df = pd.DataFrame(self.trade_log)
        df.to_csv('live_trading_验证5逻辑.csv', index=False, encoding='utf-8-sig')

    def run(self, interval_seconds: int = 300):
        """运行实盘循环"""
        print(f"\n{'='*100}")
        print(f"V8.0 实盘交易系统 - 基于验证5逻辑")
        print(f"{'='*100}")
        print(f"\n[完整逻辑]")
        print(f"  第0层: 数据获取 (BTC + DXY)")
        print(f"  第1层: 物理指标计算 (FFT + Hilbert)")
        print(f"  第2层: 市场状态诊断")
        print(f"  第3层: V8.0反向策略")
        print(f"  第4层: 风险管理")

        print(f"\n[配置]")
        print(f"  检查间隔: {interval_seconds}秒")
        print(f"  置信度阈值: 0.6")
        print(f"  风险控制: {self.risk_per_trade:.1%} per trade")

        print(f"\n[启动]")
        print(f"  系统将每{interval_seconds//60}分钟检查一次")
        print(f"  按 Ctrl+C 停止\n")

        try:
            while True:
                # 检查信号
                if self.position is None:
                    signal = self.check_signals()
                    if signal:
                        self.execute_trade(signal)
                else:
                    # 检查持仓
                    self.check_position()

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n[停止] 交易引擎已停止")
            print(f"[最终余额] ${self.account_balance:,.0f}")
            print(f"[交易日志] live_trading_验证5逻辑.csv")


if __name__ == "__main__":
    print("="*100)
    print("V8.0 实盘交易系统 - 基于验证5逻辑")
    print("="*100)

    engine = LiveTradingEngine(
        account_balance=10000,
        risk_per_trade=0.02
    )

    engine.run(interval_seconds=300)  # 每5分钟检查
