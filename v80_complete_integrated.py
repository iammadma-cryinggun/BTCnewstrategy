# -*- coding: utf-8 -*-
"""
V8.0 完整系统 - 验证5 + 期权微观结构整合版

整合内容:
1. 验证5逻辑（FFT + Hilbert张力计算）
2. 期权微观结构（Gamma陷阱、Vanna挤压）
3. 订单流监控（CVD、VPIN、有毒流量）

完整决策流程:
  第0层: 数据获取（BTC + DXY + 期权数据）
  第1层: 验证5物理指标计算
  第2层: 期权微观结构分析
  第3层: 订单流监控
  第4层: 综合诊断
  第5层: V8.0反向策略
  第6层: 风险管理

版本: v4.0 Complete Integrated
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


# ==================== 数据结构 ====================

@dataclass
class OptionsGreeks:
    """期权Greeks数据"""
    gex: float  # Gamma暴露（USD）
    call_skew: float  # Call Skew
    put_skew: float  # Put Skew
    skew_slope: float  # Skew斜率
    call_oi_above: float  # 上方Call持仓量
    iv_percentile: float  # IV历史分位数


@dataclass
class OrderFlowMetrics:
    """订单流指标"""
    cvd: float  # 累积成交量差
    cvd_trend: float  # CVD趋势
    price_trend: float  # 价格趋势
    vpin: float  # 有毒流量指标
    toxic_flow_ratio: float  # 有毒流量占比
    sell_pressure: float  # 卖压
    buy_pressure: float  # 买压
    liquidity_drop_rate: float  # 流动性撤单速度


@dataclass
class TradingSignal:
    """综合交易信号"""
    timestamp: datetime
    action: str  # LONG, SHORT, WAIT, EMERGENCY_CLOSE
    position_size: float
    reason: str
    verification5_signal: str
    microstructure_signal: str
    confidence: float
    metrics: Dict


# ==================== 验证5逻辑模块（第0-1层）====================

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
            print(f"[ERROR] BTC数据获取失败: {e}")
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
            print(f"[WARNING] DXY数据获取失败: {e}")
            return None


def calculate_tension_acceleration_verification5(prices: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    计算张力和加速度（验证5逻辑 - 核心！）

    关键：加速度 = 张力的二阶差分
    """
    if len(prices) < 3:
        return None, None

    try:
        # 1. 去趋势
        d_prices = detrend(prices)

        # 2. FFT滤波（保留前8个频率分量）
        coeffs = fft(d_prices)
        coeffs[8:] = 0
        filtered = ifft(coeffs).real

        # 3. Hilbert变换
        analytic = hilbert(filtered)
        tension = np.imag(analytic)

        # 4. 标准化张力
        if len(tension) > 1 and np.std(tension) > 0:
            norm_tension = (tension - np.mean(tension)) / np.std(tension)
        else:
            norm_tension = tension

        # 5. 计算加速度（关键！）
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
        print(f"[ERROR] 物理指标计算失败: {e}")
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
        print(f"[ERROR] DXY燃料计算失败: {e}")
        return 0.0


# ==================== 期权微观结构模块（第2层）====================

class NegativeGammaTrapDetector:
    """
    负Gamma陷阱检测器（第2层-A）

    检测即将发生的闪崩
    """

    def __init__(self):
        self.GEX_THRESHOLD = -100000000  # GEX < -1亿USD
        self.LCR_THRESHOLD = 1.0  # LCR < 1.0
        self.VPIN_THRESHOLD = 0.4  # VPIN > 0.4
        self.TOXIC_FLOW_THRESHOLD = 0.3  # 有毒流量 > 30%

    def detect(self,
               greeks: Optional[OptionsGreeks],
               orderflow: Optional[OrderFlowMetrics]) -> Tuple[str, str, float]:
        """
        检测负Gamma陷阱

        返回: (信号类型, 描述, 置信度)
        """
        if greeks is None or orderflow is None:
            return "NORMAL", "无期权数据", 0.0

        trigger_reasons = []
        confidence = 0.0

        # 条件1: 负Gamma区域
        if greeks.gex < self.GEX_THRESHOLD:
            trigger_reasons.append(f"GEX={greeks.gex/1e8:.2f}亿 (负Gamma)")
            confidence += 0.3

        # 条件2: Skew斜率
        if greeks.skew_slope > 5.0:
            trigger_reasons.append(f"Skew斜率={greeks.skew_slope:.2f} (聪明钱买保护)")
            confidence += 0.2

        # 条件3: 有毒流量
        if orderflow.toxic_flow_ratio > self.TOXIC_FLOW_THRESHOLD:
            trigger_reasons.append(f"有毒流量={orderflow.toxic_flow_ratio:.1%}")
            confidence += 0.25

        # 条件4: VPIN
        if orderflow.vpin > self.VPIN_THRESHOLD:
            trigger_reasons.append(f"VPIN={orderflow.vpin:.3f} (知情交易)")
            confidence += 0.15

        # 条件5: 流动性撤单
        if orderflow.liquidity_drop_rate < -50000:
            trigger_reasons.append("流动性快速撤离")
            confidence += 0.1

        # 判断
        if confidence >= 0.5:
            return "NEGATIVE_GAMMA_TRAP", " | ".join(trigger_reasons), min(confidence, 0.95)

        return "NORMAL", "市场正常", 0.0


class VannaSqueezeDetector:
    """
    Vanna挤压检测器（第2层-B）

    检测即将发生的暴涨
    """

    def __init__(self):
        self.IV_PERCENTILE_THRESHOLD = 30.0  # IV < 30分位
        self.CALL_SKEW_THRESHOLD = 3.0  # Call Skew > 3.0

    def detect(self,
               greeks: Optional[OptionsGreeks],
               orderflow: Optional[OrderFlowMetrics]) -> Tuple[str, str, float]:
        """
        检测Vanna挤压

        返回: (信号类型, 描述, 置信度)
        """
        if greeks is None or orderflow is None:
            return "NORMAL", "无期权数据", 0.0

        trigger_reasons = []
        confidence = 0.0

        # 条件1: IV低位
        if greeks.iv_percentile < self.IV_PERCENTILE_THRESHOLD:
            trigger_reasons.append(f"IV分位数={greeks.iv_percentile:.1f}% (低位)")
            confidence += 0.25

        # 条件2: Call Skew上升
        if greeks.call_skew > self.CALL_SKEW_THRESHOLD:
            trigger_reasons.append(f"Call Skew={greeks.call_skew:.2f} (看涨)")
            confidence += 0.2

        # 条件3: 上方Call Wall
        if greeks.call_oi_above > 500000000:
            trigger_reasons.append(f"上方Call={greeks.call_oi_above/1e8:.2f}亿 (Gamma Squeeze潜力)")
            confidence += 0.15

        # 条件4: CVD背离（吸筹）
        if orderflow.price_trend < 0 and orderflow.cvd_trend > 0.5:
            trigger_reasons.append("吸筹背离: 价格跌 + CVD涨")
            confidence += 0.25

        # 条件5: 买压 > 卖压
        if orderflow.buy_pressure > orderflow.sell_pressure * 1.5:
            trigger_reasons.append("买压优势明显")
            confidence += 0.15

        # 判断
        if confidence >= 0.5:
            return "VANNA_SQUEEZE", " | ".join(trigger_reasons), min(confidence, 0.95)

        return "NORMAL", "市场正常", 0.0


# ==================== 验证5信号分类（第2层整合）====================

class Verification5Classifier:
    """验证5信号分类器（整合微观结构）"""

    def __init__(self):
        # 验证5参数
        self.TENSION_THRESHOLD = 0.35
        self.ACCEL_THRESHOLD = 0.02
        self.OSCILLATION_BAND = 0.5

    def classify(self,
                tension: float,
                acceleration: float,
                dxy_fuel: float,
                gamma_signal: Tuple[str, str, float],
                vanna_signal: Tuple[str, str, float]) -> Tuple[str, str, float]:
        """
        综合分类市场状态

        参数:
            tension: 张力
            acceleration: 加速度
            dxy_fuel: DXY燃料
            gamma_signal: 负Gamma陷阱信号
            vanna_signal: Vanna挤压信号

        返回: (信号类型, 描述, 置信度)
        """

        # === 最高优先级：紧急平仓 ===
        if gamma_signal[0] == "NEGATIVE_GAMMA_TRAP" and gamma_signal[2] >= 0.7:
            return "EMERGENCY_CLOSE", f"【负Gamma陷阱】{gamma_signal[1]}", gamma_signal[2]

        # === 第二优先级：Vanna增强 ===
        if vanna_signal[0] == "VANNA_SQUEEZE" and vanna_signal[2] >= 0.7:
            # 如果验证5也给出做多信号，增强置信度
            if tension < -self.TENSION_THRESHOLD and acceleration > self.ACCEL_THRESHOLD:
                base_confidence = 0.6
                if dxy_fuel > 0:
                    base_confidence = 0.8
                if dxy_fuel > 0.2:
                    base_confidence = 0.95
                enhanced_confidence = min(base_confidence + vanna_signal[2] * 0.5, 0.98)
                return "VANNA_ENHANCED_LONG", f"【Vanna增强做多】{vanna_signal[1]}", enhanced_confidence

        # === 第三优先级：验证5信号（标准逻辑）===

        # 1. BEARISH_SINGULARITY（奇点看空）
        if tension > self.TENSION_THRESHOLD and acceleration < -self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.1:
                return "BEARISH_SINGULARITY", "强奇点看空 (宏观失速)", 0.9
            else:
                return "BEARISH_SINGULARITY", "奇点看空 (动力失速)", 0.7

        # 2. BULLISH_SINGULARITY（奇点看涨）
        if tension < -self.TENSION_THRESHOLD and acceleration > self.ACCEL_THRESHOLD:
            if dxy_fuel > 0.2:
                return "BULLISH_SINGULARITY", "超强奇点看涨 (燃料爆炸)", 0.95
            elif dxy_fuel > 0:
                return "BULLISH_SINGULARITY", "强奇点看涨 (动力回归)", 0.8
            else:
                return "BULLISH_SINGULARITY", "奇点看涨 (弹性释放)", 0.6

        # 3. OSCILLATION（震荡）
        if abs(tension) < self.OSCILLATION_BAND and abs(acceleration) < self.ACCEL_THRESHOLD:
            return "OSCILLATION", "系统平衡 (震荡收敛)", 0.8

        # 4. HIGH_OSCILLATION（高位震荡）
        if tension > 0.3 and abs(acceleration) < 0.01:
            return "HIGH_OSCILLATION", "高位震荡 (风险积聚)", 0.6

        # 5. LOW_OSCILLATION（低位震荡）
        if tension < -0.3 and abs(acceleration) < 0.01:
            return "LOW_OSCILLATION", "低位震荡 (机会积聚)", 0.6

        # 6. TRANSITION（过渡状态）
        if tension > 0 and acceleration > 0:
            return "TRANSITION_UP", "向上过渡 (蓄力)", 0.4
        elif tension < 0 and acceleration < 0:
            return "TRANSITION_DOWN", "向下过渡 (泄力)", 0.4

        return "TRANSITION", "体制切换中", 0.3


# ==================== V8.0反向策略（第5层）====================

class V8ReverseStrategy:
    """V8.0反向策略（整合微观结构增强）"""

    def __init__(self):
        self.strategy_map = {
            'BEARISH_SINGULARITY': ('LONG', '反向抄底'),
            'BULLISH_SINGULARITY': ('SHORT', '反向逃顶'),
            'LOW_OSCILLATION': ('LONG', '低位做多'),
            'HIGH_OSCILLATION': ('SHORT', '高位做空'),
            'VANNA_ENHANCED_LONG': ('LONG', 'Vanna增强做多'),
            'OSCILLATION': ('WAIT', '震荡观望'),
            'TRANSITION_UP': ('WAIT', '向上过渡'),
            'TRANSITION_DOWN': ('WAIT', '向下过渡'),
            'TRANSITION': ('WAIT', '体制切换'),
            'EMERGENCY_CLOSE': ('EMERGENCY_CLOSE', '紧急平仓')
        }

    def get_action(self, signal_type: str) -> Tuple[str, str]:
        """获取交易动作"""
        action, reason_base = self.strategy_map.get(signal_type, ('WAIT', '未知状态'))
        reason = f"{signal_type} → {reason_base}"
        return action, reason


# ==================== 完整整合系统 ====================

class CompleteIntegratedSystem:
    """
    完整整合系统：验证5 + 期权微观结构
    """

    def __init__(self,
                 account_balance: float = 10000,
                 risk_per_trade: float = 0.02):

        # 数据获取
        self.btc_fetcher = BTCDataFetcher()
        self.dxy_fetcher = DXYDataFetcher()

        # 数据缓存
        self.price_history: List[float] = []
        self.dxy_history: List[float] = []

        # 检测器
        self.v5_classifier = Verification5Classifier()
        self.gamma_detector = NegativeGammaTrapDetector()
        self.vanna_detector = VannaSqueezeDetector()
        self.v8_strategy = V8ReverseStrategy()

        # 账户管理
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.position: Optional[Dict] = None

        # 日志
        self.trade_log: List[Dict] = []

        print(f"[初始化] 完整整合系统 - 验证5 + 期权微观结构")
        print(f"  账户余额: ${account_balance:,.0f}")
        print(f"  单笔风险: {risk_per_trade:.1%}")

    def analyze_and_trade(self, use_options_data: bool = False):
        """
        完整分析和交易流程

        参数:
            use_options_data: 是否使用期权数据（如果有）
        """
        print(f"\n{'='*100}")
        print(f"完整分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")

        # 第0层：数据获取
        print("\n[第0层] 数据获取...")
        btc_df = self.btc_fetcher.fetch()
        if btc_df is None:
            print("  BTC数据获取失败")
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
            print(f"  数据不足 ({len(self.price_history)} < 60)")
            return

        print(f"  BTC价格: ${current_price:,.0f}")
        if dxy_df is not None:
            print(f"  DXY: {dxy_latest:.2f}")

        # 第1层：验证5物理指标计算
        print("\n[第1层] 验证5物理指标计算...")
        prices_array = np.array(self.price_history)
        tension, acceleration = calculate_tension_acceleration_verification5(prices_array)

        if tension is None:
            print("  指标计算失败")
            return

        dxy_fuel = calculate_dxy_fuel(self.dxy_history)

        print(f"  张力: {tension:.4f}")
        print(f"  加速度: {acceleration:.6f}")
        print(f"  DXY燃料: {dxy_fuel:.2f}")

        # 第2层：期权微观结构分析（如果可用）
        print("\n[第2层] 期权微观结构分析...")

        # 模拟期权数据（实际使用时应该从API获取）
        greeks = None
        orderflow = None

        if use_options_data:
            # 这里应该调用期权API获取Greeks和订单流数据
            # 暂时使用模拟数据
            print("  [模拟] 期权数据暂不可用，使用基础验证5逻辑")
        else:
            print("  期权数据未启用，使用基础验证5逻辑")

        # 检测微观结构信号
        gamma_signal = ("NORMAL", "市场正常", 0.0)
        vanna_signal = ("NORMAL", "市场正常", 0.0)

        if greeks is not None and orderflow is not None:
            gamma_signal = self.gamma_detector.detect(greeks, orderflow)
            vanna_signal = self.vanna_detector.detect(greeks, orderflow)

            print(f"  负Gamma陷阱: {gamma_signal[0]} (置信度: {gamma_signal[2]:.1%})")
            print(f"  Vanna挤压: {vanna_signal[0]} (置信度: {vanna_signal[2]:.1%})")

        # 第3层：综合诊断
        print("\n[第3层] 综合诊断...")
        signal_type, description, confidence = self.v5_classifier.classify(
            tension, acceleration, dxy_fuel,
            gamma_signal, vanna_signal
        )

        print(f"  信号类型: {signal_type}")
        print(f"  描述: {description}")
        print(f"  置信度: {confidence:.1%}")

        # 置信度过滤
        if confidence < 0.6 and signal_type != "EMERGENCY_CLOSE":
            print("\n  → 置信度不足，观望")
            return

        # 第4层：V8.0反向策略决策
        print("\n[第4层] V8.0反向策略决策...")
        action, reason = self.v8_strategy.get_action(signal_type)

        # 紧急平仓
        if action == 'EMERGENCY_CLOSE':
            print(f"\n  ⚠️  {reason}")
            if self.position is not None:
                self.close_position("紧急平仓：负Gamma陷阱", current_price)
            return

        if action == 'WAIT':
            print(f"  → 观望：{reason}")
            return

        print(f"  动作: {action}")
        print(f"  理由: {reason}")

        # 第5层：执行交易
        if self.position is None:
            # 计算仓位
            base_size = 1.0 + (confidence - 0.6) * 0.5

            # Vanna增强
            if signal_type == "VANNA_ENHANCED_LONG":
                base_size *= 1.2

            # 风险控制
            max_position = self.account_balance * (self.risk_per_trade / 0.03)
            position_value = min(self.account_balance * base_size, max_position)

            # 开仓
            self.open_position(action, current_price, position_value, signal_type, confidence, reason)

        else:
            # 检查持仓
            self.check_position(current_price, confidence)

    def open_position(self, action: str, price: float, size: float, signal_type: str, confidence: float, reason: str):
        """开仓"""
        stop_loss = price * (0.97 if action == 'LONG' else 1.03)
        take_profit = price * (1.10 if action == 'LONG' else 0.90)

        self.position = {
            'entry_time': datetime.now(),
            'entry_price': price,
            'size': size,
            'side': action,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_type': signal_type,
            'confidence': confidence,
            'reason': reason
        }

        print(f"\n  ✅ 开仓执行:")
        print(f"     方向: {action}")
        print(f"     价格: ${price:,.0f}")
        print(f"     仓位: ${size:,.0f}")
        print(f"     止损: ${stop_loss:,.0f}")
        print(f"     止盈: ${take_profit:,.0f}")

        # 记录日志
        self.trade_log.append({
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '动作': 'OPEN',
            '方向': action,
            '价格': f"${price:,.0f}",
            '仓位': f"${size:,.0f}",
            '信号': signal_type,
            '理由': reason
        })

    def check_position(self, current_price: float, current_confidence: float):
        """检查持仓"""
        if self.position is None:
            return

        # 计算盈亏
        if self.position['side'] == 'LONG':
            pnl_ratio = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:
            pnl_ratio = (self.position['entry_price'] - current_price) / self.position['entry_price']

        print(f"  持仓: {self.position['side']} | PnL {pnl_ratio:+.2%}")

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
        self.trade_log.append({
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '动作': 'CLOSE',
            '理由': reason,
            '盈亏': f"{pnl_ratio:+.2%}",
            '余额': f"${self.account_balance:,.0f}"
        })

        self.position = None

    def run(self, interval_seconds: int = 300, use_options_data: bool = False):
        """运行系统"""
        print("\n" + "="*100)
        print("V8.0 完整整合系统 - 验证5 + 期权微观结构")
        print("="*100)

        print(f"\n[完整架构]")
        print(f"  第0层: 数据获取（BTC + DXY + 期权）")
        print(f"  第1层: 验证5物理指标（FFT + Hilbert）")
        print(f"  第2层: 期权微观结构（Gamma + Vanna + 订单流）")
        print(f"  第3层: 综合诊断")
        print(f"  第4层: V8.0反向策略")
        print(f"  第5层: 风险管理")

        print(f"\n[配置]")
        print(f"  检查间隔: {interval_seconds}秒")
        print(f"  期权数据: {'启用' if use_options_data else '未启用'}")
        print(f"  置信度阈值: 0.6")

        print(f"\n[启动]")
        print(f"  系统将每{interval_seconds//60}分钟检查一次")
        print(f"  按 Ctrl+C 停止\n")

        try:
            while True:
                self.analyze_and_trade(use_options_data)
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n[停止] 系统已停止")
            print(f"[最终余额] ${self.account_balance:,.0f}")
            print(f"[交易日志] 已保存")

            # 保存日志
            if self.trade_log:
                df = pd.DataFrame(self.trade_log)
                df.to_csv('complete_integrated_trading_log.csv', index=False, encoding='utf-8-sig')
                print(f"[日志文件] complete_integrated_trading_log.csv")


if __name__ == "__main__":
    system = CompleteIntegratedSystem(
        account_balance=10000,
        risk_per_trade=0.02
    )

    # 运行系统（不启用期权数据）
    system.run(interval_seconds=300, use_options_data=False)
