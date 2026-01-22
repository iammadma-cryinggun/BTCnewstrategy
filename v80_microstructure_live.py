# -*- coding: utf-8 -*-
"""
V8.0 + 简化微观结构 实盘交易系统

完整逻辑:
1. V8.0突变检测 (测速仪)
2. Singularity信号分类
3. 简化微观结构过滤 (地图+仪表盘)
4. V8.0反向策略决策
5. 仓位管理
6. 风险管理

作者: Claude Sonnet 4.5
日期: 2026-01-22
版本: v1.0 Live Trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import time
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构 ====================

@dataclass
class MarketData:
    """市场数据"""
    timestamp: datetime
    price: float
    volume: float
    high: float
    low: float
    quote_volume: float
    price_change_percent: float


@dataclass
class TechnicalIndicators:
    """技术指标"""
    ema: float
    delta_ema: float
    delta_ema_abs: float
    tension: float
    acceleration: float
    volume_ratio: float
    delta_vol: float
    v8_score: float


@dataclass
class MicrostructureMetrics:
    """微观结构指标"""
    volatility: float
    liquidity_score: float
    is_crash_risk: bool
    crash_confidence: float
    is_squeeze: bool
    squeeze_confidence: float
    recommendation: str


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: datetime
    action: str  # LONG, SHORT, SKIP, WAIT
    position_size: float  # 杠杆倍数 0-2.5
    reason: str
    v8_score: float
    signal_type: str
    microstructure: Dict


@dataclass
class Position:
    """持仓信息"""
    entry_time: datetime
    entry_price: float
    size: float  # USD value
    side: str  # LONG, SHORT
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None


# ==================== V8.0引擎 ====================

class V80Engine:
    """
    V8.0动态策略引擎

    完整流程:
    1. 数据获取 (Binance API)
    2. 指标计算 (EMA, 量能比率, 张力, 加速度)
    3. V8.0评分
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.ema_history: List[float] = []

        print(f"[V8.0引擎初始化] lookback={lookback}")

    def fetch_market_data(self) -> Optional[MarketData]:
        """
        第0层: 数据获取

        从Binance API获取实时市场数据
        数据来源: https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT
        """
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            return MarketData(
                timestamp=datetime.now(),
                price=float(data['lastPrice']),
                volume=float(data['volume']),
                high=float(data['highPrice']),
                low=float(data['lowPrice']),
                quote_volume=float(data['quoteVolume']),
                price_change_percent=float(data['priceChangePercent'])
            )
        except Exception as e:
            print(f"[ERROR] 第0层: 数据获取失败 - {e}")
            return None

    def calculate_ema(self, price: float) -> float:
        """计算EMA"""
        if len(self.ema_history) == 0:
            self.ema_history.append(price)
            return price

        alpha = 2 / (self.lookback + 1)
        new_ema = alpha * price + (1 - alpha) * self.ema_history[-1]
        self.ema_history.append(new_ema)

        if len(self.ema_history) > 100:
            self.ema_history.pop(0)

        return new_ema

    def calculate_indicators(self, market_data: MarketData) -> TechnicalIndicators:
        """
        第1层: 基础指标计算

        从原始市场数据计算技术指标:
        - EMA (指数移动平均)
        - 张力 (Tension = (价格-EMA)/EMA)
        - 加速度 (价格变化率的变化)
        - 量能比率 (当前量/平均量)
        """
        # 更新历史
        self.price_history.append(market_data.price)
        self.volume_history.append(market_data.volume)

        if len(self.price_history) > 100:
            self.price_history.pop(0)
            self.volume_history.pop(0)

        # 计算EMA
        ema = self.calculate_ema(market_data.price)

        # Component 1: EMA突变
        delta_ema = (market_data.price - ema) / ema if ema > 0 else 0
        delta_ema_abs = abs(delta_ema)

        # 张力 = Delta_EMA
        tension = delta_ema

        # Component 2: 量能突变
        if len(self.volume_history) >= self.lookback:
            avg_volume = np.mean(self.volume_history[-self.lookback:])
        else:
            avg_volume = market_data.volume

        volume_ratio = market_data.volume / avg_volume if avg_volume > 0 else 1.0
        delta_vol = volume_ratio - 1.0

        # 加速度 (价格变化率的变化)
        if len(self.price_history) >= 3:
            change1 = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            change2 = (self.price_history[-2] - self.price_history[-3]) / self.price_history[-3]
            acceleration = change1 - change2
        else:
            acceleration = 0.0

        # V8.0评分
        score_ema = min(delta_ema_abs / 0.3, 1.0) * 0.5
        score_vol = min(abs(delta_vol) / 0.5, 1.0) * 0.3
        score_base = min(volume_ratio / 2.0, 1.0) * 0.2

        v8_score = score_ema + score_vol + score_base

        return TechnicalIndicators(
            ema=ema,
            delta_ema=delta_ema,
            delta_ema_abs=delta_ema_abs,
            tension=tension,
            acceleration=acceleration,
            volume_ratio=volume_ratio,
            delta_vol=delta_vol,
            v8_score=v8_score
        )


# ==================== Singularity分类器 ====================

class SingularityClassifier:
    """奇点信号分类器"""

    def classify(self, indicators: TechnicalIndicators) -> str:
        """分类Singularity信号"""
        tension = indicators.tension
        acceleration = indicators.acceleration
        volume_ratio = indicators.volume_ratio

        # BEARISH_SINGULARITY: 系统极度看空
        if tension < -0.7 and acceleration < -0.2:
            return 'BEARISH_SINGULARITY'

        # BULLISH_SINGULARITY: 系统极度看涨
        if tension > 0.7 and acceleration > 0.2:
            return 'BULLISH_SINGULARITY'

        # LOW_OSCILLATION: 低位震荡
        if abs(tension) < 0.3 and volume_ratio < 0.5:
            return 'LOW_OSCILLATION'

        # OSCILLATION: 震荡市
        return 'OSCILLATION'


# ==================== 微观结构分析器 ====================

class MicrostructureAnalyzer:
    """简化微观结构分析器"""

    def __init__(self):
        self.price_history: List[float] = []
        self.volume_history: List[float] = []

    def analyze(self, v8_engine: V80Engine, indicators: TechnicalIndicators) -> MicrostructureMetrics:
        """分析微观结构"""
        # 使用V8引擎的历史数据
        prices = v8_engine.price_history
        volumes = v8_engine.volume_history

        if len(prices) < 3:
            return MicrostructureMetrics(
                volatility=0,
                liquidity_score=0.5,
                is_crash_risk=False,
                crash_confidence=0,
                is_squeeze=False,
                squeeze_confidence=0,
                recommendation="数据不足"
            )

        # 计算波动率 (Proxy for IV)
        returns = pd.Series(prices).pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0

        # 价格加速度 (Proxy for GEX)
        acceleration = indicators.acceleration

        # 成交量激增 (Proxy for CVD)
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            volume_surge = volumes[-1] > avg_volume * 1.5
        else:
            volume_surge = False
            avg_volume = volumes[-1] if volumes else 1

        # 流动性评分 (Proxy for Order Book Depth)
        volume_score = min(volumes[-1] / 1000000, 1.0)
        volatility_score = max(1 - abs(acceleration) / 0.05, 0)
        liquidity = (volume_score + volatility_score) / 2

        # 检测负Gamma陷阱
        crash_conf = 0
        if volatility > 0.8:
            crash_conf += 0.4
        if acceleration < -0.2:
            crash_conf += 0.4
        if liquidity < 0.3:
            crash_conf += 0.2

        is_crash_risk = crash_conf >= 0.5

        # 检测Vanna挤压
        squeeze_conf = 0
        if volatility < 0.3:
            squeeze_conf += 0.3
        if volume_surge and abs(acceleration) < 0.1:
            squeeze_conf += 0.4
        if liquidity > 0.7:
            squeeze_conf += 0.3

        is_squeeze = squeeze_conf >= 0.6

        # 建议
        if is_crash_risk:
            recommendation = "【警告】高闪崩风险 - 减仓或观望"
        elif is_squeeze:
            recommendation = "【机会】低波动吸筹 - 准备突破"
        elif liquidity < 0.4:
            recommendation = "【注意】流动性不足 - 谨慎交易"
        else:
            recommendation = "【正常】市场平稳 - 按策略交易"

        return MicrostructureMetrics(
            volatility=volatility,
            liquidity_score=liquidity,
            is_crash_risk=is_crash_risk,
            crash_confidence=crash_conf,
            is_squeeze=is_squeeze,
            squeeze_confidence=squeeze_conf,
            recommendation=recommendation
        )


# ==================== 交易决策器 ====================

class TradingDecisionEngine:
    """交易决策引擎 - 整合所有逻辑"""

    def make_decision(self,
                     indicators: TechnicalIndicators,
                     signal_type: str,
                     micro: MicrostructureMetrics) -> TradingSignal:

        """做出交易决策"""

        # ===== 第1优先级: 负Gamma陷阱检查 =====
        if micro.is_crash_risk:
            return TradingSignal(
                timestamp=datetime.now(),
                action='SKIP',
                position_size=0,
                reason=f"负Gamma陷阱 - {micro.recommendation}",
                v8_score=indicators.v8_score,
                signal_type=signal_type,
                microstructure={
                    'volatility': micro.volatility,
                    'liquidity': micro.liquidity_score,
                    'crash_conf': micro.crash_confidence
                }
            )

        # ===== 第2优先级: Vanna挤压检查 =====
        if micro.is_squeeze:
            return TradingSignal(
                timestamp=datetime.now(),
                action='LONG',
                position_size=1.5,  # 150%仓位
                reason=f"Vanna挤压 - {micro.recommendation}",
                v8_score=indicators.v8_score,
                signal_type=signal_type,
                microstructure={
                    'volatility': micro.volatility,
                    'liquidity': micro.liquidity_score,
                    'squeeze_conf': micro.squeeze_confidence
                }
            )

        # ===== 第3优先级: V8.0反向策略 =====
        base_size = 1.0
        action = 'WAIT'
        reason = ''

        if signal_type == 'BEARISH_SINGULARITY':
            action = 'LONG'
            base_size = 1.0
            reason = 'BEARISH_SINGULARITY → 反向做多 (抄底)'

        elif signal_type == 'BULLISH_SINGULARITY':
            action = 'SHORT'
            base_size = 1.0
            reason = 'BULLISH_SINGULARITY → 反向做空 (逃顶)'

        elif signal_type == 'LOW_OSCILLATION':
            action = 'LONG'
            base_size = 1.0
            reason = 'LOW_OSCILLATION → 低位做多'

        elif signal_type == 'OSCILLATION':
            action = 'WAIT'
            base_size = 0
            reason = 'OSCILLATION → 震荡市不交易'

        # 黄金信号增强
        if indicators.v8_score > 0.85 and action in ['LONG', 'SHORT']:
            base_size = 1.5
            reason += ' + 黄金信号增强'

        # 黑天鹅增强
        if indicators.v8_score > 0.95 and abs(indicators.tension) > 1.0:
            base_size = 2.5
            reason += ' + 黑天鹅增强'

        return TradingSignal(
            timestamp=datetime.now(),
            action=action,
            position_size=base_size,
            reason=reason,
            v8_score=indicators.v8_score,
            signal_type=signal_type,
            microstructure={
                'volatility': micro.volatility,
                'liquidity': micro.liquidity_score
            }
        )


# ==================== 实盘交易引擎 ====================

class LiveTradingEngine:
    """实盘交易引擎"""

    def __init__(self,
                 account_balance: float = 10000,
                 risk_per_trade: float = 0.02,
                 data_file: str = "live_trading_log.csv"):

        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.data_file = data_file

        # 初始化组件
        self.v8_engine = V80Engine()
        self.classifier = SingularityClassifier()
        self.micro_analyzer = MicrostructureAnalyzer()
        self.decision_engine = TradingDecisionEngine()

        # 持仓
        self.position: Optional[Position] = None

        # 日志
        self.trade_log: List[Dict] = []

        print(f"[初始化] 实盘交易引擎")
        print(f"  账户余额: ${account_balance:,.0f}")
        print(f"  单笔风险: {risk_per_trade:.1%}")

    def check_signals(self) -> Optional[TradingSignal]:
        """
        完整的信号检查流程

        流程:
        第0层: 数据获取 (Binance API)
        第1层: 指标计算
        第2层: V8.0评分
        第3层: Singularity分类
        第4层: 微观结构过滤
        第5层: 交易决策
        """
        # 第0层: 获取市场数据
        market_data = self.v8_engine.fetch_market_data()
        if not market_data:
            print("[第0层] 数据获取失败")
            return None

        print(f"\n[第0层 数据获取] 价格: ${market_data.price:,.0f} ({market_data.price_change_percent:+.2f}%)")
        print(f"               成交量: {market_data.volume:,.0f} BTC")

        # 第1层: 计算技术指标
        indicators = self.v8_engine.calculate_indicators(market_data)

        print(f"[第1层 指标计算] V8_Score: {indicators.v8_score:.3f}")
        print(f"               张力: {indicators.tension:.3f}")
        print(f"               加速度: {indicators.acceleration:.3f}")
        print(f"               量能比率: {indicators.volume_ratio:.2f}")

        # 第2层: V8.0触发检查
        if indicators.v8_score < 0.7:
            print(f"[第2层 V8.0检测] 未触发 (阈值0.7)")
            return None

        print(f"[第2层 V8.0检测] 触发！V8_Score >= 0.7")

        # 第3层: Singularity分类
        signal_type = self.classifier.classify(indicators)
        print(f"[第3层 信号分类] {signal_type}")

        # 第4层: 微观结构分析
        micro = self.micro_analyzer.analyze(self.v8_engine, indicators)
        print(f"[第4层 微观结构] {micro.recommendation}")

        if micro.is_crash_risk:
            print(f"               ⚠️  负Gamma风险 (置信度: {micro.crash_confidence:.1%})")
        if micro.is_squeeze:
            print(f"               ✅ Vanna挤压 (置信度: {micro.squeeze_confidence:.1%})")

        # 第5层: 交易决策
        signal = self.decision_engine.make_decision(indicators, signal_type, micro)

        print(f"\n[第5层 交易决策] {signal.action} {signal.position_size*100:.0f}%")
        print(f"               理由: {signal.reason}")

        return signal

    def execute_trade(self, signal: TradingSignal):
        """执行交易"""
        if signal.action == 'SKIP' or signal.action == 'WAIT':
            print(f"  → 跳过交易")
            return

        # 如果有持仓，先平仓
        if self.position is not None:
            print(f"\n  ⚠️  已有持仓，先平仓")
            self.close_position(reason="信号冲突")

        # 计算仓位大小
        max_position_by_risk = self.account_balance * (self.risk_per_trade / 0.03)
        position_value = self.account_balance * signal.position_size
        actual_position = min(position_value, max_position_by_risk)

        # 第0层: 获取当前价格
        market_data = self.v8_engine.fetch_market_data()
        if not market_data:
            print(f"  [ERROR] 无法获取当前价格")
            return

        self.position = Position(
            entry_time=datetime.now(),
            entry_price=market_data.price,
            size=actual_position,
            side=signal.action,
            stop_loss=market_data.price * (0.97 if signal.action == 'LONG' else 1.03),
            take_profit=market_data.price * (1.10 if signal.action == 'LONG' else 0.90)
        )

        print(f"\n  ✅ 开仓执行:")
        print(f"     方向: {self.position.side}")
        print(f"     价格: ${self.position.entry_price:,.0f}")
        print(f"     仓位: ${self.position.size:,.0f} ({self.position.size/self.account_balance:.1%} of account)")
        print(f"     止损: ${self.position.stop_loss:,.0f} (-3%)")
        print(f"     止盈: ${self.position.take_profit:,.0f} (+10%)")

        # 记录日志
        self.log_trade(signal, 'OPEN')

    def check_position(self):
        """检查持仓状态"""
        if self.position is None:
            return

        # 第0层: 获取最新数据
        market_data = self.v8_engine.fetch_market_data()
        if not market_data:
            return

        current_price = market_data.price

        if self.position.side == 'LONG':
            pnl_ratio = (current_price - self.position.entry_price) / self.position.entry_price
        else:  # SHORT
            pnl_ratio = (self.position.entry_price - current_price) / self.position.entry_price

        # 计算微观结构
        indicators = self.v8_engine.calculate_indicators(market_data)
        micro = self.micro_analyzer.analyze(self.v8_engine, indicators)

        # 检查平仓条件
        should_close = False
        close_reason = ""

        # 1. 负Gamma陷阱 - 强制平仓
        if micro.is_crash_risk:
            should_close = True
            close_reason = "负Gamma陷阱触发"

        # 2. 止损
        elif pnl_ratio < -0.03:
            should_close = True
            close_reason = f"止损 ({pnl_ratio:.2%})"

        # 3. 止盈 (部分)
        elif pnl_ratio > 0.10:
            # 这里可以部分平仓，简化起见全部平仓
            should_close = True
            close_reason = f"止盈 ({pnl_ratio:.2%})"

        # 4. 追踪止损
        elif pnl_ratio > 0.05:
            trailing_stop = self.position.entry_price * (1.025 if self.position.side == 'LONG' else 0.975)
            if self.position.side == 'LONG' and current_price < trailing_stop:
                should_close = True
                close_reason = f"追踪止损 ({pnl_ratio:.2%})"
            elif self.position.side == 'SHORT' and current_price > trailing_stop:
                should_close = True
                close_reason = f"追踪止损 ({pnl_ratio:.2%})"

        # 5. 信号消失
        elif indicators.v8_score < 0.35:
            should_close = True
            close_reason = f"V8.0信号消失 ({indicators.v8_score:.2f})"

        if should_close:
            self.close_position(reason=close_reason, current_price=current_price)
        else:
            print(f"  持仓中: PnL {pnl_ratio:+.2%}")

    def close_position(self, reason: str, current_price: Optional[float] = None):
        """平仓"""
        if self.position is None:
            return

        if current_price is None:
            # 第0层: 获取最新价格
            market_data = self.v8_engine.fetch_market_data()
            if not market_data:
                return
            current_price = market_data.price

        if self.position.side == 'LONG':
            pnl_ratio = (current_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_ratio = (self.position.entry_price - current_price) / self.position.entry_price

        pnl_amount = self.position.size * pnl_ratio
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
        """记录交易日志"""
        log_entry = {
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '动作': action,
            'V8_Score': signal.v8_score if signal else '',
            '信号类型': signal.signal_type if signal else '',
            '交易方向': signal.action if signal else '',
            '仓位': f"{signal.position_size*100:.0f}%" if signal else '',
            '理由': signal.reason if signal else reason,
            '盈亏': f"{pnl:.2%}",
            '余额': f"${self.account_balance:,.0f}"
        }

        self.trade_log.append(log_entry)

        # 保存到CSV
        df = pd.DataFrame(self.trade_log)
        df.to_csv(self.data_file, index=False, encoding='utf-8-sig')

    def run(self, interval_seconds: int = 300):
        """
        运行实盘交易循环

        完整的7层决策流程:
        第0层: 数据获取 (Binance API)
        第1层: 指标计算 (EMA, 量能, 张力, 加速度)
        第2层: V8.0评分 (触发阈值 0.7)
        第3层: Singularity分类
        第4层: 微观结构过滤 (负Gamma, Vanna挤压)
        第5层: 反向策略决策
        第6层: 仓位管理 (100%-250%)
        第7层: 风险管理 (止损-3%, 止盈+10%)
        """
        print(f"\n{'='*100}")
        print(f"V8.0 + 简化微观结构 实盘交易")
        print(f"{'='*100}")
        print(f"\n[完整交易逻辑 - 7层决策流程]")
        print(f"  第0层: 数据获取 (Binance API)")
        print(f"  第1层: 指标计算 (EMA, 量能, 张力, 加速度)")
        print(f"  第2层: V8.0评分 (触发阈值 0.7)")
        print(f"  第3层: Singularity分类")
        print(f"  第4层: 微观结构过滤 (负Gamma, Vanna挤压)")
        print(f"  第5层: 反向策略决策")
        print(f"  第6层: 仓位管理 (100%-250%)")
        print(f"  第7层: 风险管理 (止损-3%, 止盈+10%)")

        print(f"\n[配置]")
        print(f"  检查间隔: {interval_seconds}秒")
        print(f"  V8.0阈值: 0.7")
        print(f"  风险控制: {self.risk_per_trade:.1%} per trade")

        print(f"\n[启动]")
        print(f"  系统将每{interval_seconds//60}分钟检查一次信号")
        print(f"  按 Ctrl+C 停止\n")

        try:
            while True:
                # 检查新信号
                if self.position is None:
                    signal = self.check_signals()
                    if signal:
                        self.execute_trade(signal)
                else:
                    # 检查持仓
                    self.check_position()

                # 等待下一次检查
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n[停止] 交易引擎已停止")

            if self.position:
                print(f"\n[提示] 仍有持仓，建议手动处理")
                print(f"  方向: {self.position.side}")
                print(f"  入场价: ${self.position.entry_price:,.0f}")
                print(f"  仓位: ${self.position.size:,.0f}")

            print(f"\n[最终余额] ${self.account_balance:,.0f}")
            print(f"[交易日志] {self.data_file}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("="*100)
    print("V8.0 + 简化微观结构 实盘交易系统")
    print("="*100)

    print("\n[完整交易逻辑]")
    print("  第1层: V8.0突变检测 (V8_Score >= 0.7)")
    print("  第2层: Singularity分类")
    print("  第3层: 简化微观结构过滤")
    print("  第4层: 反向策略决策")
    print("  第5层: 仓位管理")
    print("  第6层: 风险管理")

    print("\n[决策优先级]")
    print("  1. 负Gamma陷阱 → SKIP")
    print("  2. Vanna挤压 → LONG 150%")
    print("  3. V8.0反向策略 → LONG 100% / SHORT 100%")

    # 创建引擎
    engine = LiveTradingEngine(
        account_balance=10000,
        risk_per_trade=0.02,
        data_file="live_trading_log.csv"
    )

    # 运行
    engine.run(interval_seconds=300)  # 每5分钟检查一次
