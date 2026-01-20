# -*- coding: utf-8 -*-
"""
V7.0.8 升级模块 - 黄金策略识别器
基于6个月统计学分析的好机会识别系统

独立模块，可与V7.0.7系统集成使用
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class V708Config:
    """V7.0.8配置参数（基于378笔样本的统计学分析）"""

    def __init__(self):
        # ========== SHORT信号参数 ==========
        # 基础阈值
        self.SHORT_TENSION_MIN = 0.5

        # 直接开仓条件（65-70%胜率）
        self.SHORT_TENSION_DIRECT = 0.8  # 张力≥0.8
        self.SHORT_ENERGY_DIRECT_MIN = 0.5  # 量能0.5-1.0
        self.SHORT_ENERGY_DIRECT_MAX = 1.0
        self.SHORT_RATIO_DIRECT_MIN = 50  # 比例50-150
        self.SHORT_RATIO_DIRECT_MAX = 150

        # 等待确认条件（85-100%好机会率）
        self.SHORT_TENSION_WAIT_MIN = 0.5  # 张力0.5-0.7
        self.SHORT_TENSION_WAIT_MAX = 0.7
        self.SHORT_ENERGY_WAIT_MIN = 1.0  # 量能1.0-2.0
        self.SHORT_ENERGY_WAIT_MAX = 2.0

        # 确认后的黄金机会判别（Youden指数）
        self.SHORT_TENSION_CHANGE_GOLDEN = 5.31  # 张力变化>5.31%
        self.SHORT_PRICE_ADVANTAGE_GOLDEN = 0.51  # 价格优势>0.51%
        self.SHORT_RATIO_GOLDEN = 100  # 比例≥100（额外加分）

        # 等待周期
        self.SHORT_WAIT_MIN = 4
        self.SHORT_WAIT_MAX = 6

        # ========== LONG信号参数 ==========
        # 基础阈值
        self.LONG_TENSION_MAX = -0.5

        # 直接开仓和黄金开仓条件（100%好机会）
        self.LONG_TENSION_STRONG = -0.7  # 张力<-0.7
        self.LONG_RATIO_MIN = 100  # 比例≥100
        self.LONG_TENSION_CHANGE_GOLDEN = 4.77  # 张力变化>4.77%
        self.LONG_ENERGY_IDEAL_MIN = 1.0  # 能量≥1.0

        # 等待周期
        self.LONG_WAIT_MIN = 4
        self.LONG_WAIT_MAX = 6

        # ========== 平仓参数（基于最优平仓分析） ==========
        # SHORT平仓（最优第5-7周期，平均+1.20%）
        self.SHORT_EXIT_ENERGY_EXPAND = 1.0  # 52.7%触发率
        self.SHORT_EXIT_MIN_PERIOD = 5
        self.SHORT_EXIT_OPTIMAL_PERIOD = 7  # 最优第7周期
        self.SHORT_EXIT_MAX_PERIOD = 10
        self.SHORT_EXIT_TENSION_DROP = 0.14  # 平均下降14%
        self.SHORT_EXIT_PROFIT_TARGET = 0.02  # 2%

        # LONG平仓（最优第7-9周期，平均+1.35%）
        self.LONG_EXIT_ENERGY_EXPAND = 1.0  # 50.3%触发率
        self.LONG_EXIT_MIN_PERIOD = 7
        self.LONG_EXIT_OPTIMAL_PERIOD = 9  # 最优第9周期
        self.LONG_EXIT_MAX_PERIOD = 10
        self.LONG_EXIT_PROFIT_TARGET = 0.02  # 2%

        # 固定止盈止损（保留V7.0.7）
        self.FALLBACK_TP = 0.05  # +5%
        self.FALLBACK_SL = -0.025  # -2.5%


class V708GoldenDetector:
    """V7.0.8黄金机会识别器"""

    def __init__(self, config):
        self.config = config
        self.pending_signals = {}  # 待确认的信号
        self.waiting_periods = {}  # 等待周期计数

    def check_first_signal(self, tension, acceleration, volume_ratio, timestamp, price, signal_type):
        """
        检查是否为首次信号（基于统计学的直接开仓判断）

        返回: (is_signal, action, message)
        action: 'direct_enter' | 'wait_confirm' | 'ignore'
        """

        # 计算张力/加速度比
        ratio = abs(tension / acceleration) if acceleration != 0 else 0

        message_detail = f"T={tension:.4f}, a={acceleration:.6f}, E={volume_ratio:.2f}, 比例={ratio:.1f}"

        if signal_type in ['BEARISH_SINGULARITY', 'HIGH_OSCILLATION']:
            # SHORT信号判断
            if tension < self.config.SHORT_TENSION_MIN:
                return False, 'ignore', f"张力过低: {message_detail}"

            # 判断是否可以直接开仓（65-70%胜率）
            can_direct = (
                tension >= self.config.SHORT_TENSION_DIRECT and
                self.config.SHORT_ENERGY_DIRECT_MIN <= volume_ratio <= self.config.SHORT_ENERGY_DIRECT_MAX and
                self.config.SHORT_RATIO_DIRECT_MIN <= ratio <= self.config.SHORT_RATIO_DIRECT_MAX
            )

            if can_direct:
                return True, 'direct_enter', f"【直接开仓SHORT】张力≥0.8,量能{volume_ratio:.2f},比例{ratio:.1f}: {message_detail}"
            else:
                # 判断是否需要等待确认
                should_wait = (
                    (self.config.SHORT_TENSION_WAIT_MIN <= tension <= self.config.SHORT_TENSION_WAIT_MAX) or
                    (self.config.SHORT_ENERGY_WAIT_MIN <= volume_ratio <= self.config.SHORT_ENERGY_WAIT_MAX)
                )

                if should_wait:
                    # 记录为待确认信号
                    self.pending_signals[timestamp] = {
                        'direction': 'short',
                        'tension': tension,
                        'acceleration': acceleration,
                        'volume_ratio': volume_ratio,
                        'price': price,
                        'ratio': ratio,
                        'signal_type': signal_type
                    }
                    self.waiting_periods[timestamp] = 0
                    return True, 'wait_confirm', f"【等待确认SHORT】张力{tension:.2f}需确认: {message_detail}"
                else:
                    return False, 'ignore', f"SHORT信号不符合直接开仓或等待条件: {message_detail}"

        elif signal_type in ['BULLISH_SINGULARITY', 'LOW_OSCILLATION']:
            # LONG信号判断
            if tension > self.config.LONG_TENSION_MAX:
                return False, 'ignore', f"张力过高: {message_detail}"

            # 判断是否可以直接开仓（张力<-0.7, 比例≥100）
            can_direct = (
                tension <= self.config.LONG_TENSION_STRONG and
                ratio >= self.config.LONG_RATIO_MIN
            )

            if can_direct:
                return True, 'direct_enter', f"【直接开仓LONG】张力≤{self.config.LONG_TENSION_STRONG},比例≥{ratio:.1f}: {message_detail}"
            else:
                # 记录为待确认信号（等待4-6周期）
                self.pending_signals[timestamp] = {
                    'direction': 'long',
                    'tension': tension,
                    'acceleration': acceleration,
                    'volume_ratio': volume_ratio,
                    'price': price,
                    'ratio': ratio,
                    'signal_type': signal_type
                }
                self.waiting_periods[timestamp] = 0
                return True, 'wait_confirm', f"【等待确认LONG】等待4-6周期确认: {message_detail}"

        return False, 'ignore', f"非目标信号: {message_detail}"

    def check_golden_entry(self, current_tension, current_accel, current_volume,
                           current_price, current_time):
        """
        检查是否达到黄金开仓条件（基于统计学Youden指数和最优组合）

        返回: list of entry_info
        """
        confirmed_entries = []

        # 检查所有待确认信号
        for timestamp, signal in list(self.pending_signals.items()):
            self.waiting_periods[timestamp] += 1
            wait_period = self.waiting_periods[timestamp]

            direction = signal['direction']
            orig_tension = signal['tension']
            orig_price = signal['price']
            orig_ratio = signal['ratio']

            # 清理超过最大等待周期的信号
            if wait_period > 10:
                del self.pending_signals[timestamp]
                del self.waiting_periods[timestamp]
                logger.info(f"[V7.0.8] 信号超时移除: {timestamp}")
                continue

            # 检查是否在等待周期内（4-6周期）
            if not (self.config.SHORT_WAIT_MIN <= wait_period <= self.config.SHORT_WAIT_MAX or
                    self.config.LONG_WAIT_MIN <= wait_period <= self.config.LONG_WAIT_MAX):
                continue

            if direction == 'short':
                # SHORT黄金确认（基于统计学：张力变化>5.31% OR 价格优势>0.51%）
                tension_change = (current_tension - orig_tension) / orig_tension * 100
                price_advantage = (orig_price - current_price) / orig_price * 100

                # 统计学最优组合策略
                meets_tension_change = tension_change >= self.config.SHORT_TENSION_CHANGE_GOLDEN
                meets_price_advantage = price_advantage >= self.config.SHORT_PRICE_ADVANTAGE_GOLDEN
                meets_ratio = orig_ratio >= self.config.SHORT_RATIO_GOLDEN

                # 判别公式
                is_confirmed = meets_tension_change or meets_price_advantage
                is_golden = is_confirmed and (meets_ratio or (meets_tension_change and meets_price_advantage))

                if is_confirmed:
                    entry_info = {
                        'direction': 'short',
                        'entry_price': current_price,
                        'entry_tension': current_tension,
                        'entry_accel': current_accel,
                        'entry_volume': current_volume,
                        'wait_period': wait_period,
                        'tension_change': tension_change,
                        'price_advantage': price_advantage,
                        'is_golden': is_golden,
                        'original_time': timestamp,
                        'entry_time': current_time
                    }

                    confirmed_entries.append(entry_info)
                    logger.info(f"[V7.0.8] SHORT机会确认: T变化={tension_change:.2f}%, 价格优势={price_advantage:.2f}%, 黄金={is_golden}")

                    # 移除已确认的信号
                    del self.pending_signals[timestamp]
                    del self.waiting_periods[timestamp]

            elif direction == 'long':
                # LONG黄金确认（基于统计学：张力变化>4.77% OR 比例≥100）
                # LONG的张力是负数，计算变化
                tension_change = (current_tension - orig_tension) / abs(orig_tension) * 100
                price_advantage = (current_price - orig_price) / orig_price * 100

                # 统计学最优组合策略
                meets_strong_tension = orig_tension <= self.config.LONG_TENSION_STRONG
                meets_energy = current_volume >= self.config.LONG_ENERGY_IDEAL_MIN
                meets_ratio = orig_ratio >= self.config.LONG_RATIO_MIN
                meets_tension_change = abs(tension_change) >= self.config.LONG_TENSION_CHANGE_GOLDEN

                # 100%好机会的判别公式
                # 条件1+2+3 或 条件2 或 条件3
                is_confirmed = (
                    (meets_strong_tension and meets_energy and 4 <= wait_period <= 6) or
                    (meets_ratio and 4 <= wait_period <= 6) or
                    meets_tension_change
                )

                is_golden = is_confirmed  # LONG的确认条件本身就很高

                if is_confirmed:
                    entry_info = {
                        'direction': 'long',
                        'entry_price': current_price,
                        'entry_tension': current_tension,
                        'entry_accel': current_accel,
                        'entry_volume': current_volume,
                        'wait_period': wait_period,
                        'tension_change': tension_change,
                        'price_advantage': price_advantage,
                        'is_golden': is_golden,
                        'original_time': timestamp,
                        'entry_time': current_time
                    }

                    confirmed_entries.append(entry_info)
                    logger.info(f"[V7.0.8] LONG机会确认: T变化={tension_change:.2f}%, 价格优势={price_advantage:.2f}%, 黄金={is_golden}")

                    # 移除已确认的信号
                    del self.pending_signals[timestamp]
                    del self.waiting_periods[timestamp]

        return confirmed_entries

    def check_golden_exit(self, position, current_tension, current_accel,
                         current_volume, current_price, hold_periods):
        """
        检查是否达到黄金平仓条件

        返回: (should_exit, exit_reason, exit_type)
        exit_type: 'golden' | 'fallback'
        """
        direction = position['direction']
        entry_price = position['entry_price']
        entry_tension = position['entry_tension']

        # 计算当前盈亏
        if direction == 'short':
            pnl = (entry_price - current_price) / entry_price * 100
        else:
            pnl = (current_price - entry_price) / entry_price * 100

        # 先检查固定止损
        if pnl <= self.config.FALLBACK_SL * 100:
            return True, f"固定止损({pnl:.2f}%)", 'fallback'
        if pnl >= self.config.FALLBACK_TP * 100:
            return True, f"固定止盈({pnl:.2f}%)", 'fallback'

        # 检查黄金平仓条件（基于6个月统计学分析）
        if direction == 'short':
            # 张力下降比例
            tension_drop_ratio = (entry_tension - current_tension) / entry_tension

            # SHORT黄金平仓：两个条件组（AND关系）
            should_exit = (
                (current_volume > 1.0 or hold_periods >= 5)  # 条件A：量能放大 OR 时间足够
            ) and (
                tension_drop_ratio >= 0.14 or pnl >= 2  # 条件B：张力下降14% OR 盈利>2%
            )

            if should_exit:
                reasons = []
                if current_volume > 1.0:
                    reasons.append(f"量能放大({current_volume:.2f})")
                if hold_periods >= 5:
                    reasons.append(f"持仓{hold_periods}周期")
                if tension_drop_ratio >= 0.14:
                    reasons.append(f"张力下降{tension_drop_ratio*100:.1f}%")
                if pnl >= 2:
                    reasons.append(f"盈利{pnl:.2f}%")

                return True, f"黄金平仓: {', '.join(reasons)}", 'golden'

            # 强制平仓：持仓过长
            if hold_periods >= 10:  # 10个周期（40小时）
                return True, f"强制平仓: 持仓{hold_periods}周期", 'golden'

        else:  # long
            # LONG的张力是负数，使用绝对值计算变化率
            tension_change = (abs(current_tension) - abs(entry_tension)) / abs(entry_tension) * 100

            should_exit = (
                (current_volume > self.config.LONG_EXIT_ENERGY_EXPAND or
                 hold_periods >= self.config.LONG_EXIT_MIN_PERIOD)
            ) and (
                tension_change < 0 or  # 张力不再增加（绝对值开始减小）
                pnl >= self.config.LONG_EXIT_PROFIT_TARGET * 100
            )

            if should_exit:
                reasons = []
                if current_volume > self.config.LONG_EXIT_ENERGY_EXPAND:
                    reasons.append(f"量能放大({current_volume:.2f})")
                if hold_periods >= self.config.LONG_EXIT_MIN_PERIOD:
                    reasons.append(f"持仓{hold_periods}周期")
                if tension_change < 0:
                    reasons.append("张力不再增加")
                if pnl >= self.config.LONG_EXIT_PROFIT_TARGET * 100:
                    reasons.append(f"盈利{pnl:.2f}%")

                return True, f"黄金平仓: {', '.join(reasons)}", 'golden'

            # 强制平仓
            if hold_periods >= self.config.LONG_EXIT_MAX_PERIOD:
                return True, f"强制平仓: 持仓{hold_periods}周期", 'golden'

        return False, "持仓中", None


class V708TelegramNotifier:
    """V7.0.8三级通知系统"""

    def __init__(self, token, chat_id, enabled=True):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled

    def send(self, message, priority='normal'):
        """发送Telegram消息"""
        if not self.enabled:
            return

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }

            # 增加重试机制
            for attempt in range(3):
                try:
                    resp = requests.post(url, json=data, timeout=15)
                    if resp.status_code == 200:
                        logger.info(f"[Telegram] 发送成功")
                        return True
                    else:
                        logger.warning(f"[Telegram] 发送失败: {resp.status_code}, {resp.text}")
                except Exception as e:
                    logger.error(f"[Telegram] 发送异常(尝试{attempt+1}/3): {e}")
                    import time
                    time.sleep(2)

            return False

        except Exception as e:
            logger.error(f"[Telegram] 通知异常: {e}")
            return False

    def notify_first_signal(self, signal_type, tension, acceleration, volume_ratio,
                           price, timestamp, direction, ratio):
        """通知1: 原始信号"""
        emoji = "🔴" if direction == 'short' else "🟢"
        direction_cn = "做空SHORT" if direction == 'short' else "做多LONG"

        message = f"""
{emoji} 【原始信号】{direction_cn}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ 时间: {timestamp}
💰 价格: ${price:.2f}
📊 张力: {tension:.4f}
📈 加速度: {acceleration:.6f}
⚡ 量能: {volume_ratio:.2f}
📐 张力/加速度比: {ratio:.1f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏳ 等待确认中...
"""

        self.send(message, priority='normal')

    def notify_golden_entry(self, entry_info, fallback_tp, fallback_sl):
        """通知2: 黄金开仓"""
        direction = entry_info['direction']
        is_golden = entry_info['is_golden']

        if direction == 'short':
            emoji = "🔴" if is_golden else "⚪"
            direction_cn = "做空SHORT"
            emoji_level = "✨✨✨" if is_golden else "✨"
            tp_price = entry_info['entry_price'] * (1 - fallback_tp)
            sl_price = entry_info['entry_price'] * (1 - fallback_sl)
        else:
            emoji = "🟢" if is_golden else "⚪"
            direction_cn = "做多LONG"
            emoji_level = "✨✨✨" if is_golden else "✨"
            tp_price = entry_info['entry_price'] * (1 + fallback_tp)
            sl_price = entry_info['entry_price'] * (1 + fallback_sl)

        entry_price = entry_info['entry_price']
        entry_tension = entry_info['entry_tension']
        wait_period = entry_info['wait_period']
        tension_change = entry_info['tension_change']
        price_advantage = entry_info['price_advantage']

        message = f"""
{emoji_level} 【黄金开仓】{direction_cn}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ 入场时间: {entry_info['entry_time']}
💰 入场价格: ${entry_price:.2f}
📊 张力: {entry_tension:.4f}
⏳ 等待周期: {wait_period}
📈 张力变化: {tension_change:+.2f}%
💎 价格优势: {price_advantage:+.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【固定止盈止损】
🎯 止盈: ${tp_price:.2f} (+{fallback_tp*100:.1f}%)
🛡️ 止损: ${sl_price:.2f} ({fallback_sl*100:.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{emoji} {'黄金机会！' if is_golden else '普通机会'}
"""

        self.send(message, priority='high' if is_golden else 'normal')

    def notify_golden_exit(self, position, exit_reason, exit_price, pnl, exit_type):
        """通知3: 黄金平仓"""
        direction = position['direction']

        if direction == 'short':
            emoji = "🔴"
            direction_cn = "做空SHORT"
        else:
            emoji = "🟢"
            direction_cn = "做多LONG"

        exit_emoji = "✨" if exit_type == 'golden' else "⚠️"

        message = f"""
{exit_emoji} 【黄金平仓】{direction_cn}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ 入场时间: {position.get('entry_time', 'N/A')}
💰 入场价格: ${position['entry_price']:.2f}
⏰ 平仓时间: {position.get('exit_time', 'N/A')}
💰 平仓价格: ${exit_price:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 盈亏: {pnl:+.2f}%
📝 原因: {exit_reason}
🏷️ 类型: {'黄金平仓' if exit_type == 'golden' else '固定止损'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        self.send(message, priority='high' if exit_type == 'golden' else 'normal')
