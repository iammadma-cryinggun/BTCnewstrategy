# -*- coding: utf-8 -*-
"""
V8.1 + 订单流数据增强版
在V8.0基础上整合实时订单流数据，优化入场时机
"""

from main_v80 import V80TradingEngine, V80Config, TelegramNotifier, DataFetcher
from deribit_data_hub import DeribitDataHub
from order_flow_hub import OrderFlowHub
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class V81OrderFlowEnhanced(V80TradingEngine):
    """V8.1订单流增强版"""

    def __init__(self):
        super().__init__()

        # 期权数据模块
        self.deribit_hub = DeribitDataHub()
        self.options_data = None
        self.last_options_update = None
        self.options_update_interval = 3600  # 每小时更新一次期权数据

        # 订单流数据模块（新增）
        self.order_flow_hub = OrderFlowHub()
        self.order_flow_data = None
        self.last_order_flow_update = None

        logger.info("V8.1订单流增强版初始化完成")

    def fetch_options_data(self, force: bool = False) -> bool:
        """获取期权数据"""
        now = datetime.now()

        if not force and self.last_options_update:
            time_since_update = (now - self.last_options_update).total_seconds()
            if time_since_update < self.options_update_interval:
                logger.info(f"期权数据刚更新过({time_since_update:.0f}秒前)，跳过")
                return True

        logger.info("开始获取期权数据...")

        try:
            # 获取原始数据
            raw_data = self.deribit_hub.get_book_summary_by_currency("BTC")
            if not raw_data:
                logger.error("获取期权数据失败")
                return False

            # 解析数据
            self.options_data = self.deribit_hub.parse_options_data(raw_data)

            if self.options_data.empty:
                logger.warning("解析期权数据失败")
                return False

            self.last_options_update = now
            logger.info(f"期权数据更新成功: {len(self.options_data)} 个合约")

            # 计算期权指标
            self.calculate_options_indicators()

            return True

        except Exception as e:
            logger.error(f"获取期权数据失败: {e}")
            return False

    def calculate_options_indicators(self):
        """计算期权指标（同V8.0）"""
        try:
            # 1. Gamma暴露
            gamma_exp = self.deribit_hub.calculate_gamma_exposure(self.options_data)
            self.gamma_exposure = gamma_exp

            # 2. 最大痛点
            max_pain = self.deribit_hub.find_max_pain(self.options_data)
            self.max_pain = max_pain

            # 3. 订单墙（使用动态阈值）
            walls = self.deribit_hub.identify_order_walls(
                self.options_data,
                threshold_btc=None,  # 使用动态阈值
                top_n=10
            )
            self.order_walls = walls

            # 4. Vanna挤压检测
            squeeze = self.deribit_hub.detect_vanna_squeeze(self.options_data)
            self.vanna_squeeze = squeeze

            logger.info("期权微观结构指标:")
            logger.info(f"  最大痛点: ${self.max_pain:,.0f}")
            logger.info(f"  净Gamma暴露: {gamma_exp['net_gamma_exposure']:.0f}")
            logger.info(f"  订单墙数量: {len(walls)}")
            logger.info(f"  Vanna挤压: {'是' if squeeze['is_squeeze'] else '否'}")
            if squeeze['is_squeeze']:
                logger.warning(f"  ⚠️ 挤压置信度: {squeeze['confidence']:.1%}")
                logger.warning(f"  ⚠️ 原因: {squeeze['reason']}")

        except Exception as e:
            logger.error(f"计算期权指标失败: {e}")

    def fetch_order_flow_data(self) -> bool:
        """
        获取订单流数据（新增）

        Returns:
            success: 是否成功
        """
        logger.info("开始获取订单流数据...")

        try:
            # 获取综合订单流分析
            summary = self.order_flow_hub.get_order_flow_summary()

            if not summary:
                logger.error("获取订单流数据失败")
                return False

            self.order_flow_data = summary
            self.last_order_flow_update = datetime.now()

            # 记录关键信息
            if summary.get('cvd'):
                cvd = summary['cvd']
                logger.info(f"  CVD: {cvd['current_cvd']:,.0f} USD")
                logger.info(f"  买入占比: {cvd['buy_ratio']:.1%}")
                logger.info(f"  趋势: {cvd['trend']}")

            if summary.get('order_walls'):
                walls = summary['order_walls']
                logger.info(f"  订单流墙: 支撑{len(walls['support_walls'])}个, 阻力{len(walls['resistance_walls'])}个")

            if summary.get('whale_trades'):
                logger.info(f"  鲸鱼交易: {len(summary['whale_trades'])}笔")

            logger.info("订单流数据获取成功")
            return True

        except Exception as e:
            logger.error(f"获取订单流数据失败: {e}")
            return False

    def check_signals_enhanced(self):
        """增强版信号检查（整合期权数据 + 订单流数据）"""
        try:
            logger.info("=" * 70)
            logger.info("开始V8.1增强版信号检查...")

            # 1. 获取期权数据
            options_success = self.fetch_options_data(force=True)

            # 2. 获取订单流数据（新增）
            order_flow_success = self.fetch_order_flow_data()

            # 3. 获取BTC和DXY数据，计算验证5指标
            df_4h = self.fetcher.fetch_btc_data(interval='4h', limit=300)
            if df_4h is None:
                logger.error("获取4H数据失败")
                return

            logger.info(f"4H K线数据: {len(df_4h)}条")

            from main_v80 import calculate_tension_acceleration_verification5, classify_market_state

            prices = df_4h['close'].values
            tension, acceleration = calculate_tension_acceleration_verification5(prices)

            if tension is None:
                logger.error("验证5指标计算失败")
                return

            # 获取DXY数据
            dxy_df = self.fetcher.fetch_dxy_data(days_back=30)
            dxy_fuel = 0.0
            if dxy_df is not None and len(dxy_df) >= 3:
                dxy_history = dxy_df['Close'].tolist()
                from main_v80 import calculate_dxy_fuel
                dxy_fuel = calculate_dxy_fuel(dxy_history)

            # 4. 期权组合策略（增强+确认+否决）
            options_boost = 0.0
            options_warning = []
            options_veto = False
            nearest_call_wall = None
            nearest_put_wall = None

            # A. Vanna挤压检测（优先级最高）
            if options_success and self.options_data is not None:
                if hasattr(self, 'vanna_squeeze') and self.vanna_squeeze['is_squeeze']:
                    squeeze_confidence = self.vanna_squeeze['confidence']

                    if squeeze_confidence > 0.8:
                        logger.error(f"  ❌ Vanna挤压风险过高({squeeze_confidence:.1%})，期权否决交易")
                        options_veto = True

            # B. Gamma暴露调整
            if options_success and hasattr(self, 'gamma_exposure') and self.gamma_exposure:
                net_gamma = self.gamma_exposure.get('net_gamma_exposure', 0)

                # 找到最近的订单墙
                if hasattr(self, 'order_walls') and self.order_walls:
                    current_price = df_4h.iloc[-1]['close']

                    call_walls = [w for w in self.order_walls if w['is_resistance']]
                    put_walls = [w for w in self.order_walls if not w['is_resistance']]

                    if call_walls:
                        nearest_call_wall = min(call_walls,
                                              key=lambda w: abs(w['strike'] - current_price))
                    if put_walls:
                        nearest_put_wall = min(put_walls,
                                             key=lambda w: abs(w['strike'] - current_price))

                # 根据信号方向调整
                temp_signal_type, _, _ = classify_market_state(tension, acceleration, dxy_fuel)
                temp_direction, _ = self.strategy_map.get(temp_signal_type, ('wait', ''))

                if net_gamma > 0:
                    if temp_direction == 'long':
                        options_boost += 0.10
                        logger.info(f"  ✅ Gamma支持做多，置信度+10%")
                    elif temp_direction == 'short':
                        options_boost -= 0.20
                        logger.warning(f"  ⚠️ Gamma反对做空，置信度-20%")
                        options_warning.append("空头Gamma反对做空")
                elif net_gamma < 0:
                    if temp_direction == 'short':
                        options_boost += 0.10
                        logger.info(f"  ✅ Gamma支持做空，置信度+10%")
                    elif temp_direction == 'long':
                        options_boost -= 0.20
                        logger.warning(f"  ⚠️ Gamma反对做多，置信度-20%")
                        options_warning.append("多头Gamma反对做多")

            # C. 期权否决检查
            if options_veto:
                logger.error("❌ 期权数据强烈反对，取消交易")
                signal_record = {
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'VETO',
                    'confidence': 0.0,
                    'description': '期权数据否决（Vanna挤压风险过高）',
                    'price': df_4h.iloc[-1]['close'],
                    'tension': tension,
                    'acceleration': acceleration,
                    'dxy_fuel': dxy_fuel,
                    'traded': False,
                    'filtered': True,
                    'filter_reason': 'Vanna挤压风险过高'
                }
                self.config.signal_history.append(signal_record)
                self.config.save_state()
                return

            # 5. 订单流确认机制（新增）
            order_flow_boost = 0.0
            order_flow_warning = []

            if order_flow_success and self.order_flow_data:
                # A. CVD趋势确认
                cvd = self.order_flow_data.get('cvd')
                if cvd:
                    cvd_trend = cvd.get('trend', 'neutral')
                    buy_ratio = cvd.get('buy_ratio', 0.5)

                    # 找到信号方向
                    signal_type, _, _ = classify_market_state(tension, acceleration, dxy_fuel)
                    direction, _ = self.strategy_map.get(signal_type, ('wait', ''))

                    if direction == 'long':
                        if cvd_trend == 'bullish' and buy_ratio > 0.6:
                            order_flow_boost += 0.05
                            logger.info(f"  ✅ CVD看涨({buy_ratio:.1%})，置信度+5%")
                        elif cvd_trend == 'bearish' and buy_ratio < 0.4:
                            order_flow_boost -= 0.10
                            logger.warning(f"  ⚠️ CVD看跌({buy_ratio:.1%})，置信度-10%")
                            order_flow_warning.append(f"CVD看跌({buy_ratio:.1%})")

                    elif direction == 'short':
                        if cvd_trend == 'bearish' and buy_ratio < 0.4:
                            order_flow_boost += 0.05
                            logger.info(f"  ✅ CVD看跌({buy_ratio:.1%})，置信度+5%")
                        elif cvd_trend == 'bullish' and buy_ratio > 0.6:
                            order_flow_boost -= 0.10
                            logger.warning(f"  ⚠️ CVD看涨({buy_ratio:.1%})，置信度-10%")
                            order_flow_warning.append(f"CVD看涨({buy_ratio:.1%})")

                # B. 鲸鱼交易确认
                whale_trades = self.order_flow_data.get('whale_trades', [])
                if whale_trades:
                    # 计算鲸鱼买卖比例
                    whale_buy_volume = sum(t['value'] for t in whale_trades if t['side'] == 'BUY')
                    whale_sell_volume = sum(t['value'] for t in whale_trades if t['side'] == 'SELL')

                    signal_type, _, _ = classify_market_state(tension, acceleration, dxy_fuel)
                    direction, _ = self.strategy_map.get(signal_type, ('wait', ''))

                    if direction == 'long' and whale_buy_volume > whale_sell_volume * 2:
                        order_flow_boost += 0.05
                        logger.info(f"  ✅ 鲸鱼大量买入，置信度+5%")
                    elif direction == 'short' and whale_sell_volume > whale_buy_volume * 2:
                        order_flow_boost += 0.05
                        logger.info(f"  ✅ 鲸鱼大量卖出，置信度+5%")

            # 6. 市场状态分类
            signal_type, description, base_confidence = classify_market_state(
                tension, acceleration, dxy_fuel
            )

            # 7. 综合调整置信度
            final_confidence = base_confidence + options_boost + order_flow_boost
            final_confidence = max(0, min(final_confidence, 1.0))  # 限制在0-1之间

            current_price = df_4h.iloc[-1]['close']
            current_time = df_4h.index[-1]

            # 构建增强描述
            enhanced_description = description
            if options_warning or order_flow_warning:
                all_warnings = options_warning + order_flow_warning
                enhanced_description += f" | 警告: {', '.join(all_warnings)}"

            logger.info(f"检测到信号: {signal_type}")
            logger.info(f"  基础置信度: {base_confidence:.2f}")
            logger.info(f"  期权调整: {options_boost:+.2f}")
            logger.info(f"  订单流调整: {order_flow_boost:+.2f}")
            logger.info(f"  最终置信度: {final_confidence:.2f}")
            logger.info(f"价格: ${current_price:.2f} | 张力: {tension:.3f} | 加速度: {acceleration:.3f}")

            # 8. 记录信号
            signal_record = {
                'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'type': signal_type,
                'confidence': final_confidence,
                'description': enhanced_description,
                'price': current_price,
                'tension': tension,
                'acceleration': acceleration,
                'dxy_fuel': dxy_fuel,
                'traded': False,
                'filtered': False
            }
            self.config.signal_history.append(signal_record)

            if len(self.config.signal_history) > 20:
                self.config.signal_history = self.config.signal_history[-20:]

            # 9. 置信度过滤（先过滤，避免不必要的通知）
            if final_confidence < self.config.CONFIDENCE_THRESHOLD:
                logger.info(f"置信度不足 ({final_confidence:.2f} < {self.config.CONFIDENCE_THRESHOLD})，跳过")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = f'置信度不足: {final_confidence:.2f}'
                self.config.save_state()
                logger.info("置信度不足，不发送Telegram通知")
                return

            # 10. 发送信号通知（只在置信度足够时发送）
            self.notifier.notify_signal(
                signal_type, final_confidence, enhanced_description,
                current_price, tension, acceleration, dxy_fuel
            )

            # 11. 检查是否已有持仓
            if self.config.has_position:
                logger.info("已有持仓，忽略新信号")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = '已有持仓，忽略新信号'
                self.notifier.send_message(f"⏸️ 信号被忽略：已有持仓")
                self.config.save_state()
                return

            # 12. 确定入场方向
            direction, reason = self.strategy_map.get(signal_type, ('wait', '未知状态'))

            if direction == 'wait':
                logger.info(f"观望状态: {signal_type}")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = f'观望状态: {signal_type}'
                self.config.save_state()
                return

            # 13. 计算止盈止损（使用订单流订单墙优化）
            if direction == 'long':
                stop_loss = current_price * 0.97  # -3%
                take_profit = current_price * 1.10  # +10%

                # A. 使用期权订单墙调整（同V8.0）
                if nearest_call_wall:
                    if nearest_call_wall['strike'] < take_profit:
                        old_tp = take_profit
                        take_profit = nearest_call_wall['strike'] * 0.99
                        logger.info(f"  📊 期权墙止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f}")

                if nearest_put_wall:
                    if nearest_put_wall['strike'] > stop_loss:
                        old_sl = stop_loss
                        stop_loss = nearest_put_wall['strike'] * 0.99
                        logger.info(f"  📊 期权墙止损调整: ${old_sl:,.0f} → ${stop_loss:,.0f}")

                # B. 使用订单流订单墙优化（新增）
                if order_flow_success and self.order_flow_data.get('order_walls'):
                    of_walls = self.order_flow_data['order_walls']

                    # 阻力墙调整止盈
                    if of_walls['resistance_walls']:
                        nearest_resistance = of_walls['resistance_walls'][0]
                        if nearest_resistance['price'] < take_profit:
                            old_tp = take_profit
                            take_profit = nearest_resistance['price'] * 0.995  # 阻力墙之前0.5%
                            logger.info(f"  📊 订单流墙止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f}")

                    # 支撑墙调整止损
                    if of_walls['support_walls']:
                        nearest_support = of_walls['support_walls'][0]
                        if nearest_support['price'] > stop_loss:
                            old_sl = stop_loss
                            stop_loss = nearest_support['price'] * 0.995  # 支撑墙之下0.5%
                            logger.info(f"  📊 订单流墙止损调整: ${old_sl:,.0f} → ${stop_sl:,.0f}")

            else:  # short
                stop_loss = current_price * 1.03  # +3%
                take_profit = current_price * 0.90  # -10%

                # A. 使用期权订单墙调整
                if nearest_put_wall:
                    if nearest_put_wall['strike'] > take_profit:
                        old_tp = take_profit
                        take_profit = nearest_put_wall['strike'] * 1.01
                        logger.info(f"  📊 期权墙止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f}")

                if nearest_call_wall:
                    if nearest_call_wall['strike'] < stop_loss:
                        old_sl = stop_loss
                        stop_loss = nearest_call_wall['strike'] * 1.01
                        logger.info(f"  📊 期权墙止损调整: ${old_sl:,.0f} → ${stop_sl:,.0f}")

                # B. 使用订单流订单墙优化（新增）
                if order_flow_success and self.order_flow_data.get('order_walls'):
                    of_walls = self.order_flow_data['order_walls']

                    # 支撑墙调整止盈
                    if of_walls['support_walls']:
                        nearest_support = of_walls['support_walls'][0]
                        if nearest_support['price'] > take_profit:
                            old_tp = take_profit
                            take_profit = nearest_support['price'] * 1.005  # 支撑墙之上0.5%
                            logger.info(f"  📊 订单流墙止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f}")

                    # 阻力墙调整止损
                    if of_walls['resistance_walls']:
                        nearest_resistance = of_walls['resistance_walls'][0]
                        if nearest_resistance['price'] < stop_loss:
                            old_sl = stop_loss
                            stop_loss = nearest_resistance['price'] * 1.005  # 阻力墙之上0.5%
                            logger.info(f"  📊 订单流墙止损调整: ${old_sl:,.0f} → ${stop_sl:,.0f}")

            # 14. 开仓
            logger.info("=" * 70)
            logger.info("开仓决策:")
            logger.info(f"  方向: {direction.upper()}")
            logger.info(f"  入场价: ${current_price:,.2f}")
            logger.info(f"  止损: ${stop_loss:,.2f} ({(stop_loss/current_price - 1)*100:+.2f}%)")
            logger.info(f"  止盈: ${take_profit:,.2f} ({(take_profit/current_price - 1)*100:+.2f}%)")
            logger.info(f"  盈亏比: {(abs(take_profit - current_price) / abs(stop_loss - current_price)):.2f}")

            # 记录开仓
            self.config.open_position(
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            # 更新信号记录
            self.config.signal_history[-1]['traded'] = True
            self.config.signal_history[-1]['direction'] = direction
            self.config.signal_history[-1]['entry_price'] = current_price
            self.config.signal_history[-1]['stop_loss'] = stop_loss
            self.config.signal_history[-1]['take_profit'] = take_profit
            self.config.save_state()

            # 发送开仓通知
            # 构建消息（避免复杂嵌套）
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
            if options_success and hasattr(self, 'gamma_exposure'):
                lines.append("")
                lines.append("📐 期权数据:")
                lines.append(f"  净Gamma: {self.gamma_exposure['net_gamma_exposure']:.0f}")
                if hasattr(self, 'max_pain'):
                    lines.append(f"  最大痛点: ${self.max_pain:,.0f}")

            # 订单流数据
            if order_flow_success and self.order_flow_data.get('cvd'):
                lines.append("")
                lines.append("📊 订单流数据:")
                cvd = self.order_flow_data['cvd']
                lines.append(f"  CVD趋势: {cvd['trend']}")
                lines.append(f"  买入占比: {cvd['buy_ratio']:.1%}")

            lines.append("")
            lines.append(f"🚀 方向: {direction.upper()}")
            lines.append(f"💵 入场: ${current_price:,.2f}")
            lines.append(f"🛑 止损: ${stop_loss:,.2f} ({(stop_loss/current_price - 1)*100:+.2f}%)")
            lines.append(f"🎯 止盈: ${take_profit:,.2f} ({(take_profit/current_price - 1)*100:+.2f}%)")
            lines.append(f"📈 盈亏比: {(abs(take_profit - current_price) / abs(stop_loss - current_price)):.2f}")

            message = "\n".join(lines)
            self.notifier.send_message(message)
            logger.info("信号通知已发送")

        except Exception as e:
            logger.error(f"信号检查失败: {e}")
            import traceback
            traceback.print_exc()


# 主函数
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    engine = V81OrderFlowEnhanced()

    logger.info("=" * 70)
    logger.info("V8.1 订单流增强版")
    logger.info("=" * 70)

    # 运行一次信号检查
    engine.check_signals_enhanced()

    logger.info("=" * 70)
    logger.info("V8.1信号检查完成")
    logger.info("=" * 70)
