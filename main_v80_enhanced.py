# -*- coding: utf-8 -*-
"""
V8.0 + 订单墙期权增强版
整合Deribit API，获取Gamma、Vanna、订单墙等微观结构数据
"""

from main_v80 import V80TradingEngine, V80Config, TelegramNotifier, DataFetcher
from deribit_data_hub import DeribitDataHub
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class V80OptionsEnhanced(V80TradingEngine):
    """V8.0期权增强版"""

    def __init__(self):
        super().__init__()
        self.deribit_hub = DeribitDataHub()
        self.options_data = None
        self.last_options_update = None
        self.options_update_interval = 3600  # 每小时更新一次期权数据

    def fetch_options_data(self, force: bool = False) -> bool:
        """
        获取期权数据

        参数:
        - force: 是否强制更新

        返回:
        - success: 是否成功
        """
        now = datetime.now()

        # 检查是否需要更新
        if not force and self.last_options_update:
            time_since_update = (now - self.last_options_update).total_seconds()
            if time_since_update < self.options_update_interval:
                logger.info(f"期权数据刚更新过({time_since_update:.0f}秒前)，跳过")
                return True

        logger.info("开始获取期权数据...")

        try:
            # 获取期权摘要
            raw_data = self.deribit_hub.get_book_summary_by_currency("BTC")

            if not raw_data:
                logger.warning("获取期权数据失败")
                return False

            # 解析数据
            self.options_data = self.deribit_hub.parse_options_data(raw_data)

            if self.options_data.empty:
                logger.warning("解析期权数据失败")
                return False

            self.last_options_update = now
            logger.info(f"期权数据更新成功: {len(self.options_data)} 个合约")

            # 计算关键指标
            self._calculate_options_indicators()

            return True

        except Exception as e:
            logger.error(f"获取期权数据异常: {e}")
            return False

    def _calculate_options_indicators(self):
        """计算期权指标"""
        if self.options_data is None or self.options_data.empty:
            return

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

            # 记录日志
            logger.info("=" * 70)
            logger.info("期权微观结构指标:")
            logger.info(f"  最大痛点: ${self.max_pain:,.0f}")
            logger.info(f"  净Gamma暴露: {gamma_exp['net_gamma_exposure']:.0f}")
            logger.info(f"  订单墙数量: {len(walls)}")
            logger.info(f"  Vanna挤压: {'是' if squeeze['is_squeeze'] else '否'}")
            if squeeze['is_squeeze']:
                logger.warning(f"  ⚠️ 挤压置信度: {squeeze['confidence']:.1%}")
                logger.warning(f"  ⚠️ 原因: {squeeze['reason']}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"计算期权指标失败: {e}")

    def check_signals_enhanced(self):
        """增强版信号检查（整合期权数据到交易决策）"""
        try:
            logger.info("=" * 70)
            logger.info("开始增强版信号检查...")

            # 1. 获取期权数据
            options_success = self.fetch_options_data()

            # 2. 获取BTC和DXY数据，计算验证5指标
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

            # 3. 期权组合策略（增强+确认+否决）
            options_boost = 0.0  # 期权数据对置信度的提升
            options_warning = []  # 期权预警信息
            options_veto = False  # 期权否决标志
            nearest_call_wall = None  # 最近的CALL墙（用于调整止盈）
            nearest_put_wall = None  # 最近的PUT墙（用于调整止损）

            if options_success and self.options_data is not None:
                # A. Vanna挤压检测（优先级最高：风险保护）
                if hasattr(self, 'vanna_squeeze') and self.vanna_squeeze['is_squeeze']:
                    squeeze_confidence = self.vanna_squeeze['confidence']

                    if squeeze_confidence > 0.8:
                        logger.error(f"  ❌ Vanna挤压风险过高({squeeze_confidence:.1%})，期权否决交易")
                        options_warning.append(f"Vanna挤压({squeeze_confidence:.0%})")
                        options_veto = True
                    else:
                        logger.warning(f"  ⚠️ Vanna挤压风险({squeeze_confidence:.1%})，需要谨慎")
                        options_warning.append(f"Vanna挤压({squeeze_confidence:.0%})")

                # B. Gamma暴露调整（方案2：增强 + 方案1：确认）
                if hasattr(self, 'gamma_exposure') and self.gamma_exposure:
                    net_gamma = self.gamma_exposure.get('net_gamma_exposure', 0)

                    # 先确定信号方向（需要提前判断）
                    temp_signal_type, _, _ = classify_market_state(tension, acceleration, dxy_fuel)
                    temp_direction, _ = self.strategy_map.get(temp_signal_type, ('wait', ''))

                    if net_gamma > 0:
                        logger.info(f"  📐 净Gamma为正({net_gamma:,.0f})，市场多头友好")

                        # 方案2：增强机制 - Gamma与信号一致时提高置信度
                        if temp_direction == 'long':
                            options_boost += 0.10
                            logger.info(f"  ✅ Gamma支持做多，置信度+10%")
                        # 方案1：确认机制 - Gamma与信号相反时降低置信度
                        elif temp_direction == 'short':
                            options_boost -= 0.20
                            logger.warning(f"  ⚠️ Gamma反对做空，置信度-20%")

                    elif net_gamma < 0:
                        logger.info(f"  📐 净Gamma为负({net_gamma:,.0f})，市场空头友好")

                        # 方案2：增强机制 - Gamma与信号一致时提高置信度
                        if temp_direction == 'short':
                            options_boost += 0.10
                            logger.info(f"  ✅ Gamma支持做空，置信度+10%")
                        # 方案1：确认机制 - Gamma与信号相反时降低置信度
                        elif temp_direction == 'long':
                            options_boost -= 0.20
                            logger.warning(f"  ⚠️ Gamma反对做多，置信度-20%")

                # C. 最大痛点磁吸效应
                if hasattr(self, 'max_pain') and self.max_pain:
                    current_price = df_4h.iloc[-1]['close']
                    distance_to_max_pain = (self.max_pain - current_price) / current_price

                    if abs(distance_to_max_pain) < 0.02:  # 2%以内
                        logger.info(f"  🎯 价格接近最大痛点({distance_to_max_pain:.2%})，可能被吸引")
                        options_warning.append(f"接近最大痛点")

                # D. 订单墙阻挡/支撑（用于方案4：调整止盈止损）
                if hasattr(self, 'order_walls') and self.order_walls:
                    current_price = df_4h.iloc[-1]['close']

                    # 分别找最近的CALL墙和PUT墙
                    for wall in self.order_walls:
                        distance = abs(wall['strike'] - current_price) / current_price

                        if distance < 0.15:  # 15%以内的墙才考虑
                            if wall['is_resistance'] and wall['strike'] > current_price:
                                if nearest_call_wall is None or distance < abs(nearest_call_wall['strike'] - current_price) / current_price:
                                    nearest_call_wall = wall
                            elif wall['is_support'] and wall['strike'] < current_price:
                                if nearest_put_wall is None or distance < abs(nearest_put_wall['strike'] - current_price) / current_price:
                                    nearest_put_wall = wall

                            # 5%以内的墙添加到预警
                            if distance < 0.05:
                                if wall['is_resistance']:
                                    logger.warning(f"  🧱 接近阻力墙${wall['strike']:,.0f} ({distance:.2%})")
                                    options_warning.append(f"阻力墙${wall['strike']:,.0f}")
                                else:
                                    logger.info(f"  🧱 接近支撑墙${wall['strike']:,.0f} ({distance:.2%})")
                                    options_warning.append(f"支撑墙${wall['strike']:,.0f}")

            # E. 期权否决检查（方案3：保护机制）
            if options_veto:
                logger.error("❌ 期权数据强烈反对，取消交易")
                # 记录被否决的信号
                signal_record = {
                    'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': signal_type if 'signal_type' in locals() else 'UNKNOWN',
                    'confidence': 0,
                    'description': f"{description if 'description' in locals() else ''} | 期权否决",
                    'price': df_4h.iloc[-1]['close'] if 'df_4h' in locals() else 0,
                    'tension': tension if 'tension' in locals() else 0,
                    'acceleration': acceleration if 'acceleration' in locals() else 0,
                    'dxy_fuel': dxy_fuel if 'dxy_fuel' in locals() else 0,
                    'traded': False,
                    'filtered': True,
                    'filter_reason': '期权否决: Vanna挤压风险过高'
                }
                self.config.signal_history.append(signal_record)
                self.config.save_state()
                return  # 直接返回，不开仓

            # 4. 市场状态分类（基于验证5）
            signal_type, description, base_confidence = classify_market_state(
                tension, acceleration, dxy_fuel
            )

            # 5. 期权增强调整置信度
            final_confidence = base_confidence + options_boost

            current_price = df_4h.iloc[-1]['close']
            current_time = df_4h.index[-1]

            # 构建增强描述
            enhanced_description = description
            if options_warning:
                enhanced_description += f" | 期权: {', '.join(options_warning)}"

            logger.info(f"检测到信号: {signal_type} | 置信度: {final_confidence:.2f} (基础:{base_confidence:.2f} + 期权:{options_boost:.2f})")
            logger.info(f"价格: ${current_price:.2f} | 张力: {tension:.3f} | 加速度: {acceleration:.3f} | DXY燃料: {dxy_fuel:.3f}")

            # 6. 记录信号
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

            # 只保留最近20个信号
            if len(self.config.signal_history) > 20:
                self.config.signal_history = self.config.signal_history[-20:]

            # 7. 发送信号通知
            self.notifier.notify_signal(
                signal_type, final_confidence, enhanced_description,
                current_price, tension, acceleration, dxy_fuel
            )

            # 8. 置信度过滤
            if final_confidence < self.config.CONFIDENCE_THRESHOLD:
                logger.info(f"置信度不足 ({final_confidence:.2f} < {self.config.CONFIDENCE_THRESHOLD})，跳过")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = f'置信度不足: {final_confidence:.2f}'
                self.config.save_state()
                return

            # 9. 检查是否已有持仓
            if self.config.has_position:
                logger.info("已有持仓，忽略新信号")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = '已有持仓，忽略新信号'
                self.notifier.send_message(f"⏸️ 信号被忽略：已有持仓")
                self.config.save_state()
                return

            # 10. 确定入场方向（V8.0反向策略）
            direction, reason = self.strategy_map.get(signal_type, ('wait', '未知状态'))

            if direction == 'wait':
                logger.info(f"观望状态: {signal_type}")
                self.config.signal_history[-1]['filtered'] = True
                self.config.signal_history[-1]['filter_reason'] = f'观望状态: {signal_type}'
                self.config.save_state()
                return

            # 11. 检查期权阻挡（如果有强烈阻力墙，降低做空仓位）
            if options_success and hasattr(self, 'order_walls') and self.order_walls:
                if direction == 'short':
                    # 检查是否有强烈的CALL墙在上方
                    current_price = df_4h.iloc[-1]['close']
                    for wall in self.order_walls:
                        if wall['is_resistance'] and wall['strike'] > current_price:
                            distance = (wall['strike'] - current_price) / current_price
                            if distance < 0.03:  # 3%以内
                                logger.warning(f"  ⚠️ 上方有强力CALL墙${wall['strike']:,.0f}，做空风险增加")
                                # 可以考虑降低仓位或者跳过这个信号
                                # 这里我们选择继续但记录警告

            # 12. 计算止盈止损（方案4：根据订单墙调整）
            if direction == 'long':
                stop_loss = current_price * 0.97  # -3%
                take_profit = current_price * 1.10  # +10%

                # 根据订单墙调整止盈止损
                if nearest_call_wall:
                    # 如果上方有CALL墙，且在原止盈位置之前，则提前止盈
                    if nearest_call_wall['strike'] < take_profit:
                        old_tp = take_profit
                        take_profit = nearest_call_wall['strike'] * 0.99  # 阻力墙之前1%
                        logger.info(f"  📊 止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f} (阻力墙${nearest_call_wall['strike']:,.0f})")

                if nearest_put_wall:
                    # 如果下方有PUT墙，且在原止损位置之后，则延后止损（放宽保护）
                    if nearest_put_wall['strike'] > stop_loss:
                        old_sl = stop_loss
                        stop_loss = nearest_put_wall['strike'] * 0.99  # 支撑墙之下1%
                        logger.info(f"  📊 止损调整: ${old_sl:,.0f} → ${stop_loss:,.0f} (支撑墙${nearest_put_wall['strike']:,.0f})")

            else:  # short
                stop_loss = current_price * 1.03  # +3%
                take_profit = current_price * 0.90  # -10%

                # 根据订单墙调整止盈止损
                if nearest_put_wall:
                    # 如果下方有PUT墙，且在原止盈位置之前，则提前止盈
                    if nearest_put_wall['strike'] > take_profit:
                        old_tp = take_profit
                        take_profit = nearest_put_wall['strike'] * 1.01  # 支撑墙之上1%
                        logger.info(f"  📊 止盈调整: ${old_tp:,.0f} → ${take_profit:,.0f} (支撑墙${nearest_put_wall['strike']:,.0f})")

                if nearest_call_wall:
                    # 如果上方有CALL墙，且在原止损位置之后，则延后止损（放宽保护）
                    if nearest_call_wall['strike'] < stop_loss:
                        old_sl = stop_loss
                        stop_loss = nearest_call_wall['strike'] * 1.01  # 阻力墙之上1%
                        logger.info(f"  📊 止损调整: ${old_sl:,.0f} → ${stop_loss:,.0f} (阻力墙${nearest_call_wall['strike']:,.0f})")

            # 13. 开仓
            self.config.has_position = True
            self.config.position_type = direction
            self.config.entry_price = current_price
            self.config.stop_loss = stop_loss
            self.config.take_profit = take_profit
            self.config.entry_time = datetime.utcnow()
            self.config.entry_confidence = final_confidence
            self.config.entry_signal = signal_type

            # 记录交易
            trade_record = {
                'entry_time': self.config.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'type': direction,
                'entry_price': current_price,
                'signal': signal_type,
                'confidence': final_confidence
            }
            self.config.trade_history.append(trade_record)
            self.config.signal_history[-1]['traded'] = True

            # 保存状态
            self.config.save_state()

            # 发送开仓通知
            self.notifier.notify_open_position(
                direction, current_price, stop_loss, take_profit,
                signal_type, final_confidence
            )

            logger.info(f"✅ 开仓成功: {direction.upper()} @ ${current_price:.2f}")
            logger.info(f"   止损: ${stop_loss:.2f} | 止盈: ${take_profit:.2f}")

            # 14. 发送期权增强分析
            if options_success and self.options_data is not None:
                self._send_enhanced_analysis()

        except Exception as e:
            logger.error(f"增强版信号检查失败: {e}", exc_info=True)

    def _send_enhanced_analysis(self):
        """发送增强分析到Telegram"""
        try:
            message = "📊 期权微观结构分析:\n\n"

            # 最大痛点
            if hasattr(self, 'max_pain') and self.max_pain:
                message += f"🎯 最大痛点: ${self.max_pain:,.0f}\n"

            # Gamma暴露
            if hasattr(self, 'gamma_exposure') and self.gamma_exposure:
                net_gamma = self.gamma_exposure.get('net_gamma_exposure', 0)
                gamma_status = "🟢 做多友好" if net_gamma > 0 else "🔴 做空友好"
                message += f"📐 净Gamma: {net_gamma:,.0f} {gamma_status}\n"

            # 订单墙
            if hasattr(self, 'order_walls') and self.order_walls:
                message += f"\n🧱 订单墙 ({len(self.order_walls)}个):\n"
                for wall in self.order_walls[:3]:  # 只显示前3个
                    icon = "🔴" if wall['is_resistance'] else "🟢"
                    message += f"  {icon} ${wall['strike']:,.0f} - {wall['oi_btc']:.0f} BTC\n"

            # Vanna挤压
            if hasattr(self, 'vanna_squeeze'):
                squeeze = self.vanna_squeeze
                if squeeze['is_squeeze']:
                    message += f"\n⚠️ Vanna挤压风险 (置信度: {squeeze['confidence']:.1%})\n"
                    message += f"原因: {squeeze['reason']}\n"

            # 发送通知
            self.notifier.send_message(message)

        except Exception as e:
            logger.error(f"发送增强分析失败: {e}")

    def run_enhanced(self):
        """运行增强版主循环"""
        logger.info("启动V8.0期权增强版系统...")

        # 启动时更新一次期权数据
        self.fetch_options_data(force=True)

        # 发送启动通知
        self.notifier.notify_status()

        logger.info("进入主循环...")
        logger.info("=" * 70)

        last_signal_check_hour = None
        last_position_check_hour = None
        last_options_check_hour = None

        while True:
            try:
                # 获取当前北京时间
                now_beijing = datetime.utcnow() + timedelta(hours=8)
                current_hour = now_beijing.hour
                current_minute = now_beijing.minute

                # 信号检查：每4小时 (0:00, 4:00, 8:00, 12:00, 16:00, 20:00)
                if current_hour % 4 == 0 and current_minute < 5:
                    if last_signal_check_hour != current_hour:
                        logger.info(f"[定时] 触发信号检查（{now_beijing.strftime('%H:%M')}）")

                        # 使用增强版信号检查
                        self.check_signals_enhanced()

                        last_signal_check_hour = current_hour

                # 持仓检查：每1小时
                if current_minute < 1:
                    if last_position_check_hour != current_hour:
                        logger.info(f"[定时] 触发持仓检查（{now_beijing.strftime('%H:%M')}）")
                        self.check_position()
                        last_position_check_hour = current_hour

                # 期权数据更新：每1小时
                if current_minute < 1:
                    if last_options_check_hour != current_hour:
                        logger.info(f"[定时] 更新期权数据（{now_beijing.strftime('%H:%M')}）")
                        self.fetch_options_data(force=True)
                        last_options_check_hour = current_hour

                # 每秒检查一次
                import time
                time.sleep(1)

            except KeyboardInterrupt:
                logger.info("收到停止信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(60)


# ==================== 主入口 ====================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('v80_enhanced.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    system = V80OptionsEnhanced()

    try:
        system.run_enhanced()
    except KeyboardInterrupt:
        logger.info("程序已停止")
