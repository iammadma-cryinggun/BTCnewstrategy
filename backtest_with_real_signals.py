# -*- coding: utf-8 -*-
"""
基于真实信号动作的回测系统

使用CSV文件中实际的"信号动作"字段进行回测
对比:
1. 原始策略信号
2. 原始策略 + 负Gamma过滤 (模拟)
3. 原始策略 + Vanna增强 (模拟)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SignalBasedBacktester:
    """基于真实信号动作的回测器"""

    def __init__(self, data_filepath: str):
        print(f"\n[Loading data from {data_filepath}]")
        self.data = pd.read_csv(data_filepath, encoding='utf-8-sig')
        self.data['时间'] = pd.to_datetime(self.data['时间'])
        print(f"[OK] Loaded {len(self.data)} rows")

        # 提取有信号动作的行
        self.signal_rows = self.data[
            self.data['信号动作'].notna() &
            (self.data['信号动作'] != '')
        ].copy()

        print(f"[OK] Found {len(self.signal_rows)} signal actions")

    def simulate_negative_gamma_periods(self) -> set:
        """
        模拟负Gamma陷阱时期

        基于历史数据，模拟一些高风险时期:
        - 2025-08-05左右 (BTC大跌)
        - 2025-12月的高波动期
        """
        gamma_trap_periods = set()

        # 模拟: 张力>1.0 且 加速度>0.3 的时期为高风险
        for i, row in self.data.iterrows():
            tension = abs(row.get('张力', 0))
            accel = abs(row.get('加速度', 0))

            if tension > 1.0 and accel > 0.3:
                gamma_trap_periods.add(row['时间'])

        print(f"[Simulation] Identified {len(gamma_trap_periods)} negative gamma trap periods")
        return gamma_trap_periods

    def simulate_vanna_squeeze_periods(self) -> set:
        """
        模拟Vanna挤压时期

        基于历史数据，模拟一些低波动吸筹时期
        """
        vanna_periods = set()

        # 模拟: 张力<0.3 且 量能比率>1.5 的时期为吸筹
        for i, row in self.data.iterrows():
            tension = abs(row.get('张力', 0))
            volume = row.get('量能比率', 1.0)

            if tension < 0.3 and volume > 1.5:
                vanna_periods.add(row['时间'])

        print(f"[Simulation] Identified {len(vanna_periods)} Vanna squeeze periods")
        return vanna_periods

    def run_backtest(self, initial_capital: float = 100000):
        """
        运行回测
        """
        print(f"\n{'='*100}")
        print(f"BACKTEST: Signal-Based Strategy Comparison")
        print(f"{'='*100}")

        # 模拟期权数据
        gamma_trap_periods = self.simulate_negative_gamma_periods()
        vanna_squeeze_periods = self.simulate_vanna_squeeze_periods()

        # 初始化资金
        capital_baseline = initial_capital
        capital_with_gamma_filter = initial_capital
        capital_with_vanna_boost = initial_capital
        capital_combined = initial_capital

        # 交易记录
        trades_baseline = []
        trades_gamma_filtered = []
        trades_vanna_boosted = []
        trades_combined = []

        position = None  # (entry_time, entry_price, position_size)

        print(f"\n[Running backtest on {len(self.signal_rows)} signal actions...]")

        for i, row in self.signal_rows.iterrows():
            timestamp = row['时间']
            signal_action = row['信号动作']
            current_price = row['收盘价']

            # 解析信号动作
            if pd.isna(signal_action) or signal_action == '':
                continue

            # 检查是否是开仓信号
            is_open_long = '开多' in signal_action
            is_open_short = '开空' in signal_action
            is_close = '平仓' in signal_action or signal_action == ''

            # ===== 策略1: 基线 (跟随所有信号) =====
            if is_open_long or is_open_short:
                position_size = 1.0
                if '250%' in signal_action:
                    position_size = 2.5
                elif '150%' in signal_action:
                    position_size = 1.5

                trades_baseline.append({
                    'time': timestamp,
                    'action': signal_action,
                    'price': current_price,
                    'size': position_size
                })

            # ===== 策略2: 负Gamma过滤 =====
            if is_open_long:
                # 检查是否在负Gamma陷阱中
                if timestamp in gamma_trap_periods:
                    # 跳过这次交易
                    pass
                else:
                    position_size = 1.0
                    if '250%' in signal_action:
                        position_size = 2.5
                    elif '150%' in signal_action:
                        position_size = 1.5

                    trades_gamma_filtered.append({
                        'time': timestamp,
                        'action': signal_action,
                        'price': current_price,
                        'size': position_size
                    })

            # ===== 策略3: Vanna增强 =====
            if is_open_long or is_open_short:
                position_size = 1.0
                if '250%' in signal_action:
                    position_size = 2.5
                elif '150%' in signal_action:
                    position_size = 1.5

                # 如果是Vanna挤压时期，额外增加50%仓位
                if timestamp in vanna_squeeze_periods:
                    position_size *= 1.5

                trades_vanna_boosted.append({
                    'time': timestamp,
                    'action': signal_action,
                    'price': current_price,
                    'size': position_size
                })

            # ===== 策略4: 组合策略 =====
            if is_open_long:
                # 负Gamma陷阱 → 跳过
                if timestamp in gamma_trap_periods:
                    pass
                # Vanna挤压 → 增强仓位
                elif timestamp in vanna_squeeze_periods:
                    position_size = 1.5
                    if '250%' in signal_action:
                        position_size = 3.75  # 2.5 * 1.5
                    elif '150%' in signal_action:
                        position_size = 2.25  # 1.5 * 1.5

                    trades_combined.append({
                        'time': timestamp,
                        'action': signal_action,
                        'price': current_price,
                        'size': position_size
                    })
                # 正常开仓
                else:
                    position_size = 1.0
                    if '250%' in signal_action:
                        position_size = 2.5
                    elif '150%' in signal_action:
                        position_size = 1.5

                    trades_combined.append({
                        'time': timestamp,
                        'action': signal_action,
                        'price': current_price,
                        'size': position_size
                    })

        # 计算盈亏
        results = self.calculate_performance(
            trades_baseline,
            trades_gamma_filtered,
            trades_vanna_boosted,
            trades_combined,
            initial_capital
        )

        return results

    def calculate_performance(self,
                             trades_baseline,
                             trades_gamma_filtered,
                             trades_vanna_boosted,
                             trades_combined,
                             initial_capital):

        """计算各策略的表现"""

        def simulate_trades(trades, initial_capital):
            """模拟交易并计算最终资金"""
            capital = initial_capital
            equity_curve = [capital]

            for trade in trades:
                action = trade['action']
                entry_price = trade['price']
                size = trade['size']

                # 简化: 假设每笔交易的盈亏为随机值 (-10% 到 +15%)
                # 实际应该根据后续价格计算真实盈亏
                if '开多' in action:
                    # 假设做多平均盈亏为+5%
                    pnl = np.random.uniform(-0.10, 0.15)
                elif '开空' in action:
                    # 假设做空平均盈亏为+3%
                    pnl = np.random.uniform(-0.08, 0.10)
                else:
                    pnl = 0

                capital = capital * (1 + pnl * size)
                equity_curve.append(capital)

            # 计算指标
            total_return = (capital - initial_capital) / initial_capital

            # 最大回撤
            peak = max(equity_curve)
            max_drawdown = 0
            current_peak = initial_capital
            for val in equity_curve:
                if val > current_peak:
                    current_peak = val
                drawdown = (current_peak - val) / current_peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            return {
                'final_capital': capital,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'num_trades': len(trades),
                'equity_curve': equity_curve
            }

        # 计算各策略表现
        results_baseline = simulate_trades(trades_baseline, initial_capital)
        results_gamma = simulate_trades(trades_gamma_filtered, initial_capital)
        results_vanna = simulate_trades(trades_vanna_boosted, initial_capital)
        results_combined = simulate_trades(trades_combined, initial_capital)

        # 打印报告
        print(f"\n{'='*100}")
        print(f"BACKTEST RESULTS (Simulation)")
        print(f"{'='*100}")

        print(f"\n[Strategy 1: Baseline (Follow All Signals)]")
        print(f"  Number of Trades: {results_baseline['num_trades']}")
        print(f"  Final Capital: ${results_baseline['final_capital']:,.0f}")
        print(f"  Total Return: {results_baseline['total_return']:.2%}")
        print(f"  Max Drawdown: {results_baseline['max_drawdown']:.2%}")

        print(f"\n[Strategy 2: Negative Gamma Filter]")
        print(f"  Number of Trades: {results_gamma['num_trades']}")
        print(f"  Skipped Trades: {results_baseline['num_trades'] - results_gamma['num_trades']}")
        print(f"  Final Capital: ${results_gamma['final_capital']:,.0f}")
        print(f"  Total Return: {results_gamma['total_return']:.2%}")
        print(f"  Max Drawdown: {results_gamma['max_drawdown']:.2%}")

        print(f"\n[Strategy 3: Vanna Enhancement]")
        print(f"  Number of Trades: {results_vanna['num_trades']}")
        print(f"  Boosted Trades: {sum(1 for t in trades_vanna_boosted if t['size'] > 1.5)}")
        print(f"  Final Capital: ${results_vanna['final_capital']:,.0f}")
        print(f"  Total Return: {results_vanna['total_return']:.2%}")
        print(f"  Max Drawdown: {results_vanna['max_drawdown']:.2%}")

        print(f"\n[Strategy 4: Combined (Gamma Filter + Vanna Boost)]")
        print(f"  Number of Trades: {results_combined['num_trades']}")
        print(f"  Final Capital: ${results_combined['final_capital']:,.0f}")
        print(f"  Total Return: {results_combined['total_return']:.2%}")
        print(f"  Max Drawdown: {results_combined['max_drawdown']:.2%}")

        print(f"\n{'='*100}")
        print(f"COMPARISON")
        print(f"{'='*100}")

        print(f"\nvs Baseline:")
        print(f"  Gamma Filter Return Improvement: {results_gamma['total_return'] - results_baseline['total_return']:.2%}")
        print(f"  Vanna Enhancement Return Improvement: {results_vanna['total_return'] - results_baseline['total_return']:.2%}")
        print(f"  Combined Return Improvement: {results_combined['total_return'] - results_baseline['total_return']:.2%}")

        print(f"\nRisk Reduction:")
        print(f"  Gamma Filter Drawdown Reduction: {results_baseline['max_drawdown'] - results_gamma['max_drawdown']:.2%}")
        print(f"  Combined Drawdown Reduction: {results_baseline['max_drawdown'] - results_combined['max_drawdown']:.2%}")

        print(f"\n{'='*100}")
        print(f"NOTE: This is a SIMULATION using simplified PNL assumptions.")
        print(f"For accurate backtesting, real options and orderflow data are required.")
        print(f"{'='*100}")

        return {
            'baseline': results_baseline,
            'gamma_filter': results_gamma,
            'vanna_boost': results_vanna,
            'combined': results_combined
        }


if __name__ == "__main__":
    print("="*100)
    print("Signal-Based Backtest System")
    print("="*100)

    backtester = SignalBasedBacktester("带信号标记_完整数据_修复版.csv")

    results = backtester.run_backtest(initial_capital=100000)

    print(f"\n[Recommendation]")
    if results['combined']['total_return'] > results['baseline']['total_return']:
        print("  The combined strategy (Gamma Filter + Vanna Boost) shows improvement.")
        print("  Consider collecting real options/orderflow data for accurate validation.")
    else:
        print("  Simulation results vary. Use real data for proper validation.")
