# -*- coding: utf-8 -*-
"""
期权微观结构策略回测系统

使用真实历史数据回测:
1. 期权数据 (GEX, IV, Skew) - 需要从Greeks.live历史数据获取
2. 订单流数据 (CVD, VPIN) - 需要从交易所历史数据获取
3. 价格数据 - 已有CSV文件

回测目标:
- 对比V8.0单独 vs V8.0 + 期权微观结构
- 计算收益率、Sharpe Ratio、Max Drawdown
- 验证负Gamma闪崩检测的规避效果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class OptionsBacktestData:
    """期权历史数据容器"""

    def __init__(self):
        self.data: pd.DataFrame = None

    def load_from_csv(self, filepath: str):
        """
        从CSV加载期权历史数据

        需要的列:
        - 时间
        - GEX (Gamma Exposure)
        - ATM_IV
        - IV_1D, IV_1W, IV_1M
        - Skew_Slope
        - Call_Skew
        - Put_Skew
        """
        try:
            self.data = pd.read_csv(filepath, encoding='utf-8-sig')
            self.data['时间'] = pd.to_datetime(self.data['时间'])
            print(f"[OK] Loaded options data: {len(self.data)} rows")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {filepath}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load options data: {e}")
            return False

    def get_greeks_at_time(self, timestamp: datetime) -> Optional[Dict]:
        """获取指定时间点的Greeks数据"""
        if self.data is None:
            return None

        row = self.data[self.data['时间'] == timestamp]
        if row.empty:
            return None

        return {
            'gex': row['GEX'].iloc[0],
            'atm_iv': row['ATM_IV'].iloc[0],
            'iv_term_structure': {
                '1D': row['IV_1D'].iloc[0],
                '1W': row['IV_1W'].iloc[0],
                '1M': row['IV_1M'].iloc[0]
            },
            'skew_slope': row['Skew_Slope'].iloc[0],
            'call_skew': row['Call_Skew'].iloc[0],
            'put_skew': row['Put_Skew'].iloc[0]
        }


class OrderFlowBacktestData:
    """订单流历史数据容器"""

    def __init__(self):
        self.data: pd.DataFrame = None

    def load_from_csv(self, filepath: str):
        """
        从CSV加载订单流历史数据

        需要的列:
        - 时间
        - CVD
        - VPIN
        - Bid_Quantity_1pct
        - Ask_Quantity_1pct
        - Sell_Pressure
        - Buy_Pressure
        """
        try:
            self.data = pd.read_csv(filepath, encoding='utf-8-sig')
            self.data['时间'] = pd.to_datetime(self.data['时间'])
            print(f"[OK] Loaded orderflow data: {len(self.data)} rows")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {filepath}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load orderflow data: {e}")
            return False

    def get_metrics_at_time(self, timestamp: datetime) -> Optional[Dict]:
        """获取指定时间点的订单流指标"""
        if self.data is None:
            return None

        row = self.data[self.data['时间'] == timestamp]
        if row.empty:
            return None

        return {
            'cvd': row['CVD'].iloc[0],
            'cvd_trend': row.get('CVD_Trend', 0).iloc[0],
            'price_trend': row.get('Price_Trend', 0).iloc[0],
            'bid_quantity_1pct': row['Bid_Quantity_1pct'].iloc[0],
            'vpin': row['VPIN'].iloc[0],
            'sell_pressure': row['Sell_Pressure'].iloc[0],
            'buy_pressure': row['Buy_Pressure'].iloc[0],
            'toxic_flow_ratio': row.get('Toxic_Flow_Ratio', 0).iloc[0]
        }


class MicrostructureBacktester:
    """期权微观结构策略回测引擎"""

    def __init__(self,
                 price_data_filepath: str,
                 options_data_filepath: Optional[str] = None,
                 orderflow_data_filepath: Optional[str] = None):

        # 加载价格数据
        print(f"\n[1] Loading price data...")
        self.price_data = pd.read_csv(price_data_filepath, encoding='utf-8-sig')
        self.price_data['时间'] = pd.to_datetime(self.price_data['时间'])
        print(f"[OK] Loaded price data: {len(self.price_data)} rows")

        # 加载期权数据
        self.options_data = OptionsBacktestData()
        if options_data_filepath:
            print(f"\n[2] Loading options data...")
            self.options_data.load_from_csv(options_data_filepath)
        else:
            print(f"\n[2] Options data: SKIPPED (no file provided)")

        # 加载订单流数据
        self.orderflow_data = OrderFlowBacktestData()
        if orderflow_data_filepath:
            print(f"\n[3] Loading orderflow data...")
            self.orderflow_data.load_from_csv(orderflow_data_filepath)
        else:
            print(f"\n[3] Orderflow data: SKIPPED (no file provided)")

        # 回测结果
        self.trades = []
        self.equity_curve = []

    def calculate_v8_score(self, row: pd.Series) -> float:
        """
        计算V8.0评分

        使用现有的指标:
        - 加速度 (加速度指标)
        - 张力 (张力指标)
        - 量能比率 (量能指标)
        """
        # 归一化处理
        accel = abs(row.get('加速度', 0))
        tension = abs(row.get('张力', 0))
        volume = row.get('量能比率', 1.0)

        # 简化的V8.0评分 (模拟)
        score = (
            min(accel / 0.3, 1.0) * 0.5 +  # EMA突变 50%
            min(tension / 1.0, 1.0) * 0.3 +  # 量能突变 30%
            min(volume / 2.0, 1.0) * 0.2    # 基础量能 20%
        )

        return score

    def check_negative_gamma_trap(self,
                                   greeks: Optional[Dict],
                                   orderflow: Optional[Dict],
                                   current_price: float) -> Tuple[bool, float]:
        """
        检查负Gamma陷阱

        返回: (是否陷阱, 置信度)
        """
        if not greeks or not orderflow:
            return False, 0.0

        confidence = 0.0

        # 条件1: GEX < -1亿
        if greeks['gex'] < -100000000:
            confidence += 0.4

        # 条件2: LCR < 1.0
        hedging_need = abs(greeks['gex']) * 0.01
        lcr = orderflow['bid_quantity_1pct'] / hedging_need if hedging_need > 0 else float('inf')
        if lcr < 1.0:
            confidence += 0.3

        # 条件3: Skew斜率 > 5.0
        if greeks['skew_slope'] > 5.0:
            confidence += 0.15

        # 条件4: VPIN > 0.4
        if orderflow['vpin'] > 0.4:
            confidence += 0.15

        is_trap = confidence >= 0.5
        return is_trap, confidence

    def check_vanna_squeeze(self,
                           greeks: Optional[Dict],
                           orderflow: Optional[Dict],
                           iv_percentile: float = 50.0) -> Tuple[bool, float]:
        """
        检查Vanna挤压

        返回: (是否Vanna挤压, 置信度)
        """
        if not greeks or not orderflow:
            return False, 0.0

        confidence = 0.0

        # 条件1: IV分位数 < 30%
        if iv_percentile < 30.0:
            confidence += 0.25

        # 条件2: Call Skew > 3.0
        if greeks['call_skew'] > 3.0:
            confidence += 0.20

        # 条件3: CVD背离 (价格跌, CVD涨)
        if orderflow['price_trend'] < 0 and orderflow['cvd_trend'] > 0:
            confidence += 0.35

        # 条件4: 买压 > 卖压
        if orderflow['buy_pressure'] > orderflow['sell_pressure'] * 1.5:
            confidence += 0.20

        is_squeeze = confidence >= 0.6
        return is_squeeze, confidence

    def run_backtest(self,
                    initial_capital: float = 100000,
                    v8_threshold: float = 0.7):
        """
        运行回测

        策略对比:
        1. V8.0单独 (Baseline)
        2. V8.0 + 负Gamma过滤
        3. V8.0 + Vanna过滤
        4. V8.0 + 完整期权微观结构
        """
        print(f"\n{'='*100}")
        print(f"BACKTEST: Options Microstructure Strategy")
        print(f"{'='*100}")

        capital_v8_only = initial_capital
        capital_with_gamma_filter = initial_capital
        capital_with_vanna_filter = initial_capital
        capital_with_full_filter = initial_capital

        position_v8_only = 0  # 0=空仓, 1=多仓
        position_with_gamma_filter = 0
        position_with_vanna_filter = 0
        position_with_full_filter = 0

        entry_price_v8_only = 0
        entry_price_with_gamma_filter = 0
        entry_price_with_vanna_filter = 0
        entry_price_with_full_filter = 0

        equity_curve = {
            'v8_only': [],
            'with_gamma_filter': [],
            'with_vanna_filter': [],
            'with_full_filter': []
        }

        skipped_trades_gamma = 0  # 被负Gamma过滤掉的交易
        boosted_trades_vanna = 0  # 被Vanna增强的交易

        print(f"\n[Backtest Parameters]")
        print(f"  Initial Capital: ${initial_capital:,.0f}")
        print(f"  V8 Threshold: {v8_threshold}")
        print(f"  Data Points: {len(self.price_data)}")

        print(f"\n[Running Backtest...]")

        for i, row in self.price_data.iterrows():
            if i < 10:  # 跳过前10行用于计算指标
                continue

            timestamp = row['时间']
            current_price = row['收盘价']

            # 获取期权和订单流数据
            greeks = self.options_data.get_greeks_at_time(timestamp) if self.options_data.data is not None else None
            orderflow = self.orderflow_data.get_metrics_at_time(timestamp) if self.orderflow_data.data is not None else None

            # 计算V8.0评分
            v8_score = self.calculate_v8_score(row)

            # 检查负Gamma陷阱
            is_gamma_trap, gamma_confidence = self.check_negative_gamma_trap(
                greeks, orderflow, current_price
            )

            # 检查Vanna挤压
            is_vanna_squeeze, vanna_confidence = self.check_vanna_squeeze(
                greeks, orderflow
            )

            # ==================== 策略1: V8.0单独 ====================
            if position_v8_only == 0 and v8_score >= v8_threshold:
                # 开多
                signal_action = row.get('信号动作', '')
                if pd.notna(signal_action) and '开多' in str(signal_action):
                    position_v8_only = 1
                    entry_price_v8_only = current_price

            elif position_v8_only == 1:
                # 平仓条件
                should_close = False
                if v8_score < v8_threshold * 0.5:  # V8分数下降
                    should_close = True
                elif pd.isna(row.get('信号动作', '')) or row.get('信号动作', '') == '':
                    should_close = True

                if should_close:
                    # 计算盈亏
                    pnl = (current_price - entry_price_v8_only) / entry_price_v8_only
                    capital_v8_only *= (1 + pnl)
                    position_v8_only = 0

            # ==================== 策略2: V8.0 + 负Gamma过滤 ====================
            if position_with_gamma_filter == 0 and v8_score >= v8_threshold:
                signal_action = row.get('信号动作', '')
                if pd.notna(signal_action) and '开多' in str(signal_action):
                    # 检查是否在负Gamma陷阱中
                    if is_gamma_trap:
                        # 跳过这次交易
                        skipped_trades_gamma += 1
                    else:
                        position_with_gamma_filter = 1
                        entry_price_with_gamma_filter = current_price

            elif position_with_gamma_filter == 1:
                should_close = False
                if v8_score < v8_threshold * 0.5:
                    should_close = True
                elif is_gamma_trap:  # 紧急平仓
                    should_close = True
                elif pd.isna(row.get('信号动作', '')) or row.get('信号动作', '') == '':
                    should_close = True

                if should_close:
                    pnl = (current_price - entry_price_with_gamma_filter) / entry_price_with_gamma_filter
                    capital_with_gamma_filter *= (1 + pnl)
                    position_with_gamma_filter = 0

            # ==================== 策略3: V8.0 + Vanna增强 ====================
            if position_with_vanna_filter == 0 and v8_score >= v8_threshold:
                signal_action = row.get('信号动作', '')
                if pd.notna(signal_action) and '开多' in str(signal_action):
                    position_with_vanna_filter = 1
                    entry_price_with_vanna_filter = current_price

                    # 如果是Vanna挤压，加大仓位 (模拟1.5倍)
                    if is_vanna_squeeze:
                        boosted_trades_vanna += 1

            elif position_with_vanna_filter == 1:
                should_close = False
                if v8_score < v8_threshold * 0.5:
                    should_close = True
                elif pd.isna(row.get('信号动作', '')) or row.get('信号动作', '') == '':
                    should_close = True

                if should_close:
                    pnl = (current_price - entry_price_with_vanna_filter) / entry_price_with_vanna_filter

                    # Vanna挤压时盈亏也放大1.5倍
                    if is_vanna_squeeze:
                        pnl *= 1.5

                    capital_with_vanna_filter *= (1 + pnl)
                    position_with_vanna_filter = 0

            # ==================== 策略4: V8.0 + 完整期权微观结构 ====================
            if position_with_full_filter == 0 and v8_score >= v8_threshold:
                signal_action = row.get('信号动作', '')
                if pd.notna(signal_action) and '开多' in str(signal_action):
                    # 负Gamma陷阱 → 跳过
                    if is_gamma_trap:
                        skipped_trades_gamma += 1
                    # Vanna挤压 → 开仓
                    elif is_vanna_squeeze:
                        position_with_full_filter = 1
                        entry_price_with_full_filter = current_price
                        boosted_trades_vanna += 1
                    # 正常情况 → 正常开仓
                    else:
                        position_with_full_filter = 1
                        entry_price_with_full_filter = current_price

            elif position_with_full_filter == 1:
                should_close = False
                if v8_score < v8_threshold * 0.5:
                    should_close = True
                elif is_gamma_trap:  # 紧急平仓
                    should_close = True
                elif pd.isna(row.get('信号动作', '')) or row.get('信号动作', '') == '':
                    should_close = True

                if should_close:
                    pnl = (current_price - entry_price_with_full_filter) / entry_price_with_full_filter
                    capital_with_full_filter *= (1 + pnl)
                    position_with_full_filter = 0

            # 记录净值曲线
            equity_curve['v8_only'].append(capital_v8_only)
            equity_curve['with_gamma_filter'].append(capital_with_gamma_filter)
            equity_curve['with_vanna_filter'].append(capital_with_vanna_filter)
            equity_curve['with_full_filter'].append(capital_with_full_filter)

        # 计算回测结果
        results = self.calculate_results(
            initial_capital,
            capital_v8_only,
            capital_with_gamma_filter,
            capital_with_vanna_filter,
            capital_with_full_filter,
            equity_curve,
            skipped_trades_gamma,
            boosted_trades_vanna
        )

        return results

    def calculate_results(self,
                         initial_capital: float,
                         final_capital_v8: float,
                         final_capital_gamma: float,
                         final_capital_vanna: float,
                         final_capital_full: float,
                         equity_curve: Dict,
                         skipped_trades: int,
                         boosted_trades: int) -> Dict:

        """计算回测结果"""

        def calculate_metrics(final_capital: float, curve: List[float]) -> Dict:
            """计算单个策略的指标"""
            total_return = (final_capital - initial_capital) / initial_capital

            # 计算最大回撤
            peak = initial_capital
            max_drawdown = 0
            for value in curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            # 计算Sharpe Ratio (简化版)
            returns = pd.Series(curve).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            return {
                'final_capital': final_capital,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe
            }

        results_v8 = calculate_metrics(final_capital_v8, equity_curve['v8_only'])
        results_gamma = calculate_metrics(final_capital_gamma, equity_curve['with_gamma_filter'])
        results_vanna = calculate_metrics(final_capital_vanna, equity_curve['with_vanna_filter'])
        results_full = calculate_metrics(final_capital_full, equity_curve['with_full_filter'])

        # 打印报告
        print(f"\n{'='*100}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*100}")

        print(f"\n[Strategy 1: V8.0 Only (Baseline)]")
        print(f"  Final Capital: ${results_v8['final_capital']:,.0f}")
        print(f"  Total Return: {results_v8['total_return']:.2%}")
        print(f"  Max Drawdown: {results_v8['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {results_v8['sharpe_ratio']:.2f}")

        print(f"\n[Strategy 2: V8.0 + Negative Gamma Filter]")
        print(f"  Final Capital: ${results_gamma['final_capital']:,.0f}")
        print(f"  Total Return: {results_gamma['total_return']:.2%}")
        print(f"  Max Drawdown: {results_gamma['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {results_gamma['sharpe_ratio']:.2f}")
        print(f"  Skipped Trades: {skipped_trades}")

        print(f"\n[Strategy 3: V8.0 + Vanna Enhancement]")
        print(f"  Final Capital: ${results_vanna['final_capital']:,.0f}")
        print(f"  Total Return: {results_vanna['total_return']:.2%}")
        print(f"  Max Drawdown: {results_vanna['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {results_vanna['sharpe_ratio']:.2f}")
        print(f"  Boosted Trades: {boosted_trades}")

        print(f"\n[Strategy 4: V8.0 + Full Microstructure Filter]")
        print(f"  Final Capital: ${results_full['final_capital']:,.0f}")
        print(f"  Total Return: {results_full['total_return']:.2%}")
        print(f"  Max Drawdown: {results_full['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {results_full['sharpe_ratio']:.2f}")

        # 对比分析
        print(f"\n{'='*100}")
        print(f"COMPARISON ANALYSIS")
        print(f"{'='*100}")

        print(f"\nvs Baseline (V8.0 Only):")
        print(f"  Gamma Filter Return Improvement: {(results_gamma['total_return'] - results_v8['total_return']):.2%}")
        print(f"  Vanna Enhancement Return Improvement: {(results_vanna['total_return'] - results_v8['total_return']):.2%}")
        print(f"  Full Filter Return Improvement: {(results_full['total_return'] - results_v8['total_return']):.2%}")

        print(f"\nRisk Reduction:")
        print(f"  Gamma Filter Drawdown Reduction: {(results_v8['max_drawdown'] - results_gamma['max_drawdown']):.2%}")
        print(f"  Full Filter Drawdown Reduction: {(results_v8['max_drawdown'] - results_full['max_drawdown']):.2%}")

        print(f"\n{'='*100}")

        return {
            'v8_only': results_v8,
            'with_gamma_filter': results_gamma,
            'with_vanna_filter': results_vanna,
            'with_full_filter': results_full,
            'skipped_trades': skipped_trades,
            'boosted_trades': boosted_trades
        }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("期权微观结构策略回测系统")
    print("="*100)

    print("\n[Data Requirements]")
    print("This backtest requires THREE data sources:")
    print("  1. Price data (CSV) - You have this")
    print("  2. Options data (CSV) - Greeks.live historical data")
    print("  3. Orderflow data (CSV) - Exchange historical data")

    print("\n[Current Status]")
    print("  Price data: Available")
    print("  Options data: NOT AVAILABLE - Need to collect from Greeks.live")
    print("  Orderflow data: NOT AVAILABLE - Need to collect from exchange")

    print("\n" + "="*100)
    print("OPTIONS:")
    print("="*100)
    print("1. Run baseline backtest (V8.0 only, using price data)")
    print("2. Instructions for collecting options data")
    print("3. Instructions for collecting orderflow data")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        # 运行基线回测 (仅价格数据)
        backtester = MicrostructureBacktester(
            price_data_filepath="带信号标记_完整数据_修复版.csv"
        )

        results = backtester.run_backtest(
            initial_capital=100000,
            v8_threshold=0.7
        )

    elif choice == "2":
        print("\n" + "="*100)
        print("OPTIONS DATA COLLECTION INSTRUCTIONS")
        print("="*100)

        print("\nTo collect historical options data from Greeks.live:")
        print("\nStep 1: Apply for Greeks.live API access")
        print("  Website: https://greeks.live/")
        print("  Contact: support@greeks.live")

        print("\nStep 2: Use this script to download historical data:")
        print("""
import requests
import pandas as pd
from datetime import datetime, timedelta

def download_options_history(symbol='BTC', start_date='2024-01-01', end_date='2025-01-22'):
    api_url = 'https://api.greeks.live/v1/historical/options-chain'

    params = {
        'symbol': symbol,
        'start': start_date,
        'end': end_date,
        'interval': '4h'  # 4小时 candles
    }

    response = requests.get(api_url, params=params)
    data = response.json()

    # Process data
    df = pd.DataFrame(data['data'])
    df.to_csv('btc_options_history.csv', index=False, encoding='utf-8-sig')

    print(f"Downloaded {len(df)} records")

if __name__ == '__main__':
    download_options_history()
        """)

    elif choice == "3":
        print("\n" + "="*100)
        print("ORDERFLOW DATA COLLECTION INSTRUCTIONS")
        print("="*100)

        print("\nTo collect historical orderflow data from Binance:")
        print("\nStep 1: Install kline library")
        print("  pip install binance-connector")

        print("\nStep 2: Use this script to download historical data:")
        print("""
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

def download_orderflow_history(symbol='BTCUSDT', interval='1h', days=365):
    client = Client()

    # Download klines (candlestick data)
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_str=str(datetime.now() - timedelta(days=days))
    )

    # Process data
    data = []
    for k in klines:
        data.append({
            '时间': datetime.fromtimestamp(k[0]//1000),
            'Open': float(k[1]),
            'High': float(k[2]),
            'Low': float(k[3]),
            'Close': float(k[4]),
            'Volume': float(k[5])
        })

    df = pd.DataFrame(data)

    # Calculate CVD (simplified)
    df['CVD'] = df['Volume'].cumsum()
    df['VPIN'] = 0.3  # Placeholder

    df.to_csv('btc_orderflow_history.csv', index=False, encoding='utf-8-sig')
    print(f"Downloaded {len(df)} records")

if __name__ == '__main__':
    download_orderflow_history()
        """)

    else:
        print("\nInvalid choice. Exiting.")

    print("\n" + "="*100)
    print("For full backtesting with options microstructure filters,")
    print("you need to collect historical options and orderflow data first.")
    print("="*100)
