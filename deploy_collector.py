# -*- coding: utf-8 -*-
"""
自动数据收集系统

从Binance收集实时BTC价格数据，为未来的微观结构回测做准备
每4小时自动收集一次
"""

import time
import schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


# ==================== 简化微观结构监控器 ====================

class SimplifiedMicrostructureMonitor:
    """简化版微观结构监控器"""

    def __init__(self, lookback_periods: int = 20):
        self.lookback = lookback_periods
        self.price_history: List[float] = []
        self.volume_history: List[float] = []

    def calculate_volatility(self, prices: List[float]) -> float:
        """计算历史波动率"""
        if len(prices) < 2:
            return 0.0

        returns = pd.Series(prices).pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        volatility = returns.std() * np.sqrt(252)
        return volatility

    def calculate_volume_surge(self, current_volume: float) -> bool:
        """检测量能激增"""
        if len(self.volume_history) < self.lookback:
            return False

        avg_volume = np.mean(self.volume_history[-self.lookback:])
        is_surge = current_volume > avg_volume * 1.5

        return is_surge

    def calculate_liquidity_score(self, price_change: float, volume: float) -> float:
        """计算流动性评分"""
        volume_score = min(volume / 1000000, 1.0)
        volatility_score = max(1 - abs(price_change) / 0.05, 0)
        liquidity = (volume_score + volatility_score) / 2
        return liquidity

    def detect_crash_risk(self, volatility: float, acceleration: float, liquidity_score: float) -> tuple:
        """检测闪崩风险"""
        reasons = []
        confidence = 0.0

        if volatility > 0.8:
            reasons.append(f"Volatility spike: {volatility:.1%}")
            confidence += 0.4

        if acceleration < -0.2:
            reasons.append(f"Price crash: accel={acceleration:.3f}")
            confidence += 0.4

        if liquidity_score < 0.3:
            reasons.append(f"Liquidity dry: score={liquidity_score:.2f}")
            confidence += 0.2

        is_crash_risk = confidence >= 0.5
        return is_crash_risk, confidence, reasons

    def detect_squeeze_setup(self, volatility: float, volume_surge: bool, liquidity_score: float, price_acceleration: float) -> tuple:
        """检测暴涨设置"""
        reasons = []
        confidence = 0.0

        if volatility < 0.3:
            reasons.append(f"Low volatility: {volatility:.1%}")
            confidence += 0.3

        if volume_surge and abs(price_acceleration) < 0.1:
            reasons.append("Volume surge without price move")
            confidence += 0.4

        if liquidity_score > 0.7:
            reasons.append(f"High liquidity: score={liquidity_score:.2f}")
            confidence += 0.3

        is_squeeze = confidence >= 0.6
        return is_squeeze, confidence, reasons

    def analyze_row(self, row: pd.Series) -> Dict:
        """分析单行数据"""
        timestamp = pd.to_datetime(row['时间'])
        price = row['收盘价']
        volume = row.get('成交量', 1000000)

        # 更新历史
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > 100:
            self.price_history.pop(0)
            self.volume_history.pop(0)

        # 计算指标
        volatility = self.calculate_volatility(self.price_history)

        if len(self.price_history) >= 3:
            price_change_1 = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            price_change_2 = (self.price_history[-2] - self.price_history[-3]) / self.price_history[-3]
            acceleration = price_change_1 - price_change_2
        else:
            acceleration = 0

        volume_surge = self.calculate_volume_surge(volume)
        liquidity_score = self.calculate_liquidity_score(acceleration, volume)

        # 检测
        is_crash, crash_conf, crash_reasons = self.detect_crash_risk(volatility, acceleration, liquidity_score)
        is_squeeze, squeeze_conf, squeeze_reasons = self.detect_squeeze_setup(volatility, volume_surge, liquidity_score, acceleration)

        # 建议
        if is_crash:
            recommendation = "【警告】高闪崩风险 - 减仓或观望"
        elif is_squeeze:
            recommendation = "【机会】低波动吸筹 - 准备突破"
        elif liquidity_score < 0.4:
            recommendation = "【注意】流动性不足 - 谨慎交易"
        else:
            recommendation = "【正常】市场平稳 - 按策略交易"

        return {
            'timestamp': timestamp,
            'volatility': volatility,
            'liquidity_score': liquidity_score,
            'is_crash_risk': is_crash,
            'crash_confidence': crash_conf,
            'is_squeeze': is_squeeze,
            'squeeze_confidence': squeeze_conf,
            'recommendation': recommendation
        }


# ==================== 数据收集器 ====================

class BinanceDataCollector:
    """Binance数据收集器"""

    def __init__(self, output_file: str = "realtime_microstructure_data.csv"):
        self.output_file = output_file
        self.monitor = SimplifiedMicrostructureMonitor()
        self.data_buffer = []

        # 尝试加载现有数据
        try:
            self.existing_df = pd.read_csv(output_file, encoding='utf-8-sig')
            self.existing_df['时间'] = pd.to_datetime(self.existing_df['时间'])
            print(f"[OK] 找到历史数据: {len(self.existing_df)} 条记录")
        except FileNotFoundError:
            self.existing_df = pd.DataFrame()
            print(f"[INFO] 未找到历史数据，将创建新文件")

    def get_binance_data(self) -> dict:
        """从Binance获取实时数据"""
        try:
            # 24小时ticker数据
            url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                'price': float(data['lastPrice']),
                'volume': float(data['volume']),
                'quote_volume': float(data['quoteVolume']),
                'price_change_percent': float(data['priceChangePercent']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice'])
            }
        except Exception as e:
            print(f"[ERROR] Binance API失败: {e}")
            return None

    def collect_and_analyze(self):
        """收集数据并分析"""
        timestamp = datetime.now()

        print(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] 收集数据...")

        # 获取市场数据
        market_data = self.get_binance_data()
        if not market_data:
            return False

        # 构造DataFrame行
        row = pd.Series({
            '时间': timestamp,
            '收盘价': market_data['price'],
            '成交量': market_data['volume'],
            '最高价': market_data['high'],
            '最低价': market_data['low']
        })

        # 使用简化微观结构分析
        try:
            analysis = self.monitor.analyze_row(row)

            # 保存数据点
            data_point = {
                '时间': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                '价格': market_data['price'],
                '成交量': market_data['volume'],
                '成交额': market_data['quote_volume'],
                '涨跌幅': market_data['price_change_percent'],
                '波动率': f"{analysis['volatility']:.4f}",
                '流动性评分': f"{analysis['liquidity_score']:.2f}",
                '闪崩风险': 'YES' if analysis['is_crash_risk'] else 'No',
                '闪崩置信度': f"{analysis['crash_confidence']:.2f}",
                '暴涨设置': 'YES' if analysis['is_squeeze'] else 'No',
                '暴涨置信度': f"{analysis['squeeze_confidence']:.2f}",
                '建议': analysis['recommendation']
            }

            self.data_buffer.append(data_point)

            # 保存到文件
            self.save()

            print(f"[OK] 价格: ${market_data['price']:,.0f}")
            print(f"     成交量: {market_data['volume']:,.0f} BTC")
            print(f"     涨跌幅: {market_data['price_change_percent']:+.2f}%")
            print(f"     波动率: {analysis['volatility']:.2%}")
            print(f"     建议: {analysis['recommendation']}")

            return True

        except Exception as e:
            print(f"[ERROR] 分析失败: {e}")
            return False

    def save(self):
        """保存数据到CSV"""
        try:
            # 转换为DataFrame
            new_df = pd.DataFrame(self.data_buffer)

            # 追加到现有数据
            if not self.existing_df.empty:
                combined_df = pd.concat([self.existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            # 保存
            combined_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')

            print(f"[SAVED] {len(new_df)} 条记录 → {self.output_file} (总计: {len(combined_df)} 条)")

            # 清空buffer
            self.data_buffer = []

            # 更新existing_df
            self.existing_df = combined_df

        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")

    def get_statistics(self):
        """获取数据统计"""
        if self.existing_df.empty:
            return None

        df = self.existing_df

        stats = {
            'total_records': len(df),
            'date_range': f"{df['时间'].min()} to {df['时间'].max()}",
            'price_range': f"${df['价格'].min():,.0f} - ${df['价格'].max():,.0f}",
            'avg_volume': df['成交量'].mean(),
            'crash_risk_count': len(df[df['闪崩风险'] == 'YES']),
            'squeeze_setup_count': len(df[df['暴涨设置'] == 'YES'])
        }

        return stats


def main():
    """主函数"""
    print("="*100)
    print("BTC实时数据收集系统")
    print("="*100)

    print("\n[系统配置]")
    print("  数据源: Binance Public API (免费)")
    print("  交易对: BTCUSDT")
    print("  收集频率: 每4小时")
    print("  输出文件: realtime_microstructure_data.csv")

    print("\n[功能]")
    print("  1. 实时价格收集")
    print("  2. 简化微观结构分析")
    print("  3. 自动保存到CSV")

    # 创建收集器
    collector = BinanceDataCollector()

    # 显示统计
    stats = collector.get_statistics()
    if stats:
        print(f"\n[已有数据统计]")
        print(f"  总记录数: {stats['total_records']}")
        print(f"  时间范围: {stats['date_range']}")
        print(f"  价格范围: {stats['price_range']}")
        print(f"  平均成交量: {stats['avg_volume']:,.0f} BTC")
        print(f"  闪崩风险: {stats['crash_risk_count']} 次")
        print(f"  暴涨设置: {stats['squeeze_setup_count']} 次")

    print("\n" + "="*100)
    print("启动选项:")
    print("="*100)
    print("1. 立即收集一次，然后退出")
    print("2. 启动自动收集 (每4小时)")
    print("3. 查看统计，然后退出")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        print("\n[收集数据...]")
        success = collector.collect_and_analyze()

        if success:
            print("\n[完成] 数据已收集并保存")
        else:
            print("\n[失败] 数据收集失败")

    elif choice == "2":
        print("\n[启动自动收集]")
        print("  系统将每4小时自动收集一次数据")
        print("  按 Ctrl+C 停止")
        print("\n收集中... (首次立即执行)")

        # 立即执行一次
        collector.collect_and_analyze()

        # 定时任务
        schedule.every(4).hours.do(collector.collect_and_analyze)

        # 保持运行
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次

        except KeyboardInterrupt:
            print("\n\n[停止] 数据收集已停止")
            print(f"[总计] 数据已保存到 {collector.output_file}")

            # 显示最终统计
            stats = collector.get_statistics()
            if stats:
                print(f"\n[最终统计]")
                print(f"  总记录数: {stats['total_records']}")
                print(f"  闪崩风险: {stats['crash_risk_count']} 次")
                print(f"  暴涨设置: {stats['squeeze_setup_count']} 次")

    elif choice == "3":
        stats = collector.get_statistics()
        if stats:
            print(f"\n[数据统计]")
            print(f"  总记录数: {stats['total_records']}")
            print(f"  时间范围: {stats['date_range']}")
            print(f"  价格范围: {stats['price_range']}")
            print(f"  平均成交量: {stats['avg_volume']:,.0f} BTC")
            print(f"  闪崩风险: {stats['crash_risk_count']} 次 ({stats['crash_risk_count']/stats['total_records']*100:.1f}%)")
            print(f"  暴涨设置: {stats['squeeze_setup_count']} 次 ({stats['squeeze_setup_count']/stats['total_records']*100:.1f}%)")
        else:
            print("\n[INFO] 暂无数据")

    else:
        print("\n[无效选择]")

    print("\n" + "="*100)
    print("提示:")
    print("="*100)
    print("  - 数据保存到: realtime_microstructure_data.csv")
    print("  - 3个月后将有 ~540 条数据点")
    print("  - 届时可以进行简化版微观结构回测")
    print("  - 使用 python analyze_collected_data.py 查看分析")


if __name__ == "__main__":
    main()
