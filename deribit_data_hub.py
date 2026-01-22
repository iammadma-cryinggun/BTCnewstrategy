# -*- coding: utf-8 -*-
"""
Deribit期权数据中台

数据来源:
- Deribit public API (无需认证)
- 接口: get_book_summary_by_currency, get_order_book

功能:
1. 获取所有期权Greeks (Delta, Gamma, Vega, Theta)
2. 获取订单深度和隐含波动率
3. 计算订单墙位置
4. 识别最大痛点、Gamma暴露
5. Vanna挤压检测

作者: Claude Sonnet 4.5
日期: 2026-01-22
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DeribitDataHub:
    """Deribit数据中台"""

    def __init__(self):
        self.base_url = "https://www.deribit.com/api/v2/public"
        self.currency = "BTC"  # BTC or ETH
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'V8.0-Trading-System/1.0'
        })

    def get_book_summary_by_currency(self, currency: str = None) -> Optional[Dict]:
        """
        获取所有期权的Greeks摘要

        返回数据包括:
        - delta: Delta值
        - gamma: Gamma值
        - vega: Vega值
        - theta: Theta值
        - mark_iv: 标记隐含波动率
        - open_interest: 未平仓合约(OI)

        API文档: https://docs.deribit.com/#public-get_book_summary_by_currency
        """
        if currency is None:
            currency = self.currency

        url = f"{self.base_url}/get_book_summary_by_currency"
        params = {
            'currency': currency,
            'kind': 'option'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('error'):
                logger.error(f"Deribit API错误: {data['error']}")
                return None

            result = data.get('result', [])
            logger.info(f"获取到 {len(result)} 个期权合约")

            return result

        except Exception as e:
            logger.error(f"获取期权摘要失败: {e}")
            return None

    def get_order_book(self, instrument_name: str, depth: int = 5) -> Optional[Dict]:
        """
        获取特定期权的订单簿深度

        API文档: https://docs.deribit.com/#public-get_order_book
        """
        url = f"{self.base_url}/get_order_book"
        params = {
            'instrument_name': instrument_name,
            'depth': depth
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('error'):
                logger.error(f"获取订单簿失败 {instrument_name}: {data['error']}")
                return None

            return data.get('result')

        except Exception as e:
            logger.error(f"获取订单簿异常: {e}")
            return None

    def parse_options_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """解析期权数据为DataFrame"""
        if not raw_data:
            return pd.DataFrame()

        records = []
        for item in raw_data:
            try:
                option_name = item.get('instrument_name', '')
                parts = option_name.split('-')

                if len(parts) < 4:
                    continue

                expiry_str = parts[1]
                strike = float(parts[2])
                option_type = parts[3]

                record = {
                    'option_name': option_name,
                    'strike': strike,
                    'expiry': expiry_str,
                    'type': option_type,
                    'underlying_price': item.get('underlying_price', 0),
                    'mark_price': item.get('mark_price', 0),
                    'mark_iv': item.get('mark_iv', 0),
                    'delta': item.get('delta', 0),
                    'gamma': item.get('gamma', 0),
                    'vega': item.get('vega', 0),
                    'theta': item.get('theta', 0),
                    'oi': item.get('open_interest', 0),
                    'volume_24h': item.get('volume_24h', 0),
                }
                records.append(record)

            except Exception as e:
                continue

        df = pd.DataFrame(records)
        logger.info(f"解析完成: {len(df)} 个有效期权合约")

        return df

    def calculate_gamma_exposure(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        计算Gamma暴露

        Gamma暴露 = sum(Gamma × OI × 标的价格 × 100)

        优化：只使用Gamma不为0的期权（活跃期权）
        """
        if df.empty:
            return {}

        # 基础过滤：有OI且标的价格有效
        df_valid = df[(df['oi'] > 0) & (df['underlying_price'] > 0)].copy()

        if df_valid.empty:
            logger.warning("没有有效的期权数据用于Gamma计算")
            return {}

        # 关键优化：只使用Gamma不为0的期权（活跃的近月期权）
        df_active = df_valid[df_valid['gamma'] != 0].copy()

        if df_active.empty:
            logger.warning(f"所有期权Gamma都为0（可能是远期期权），使用全部{len(df_valid)}个期权")
            df_active = df_valid
        else:
            logger.info(f"使用{len(df_active)}/{len(df_valid)}个活跃期权计算Gamma")

        underlying_price = df_active['underlying_price'].iloc[0]
        multiplier = 100

        df_active['gamma_exposure'] = (
            df_active['gamma'] *
            df_active['oi'] *
            underlying_price *
            multiplier
        )

        call_gamma = df_active[df_active['type'] == 'C']['gamma_exposure'].sum()
        put_gamma = df_active[df_active['type'] == 'P']['gamma_exposure'].sum()

        result = {
            'total_gamma_exposure': call_gamma + put_gamma,
            'call_gamma_exposure': call_gamma,
            'put_gamma_exposure': put_gamma,
            'net_gamma_exposure': call_gamma - put_gamma,
            'underlying_price': underlying_price,
            'active_options_count': len(df_active),
            'total_options_count': len(df_valid)
        }

        logger.info(f"Gamma暴露: Call={call_gamma:.0f}, Put={put_gamma:.0f}, Net={result['net_gamma_exposure']:.0f}")
        logger.info(f"  活跃期权: {len(df_active)}/{len(df_valid)} ({len(df_active)/len(df_valid)*100:.1f}%)")

        return result

    def find_max_pain(self, df: pd.DataFrame) -> Optional[float]:
        """寻找最大痛点"""
        if df.empty:
            return None

        df_valid = df[df['oi'] > 0].copy()

        if df_valid.empty:
            return None

        underlying_price = df_valid['underlying_price'].iloc[0]

        grouped = df_valid.groupby('strike').agg({
            'oi': 'sum',
            'type': lambda x: list(x)
        }).reset_index()

        max_pain_value = 0
        max_pain_price = underlying_price

        for _, row in grouped.iterrows():
            strike = row['strike']
            total_oi = row['oi']
            types = row['type']

            pain_value = 0
            if 'C' in types:
                call_oi = df_valid[(df_valid['strike'] == strike) & (df_valid['type'] == 'C')]['oi'].sum()
                pain_value += call_oi * max(0, underlying_price - strike)

            if 'P' in types:
                put_oi = df_valid[(df_valid['strike'] == strike) & (df_valid['type'] == 'P')]['oi'].sum()
                pain_value += put_oi * max(0, strike - underlying_price)

            if pain_value > max_pain_value:
                max_pain_value = pain_value
                max_pain_price = strike

        logger.info(f"最大痛点: ${max_pain_price:,.0f} (当前价: ${underlying_price:,.0f})")

        return max_pain_price

    def identify_order_walls(self, df: pd.DataFrame, threshold_btc: float = None, top_n: int = 10) -> List[Dict]:
        """
        识别订单墙 (大额OI集中区)

        参数:
        - threshold_btc: 固定阈值（BTC）。如果为None，则使用动态阈值（前90%分位数）
        - top_n: 返回最多的墙数量
        """
        if df.empty:
            return []

        df_valid = df[df['oi'] > 0].copy()

        if df_valid.empty:
            return []

        underlying_price = df_valid['underlying_price'].iloc[0]

        grouped = df_valid.groupby(['strike', 'type'])['oi'].sum().reset_index()
        grouped['oi_btc'] = grouped['oi'] / 100

        # 动态阈值：如果未指定，使用75%分位数
        if threshold_btc is None:
            threshold_btc = grouped['oi_btc'].quantile(0.75)
            # 最小阈值100 BTC
            threshold_btc = max(threshold_btc, 100)
            logger.info(f"使用动态阈值: {threshold_btc:.0f} BTC (75%分位数)")

        order_walls = []
        for _, row in grouped.iterrows():
            oi_btc = row['oi_btc']

            if oi_btc >= threshold_btc:
                distance_pct = (row['strike'] - underlying_price) / underlying_price * 100

                wall = {
                    'strike': row['strike'],
                    'oi_btc': oi_btc,
                    'type': 'CALL' if row['type'] == 'C' else 'PUT',
                    'distance_pct': distance_pct,
                    'is_resistance': row['type'] == 'C',
                    'is_support': row['type'] == 'P',
                }
                order_walls.append(wall)

        # 按OI大小排序，只返回前N个
        order_walls.sort(key=lambda x: x['oi_btc'], reverse=True)
        order_walls = order_walls[:top_n]

        if order_walls:
            logger.info(f"发现 {len(order_walls)} 个订单墙 (阈值: {threshold_btc:.0f} BTC):")
            for i, wall in enumerate(order_walls[:5], 1):  # 只打印前5个
                dist_str = f"{wall['distance_pct']:+.2f}%"
                logger.info(f"  {i}. {wall['type']} ${wall['strike']:,.0f} - {wall['oi_btc']:.0f} BTC ({dist_str})")

        return order_walls

    def detect_vanna_squeeze(self, df: pd.DataFrame) -> Dict[str, any]:
        """检测Vanna挤压风险"""
        result = {
            'is_squeeze': False,
            'confidence': 0.0,
            'reason': ''
        }

        if df.empty:
            return result

        gamma_exp = self.calculate_gamma_exposure(df)

        if not gamma_exp:
            return result

        net_gamma = gamma_exp.get('net_gamma_exposure', 0)
        underlying_price = gamma_exp.get('underlying_price', 0)

        gamma_threshold = -1000000000
        gamma_score = min(abs(net_gamma) / abs(gamma_threshold), 1.0)

        avg_iv = df['mark_iv'].mean()
        iv_threshold = 0.5
        iv_score = min(avg_iv / iv_threshold, 1.0)

        walls = self.identify_order_walls(df, threshold_btc=1000)
        nearest_wall_distance = float('inf')

        for wall in walls:
            distance = abs(wall['distance_pct'])
            if distance < nearest_wall_distance:
                nearest_wall_distance = distance

        wall_score = max(1 - nearest_wall_distance / 10, 0)

        confidence = (gamma_score * 0.5 + iv_score * 0.3 + wall_score * 0.2)

        if confidence > 0.7:
            result['is_squeeze'] = True
            result['confidence'] = confidence
            result['reason'] = f"Net Gamma: {net_gamma:.0f}, IV: {avg_iv:.2%}, Wall: {nearest_wall_distance:.2f}%"

        return result


# ==================== 测试代码 ====================

def test_deribit_hub():
    """测试Deribit数据中台"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("Deribit数据中台测试")
    print("=" * 70)

    hub = DeribitDataHub()

    # 1. 获取期权摘要
    print("\n1. 获取期权数据...")
    raw_data = hub.get_book_summary_by_currency("BTC")

    if not raw_data:
        print("❌ 获取数据失败")
        return

    # 2. 解析数据
    print("\n2. 解析期权数据...")
    df = hub.parse_options_data(raw_data)

    if df.empty:
        print("❌ 解析失败")
        return

    print(f"✅ 解析成功: {len(df)} 个期权合约")
    print(df[['strike', 'type', 'delta', 'gamma', 'oi']].head(10))

    # 3. 计算Gamma暴露
    print("\n3. 计算Gamma暴露...")
    gamma_exp = hub.calculate_gamma_exposure(df)

    if gamma_exp:
        for key, value in gamma_exp.items():
            if key != 'underlying_price':
                print(f"  {key}: {value:,.0f}")

    # 4. 寻找最大痛点
    print("\n4. 寻找最大痛点...")
    max_pain = hub.find_max_pain(df)

    if max_pain:
        print(f"  最大痛点: ${max_pain:,.0f}")

    # 5. 识别订单墙
    print("\n5. 识别订单墙...")
    walls = hub.identify_order_walls(df, threshold_btc=None)  # 使用动态阈值

    for i, wall in enumerate(walls[:5], 1):
        print(f"  墙{i}: ${wall['strike']:,.0f} ({wall['type']}) - {wall['oi_btc']:.0f} BTC")

    # 6. 检测Vanna挤压
    print("\n6. 检测Vanna挤压...")
    squeeze = hub.detect_vanna_squeeze(df)

    if squeeze['is_squeeze']:
        print(f"  ⚠️ 检测到Vanna挤压! 置信度: {squeeze['confidence']:.2%}")
        print(f"  原因: {squeeze['reason']}")
    else:
        print("  ✅ 未检测到挤压风险")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_deribit_hub()
