# -*- coding: utf-8 -*-
"""
时间工具模块 - 统一处理时区转换
=================================

所有时间统一使用北京时间（UTC+8）
Binance API返回UTC时间，需要转换
"""

import pandas as pd
from datetime import datetime, timezone, timedelta

# 北京时区（UTC+8）
BEIJING_TZ = timezone(timedelta(hours=8))
# UTC时区
UTC_TZ = timezone.utc


def convert_to_beijing(timestamp):
    """
    将时间戳转换为北京时间

    参数:
        timestamp: 可以是多种格式
            - pandas Timestamp
            - datetime object
            - Unix timestamp (ms)

    返回:
        datetime: 北京时间的datetime对象（无时区信息，但值是北京时间）
    """
    if pd.isna(timestamp):
        return None

    # 如果是pandas Timestamp
    if isinstance(timestamp, pd.Timestamp):
        # 如果没有时区信息，假设是UTC
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize(UTC_TZ)
        # 转换为北京时间
        beijing_time = timestamp.tz_convert(BEIJING_TZ)
        # 返回无时区的datetime（但值是北京时间）
        return beijing_time.tz_localize(None).to_pydatetime()

    # 如果是datetime对象
    if isinstance(timestamp, datetime):
        # 如果没有时区信息，假设是UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC_TZ)
        # 转换为北京时间
        beijing_time = timestamp.astimezone(BEIJING_TZ)
        # 返回无时区的datetime（但值是北京时间）
        return beijing_time.replace(tzinfo=None)

    # 如果是数字（Unix timestamp毫秒）
    if isinstance(timestamp, (int, float)):
        # 从毫秒转换为秒
        utc_time = datetime.fromtimestamp(timestamp / 1000, tz=UTC_TZ)
        beijing_time = utc_time.astimezone(BEIJING_TZ)
        return beijing_time.replace(tzinfo=None)

    return None


def format_beijing_time(timestamp, format_str='%Y-%m-%d %H:%M:%S'):
    """
    格式化时间为北京时间字符串

    参数:
        timestamp: 时间戳（多种格式）
        format_str: 格式化字符串

    返回:
        str: 格式化后的北京时间字符串
    """
    beijing_time = convert_to_beijing(timestamp)
    if beijing_time is None:
        return 'N/A'
    return beijing_time.strftime(format_str)


def get_current_beijing_time():
    """
    获取当前北京时间

    返回:
        datetime: 当前北京时间（无时区信息）
    """
    utc_now = datetime.now(UTC_TZ)
    beijing_now = utc_now.astimezone(BEIJING_TZ)
    return beijing_now.replace(tzinfo=None)


def is_4h_kline_close_time(beijing_time):
    """
    检查是否是4小时K线收盘时间（北京时间）

    北京时间4小时K线收盘时间:
    00:00, 04:00, 08:00, 12:00, 16:00, 20:00

    参数:
        beijing_time: 北京时间的datetime对象

    返回:
        bool: 是否是收盘时间
    """
    if beijing_time is None:
        return False

    hour = beijing_time.hour
    minute = beijing_time.minute

    # 必须是整点，且小时是4的倍数
    return minute == 0 and hour % 4 == 0


def get_next_4h_close_time(beijing_time=None):
    """
    获取下一个4小时K线收盘时间

    参数:
        beijing_time: 基准时间（北京时间），默认为当前时间

    返回:
        datetime: 下一个收盘时间（北京时间）
    """
    if beijing_time is None:
        beijing_time = get_current_beijing_time()

    hour = beijing_time.hour
    minute = beijing_time.minute

    # 计算下一个4小时的倍数
    next_hour = ((hour // 4) + 1) * 4
    if next_hour >= 24:
        next_hour = 0
        next_date = beijing_time + timedelta(days=1)
    else:
        next_date = beijing_time

    return datetime(next_date.year, next_date.month, next_date.day, next_hour, 0, 0)


def seconds_until_next_4h_close(beijing_time=None):
    """
    计算距离下一个4小时K线收盘的秒数

    参数:
        beijing_time: 基准时间（北京时间），默认为当前时间

    返回:
        int: 秒数
    """
    next_close = get_next_4h_close_time(beijing_time)
    now = beijing_time if beijing_time else get_current_beijing_time()
    delta = next_close - now
    return int(delta.total_seconds())


# 测试代码
if __name__ == "__main__":
    # 测试时区转换
    test_timestamp = 1736913600000  # 2025-01-15 00:00:00 UTC
    beijing_time = convert_to_beijing(test_timestamp)
    print(f"UTC时间戳: {test_timestamp}")
    print(f"北京时间: {format_beijing_time(test_timestamp)}")
    print(f"是否收盘时间: {is_4h_kline_close_time(beijing_time)}")

    # 当前时间
    now_beijing = get_current_beijing_time()
    print(f"\n当前北京时间: {format_beijing_time(now_beijing)}")
    print(f"下一个收盘: {format_beijing_time(get_next_4h_close_time())}")
    print(f"距离收盘: {seconds_until_next_4h_close()}秒")
