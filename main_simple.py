# -*- coding: utf-8 -*-
"""
简化的云端测试系统 - 用于验证Zeabur部署
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import logging
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('simple.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主循环"""
    logger.info("=" * 70)
    logger.info("简化云端测试系统启动")
    logger.info("=" * 70)

    loop_count = 0
    heartbeat_interval = 30  # 每30次循环打印心跳（约30秒）

    try:
        while True:
            loop_count += 1

            # 心跳日志
            if loop_count % heartbeat_interval == 0:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"♥ [{current_time}] 系统运行中 - 循环次数: {loop_count}")

                # 模拟一些工作
                logger.info(f"  正在检查BTC价格...")
                logger.info(f"  当前价格: $90,000 (模拟)")
                logger.info(f"  置信度: 65% (模拟)")
                logger.info(f"  无交易信号，继续监控...")

            # 每秒检查一次
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n系统停止")
    except Exception as e:
        logger.error(f"系统异常: {e}", exc_info=True)

if __name__ == "__main__":
    main()
