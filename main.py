#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V7.0.7 交易系统 - Zeabur生产入口
⭐ 使用Gunicorn生产服务器启动Flask应用
"""

import os
import sys
import subprocess

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    # ⭐ 生产环境：使用Gunicorn启动
    print("[启动] V7.0.7交易系统 - Gunicorn模式")
    print("[启动] 启动Gunicorn生产服务器...")

    # 获取端口
    port = int(os.environ.get('PORT', 8080))

    # Gunicorn启动命令
    cmd = [
        'gunicorn',
        '-c', 'gunicorn_config.py',
        'main_production:app',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '2',
        '--timeout', '120',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info'
    ]

    print(f"[启动] 命令: {' '.join(cmd)}")
    print(f"[启动] 端口: {port}")
    print("[启动] 正在启动...")

    # 启动Gunicorn（这会阻塞并持续运行）
    sys.exit(subprocess.run(cmd).returncode)
