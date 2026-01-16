#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V7.0.7 交易系统 - Zeabur/GitHub自动部署入口
⭐ 这个文件会被Zeabur自动检测并运行
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    # ⭐ 检测是否在Zeabur/生产环境
    IS_ZEABUR = os.environ.get('ZEABUR') or os.environ.get('PORT')

    if IS_ZEABUR:
        # ⭐ Zeabur生产环境：使用Gunicorn
        print("[启动] Zeabur生产环境检测到")
        print("[启动] 导入Flask应用...")

        # 导入Flask应用（供Gunicorn使用）
        from main_production import app

        print("[启动] Flask应用已加载")
        print(f"[启动] 监听端口: {os.environ.get('PORT', 8080)}")

        # ⚠️ 注意：实际启动由Gunicorn完成
        # 这个文件只需要正确导入app即可
        # Gunicorn会调用: gunicorn -c gunicorn_config.py main_production:app
    else:
        # ⭐ 本地开发环境
        print("[启动] 本地开发环境")
        print("[启动] 使用Gunicorn生产服务器")

        # 使用subprocess启动Gunicorn（本地测试）
        import subprocess
        port = os.environ.get('PORT', 8080)
        cmd = [
            'gunicorn',
            '-c', 'gunicorn_config.py',
            'main_production:app',
            '--bind', f'0.0.0.0:{port}'
        ]
        print(f"[启动] 命令: {' '.join(cmd)}")
        subprocess.run(cmd)
