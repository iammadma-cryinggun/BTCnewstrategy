#!/bin/bash
echo "=========================================="
echo "清理Zeabur上的重复Bot服务"
echo "=========================================="

# 检查登录状态
echo "1. 检查Zeabur登录状态..."
zeabur profile info 2>&1 || {
    echo "❌ 未登录Zeabur，请先登录:"
    echo "   zeabur auth login"
    exit 1
}

# 列出所有项目
echo ""
echo "2. 列出所有项目..."
zeabur project list

echo ""
echo "3. 列出服务..."
zeabur service list

echo ""
echo "=========================================="
echo "如果发现有多个btc_4hour_alert相关的服务："
echo "  - 停止不需要的服务: zeabur service stop <service-name>"
echo "  - 或者删除: zeabur service delete <service-name>"
echo "=========================================="
