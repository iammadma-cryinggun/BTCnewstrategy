#!/bin/bash
echo "=========================================="
echo "快速修复Telegram 409错误"
echo "=========================================="
echo ""
echo "步骤1：检查Zeabur登录状态"
zeabur profile info 2>&1 || {
    echo "❌ 未登录，正在打开浏览器..."
    zeabur auth login
}

echo ""
echo "步骤2：列出所有服务"
zeabur service list

echo ""
echo "步骤3：检查是否有重复的btc_4hour_alert服务"
echo "   如果有多个，请停止不需要的："
echo "   zeabur service stop <service-name>"
echo ""
echo "步骤4：重新部署（确保只有一个实例）"
echo "   zeabur deploy"
echo ""
echo "=========================================="
