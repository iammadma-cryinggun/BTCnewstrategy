# 分析Zeabur日志，检查程序是否真正启动

import re

log_content = """
2026-01-20 23:03:26,209 - INFO - [Telegram] Polling启动...
2026-01-20 22:46:07,917 - INFO - V7.0.8 智能交易系统启动（V7.0.7 + 黄金策略）
2026-01-20 22:46:08,663 - INFO - [系统] Telegram Polling已启动（后台线程）
2026-01-20 22:46:08,664 - INFO - 定时任务已设置：
2026-01-20 22:46:08,664 - INFO -   - 信号检查: 北京时间 0:00, 4:00, 8:00, 12:00, 16:00, 20:00
2026-01-20 22:46:08,664 - INFO -   - 黄金开仓检查: 每小时（V7.0.8新增）
2026-01-20 22:46:08,664 - INFO -   - 持仓检查: 每1小时
2026-01-20 22:46:08,664 - INFO - 
2026-01-20 22:46:08,664 - INFO - 进入主循环...
"""

# 检查关键启动信息
checks = {
    "V7.0.8系统启动": "V7.0.8 智能交易系统启动" in log_content,
    "定时任务设置": "定时任务已设置" in log_content,
    "进入主循环": "进入主循环" in log_content,
    "Telegram Polling": "Telegram Polling已启动" in log_content,
}

print("=== Zeabur程序启动状态检查 ===\n")
for check, result in checks.items():
    status = "[OK]" if result else "[FAIL]"
    print(f"{status} {check}")

# 检查是否有错误
errors = re.findall(r'ERROR - (.+)', log_content)
if errors:
    print(f"\n发现 {len(errors)} 个错误:")
    for err in errors[:3]:  # 只显示前3个
        print(f"  - {err[:80]}")
