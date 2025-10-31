#!/bin/bash
# 全数据集实验监控脚本

LOG_FILE="/root/autodl-tmp/full_dataset_experiment.log"

echo "================================================"
echo "🔍 MRAG-Bench全数据集实验监控"
echo "================================================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查进程
if ps aux | grep -q "[r]un_all_baselines_100samples.py"; then
    echo "✅ 实验进程运行中"
    UPTIME=$(ps -p $(pgrep -f "run_all_baselines_100samples.py") -o etime= | tr -d ' ')
    echo "运行时长: $UPTIME"
else
    echo "❌ 实验进程未运行"
fi

echo ""
echo "------------------------------------------------"
echo "📊 当前进度"
echo "------------------------------------------------"

# 获取当前评测的方法
CURRENT_METHOD=$(tail -100 "$LOG_FILE" | grep "评测方法:" | tail -1)
if [ ! -z "$CURRENT_METHOD" ]; then
    echo "$CURRENT_METHOD"
fi

# 获取最新进度条
PROGRESS_LINE=$(tail -100 "$LOG_FILE" | grep "运行.*%\|" | tail -1)
if [ ! -z "$PROGRESS_LINE" ]; then
    echo "$PROGRESS_LINE"
fi

echo ""
echo "------------------------------------------------"
echo "📈 已完成方法"
echo "------------------------------------------------"

# 统计已完成的方法
grep "✅.*完成:" "$LOG_FILE" | tail -20

echo ""
echo "------------------------------------------------"
echo "📝 最近日志（最后10行）"
echo "------------------------------------------------"
tail -10 "$LOG_FILE"

echo ""
echo "================================================"
echo "监控命令:"
echo "  tail -f $LOG_FILE  # 实时查看"
echo "  ./monitor_full_experiment.sh  # 再次检查"
echo "================================================"

