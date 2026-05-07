#!/bin/bash
# 修复不确定性估计器问题并重新运行实验
# 生成时间: 2025-11-01

echo "========================================="
echo "修复不确定性估计器并重新运行实验"
echo "========================================="

# Step 1: 停止当前错误的实验
echo ""
echo "Step 1: 停止当前实验..."
ps aux | grep "run_all_baselines_100samples.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 2
echo "  ✅ 已停止当前实验"

# Step 2: 备份当前错误的日志
echo ""
echo "Step 2: 备份错误的实验日志..."
BACKUP_DIR="_BACKUP_CrossModalEstimator_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
mv full_dataset_CrossModal_experiment.log "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✅ 日志已备份到: $BACKUP_DIR"

# Step 3: 修改配置（改回ImprovedUncertaintyEstimator）
echo ""
echo "Step 3: 修改配置文件..."
cd /root/autodl-tmp/FlashRAG/experiments

# 备份原配置
cp run_all_baselines_100samples.py run_all_baselines_100samples.py.bak_$(date +%Y%m%d_%H%M%S)

# 修改配置：use_improved_estimator: False → True
sed -i "s/'use_improved_estimator': False/'use_improved_estimator': True/g" run_all_baselines_100samples.py

# 验证修改
if grep -q "'use_improved_estimator': True" run_all_baselines_100samples.py; then
    echo "  ✅ 配置已修改: use_improved_estimator = True"
else
    echo "  ❌ 配置修改失败，请手动修改"
    exit 1
fi

# Step 4: 重新运行实验
echo ""
echo "Step 4: 启动修复后的实验..."
cd /root/autodl-tmp

# 激活conda环境并运行
nohup bash -c "
    source ~/.bashrc
    conda activate multirag
    python -u /root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py 2>&1 | tee full_dataset_FIXED_experiment.log
" &

sleep 5

# 检查进程是否启动
if ps aux | grep "run_all_baselines_100samples.py" | grep -v grep > /dev/null; then
    echo "  ✅ 实验已启动"
    echo ""
    echo "查看进程:"
    ps aux | grep "run_all_baselines_100samples.py" | grep -v grep
    echo ""
    echo "实时日志: tail -f /root/autodl-tmp/full_dataset_FIXED_experiment.log"
else
    echo "  ❌ 实验启动失败"
    exit 1
fi

echo ""
echo "========================================="
echo "修复完成！"
echo "========================================="
echo ""
echo "预期改善:"
echo "  - 不确定性分布: 0.25-0.40 (不再是固定的0.47)"
echo "  - EM性能: 58-62% (目前47.3%)"
echo "  - 检索率: 60-65% (不再是100%)"
echo ""
echo "监控命令:"
echo "  1. 查看日志: tail -f full_dataset_FIXED_experiment.log"
echo "  2. 检查不确定性: grep 'uncertainty=' full_dataset_FIXED_experiment.log | head -20"
echo "  3. 查看进度: ps aux | grep run_all_baselines"
echo ""

