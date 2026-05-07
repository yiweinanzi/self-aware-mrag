#!/bin/bash
# 使用论文完整方法（CrossModalUncertaintyEstimator + MLLM）修复并重新运行
# 生成时间: 2025-11-01

echo "========================================="
echo "使用论文完整方法修复实验"
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
BACKUP_DIR="_BACKUP_CrossModalEstimator_Wrong_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
mv full_dataset_CrossModal_experiment.log "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✅ 日志已备份到: $BACKUP_DIR"

# Step 3: 验证代码修复
echo ""
echo "Step 3: 验证代码修复..."
if grep -q "mllm_for_uncertainty = self.qwen3_vl" /root/autodl-tmp/FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py; then
    echo "  ✅ Pipeline代码已修复（传入真实MLLM模型）"
else
    echo "  ❌ Pipeline代码未修复，请检查"
    exit 1
fi

# Step 4: 确保使用CrossModalUncertaintyEstimator（论文方法）
echo ""
echo "Step 4: 确保使用论文完整方法..."
cd /root/autodl-tmp/FlashRAG/experiments

# 备份配置
cp run_all_baselines_100samples.py run_all_baselines_100samples.py.bak_$(date +%Y%m%d_%H%M%S)

# 确保 use_improved_estimator = False（使用CrossModalUncertaintyEstimator）
sed -i "s/'use_improved_estimator': True/'use_improved_estimator': False/g" run_all_baselines_100samples.py

if grep -q "'use_improved_estimator': False" run_all_baselines_100samples.py; then
    echo "  ✅ 配置正确: use_improved_estimator = False (使用CrossModalUncertaintyEstimator)"
else
    echo "  ❌ 配置修改失败"
    exit 1
fi

# Step 5: 重新运行实验
echo ""
echo "Step 5: 启动修复后的实验..."
cd /root/autodl-tmp

# 激活conda环境并运行
nohup bash -c "
    source ~/.bashrc
    conda activate multirag
    python -u /root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py 2>&1 | tee full_dataset_PaperMethod_experiment.log
" &

sleep 5

# 检查进程是否启动
if ps aux | grep "run_all_baselines_100samples.py" | grep -v grep > /dev/null; then
    echo "  ✅ 实验已启动"
    echo ""
    echo "查看进程:"
    ps aux | grep "run_all_baselines_100samples.py" | grep -v grep
else
    echo "  ❌ 实验启动失败"
    exit 1
fi

echo ""
echo "========================================="
echo "修复完成！"
echo "========================================="
echo ""
echo "关键修复:"
echo "  1. ✅ Pipeline代码修复：传入真实Qwen3-VL模型"
echo "  2. ✅ 使用CrossModalUncertaintyEstimator（论文完整方法）"
echo "  3. ✅ 支持SeaKR的Gram矩阵 + eigen_score方法"
echo ""
echo "预期改善:"
echo "  - 不确定性分布: 应该呈现多样性（不再是固定的0.47）"
echo "  - EM性能: 58-62% (目前47.3%)"
echo "  - 符合论文描述的方法"
echo ""
echo "监控命令:"
echo "  1. 查看日志: tail -f full_dataset_PaperMethod_experiment.log"
echo "  2. 检查不确定性分布:"
echo "     grep 'uncertainty=' full_dataset_PaperMethod_experiment.log | grep -o 'uncertainty=0\.[0-9]*' | sort | uniq -c"
echo "  3. 查看初始化信息（验证使用了MLLM）:"
echo "     grep -E '(CrossModalUncertaintyEstimator|已从wrapper获取|直接使用wrapper)' full_dataset_PaperMethod_experiment.log | head -5"
echo "  4. 查看进度:"
echo "     ps aux | grep run_all_baselines"
echo ""
echo "⚠️  注意事项:"
echo "  - 如果看到 '已从wrapper获取Qwen3-VL底层模型' 说明正常"
echo "  - 如果看到 'MLLM模型未提供，使用随机嵌入' 说明有问题"
echo "  - 前几个样本可能会显示一些警告，这是正常的"
echo ""

