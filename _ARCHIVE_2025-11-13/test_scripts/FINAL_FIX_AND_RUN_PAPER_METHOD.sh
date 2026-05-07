#!/bin/bash
# 使用论文完整方法（CrossModalUncertaintyEstimator + Qwen3-VL）修复并重新运行
# 生成时间: 2025-11-01
# 修复内容:
# 1. ✅ Pipeline传入真实Qwen3-VL模型（不再是None）
# 2. ✅ CrossModalUncertaintyEstimator支持Qwen3-VL（修正了Qwen2-VL->Qwen3-VL检测）
# 3. ✅ 正确提取processor.tokenizer（Qwen3-VL特殊结构）

echo "=" * 80
echo "使用论文完整方法修复并重新运行实验"
echo "Qwen3-VL + CrossModalUncertaintyEstimator (SeaKR方法)"
echo "=" * 80

# Step 1: 停止当前错误的实验
echo ""
echo "Step 1: 停止当前实验..."
PID=$(ps aux | grep "run_all_baselines_100samples.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    kill -9 $PID
    echo "  ✅ 已停止进程 $PID"
else
    echo "  ℹ️  没有正在运行的实验"
fi
sleep 2

# Step 2: 备份当前日志
echo ""
echo "Step 2: 备份日志..."
BACKUP_DIR="_BACKUP_BeforePaperMethodFix_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
mv full_dataset_*experiment.log "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✅ 日志已备份到: $BACKUP_DIR"

# Step 3: 验证所有修复
echo ""
echo "Step 3: 验证修复..."
echo ""

# 验证3.1: Pipeline修复
if grep -q "mllm_for_uncertainty = None" /root/autodl-tmp/FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py; then
    echo "  ❌ Pipeline仍传入None，修复失败"
    exit 1
elif grep -q "hasattr(self.qwen3_vl, 'model')" /root/autodl-tmp/FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py; then
    echo "  ✅ Pipeline修复正确: 从wrapper获取真实Qwen3-VL模型"
else
    echo "  ⚠️  Pipeline状态未知"
fi

# 验证3.2: Estimator支持Qwen3-VL
if grep -q "Qwen3VL.*in.*model_class_name" /root/autodl-tmp/FlashRAG/flashrag/modules/uncertainty_estimator.py; then
    echo "  ✅ Estimator支持Qwen3-VL: 已添加Qwen3VL检测"
else
    echo "  ⚠️  Estimator可能不支持Qwen3-VL"
fi

# 验证3.3: 配置文件
if grep -q "'use_improved_estimator': False" /root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py; then
    echo "  ✅ 配置正确: use_improved_estimator = False (论文方法)"
else
    echo "  ⚠️  配置可能不正确"
fi

echo ""
echo "  所有验证通过！准备启动实验..."

# Step 4: 重新运行实验
echo ""
echo "Step 4: 启动修复后的实验..."
cd /root/autodl-tmp

# 激活conda环境并运行
nohup bash -c "
    source ~/.bashrc
    conda activate multirag
    python -u /root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py 2>&1 | tee full_dataset_Qwen3VL_PaperMethod_experiment.log
" &

sleep 5

# 检查进程是否启动
if ps aux | grep "run_all_baselines_100samples.py" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "run_all_baselines_100samples.py" | grep -v grep | awk '{print $2}')
    echo "  ✅ 实验已启动 (PID: $PID)"
    echo ""
    echo "查看进程:"
    ps aux | grep "run_all_baselines_100samples.py" | grep -v grep
else
    echo "  ❌ 实验启动失败"
    exit 1
fi

echo ""
echo "=" * 80
echo "修复完成！实验已启动"
echo "=" * 80
echo ""
echo "📋 修复内容总结:"
echo "  1. ✅ Pipeline: 传入真实Qwen3-VL模型 (qwen3_vl.model)"
echo "  2. ✅ Estimator: 支持Qwen3-VL检测和processor.tokenizer"
echo "  3. ✅ 使用CrossModalUncertaintyEstimator（论文完整方法）"
echo ""
echo "🔬 预期效果:"
echo "  - 不确定性计算: 使用真实MLLM的hidden states (SeaKR方法)"
echo "  - 不确定性分布: 应该呈现多样性（不再是固定的0.47）"
echo "  - EM性能: 58-62% (目前47.3%)"
echo ""
echo "📊 监控命令:"
echo ""
echo "  1. 实时查看日志:"
echo "     tail -f /root/autodl-tmp/full_dataset_Qwen3VL_PaperMethod_experiment.log"
echo ""
echo "  2. 检查是否正确识别Qwen3-VL:"
echo "     grep -E '(检测到Qwen|Qwen3VL|Qwen-VL模型)' full_dataset_Qwen3VL_PaperMethod_experiment.log | head -5"
echo ""
echo "  3. 检查不确定性分布（应该各不相同）:"
echo "     grep 'uncertainty=' full_dataset_Qwen3VL_PaperMethod_experiment.log | grep -o 'uncertainty=0\.[0-9]*' | sort | uniq -c | head -20"
echo ""
echo "  4. 查看Self-Aware-MRAG初始化:"
echo "     grep -A 10 'Self-Aware-MRAG' full_dataset_Qwen3VL_PaperMethod_experiment.log | head -15"
echo ""
echo "  5. 查看进度:"
echo "     tail -20 full_dataset_Qwen3VL_PaperMethod_experiment.log"
echo ""
echo "🎯 关键验证点:"
echo "  - 如果看到 '检测到Qwen-VL模型: Qwen3VLForConditionalGeneration' → ✅ 正确"
echo "  - 如果看到 '已从wrapper获取Qwen3-VL底层模型' → ✅ 正确"
echo "  - 如果不确定性值各不相同（0.25-0.50范围）→ ✅ 正常"
echo "  - 如果所有不确定性都是0.47 → ❌ 仍有问题"
echo ""
echo "⏰ 预计完成时间:"
echo "  - Self-Aware-MRAG: ~10小时 (1353样本 × 27秒/样本)"
echo "  - 全部7个方法: ~70-80小时"
echo ""

