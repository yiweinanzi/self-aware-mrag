#!/bin/bash

#SBATCH --job-name=full_baselines_comparison
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/comparison_%j.out
#SBATCH --error=logs/comparison_%j.err

# 创建日志目录
mkdir -p logs

echo "========================================"
echo "完整基线对比实验"
echo "========================================"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "========================================"

# 初始化conda并激活环境
eval "$(conda shell.bash hook)"
conda activate multirag

echo "环境信息:"
echo "  Python: $(python --version)"
echo "  Conda环境: $CONDA_DEFAULT_ENV"
echo "  GPU信息:"
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/data0/home/zqwang/ACL/FlashRAG:$PYTHONPATH"

# 第一步：运行原始的单数据集实验
echo ""
echo "========================================"
echo "步骤1: 运行OK-VQA单数据集实验"
echo "========================================"

python experiments/run_all_baselines_100samples.py 2>&1 | tee logs/step1_okvqa.log

# 检查第一步是否成功
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 步骤1失败，查看日志 logs/step1_okvqa.log"
    exit 1
fi

echo "✅ 步骤1完成"

# 第二步：运行多数据集对比（少量样本）
echo ""
echo "========================================"
echo "步骤2: 运行多数据集对比（50样本）"
echo "========================================"

python experiments/run_all_baselines_multi_datasets.py \
    --datasets okvqa mrag-bench \
    --methods Self-Aware-MRAG ViDoRAG \
    --max-samples 50 \
    --gpu-id 0 2>&1 | tee logs/step2_multi_datasets.log

# 检查第二步是否成功
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 步骤2失败，查看日志 logs/step2_multi_datasets.log"
    exit 1
fi

echo "✅ 步骤2完成"

# 第三步：运行完整的7种方��对比（如果时间允许）
echo ""
echo "========================================"
echo "步骤3: 运行完整方法对比（10样本）"
echo "========================================"

python experiments/run_all_baselines_multi_datasets.py \
    --datasets okvqa \
    --methods Self-Aware-MRAG SAM-RAG mR2AG VisRAG ViDoRAG RagVL MuRAG \
    --max-samples 10 \
    --gpu-id 0 2>&1 | tee logs/step3_all_methods.log

echo "✅ 步骤3完成"

# 整理结果
echo ""
echo "========================================"
echo "整理实验结果"
echo "========================================"

# 创建结果目录
RESULT_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/results_final_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

# 复制日志和结果文件
echo "结果目录: $RESULT_DIR"
cp -r logs $RESULT_DIR/
find experiments/results_* -maxdepth 1 -name "*.json" -exec cp {} $RESULT_DIR/ \;
find experiments/results_* -maxdepth 1 -name "*.md" -exec cp {} $RESULT_DIR/ \;

# 生成总结报告
cat > $RESULT_DIR/experiment_summary.md << EOF
# 基线对比实验总结

**实验时间**: $(date)
**作业ID**: $SLURM_JOB_ID
**节点**: $SLURM_NODELIST

## 实验步骤

1. ✅ OK-VQA单数据集实验 - 100样本
   - 日志: step1_okvqa.log
   - 7种方法对比

2. ✅ 多数据集对比 - 50样本
   - 日志: step2_multi_datasets.log
   - 数据集: OK-VQA, MRAG-Bench
   - 方法: Self-Aware-MRAG, ViDoRAG

3. ✅ 完整方法对比 - 10样本
   - 日志: step3_all_methods.log
   - 数据集: OK-VQA
   - 方法: 全部7种

## 结果文件

- JSON结果文件: *.json
- 日志文件: logs/*.log
- 配置信息: 包含在各个JSON文件中

## 注意事项

- 确保ViDoRAG已正确替换REVEAL
- 7个核心指标的计算需要检查
- GPU内存使用情况请查看日志

EOF

echo ""
echo "========================================"
echo "实验完成！"
echo "========================================"
echo "结果保存在: $RESULT_DIR"
echo "结束时间: $(date)"
echo ""