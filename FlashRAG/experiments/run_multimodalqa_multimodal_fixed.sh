#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_multimodal_fixed
#SBATCH --output=multimodalqa_multimodal_fixed_%j.out
#SBATCH --error=multimodalqa_multimodal_fixed_%j.err
#SBATCH --dependency=afterok:505

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA多模态检索实验 - 10样本（使用MultiModalQA索引）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "依赖索引构建作业: 505"
echo "输出目录: ${PWD}/results_multimodalqa_multimodal_fixed"
echo "============================================"

# 等待索引构建完成
echo "等待索引构建完成..."
while [ ! -f "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index" ]; do
    echo "等待BGE索引..."
    sleep 30
done

while [ ! -f "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/clip_Flat.index" ]; do
    echo "等待CLIP索引..."
    sleep 30
done

echo "✅ 索引文件已就绪！"

# 创建输出目录
mkdir -p results_multimodalqa_multimodal_fixed

# 运行MultiModalQA 10样本测试（多模态检索）
echo -e "\n开始运行MultiModalQA 10样本测试（多模态检索）..."
# 使用 --ntasks=1 确保只有一个进程运行，避免重复日志
# 启用多模态检索
srun --ntasks=1 --gres=gpu:2 python run_all_baselines_MultimodalVQA_multimodal.py \
    --max_samples 10 \
    --split dev \
    --output_dir results_multimodalqa_multimodal_fixed \
    --use_multimodal_retrieval

echo -e "\n实验完成时间: $(date)"
echo "所有结果保存在: ${PWD}/results_multimodalqa_multimodal_fixed"