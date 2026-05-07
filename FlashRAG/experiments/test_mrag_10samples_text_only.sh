#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_test_10samples_text
#SBATCH --output=mrag_test_10samples_text_%j.out
#SBATCH --error=mrag_test_10samples_text_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MRAG-Bench 10样本测试（仅文本检索+Reranker）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 临时修改：禁用多模态，只测试文本检索+Reranker
echo -e "\n临时修改配置：使用纯文本检索（启用Reranker）"
sed -i 's/multimodal_retriever = init_retriever(CONFIG, use_multimodal=True)/multimodal_retriever = init_retriever(CONFIG, use_multimodal=False)/' run_all_baselines_MRAG.py

# 运行MRAG实验（10个样本）
echo -e "\n开始运行MRAG实验（10个样本）..."
srun --gres=gpu:2 python run_all_baselines_MRAG.py \
    --max_samples 10

# 恢复多模态配置
echo -e "\n恢复多模态配置..."
sed -i 's/multimodal_retriever = init_retriever(CONFIG, use_multimodal=False)/multimodal_retriever = init_retriever(CONFIG, use_multimodal=True)/' run_all_baselines_MRAG.py

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: ${PWD}/results_mrag_with_reranker"
echo "============================================"