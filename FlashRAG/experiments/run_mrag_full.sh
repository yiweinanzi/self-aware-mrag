#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_full
#SBATCH --output=mrag_full_%j.out
#SBATCH --error=mrag_full_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================" | tee mrag_full_${SLURM_JOB_ID}.out
echo "MRAG-Bench Baselines对比实验 - 全样本" | tee -a mrag_full_${SLURM_JOB_ID}.out
echo "开始时间: $(date)" | tee -a mrag_full_${SLURM_JOB_ID}.out
echo "Job ID: ${SLURM_JOB_ID}" | tee -a mrag_full_${SLURM_JOB_ID}.out
echo "输出目录: ${PWD}/results_mragbench_baseline" | tee -a mrag_full_${SLURM_JOB_ID}.out
echo "============================================" | tee -a mrag_full_${SLURM_JOB_ID}.out

# 运行MRAG全样本实验（不指定max_samples以使用全部数据）
echo -e "\n开始运行MRAG全样本实验..." | tee -a mrag_full_${SLURM_JOB_ID}.out
srun --gres=gpu:2 python run_all_baselines_MRAG.py 2>&1 | tee -a mrag_full_${SLURM_JOB_ID}.out

echo -e "\n实验完成时间: $(date)" | tee -a mrag_full_${SLURM_JOB_ID}.out
echo "所有结果保存在: ${PWD}/results_mragbench_baseline" | tee -a mrag_full_${SLURM_JOB_ID}.out