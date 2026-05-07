#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=okvqa_fresh
#SBATCH --output=okvqa_fresh_%j.out
#SBATCH --error=okvqa_fresh_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================" | tee okvqa_fresh_${SLURM_JOB_ID}.out
echo "OK-VQA Baselines对比实验 - 完整运行" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out
echo "开始时间: $(date)" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out
echo "Job ID: ${SLURM_JOB_ID}" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out
echo "输出目录: ${PWD}/results_okvqa_baselines" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out
echo "============================================" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out

# 运行完整的OK-VQA baselines实验
echo -e "\n开始运行OK-VQA baselines实验..." | tee -a okvqa_fresh_${SLURM_JOB_ID}.out
srun --gres=gpu:2 python run_okvqa_baselines.py \
    --max-samples 5046 \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --output-dir results_okvqa_baselines \
    2>&1 | tee -a okvqa_fresh_${SLURM_JOB_ID}.out

echo -e "\n实验完成时间: $(date)" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out
echo "所有结果保存在: ${PWD}/results_okvqa_baselines" | tee -a okvqa_fresh_${SLURM_JOB_ID}.out