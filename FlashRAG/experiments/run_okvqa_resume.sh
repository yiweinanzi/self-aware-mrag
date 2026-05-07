#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=okvqa_resume
#SBATCH --output=okvqa_resume_%j.out
#SBATCH --error=okvqa_resume_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 创建时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================" | tee okvqa_resume_${SLURM_JOB_ID}.out
echo "OK-VQA Baselines实验 (从MuRAG继续)" | tee -a okvqa_resume_${SLURM_JOB_ID}.out
echo "开始时间: $(date)" | tee -a okvqa_resume_${SLURM_JOB_ID}.out
echo "Job ID: ${SLURM_JOB_ID}" | tee -a okvqa_resume_${SLURM_JOB_ID}.out
echo "============================================" | tee -a okvqa_resume_${SLURM_JOB_ID}.out

# 运行实验，跳过Self-Aware-MRAG
echo -e "\n开始运行OK-VQA baselines实验..." | tee -a okvqa_resume_${SLURM_JOB_ID}.out
srun --gres=gpu:2 python run_okvqa_baselines.py --max-samples 5046 --save-detailed-results --save-sample-results --enable-complete-metrics --skip-method Self-Aware-MRAG 2>&1 | tee -a okvqa_resume_${SLURM_JOB_ID}.out

echo -e "\n实验完成时间: $(date)" | tee -a okvqa_resume_${SLURM_JOB_ID}.out