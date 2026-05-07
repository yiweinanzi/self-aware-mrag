#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=okvqa_continue
#SBATCH --output=okvqa_continue_%j.out
#SBATCH --error=okvqa_continue_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================" | tee okvqa_continue_${SLURM_JOB_ID}.out
echo "OK-VQA Baselines实验 (从MuRAG继续)" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "开始时间: $(date)" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "Job ID: ${SLURM_JOB_ID}" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "输出目录: ${PWD}/results_okvqa_baselines" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "============================================" | tee -a okvqa_continue_${SLURM_JOB_ID}.out

# 记录Self-Aware-MRAG的完整指标
echo -e "\nSelf-Aware-MRAG 已完成指标:" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "- 准确率: 62.90% (3174/5046)" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "- 检索率: 92.8%" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "- 耗时: 46251.9秒 (9.17秒/样本)" | tee -a okvqa_continue_${SLURM_JOB_ID}.out

# 运行实验，从MuRAG开始，使用--methods参数指定方法
echo -e "\n开始运行剩余的baseline方法..." | tee -a okvqa_continue_${SLURM_JOB_ID}.out
srun --gres=gpu:2 python run_okvqa_baselines.py \
    --max-samples 5046 \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --methods MuRAG VisRAG ViDoRAG RagVL SAM-RAG "mR²AG" \
    2>&1 | tee -a okvqa_continue_${SLURM_JOB_ID}.out

echo -e "\n实验完成时间: $(date)" | tee -a okvqa_continue_${SLURM_JOB_ID}.out
echo "所有结果保存在: ${PWD}/results_okvqa_baselines" | tee -a okvqa_continue_${SLURM_JOB_ID}.out