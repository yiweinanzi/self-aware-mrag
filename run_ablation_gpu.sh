#!/bin/bash
#SBATCH -p 5090
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --output=ablation_100samples.out
#SBATCH --error=ablation_100samples.err

echo "=== 激活环境和运行消融实验 ==="
source ~/.bashrc
source /data0/home/zqwang/miniconda3/bin/activate multirag

echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "Installing missing dependencies..."
pip install termcolor

cd /data0/home/zqwang/ACL/FlashRAG/experiments
echo "Running ablation study..."
python run_fixed_ablation_100samples.py

echo "=== 实验完成 ==="