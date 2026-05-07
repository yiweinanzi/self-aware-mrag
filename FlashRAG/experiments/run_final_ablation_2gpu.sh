#!/bin/bash
#SBATCH -p 5090
#SBATCH --gres=gpu:2
#SBATCH -t 03:00:00
#SBATCH --output=final_ablation_2gpu.out
#SBATCH --error=final_ablation_2gpu.err

echo "=== 100样本2GPU消融实验 - 最终统一版本 ==="
echo "开始时间: $(date)"

# 激活环境
source ~/.bashrc
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 显示环境信息
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 检查GPU数量
python -c "
import torch
print(f'GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB')
"

# 进入实验目录
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 快速测试统一版本
echo "快速测试统一版本..."
python test_unified_version.py

echo ""
echo "开始运行100样本2GPU消融实验..."
echo "配置: --max-samples 100 --use-multi-gpu --num-gpus 2 --dataset okvqa --use-multimodal-retrieval"

# 运行最终实验
python run_unified_ablation.py \
    --max-samples 100 \
    --use-multi-gpu \
    --num-gpus 2 \
    --dataset okvqa \
    --use-multimodal-retrieval \
    --use-improved-estimator \
    --text-retrieval-weight 0.6 \
    --visual-retrieval-weight 0.4 \
    --uncertainty-threshold 0.43 \
    --text-weight 0.4 \
    --visual-weight 0.3 \
    --alignment-weight 0.3

echo ""
echo "实验完成时间: $(date)"
echo "=== 实验结束 ==="