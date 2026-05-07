#!/bin/bash
# -*- coding: utf-8 -*-
# 环境设置和依赖安装脚本

set -e

echo "================================================================="
echo "环境设置和依赖安装"
echo "================================================================="
echo "开始时间: $(date)"
echo

# ============================================================================
# 检查系统环境
# ============================================================================

echo "🔍 检查系统环境..."

# 检查操作系统
OS=$(uname -s)
echo "操作系统: $OS"

# 检查Python版本
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "Python版本: $PYTHON_VERSION"
else
    echo "❌ Python3未安装"
    exit 1
fi

# 检查conda
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version | cut -d' ' -f2)
    echo "Conda版本: $CONDA_VERSION"
else
    echo "❌ Conda未安装，请先安装Miniconda或Anaconda"
    exit 1
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
    echo "GPU数量: $GPU_COUNT"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU未检测到"
fi

# ============================================================================
# 创建或激活环境
# ============================================================================

ENV_NAME="multirag"
echo "🔄 设置conda环境: $ENV_NAME"

# 检查环境是否存在
if conda env list | grep -q "^$ENV_NAME "; then
    echo "✅ 环境已存在: $ENV_NAME"
else
    echo "📝 创建新环境: $ENV_NAME"
    conda create -n $ENV_NAME python=3.9 -y
fi

# 激活环境
echo "🔄 激活环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "✅ 环境激活成功: $(which python)"

# ============================================================================
# 安装PyTorch（GPU版本）
# ============================================================================

echo "📦 安装PyTorch (GPU版本)..."

if python -c "import torch; print('PyTorch已安装:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch已安装"
else
    echo "🔄 安装PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # 验证CUDA支持
    python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
    print('GPU数量:', torch.cuda.device_count())
"
fi

# ============================================================================
# 安装FlashRAG依赖
# ============================================================================

echo "📦 安装FlashRAG依赖..."

cd /data0/home/zqwang/ACL/FlashRAG

# 检查requirements.txt
if [ -f "requirements.txt" ]; then
    echo "🔄 安装requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt不存在，安装基本依赖..."

    # 基础依赖
    pip install numpy pandas tqdm
    pip install matplotlib seaborn plotly
    pip install transformers
    pip install accelerate
    pip install datasets
    pip install sentence-transformers
    pip install faiss-cpu

    # 可视化依赖
    pip install pillow
    pip install opencv-python

    # 高级归因（可选）
    pip install captum pytorch-grad-cam
fi

# ============================================================================
# 安装评估指标依赖
# ============================================================================

echo "📦 安装评估指标依赖..."

pip install nltk  # 用于文本处理
pip install rouge-score  # 用于ROUGE指标
pip install sacrebleu  # 用于BLEU指标

# 下载NLTK数据
python -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print('下载NLTK数据...')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
"

# ============================================================================
# 检查FlashRAG安装
# ============================================================================

echo "🔍 检查FlashRAG安装..."

cd /data0/home/zqwang/ACL/FlashRAG

# 尝试导入FlashRAG
python -c "
import sys
sys.path.insert(0, '.')

try:
    from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator
    print('✅ FlashRAG评估指标导入成功')
except ImportError as e:
    print(f'⚠️  FlashRAG评估指标导入失败: {e}')

try:
    from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
    print('✅ OK-VQA数据集加载器导入成功')
except ImportError as e:
    print(f'⚠️  OK-VQA数据集加载器导入失败: {e}')

try:
    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    print('✅ Qwen3-VL模块导入成功')
except ImportError as e:
    print(f'⚠️  Qwen3-VL模块导入失败: {e}')

try:
    from flashrag.retriever import DenseRetriever
    print('✅ 检索器模块导入成功')
except ImportError as e:
    print(f'⚠️  检索器模块导入失败: {e}')
"

# ============================================================================
# 检查数据集
# ============================================================================

echo "🔍 检查数据集..."

DATA_DIR="/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA"

if [ -d "$DATA_DIR" ]; then
    echo "✅ VQA数据目录存在: $DATA_DIR"

    # 检查关键文件
    FILES=(
        "OpenEnded_mscoco_val2014_questions.json"
        "mscoco_val2014_annotations.json"
        "val2014"
    )

    for file in "${FILES[@]}"; do
        if [ -e "$DATA_DIR/$file" ]; then
            echo "✅ 数据文件存在: $file"
        else
            echo "❌ 数据文件缺失: $file"
        fi
    done
else
    echo "❌ VQA数据目录不存在: $DATA_DIR"
fi

# ============================================================================
# 检查模型
# ============================================================================

echo "🔍 检查模型..."

MODEL_DIR="/data0/home/zqwang/ACL/models"

if [ -d "$MODEL_DIR" ]; then
    echo "✅ 模型目录存在: $MODEL_DIR"

    # 列出可用模型
    for model_dir in "$MODEL_DIR"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            echo "   📁 $model_name"
        fi
    done
else
    echo "⚠️  模型目录不存在: $MODEL_DIR"
    echo "   请下载所需模型到该目录"
fi

# ============================================================================
# 测试运行
# ============================================================================

echo "🧪 运行环境测试..."

cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 运行快速测试
if python quick_test_run.py > /tmp/quick_test.log 2>&1; then
    echo "✅ 快速测试通过"
else
    echo "⚠️  快速测试失败，查看日志: /tmp/quick_test.log"
    tail -20 /tmp/quick_test.log
fi

# ============================================================================
# 生成环境报告
# ============================================================================

echo "📝 生成环境报告..."

REPORT_FILE="/data0/home/zqwang/ACL/FlashRAG/experiments/ENVIRONMENT_REPORT_$(date +%Y%m%d_%H%M%S).txt"

cat > $REPORT_FILE << EOF
环境设置报告
生成时间: $(date)
环境名称: $ENV_NAME
操作系统: $OS
Python版本: $PYTHON_VERSION
Conda版本: $CONDA_VERSION
GPU信息: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "未检测到GPU")

已安装的主要包:
$(pip list | grep -E "(torch|transformers|datasets|numpy|pandas|matplotlib)")

FlashRAG状态:
- 路径: /data0/home/zqwang/ACL/FlashRAG
- 可导入: $(python -c "import sys; sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG'); from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator; print('是')" 2>/dev/null || echo "否")

数据集状态:
- VQA数据: $([ -d "$DATA_DIR" ] && echo "存在" || echo "缺失")
- 模型目录: $([ -d "$MODEL_DIR" ] && echo "存在" || echo "缺失")

建议:
1. 确保所有测试通过后再运行完整实验
2. 对于大模型推理，建议使用多GPU
3. 定期检查GPU内存使用情况
4. 及时备份实验结果
EOF

echo "✅ 环境报告生成: $REPORT_FILE"

# ============================================================================
# 完成
# ============================================================================

echo
echo "================================================================="
echo "环境设置完成!"
echo "================================================================="
echo "完成时间: $(date)"
echo

echo "📋 下一步操作:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 运行测试: python /data0/home/zqwang/ACL/FlashRAG/experiments/quick_test_run.py"
echo "3. 运行实验: ./run_ablation_multigpu.py"

echo
echo "📁 重要文件:"
echo "   - 环境报告: $REPORT_FILE"
echo "   - 快速测试: /data0/home/zqwang/ACL/FlashRAG/experiments/quick_test_run.py"
echo "   - 多GPU实验: /data0/home/zqwang/ACL/FlashRAG/experiments/run_ablation_multigpu.py"

echo "================================================================="