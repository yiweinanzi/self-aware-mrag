#!/bin/bash
# 2GPU Clean版本 - 彻底解决template问题
# Author: Claude Code
# Date: 2025-11-25

set -e

# 环境设置
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 重置Python环境变量以避免template冲突
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建日志目录
LOG_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/logs"
mkdir -p "$LOG_DIR"

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ablation_2gpu_clean_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/ablation_2gpu_clean_${TIMESTAMP}.pid"

echo "================================================================================"
echo "2GPU Clean版本 - 100样本消融实验"
echo "开始时间: $(date)"
echo "GPU数量: 2 x RTX 5090 (64GB total)"
echo "策略: 环境隔离 + 原生加载"
echo "日志文件: $LOG_FILE"
echo "================================================================================"

# 启动完全重置环境的2GPU实验
echo "启动2GPU Clean版本实验..."
nohup srun --partition=5090 --gres=gpu:2 python -c "
import sys
import os
import gc
import torch
import warnings

# 完全重置transformers环境
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print('='*80)
print('环境重置和GPU检测')
print('='*80)

# 清理可能存在的模块缓存
for module in list(sys.modules.keys()):
    if 'transformers' in module or 'flashrag' in module:
        if module in sys.modules:
            del sys.modules[module]

# 重置CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f'✅ CUDA已重置，GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({memory:.1f}GB)')

# 完全隔离的实验代码
print('='*80)
print('开始加载Qwen3-VL模型...')
print('='*80)

try:
    # 使用transformers原生加载避免template问题
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    model_path = '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct'

    # 检测GPU并配置
    num_gpus = torch.cuda.device_count()
    print(f'✅ 检测到{num_gpus}个GPU')

    if num_gpus >= 2:
        print('✅ 使用2GPU模型并行')
        device_map = 'auto'  # 让transformers自动分配
    else:
        print(f'⚠️ 只有{num_gpus}个GPU')
        device_map = 'auto'

    # 8bit量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    print('正在加载���型...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    print('正在加载处理器...')
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print('✅ Qwen3-VL模型和处理器加载成功!')
    print(f'模型参数量: {sum(p.numel() for p in model.parameters())/1e9:.1f}B')

    # 显示内存使用
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f'GPU {i}内存: {allocated:.1f}GB已分配, {reserved:.1f}GB已保留')

    # 运行消融实验
    print('='*80)
    print('开始运行消融实验...')
    print('='*80)

    # 导入实验配置
    from run_real_model_ablation import CONFIG, AblationExperiment

    # 更新配置为2GPU模式
    CONFIG['model_path'] = model_path
    CONFIG['load_in_8bit'] = True
    CONFIG['max_samples'] = 100

    # 创建实验实例并设置全局模型
    experiment = AblationExperiment(CONFIG)
    experiment.model_processor = type('ModelProcessor', (), {
        'model': model,
        'processor': processor,
        'available': True
    })()

    # 强制设置全局实例
    AblationExperiment._global_model_processor = experiment.model_processor

    print('✅ 开始运行100样本消融实验...')
    experiment.run_experiment()

except Exception as e:
    print(f'❌ 实验失败: {e}')
    import traceback
    traceback.print_exc()
    raise e

" > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "实验已启动！"
echo "PID: $(cat $PID_FILE)"
echo "监控日志: tail -f $LOG_FILE"
echo ""
echo "使用以下命令监控进度:"
echo "  tail -f $LOG_FILE"
echo "  grep -E '✅|❌|准确率|样本|GPU' $LOG_FILE"
echo ""
echo "停止实验:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "================================================================================"