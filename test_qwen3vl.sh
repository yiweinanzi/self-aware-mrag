#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG

echo "=== PyTorch和GPU检查 ==="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU capability:', torch.cuda.get_device_capability(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
"

echo -e "\n=== 测试Qwen3-VL模型 ==="
python -c "
import sys
sys.path.insert(0, '.')

print('测试Qwen3-VL处理器...')
try:
    from flashrag.modules.qwen3_vl import Qwen3VLProcessor

    # 测试处理器初始化
    print('正在初始化Qwen3-VL处理器...')
    processor = Qwen3VLProcessor('/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct', device='cuda')
    print('✅ Qwen3-VL模型加载成功!')

    # 测试简单推理
    print('测试推理功能...')
    result = processor.generate('What color is the cat?', None)
    print(f'✅ 推理成功，答案: {result}')

    # 测试多模态推理
    result2 = processor.generate('How many animals are in the image?', None)
    print(f'✅ 多模态推理成功，答案: {result2}')

except Exception as e:
    import traceback
    print(f'❌ 错误: {e}')
    print('详细错误信息:')
    traceback.print_exc()
"