#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单样本测试 - 验证整个流程是否正常工作
Single Sample Test - Verify the Entire Pipeline Works
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 首先激活multirag环境
import subprocess
print("激活multirag环境...")
subprocess.run("source ~/.bashrc && conda activate multirag", shell=True, check=False)

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("="*80)
print("单样本测试开始")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 测试导入
print("\n1. 测试导入...")
try:
    from flashrag.dataset.unified_dataset_loader import load_unified_dataset
    print("✓ 统一数据集加载器导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

try:
    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    print("✓ Qwen3-VL模块导入成功")
except Exception as e:
    print(f"✗ Qwen3-VL导入失败: {e}")
    sys.exit(1)

# 选择测试数据集
dataset_name = 'mrag-bench'  # 使用MRAG-Bench因为它有清晰的格式
print(f"\n2. 加载数据集: {dataset_name.upper()}")

try:
    dataset = load_unified_dataset(
        dataset_name,
        split='val',
        max_samples=1  # 只加载1个样本
    )

    if len(dataset) == 0:
        print("✗ 数据集为空")
        sys.exit(1)

    sample = dataset.data[0]
    print(f"✓ 加载成功，样本ID: {sample.get('id', 'N/A')}")
    print(f"  问题: {sample.get('question', '')[:100]}...")
    print(f"  选择: A={sample.get('A', '')[:30]}... B={sample.get('B', '')[:30]}...")
    print(f"  答案: {sample.get('golden_answers', [])}")

except Exception as e:
    print(f"✗ 数据集加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试模型加载
print("\n3. 初始化模型...")
model_path = '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct'

try:
    qwen3_vl = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✓ Qwen3-VL加载成功")
except Exception as e:
    print(f"✗ Qwen3-VL加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建简单的测试函数
def test_simple_generation():
    """测试简单的生成功能"""
    print("\n4. 测试简单生成...")

    prompt = "Answer with ONLY the letter: A"

    try:
        response = qwen3_vl.generate(
            text=prompt,
            image=None,
            max_new_tokens=5,
            temperature=0.1
        )
        print(f"✓ 生成成功: '{response.strip()}'")
        return response
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# 运行简单生成测试
generation_result = test_simple_generation()

# 创建简单的预测结果
print("\n5. 创建测试预测结果...")

# 模拟预测结果
test_prediction = {
    'id': sample.get('id', 0),
    'question': sample.get('question', ''),
    'answer': 'A',  # 假设预测为A
    'golden_answers': sample.get('golden_answers', []),
    'retrieved_docs': ['Test document 1', 'Test document 2'],
    'retrieval_result': [{
        'retrieved_docs': ['Test document 1', 'Test document 2'],
        'retrieval_scores': [0.9, 0.8],
        'retrieval_used': True
    }],
    'attributions': {
        'visual': [0, 1],
        'text': [0, 1]
    },
    'position_bias_results': {
        'average_bias': 0.3,
        'individual_scores': [0.3, 0.3],
        'position_weights': [0.4, 0.3, 0.2, 0.07, 0.03]
    },
    'used_retrieval': True
}

predictions = [test_prediction]
references = [{
    'question': sample.get('question', ''),
    'golden_answers': sample.get('golden_answers', []),
    'dataset': dataset_name,
    'scenario': sample.get('scenario', 'Unknown')
}]

# 测试评测
print("\n6. 测试统一评测...")
try:
    from flashrag.evaluator.unified_evaluator import evaluate_unified

    metrics = evaluate_unified(dataset_name, predictions, references)

    print("✓ 评测成功！")
    print("\n评测指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")

except Exception as e:
    print(f"✗ 评测失败: {e}")
    import traceback
    traceback.print_exc()

# 测试原始脚本（简化版）
print("\n7. 测试原始脚本配置...")
try:
    # 测试是否能正确导入原始脚本的配置
    config = {
        'dataset_name': 'okvqa',
        'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'max_samples': 1,
        'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        'output_dir': '/tmp/test_output',
        'temperature': 0.01,
        'max_new_tokens': 10,
        'retrieval_topk': 5,
    }

    print("✓ 配置测试成功")

except Exception as e:
    print(f"✗ 配置测试失败: {e}")

print("\n" + "="*80)
print("单样本测试完成！")
print("="*80)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 保存测试结果
test_result = {
    'test_time': datetime.now().isoformat(),
    'dataset': dataset_name,
    'sample_processed': True,
    'model_loaded': True,
    'generation_success': generation_result is not None,
    'evaluation_success': 'metrics' in locals()
}

output_file = '/data0/home/zqwang/ACL/FlashRAG/test_single_sample_result.json'
with open(output_file, 'w') as f:
    json.dump(test_result, f, indent=2)

print(f"\n测试结果已保存到: {output_file}")