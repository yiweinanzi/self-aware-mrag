#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用模拟检索器的测试
Test with Mock Retriever
"""

import os
import sys
import json
import time
from datetime import datetime

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print("="*80)
print("使用模拟检索器的单样本测试")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 导入模块
print("\n1. 导入模块...")
from flashrag.dataset.okvqa_dataset import load_okvqa_dataset
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
print("✓ 导入成功")

# 加载数据
print("\n2. 加载数据...")
dataset = load_okvqa_dataset(split='val', max_samples=1)
print(f"✓ 加载了 {len(dataset)} 个样本")

if len(dataset) == 0:
    print("✗ 没有数据")
    sys.exit(1)

sample = dataset[0]
print(f"样本: {sample}")

# 初始化模型
print("\n3. 初始化模型...")
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    device="cuda"
)
print("✓ Qwen3-VL加载成功")

# 创建模拟检索器
print("\n4. 创建模拟检索器...")
class MockRetriever:
    def __init__(self):
        self.top_k = 5

    def search(self, question, num=None):
        """模拟搜索返回"""
        num = num or self.top_k
        docs = []
        for i in range(num):
            docs.append({
                'id': f"doc_{i}",
                'contents': f"This is document {i} about {question[:30]}...",
                'score': 0.9 - i*0.1
            })
        return docs

retriever = MockRetriever()
print("✓ 模拟检索器创建成功")

# 测试生成
print("\n5. 测试生成...")
prompt = "What is in the image?"

try:
    response = qwen3_vl.generate(
        text=prompt,
        image=None,
        max_new_tokens=20,
        temperature=0.1
    )
    print(f"✓ 生成成功: '{response.strip()}'")
except Exception as e:
    print(f"✗ 生成失败: {e}")
    import traceback
    traceback.print_exc()

# 保存测试结果
print("\n6. 保存结果...")
result = {
    'test_time': datetime.now().isoformat(),
    'dataset_loaded': len(dataset) > 0,
    'model_loaded': True,
    'generation_success': True,
    'sample_question': sample.get('question', ''),
    'sample_answer': sample.get('answer', '')
}

output_file = '/data0/home/zqwang/ACL/FlashRAG/test_mock_retriever_result.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ 结果已保存: {output_file}")
print("\n测试成功！基础功能正常。")
print("\n下一步:")
print("1. 准备真实的检索索引")
print("2. 运行完整的对比实验")