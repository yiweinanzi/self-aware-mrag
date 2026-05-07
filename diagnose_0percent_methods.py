#!/usr/bin/env python3
"""
诊断4个0%准确率方法的具体问题
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
sys.path.insert(0, '/data0/home/zqwang/ACL')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from flashrag.utils.qwen3_vl import Qwen3VLWrapper
from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline
from experiments.baselines.ragvl_enhanced import create_ragvl_enhanced
from experiments.baselines.samrag_adapted import create_samrag_adapted
from experiments.baselines.mr2ag_enhanced import create_mr2ag_enhanced
from flashrag.retriever.retriever import DenseRetriever
from flashrag.utils.vqa_evaluator import VQAEvaluator

# 创建测试样本
test_samples = [
    {
        'question': 'What sport can you use this for?',
        'image': 'test_image_placeholder',  # 实际测试时需要真实图像
        'golden_answers': ['race', 'race', 'race', 'motocross']
    }
]

print("="*60)
print("诊断4个0%准确率方法")
print("="*60)

# 初始化模型（仅测试，不实际加载）
print("\n1. 检查方法创建...")
methods = {
    'ViDoRAG': lambda qwen3vl, retriever, config: create_vidorag_pipeline(qwen3vl, retriever, config),
    'RagVL': lambda qwen3vl, retriever, config: create_ragvl_enhanced(qwen3vl, retriever, config),
    'SAM-RAG': lambda qwen3vl, retriever, config: create_samrag_adapted(qwen3vl, retriever, config),
    'mR²AG': lambda qwen3vl, retriever, config: create_mr2ag_enhanced(qwen3vl, retriever, config)
}

config = {'retrieval_topk': 5, 'max_new_tokens': 20}

for method_name, create_func in methods.items():
    print(f"\n{method_name}:")
    try:
        # 模拟创建（不需要真实模型）
        class MockQwen3VL:
            def generate(self, text, images=None, **kwargs):
                if 'sport' in text.lower():
                    return "racing"
                elif 'knowledge' in text.lower():
                    return "NEED"
                else:
                    return "unknown"
            def __call__(self):
                return self

        class MockRetriever:
            def search(self, query, num=5):
                # 返回5个假文档
                docs = [
                    f"Document about {query}",
                    f"Information related to {query}",
                    f"Facts about {query}",
                    f"Details concerning {query}",
                    f"Data on {query}"
                ]
                scores = [0.9, 0.8, 0.7, 0.6, 0.5]
                return docs, scores

        mock_qwen3vl = MockQwen3VL()
        mock_retriever = MockRetriever()

        # 创建pipeline
        pipeline = create_func(mock_qwen3vl, mock_retriever, config)
        print(f"  ✅ 创建成功")

        # 测试run_single
        print(f"  测试 run_single...")
        sample = {
            'question': 'What sport can you use this for?',
            'golden_answers': ['race', 'motocross']
        }

        result = pipeline.run_single(sample)
        print(f"  答案: '{result.get('answer', '')}'")
        print(f"  检索成功: {result.get('retrieved', False)}")
        print(f"  Correct字段: {result.get('correct', False)}")

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("诊断完成")
print("="*60)

print("\n可能的问题:")
print("1. SAM-RAG - 可能返回空答案")
print("2. mR²AG - 检索反射可能返回NO")
print("3. ViDoRAG/RagVL - 答案可能不匹配期望")
print("\n建议:")
print("- 检查实际生成的答案内容")
print("- 验证golden_answers是否正确传递")
print("- 确认correct字段计算逻辑")