#!/usr/bin/env python3
"""
调试评估系统
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import VQAEvaluator, extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart

# 创建评估器
evaluator = VQAEvaluator()

# 测试案例
test_cases = [
    {
        'answer': 'tennis',
        'golden_answers': ['race', 'race', 'race', 'race', 'race', 'motocross', 'motocross', 'ride', 'ride'],
        'method': 'MuRAG'
    },
    {
        'answer': 'teddy bear',
        'golden_answers': ['stuffed animal', 'stuffed animal', 'stuffed animal', 'stuffed animal', 'teddy bear', 'teddy bear'],
        'method': 'MuRAG'
    },
    {
        'answer': 'racket sports',
        'golden_answers': ['race', 'race', 'race', 'motocross', 'motocross', 'ride', 'ride'],
        'method': 'Self-Aware-MRAG'
    },
    {
        'answer': 'none',
        'golden_answers': ['race', 'race', 'race', 'motocross', 'ride'],
        'method': 'VisRAG'
    },
    {
        'answer': 'houseplant',
        'golden_answers': ['vine', 'climb', 'ficus', 'ivy'],
        'method': 'VisRAG'
    }
]

print("🔍 调试评估系统")
print("="*60)

for case in test_cases:
    answer = case['answer']
    golden = case['golden_answers']
    method = case['method']

    print(f"\n方法: {method}")
    print(f"生成答案: '{answer}'")
    print(f"正确答案: {golden}")

    # 使用不同的评估方法
    print("\n评估方法对比:")

    # 方法1: 直接匹配
    direct_match = answer in golden
    print(f"  直接匹配: {'✅' if direct_match else '❌'}")

    # 方法2: 使用extract_okvqa_answer
    extracted_old = extract_okvqa_answer(answer)
    old_match = extracted_old in golden
    print(f"  extract_okvqa_answer: '{extracted_old}' -> {'✅' if old_match else '❌'}")

    # 方法3: 使用改进的提取器
    extracted_new = extract_answer_smart(answer)
    new_match = extracted_new in golden
    print(f"  extract_answer_smart: '{extracted_new}' -> {'✅' if new_match else '❌'}")

    # 方法4: 使用VQAEvaluator
    if hasattr(evaluator, 'evaluate_answer'):
        evaluator_result = evaluator.evaluate_answer(answer, golden)
        print(f"  VQAEvaluator.evaluate_answer: {evaluator_result}")
    elif hasattr(evaluator, 'evaluate_okvqa'):
        evaluator_result = evaluator.evaluate_okvqa(answer, golden)
        print(f"  VQAEvaluator.evaluate_okvqa: {evaluator_result}")
    else:
        print("  ⚠️ VQAEvaluator没有evaluate_answer或evaluate_okvqa方法")

# 检查标准化函数
print("\n" + "="*60)
print("📝 检查标准化函数")

if hasattr(evaluator, 'standardize_answer'):
    test_answers = ['teddy bear', 'Tennis', 'TENNIS', 'tennis']
    print("\n标准化测试:")
    for ans in test_answers:
        std = evaluator.standardize_answer(ans)
        print(f"  '{ans}' -> '{std}'")
else:
    print("⚠️ VQAEvaluator没有standardize_answer方法")

# 查看VQAEvaluator的方法
print("\nVQAEvaluator可用方法:")
methods = [m for m in dir(evaluator) if not m.startswith('_')]
for method in methods[:20]:  # 只显示前20个
    print(f"  - {method}")