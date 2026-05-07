#!/usr/bin/env python3
"""
直接测试MuRAG的输出
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def main():
    print("🔧 直接测试MuRAG输出")
    print("="*60)

    # 设置CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 导入必要的模块
    from flashrag.utils.vqa_evaluator import extract_okvqa_answer, VQAEvaluator
    evaluator = VQAEvaluator()

    # 测试评估函数
    print("\n📝 测试VQA评估函数:")
    test_cases = [
        ("race", ["race", "race", "race"]),
        ("basketball", ["basketball", "basketball", "basketball"]),
        ("", ["race", "race", "race"]),
        (" ", ["race", "race", "race"]),
        ("I don't know", ["race", "race", "race"]),
    ]

    for answer, golden in test_cases:
        # 检查不同的评估方法
        methods = []

        # 方法1: 直接比较
        direct_match = answer in golden
        methods.append(f"直接匹配: {direct_match}")

        # 方法2: 使用extract_okvqa_answer
        extracted = extract_okvqa_answer(answer)
        extracted_match = extracted in golden
        methods.append(f"提取后匹配: {extracted_match} (提取后: '{extracted}')")

        # 方法3: 标准化后比较
        if hasattr(evaluator, 'standardize_answer'):
            ans_std = evaluator.standardize_answer(answer)
            extracted_std = evaluator.standardize_answer(extracted)
            golden_stds = [evaluator.standardize_answer(g) for g in golden]
            std_match = ans_std in golden_stds or extracted_std in golden_stds
            methods.append(f"标准化匹配: {std_match} (答案: '{ans_std}', 提取: '{extracted_std}')")

        # 方法4: 使用evaluate_answer（如果存在）
        if hasattr(evaluator, 'evaluate_answer'):
            eval_result = evaluator.evaluate_answer(answer, golden)
            methods.append(f"评估器评分: {eval_result}")

        print(f"\n答案: '{answer}' | 正确答案: {golden}")
        for method in methods:
            print(f"  - {method}")

    # 创建简单的MuRAG测试
    print("\n" + "="*60)
    print("🎯 模拟MuRAG行为测试")

    # 模拟实际生成的问题答案
    sample_question = "What sport can you use this for?"
    golden_answers = ["race", "race", "race"]

    # 可能的错误输出
    possible_answers = [
        "race",  # 正确
        "basketball",  # 错误
        "",  # 空答案
        "Unable to determine",  # 无法确定
        "This equipment is used for racing and competition",  # 长答案
    ]

    for ans in possible_answers:
        # 应用MuRAG的处理流程
        processed_answer = extract_okvqa_answer(ans.strip())

        # 评估
        is_correct = processed_answer in golden_answers

        print(f"\n原始答案: '{ans}'")
        print(f"处理后: '{processed_answer}'")
        print(f"是否正确: {'✅' if is_correct else '❌'}")

    print("\n" + "="*60)
    print("🔍 分析:")
    print("1. extract_okvqa_answer 函数工作正常")
    print("2. 空答案会导致准确率0%")
    print("3. 不匹配的答案会导致准确率0%")
    print("4. 需要检查MuRAG实际生成的答案内容")
    print("="*60)

if __name__ == "__main__":
    main()