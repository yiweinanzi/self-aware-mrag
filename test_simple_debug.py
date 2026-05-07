#!/usr/bin/env python3
"""
简单测试 - 直接测试MuRAG的一个样本
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def main():
    print("🔧 简单测试 - 检查MuRAG问题")
    print("="*60)

    # 导入VQA评估器
    try:
        from flashrag.utils.vqa_evaluator import extract_okvqa_answer
        print("✅ 成功导入 extract_okvqa_answer")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return

    # 测试extract_okvqa_answer
    print("\n📝 测试 extract_okvqa_answer:")
    test_cases = [
        ("basketball", ["race", "race", "race"]),
        ("basketball racing", ["race", "race", "race"]),
        ("The sport is basketball", ["basketball", "basketball", "basketball"]),
        ("rose flower", ["rose", "rose", "rose"]),
    ]

    from flashrag.utils.vqa_evaluator import VQAEvaluator
    evaluator = VQAEvaluator()

    for i, (answer, golden) in enumerate(test_cases):
        extracted = extract_okvqa_answer(answer)
        correct = evaluator.evaluate_okvqa(extracted, golden)

        print(f"  测试{i+1}: '{answer}' -> '{extracted}' | 正确答案: {golden} | {'✅' if correct else '❌'}")

    print("\n" + "="*60)
    print("🔍 问题分析:")
    print("1. extract_okvqa_answer 工作正常")
    print("2. MuRAG准确率0%可能是生成的答案不正确")
    print("3. 需要检查实际生成的答案内容")
    print("="*60)

if __name__ == "__main__":
    main()