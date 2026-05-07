#!/usr/bin/env python3
"""
测试MuRAG输出问题
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/data0/home/zqwang/ACL/models/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/data0/home/zqwang/ACL/models/huggingface/transformers'

def main():
    print("🔧 测试MuRAG输出问题")
    print("="*60)

    # 创建一个简化的测试环境
    sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

    # 模拟MuRAG的答案生成
    from flashrag.utils.vqa_evaluator import extract_okvqa_answer, VQAEvaluator
    evaluator = VQAEvaluator()

    # 测试数据
    test_cases = [
        {
            'question': 'What sport can you use this for?',
            'golden_answers': ['race', 'race', 'race'],
            'possible_outputs': [
                'race',
                'racing',
                'basketball',
                '',
                "I don't know",
                'This is used for racing',
                'The sport is racing'
            ]
        },
        {
            'question': 'Name the type of plant this is?',
            'golden_answers': ['rose', 'rose', 'rose'],
            'possible_outputs': [
                'rose',
                'flower',
                'tulip',
                '',
                'Cannot identify',
                'This is a rose'
            ]
        }
    ]

    print("\n📊 测试不同的输出情况:")
    for i, case in enumerate(test_cases):
        print(f"\n--- 案例 {i+1} ---")
        print(f"问题: {case['question']}")
        print(f"正确答案: {case['golden_answers']}")

        for output in case['possible_outputs']:
            # 应用MuRAG的处理流程
            processed = extract_okvqa_answer(output)

            # 检查是否正确
            is_correct = processed in case['golden_answers']

            print(f"\n  原始输出: '{output}'")
            print(f"  处理后: '{processed}'")
            print(f"  结果: {'✅ 正确' if is_correct else '❌ 错误'}")

    # 检查问题模式
    print("\n" + "="*60)
    print("🔍 问题分析:")

    # 可能的问题
    issues = [
        "1. 模型生成了错误答案（如用basketball回答race的问题）",
        "2. 模型生成了空答案",
        "3. 模型生成了'I don't know'类答案",
        "4. 模型生成了长句子，extract_okvqa_answer提取了错误部分",
        "5. VQA评估器的评估逻辑问题"
    ]

    for issue in issues:
        print(f"  {issue}")

    # 建议的修复方案
    print("\n💡 建议的修复方案:")
    fixes = [
        "1. 添加调试输出，查看实际生成的答案",
        "2. 改进提示词，明确要求生成1-3词的答案",
        "3. 在extract_okvqa_answer之前进行后处理",
        "4. 检查是否是数据集问题（golden_answers格式）",
        "5. 验证VQA评估器的实际工作方式"
    ]

    for fix in fixes:
        print(f"  {fix}")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()