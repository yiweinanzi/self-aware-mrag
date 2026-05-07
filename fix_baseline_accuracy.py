#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复baseline方法准确率为0%的问题

主要问题：
1. 答案太长，需要提取核心信息
2. 需要确保答案格式符合VQA评估要求
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import extract_okvqa_answer

def create_better_answer_generator():
    """创建更好的答案生成器，确保答案简洁且包含核心信息"""

    prompts = {
        'direct': """Answer the question with 1-3 words.

Question: {question}

Answer:""",

        'retrieval': """Based ONLY on the given evidence, answer with 1-3 words.

Evidence: {context}

Question: {question}

Answer:""",

        'multichoice': """Choose the correct letter.

{context}

Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Answer (letter only):"""
    }

    return prompts

def fix_murag_answer_generation():
    """修复MuRAG的答案生成"""

    print("\n=== MuRAG修复方案 ===")
    print("1. 使用extract_okvqa_answer提取核心答案")
    print("2. 改进prompt，要求生成1-3个单词的短答案")
    print("3. 对于选择题，返回选项字母而不是完整文本")

    example_fix = """
原始prompt:
"Based ONLY on this evidence document, answer the question.
Evidence: {doc}
Question: {question}
Answer:"

改进prompt:
"Answer with 1-3 words only.
Question: {question}
Evidence: {doc}
Answer:"
    """

    print(example_fix)

def fix_visrag_answer_generation():
    """修复VisRAG的答案生成"""

    print("\n=== VisRAG修复方案 ===")
    print("1. 确保最终答案经过extract_okvqa_answer处理")
    print("2. 使用更直接的prompt避免冗长答案")

def fix_vidorag_retrieval():
    """修复ViDoRAG的检索问题"""

    print("\n=== ViDoRAG检索修复方案 ===")
    print("1. 检查retriever初始化是否正确")
    print("2. 确保search方法被正确调用")
    print("3. 添加调试日志以追踪检索过程")

def main():
    """总结所有修复方案"""

    print("="*70)
    print("Baseline方法准确率修复方案")
    print("="*70)

    # 测试答案提取
    print("\n测试答案提取功能：")
    test_answers = [
        "This is a racing sport that involves cars competing against each other",
        "The answer is race car competition",
        "Baseball is played with a bat and ball",
        "You can play tennis with this equipment"
    ]

    for ans in test_answers:
        extracted = extract_okvqa_answer(ans)
        print(f"原答案: '{ans[:50]}...'")
        print(f"提取后: '{extracted}'")
        print()

    # 修复方案
    fix_murag_answer_generation()
    fix_visrag_answer_generation()
    fix_vidorag_retrieval()

    print("\n" + "="*70)
    print("总结:")
    print("1. 所有方法都应该使用extract_okvqa_answer确保答案简洁")
    print("2. prompt应明确要求生成1-3个单词的答案")
    print("3. ViDoRAG需要修复检索器初始化问题")
    print("4. 评估时使用comprehensive_evaluator计算完整指标")
    print("="*70)

if __name__ == "__main__":
    main()