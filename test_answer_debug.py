#!/usr/bin/env python3
"""
调试答案提取器
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from experiments.baselines.answer_extractor import ImprovedAnswerExtractor

def test_debug():
    extractor = ImprovedAnswerExtractor()

    # 测试用例
    test_text = "This is a rose"

    print(f"输入: '{test_text}'")
    print(f"停用词: {extractor.stopwords}")

    # 手动测试过滤
    words = test_text.split()
    print(f"分词后: {words}")

    # 转换为小写再过滤
    filtered = [w for w in words if w.lower() not in extractor.stopwords]
    print(f"过滤停用词后: {filtered}")

    # 进一步过滤
    filtered = [w for w in filtered if len(w) > 1 or w in ['i', 'ok', 'no', 'yes']]
    print(f"过滤单个字母后: {filtered}")

    # 确保没有"a"
    filtered = [w for w in filtered if w != 'a']
    print(f"移除a后: {filtered}")

    # 提取答案
    answer = extractor.extract_answer(test_text)
    print(f"最终提取: '{answer}'")

    # 测试其他输入
    test_cases = [
        "The sport is racing",
        "Answer: basketball",
        "It is a basketball",
    ]

    for text in test_cases:
        print(f"\n输入: '{text}'")
        result = extractor.extract_answer(text)
        print(f"输出: '{result}'")

if __name__ == "__main__":
    test_debug()