"""
评估辅助函数
用于baseline方法正确计算correct字段
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import VQAEvaluator
from .answer_matcher import smart_answer_match

# 全局评估器实例
_evaluator = VQAEvaluator()

def evaluate_answer_correctness(answer: str, golden_answers: list) -> bool:
    """
    评估答案是否正确 - 使用智能匹配器处理词汇变体

    Args:
        answer: 生成的答案
        golden_answers: 正确答案列表

    Returns:
        bool: 是否正确
    """
    if not answer or not golden_answers:
        return False

    # 首先尝试智能匹配（处理词汇变体如racing -> race）
    if smart_answer_match(answer, golden_answers):
        return True

    # 如果智能匹配失败，使用VQA评估器
    result = _evaluator.calculate_vqa_accuracy(answer, golden_answers)
    return result.get('is_correct', False)

# 测试函数
def test_evaluation():
    """测试评估函数"""
    test_cases = [
        ('tennis', ['race', 'motocross', 'ride']),
        ('teddy bear', ['stuffed animal', 'teddy bear']),
        ('lunch', ['cloth', 'food', 'lunch']),
    ]

    print("测试评估函数:")
    for answer, golden in test_cases:
        correct = evaluate_answer_correctness(answer, golden)
        print(f"  '{answer}' vs {golden} -> {'✅' if correct else '❌'}")

if __name__ == "__main__":
    test_evaluation()