"""
智能答案匹配器
处理词汇变体和相似性匹配
"""

import re
from typing import List, Set

class SmartAnswerMatcher:
    """智能答案匹配器"""

    def __init__(self):
        # 常见的词汇变换
        self.transformations = {
            # 名词变体
            'race': ['racing', 'racer', 'races', 'race car'],
            'racing': ['race', 'racer', 'races'],
            'basketball': ['basket ball', 'basketballs'],
            'football': ['foot ball', 'footballs'],
            'tennis': ['tennis ball', 'tennis racket'],
            'soccer': ['football', 'soccer ball'],
            'baseball': ['base ball', 'baseballs'],
            'red': ['reddish', 'red color', 'red colored'],
            'blue': ['bluish', 'blue color', 'blue colored'],
            'green': ['greenish', 'green color', 'green colored'],
            'yellow': ['yellowish', 'yellow color', 'yellow colored'],
            'black': ['blackish', 'black color', 'black colored'],
            'white': ['whitish', 'white color', 'white colored'],
            'dog': ['dogs', 'puppy', 'puppies'],
            'cat': ['cats', 'kitten', 'kitties'],
            'car': ['cars', 'automobile', 'auto', 'vehicle'],
            'bicycle': ['bike', 'bikes', 'bicycling'],
            'motorcycle': ['motorbike', 'motorcycles'],
            'flower': ['flowers', 'blossom', 'blossoms'],
            'tree': ['trees', 'treelike'],
            'house': ['houses', 'home', 'homes', 'building'],
        }

        # 构建反向映射
        self.reverse_map = {}
        for main_word, variants in self.transformations.items():
            for variant in variants:
                self.reverse_map[variant] = main_word

    def normalize_answer(self, answer: str) -> str:
        """标准化答案"""
        if not answer:
            return ""

        # 转小写并清理
        answer = str(answer).lower().strip()
        answer = re.sub(r'[^\w\s]', ' ', answer)  # 替换标点
        words = answer.split()

        # 移除常见的不重要词
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'it', 'this', 'that', 'for', 'with', 'by', 'sport', 'color'}
        words = [w for w in words if w not in stopwords]

        # 返回标准化后的答案（只取前3个词）
        if words:
            return ' '.join(words[:3])
        return ""

    def is_match(self, answer: str, golden_answers: List[str]) -> bool:
        """检查答案是否匹配"""
        if not answer:
            return False

        # 标准化答案
        answer = str(answer).lower().strip()

        # 检查每个黄金答案
        for golden in golden_answers[:3]:
            golden = str(golden).lower().strip()

            # 完全匹配
            if answer == golden:
                return True

            # 词根匹配
            answer_words = answer.split()
            golden_words = golden.split()

            # 特殊变换映射
            for a_word in answer_words:
                for g_word in golden_words:
                    # 使用预定义的变换
                    if a_word in self.transformations and g_word in self.transformations[a_word]:
                        return True
                    if g_word in self.transformations and a_word in self.transformations[g_word]:
                        return True

                    # 词根匹配
                    a_stem = self._get_stem(a_word)
                    g_stem = self._get_stem(g_word)
                    if a_stem == g_stem and len(a_stem) > 2:
                        return True

            # 部分匹配
            if answer in golden or golden in answer:
                return True

        return False

    def _get_stem(self, word: str) -> str:
        """获取词根"""
        if len(word) <= 3:
            return word

        # 移除常见后缀
        suffixes = ['ing', 'ed', 'es', 's', 'er', 'ly']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    def evaluate_batch(self, answers: List[str], golden_answers_list: List[List[str]]) -> float:
        """批量评估准确率"""
        if not answers or not golden_answers_list:
            return 0.0

        correct = 0
        for answer, golden in zip(answers, golden_answers_list):
            if self.is_match(answer, golden):
                correct += 1

        return correct / len(answers)

# 全局实例
_matcher = SmartAnswerMatcher()

def smart_answer_match(answer: str, golden_answers: List[str]) -> bool:
    """智能答案匹配"""
    return _matcher.is_match(answer, golden_answers)

def smart_evaluate_correctness(answer: str, golden_answers: List[str]) -> bool:
    """智能评估答案正确性"""
    return smart_answer_match(answer, golden_answers)

# 测试
if __name__ == "__main__":
    test_cases = [
        ("racing", ["race", "race", "race"]),
        ("race car", ["race", "race", "race"]),
        ("The sport is racing", ["race", "race", "race"]),
        ("basket ball", ["basketball", "basketball", "basketball"]),
        ("reddish", ["red", "red", "red"]),
        ("a bicycle", ["bicycle", "bicycle", "bicycle"]),
    ]

    print("测试智能答案匹配：")
    for answer, golden in test_cases:
        result = smart_answer_match(answer, golden)
        print(f"答案: '{answer}' vs 期望: {golden} -> 匹配: {result}")