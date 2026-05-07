"""
改进的答案提取器
针对VQA任务优化，提取更准确的答案
"""

import re
from typing import List

class ImprovedAnswerExtractor:
    """改进的答案提取器"""

    def __init__(self):
        # 常见的答案引导词
        self.answer_prefixes = [
            "answer is", "answer:", "the answer is",
            "it is", "it's", "it is a",
            "this is", "this is a",
            "that is", "that is a",
            "the sport is", "the plant is", "the object is",
            "name is", "called"
        ]

        # 需要移除的词
        self.stopwords = {
            'a', 'an', 'the', 'it', 'this', 'that', 'is', 'are', 'was', 'were',
            'to', 'for', 'in', 'on', 'at', 'by', 'with', 'from', 'used'
        }

    def extract_answer(self, text: str) -> str:
        """提取答案"""
        if not text:
            return ""

        # 转换为小写
        text = text.lower().strip()

        # 移除句号
        if text.endswith('.'):
            text = text[:-1]

        # 尝试模式1：寻找"answer is X"模式
        for prefix in self.answer_prefixes:
            if prefix in text:
                parts = text.split(prefix)
                if len(parts) > 1:
                    answer_part = parts[-1].strip()
                    # 提取前1-3个词，并过滤停用词
                    words = answer_part.split()
                    if words:
                        # 过滤停用词
                        filtered_words = [w for w in words if w.lower() not in self.stopwords]
                        # 确保不是空的
                        if filtered_words:
                            return ' '.join(filtered_words[:3])
                        else:
                            # 如果过滤后为空，返回前3个词
                            return ' '.join(words[:3])

        # 尝试模式2：如果句子很短（<5词），直接返回
        words = text.split()
        if len(words) <= 5:
            # 转换为小写再过滤停用词
            filtered = [w for w in words if w.lower() not in self.stopwords]
            # 进一步过滤单个字母的词
            filtered = [w for w in filtered if len(w) > 1 or w.lower() in ['i', 'ok', 'no', 'yes']]
            # 确保没有"a"
            filtered = [w for w in filtered if w.lower() != 'a']
            if filtered:
                return ' '.join(filtered[:3])

        # 尝试模式3：寻找关键词
        keywords = self._extract_keywords(text)
        if keywords:
            return ' '.join(keywords[:3])

        # 默认：返回前3个词
        return ' '.join(words[:3])

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        # 移除常见的前缀
        prefixes = ["the answer is", "answer is", "it is", "this is", "that is"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        # 分词并过滤
        words = text.split()
        keywords = [w for w in words if w.lower() not in self.stopwords and len(w) > 1]

        # 进一步清理：移除单个字母的词（如"a"）
        keywords = [w for w in keywords if len(w) > 1 or w.lower() in ['i', 'ok', 'no', 'yes']]

        return keywords

# 全局实例
_extractor = ImprovedAnswerExtractor()

def extract_answer_smart(text: str) -> str:
    """智能提取答案"""
    return _extractor.extract_answer(text)

# 测试函数
def test_extractor():
    """测试答案提取器"""
    test_cases = [
        ("The sport is racing", "racing"),
        ("This is a rose", "rose"),
        ("Answer: basketball", "basketball"),
        ("It is a basketball", "basketball"),
        ("race", "race"),
        ("Used for racing", "racing"),
        ("I think it's a basketball", "basketball"),
    ]

    print("测试答案提取器:")
    for text, expected in test_cases:
        result = extract_answer_smart(text)
        print(f"'{text}' -> '{result}' (期望: '{expected}') {'✅' if result == expected else '❌'}")

if __name__ == "__main__":
    test_extractor()