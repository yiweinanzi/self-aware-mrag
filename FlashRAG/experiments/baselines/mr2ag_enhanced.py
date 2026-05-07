#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mR²AG Fixed - 修复版本

主要修复：
1. Retrieval-Reflection更宽松
2. 改进答案生成prompt
3. Relevance-Reflection更宽松
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from typing import Dict, Any, List, Tuple
import warnings
from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart


class MR2AGFixed:
    """
    mR²AG Fixed - 修复低准确率问题
    """

    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}

        self.top_k = self.config.get('retrieval_topk', 5)
        self.temperature = self.config.get('temperature', 0.1)  # 提高温度获得更好答案

        # 段落切分参数
        self.para_min_len = self.config.get('para_min_len', 30)  # 降低最小长度
        self.para_max_len = self.config.get('para_max_len', 200)  # 增加最大长度

    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample['question']
        image = sample.get('image')

        # === Step 1: Retrieval-Reflection (更宽松) ===
        need_retrieval = self._retrieval_reflection(question, image)

        if not need_retrieval:
            # 直接回答，但使用更好的prompt
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],
                'retrieved': False,
                'retrieval_decision': 'No Retrieval',
                'total_paragraphs': 0,
                'relevant_paragraphs': 0,
                'golden_answers': sample.get('golden_answers', []),
                'correct': False
            }

        # === Step 2: 检索条目 ===
        entries = self._retrieve_documents(question)

        if not entries:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],
                'retrieved': False,
                'retrieval_decision': 'Retrieval (no docs)',
                'total_paragraphs': 0,
                'relevant_paragraphs': 0,
                'golden_answers': sample.get('golden_answers', []),
                'correct': False
            }

        # === Step 3: 段落级处理 ===
        candidates = []
        total_paras = 0

        for entry_idx, entry in enumerate(entries):
            paragraphs = self._split_into_paragraphs(entry)
            total_paras += len(paragraphs)

            for para in paragraphs:
                # Relevance-Reflection (更宽松)
                is_relevant, rel_score = self._relevance_reflection(question, para)

                if is_relevant:
                    # 生成答案（带分数）
                    answer, ans_score = self._generate_with_paragraph(sample, para)

                    # 层级打分
                    ret_score = 0.9 ** entry_idx
                    total_score = ret_score * rel_score * ans_score

                    candidates.append({
                        'answer': answer,
                        'score': total_score,
                        'paragraph': para,
                        'entry_idx': entry_idx
                    })

        # === Step 4: 选择最高分答案 ===
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            answer = best['answer']
        else:
            # 如果没有相关段落，仍然使用第一个段落生成答案
            first_para = self._split_into_paragraphs(entries[0])[0] if entries else ""
            if first_para:
                answer, _ = self._generate_with_paragraph(sample, first_para)
            else:
                answer = self._direct_answer(sample)

        # 转换为文档字典格式
        retrieved_docs_dict = []
        for i, entry in enumerate(entries):
            retrieved_docs_dict.append({
                'contents': entry,
                'id': f'mr2ag_doc_{i}',
                'title': '',
                'source': 'mr2ag_retriever'
            })

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs_dict,
            'retrieved': True,
            'retrieval_decision': 'Retrieval',
            'total_paragraphs': total_paras,
            'relevant_paragraphs': len(candidates),
            'golden_answers': sample.get('golden_answers', []),
            'correct': False
        }

    def _retrieval_reflection(self, question: str, image=None) -> bool:
        """
        判断是否需要检索 - 更宽松的版本
        默认需要检索，只有明显是常识才不检索
        """
        # 常识问题关键词
        common_sense_keywords = [
            'color', 'color is', 'what color', 'how many', 'count', 'number of',
            'is this', 'are these', 'do you see', 'what is in the picture'
        ]

        question_lower = question.lower()

        # 如果包含常识关键词，可能不需要检索
        for keyword in common_sense_keywords:
            if keyword in question_lower:
                return False  # 可能是常识问题

        # 特殊类型的问题（需要知识）
        knowledge_keywords = [
            'what year', 'when was', 'who designed', 'what is the name of',
            'what type of', 'what kind of', 'what brand', 'what model',
            'why might', 'how does', 'how do', 'what could'
        ]

        for keyword in knowledge_keywords:
            if keyword in question_lower:
                return True  # 需要外部知识

        # 默认需要检索（更积极）
        return True

    def _retrieve_documents(self, question: str) -> List[str]:
        """检索文档"""
        if self.retriever is None:
            return []

        try:
            if hasattr(self.retriever, 'search'):
                results = self.retriever.search(question, num=self.top_k)
            elif hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query_text=question, top_k=self.top_k)
            else:
                return []

            # 提取文档
            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                docs = results
                scores = [1.0 - i*0.1 for i in range(len(docs))]
            else:
                docs = [results]
                scores = [1.0]

            # 提取文本
            docs_text = []
            for doc in docs:
                if isinstance(doc, dict):
                    text = doc.get('contents', doc.get('text', str(doc)))
                else:
                    text = str(doc)
                docs_text.append(text)

            return docs_text

        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return []

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """段落切分"""
        # 按句号切分
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        paragraphs = []
        current_para = ""

        for sent in sentences:
            if len(current_para) + len(sent) < self.para_max_len:
                current_para += " " + sent
            else:
                if len(current_para) > self.para_min_len:
                    paragraphs.append(current_para.strip())
                current_para = sent

        if len(current_para) > self.para_min_len:
            paragraphs.append(current_para.strip())

        # 如果没有段落，返回原文
        return paragraphs if paragraphs else [text[:self.para_max_len]]

    def _relevance_reflection(self, question: str, paragraph: str) -> Tuple[bool, float]:
        """
        段落相关性判断 - 更宽松的版本
        """
        question_words = set(question.lower().split())
        paragraph_words = set(paragraph.lower().split())

        # 移除停用词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those'}
        question_words = question_words - stopwords
        paragraph_words = paragraph_words - stopwords

        # 计算相关性分数
        if question_words:
            common_words = question_words & paragraph_words
            score = len(common_words) / len(question_words)
        else:
            score = 0.0

        # 更宽松的阈值
        return (score > 0.2, score)  # 从0.3降到0.2

    def _generate_with_paragraph(self, sample: Dict, paragraph: str) -> Tuple[str, float]:
        """
        基于段落生成答案 - 改进的prompt
        """
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based on the evidence, answer the question.

Evidence: {paragraph}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Give the answer letter only (A/B/C/D):"""
        else:
            prompt = f"""Using the information below, answer the question with a short phrase.

Information: {paragraph}

Question: {sample['question']}

Answer:"""

        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=30,  # 增加到30
                temperature=self.temperature,
                do_sample=False
            )

            # 多选题答案映射
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                answer = self._map_mc_answer(answer, sample)
            else:
                # 使用智能答案提取
                answer = extract_answer_smart(answer.strip())

            # 置信度：固定值
            confidence = 0.8

            return (answer, confidence)

        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ("", 0.0)

    def _direct_answer(self, sample: Dict) -> str:
        """直接回答 - 改进的prompt"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Answer this multiple choice question.

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
        else:
            prompt = f"""Answer this question briefly in 1-5 words.

Question: {sample['question']}

Answer:"""

        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=30,
                temperature=self.temperature,
                do_sample=False
            )

            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)

            # 使用智能答案提取
            return extract_answer_smart(answer.strip())

        except Exception as e:
            warnings.warn(f"直接回答失败: {e}")
            return ""

    def _map_mc_answer(self, response: str, sample: Dict) -> str:
        """映射多选题答案"""
        response_upper = response.strip().upper()

        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return sample.get(letter, response.strip())

        return response.strip()


def create_mr2ag_fixed(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建mR²AG Fixed"""
    return MR2AGFixed(qwen3vl_wrapper, retriever, kwargs)


# 为了兼容性，添加原始创建函数
def create_mr2ag_enhanced(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建mR²AG Fixed"""
    return MR2AGFixed(qwen3vl_wrapper, retriever, kwargs)


def adapt_mr2ag_for_okvqa(qwen3vl_wrapper, retriever=None):
    """适配OK-VQA数据集"""
    return MR2AGFixed(qwen3vl_wrapper, retriever)


if __name__ == '__main__':
    print("mR²AG Fixed - 修复版本")
    print("=" * 70)
    print("主要修复:")
    print("  1. Retrieval-Reflection更宽松（默认需要检索）")
    print("  2. 改进答案生成prompt")
    print("  3. Relevance-Reflection阈值降低（0.3→0.2）")
    print("  4. 使用extract_answer_smart提取答案")
    print("=" * 70)