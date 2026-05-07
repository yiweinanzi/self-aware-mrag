#!/usr/bin/env python3
"""
MuRAG Enhanced - 调试版本
添加了详细的输出调试信息
"""

import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# 复制原始MuRAG代码，但添加调试输出
from murag_enhanced import MuRAGEnhanced

class MuRAGEnhancedDebug(MuRAGEnhanced):
    """MuRAG with debug output"""

    def _generate_with_single_doc(self, sample: Dict, doc: Dict) -> str:
        """
        基于单个文档独立生成答案（调试版本）
        """
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based ONLY on this single evidence document, answer the question.

Evidence: {doc.get('contents', doc.get('text', str(doc)))[:300]}...

Question: {sample['question']}

Choices:
-A. {sample['A']}
-B. {sample['B']}
-C. {sample['C']}
-D. {sample['D']}

Answer (letter only):"""
        else:
            prompt = f"""Answer with 1-3 words only based on the evidence.

Evidence: {doc.get('contents', doc.get('text', str(doc)))[:300]}...

Question: {sample['question']}

Answer:"""

        try:
            print(f"[MuRAG DEBUG] 生成中...")
            print(f"[MuRAG DEBUG] 问题: {sample['question']}")
            print(f"[MuRAG DEBUG] 提示: {prompt[:200]}...")

            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=False
            )

            print(f"[MuRAG DEBUG] 原始答案: '{answer}'")

            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                final_answer = self._map_mc_answer(answer, sample)
                print(f"[MuRAG DEBUG] 选择题答案: {final_answer}")
            else:
                from flashrag.utils.vqa_evaluator import extract_okvqa_answer
                final_answer = extract_okvqa_answer(answer.strip())
                print(f"[MuRAG DEBUG] 提取后答案: '{final_answer}'")

            print(f"[MuRAG DEBUG] 最终答案: '{final_answer}'")
            print(f"[MuRAG DEBUG] 正确答案: {sample.get('golden_answers', [])}")

            # 检查答案是否正确
            golden = sample.get('golden_answers', [])
            is_correct = final_answer in golden
            print(f"[MuRAG DEBUG] 是否正确: {'✅' if is_correct else '❌'}")

            return final_answer

        except Exception as e:
            warnings.warn(f"单文档生成失败: {e}")
            print(f"[MuRAG DEBUG] 生成失败: {e}")
            return ""

    def _direct_answer(self, sample: Dict) -> str:
        """直接回答（调试版本）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Question: {sample['question']}

Choices:
-A. {sample['A']}
-B. {sample['B']}
-C. {sample['C']}
-D. {sample['D']}

Answer (letter only):"""
        else:
            prompt = f"Answer with 1-3 words only.\n\nQuestion: {sample['question']}\n\nAnswer:"

        try:
            print(f"[MuRAG DEBUG] 直接回答生成...")
            print(f"[MuRAG DEBUG] 问题: {sample['question']}")

            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=False
            )

            print(f"[MuRAG DEBUG] 直接回答原始答案: '{answer}'")

            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                final_answer = self._map_mc_answer(answer, sample)
            else:
                from flashrag.utils.vqa_evaluator import extract_okvqa_answer
                final_answer = extract_okvqa_answer(answer.strip())

            print(f"[MuRAG DEBUG] 直接回答最终: '{final_answer}'")

            return final_answer

        except Exception as e:
            print(f"[MuRAG DEBUG] 直接回答失败: {e}")
            return ""

    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个样本（调试版本）"""
        question = sample['question']
        golden_answers = sample.get('golden_answers', [])

        print(f"\n{'='*60}")
        print(f"[MuRAG DEBUG] 处理样本: {question}")
        print(f"[MuRAG DEBUG] 正确答案: {golden_answers}")
        print(f"{'='*60}")

        # Step 1: 检索多个证据
        docs = self._retrieve_documents(question)

        if not docs:
            print("[MuRAG DEBUG] 没有检索到文档，使用直接回答")
            answer = self._direct_answer(sample)

            # 检查答案
            is_correct = answer in golden_answers if golden_answers else False
            print(f"[MuRAG DEBUG] 直接回答正确性: {'✅' if is_correct else '❌'}")

            return {
                'question': question,
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'reasoning': 'No documents retrieved',
                'method': 'MuRAG',
                'is_correct': is_correct
            }

        print(f"[MuRAG DEBUG] 检索到 {len(docs)} 个文档")

        # Step 2: 使用多证据生成子答案
        sub_answers = []
        selected_docs = docs[:self.ensemble_k]

        print(f"[MuRAG DEBUG] 使用前 {len(selected_docs)} 个文档生成答案")

        for i, doc in enumerate(selected_docs):
            print(f"\n[MuRAG DEBUG] 文档 {i+1}/{len(selected_docs)}:")
            sub_answer = self._generate_with_single_doc(sample, doc)
            if sub_answer:
                sub_answers.append(sub_answer)

        print(f"\n[MuRAG DEBUG] 子答案列表: {sub_answers}")

        # Step 3: 投票融合
        if sub_answers:
            final_answer = self._voting_fusion(sub_answers)
            print(f"[MuRAG DEBUG] 投票融合后: '{final_answer}'")
        else:
            final_answer = self._direct_answer(sample)
            print(f"[MuRAG DEBUG] 投票失败，使用直接回答: '{final_answer}'")

        # 转换文档格式
        retrieved_docs = []
        for i, doc in enumerate(selected_docs):
            if isinstance(doc, dict):
                retrieved_docs.append(doc)
            else:
                retrieved_docs.append({
                    'contents': str(doc),
                    'id': f"murag_doc_{i}",
                    'title': '',
                    'source': 'murag_retriever'
                })

        # 检查最终答案
        is_correct = final_answer in golden_answers if golden_answers else False
        print(f"\n[MuRAG DEBUG] 最终答案: '{final_answer}'")
        print(f"[MuRAG DEBUG] 最终正确性: {'✅' if is_correct else '❌'}")

        result = {
            'answer': final_answer,
            'raw_prediction': final_answer,
            'retrieved_docs': retrieved_docs,
            'used_retrieval': True,
            'reasoning': f'MuRAG processed {len(sub_answers)} sub-answers',
            'method': 'MuRAG',
            'sub_answers': sub_answers,
            'is_correct': is_correct
        }

        return result


def create_murag_enhanced_debug(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建调试版MuRAG"""
    return MuRAGEnhancedDebug(qwen3vl_wrapper, retriever, kwargs)