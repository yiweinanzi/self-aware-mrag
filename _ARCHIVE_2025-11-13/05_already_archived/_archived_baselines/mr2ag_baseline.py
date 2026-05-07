#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mR²AG Baseline - 简化版

基于论文: mR2AG: Multimodal Retrieval-Reflection-Augmented Generation
参考文档: /root/autodl-tmp/open_resource/m_r_ag_复现指南（面向_cursor）.md

核心思想：
1. Retrieval-Reflection: 判断是否需要检索 ([Retrieval] 或 [No Retrieval])
2. Relevance-Reflection: 判断段落是否相关 ([Relevant] 或 [Irrelevant])
3. Answer Generation: 基于相关段落生成答案
4. 层级后处理: S_ret × S_rel × S_ans

简化说明：
- 使用提示词工程而非完整的mR²AG-IT训练
- 使用LLaVA的生成能力和提示词来模拟三阶段推理
- 专用token通过提示词引导而非词表扩展
"""

import torch
import warnings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class MR2AGBaseline:
    """
    mR²AG Baseline (简化版)
    
    三阶段推理流程：
    1. Retrieval-Reflection: 判断是否需要检索
    2. Relevance-Reflection: 判断段落相关性
    3. Answer Generation: 基于相关段落生成答案
    
    使用示例：
    ```python
    baseline = MR2AGBaseline(
        llava_wrapper=llava,
        retriever=retriever
    )
    results = baseline.run(dataset)
    ```
    """
    
    def __init__(self, llava_wrapper, retriever=None, config=None):
        """
        初始化mR²AG baseline
        
        Args:
            llava_wrapper: LLaVA模型封装器
            retriever: 检索器
            config: 配置字典
        """
        self.llava = llava_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        # 配置
        self.top_k = self.config.get('retrieval_topk', 5)
        self.max_new_tokens = self.config.get('max_new_tokens', 100)
        self.temperature = self.config.get('temperature', 0.2)
        
        # mR²AG特定配置
        self.retrieval_threshold = self.config.get('retrieval_threshold', 0.5)
        self.relevance_threshold = self.config.get('relevance_threshold', 0.5)
        self.段落最大长度 = self.config.get('max_paragraph_len', 200)
        
        # 提示词模板
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化提示词模板"""
        # 1. Retrieval-Reflection 提示词
        self.retrieval_prompt_template = """Given this image and question, do you need external knowledge to answer?

Question: {question}

Respond with ONLY ONE WORD:
- "YES" if you need external knowledge
- "NO" if you can answer from the image alone

Answer:"""
        
        # 2. Relevance-Reflection 提示词
        self.relevance_prompt_template = """Given the question and this paragraph, is the paragraph relevant to answering the question?

Question: {question}

Paragraph: {paragraph}

Respond with ONLY ONE WORD:
- "RELEVANT" if the paragraph helps answer the question
- "IRRELEVANT" if it doesn't help

Answer:"""
        
        # 3. Answer Generation 提示词
        self.answer_prompt_template = """Use the following relevant information to answer the question. Be concise.

Question: {question}

Relevant Information:
{evidence}

Answer:"""
    
    # =========================================================================
    # 阶段1: Retrieval-Reflection
    # =========================================================================
    
    def retrieval_reflection(self, question: str, image=None) -> Tuple[bool, float]:
        """
        判断是否需要检索外部知识
        
        Args:
            question: 问题文本
            image: 图像
            
        Returns:
            (should_retrieve, confidence)
        """
        prompt = self.retrieval_prompt_template.format(question=question)
        
        try:
            # 生成判断
            response = self.llava.generate(
                text=prompt,
                image=image,
                max_new_tokens=10,
                temperature=0.1  # 低温度获得确定性输出
            )
            
            response_lower = response.strip().lower()
            
            # 解析响应
            if 'yes' in response_lower:
                should_retrieve = True
                confidence = 1.0
            elif 'no' in response_lower:
                should_retrieve = False
                confidence = 1.0
            else:
                # 默认：如果不确定，选择检索（保守策略）
                should_retrieve = True
                confidence = 0.5
                warnings.warn(f"Retrieval-Reflection响应不明确: {response}，默认检索")
            
            return should_retrieve, confidence
        
        except Exception as e:
            warnings.warn(f"Retrieval-Reflection失败: {e}，默认检索")
            return True, 0.5
    
    # =========================================================================
    # 阶段2: Relevance-Reflection + Retrieval
    # =========================================================================
    
    def retrieve_and_filter(self, question: str, image=None, 
                           top_k: int = None) -> List[Dict[str, Any]]:
        """
        检索并过滤相关段落
        
        Args:
            question: 问题
            image: 图像
            top_k: 检索数量
            
        Returns:
            List[Dict]: 相关段落列表，每个包含{'text', 'score', 'relevance'}
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.retriever is None:
            warnings.warn("未提供检索器")
            return []
        
        try:
            # 检索Top-K条目
            retrieved = self.retriever.retrieve(
                query_text=question,
                query_image=image,
                top_k=top_k
            )
            
            # 处理返回格式
            if isinstance(retrieved, tuple):
                docs, scores = retrieved
            else:
                docs = retrieved
                scores = [1.0] * len(docs)
            
            # 切分段落并进行Relevance-Reflection
            relevant_paragraphs = []
            
            for doc, ret_score in zip(docs, scores):
                # 简单切分段落（按句子）
                paragraphs = self._split_paragraphs(doc)
                
                for para in paragraphs:
                    # Relevance-Reflection
                    is_relevant, rel_score = self.relevance_reflection(
                        question, para, image
                    )
                    
                    if is_relevant:
                        relevant_paragraphs.append({
                            'text': para,
                            'retrieval_score': ret_score,
                            'relevance_score': rel_score
                        })
            
            return relevant_paragraphs
        
        except Exception as e:
            warnings.warn(f"检索和过滤失败: {e}")
            return []
    
    def relevance_reflection(self, question: str, paragraph: str, 
                            image=None) -> Tuple[bool, float]:
        """
        判断段落是否相关
        
        Args:
            question: 问题
            paragraph: 段落文本
            image: 图像（可选）
            
        Returns:
            (is_relevant, confidence)
        """
        # 截断过长段落
        if len(paragraph) > self.段落最大长度:
            paragraph = paragraph[:self.段落最大长度] + "..."
        
        prompt = self.relevance_prompt_template.format(
            question=question,
            paragraph=paragraph
        )
        
        try:
            response = self.llava.generate(
                text=prompt,
                image=image,
                max_new_tokens=10,
                temperature=0.1
            )
            
            response_lower = response.strip().lower()
            
            if 'relevant' in response_lower and 'irrelevant' not in response_lower:
                return True, 1.0
            elif 'irrelevant' in response_lower:
                return False, 0.0
            else:
                # 默认相关（保守）
                return True, 0.5
        
        except Exception as e:
            warnings.warn(f"Relevance-Reflection失败: {e}")
            return True, 0.5
    
    def _split_paragraphs(self, text: str, max_len: int = None) -> List[str]:
        """
        简单的段落切分
        
        Args:
            text: 文本
            max_len: 最大长度
            
        Returns:
            List[str]: 段落列表
        """
        if max_len is None:
            max_len = self.段落最大长度
        
        # 按句号切分
        sentences = text.split('. ')
        
        paragraphs = []
        current_para = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if len(current_para) + len(sent) < max_len:
                current_para += sent + ". "
            else:
                if current_para:
                    paragraphs.append(current_para.strip())
                current_para = sent + ". "
        
        if current_para:
            paragraphs.append(current_para.strip())
        
        return paragraphs if paragraphs else [text[:max_len]]
    
    # =========================================================================
    # 阶段3: Answer Generation with Hierarchical Scoring
    # =========================================================================
    
    def generate_answer_with_evidence(self, question: str, 
                                       relevant_paragraphs: List[Dict],
                                       image=None) -> Tuple[str, float]:
        """
        基于相关段落生成答案，使用层级打分
        
        Args:
            question: 问题
            relevant_paragraphs: 相关段落列表
            image: 图像
            
        Returns:
            (answer, confidence)
        """
        if not relevant_paragraphs:
            # 无相关证据，直接回答
            return self._direct_answer(question, image)
        
        # 选择最佳段落（层级打分：S_ret × S_rel）
        best_para = max(
            relevant_paragraphs,
            key=lambda p: p['retrieval_score'] * p['relevance_score']
        )
        
        # 组织证据
        evidence = best_para['text']
        
        # 也可以使用Top-3段落
        if len(relevant_paragraphs) > 1:
            top3 = sorted(
                relevant_paragraphs,
                key=lambda p: p['retrieval_score'] * p['relevance_score'],
                reverse=True
            )[:3]
            evidence = "\n\n".join([p['text'] for p in top3])
        
        # 生成答案
        prompt = self.answer_prompt_template.format(
            question=question,
            evidence=evidence
        )
        
        try:
            answer = self.llava.generate(
                text=prompt,
                image=image,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            
            # 计算总体置信度（简化版）
            confidence = best_para['retrieval_score'] * best_para['relevance_score']
            
            return answer, confidence
        
        except Exception as e:
            warnings.warn(f"答案生成失败: {e}")
            return "", 0.0
    
    def _direct_answer(self, question: str, image=None) -> Tuple[str, float]:
        """无证据时的直接回答"""
        prompt = f"Question: {question}\nAnswer:"
        
        try:
            answer = self.llava.generate(
                text=prompt,
                image=image,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            return answer, 0.5  # 较低置信度
        except Exception as e:
            warnings.warn(f"直接回答失败: {e}")
            return "", 0.0
    
    # =========================================================================
    # 完整Pipeline
    # =========================================================================
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整的mR²AG三阶段推理
        
        Args:
            sample: 样本字典
            
        Returns:
            Dict: 结果字典
        """
        question = sample['question']
        image = sample.get('image', None)
        
        # 阶段1: Retrieval-Reflection
        should_retrieve, ret_confidence = self.retrieval_reflection(question, image)
        
        if not should_retrieve:
            # 不需要检索，直接回答
            answer, ans_confidence = self._direct_answer(question, image)
            
            return {
                'id': sample.get('id'),
                'question': question,
                'prediction': answer,
                'golden_answers': sample.get('golden_answers', []),
                'retrieval_decision': 'No Retrieval',
                'relevant_paragraphs': 0,
                'confidence': ans_confidence
            }
        
        # 阶段2: 检索 + Relevance-Reflection
        relevant_paragraphs = self.retrieve_and_filter(question, image, self.top_k)
        
        # 阶段3: Answer Generation with Hierarchical Scoring
        answer, ans_confidence = self.generate_answer_with_evidence(
            question, relevant_paragraphs, image
        )
        
        return {
            'id': sample.get('id'),
            'question': question,
            'prediction': answer,
            'golden_answers': sample.get('golden_answers', []),
            'retrieval_decision': 'Retrieval',
            'relevant_paragraphs': len(relevant_paragraphs),
            'confidence': ans_confidence,
            'evidence': [p['text'][:100] + '...' for p in relevant_paragraphs[:3]]
        }
    
    def run(self, dataset, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        在数据集上运行mR²AG
        
        Args:
            dataset: 数据集
            verbose: 是否显示进度
            
        Returns:
            List[Dict]: 结果列表
        """
        results = []
        
        if verbose:
            print("mR²AG Baseline运行中...")
            print("三阶段推理：Retrieval-Reflection → Relevance-Reflection → Answer Generation")
            print()
        
        retrieval_count = 0
        relevant_para_total = 0
        
        for i, sample in enumerate(dataset):
            try:
                result = self.run_single(sample)
                results.append(result)
                
                if result['retrieval_decision'] == 'Retrieval':
                    retrieval_count += 1
                    relevant_para_total += result['relevant_paragraphs']
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
                    print(f"  检索率: {retrieval_count/(i+1)*100:.1f}%")
                    print(f"  平均相关段落: {relevant_para_total/max(retrieval_count,1):.1f}")
            
            except Exception as e:
                warnings.warn(f"样本{i}处理失败: {e}")
                continue
        
        if verbose:
            print(f"\n✅ 完成！")
            print(f"  总样本: {len(results)}")
            print(f"  触发检索: {retrieval_count} ({retrieval_count/len(results)*100:.1f}%)")
            print(f"  平均相关段落: {relevant_para_total/max(retrieval_count,1):.1f}")
        
        return results


# 工厂函数
def create_mr2ag_baseline(llava_wrapper, retriever=None, **kwargs):
    """创建mR²AG baseline"""
    return MR2AGBaseline(llava_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("mR²AG Baseline (简化版)")
    print("=" * 70)
    print("基于论文: mR2AG: Multimodal Retrieval-Reflection-Augmented Generation")
    print("\n三阶段推理流程:")
    print("  1. Retrieval-Reflection: 判断是否需要检索")
    print("  2. Relevance-Reflection: 判断段落相关性")
    print("  3. Answer Generation: 基于相关段落生成答案")
    print("\n简化说明:")
    print("  - 使用提示词工程而非完整训练")
    print("  - 保留核心的三阶段推理框架")
    print("  - 层级打分: S_ret × S_rel × S_ans")
    print("\n使用方法:")
    print("  from experiments.baselines.mr2ag_baseline import MR2AGBaseline")
    print("  baseline = MR2AGBaseline(llava, retriever)")
    print("  results = baseline.run(dataset)")
    print("=" * 70)


