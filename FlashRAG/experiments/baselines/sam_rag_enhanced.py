#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM-RAG Enhanced - 完整实现（基于Qwen3-VL）

参考论文: Self-adaptive Multimodal Retrieval-Augmented Generation
核心特色:
1. 批次检索 (Batch Retrieval): 逐批检索文档，找到相关内容后停止
2. 相关性判断 (Relevance Judgment): 判断文本/图像是否与问题相关
3. 答案质量评估 (Answer Quality): 评估答案是否被内容支持
4. 自适应迭代 (Adaptive Iteration): 如果答案不满足条件，继续检索

适配说明:
- 原始SAM-RAG使用LLaVA，我们使用Qwen3-VL
- 原始使用MultiModalQA数据集，我们使用MRAG-Bench
- 简化版本：不使用self-consistency voting（太慢）
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List
import warnings
import re
import json


class SAMRAGEnhanced:
    """
    SAM-RAG Enhanced完整实现
    
    自适应批次检索流程:
    1. 批次检索文档
    2. 判断文档相关性 (isRel)
    3. 生成答案
    4. 判断答案支持度 (isSup)
    5. 判断答案有用性 (isUse)
    6. 如果不满足条件，继续下一批
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        """
        初始化SAM-RAG Enhanced
        
        Args:
            qwen3vl_wrapper: Qwen3-VL封装器
            retriever: 检索器
            config: 配置
        """
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        # SAM-RAG参数
        self.batch_size = self.config.get('sam_batch_size', 5)  # 每批检索的文档数
        self.max_batches = self.config.get('sam_max_batches', 4)  # 最多检索批次
        self.total_docs = self.batch_size * self.max_batches  # 总文档数
        
        # 判断温度（低温度获得更确定的判断）
        self.decision_temp = 0.05
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个样本
        
        Args:
            sample: 样本数据
            
        Returns:
            结果字典
        """
        question = sample['question']
        query_image = sample.get('query_image')
        
        # 1. 检索所有文档
        all_docs = self.retriever.retrieve(
            query=question,
            query_image=query_image,
            top_k=self.total_docs
        )
        
        # 2. 批次处理
        relevant_contents = []
        relevant_ids = []
        answer = None
        final_batch = 0
        
        for batch_idx in range(self.max_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_docs))
            batch_docs = all_docs[start_idx:end_idx]
            
            if not batch_docs:
                break
            
            # 2.1 判断相关性
            for doc in batch_docs:
                is_relevant = self._judge_relevance(doc, question)
                
                if is_relevant:
                    relevant_contents.append(doc['content'])
                    relevant_ids.append(doc.get('doc_id', f"doc_{len(relevant_ids)}"))
            
            # 2.2 如果找到相关内容，尝试回答
            if relevant_contents:
                answer = self._generate_answer(question, relevant_contents)
                
                # 2.3 评估答案质量
                is_supported = self._judge_support(relevant_contents, question, answer)
                is_useful = self._judge_usefulness(relevant_contents, question, answer)
                
                # 2.4 如果答案满足条件，返回
                if is_supported == 'True' and is_useful:
                    final_batch = batch_idx + 1
                    break
                
                # 2.5 如果不满足，继续下一批
                if not is_useful:
                    # 重新生成答案
                    answer = self._generate_answer(question, relevant_contents)
                    is_useful = self._judge_usefulness(relevant_contents, question, answer)
                
                if is_supported == 'Partial':
                    # 部分支持，继续检索
                    final_batch = batch_idx + 1
                    continue
                elif is_supported == 'False':
                    # 不支持，清空并继续
                    relevant_contents = []
                    relevant_ids = []
                    answer = None
                    final_batch = batch_idx + 1
                    continue
                else:
                    # True，满足条件
                    final_batch = batch_idx + 1
                    break
            
            final_batch = batch_idx + 1
        
        # 3. 如果所有批次都处理完，返回最终答案
        if not answer:
            if relevant_contents:
                answer = self._generate_answer(question, relevant_contents)
            else:
                answer = self._generate_answer(question, [])  # 无检索答案
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': relevant_ids,
            'num_retrieved': len(relevant_ids),
            'num_batches_used': final_batch,
        }

    def _judge_relevance(self, doc: Dict, question: str) -> bool:
        """判断文档是否与问题相关"""
        doc_type = doc.get('type', 'text')

        if doc_type == 'image':
            # 图像相关性判断
            prompt = self._build_image_relevance_prompt(doc, question)
            image = doc.get('image')
        else:
            # 文本相关性判断
            prompt = self._build_text_relevance_prompt(doc, question)
            image = None

        response = self.qwen3vl.generate(
            prompt=prompt,
            image=image,
            temperature=self.decision_temp,
            max_new_tokens=512
        )

        # 解析响应
        try:
            result = self._parse_json_response(response)
            return result.get('Response', 'False') in ['True', True]
        except:
            # 如果解析失败，使用简单的关键词匹配
            return 'true' in response.lower()

    def _generate_answer(self, question: str, contents: List[str]) -> str:
        """生成答案"""
        if contents:
            context = '\n\n'.join(contents[:10])  # 最多使用10个文档
            prompt = f"""Based on the following content, answer the question concisely.

Content:
{context}

Question: {question}

Provide a direct and concise answer (no explanation needed):"""
        else:
            prompt = f"""Answer the following question based on your knowledge:

Question: {question}

Provide a direct and concise answer:"""

        response = self.qwen3vl.generate(
            prompt=prompt,
            temperature=0.1,
            max_new_tokens=256
        )

        return response.strip()

    def _judge_support(self, contents: List[str], question: str, answer: str) -> str:
        """
        判断答案是否被内容支持

        Returns:
            'True': 完全支持
            'Partial': 部分支持
            'False': 不支持
        """
        context = '\n\n'.join(contents[:10])
        prompt = f"""Determine if the answer is supported by the content.

Content:
{context}

Question: {question}
Answer: {answer}

Is the answer fully supported by the content?
- "True": The answer is fully supported
- "Partial": The answer is partially supported
- "False": The answer is not supported

Respond with JSON format:
{{"Response": "True"/"Partial"/"False"}}"""

        response = self.qwen3vl.generate(
            prompt=prompt,
            temperature=self.decision_temp,
            max_new_tokens=256
        )

        try:
            result = self._parse_json_response(response)
            return result.get('Response', 'False')
        except:
            # 简单判断
            if 'true' in response.lower():
                return 'True'
            elif 'partial' in response.lower():
                return 'Partial'
            else:
                return 'False'

    def _judge_usefulness(self, contents: List[str], question: str, answer: str) -> bool:
        """判断答案是否正确使用了内容"""
        context = '\n\n'.join(contents[:10])
        prompt = f"""Determine if the answer correctly uses the content to answer the question.

Content:
{context}

Question: {question}
Answer: {answer}

Is the answer appropriate and correctly uses the content?

Respond with JSON format:
{{"Response": "True"/"False"}}"""

        response = self.qwen3vl.generate(
            prompt=prompt,
            temperature=self.decision_temp,
            max_new_tokens=256
        )

        try:
            result = self._parse_json_response(response)
            return result.get('Response', 'False') in ['True', True]
        except:
            return 'true' in response.lower()

    def _build_text_relevance_prompt(self, doc: Dict, question: str) -> str:
        """构建文本相关性判断prompt"""
        title = doc.get('title', '')
        content = doc.get('content', '')

        prompt = f"""Determine if this text is related to the question.

Title: {title}
Content: {content}

Question: {question}

Is this text related to answering the question?

Respond with JSON format:
{{"Response": "True"/"False"}}"""

        return prompt

    def _build_image_relevance_prompt(self, doc: Dict, question: str) -> str:
        """构建图像相关性判断prompt"""
        title = doc.get('title', 'Image')

        prompt = f"""Determine if this image is related to the question.

Image Title: {title}
Question: {question}

Is this image related to answering the question?

Respond with JSON format:
{{"Response": "True"/"False"}}"""

        return prompt

    def _parse_json_response(self, response: str) -> Dict:
        """解析JSON响应"""
        # 提取JSON部分
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            # 处理True/False
            json_str = json_str.replace('True', '"True"').replace('False', '"False"')
            json_str = json_str.replace('Partial', '"Partial"')
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

