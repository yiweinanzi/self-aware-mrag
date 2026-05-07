"""
SAM-RAG (Self-Adaptive Multimodal RAG) Baseline

参考论文: Self-adaptive Multimodal Retrieval-Augmented Generation
核心思想:
1. 批次检索 (Batch Retrieval): 逐批检索文档，找到相关内容后停止
2. 相关性判断 (Relevance Judgment): 判断文本/图像是否与问题相关
3. 答案质量评估 (Answer Quality): 评估答案是否被内容支持
4. 自适应迭代 (Adaptive Iteration): 如果答案不满足条件，继续检索

适配说明:
- 原始SAM-RAG使用LLaVA，我们使用Qwen3-VL
- 原始使用MultiModalQA数据集，我们使用MRAG-Bench
- 简化版本：不使用self-consistency voting（太慢）
"""

import json
import re
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image


class SAMRAG:
    """SAM-RAG Pipeline"""
    
    def __init__(self, mllm_model, retriever, config: Dict):
        """
        Args:
            mllm_model: Qwen3-VL模型
            retriever: 检索器
            config: 配置参数
        """
        self.mllm = mllm_model
        self.retriever = retriever
        self.config = config
        
        # SAM-RAG参数
        self.batch_size = config.get('sam_batch_size', 5)  # 每批检索的文档数
        self.max_batches = config.get('sam_max_batches', 4)  # 最多检索批次
        self.use_self_consistency = config.get('sam_self_consistency', False)  # 是否使用self-consistency
        
    def run(self, question: str, images: List[Image.Image] = None) -> Dict:
        """
        运行SAM-RAG pipeline
        
        Args:
            question: 问题
            images: 查询图像（可选）
            
        Returns:
            结果字典
        """
        # 1. 检索文档
        all_docs = self.retriever.retrieve(question, images, top_k=self.batch_size * self.max_batches)
        
        # 2. 批次处理
        relevant_contents = []
        relevant_ids = []
        
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
                    relevant_ids.append(doc.get('doc_id', ''))
            
            # 2.2 如果找到相关内容，尝试回答
            if relevant_contents:
                answer = self._generate_answer(question, relevant_contents)
                
                # 2.3 评估答案质量
                is_supported = self._judge_support(relevant_contents, question, answer)
                is_useful = self._judge_usefulness(relevant_contents, question, answer)
                
                # 2.4 如果答案满足条件，返回
                if is_supported and is_useful:
                    return {
                        'answer': answer,
                        'retrieved_docs': relevant_ids,
                        'num_batches': batch_idx + 1,
                        'is_supported': is_supported,
                        'is_useful': is_useful,
                    }
                
                # 2.5 如果不满足，继续下一批
                if not is_useful:
                    # 重新生成答案
                    answer = self._generate_answer(question, relevant_contents)
                
                if is_supported == 'partial':
                    # 部分支持，继续检索
                    continue
                elif not is_supported:
                    # 不支持，清空并继续
                    relevant_contents = []
                    relevant_ids = []
                    continue
        
        # 3. 如果所有批次都处理完，返回最终答案
        if relevant_contents:
            answer = self._generate_answer(question, relevant_contents)
        else:
            answer = self._generate_answer(question, [])  # 无检索答案
        
        return {
            'answer': answer,
            'retrieved_docs': relevant_ids,
            'num_batches': self.max_batches,
            'is_supported': False,
            'is_useful': False,
        }
    
    def _judge_relevance(self, doc: Dict, question: str) -> bool:
        """判断文档是否与问题相关"""
        # 简化版：使用MLLM判断相关性
        prompt = self._build_relevance_prompt(doc, question)
        response = self._mllm_inference(prompt, doc.get('image'))

        # 解析响应
        try:
            result = self._parse_json_response(response)
            return result.get('Response', False) in [True, 'True']
        except:
            return False

    def _generate_answer(self, question: str, contents: List[str]) -> str:
        """生成答案"""
        if contents:
            context = '\n'.join(contents)
            prompt = f"""Based on the following content, answer the question.

Content: {context}

Question: {question}

Provide a concise answer:"""
        else:
            prompt = f"""Answer the following question based on your knowledge:

Question: {question}

Provide a concise answer:"""

        response = self._mllm_inference(prompt)
        return response.strip()

    def _judge_support(self, contents: List[str], question: str, answer: str) -> bool:
        """判断答案是否被内容支持"""
        context = '\n'.join(contents)
        prompt = f"""Determine if the answer is supported by the content.

Content: {context}
Question: {question}
Answer: {answer}

Is the answer fully supported? Respond with JSON:
{{"Response": "True"/"Partial"/"False"}}"""

        response = self._mllm_inference(prompt)
        try:
            result = self._parse_json_response(response)
            return result.get('Response', 'False')
        except:
            return 'False'

    def _judge_usefulness(self, contents: List[str], question: str, answer: str) -> bool:
        """判断答案是否正确使用了内容"""
        context = '\n'.join(contents)
        prompt = f"""Determine if the answer correctly uses the content to answer the question.

Content: {context}
Question: {question}
Answer: {answer}

Is the answer appropriate? Respond with JSON:
{{"Response": "True"/"False"}}"""

        response = self._mllm_inference(prompt)
        try:
            result = self._parse_json_response(response)
            return result.get('Response', False) in [True, 'True']
        except:
            return False

    def _build_relevance_prompt(self, doc: Dict, question: str) -> str:
        """构建相关性判断prompt"""
        if doc.get('type') == 'image' or doc.get('image'):
            # 图像相关性
            title = doc.get('title', 'Image')
            return f"""Determine if this image is related to the question.

Title: {title}
Question: {question}

Respond with JSON:
{{"Reasoning": "...", "Response": "True"/"False"}}"""
        else:
            # 文本相关性
            title = doc.get('title', '')
            content = doc.get('content', '')
            return f"""Determine if this text is related to the question.

Title: {title}
Content: {content}
Question: {question}

Respond with JSON:
{{"Reasoning": "...", "Response": "True"/"False"}}"""

    def _mllm_inference(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """MLLM推理"""
        if image:
            # 多模态推理
            response = self.mllm.generate(prompt, image)
        else:
            # 纯文本推理
            response = self.mllm.generate(prompt)

        return response

    def _parse_json_response(self, response: str) -> Dict:
        """解析JSON响应"""
        # 提取JSON部分
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            # 处理True/False
            json_str = json_str.replace('True', 'true').replace('False', 'false')
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")


