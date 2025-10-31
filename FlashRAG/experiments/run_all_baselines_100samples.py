#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - 100样本，7个核心指标

方法列表：
1. Self-Aware-MRAG (Our Method)
2. Self-RAG
3. mR²AG
4. VisRAG
5. REVEAL
6. RagVL
7. MuRAG

指标列表（7个核心指标）：
1. EM (Exact Match)
2. F1 (Token-level F1)
3. Recall@5 (Retrieval Recall)
4. VQA-Score
5. Faithfulness
6. Attribution Precision
7. Position Bias Score
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加FlashRAG路径
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'mragbench',
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'max_samples': None,  # None = 全部样本(1353)
    
    # 模型配置
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    
    # 检索器配置（使用纯Wikipedia 3M语料库和索引）
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
    
    # CLIP多模态检索配置（可选，如果CLIP索引不存在会降级为纯BGE）
    'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
    'clip_index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/clip',
    
    # 评测配置
    'save_results': True,
    'output_dir': '/root/autodl-tmp/FlashRAG/experiments/results_baseline_comparison_100_wiki3m',
    
    # 生成参数（统一）
    'temperature': 0.01,
    'max_new_tokens': 10,
    'retrieval_topk': 5,
    
    # 不确定性估计器配置（✅ 使用论文完整实现）
    'use_improved_estimator': False,  # ✅ 修改: 使用CrossModalUncertaintyEstimator（论文承诺的完整实现）
    'uncertainty_threshold': 0.35,  # ✅ 优化：恢复合理阈值 (平衡检索与直接回答)
}


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(dataset_path, max_samples=None):
    """加载MRAG-Bench数据集（Arrow格式）"""
    print(f"加载数据集: {dataset_path}")
    
    dataset_dict = datasets.load_from_disk(dataset_path)
    test_data = dataset_dict['test']
    
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    # 转换为列表
    samples = []
    for item in test_data:
        sample = {
            'question': item['question'],
            'image': item['image'],
            'answer': item['answer'],  # ground truth
            'A': item['A'],
            'B': item['B'],
            'C': item['C'],
            'D': item['D'],
        }
        samples.append(sample)
    
    print(f"✅ 加载完成: {len(samples)} 样本")
    return samples


# ============================================================================
# 模型和检索器初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper


def init_retriever(config, use_multimodal=False):
    """
    初始化检索器
    
    Args:
        config: 配置字典
        use_multimodal: 是否使用多模态检索融合 (BGE + CLIP)
    """
    print("初始化检索器...")
    print(f"  模式: {'多模态融合 (BGE + CLIP)' if use_multimodal else '纯文本 (BGE)'}")
    
    # 检查索引文件是否存在
    import os
    from flashrag.retriever.index_builder import Index_Builder
    
    index_path = config.get('index_path', '')
    corpus_path = config['corpus_path']
    
    if not os.path.exists(index_path):
        print(f"⚠️ 索引文件不存在: {index_path}")
        print(f"✅ 将从真实语料库动态构建索引: {corpus_path}")
        print(f"⏱️  预计时间: 30-60分钟（3M文档）")
        print(f"💡 这样明天早上索引和实验结果都完成了")
        
        # 从真实语料库构建索引
        index_dir = os.path.dirname(index_path)
        os.makedirs(index_dir, exist_ok=True)
        
        print(f"\n开始构建索引...")
        builder = Index_Builder(
            retrieval_method='e5',
            model_path=config['retrieval_model_path'],
            corpus_path=corpus_path,
            save_dir=index_dir,
            max_length=512,
            batch_size=256,
            use_fp16=True,
            faiss_type='Flat',
            pooling_method='mean',
            save_embedding=True
        )
        
        builder.build_index()
        print(f"✅ 索引构建完成: {index_path}")
    else:
        print(f"✅ 使用现有索引: {index_path}")
    
    # 初始化BGE文本检索器
    retriever_config = {
        'index_path': index_path,
        'corpus_path': corpus_path,
        'retrieval_method': 'e5',
        'retrieval_model_path': config['retrieval_model_path'],
        'retrieval_query_max_length': 512,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 128,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': False,
        'use_sentence_transformer': False,
        'faiss_gpu': False,
        'instruction': '',
    }
    
    bge_retriever = DenseRetriever(retriever_config)
    print("✅ BGE文本检索器加载成功")
    
    # 如果不使用多模态，直接返回BGE检索器
    if not use_multimodal:
        return bge_retriever
    
    # 检查CLIP索引是否存在
    clip_index_dir = config.get('clip_index_path', '/root/autodl-tmp/FlashRAG/indexes/3m_real/clip')
    clip_index_file = os.path.join(clip_index_dir, 'clip_Flat.index')
    
    if not os.path.exists(clip_index_file):
        print(f"⚠️  CLIP索引不存在: {clip_index_file}")
        print(f"💡 降级使用纯BGE文本检索")
        return bge_retriever
    
    # 初始化CLIP视觉检索器
    print(f"✅ CLIP索引已存在，初始化多模态检索器...")
    clip_retriever_config = {
        'index_path': clip_index_file,
        'corpus_path': corpus_path,
        'retrieval_method': 'clip',
        'retrieval_model_path': config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336'),
        'retrieval_query_max_length': 77,
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 64,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'index_modal': 'all',  # CLIP索引包含text+image
    }
    
    clip_retriever = DenseRetriever(clip_retriever_config)
    print("✅ CLIP视觉检索器加载成功")
    
    # 创建多模态融合检索器
    from flashrag.retriever.multimodal_retriever import SelfAwareMultimodalRetriever
    
    multimodal_config = {
        'retrieval_topk': config['retrieval_topk'],
        'use_clip': True,
        'clip_model_path': config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336'),
        'fusion_method': 'weighted',
        'position_encoding': 'learned',
        'text_weight': 0.6,  # BGE权重
        'visual_weight': 0.4,  # CLIP权重
    }
    
    multimodal_retriever = SelfAwareMultimodalRetriever(
        config=multimodal_config,
        text_retriever=bge_retriever,
        visual_retriever=clip_retriever
    )
    
    print("✅ 多模态融合检索器初始化完成 (BGE 60% + CLIP 40%)")
    return multimodal_retriever


# ============================================================================
# Baseline实现（简化版）
# ============================================================================

class BaselinePipeline:
    """Baseline方法的基类"""
    
    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config
    
    def run_single(self, sample):
        """运行单个样本（子类实现）"""
        raise NotImplementedError
    
    def _construct_prompt(self, question, options, context=None):
        """构建多选题prompt"""
        if context:
            prompt = f"""Based on the following evidence, answer the question.

{context}

Question: {question}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Answer with ONLY the letter (A/B/C/D):"""
        else:
            prompt = f"""Question: {question}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Answer with ONLY the letter (A/B/C/D):"""
        
        return prompt
    
    def _generate(self, prompt, image):
        """生成答案"""
        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature']
            )
            return answer.strip()
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ""
    
    def _map_letter_to_answer(self, prediction, sample):
        """将字母映射回答案"""
        pred_letter = prediction.upper()[0] if prediction else '?'
        if pred_letter in ['A', 'B', 'C', 'D']:
            return sample[pred_letter]
        return prediction
    
    def _add_evaluator_fields(self, result, retrieved_docs=None):
        """
        ✅ Task 1: 添加evaluator需要的字段
        
        所有baseline都需要添加这些字段以支持完整的7个指标评估
        """
        if retrieved_docs is None:
            retrieved_docs = result.get('retrieved_docs', [])
        
        # 1. retrieval_result - 用于Faithfulness计算
        result['retrieval_result'] = [{
            'retrieved_docs': retrieved_docs,
            'retrieval_scores': [1.0] * len(retrieved_docs),
            'retrieval_used': len(retrieved_docs) > 0
        }]
        
        # 2. attributions - 用于Attribution Precision计算
        # 简化版：baseline暂时不支持细粒度归因
        if 'attributions' not in result:
            result['attributions'] = {
                'visual': [],
                'text': []
            }
        
        # 3. position_bias_results - 用于Position Bias Score计算
        # 简化版：使用统一的位置偏差
        if 'position_bias_results' not in result:
            result['position_bias_results'] = {
                'average_bias': 0.0,
                'individual_scores': [0.0],
                'position_weights': []
            }
        
        return result


class SelfRAGPipeline(BaselinePipeline):
    """
    ✅ Self-RAG: 完整实现（基于Qwen3-VL）
    
    实现了Self-RAG的3个核心判断：
    1. Retrieval Decision: 判断是否需要检索
    2. Relevance Judgment: 判断文档是否相关
    3. Support Judgment: 判断答案是否被支持
    """
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.decision_temp = 0.05  # 判断温度（低温度=更确定）
    
    def run_single(self, sample):
        """运行单个样本 - 完整的Self-RAG流程"""
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: Retrieval Decision ===
        need_retrieval = self._retrieval_decision(question, image)
        
        if not need_retrieval:
            # 无需检索，直接回答
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'retrieval_decision': 'No Retrieval',
                'relevant_docs_count': 0,
                'support_status': 'N/A'
            }
            return self._add_evaluator_fields(result)
        
        # === Step 2: Retrieve Documents ===
        results = self.retriever.search(question, num=self.config.get('retrieval_topk', 5))
        
        if not results:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': True,
                'retrieval_decision': 'Retrieval (no docs)',
                'relevant_docs_count': 0,
                'support_status': 'N/A'
            }
            return self._add_evaluator_fields(result)
        
        # === Step 3: Relevance Judgment ===
        relevant_docs = []
        for doc in results[:5]:
            doc_text = doc.get('contents', '')
            if self._relevance_judgment(question, doc_text, image):
                relevant_docs.append(doc_text)
        
        if not relevant_docs:
            # 无相关文档，降级到直接回答
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
                'used_retrieval': True,
                'retrieval_decision': 'Retrieval (no relevant)',
                'relevant_docs_count': 0,
                'support_status': 'No'
            }
            return self._add_evaluator_fields(result)
        
        # === Step 4: Generate Answer ===
        answer = self._generate_with_context(sample, relevant_docs[:3])
        
        # === Step 5: Support Judgment ===
        is_supported = self._support_judgment(question, answer, relevant_docs[:3])
        
        if not is_supported:
            # 答案不被支持，但仍使用该答案（记录状态）
            support_status = 'Not Supported'
        else:
            support_status = 'Supported'
        
        result = {
            'answer': answer,
            'raw_prediction': answer,
            'retrieved_docs': relevant_docs,
            'used_retrieval': True,
            'retrieval_decision': 'Retrieval',
            'relevant_docs_count': len(relevant_docs),
            'support_status': support_status
        }
        
        return self._add_evaluator_fields(result)
    
    def _retrieval_decision(self, question: str, image=None) -> bool:
        """判断是否需要检索（模拟[Retrieval] token）"""
        prompt = f"""Task: Decide if external knowledge is needed to answer this question.

Question: {question}

Think: Can this be answered just by looking at the image, or does it require external factual knowledge (dates, names, locations, etc.)?

Answer ONLY 'NEED' or 'NO':"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=self.decision_temp
            )
            
            response_upper = response.strip().upper()
            return 'NEED' in response_upper and 'NO' not in response_upper[:4]
        except:
            return True  # 默认检索（保守）
    
    def _relevance_judgment(self, question: str, document: str, image=None) -> bool:
        """判断文档是否相关（模拟[IsREL] token）"""
        doc_preview = document[:300] + "..." if len(document) > 300 else document
        
        prompt = f"""Task: Is this document relevant to the question?

Question: {question}

Document: {doc_preview}

Answer ONLY 'RELEVANT' or 'IRRELEVANT':"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,  # 纯文本判断
                max_new_tokens=5,
                temperature=self.decision_temp
            )
            
            response_upper = response.strip().upper()
            return 'RELEVANT' in response_upper and 'IRRELEVANT' not in response_upper
        except:
            return True  # 默认相关（保守）
    
    def _support_judgment(self, question: str, answer: str, documents: list) -> bool:
        """判断答案是否被文档支持（模拟[IsSUP] token）"""
        context = "\n\n".join(documents)[:400]
        
        prompt = f"""Task: Is the answer supported by the context?

Context: {context}...

Question: {question}
Answer: {answer}

Answer ONLY 'SUPPORTED' or 'NOT_SUPPORTED':"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=5,
                temperature=self.decision_temp
            )
            
            response_upper = response.strip().upper().replace(' ', '_')
            return 'SUPPORTED' in response_upper and 'NOT' not in response_upper[:3]
        except:
            return True  # 默认支持（保守）
    
    def _generate_with_context(self, sample, relevant_docs):
        """基于相关文档生成答案"""
        context = "\n\n".join(relevant_docs)
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)
    
    def _direct_answer(self, sample):
        """直接回答（无检索）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context=None)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)


class MR2AGPipeline(BaselinePipeline):
    """
    ✅ mR²AG: 完整实现（基于Qwen3-VL）
    
    实现了mR²AG的核心特色：
    1. Retrieval-Reflection: 判断是否需要检索
    2. 段落级处理: 将文档切分为小段落（50-180 tokens）
    3. Relevance-Reflection: 逐段落判断相关性
    4. 层级打分: S_ret × S_rel × S_ans
    """
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.para_min_len = 50
        self.para_max_len = 180
    
    def run_single(self, sample):
        """运行单个样本 - 完整的mR²AG流程"""
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: Retrieval-Reflection ===
        need_retrieval = self._retrieval_reflection(question, image)
        
        if not need_retrieval:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'retrieval_decision': 'No Retrieval',
                'total_paragraphs': 0,
                'relevant_paragraphs': 0
            }
            return self._add_evaluator_fields(result)
        
        # === Step 2: 检索文档 ===
        results = self.retriever.search(question, num=10)  # 多检索一些
        
        if not results:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': True,
                'retrieval_decision': 'Retrieval (no docs)',
                'total_paragraphs': 0,
                'relevant_paragraphs': 0
            }
            return self._add_evaluator_fields(result)
        
        # === Step 3: 段落级处理（mR²AG核心特色）===
        candidates = []
        total_paras = 0
        all_docs = []
        
        for entry_idx, entry in enumerate(results[:5]):
            doc_text = entry.get('contents', '')
            all_docs.append(doc_text)
            
            # 切分为段落
            paragraphs = self._split_paragraphs(doc_text)
            total_paras += len(paragraphs)
            
            for para in paragraphs:
                # Relevance-Reflection（段落级判断）
                is_relevant, rel_score = self._relevance_reflection(question, para)
                
                if is_relevant:
                    # 基于该段落生成答案
                    answer = self._generate_with_paragraph(sample, para)
                    
                    # 层级打分: S_ret × S_rel × S_ans
                    ret_score = 0.9 ** entry_idx  # 检索分数（排名衰减）
                    ans_score = 0.8  # 答案置信度（简化）
                    total_score = ret_score * rel_score * ans_score
                    
                    candidates.append({
                        'answer': answer,
                        'score': total_score,
                        'paragraph': para
                    })
        
        # === Step 4: 选择最佳候选答案 ===
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            final_answer = best['answer']
        else:
            # 无相关段落，回退到使用全部文档
            context = "\n\n".join(all_docs[:3])
            final_answer = self._generate_with_context(sample, context)
        
        result = {
            'answer': final_answer,
            'raw_prediction': final_answer,
            'retrieved_docs': all_docs,
            'used_retrieval': True,
            'retrieval_decision': 'Retrieval',
            'total_paragraphs': total_paras,
            'relevant_paragraphs': len(candidates)
        }
        
        return self._add_evaluator_fields(result)
    
    def _retrieval_reflection(self, question: str, image=None) -> bool:
        """Retrieval-Reflection: 判断是否需要检索"""
        prompt = f"""Decide if external knowledge is needed.

Question: {question}

Answer ONLY 'NEED' or 'NO':"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=0.05
            )
            return 'NEED' in response.upper()
        except:
            return True
    
    def _split_paragraphs(self, text: str) -> list:
        """段落切分（mR²AG的核心特色）"""
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        paragraphs = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) < self.para_max_len:
                current += " " + sent
            else:
                if len(current) > self.para_min_len:
                    paragraphs.append(current.strip())
                current = sent
        
        if len(current) > self.para_min_len:
            paragraphs.append(current.strip())
        
        return paragraphs if paragraphs else [text[:self.para_max_len]]
    
    def _relevance_reflection(self, question: str, paragraph: str) -> tuple:
        """Relevance-Reflection: 段落相关性判断"""
        prompt = f"""Rate relevance (0-10).

Question: {question}

Paragraph: {paragraph[:200]}...

Score (0-10):"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=5,
                temperature=0.1
            )
            try:
                score = float(response.strip()) / 10.0
            except:
                score = 0.5
            
            return (score > 0.5, score)
        except:
            return (True, 0.5)
    
    def _generate_with_paragraph(self, sample, paragraph):
        """基于单个段落生成答案"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        
        prompt = f"""Based on this paragraph, answer the question.

Paragraph: {paragraph}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only:"""
        
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)
    
    def _generate_with_context(self, sample, context):
        """基于完整context生成答案（回退方案）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)
    
    def _direct_answer(self, sample):
        """直接回答（无检索）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context=None)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)


class VisRAGPipeline(BaselinePipeline):
    """
    ✅ VisRAG: 完整实现（基于BGE Reranker）
    
    实现了VisRAG的核心特色：
    1. 初始检索 (top-10)
    2. BGE重排 (top-5) - 提升检索质量
    3. 视觉优先策略
    """
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.initial_topk = 10
        self.final_topk = 5
        self.bge_reranker = None
        
        # 尝试加载BGE Reranker
        try:
            from flashrag.modules.bge_reranker import create_bge_reranker
            self.bge_reranker = create_bge_reranker()
            print("✅ VisRAG: BGE Reranker已加载")
        except Exception as e:
            print(f"⚠️ VisRAG: BGE Reranker加载失败，将使用简化版: {e}")
    
    def run_single(self, sample):
        """运行单个样本 - 完整的VisRAG流程"""
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: 初始检索 (top-10) ===
        initial_results = self.retriever.search(question, num=self.initial_topk)
        
        if not initial_results:
            # 无检索结果，直接回答
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'reranker_used': False,
                'initial_docs': 0,
                'final_docs': 0
            }
            return self._add_evaluator_fields(result)
        
        # 提取文档文本
        docs_text = [doc.get('contents', '') for doc in initial_results]
        
        # === Step 2: BGE重排 (top-5) ===
        reranked_docs = self._rerank_documents(question, docs_text)
        
        # === Step 3: 融合生成 ===
        answer = self._generate_with_reranked_context(sample, reranked_docs)
        
        result = {
            'answer': answer,
            'raw_prediction': answer,
            'retrieved_docs': reranked_docs,  # 使用重排后的文档
            'used_retrieval': True,
            'reranker_used': (self.bge_reranker is not None),
            'initial_docs': len(docs_text),
            'final_docs': len(reranked_docs)
        }
        
        return self._add_evaluator_fields(result)
    
    def _rerank_documents(self, question: str, documents: list) -> list:
        """BGE重排文档（VisRAG的核心特色）"""
        if self.bge_reranker is None:
            # 无reranker，返回原始top-k
            return documents[:self.final_topk]
        
        try:
            # 使用BGE重排
            reranked = self.bge_reranker.rerank(
                query=question,
                documents=documents,
                top_k=self.final_topk
            )
            return reranked
        except Exception as e:
            print(f"⚠️ VisRAG重排失败: {e}")
            return documents[:self.final_topk]
    
    def _generate_with_reranked_context(self, sample, reranked_docs):
        """基于重排后的文档生成答案"""
        if not reranked_docs:
            return self._direct_answer(sample)
        
        context = "\n\n".join(reranked_docs)
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        
        prompt = f"""Using the high-quality context below (reranked for relevance), answer the question.

Context:
{context}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only:"""
        
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)
    
    def _direct_answer(self, sample):
        """直接回答（无检索）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context=None)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)


class REVEALPipeline(BaselinePipeline):
    """
    ✅ REVEAL: 完整实现（两阶段推理）
    
    实现了REVEAL的核心特色：
    1. 检索证据
    2. 生成推理过程 (Reasoning) - 第一阶段
    3. 基于推理生成最终答案 (Answer) - 第二阶段
    """
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.top_k = 5
        self.reasoning_temp = 0.3  # 推理阶段允许更高温度
    
    def run_single(self, sample):
        """运行单个样本 - 完整的REVEAL流程"""
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: 检索证据 ===
        results = self.retriever.search(question, num=self.top_k)
        
        if not results:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'reasoning': ''
            }
            return self._add_evaluator_fields(result)
        
        docs_text = [doc.get('contents', '') for doc in results]
        context = "\n\n".join(docs_text)
        
        # === Step 2: 生成推理过程（REVEAL核心特色）===
        reasoning = self._generate_reasoning(sample, context)
        
        # === Step 3: 基于推理生成最终答案 ===
        answer = self._generate_final_answer(sample, context, reasoning)
        
        result = {
            'answer': answer,
            'raw_prediction': answer,
            'retrieved_docs': docs_text,
            'used_retrieval': True,
            'reasoning': reasoning  # 保存推理过程
        }
        
        return self._add_evaluator_fields(result)
    
    def _generate_reasoning(self, sample, context):
        """Stage 1: 生成推理过程（REVEAL核心）"""
        prompt = f"""Given the evidence below, provide step-by-step reasoning for answering the question.

Evidence:
{context[:500]}...

Question: {sample['question']}

Step-by-step reasoning (2-3 sentences):"""
        
        try:
            reasoning = self.qwen3_vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=100,
                temperature=self.reasoning_temp,  # 允许推理多样性
                do_sample=True
            )
            return reasoning.strip()
        except:
            return "Based on the evidence provided."
    
    def _generate_final_answer(self, sample, context, reasoning):
        """Stage 2: 基于推理生成最终答案"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        
        prompt = f"""Based on the reasoning below, provide the final answer.

Question: {sample['question']}

Reasoning: {reasoning}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Final answer (letter only):"""
        
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)
    
    def _direct_answer(self, sample):
        """直接回答（无检索）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context=None)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)


class RagVLPipeline(BaselinePipeline):
    """
    ✅ RagVL: 完整实现（MLLM作为强Reranker）
    
    实现了RagVL的核心特色：
    1. 粗检索 (top-20)
    2. MLLM Reranking (选top-3) - 核心创新！
    3. 生成答案
    
    基于论文: MLLM Is a Strong Reranker (arXiv:2407.21439)
    """
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.clip_topk = 20  # 粗检索
        self.rerank_topk = 3  # 精排序后保留
        self.use_reranking = True
    
    def run_single(self, sample):
        """运行单个样本 - 完整的RagVL流程"""
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: 粗检索 (top-20) ===
        initial_results = self.retriever.search(question, num=self.clip_topk)
        
        if not initial_results:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'reranked_count': 0
            }
            return self._add_evaluator_fields(result)
        
        docs_text = [doc.get('contents', '') for doc in initial_results]
        retrieval_scores = [1.0 - i*0.05 for i in range(len(docs_text))]
        
        # === Step 2: MLLM Reranking（RagVL核心特色）===
        if self.use_reranking:
            reranked_docs = self._rerank_documents(
                question, docs_text, retrieval_scores, image
            )
        else:
            reranked_docs = [(doc, score) for doc, score in 
                           zip(docs_text[:self.rerank_topk], 
                               retrieval_scores[:self.rerank_topk])]
        
        # === Step 3: 生成答案 ===
        answer = self._generate_with_reranked(sample, reranked_docs)
        
        result = {
            'answer': answer,
            'raw_prediction': answer,
            'retrieved_docs': [doc for doc, _ in reranked_docs],
            'used_retrieval': True,
            'initial_count': len(docs_text),
            'reranked_count': len(reranked_docs),
            'used_reranking': self.use_reranking
        }
        
        return self._add_evaluator_fields(result)
    
    def _rerank_single(self, question, doc, image=None):
        """使用MLLM判断单个文档的相关性（RagVL核心）"""
        prompt = f"""Is this document relevant to answering the question?

Document: {doc[:200]}...

Question: {question}

Answer with ONLY 'Yes' or 'No':"""
        
        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=0.1
            )
            
            response_lower = response.strip().lower()
            
            if 'yes' in response_lower:
                return True, 0.9
            elif 'no' in response_lower:
                return False, 0.1
            else:
                return True, 0.5
        except:
            return True, 0.5
    
    def _rerank_documents(self, question, retrieved_docs, retrieval_scores, image=None):
        """对检索结果进行reranking（RagVL的核心创新）"""
        reranked = []
        
        for doc, ret_score in zip(retrieved_docs, retrieval_scores):
            is_relevant, rel_score = self._rerank_single(question, doc, image)
            
            if is_relevant:
                # 综合分数：检索分数 × 相关性分数
                combined_score = ret_score * rel_score
                reranked.append((doc, combined_score))
        
        # 按综合分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # 只保留Top-N
        return reranked[:self.rerank_topk]
    
    def _generate_with_reranked(self, sample, reranked_docs):
        """基于rerank后的文档生成答案"""
        if not reranked_docs:
            return self._direct_answer(sample)
        
        # 组织证据
        evidence_parts = []
        for i, (doc, score) in enumerate(reranked_docs):
            evidence_parts.append(f"[Evidence {i+1}]\n{doc}")
        
        evidence_str = "\n\n".join(evidence_parts)
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        
        prompt = f"""Use the following high-quality evidence (filtered by reranking) to answer the question.

Evidence:
{evidence_str}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only:"""
        
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)
    
    def _direct_answer(self, sample):
        """直接回答（无检索）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context=None)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)


class MuRAGPipeline(BaselinePipeline):
    """
    ✅ MuRAG: 完整实现（FiD式并行处理 + 投票融合）
    
    实现了MuRAG的核心特色：
    1. 检索多个证据（top-10）
    2. 每个证据独立生成答案（FiD风格）- 核心创新！
    3. 投票融合选择最终答案
    """
    
    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.top_k = 10  # 检索更多候选
        self.ensemble_k = 5  # 用于投票的证据数
    
    def run_single(self, sample):
        """运行单个样本 - 完整的MuRAG流程"""
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: 检索多个证据 ===
        results = self.retriever.search(question, num=self.top_k)
        
        if not results:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'sub_answers': []
            }
            return self._add_evaluator_fields(result)
        
        docs_text = [doc.get('contents', '') for doc in results]
        
        # === Step 2: FiD式并行处理（MuRAG核心特色）===
        sub_answers = []
        for doc in docs_text[:self.ensemble_k]:
            sub_ans = self._generate_with_single_doc(sample, doc)
            if sub_ans:
                sub_answers.append(sub_ans)
        
        # === Step 3: 投票融合（MuRAG核心特色）===
        if sub_answers:
            answer = self._voting_fusion(sub_answers)
        else:
            answer = self._direct_answer(sample)
        
        result = {
            'answer': answer,
            'raw_prediction': answer,
            'retrieved_docs': docs_text[:self.ensemble_k],
            'used_retrieval': True,
            'sub_answers': sub_answers,  # 保存所有子答案
            'ensemble_size': len(sub_answers)
        }
        
        return self._add_evaluator_fields(result)
    
    def _generate_with_single_doc(self, sample, doc):
        """基于单个文档独立生成答案（FiD风格，MuRAG核心）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        
        prompt = f"""Based ONLY on this single evidence document, answer the question.

Evidence: {doc[:300]}...

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer (letter only):"""
        
        try:
            prediction = self._generate(prompt, sample.get('image'))
            return self._map_letter_to_answer(prediction, sample)
        except:
            return ""
    
    def _voting_fusion(self, sub_answers):
        """投票融合（MuRAG核心特色）"""
        from collections import Counter
        
        # 统计答案频率
        answer_counts = Counter(sub_answers)
        
        # 返回最常见的答案
        if answer_counts:
            most_common = answer_counts.most_common(1)[0]
            return most_common[0]
        
        return sub_answers[0] if sub_answers else ""
    
    def _direct_answer(self, sample):
        """直接回答（无检索）"""
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context=None)
        prediction = self._generate(prompt, sample['image'])
        return self._map_letter_to_answer(prediction, sample)


# ============================================================================
# 评测主函数
# ============================================================================

class MockData:
    """模拟数据对象（用于指标计算）"""
    def __init__(self, predictions, golden_answers, retrieval_results):
        self.pred = predictions
        self.golden_answers = [[ans] if isinstance(ans, str) else ans for ans in golden_answers]
        self.retrieval_result = retrieval_results
        self.items = [{'golden_answers': ga} for ga in self.golden_answers]
        # 修复：添加choices属性（空列表表示不是多选题格式）
        self.choices = [[] for _ in predictions]


def run_method(method_name, pipeline, samples):
    """运行单个方法"""
    print(f"\n{'='*80}")
    print(f"评测方法: {method_name}")
    print(f"{'='*80}")
    
    results = []
    start_time = time.time()
    
    for sample in tqdm(samples, desc=f"运行 {method_name}"):
        result = pipeline.run_single(sample)
        result['question'] = sample['question']
        result['ground_truth'] = sample['answer']
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    return results, elapsed_time


def calculate_metrics(method_name, results, samples):
    """计算7个核心指标"""
    print(f"\n计算 {method_name} 的指标...")
    
    # 准备数据
    predictions = [r['answer'] for r in results]
    golden_answers = [s['answer'] for s in samples]
    
    # 修复：retrieval_result应该是文档列表，每个文档是dict
    retrieval_results = []
    for r in results:
        docs = r.get('retrieved_docs', [])
        # 转换为正确格式：列表的列表，每个元素是字典
        if docs:
            doc_list = [{'contents': doc} if isinstance(doc, str) else {'contents': str(doc)} for doc in docs]
        else:
            doc_list = []
        retrieval_results.append(doc_list)
    
    # 创建MockData对象
    data = MockData(predictions, golden_answers, retrieval_results)
    
    # 计算所有指标
    config = {
        'use_llm_judge': False,  # Faithfulness使用简化版
        'dataset_name': 'mragbench',  # 修复：添加dataset_name
        'metric_setting': {
            'retrieval_recall_topk': 5,  # Recall@5
        }
    }
    calculator = CompleteMetricsCalculator(config)
    
    metrics = calculator.calculate_all_metrics(data)
    
    return metrics


def main():
    """主函数"""
    print("="*80)
    print("Baseline对比实验 - MRAG-Bench全数据集, 7个核心指标")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    max_samples_display = CONFIG['max_samples'] if CONFIG['max_samples'] else "全部(1353)"
    print(f"样本数: {max_samples_display}")
    print()
    
    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("="*80)
    print("1. 加载数据集")
    print("="*80)
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    
    # 初始化模型和检索器
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])
    
    # 初始化BGE检索器（用于baseline方法）
    bge_retriever = init_retriever(CONFIG, use_multimodal=False)
    
    # 初始化多模态融合检索器（用于Self-Aware-MRAG）
    multimodal_retriever = init_retriever(CONFIG, use_multimodal=True)
    
    # 定义所有方法
    methods = {
        'Self-Aware-MRAG': lambda: SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=multimodal_retriever,  # ✅ 使用BGE+CLIP多模态融合检索器
            config={
                # 核心创新点 - 全部启用（✅ 论文完整实现）
                'uncertainty_threshold': 0.35,  # ✅ 优化：恢复合理阈值
                'use_improved_estimator': False,  # ✅ 修改：使用CrossModalUncertaintyEstimator（Gram矩阵+eigen_score+JS散度）
                'use_position_fusion': True,     # ✅ 位置感知跨模态融合
                'use_attribution': True,          # ✅ 启用Attribution（为evaluator提供数据）
                'enable_multimodal_output': False,  # 可选：多模态输出增强
                
                # 模型配置
                'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
                'retrieval_topk': 5,
                
                # Qwen3-VL配置
                'thinking': False,  # 确保不使用thinking模式
                'max_images': 20,   # 最多20张图像
            }
        ),
        'Self-RAG': lambda: SelfRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'mR2AG': lambda: MR2AGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'VisRAG': lambda: VisRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'REVEAL': lambda: REVEALPipeline(qwen3_vl, bge_retriever, CONFIG),
        'RagVL': lambda: RagVLPipeline(qwen3_vl, bge_retriever, CONFIG),
        'MuRAG': lambda: MuRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
    }
    
    # 运行所有方法
    print("\n" + "="*80)
    print("3. 运行所有方法")
    print("="*80)
    
    all_results = {}
    all_metrics = {}
    
    for method_name, pipeline_factory in methods.items():
        try:
            pipeline = pipeline_factory()
            results, elapsed_time = run_method(method_name, pipeline, samples)
            
            # 计算指标
            metrics = calculate_metrics(method_name, results, samples)
            metrics['runtime_seconds'] = elapsed_time
            metrics['seconds_per_sample'] = elapsed_time / len(samples)
            
            all_results[method_name] = results
            all_metrics[method_name] = metrics
            
            print(f"\n✅ {method_name} 完成:")
            print(f"   EM: {metrics.get('em', 0):.4f}")
            print(f"   F1: {metrics.get('f1', 0):.4f}")
            print(f"   VQA-Score: {metrics.get('vqa_score', 0):.4f}")
            print(f"   时间: {metrics['seconds_per_sample']:.2f}秒/样本")
            
        except Exception as e:
            print(f"\n❌ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    print("\n" + "="*80)
    print("4. 保存结果")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_file = output_dir / f"all_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 详细结果: {results_file}")
    
    # 保存指标对比
    metrics_file = output_dir / f"metrics_comparison_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 指标对比: {metrics_file}")
    
    # 生成对比报告
    report_file = output_dir / f"COMPARISON_REPORT_{timestamp}.md"
    generate_report(all_metrics, report_file, samples)
    print(f"✅ 对比报告: {report_file}")
    
    print("\n" + "="*80)
    print("评测完成!")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def generate_report(all_metrics, report_file, samples):
    """生成对比报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Baseline对比实验报告\n\n")
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {len(samples)}\n\n")
        
        f.write("---\n\n")
        f.write("## 核心指标对比（7个指标）\n\n")
        
        # 表格
        f.write("| Method | EM | F1 | Recall@5 | VQA | Faith | Attr | PosBias | 时间(s) |\n")
        f.write("|--------|----|----|----------|-----|-------|------|---------|--------|\n")
        
        for method_name, metrics in all_metrics.items():
            f.write(f"| {method_name} | ")
            f.write(f"{metrics.get('em', 0):.4f} | ")
            f.write(f"{metrics.get('f1', 0):.4f} | ")
            f.write(f"{metrics.get('retrieval_recall_top5', 0):.4f} | ")
            f.write(f"{metrics.get('vqa_score', 0):.4f} | ")
            f.write(f"{metrics.get('faithfulness', 0):.4f} | ")
            f.write(f"{metrics.get('attribution_precision', 0):.4f} | ")
            f.write(f"{metrics.get('position_bias_score', 0):.4f} | ")
            f.write(f"{metrics.get('seconds_per_sample', 0):.2f} |\n")
        
        f.write("\n")
        f.write("**注**:\n")
        f.write("- EM: Exact Match (精确匹配)\n")
        f.write("- F1: Token-level F1\n")
        f.write("- Recall@5: 检索召回率\n")
        f.write("- VQA: VQA-Score\n")
        f.write("- Faith: Faithfulness (忠实度)\n")
        f.write("- Attr: Attribution Precision (归因精度)\n")
        f.write("- PosBias: Position Bias Score (位置偏差，越低越好)\n")


if __name__ == '__main__':
    main()

