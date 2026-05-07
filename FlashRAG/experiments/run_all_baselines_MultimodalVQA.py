#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - MultiModalQA数据集，7个核心指标

方法列表：
1. Self-Aware-MRAG (Our Method)
2. SAM-RAG (替���Self-RAG)
3. mR²AG
4. VisRAG
5. ViDoRAG (替换REVEAL)
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

MultiModalQA特点：
- 支持文本、表格、图像的多模态问答
- 需要结合多种模态的信息来回答问题
- 包含TableQ、TextQ、ImageQ、Compose等多种类型
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
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics
from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline

# 导入RagVL
from experiments.baselines.ragvl_enhanced import RagVLEnhanced
from experiments.baselines.samrag_adapted import SAMRAGAdapted
from experiments.baselines.mr2ag_enhanced import MR2AGFixed
from experiments.baselines.visrag_enhanced import VisRAGEnhanced
from experiments.baselines.murag_enhanced import MuRAGEnhanced


# ============================================================================
# 命令行参数解析
# ============================================================================

import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MultiModalQA Baseline对比实验')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='最大样本数 (默认: 10)')
    parser.add_argument('--dataset_path', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA',
                       help='数据集路径')
    parser.add_argument('--split', type=str, default='dev',
                       choices=['train', 'dev', 'test'],
                       help='数据集划分 (默认: dev)')
    parser.add_argument('--output_dir', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/experiments/results_multimodalqa_baseline',
                       help='输出目录')
    parser.add_argument('--use_dataset_docs', action='store_true',
                       help='使用数据集提供的文档（不进行检索）')
    parser.add_argument('--simple_table', action='store_true',
                       help='使用简化的表格处理（MOQAGPT风格）')
    return parser.parse_args()

# ============================================================================
# 配置
# ============================================================================

# 解析命令行参数
args = parse_args()

CONFIG = {
    # 数据集配置 - MultiModalQA
    'dataset_name': 'multimodalqa',
    'dataset_path': args.dataset_path,
    'split': args.split,
    'max_samples': args.max_samples,  # 使用命令行参数
    'simple_table_processing': args.simple_table,  # 是否使用简化的表格处理

    # 输出配置
    'output_dir': args.output_dir,
    'save_detailed_results': True,
    'save_sample_results': True,
    'enable_complete_metrics': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'max_new_tokens': 50,  # MultiModalQA需要更长的答案
    'temperature': 0.01,

    # 检索器配置
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # CLIP多模态检索配置
    'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
    'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa',  # MultiModalQA专用
    'image_corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/image_corpus.jsonl',  # 明确指定图像语料库
    'use_multimodal_retrieval': True,

    # 多模态检索权重（BGE 60% + CLIP 40%）
    'text_retrieval_weight': 0.6,
    'visual_retrieval_weight': 0.4,

    # 不确定性估计器配置
    'use_improved_estimator': True,
    'uncertainty_threshold': 0.43,

    # 不确定性权重配置
    'text_weight': 0.4,
    'visual_weight': 0.4,
    'alignment_weight': 0.2,

    # Reranker配置
    'use_reranker': True,  # 启用reranker
    'rerank_model_name': 'bge-reranker-v2-m3',
    'rerank_model_path': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
    'rerank_topk': 5,  # rerank后保留的文档数
    'rerank_max_length': 512,  # reranker最大长度
    'rerank_batch_size': 32,  # reranker批次大小（减少以适应GPU内存）
    'rerank_use_fp16': True,  # 使用fp16加速

    # GPU配置
    'use_multi_gpu': False,
    'num_gpus': 1,
    'batch_size_per_gpu': 1,
}


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(dataset_path, split='dev', max_samples=None):
    """加载MultiModalQA数据集 - 参考 multimodalqa-master 数据格式"""
    print(f"加载数据集: MultiModalQA")
    print(f"数据路径: {dataset_path}")
    print(f"数据划分: {split}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    import gzip
    import json

    samples = []

    try:
        # 根据split选择数据文件
        data_file = os.path.join(dataset_path, f'MMQA_{split}.jsonl.gz')

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"找不到数据文件: {data_file}")

        print(f"使用MMQA_{split}.jsonl.gz")

        # 读取数据
        print(f"读取文件: {data_file}")
        with gzip.open(data_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                item = json.loads(line.strip())

                # 提取问题
                question = item.get('question', '')

                # 提取答案
                answers = item.get('answers', [])
                golden_answers = []
                answer_type = 'string'
                modality = ''

                for ans in answers:
                    golden_answers.append(ans.get('answer', ''))
                    answer_type = ans.get('type', 'string')
                    modality = ans.get('modality', '')

                # 获取元数据
                metadata = item.get('metadata', {})
                question_type = metadata.get('type', 'Unknown')
                modalities = metadata.get('modalities', [])

                # 构建样本，添加图像路径
                image_path = None
                if 'image_doc_ids' in metadata and metadata['image_doc_ids']:
                    # 获取第一个图像ID
                    image_id = metadata['image_doc_ids'][0]
                    # 从图像字典中查找路径，支持.jpg和.png扩展名
                    jpg_path = f"/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/images/final_dataset_images/{image_id}.jpg"
                    png_path = f"/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/images/final_dataset_images/{image_id}.png"
                    if os.path.exists(jpg_path):
                        image_path = jpg_path
                    elif os.path.exists(png_path):
                        image_path = png_path
                    else:
                        # 尝试大写.JPG
                        jpg_path_upper = f"/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/images/final_dataset_images/{image_id}.JPG"
                        if os.path.exists(jpg_path_upper):
                            image_path = jpg_path_upper

                sample = {
                    'id': item.get('qid', f'mmqa_{i}'),
                    'question': question,
                    'answer': golden_answers[0] if golden_answers else '',
                    'golden_answers': golden_answers,
                    'question_type': question_type,
                    'modalities': modalities,
                    'answer_type': answer_type,
                    'answer_modality': modality,
                    'metadata': metadata,
                    'image': image_path,  # 添加图像路径
                    # 支持的上下文文档ID
                    'supporting_context': item.get('supporting_context', []),
                    'image_doc_ids': metadata.get('image_doc_ids', []),
                    'text_doc_ids': metadata.get('text_doc_ids', []),
                    'table_id': metadata.get('table_id', ''),
                }

                samples.append(sample)

        print(f"✅ MultiModalQA加载成功: {len(samples)} 样本")

        # 打印类型统计
        from collections import Counter
        type_counter = Counter([s['question_type'] for s in samples])
        modality_counter = Counter([','.join(s['modalities']) for s in samples])
        print(f"问题类型分布: {dict(type_counter)}")
        print(f"模态分布: {dict(modality_counter)}")

        return samples

    except Exception as e:
        print(f"MultiModalQA加载失败: {e}")
        import traceback
        traceback.print_exc()

        # 使用示例数据
        print("使用示例数据进行测试...")
        samples = [
            {
                'id': 'mmqa_sample_0',
                'question': 'For which film did Ben Piazza play the role of Mr. Simms?',
                'answer': 'Mask',
                'golden_answers': ['Mask'],
                'question_type': 'TableQ',
                'modalities': ['table'],
                'answer_type': 'string',
                'answer_modality': 'table',
                'metadata': {'type': 'TableQ', 'modalities': ['table']},
                'supporting_context': [],
                'image_doc_ids': [],
                'text_doc_ids': [],
                'table_id': 'sample_table',
            },
            {
                'id': 'mmqa_sample_1',
                'question': 'What sports is shown in the image?',
                'answer': 'baseball',
                'golden_answers': ['baseball'],
                'question_type': 'ImageQ',
                'modalities': ['image'],
                'answer_type': 'string',
                'answer_modality': 'image',
                'metadata': {'type': 'ImageQ', 'modalities': ['image']},
                'supporting_context': [],
                'image_doc_ids': ['sample_img'],
                'text_doc_ids': [],
                'table_id': '',
            }
        ]

        if max_samples:
            samples = samples[:max_samples]

        print(f"✅ 使用示例数据: {len(samples)} 样本")
        return samples


# ============================================================================
# MultiModalQA官方评测函数
# ============================================================================

def normalize_answer(s):
    """来自MultiModalQA官方评测的答案标准化"""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """计算Exact Match分数"""
    if isinstance(ground_truth, list):
        return any(normalize_answer(prediction) == normalize_answer(gt) for gt in ground_truth)
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    """计算F1分数"""
    if isinstance(ground_truth, list):
        # 多个答案时取最大值
        scores = [f1_score(prediction, gt) for gt in ground_truth]
        return max(scores)

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# ============================================================================
# 模型和检索器初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    print(f"初始化Qwen3-VL: {model_path}")
    print(f"正在加载Qwen3-VL-8B-Instruct: {model_path}")
    print(f"设备: cuda, 8bit量化: False, dtype: {CONFIG['torch_dtype']}")

    wrapper = create_qwen3_vl_wrapper(
        model_path=model_path,
        device="cuda",
        torch_dtype=CONFIG['torch_dtype']
    )

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
        'use_reranker': config.get('use_reranker', False),
        'rerank_model_name': config.get('rerank_model_name'),
        'rerank_model_path': config.get('rerank_model_path'),
        'rerank_topk': config.get('rerank_topk', 5),
        'rerank_max_length': config.get('rerank_max_length', 512),
        'rerank_batch_size': config.get('rerank_batch_size', 32),
        'rerank_use_fp16': config.get('rerank_use_fp16', True),
        'device': 'cuda',  # 添加device参数
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
        'retrieval_cache_path': None,
        'index_modal': 'all',  # CLIP索引包含text+image
        'faiss_gpu': False,
        'retrieval_pooling_method': 'mean',
        'use_reranker': False,
        'instruction': '',
        'use_sentence_transformer': False,
    }
    
    clip_retriever = DenseRetriever(clip_retriever_config)
    print("✅ CLIP视觉检索器加载成功")
    
    # 创建多模态融合检索器
    from flashrag.retriever.multimodal_retriever import SelfAwareMultimodalRetriever
    
    multimodal_config = {
        'retrieval_topk': config['retrieval_topk'],
        'use_clip': True,
        'clip_model_path': config.get('clip_model_path', '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336'),
        'clip_index_path': config.get('clip_index_path'),
        'image_corpus_path': config.get('image_corpus_path'),
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


class SAMRAGPipeline(BaselinePipeline):
    """
    ✅ SAM-RAG: 完整实现（基于Qwen3-VL）

    参考论文: Self-adaptive Multimodal Retrieval-Augmented Generation
    实现了SAM-RAG的核心特色：
    1. Batch Retrieval: 批次检索，找到相关内容后停止
    2. Relevance Judgment: 判断文档是否相关 (isRel)
    3. Answer Quality: 评估答案质量 (isSup + isUse)
    4. Adaptive Iteration: 自适应迭代检索
    """

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.decision_temp = 0.05  # 判断温度（低温度=更确定）
        self.batch_size = config.get('sam_batch_size', 5)  # 每批检索的文档数
        self.max_batches = config.get('sam_max_batches', 4)  # 最多检索批次
    
    def run_single(self, sample):
        """运行单个样本 - 完整的SAM-RAG批次检索流程"""
        question = sample['question']
        image = sample.get('image')

        # === Step 1: 检索所有文档 ===
        total_docs = self.batch_size * self.max_batches
        all_results = self.retriever.search(question, num=total_docs)

        if not all_results:
            answer = self._direct_answer(sample)
            result = {
                'answer': answer,
                'raw_prediction': answer,
                'retrieved_docs': [],
                'used_retrieval': False,
                'num_batches_used': 0,
                'relevant_docs_count': 0,
                'support_status': 'N/A',
                'usefulness_status': 'N/A'
            }
            return self._add_evaluator_fields(result)

        # === Step 2: 批次处理 ===
        relevant_contents = []
        relevant_ids = []
        answer = None
        final_batch = 0

        for batch_idx in range(self.max_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_results))
            batch_docs = all_results[start_idx:end_idx]

            if not batch_docs:
                break

            # 2.1 判断相关性
            for doc in batch_docs:
                doc_text = doc.get('contents', '')
                if self._relevance_judgment(question, doc_text, image):
                    relevant_contents.append(doc_text)
                    relevant_ids.append(doc.get('id', f"doc_{len(relevant_ids)}"))

            # 2.2 如果找到相关内容，尝试回答
            if relevant_contents:
                answer = self._generate_with_context_simple(question, relevant_contents, image)

                # 2.3 评估答案质量
                is_supported = self._support_judgment_samrag(question, answer, relevant_contents)
                is_useful = self._usefulness_judgment(question, answer, relevant_contents)

                # 2.4 如果答案满足条件，返回
                if is_supported == 'True' and is_useful:
                    final_batch = batch_idx + 1
                    break

                # 2.5 如果不满足，继续下一批
                if not is_useful:
                    # 重新生成答案
                    answer = self._generate_with_context_simple(question, relevant_contents, image)
                    is_useful = self._usefulness_judgment(question, answer, relevant_contents)

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

        # === Step 3: 返回最终答案 ===
        if not answer:
            if relevant_contents:
                answer = self._generate_with_context_simple(question, relevant_contents, image)
            else:
                answer = self._direct_answer(sample)

        result = {
            'answer': answer,
            'raw_prediction': answer,
            'retrieved_docs': relevant_contents,
            'used_retrieval': len(relevant_contents) > 0,
            'num_batches_used': final_batch,
            'relevant_docs_count': len(relevant_contents),
            'support_status': is_supported if answer and relevant_contents else 'N/A',
            'usefulness_status': is_useful if answer and relevant_contents else 'N/A'
        }

        return self._add_evaluator_fields(result)
    
    def _relevance_judgment(self, question: str, document: str, image=None) -> bool:
        """判断文档是否相关（SAM-RAG的isRel判断）"""
        doc_preview = document[:300] + "..." if len(document) > 300 else document

        prompt = f"""Determine if this text is related to the question.

Content: {doc_preview}
Question: {question}

Is this text related to answering the question?
Answer ONLY 'True' or 'False':"""

        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,  # 纯文本判断
                max_new_tokens=10,
                temperature=self.decision_temp
            )

            response_lower = response.strip().lower()
            return 'true' in response_lower and 'false' not in response_lower
        except:
            return True  # 默认相关（保守）

    def _support_judgment_samrag(self, question: str, answer: str, documents: list) -> str:
        """
        判断答案是否被文档支持（SAM-RAG的isSup判断）

        Returns:
            'True': 完全支持
            'Partial': 部分支持
            'False': 不支持
        """
        context = "\n\n".join(documents[:5])[:500]

        prompt = f"""Determine if the answer is supported by the content.

Content: {context}
Question: {question}
Answer: {answer}

Is the answer fully supported by the content?
- "True": The answer is fully supported
- "Partial": The answer is partially supported
- "False": The answer is not supported

Answer ONLY 'True', 'Partial', or 'False':"""

        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=10,
                temperature=self.decision_temp
            )

            response_lower = response.strip().lower()
            if 'partial' in response_lower:
                return 'Partial'
            elif 'true' in response_lower:
                return 'True'
            else:
                return 'False'
        except:
            return 'True'  # 默认支持（保守）

    def _usefulness_judgment(self, question: str, answer: str, documents: list) -> bool:
        """判断答案是否正确使用了内容（SAM-RAG的isUse判断）"""
        context = "\n\n".join(documents[:5])[:500]

        prompt = f"""Determine if the answer correctly uses the content to answer the question.

Content: {context}
Question: {question}
Answer: {answer}

Is the answer appropriate and correctly uses the content?
Answer ONLY 'True' or 'False':"""

        try:
            response = self.qwen3_vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=10,
                temperature=self.decision_temp
            )

            response_lower = response.strip().lower()
            return 'true' in response_lower and 'false' not in response_lower
        except:
            return True  # 默认有用（保守）

    def _generate_with_context_simple(self, question: str, relevant_docs: list, image=None) -> str:
        """基于相关文档生成答案（简化版）"""
        context = "\n\n".join(relevant_docs[:10])

        prompt = f"""Based on the following content, answer the question concisely.

Content:
{context}

Question: {question}

Provide a direct and concise answer:"""

        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=50,
                temperature=0.1
            )
            return answer.strip()
        except:
            return ""

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


# REVEAL Pipeline replaced by ViDoRAG
# class REVEALPipeline(BaselinePipeline):
#     """
#     �� REVEAL: 完整实现（两阶段推理）
#
#     实现了REVEAL的核心特色：
#     1. 检索证据
#     2. 生成推理过程 (Reasoning) - 第一阶段
#     3. 基于推理生成最终答案 (Answer) - 第二阶段
#     """
#
#     def __init__(self, qwen3_vl, retriever, config):
#         super().__init__(qwen3_vl, retriever, config)
#         self.top_k = 5
#         self.reasoning_temp = 0.3  # 推理阶段允许更高温度
#     
#     def run_single(self, sample):
#         """运行单个样本 - 完整的REVEAL流程"""
#         question = sample['question']
#         image = sample.get('image')
#         
#         # === Step 1: 检索证据 ===
#         results = self.retriever.search(question, num=self.top_k)
#         
#         if not results:
#             answer = self._direct_answer(sample)
#             result = {
#                 'answer': answer,
#                 'raw_prediction': answer,
#                 'retrieved_docs': [],
#                 'used_retrieval': False,
#                 'reasoning': ''
#             }
#             return self._add_evaluator_fields(result)
#         
#         docs_text = [doc.get('contents', '') for doc in results]
#         context = "\n\n".join(docs_text)
#         
#         # === Step 2: 生成推理过程（REVEAL核心特色）===
#         reasoning = self._generate_reasoning(sample, context)
#         
#         # === Step 3: 基于推理生成最终答案 ===
#         answer = self._generate_final_answer(sample, context, reasoning)
#         
#         result = {
#             'answer': answer,
#             'raw_prediction': answer,
#             'retrieved_docs': docs_text,
#             'used_retrieval': True,
#             'reasoning': reasoning  # 保存推理过程
#         }
#         
#         return self._add_evaluator_fields(result)
#     
#     def _generate_reasoning(self, sample, context):
#         """Stage 1: 生成推理过程（REVEAL核心）"""
#         prompt = f"""Given the evidence below, provide step-by-step reasoning for answering the question.
# 
# Evidence:
# {context[:500]}...
# 
# Question: {sample['question']}
# 
# Step-by-step reasoning (2-3 sentences):"""
#         
#         try:
#             reasoning = self.qwen3_vl.generate(
#                 text=prompt,
#                 image=sample.get('image'),
#                 max_new_tokens=100,
#                 temperature=self.reasoning_temp,  # 允许推理多样性
#                 do_sample=True
#             )
#             return reasoning.strip()
#         except:
#             return "Based on the evidence provided."
#     
#     def _generate_final_answer(self, sample, context, reasoning):
#         """Stage 2: 基于推理生成最终答案"""
#         options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
#         
#         prompt = f"""Based on the reasoning below, provide the final answer.
# 
# Question: {sample['question']}
# 
# Reasoning: {reasoning}
# 
# Choices:
# A. {sample['A']}
# B. {sample['B']}
# C. {sample['C']}
# D. {sample['D']}
# 
# Final answer (letter only):"""
#         
#         prediction = self._generate(prompt, sample['image'])
#         return self._map_letter_to_answer(prediction, sample)
#     
#     def _direct_answer(self, sample):
#         """直接回答（无检索）"""
#         options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
#         prompt = self._construct_prompt(sample['question'], options, context=None)
#         prediction = self._generate(prompt, sample['image'])
#         return self._map_letter_to_answer(prediction, sample)
# 

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
    """计算指标 - 结合MultiModalQA官方评测和FlashRAG的7个核心指标"""
    print(f"\n计算 {method_name} 的指标...")

    # ========== MultiModalQA官方评测 ==========
    # 计算准确率、EM、F1
    correct = 0
    total = len(results)
    em_scores = []
    f1_scores = []

    for i, (r, s) in enumerate(zip(results, samples)):
        pred = r.get('answer', '')
        gt = s.get('answer', '')
        golden_answers = s.get('golden_answers', [gt])

        # 计算准确率
        if gt and exact_match_score(pred, golden_answers):
            correct += 1

        # 计算EM和F1
        if golden_answers:
            em_scores.append(float(exact_match_score(pred, golden_answers)))
            f1_scores.append(f1_score(pred, golden_answers))

    accuracy = correct / total if total > 0 else 0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    print(f"  ✅ MultiModalQA官方评测:")
    print(f"    - Accuracy: {accuracy:.4f}")
    print(f"    - EM (Exact Match): {avg_em:.4f}")
    print(f"    - F1: {avg_f1:.4f}")

    # ========== FlashRAG的7个核心指标评估器 ==========
    # 准备数据以匹配评估器格式
    formatted_results = []
    for i, r in enumerate(results):
        formatted_result = {
            'answer': r.get('answer', ''),
            'golden_answers': samples[i].get('golden_answers', [samples[i].get('answer', '')]),
            'retrieved_docs': r.get('retrieved_docs', []),
            'question': samples[i].get('question', ''),
            'id': samples[i].get('id', f'sample_{i}'),
        }

        # 转换retrieved_docs为标准格式
        docs = r.get('retrieved_docs', [])
        if docs:
            formatted_result['retrieved_docs'] = [
                {'contents': doc} if isinstance(doc, str) else {'contents': str(doc)}
                for doc in docs
            ]
        else:
            formatted_result['retrieved_docs'] = []

        formatted_results.append(formatted_result)

    # 使用综合评估器计算其他指标
    try:
        metrics = evaluate_comprehensive_metrics(formatted_results)

        unified_metrics = {
            'method': method_name,
            # MultiModalQA官方指标
            'accuracy': accuracy,
            'em': avg_em,
            'f1': avg_f1,
            # FlashRAG核心指标
            'retrieval_rate': metrics.get('retrieval_rate', 0),
            'retrieval_recall_top5': metrics.get('avg_Recall@5', 0),
            'vqa_score': metrics.get('vqa_score', 0),
            'faithfulness': metrics.get('avg_Faithfulness', 0),
            'attribution_precision': metrics.get('avg_Attribution_Precision', 0),
            'position_bias_score': metrics.get('avg_Position_Bias_Score', 0),
        }

        # 打印所有指标
        print(f"  ✅ FlashRAG核心指标:")
        print(f"    - Retrieval Rate: {unified_metrics['retrieval_rate']:.4f}")
        print(f"    - Recall@5: {unified_metrics['retrieval_recall_top5']:.4f}")
        print(f"    - VQA Score: {unified_metrics['vqa_score']:.4f}")
        print(f"    - Faithfulness: {unified_metrics['faithfulness']:.4f}")
        print(f"    - Attribution Precision: {unified_metrics['attribution_precision']:.4f}")
        print(f"    - Position Bias Score: {unified_metrics['position_bias_score']:.4f}")

        return unified_metrics

    except Exception as e:
        print(f"  ❌ FlashRAG指标计算失败: {e}")
        # 返回默认指标
        return {
            'method': method_name,
            'accuracy': 0.0,
            'retrieval_rate': 0.0,
            'avg_F1': 0.0,
            'avg_Recall@5': 0.0,
            'avg_Faithfulness': 0.0,
            'avg_Attribution_Precision': 0.0,
            'avg_Position_Bias_Score': 0.0
        }


def main():
    """主函数"""
    print("="*80)
    print("Baseline对比实验 - MultiModalQA 10样本, 7个核心指标")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    max_samples_display = CONFIG['max_samples'] if CONFIG['max_samples'] else "全部"
    print(f"样本数: {max_samples_display}")
    print(f"数据集: {CONFIG['dataset_name'].upper()}")
    print(f"数据划分: {CONFIG['split']}")
    print()
    
    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("="*80)
    print("1. 加载数据集")
    print("="*80)
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['split'], CONFIG['max_samples'])
    
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
                # 核心创新点 - 全部启用（✅ 优化C: threshold+文档过滤+visual优化+答案验证）
                'uncertainty_threshold': 0.35,  # ✅ 降低阈值以鼓励更多检索
                'use_improved_estimator': True,  # ✅ 使用改进版不确定性估计器
                'use_position_fusion': True,     # ✅ 位置感知跨模态融合
                'use_attribution': True,          # ✅ 启用Attribution（为evaluator提供数据）
                'enable_multimodal_output': False,  # 可选：多模态输出增强

                # 新增：使用数据集提供的文档（MultiModalQA）
                'use_dataset_docs': args.use_dataset_docs,  # 使用命令行参数控制

                # 新增：简化表格处理（MOQAGPT风格）
                'simple_table_processing': args.simple_table,  # 使用命令行参数控制

                # 模型配置
                'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
                'retrieval_topk': 5,

                # Qwen3-VL配置
                'thinking': False,
                'max_images': 20,
                'temperature': CONFIG['temperature'],
                'max_new_tokens': CONFIG['max_new_tokens'],  # 使用配置中的值（已设为50）
            }
        ),
        'SAM-RAG': lambda: SAMRAGAdapted(qwen3_vl, bge_retriever, {
            **CONFIG,
            'sam_batch_size': 5,  # 每批检索5个文档
            'sam_max_batches': 4,  # 最多4批（总共20个文档）
        }),
        'mR2AG': lambda: MR2AGFixed(qwen3_vl, bge_retriever, CONFIG),
        'VisRAG': lambda: VisRAGEnhanced(qwen3_vl, bge_retriever, CONFIG),
        'ViDoRAG': lambda: create_vidorag_pipeline(qwen3_vl, bge_retriever, CONFIG),
        'RagVL': lambda: RagVLEnhanced(qwen3_vl, None, {**CONFIG, **{  # 传入None作为retriever
            'use_reranking': False,
            'rerank_topk': 0,  # 不检索任何文档
            'clip_topk': 0,
            'no_retrieval': True  # 标记为不检索模式
        }}),
        'MuRAG': lambda: MuRAGEnhanced(qwen3_vl, bge_retriever, CONFIG),
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
    """生成MultiModalQA对比报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MultiModalQA Baseline对比实验报告\n\n")
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {len(samples)}\n\n")

        # 统计问题类型和模态分布
        from collections import Counter
        question_types = Counter([s.get('question_type', 'Unknown') for s in samples])
        modalities = Counter([','.join(s.get('modalities', [])) for s in samples])

        f.write("## 数据集统计\n\n")
        f.write("### 问题类型分布\n\n")
        for qtype, count in question_types.items():
            f.write(f"- {qtype}: {count} 样本\n")

        f.write("\n### 模态分布\n\n")
        for mod, count in modalities.items():
            f.write(f"- {mod}: {count} 样本\n")
        f.write("\n")

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

        # MultiModalQA特有分析
        f.write("\n---\n\n")
        f.write("## MultiModalQA特性分析\n\n")

        # 按准确率排序
        sorted_methods = sorted(all_metrics.items(),
                               key=lambda x: x[1].get('em', 0),
                               reverse=True)

        f.write("### 方法排名\n\n")
        for i, (method_name, metrics) in enumerate(sorted_methods, 1):
            em_score = metrics.get('em', 0) * 100
            retrieval_rate = metrics.get('retrieval_rate', 0) * 100
            f.write(f"{i}. **{method_name}**: EM={em_score:.2f}%, 检索率={retrieval_rate:.1f}%\n")

        f.write("\n### 关键发现\n\n")
        best_method = sorted_methods[0][0]
        f.write(f"- **最佳方法**: {best_method}\n")
        f.write(f"- MultiModalQA需要处理多种模态信息（文本、表格、图像）\n")
        f.write(f"- 包含复合问题类型（Compose）需要多步推理\n")
        f.write(f"- 检索系统需要能够跨模态检索相关信息\n")


if __name__ == '__main__':
    main()

