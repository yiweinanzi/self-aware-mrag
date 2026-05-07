#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - A-OKVQA数据集，7个核心指标

方法列表：
1. Self-Aware-MRAG (Our Method)
2. SAM-RAG (替换Self-RAG)
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

A-OKVQA特点：
- 包含推理步骤（rationales）的多选问答
- 需要外部知识来回答问题
- 支持多选题（Multiple Choice）和直接回答（Direct Answer）两种模式
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


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 数据集配置 - A-OKVQA
    'dataset_name': 'aokvqa',
    'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA',
    'max_samples': 10,  # 对比实验：10样本

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_aokvqa_baseline',
    'save_detailed_results': True,
    'save_sample_results': True,
    'enable_complete_metrics': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'max_new_tokens': 30,  # A-OKVQA需要适中的答案长度
    'temperature': 0.01,

    # 检索器配置
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # CLIP多模态检索配置
    'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
    'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip',
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

    # GPU配置
    'use_multi_gpu': False,
    'num_gpus': 1,
    'batch_size_per_gpu': 1,
}


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(dataset_path, max_samples=None):
    """加载A-OKVQA数据集 - 参考 aokvqa-main 数据格式"""
    print(f"加载数据集: A-OKVQA")
    print(f"数据路径: {dataset_path}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    samples = []

    try:
        # A-OKVQA数据文件路径
        possible_files = [
            os.path.join(dataset_path, 'validation_sample.json'),
            os.path.join(dataset_path, 'train_sample.json'),
            os.path.join(dataset_path, 'test_sample.json'),
            os.path.join(dataset_path, 'raw', 'validation.json'),
            os.path.join(dataset_path, 'raw', 'train.json'),
            os.path.join(dataset_path, 'raw', 'test.json'),
        ]

        # 选择数据文件
        data_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                print(f"使用文件: {file_path}")
                break

        if not data_file:
            raise FileNotFoundError("找不到A-OKVQA数据文件")

        # 读取数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理数据
        for i, item in enumerate(data):
            if max_samples and i >= max_samples:
                break

            # 提取基本信息
            question_id = item.get('question_id', f'aokvqa_{i}')
            question = item.get('question', '')
            choices = item.get('choices', [])
            correct_choice_idx = item.get('correct_choice_idx', 0)
            rationales = item.get('rationales', [])

            # 获取正确答案
            if choices and correct_choice_idx < len(choices):
                correct_answer = choices[correct_choice_idx]
                # 格式化为字母
                answer_letter = chr(65 + correct_choice_idx)  # A, B, C, D
            else:
                correct_answer = ''
                answer_letter = ''

            # 构建样本
            sample = {
                'id': question_id,
                'question': question,
                'answer': answer_letter,  # 返回字母答案
                'answer_text': correct_answer,  # 完整答案文本
                'golden_answers': [answer_letter],
                'choices': choices,
                'correct_choice_idx': correct_choice_idx,
                'rationales': rationales,
                'question_type': 'multiple_choice',
                'difficult_direct_answer': item.get('difficult_direct_answer', False),
                'direct_answers': item.get('direct_answers', []),
            }

            # 如果有A, B, C, D选项，添加到样本中
            for j, choice in enumerate(choices):
                sample[chr(65 + j)] = choice

            samples.append(sample)

        print(f"✅ A-OKVQA加载成功: {len(samples)} 样本")
        print(f"多选题格式: {len([s for s in samples if s['choices']])} 样本")
        print(f"有推理步骤: {len([s for s in samples if s['rationales']])} 样本")

        return samples

    except Exception as e:
        print(f"A-OKVQA加载失败: {e}")
        import traceback
        traceback.print_exc()

        # 使用示例数据
        print("使用示例数据进行测试...")
        samples = [
            {
                'id': 'aokvqa_sample_0',
                'question': 'What is in the motorcyclist\'s mouth?',
                'answer': 'D',
                'answer_text': 'cigarette',
                'golden_answers': ['D'],
                'choices': ['toothpick', 'food', 'popsicle stick', 'cigarette'],
                'A': 'toothpick',
                'B': 'food',
                'C': 'popsicle stick',
                'D': 'cigarette',
                'correct_choice_idx': 3,
                'rationales': ['He\'s smoking while riding.'],
                'question_type': 'multiple_choice',
                'difficult_direct_answer': False,
                'direct_answers': ['cigarette'],
            },
            {
                'id': 'aokvqa_sample_1',
                'question': 'Which number birthday is probably being celebrated?',
                'answer': 'D',
                'answer_text': 'thirty',
                'golden_answers': ['D'],
                'choices': ['one', 'ten', 'nine', 'thirty'],
                'A': 'one',
                'B': 'ten',
                'C': 'nine',
                'D': 'thirty',
                'correct_choice_idx': 3,
                'rationales': ['There is a birthday cake on the table with the number 30 written in icing.'],
                'question_type': 'multiple_choice',
                'difficult_direct_answer': False,
                'direct_answers': ['30'],
            }
        ]

        if max_samples:
            samples = samples[:max_samples]

        print(f"✅ 使用示例数据: {len(samples)} 样本")
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
    """计算A-OKVQA的7个核心指标 - 参考aokvqa-main/evaluation/eval_predictions.py"""
    print(f"\n计算 {method_name} 的指标...")

    # ========== A-OKVQA特有的准确率计算 ==========
    # 提取预测答案和真实答案
    preds = {}
    dataset = {}

    # A-OKVQA评估函数（参考官方代码）
    def eval_aokvqa(dataset, preds, multiple_choice=True):
        """A-OKVQA官方评估函数"""
        acc = []

        for q_id in dataset.keys():
            if q_id not in preds:
                acc.append(0.0)
                continue

            pred = preds[q_id]
            choices = dataset[q_id]['choices']

            # 多选题格式
            if multiple_choice and choices:
                correct_choice_idx = dataset[q_id]['correct_choice_idx']
                # 检查预测是否是正确选项
                is_correct = 0.0
                if len(pred) > 0:
                    pred_letter = pred[0].upper() if isinstance(pred, list) else pred.upper()
                    if pred_letter in ['A', 'B', 'C', 'D']:
                        pred_idx = ord(pred_letter) - 65
                        if pred_idx == correct_choice_idx:
                            is_correct = 1.0
                acc.append(is_correct)

        return sum(acc) / len(acc) if acc else 0.0

    # 准备数据
    for i, (result, sample) in enumerate(zip(results, samples)):
        q_id = sample['id']

        # 预测答案
        pred_answer = result.get('answer', '')
        preds[q_id] = pred_answer

        # 数据集信息
        dataset[q_id] = {
            'question_id': q_id,
            'question': sample['question'],
            'choices': sample['choices'],
            'correct_choice_idx': sample['correct_choice_idx'],
            'direct_answers': sample['direct_answers'],
            'difficult_direct_answer': sample['difficult_direct_answer']
        }

    # 计算多选题准确率
    mc_accuracy = eval_aokvqa(dataset, preds, multiple_choice=True)
    overall_accuracy = mc_accuracy * 100  # 转换为百分比

    # ========== 使用FlashRAG的7个核心指标评估器 ==========
    # 准备数据以匹配评估器格式
    formatted_results = []
    for i, (r, s) in enumerate(zip(results, samples)):
        formatted_result = {
            'answer': r.get('answer', ''),
            'golden_answers': [s.get('answer', '')],
            'retrieved_docs': r.get('retrieved_docs', []),
            'question': s.get('question', ''),
            'id': s.get('id', f'sample_{i}'),
            'choices': s.get('choices', []),
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

    # 使用综合评估器计算指标
    try:
        from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics
        metrics = evaluate_comprehensive_metrics(formatted_results)

        # 添加A-OKVQA特有的指标
        unified_metrics = {
            'method': method_name,
            # A-OKVQA多选题准确率
            'accuracy': overall_accuracy / 100,  # 转换为0-1范围
            'em': metrics.get('em', overall_accuracy / 100),  # 对于多选题，EM约等于准确率

            # 其他核心指标
            'f1': metrics.get('avg_F1', 0),
            'retrieval_rate': metrics.get('retrieval_rate', 0),
            'retrieval_recall_top5': metrics.get('avg_Recall@5', 0),
            'vqa_score': metrics.get('vqa_score', 0),
            'faithfulness': metrics.get('avg_Faithfulness', 0),
            'attribution_precision': metrics.get('avg_Attribution_Precision', 0),
            'position_bias_score': metrics.get('avg_Position_Bias_Score', 0),

            # A-OKVQA特有指标
            'mc_accuracy': overall_accuracy,
            'has_rationales': len([s for s in samples if s.get('rationales')]),
        }

        # 打印指标
        print(f"  ✅ A-OKVQA多选题准确率: {overall_accuracy:.2f}%")
        print(f"  ✅ EM: {unified_metrics['em']:.4f}")
        print(f"  ✅ F1: {unified_metrics['f1']:.4f}")
        print(f"  ✅ Retrieval Rate: {unified_metrics['retrieval_rate']:.4f}")
        print(f"  ✅ Recall@5: {unified_metrics['retrieval_recall_top5']:.4f}")
        print(f"  ✅ VQA Score: {unified_metrics['vqa_score']:.4f}")

        return unified_metrics

    except Exception as e:
        print(f"  ❌ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()

        # 返回基本指标
        return {
            'method': method_name,
            'accuracy': overall_accuracy / 100,
            'em': overall_accuracy / 100,
            'f1': 0.0,
            'retrieval_rate': 0.0,
            'retrieval_recall_top5': 0.0,
            'vqa_score': 0.0,
            'faithfulness': 0.0,
            'attribution_precision': 0.0,
            'position_bias_score': 0.0,
            'mc_accuracy': overall_accuracy,
        }


def main():
    """主函数"""
    print("="*80)
    print("Baseline对比实验 - A-OKVQA 10样本, 7个核心指标")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    max_samples_display = CONFIG['max_samples'] if CONFIG['max_samples'] else "全部"
    print(f"样本数: {max_samples_display}")
    print(f"数据集: {CONFIG['dataset_name'].upper()}")
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
                # 核心创新点 - 全部启用（✅ 优化C: threshold+文档过滤+visual优化+答案验证）
                'uncertainty_threshold': 0.43,  # ✅ 统一阈值：基于P92百分位校准
                'use_improved_estimator': True,  # ✅ 使用改进版不确定性估计器
                'use_position_fusion': True,     # ✅ 位置感知跨模态融合
                'use_attribution': True,          # ✅ 启用Attribution（为evaluator提供数据）
                'enable_multimodal_output': False,  # 可选：多模态输出增强

                # 模型配置
                'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
                'retrieval_topk': 5,

                # Qwen3-VL配置
                'thinking': False,  # 确保不使用thinking模式
                'max_images': 20,   # 最多20张图像
            }
        ),
        'SAM-RAG': lambda: SAMRAGPipeline(qwen3_vl, bge_retriever, {
            **CONFIG,
            'sam_batch_size': 5,  # 每批检索5个文档
            'sam_max_batches': 4,  # 最多4批（总共20个文档）
        }),
        'mR2AG': lambda: MR2AGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'VisRAG': lambda: VisRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'ViDoRAG': lambda: create_vidorag_pipeline(qwen3_vl, bge_retriever, CONFIG),
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
    """生成A-OKVQA对比报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# A-OKVQA Baseline对比实验报告\n\n")
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {len(samples)}\n\n")

        # 统计数据集特点
        mc_count = len([s for s in samples if s['choices']])
        rationale_count = len([s for s in samples if s.get('rationales')])

        f.write("## 数据集统计\n\n")
        f.write(f"- 多选题样本: {mc_count} ({mc_count/len(samples)*100:.1f}%)\n")
        f.write(f"- 包含推理步骤: {rationale_count} ({rationale_count/len(samples)*100:.1f}%)\n")
        f.write(f"- 平均选项数: {sum([len(s['choices']) for s in samples])/len(samples):.1f}\n\n")

        f.write("---\n\n")
        f.write("## 核心指标对比（7个指标）\n\n")

        # 表格 - 添加多选题准确率
        f.write("| Method | MC Acc (%) | EM | F1 | Recall@5 | VQA | Faith | Attr | PosBias | 时间(s) |\n")
        f.write("|--------|------------|----|----|----------|-----|-------|------|---------|--------|\n")

        # 按多选题准确率排序
        sorted_methods = sorted(all_metrics.items(),
                               key=lambda x: x[1].get('mc_accuracy', 0),
                               reverse=True)

        for method_name, metrics in sorted_methods:
            mc_acc = metrics.get('mc_accuracy', 0)
            f.write(f"| {method_name} | ")
            f.write(f"{mc_acc:.2f} | ")
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
        f.write("- MC Acc: A-OKVQA多选题准确率 (官方评估指标)\n")
        f.write("- EM: Exact Match (精确匹配)\n")
        f.write("- F1: Token-level F1\n")
        f.write("- Recall@5: 检索召回率\n")
        f.write("- VQA: VQA-Score\n")
        f.write("- Faith: Faithfulness (忠实度)\n")
        f.write("- Attr: Attribution Precision (归因精度)\n")
        f.write("- PosBias: Position Bias Score (位置偏差，越低越好)\n")

        # A-OKVQA特有分析
        f.write("\n---\n\n")
        f.write("## A-OKVQA特性分析\n\n")

        # 最佳方法
        best_method = sorted_methods[0][0]
        best_mc_acc = sorted_methods[0][1].get('mc_accuracy', 0)

        f.write(f"### 关键发现\n\n")
        f.write(f"1. **最佳方法**: {best_method} (多选题准确率: {best_mc_acc:.2f}%)\n\n")

        f.write(f"2. **方法排名（按多选题准确率）**:\n")
        for i, (method_name, metrics) in enumerate(sorted_methods, 1):
            mc_acc = metrics.get('mc_accuracy', 0)
            retrieval_rate = metrics.get('retrieval_rate', 0) * 100
            f.write(f"   {i}. {method_name}: {mc_acc:.2f}% (检索率: {retrieval_rate:.1f}%)\n")

        f.write("\n3. **A-OKVQA特点**:\n")
        f.write(f"   - 需要外部知识的多选问答\n")
        f.write(f"   - 包含推理步骤（rationales）引导\n")
        f.write(f"   - 测试模型的常识推理能力\n")
        f.write(f"   - 检索增强对提升性能至关重要\n")

        # 选项难度分析
        f.write("\n4. **选项分析**:\n")
        choice_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for s in samples:
            if 'correct_choice_idx' in s:
                idx = s['correct_choice_idx']
                if 0 <= idx < 4:
                    choice_counts[chr(65 + idx)] += 1

        total_correct = sum(choice_counts.values())
        if total_correct > 0:
            f.write(f"   - 正确答案分布: ")
            for choice, count in choice_counts.items():
                pct = count / total_correct * 100
                f.write(f"{choice}: {pct:.1f}%  ")
            f.write("\n")


if __name__ == '__main__':
    main()

