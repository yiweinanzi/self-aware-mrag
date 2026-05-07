#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - MultiModalQA数据集，7个核心指标

方法列表：
1. Self-Aware-MRAG (Our Method)
2. SAM-RAG (替换Self-RAG)
3. mR²AG
4. VisRAG
5. ViDoRAG
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
import argparse
import gzip
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
from experiments.baselines.samrag_adapted import SAMRAGPipeline
from experiments.baselines.mr2ag_enhanced import MR2AGPipeline
from experiments.baselines.visrag_enhanced import VisRAGPipeline
from experiments.baselines.murag_enhanced import MuRAGPipeline


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MultiModalQA Baseline对比实验')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='最大样本数 (默认: 10)')
    parser.add_argument('--dataset_path', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA',
                       help='数据集路径')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'dev', 'test'],
                       help='数据集划分 (默认: test)')
    return parser.parse_args()


# ============================================================================
# 配置
# ============================================================================

# 解析命令行参数
args = parse_args()

CONFIG = {
    # 数据集配置
    'dataset_name': 'multimodalqa',
    'dataset_path': args.dataset_path,
    'split': args.split,
    'max_samples': args.max_samples,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_multimodalqa_baseline',
    'save_detailed_results': True,
    'save_sample_results': True,
    'enable_complete_metrics': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'max_new_tokens': 50,
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
    'num_gpus': 2,
    'batch_size_per_gpu': 1,
}


# ============================================================================
# MultiModalQA数据加载
# ============================================================================

def load_multimodalqa_dataset(dataset_path, split='test', max_samples=None):
    """加载MultiModalQA数据集"""
    print(f"加载数据集: MultiModalQA")
    print(f"数据路径: {dataset_path}")
    print(f"数据划分: {split}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    # 加载问题数据
    qa_file = os.path.join(dataset_path, f'MMQA_{split}.jsonl.gz')

    questions = []
    answers = []

    with gzip.open(qa_file, 'rt') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append({
                'qid': data['qid'],
                'question': data['question'],
                'metadata': data.get('metadata', {}),
                'answer': data.get('answer', '') if split != 'test' else ''
            })
            if split != 'test':
                answers.append(data.get('answer', ''))

    if max_samples:
        questions = questions[:max_samples]
        if answers:
            answers = answers[:max_samples]

    # 加载文本数据
    texts_file = os.path.join(dataset_path, 'MMQA_texts.jsonl.gz')
    texts = {}
    with gzip.open(texts_file, 'rt') as f:
        for line in f:
            data = json.loads(line.strip())
            texts[data['id']] = data

    # 加载表格数据
    tables_file = os.path.join(dataset_path, 'MMQA_tables.jsonl.gz')
    tables = {}
    with gzip.open(tables_file, 'rt') as f:
        for line in f:
            data = json.loads(line.strip())
            tables[data['id']] = data

    # 加载图像数据
    images_file = os.path.join(dataset_path, 'MMQA_images.jsonl.gz')
    images = {}
    with gzip.open(images_file, 'rt') as f:
        for line in f:
            data = json.loads(line.strip())
            images[data['id']] = data

    # 构建corpus
    corpus = []
    for text_id, text_data in texts.items():
        corpus.append({
            'id': text_id,
            'contents': text_data['text'],
            'title': text_data.get('title', ''),
            'type': 'text'
        })

    for table_id, table_data in tables.items():
        # 将表格转换为文本
        table_text = f"Table: {table_data.get('title', '')}\n"
        for row in table_data['table']['table_rows']:
            row_text = " | ".join([cell.get('text', '') for cell in row])
            table_text += f"{row_text}\n"
        corpus.append({
            'id': table_id,
            'contents': table_text,
            'title': table_data.get('title', ''),
            'type': 'table'
        })

    for image_id, image_data in images.items():
        corpus.append({
            'id': image_id,
            'contents': f"Image: {image_data.get('title', '')}",
            'title': image_data.get('title', ''),
            'type': 'image'
        })

    print(f"✅ 加载成功:")
    print(f"   - 问题数: {len(questions)}")
    print(f"   - 文档数: {len(corpus)}")
    print(f"   - 文本: {len(texts)}")
    print(f"   - 表格: {len(tables)}")
    print(f"   - 图像: {len(images)}")

    return questions, answers, corpus


# ============================================================================
# 评测相关
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
# 模型初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL模型"""
    print(f"初始化Qwen3-VL: {model_path}")

    return create_qwen3_vl_wrapper(
        model_path=model_path,
        device_map='auto',
        torch_dtype=CONFIG['torch_dtype'],
        max_new_tokens=CONFIG['max_new_tokens'],
        temperature=CONFIG['temperature'],
        thinking=False
    )


def init_retriever(config, use_multimodal=False):
    """初始化检索器"""
    print(f"初始化检索器...")

    if use_multimodal and config['use_multimodal_retrieval']:
        print(f"  模式: 多模态融合 (BGE + CLIP)")

        # BGE文本检索器
        text_retriever = DenseRetriever(
            model_path=config['retrieval_model_path'],
            index_path=config['index_path'],
            corpus_path=config['corpus_path'],
            batch_size=64,
            max_length=512,
            retrieval_topk=config['retrieval_topk']
        )

        # CLIP视觉检索器 - 如果有索引的话
        try:
            from flashrag.retriever.clip_retriever import CLIPRetriever
            visual_retriever = CLIPRetriever(
                model_path=config['clip_model_path'],
                index_path=config['clip_index_path'],
                retrieval_topk=config['retrieval_topk']
            )

            # 创建多模态融合检索器
            from flashrag.retriever.multimodal_retriever import MultimodalRetriever
            multimodal_retriever = MultimodalRetriever(
                text_retriever=text_retriever,
                visual_retriever=visual_retriever,
                text_weight=config['text_retrieval_weight'],
                visual_weight=config['visual_retrieval_weight']
            )

            print(f"  ✅ 多模态检索器初始化成功")
            return multimodal_retriever

        except Exception as e:
            print(f"  ⚠️  CLIP检索器初始化失败: {e}")
            print(f"  💡 降级使用纯BGE文本检索")
            return text_retriever
    else:
        print(f"  模式: 纯文本 (BGE)")
        return DenseRetriever(
            model_path=config['retrieval_model_path'],
            index_path=config['index_path'],
            corpus_path=config['corpus_path'],
            batch_size=64,
            max_length=512,
            retrieval_topk=config['retrieval_topk']
        )


# ============================================================================
# 运行单个方法
# ============================================================================

def run_method(method_name, pipeline, questions, corpus):
    """运行单个方法"""
    print(f"\n{'='*80}")
    print(f"运行方法: {method_name}")
    print(f"{'='*80}")

    results = []
    start_time = time.time()

    for i, question_data in enumerate(tqdm(questions, desc=f"Processing {method_name}")):
        # 准备sample格式
        sample = {
            'id': question_data['qid'],
            'question': question_data['question'],
            'answer': question_data.get('answer', ''),
            'image': None,  # MultiModalQA可能需要处理多个图像
            'metadata': question_data.get('metadata', {}),
        }

        # 获取相关文档ID
        metadata = question_data.get('metadata', {})
        doc_ids = []
        if 'text_doc_ids' in metadata:
            doc_ids.extend(metadata['text_doc_ids'])
        if 'table_id' in metadata:
            doc_ids.append(metadata['table_id'])
        if 'image_doc_ids' in metadata:
            doc_ids.extend(metadata['image_doc_ids'])

        # 运行pipeline
        try:
            result = pipeline.process(sample)
            results.append(result)
        except Exception as e:
            print(f"  ❌ 处理样本 {question_data['qid']} 时出错: {e}")
            results.append({
                'question': question_data['question'],
                'answer': '',
                'retrieved_docs': [],
                'error': str(e)
            })

    elapsed_time = time.time() - start_time
    return results, elapsed_time


# ============================================================================
# 指标计算
# ============================================================================

def calculate_metrics(method_name, results, questions):
    """计算指标"""
    print(f"\n{'='*80}")
    print(f"评测方法: {method_name}")
    print(f"{'='*80}")

    # 基本准确率计算
    correct = 0
    total = len(results)

    for r, q in zip(results, questions):
        pred = r.get('answer', '')
        gt = q.get('answer', '')

        if gt and exact_match_score(pred, gt):
            correct += 1

    accuracy = correct / total if total > 0 else 0

    # 准备数据以匹配FlashRAG评估器格式
    formatted_results = []
    for i, (r, q) in enumerate(zip(results, questions)):
        formatted_result = {
            'answer': r.get('answer', ''),
            'golden_answers': [q.get('answer', '')] if q.get('answer') else [],
            'retrieved_docs': r.get('retrieved_docs', []),
            'question': q.get('question', ''),
            'id': q.get('qid', f'sample_{i}'),
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
        metrics = evaluate_comprehensive_metrics(formatted_results)

        unified_metrics = {
            'method': method_name,
            'accuracy': accuracy,
            'em': metrics.get('em', accuracy),
            'f1': metrics.get('avg_F1', 0),
            'retrieval_rate': metrics.get('retrieval_rate', 0),
            'retrieval_recall_top5': metrics.get('avg_Recall@5', 0),
            'vqa_score': metrics.get('vqa_score', 0),
            'faithfulness': metrics.get('avg_Faithfulness', 0),
            'attribution_precision': metrics.get('avg_Attribution_Precision', 0),
            'position_bias_score': metrics.get('avg_Position_Bias_Score', 0),
        }

        # 打印指标
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  ✅ EM: {unified_metrics['em']:.4f}")
        print(f"  ✅ F1: {unified_metrics['f1']:.4f}")
        print(f"  ✅ Retrieval Rate: {unified_metrics['retrieval_rate']:.4f}")
        print(f"  ✅ Recall@5: {unified_metrics['retrieval_recall_top5']:.4f}")

        return unified_metrics

    except Exception as e:
        print(f"  ❌ 指标计算失败: {e}")

        # 返回基本指标
        return {
            'method': method_name,
            'accuracy': accuracy,
            'em': accuracy,
            'f1': 0.0,
            'retrieval_rate': 0.0,
            'retrieval_recall_top5': 0.0,
            'vqa_score': 0.0,
            'faithfulness': 0.0,
            'attribution_precision': 0.0,
            'position_bias_score': 0.0,
        }


# ============================================================================
# 保存结果
# ============================================================================

def save_results(all_results, all_metrics):
    """保存结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果
    output_file = Path(CONFIG['output_dir']) / f"multimodalqa_results_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'config': CONFIG,
            'metrics': all_metrics,
            'results': all_results,
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存到: {output_file}")

    # 生成对比表格
    metrics_file = Path(CONFIG['output_dir']) / f"multimodalqa_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ 指标已保存到: {metrics_file}")

    return output_file, metrics_file


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*80)
    print("MultiModalQA Baseline对比实验")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")
    print()

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("="*80)
    print("1. 加载数据集")
    print("="*80)
    questions, answers, corpus = load_multimodalqa_dataset(
        CONFIG['dataset_path'],
        CONFIG['split'],
        CONFIG['max_samples']
    )

    # 初始化模型和检索器
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])

    # 初始化BGE检索器
    bge_retriever = init_retriever(CONFIG, use_multimodal=False)

    # 初始化多模态融合检索器
    multimodal_retriever = init_retriever(CONFIG, use_multimodal=True)

    # 定义所有方法
    methods = {
        'Self-Aware-MRAG': lambda: SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=multimodal_retriever,
            config={
                'uncertainty_threshold': CONFIG['uncertainty_threshold'],
                'use_improved_estimator': CONFIG['use_improved_estimator'],
                'use_position_fusion': True,
                'use_attribution': True,
                'clip_model_path': CONFIG['clip_model_path'],
                'retrieval_topk': CONFIG['retrieval_topk'],
                'text_weight': CONFIG['text_weight'],
                'visual_weight': CONFIG['visual_weight'],
                'alignment_weight': CONFIG['alignment_weight'],
                'thinking': False,
                'max_images': 10,
                'temperature': CONFIG['temperature'],
                'max_new_tokens': CONFIG['max_new_tokens'],
            }
        ),
        'SAM-RAG': lambda: SAMRAGPipeline(qwen3_vl, bge_retriever, {
            **CONFIG,
            'sam_batch_size': 5,
            'sam_max_batches': 4,
        }),
        'mR2AG': lambda: MR2AGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'VisRAG': lambda: VisRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'ViDoRAG': lambda: create_vidorag_pipeline(qwen3_vl, bge_retriever, CONFIG),
        'RagVL': lambda: RagVLEnhanced(qwen3_vl, None, {**CONFIG, **{
            'use_reranking': False,
            'rerank_topk': 0,
            'clip_topk': 0,
            'no_retrieval': True
        }}),
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
            results, elapsed_time = run_method(method_name, pipeline, questions, corpus)

            # 计算指标
            metrics = calculate_metrics(method_name, results, questions)
            metrics['runtime_seconds'] = elapsed_time
            metrics['seconds_per_sample'] = elapsed_time / len(questions)

            all_results[method_name] = results
            all_metrics[method_name] = metrics

            print(f"\n✅ {method_name} 完成:")
            print(f"   准确率: {metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   EM: {metrics.get('em', 0):.4f}")
            print(f"   F1: {metrics.get('f1', 0):.4f}")
            print(f"   耗时: {elapsed_time:.1f}秒 ({metrics['seconds_per_sample']:.2f}s/样本)")

        except Exception as e:
            print(f"\n❌ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    print("\n" + "="*80)
    print("4. 保存结果")
    print("="*80)
    save_results(all_results, all_metrics)

    print(f"\n{'='*80}")
    print("✅ MultiModalQA Baseline对比实验完成！")
    print(f"📁 结果保存在: {CONFIG['output_dir']}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()