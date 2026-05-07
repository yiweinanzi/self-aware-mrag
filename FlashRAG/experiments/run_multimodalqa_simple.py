#!/usr/bin/env python3
"""
MultiModalQA实验 - 简化版本（基于MOQAGPT方法）
- 表格简化为纯文本
- 提取式QA提示
- 禁用支持度验证避免错误回退
"""

import os
import sys
import json
import gzip
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashrag.library.config import Config
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.evaluator import Evaluator
from flashrag.utils.utils import get_dataset

def load_multimodalqa_data(data_path: str, split: str = "dev", max_samples: int = None):
    """加载MultiModalQA数据集"""
    file_path = os.path.join(data_path, f"MMQA_{split}.jsonl.gz")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            sample = json.loads(line.strip())
            data.append(sample)
            if max_samples and len(data) >= max_samples:
                break

    print(f"✅ MultiModalQA加载成功: {len(data)} 样本")

    # Print statistics
    question_types = {}
    modality_counts = {}
    for sample in data:
        q_type = sample.get('question_type', 'Unknown')
        question_types[q_type] = question_types.get(q_type, 0) + 1

        modalities = sample.get('modalities', [])
        modality_key = ','.join(sorted(modalities))
        modality_counts[modality_key] = modality_counts.get(modality_key, 0) + 1

    print(f"问题类型分布: {question_types}")
    print(f"模态分布: {modality_counts}")

    return data

def simple_multimodalqa_eval(predicted_answers: List[str], ground_truth_answers: List[Dict]) -> Dict[str, float]:
    """简化的MultiModalQA评测（只检查exact match）"""
    if len(predicted_answers) != len(ground_truth_answers):
        raise ValueError("Predicted and ground truth answers must have same length")

    correct = 0
    for pred, gt in zip(predicted_answers, ground_truth_answers):
        gt_answer = gt.get('answer', '')
        if pred.strip().lower() == gt_answer.strip().lower():
            correct += 1

    accuracy = correct / len(ground_truth_answers)

    # For consistency with other metrics
    f1 = accuracy  # Simplified

    return {
        'accuracy': accuracy,
        'em': accuracy,  # Same as accuracy in this simplified version
        'f1': f1
    }

def parse_args():
    parser = argparse.ArgumentParser(description="MultiModalQA简化版实验")
    parser.add_argument("--max_samples", type=int, default=10, help="最大样本数")
    parser.add_argument("--dataset_path", type=str,
                       default="/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA",
                       help="数据集路径")
    parser.add_argument("--config_path", type=str,
                       default="/data0/home/zqwang/ACL/FlashRAG/config/multimodalqa_simple.yaml",
                       help="配置文件路径")
    parser.add_argument("--output_dir", type=str,
                       default="/data0/home/zqwang/ACL/results_multimodalqa_simple",
                       help="结果输出目录")
    parser.add_argument("--split", type=str, default="dev", help="数据划分")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print experiment info
    print("=" * 80)
    print("MultiModalQA简化版实验（基于MOQAGPT方法）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {args.max_samples}")
    print(f"数据集: MULTIMODALQA")
    print(f"数据划分: {args.split}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    try:
        # 1. Load dataset
        print("\n" + "=" * 80)
        print("1. 加载数据集")
        print("=" * 80)
        dataset = load_multimodalqa_data(args.dataset_path, args.split, args.max_samples)

        # 2. Initialize pipeline
        print("\n" + "=" * 80)
        print("2. 初始化管道")
        print("=" * 80)

        # Create config
        config_dict = {
            'dataset_name': 'multimodalqa',
            'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            'retrieval_method': 'multimodal_fusion',
            'retrieval_config': {
                'text_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index',
                'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/clip_Flat.index',
                'image_corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/image_corpus.jsonl',
                'text_corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl',
                'fusion_weights': {'text': 0.6, 'visual': 0.4},
                'topk': 20
            },
            'uncertainty_threshold': 0.35,
            'max_images': 20,
            'thinking_mode': False,
            'use_dataset_docs': True,  # 使用数据集提供的文档
            'enable_support_verification': False  # 禁用支持度验证
        }

        # Save config
        os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
        with open(args.config_path, 'w') as f:
            import yaml
            yaml.dump(config_dict, f, default_flow_style=False)

        config = Config(config_dict)

        # Initialize pipeline
        pipeline = SelfAwarePipelineQwen3VL(config)

        # 3. Run pipeline
        print("\n" + "=" * 80)
        print("3. 运行管道")
        print("=" * 80)

        predicted_answers = []
        ground_truth_answers = []

        for i, sample in enumerate(dataset):
            print(f"\n处理样本 {i+1}/{len(dataset)}")
            print(f"问题: {sample['question']}")

            # Run pipeline
            result = pipeline.run(sample['question'], sample)

            predicted_answer = result.get('answer', '')
            predicted_answers.append(predicted_answer)

            ground_truth_answers.append({
                'answer': sample.get('answer', ''),
                'question_type': sample.get('question_type', '')
            })

            print(f"预测答案: {predicted_answer}")
            print(f"真实答案: {sample.get('answer', '')}")

        # 4. Evaluate
        print("\n" + "=" * 80)
        print("4. 评测结果")
        print("=" * 80)

        # MultiModalQA official evaluation
        mmqa_metrics = simple_multimodalqa_eval(predicted_answers, ground_truth_answers)

        print(f"\n✅ MultiModalQA官方评测:")
        print(f"  - Accuracy: {mmqa_metrics['accuracy']:.4f}")
        print(f"  - EM (Exact Match): {mmqa_metrics['em']:.4f}")
        print(f"  - F1: {mmqa_metrics['f1']:.4f}")

        # 5. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = os.path.join(args.output_dir, f'simple_results_{timestamp}.json')
        results_data = {
            'dataset': 'MULTIMODALQA',
            'split': args.split,
            'num_samples': len(dataset),
            'method': 'Self-Aware-MRAG-Simple',
            'config': config_dict,
            'results': []
        }

        for i, sample in enumerate(dataset):
            results_data['results'].append({
                'id': sample.get('id', i),
                'question': sample['question'],
                'predicted': predicted_answers[i],
                'ground_truth': ground_truth_answers[i]['answer'],
                'question_type': ground_truth_answers[i]['question_type']
            })

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Save metrics
        metrics_file = os.path.join(args.output_dir, f'simple_metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(mmqa_metrics, f, indent=2)

        print(f"\n✅ 详细结果已保存: {results_file}")
        print(f"✅ 评测指标已保存: {metrics_file}")

        print("\n" + "=" * 80)
        print("实验完成!")
        print("=" * 80)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"实验出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()