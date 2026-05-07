#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速评测脚本 - MRAG-Bench上的简化评测

适用场景：
- 快速验证所有方法是否能正常运行
- 测试7个核心指标的计算
- 生成初步的对比报告

运行方式:
```bash
cd /root/autodl-tmp/FlashRAG
conda activate multirag
python experiments/run_quick_evaluation.py
```
"""

import os
import sys
import json
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加FlashRAG路径
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import torch
import numpy as np
from tqdm import tqdm

# 设置警告过滤
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 数据集配置（中等规模：100样本）
    'dataset_name': 'mragbench',
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'max_samples': 100,  # 中等规模测试
    
    # 模型配置
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    
    # 评测配置
    'save_results': True,
    'output_dir': '/root/autodl-tmp/FlashRAG/experiments/results_full_evaluation',
    
    # 通用参数（优化后）
    'temperature': 0.05,  # 降低温度提高确定性
    'max_new_tokens': 10,  # 多选题只需要很短的回答
    'retrieval_topk': 5,
}


# ============================================================================
# 检索器初始化（使用真实检索器或Mock）
# ============================================================================

def init_retriever(use_real_retriever=True):
    """
    初始化检索器
    
    Args:
        use_real_retriever: 是否使用真实检索器
    
    Returns:
        retriever对象
    """
    if not use_real_retriever:
        # Mock检索器（快速测试）
        class SimpleRetriever:
            def __init__(self):
                print("✅ 使用SimpleRetriever（Mock模式，返回空结果）")
            
            def retrieve(self, query_text, query_image=None, top_k=5):
                return [], []
        
        return SimpleRetriever()
    
    # 使用真实检索器
    try:
        from flashrag.retriever import DenseRetriever
        
        print("✅ 尝试加载真实检索器（DenseRetriever）...")
        
        # FlashRAG DenseRetriever需要完整的config
        retriever_config = {
            # 基础配置
            'index_path': '/root/autodl-tmp/FlashRAG/indexes/3m/bge/e5_Flat.index',
            'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
            
            # 检索方法配置
            'retrieval_method': 'e5',  # 或'bge'
            'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 128,
            'retrieval_topk': 5,  # 检索Top-K
            
            # 缓存配置
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            
            # Reranker配置
            'use_reranker': False,
            
            # 其他配置
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
        }
        
        retriever = DenseRetriever(retriever_config)
        print("✅ DenseRetriever加载成功")
        
        return retriever
        
    except Exception as e:
        print(f"⚠️  真实检索器加载失败: {e}")
        print("降级到Mock检索器...")
        
        class SimpleRetriever:
            def __init__(self):
                print("⚠️  使用SimpleRetriever（Mock模式）")
            
            def retrieve(self, query_text, query_image=None, top_k=5):
                return [], []
        
        return SimpleRetriever()


# ============================================================================
# 简化的评估数据类
# ============================================================================

class EvalData:
    """评估数据封装"""
    
    def __init__(self, results):
        self.pred = [r.get('answer', '') for r in results]
        self.golden_answers = [r.get('golden_answers', []) for r in results]
        self.choices = [[] for _ in results]


# ============================================================================
# 简化的指标计算
# ============================================================================

def calculate_simple_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    计算简化版的7个核心指标
    
    Args:
        results: 结果列表
    
    Returns:
        Dict[str, float]: 指标字典
    """
    from flashrag.evaluator.utils import normalize_answer
    
    total = len(results)
    if total == 0:
        return {
            'em': 0.0, 'f1': 0.0, 'retrieval_recall_top5': 0.0,
            'vqa_score': 0.0, 'faithfulness': 0.0,
            'attribution_precision': 0.0, 'position_bias_score': 0.0
        }
    
    em_count = 0
    f1_scores = []
    
    for result in results:
        pred = result.get('answer', '')
        golden = result.get('golden_answers', [])
        
        if not golden:
            continue
        
        # EM
        pred_norm = normalize_answer(pred)
        for g in golden:
            if pred_norm == normalize_answer(g):
                em_count += 1
                break
        
        # F1（简化版）
        pred_tokens = set(pred_norm.split())
        max_f1 = 0.0
        for g in golden:
            g_tokens = set(normalize_answer(g).split())
            if not pred_tokens or not g_tokens:
                continue
            
            common = pred_tokens & g_tokens
            if not common:
                continue
            
            p = len(common) / len(pred_tokens)
            r = len(common) / len(g_tokens)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            max_f1 = max(max_f1, f1)
        
        f1_scores.append(max_f1)
    
    metrics = {
        'em': em_count / total,
        'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'retrieval_recall_top5': 0.0,  # 需要检索结果
        'vqa_score': em_count / total,  # 简化：与EM相同
        'faithfulness': 0.0,  # 需要检索文档
        'attribution_precision': 0.0,  # 需要归因结果
        'position_bias_score': 0.0,  # 需要特殊测试
    }
    
    return metrics


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(dataset_path: str, max_samples: int = None):
    """加载MRAG-Bench数据集（从Arrow格式）"""
    print(f"\n{'='*80}")
    print("加载MRAG-Bench数据集...")
    print(f"{'='*80}")
    
    import datasets
    
    # 加载Arrow格式的数据集
    dataset_dict = datasets.load_from_disk(dataset_path)
    dataset = dataset_dict['test']
    
    # 转换为列表格式
    samples = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    for i in range(num_samples):
        sample = dataset[i]
        samples.append(sample)
    
    print(f"✅ 加载完成: {len(samples)} 样本")
    
    return samples


# ============================================================================
# 单个方法评测
# ============================================================================

def evaluate_method(method_name: str, method, dataset: List[Dict],
                   config: Dict) -> Dict[str, Any]:
    """评测单个方法"""
    print(f"\n{'='*80}")
    print(f"评测方法: {method_name}")
    print(f"{'='*80}")
    
    results = []
    start_time = time.time()
    
    # 运行方法
    for sample in tqdm(dataset, desc=f"运行 {method_name}"):
        try:
            # 调用方法生成答案
            if hasattr(method, 'run_single'):
                result = method.run_single(sample)
            elif hasattr(method, 'generate'):
                # MRAG-Bench使用'answer'字段，转换为列表格式
                golden = sample.get('answer', '')
                golden_list = [golden] if golden else []
                
                result = {
                    'question': sample.get('question', ''),
                    'answer': method.generate(sample),
                    'golden_answers': golden_list
                }
            else:
                warnings.warn(f"{method_name}没有run_single或generate方法")
                
                golden = sample.get('answer', '')
                golden_list = [golden] if golden else []
                
                result = {
                    'question': sample.get('question', ''),
                    'answer': '',
                    'golden_answers': golden_list
                }
            
            results.append(result)
            
        except Exception as e:
            warnings.warn(f"样本处理失败: {e}")
            golden = sample.get('answer', '')
            golden_list = [golden] if golden else []
            
            results.append({
                'question': sample.get('question', ''),
                'answer': '',
                'golden_answers': golden_list
            })
            continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 计算指标
    print(f"\n计算指标...")
    metrics = calculate_simple_metrics(results)
    
    # 汇总结果
    summary = {
        'method_name': method_name,
        'num_samples': len(results),
        'elapsed_time': elapsed_time,
        'avg_time_per_sample': elapsed_time / len(results) if results else 0,
        'metrics': metrics,
    }
    
    # 打印结果
    print(f"\n指标结果:")
    print(f"  EM: {metrics['em']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  VQA-Score: {metrics['vqa_score']:.4f}")
    print(f"\n运行时间: {elapsed_time:.2f}秒 ({elapsed_time/len(results):.2f}秒/样本)")
    
    return summary


# ============================================================================
# 保存结果
# ============================================================================

def save_results(all_results: Dict[str, Any], config: Dict):
    """保存评测结果"""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存JSON
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 保存结果到: {results_file}")
    
    # 2. 生成Markdown报告
    report_file = output_dir / f"REPORT_{timestamp}.md"
    generate_report(all_results, report_file, config)
    print(f"✅ 生成报告: {report_file}")


def generate_report(all_results: Dict[str, Any], report_file: Path, config: Dict):
    """生成对比报告"""
    lines = []
    
    lines.append("# MRAG-Bench快速评测报告")
    lines.append("")
    lines.append(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**数据集**: MRAG-Bench ({config['max_samples']} 样本)")
    lines.append(f"**模型**: Qwen3-VL-8B-Instruct")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 指标对比表
    lines.append("## 核心指标对比")
    lines.append("")
    lines.append("| Method | EM | F1 | VQA-Score | 时间(秒/样本) |")
    lines.append("|--------|----|----|-----------|-------------|")
    
    for method_name, result in all_results.items():
        metrics = result['metrics']
        line = f"| **{method_name}** |"
        line += f" {metrics.get('em', 0):.4f} |"
        line += f" {metrics.get('f1', 0):.4f} |"
        line += f" {metrics.get('vqa_score', 0):.4f} |"
        line += f" {result['avg_time_per_sample']:.3f} |"
        lines.append(line)
    
    lines.append("")
    
    # 写入文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主评测流程"""
    
    print("\n" + "="*80)
    print("MRAG-Bench快速评测")
    print("="*80)
    print(f"\n配置:")
    print(f"  - 样本数: {CONFIG['max_samples']}")
    print(f"  - 模型: Qwen3-VL-8B-Instruct")
    print(f"  - 输出: {CONFIG['output_dir']}")
    
    # 1. 加载数据集
    dataset = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    
    # 2. 初始化Qwen3-VL
    print(f"\n{'='*80}")
    print("初始化Qwen3-VL...")
    print(f"{'='*80}")
    
    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    
    try:
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=CONFIG['qwen3_vl_path'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ Qwen3-VL加载成功")
    except Exception as e:
        print(f"⚠️  Qwen3-VL加载失败: {e}")
        print("使用Mock模型继续测试...")
        qwen3_vl = None
    
    # 3. 初始化检索器（尝试使用真实检索器）
    retriever = init_retriever(use_real_retriever=True)
    
    # 4. 初始化方法并评测
    all_results = {}
    
    # 方法1: Self-Aware MRAG (Our Method)
    if qwen3_vl:
        try:
            print(f"\n{'='*80}")
            print("初始化 Self-Aware MRAG...")
            print(f"{'='*80}")
            
            from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
            
            method = SelfAwarePipelineQwen3VL(
                qwen3_vl_wrapper=qwen3_vl,
                retriever=retriever,
                config={
                    'temperature': CONFIG['temperature'],
                    'max_new_tokens': CONFIG['max_new_tokens'],
                    'uncertainty_threshold': 0.35,
                }
            )
            
            result = evaluate_method('Self-Aware-MRAG', method, dataset, CONFIG)
            all_results['Self-Aware-MRAG'] = result
            
        except Exception as e:
            print(f"⚠️  Self-Aware MRAG初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 添加其他方法...（根据需要）
    
    # 5. 保存结果
    if all_results and CONFIG['save_results']:
        save_results(all_results, CONFIG)
    
    # 6. 总结
    print(f"\n{'='*80}")
    print("评测完成!")
    print(f"{'='*80}")
    print(f"\n共评测 {len(all_results)} 个方法")
    print(f"样本数: {CONFIG['max_samples']}")
    
    if all_results:
        print("\n结果摘要:")
        for method_name, result in all_results.items():
            metrics = result['metrics']
            print(f"\n{method_name}:")
            print(f"  EM: {metrics['em']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  时间: {result['avg_time_per_sample']:.3f}秒/样本")


if __name__ == '__main__':
    main()

