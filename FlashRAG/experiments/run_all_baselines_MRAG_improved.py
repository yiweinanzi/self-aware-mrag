#!/usr/bin/env python3
"""
MRAG-Bench Baselines对比实验 - 改进版
解决准确率低和性能问题

主要改进：
1. 修复答案解析，正确处理content-based answers
2. RagVL使用BGE reranker替代MLLM reranker
3. 改进多选题答案提取逻辑
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import torch
import gc

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
sys.path.insert(0, '/data0/home/zqwang/ACL/MRAG-Bench-main/eval/utils')

# 导入MRAG-Bench官方解析函数
try:
    from automatic_extract import parse_multi_choice_response as official_parse
except ImportError:
    print("⚠️ 无法导入MRAG-Bench官方解析函数，将使用备用方案")
    official_parse = None

# 导入FlashRAG模块
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 导入基线方法
from experiments.baselines.samrag_adapted import SAMRAGBaseline
from experiments.baselines.mr2ag_enhanced import MR2AGEnhancedBaseline
from experiments.baselines.vidorag_pipeline import ViDoRAGPipeline
from experiments.baselines.ragvl_enhanced import RagVLEnhancedPipeline
from experiments.baselines.visrag_pipeline import VisRAGPipeline
from experiments.baselines.murag_pipeline import MuRAGPipeline
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL


class ImprovedMRAGEvaluator:
    """改进版MRAG评估器"""

    def __init__(self):
        self.results = {}

    def extract_answer_improved(self, response: str, sample: Dict) -> str:
        """
        改进的答案提取，完全参考MRAG-Bench官方逻辑

        Args:
            response: 模型响应
            sample: 包含选项的样本

        Returns:
            提取的选项字母 (A/B/C/D)
        """
        response = response.strip()

        # 1. 首先尝试直接匹配选项
        for choice in ['A', 'B', 'C', 'D']:
            if response == choice:
                return choice

        # 2. 使用MRAG-Bench官方解析函数
        if official_parse:
            # 构建index2ans映射
            index2ans = {}
            for choice in ['A', 'B', 'C', 'D']:
                if choice in sample:
                    index2ans[choice.lower()] = sample[choice].lower()

            try:
                result = official_parse(response, ['A', 'B', 'C', 'D'], index2ans)
                if result in ['A', 'B', 'C', 'D']:
                    return result
            except:
                pass

        # 3. 检查是否直接回答了选项内容
        for choice in ['A', 'B', 'C', 'D']:
            if choice in sample:
                # 检查答案是否与选项内容匹配（不区分大小写）
                if sample[choice].lower() in response.lower():
                    return choice

        # 4. 尝试各种格式模式
        import re
        patterns = [
            r'答案[是]?[:]*\s*([ABCD])',
            r'选择[是]?[:]*\s*([ABCD])',
            r'正确答案[是]?[:]*\s*([ABCD])',
            r'Answer is[:]*\s*([ABCD])',
            r'The answer is[:]*\s*([ABCD])',
            r'Choice is[:]*\s*([ABCD])',
            r'([ABCD])\.',
            r'^([ABCD])\s*$',
            r'\(([ABCD])\)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 5. 如果都失败了，返回原响应
        return response.strip()

    def calculate_metrics(self, method_name: str, results: List[Dict], samples: List[Dict]) -> Dict:
        """计算指标"""
        print(f"\n计算 {method_name} 的指标...")

        correct = 0
        total = 0
        scenario_correct = {}
        scenario_total = {}

        # 详细记录每个样本的结果
        detailed_results = []

        for i, (result, sample) in enumerate(zip(results, samples)):
            # 获取真实答案
            gt = sample.get('answer_choice', '').upper()
            if not gt:
                print(f"⚠️ 样本 {i} 缺少answer_choice，跳过")
                continue

            # 获取预测答案
            pred = result.get('answer', '').strip()
            pred_parsed = self.extract_answer_improved(pred, sample)

            # 记录详细信息
            detailed_info = {
                'sample_id': i,
                'question': sample.get('question', ''),
                'gt_choice': gt,
                'gt_text': sample.get(gt, '') if gt else '',
                'pred_raw': pred,
                'pred_parsed': pred_parsed,
                'is_correct': gt == pred_parsed.upper() if pred_parsed in ['A', 'B', 'C', 'D'] else False,
                'scenario': sample.get('scenario', 'Unknown')
            }
            detailed_results.append(detailed_info)

            # 统计准确率
            if gt and pred_parsed and gt.upper() == pred_parsed.upper():
                correct += 1
            total += 1

            # 分场景统计
            scenario = sample.get('scenario', 'Unknown')
            if scenario not in scenario_correct:
                scenario_correct[scenario] = 0
                scenario_total[scenario] = 0
            scenario_total[scenario] += 1
            if gt and pred_parsed and gt.upper() == pred_parsed.upper():
                scenario_correct[scenario] += 1

        # 保存详细结果
        self.results[method_name] = {
            'detailed_results': detailed_results,
            'sample_count': total
        }

        overall_accuracy = correct / total * 100 if total > 0 else 0

        # 使用FlashRAG评估器计算其他指标
        formatted_results = []
        for i, (r, s) in enumerate(zip(results, samples)):
            formatted_results.append({
                'answer': r.get('answer', ''),
                'golden_answers': [s.get('answer_choice', '')],
                'retrieved_docs': r.get('retrieved_docs', []),
                'question': s.get('question', ''),
                'id': s.get('id', f'sample_{i}'),
            })

        try:
            unified_metrics = evaluate_comprehensive_metrics(formatted_results)
        except:
            unified_metrics = {
                'em': correct / total if total > 0 else 0,
                'f1': 0,
                'retrieval_rate': 0,
                'retrieval_recall_top5': 0,
                'scenario_accuracy': {}
            }

        # 打印一些错误案例分析
        print(f"\n📋 {method_name} 错误案例分析:")
        errors = [d for d in detailed_results if not d['is_correct']]
        if errors and len(errors) <= 3:
            for error in errors[:3]:
                print(f"  样本 {error['sample_id']}:")
                print(f"    问题: {error['question'][:50]}...")
                print(f"    正确答案: {error['gt_choice']} ({error['gt_text']})")
                print(f"    模型输出: {error['pred_raw'][:50]}...")
                print(f"    解析结果: {error['pred_parsed']}")

        return {
            'method': method_name,
            'accuracy': overall_accuracy / 100,
            'em': unified_metrics['em'],
            'f1': unified_metrics['f1'],
            'retrieval_rate': unified_metrics.get('retrieval_rate', 0),
            'recall_at_5': unified_metrics.get('retrieval_recall_top5', 0),
            'scenario_accuracy': scenario_correct,
            'detailed_errors': errors[:5] if errors else []
        }


def run_improved_mrag_experiment(max_samples: int = 10):
    """运行改进版MRAG实验"""
    print("="*80)
    print("MRAG-Bench Baselines对比实验 - 改进版")
    print("="*80)

    # 加载数据集
    print("\n1. 加载数据集")
    print("-"*40)
    from datasets import load_dataset

    dataset = load_dataset('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench')
    samples = dataset['test'].select(range(min(max_samples, len(dataset['test']))))
    print(f"✅ 加载成功: {len(samples)} 样本")

    # 初始化模型
    print("\n2. 初始化模型")
    print("-"*40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    model_path = "/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct"

    # 配置基线方法
    baselines = [
        {
            'name': 'Self-Aware-MRAG',
            'class': SelfAwarePipelineQwen3VL,
            'config': {
                'model_path': model_path,
                'device': device,
                'uncertainty_threshold': 0.43,
                'max_images': 20,
                'retriever_config': {
                    'retriever_type': 'bm25',
                    'top_k': 5
                }
            }
        },
        {
            'name': 'SAM-RAG',
            'class': SAMRAGBaseline,
            'config': {
                'model_path': model_path,
                'device': device,
                'max_memory': 1000,
                'retrieval_top_k': 5
            }
        },
        {
            'name': 'MuRAG',
            'class': MuRAGPipeline,
            'config': {
                'model_path': model_path,
                'device': device,
                'top_k': 5
            }
        },
        {
            'name': 'VisRAG',
            'class': VisRAGPipeline,
            'config': {
                'model_path': model_path,
                'device': device,
                'max_new_tokens': 10,
                'top_k': 5
            }
        },
        {
            'name': 'ViDoRAG',
            'class': ViDoRAGPipeline,
            'config': {
                'model_path': model_path,
                'device': device,
                'max_new_tokens': 10
            }
        },
        {
            'name': 'RagVL',
            'class': RagVLEnhancedPipeline,
            'config': {
                'model_path': model_path,
                'device': device,
                'use_reranking': False,  # 禁用MLLM重排
                'use_bge_reranker': True,  # 使用BGE重排器
                'rerank_topk': 3,
                'top_k': 5
            }
        },
        {
            'name': 'mR2AG',
            'class': MR2AGEnhancedBaseline,
            'config': {
                'model_path': model_path,
                'device': device,
                'max_new_tokens': 10,
                'top_k': 5
            }
        }
    ]

    # 运行实验
    print("\n3. 运行基线方法")
    print("-"*40)

    evaluator = ImprovedMRAGEvaluator()
    all_metrics = {}

    for baseline in baselines:
        print(f"\n{'='*60}")
        print(f"运行方法: {baseline['name']}")
        print(f"{'='*60}")

        try:
            # 初始化方法
            pipeline = baseline['class'](**baseline['config'])

            # 运行实验
            start_time = time.time()
            results = []

            for i, sample in enumerate(samples):
                print(f"\r进度: {i+1}/{len(samples)}", end='', flush=True)
                try:
                    result = pipeline.run_single(sample)
                    results.append(result)
                except Exception as e:
                    print(f"\n[ERROR] 样本 {i} 失败: {e}")
                    results.append({
                        'answer': '',
                        'retrieved_docs': []
                    })

            print()  # 换行
            end_time = time.time()

            # 计算指标
            metrics = evaluator.calculate_metrics(baseline['name'], results, samples)
            metrics['execution_time'] = end_time - start_time
            metrics['samples_per_second'] = len(samples) / (end_time - start_time)

            all_metrics[baseline['name']] = metrics

            # 打印结果
            print(f"\n✅ {baseline['name']} 完成:")
            print(f"   准确率: {metrics['accuracy']*100:.2f}%")
            print(f"   EM: {metrics['em']:.4f}")
            print(f"   F1: {metrics['f1']:.4f}")
            print(f"   耗时: {metrics['execution_time']:.1f}秒 ({metrics['samples_per_second']:.2f}样本/秒)")

            # 打印错误分析
            if 'detailed_errors' in metrics and metrics['detailed_errors']:
                print(f"   错误样本数: {len(metrics['detailed_errors'])}")

        except Exception as e:
            print(f"\n❌ {baseline['name']} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    print("\n4. 保存结果")
    print("-"*40)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_improved")
    output_dir.mkdir(exist_ok=True)

    # 保存指标
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 指标已保存: {metrics_file}")

    # 保存详细结果（包括错误分析）
    results_file = output_dir / f"detailed_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluator.results, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 详细结果已保存: {results_file}")

    # 打印汇总
    print(f"\n{'='*80}")
    print("📊 实验结果汇总")
    print(f"{'='*80}")
    print(f"{'方法':<20} {'准确率':<10} {'EM':<10} {'F1':<10} {'耗时(秒)':<10}")
    print("-"*80)

    for method, metrics in all_metrics.items():
        print(f"{method:<20} {metrics['accuracy']*100:>8.2f}% "
              f"{metrics['em']:>8.4f} {metrics['f1']:>8.4f} "
              f"{metrics.get('execution_time', 0):>8.1f}")

    print(f"\n🎯 最佳方法: {max(all_metrics.items(), key=lambda x: x[1]['accuracy'])[0]}")
    print(f"   准确率: {max(all_metrics.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']*100:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行改进版MRAG实验")
    parser.add_argument('--max_samples', type=int, default=10, help='最大样本数')
    args = parser.parse_args()

    run_improved_mrag_experiment(max_samples=args.max_samples)