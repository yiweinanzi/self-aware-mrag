#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集评测管理器 - 统一管理4个数据集的评测
Dataset Evaluation Manager for Unified Evaluation

功能：
1. 加载4个数据集
2. 运行模型推理
3. 统一评测指标
4. 生成评测报告
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.unified_dataset_loader import load_unified_dataset
from flashrag.evaluator.unified_evaluator import evaluate_unified


class DatasetEvaluationManager:
    """数据集评测管理器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 支持的数据集
        self.supported_datasets = ['okvqa', 'a-okvqa', 'multimodalqa', 'mrag-bench']

        # 默认配置
        self.default_config = {
            'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_unified_evaluation',
            'max_samples': 100,  # 每个数据集的最大样本数
            'save_results': True,
            'generate_report': True,
            'run_inference': False  # 是否运行推理
        }

        # 合并配置
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

    def run_evaluation(self, datasets: Optional[List[str]] = None,
                      pipeline=None, results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        运行数据集评测

        Args:
            datasets: 要评测的数据集列表，None表示评测所有
            pipeline: 推理pipeline（如果需要运行推理）
            results_dir: 预先存在的结果目录

        Returns:
            所有数据集的评测结果
        """
        if datasets is None:
            datasets = self.supported_datasets

        # 验证数据集
        for ds in datasets:
            if ds not in self.supported_datasets:
                raise ValueError(f"不支持的数据集: {ds}")

        print("\n" + "="*80)
        print("统一数据集评测开始")
        print("="*80)
        print(f"数据集: {', '.join(datasets)}")
        print(f"最大样本数: {self.config['max_samples']}")
        print(f"输出目录: {self.config['output_dir']}")
        print("="*80)

        # 创建输出目录
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 评测结果
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for dataset_name in datasets:
            print(f"\n{'#'*60}")
            print(f"评测数据集: {dataset_name.upper()}")
            print(f"{'#'*60}")

            try:
                # 加载数据集
                dataset = load_unified_dataset(
                    dataset_name,
                    split='val',
                    max_samples=self.config['max_samples']
                )

                # 获取预测结果
                if results_dir:
                    # 从已有结果加载
                    predictions = self._load_predictions(results_dir, dataset_name)
                else:
                    # 运行推理
                    if self.config['run_inference'] and pipeline:
                        predictions = self._run_inference(pipeline, dataset)
                    else:
                        # 使用示例预测
                        print("⚠️ 使用示例预测结果（未提供pipeline或results_dir）")
                        predictions = self._get_dummy_predictions(dataset)

                # 准备参考答案
                references = self._prepare_references(dataset)

                # 评测
                metrics = evaluate_unified(dataset_name, predictions, references)

                # 保存结果
                all_results[dataset_name] = {
                    'dataset': dataset_name,
                    'metrics': metrics,
                    'num_samples': len(dataset),
                    'predictions': predictions if self.config['save_results'] else None,
                    'references': references if self.config['save_results'] else None
                }

            except Exception as e:
                print(f"❌ {dataset_name} 评测失败: {e}")
                import traceback
                traceback.print_exc()
                all_results[dataset_name] = {
                    'dataset': dataset_name,
                    'error': str(e),
                    'metrics': {},
                    'num_samples': 0
                }

        # 保存结果
        if self.config['save_results']:
            self._save_results(all_results, timestamp)

        # 生成报告
        if self.config['generate_report']:
            self._generate_report(all_results, timestamp)

        print("\n" + "="*80)
        print("评测完成!")
        print("="*80)

        return all_results

    def _load_predictions(self, results_dir: str, dataset_name: str) -> List[Dict]:
        """从目录加载预测结果"""
        results_path = Path(results_dir) / f"{dataset_name}_predictions.json"

        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ 预测结果文件不存在: {results_path}")
            return []

    def _run_inference(self, pipeline, dataset) -> List[Dict]:
        """运行推理"""
        print(f"\n运行 {dataset.name} 推理...")

        predictions = []
        start_time = time.time()

        for i, sample in enumerate(tqdm(dataset, desc=f"推理 {dataset.name}")):
            try:
                # 运行单个样本
                result = pipeline.run_single(sample)

                prediction = {
                    'id': sample.get('id', i),
                    'answer': result.get('answer', ''),
                    'retrieved_docs': result.get('retrieved_docs', []),
                    'retrieval_result': result.get('retrieval_result', []),
                    'attributions': result.get('attributions', {}),
                    'position_bias_results': result.get('position_bias_results', {}),
                    'used_retrieval': result.get('used_retrieval', False)
                }

                predictions.append(prediction)

            except Exception as e:
                print(f"⚠️ 样本 {i} 推理失败: {e}")
                predictions.append({
                    'id': sample.get('id', i),
                    'answer': '',
                    'error': str(e)
                })

        elapsed_time = time.time() - start_time
        print(f"✅ 推理完成: {len(predictions)} 样本, {elapsed_time:.1f}秒")

        return predictions

    def _get_dummy_predictions(self, dataset) -> List[Dict]:
        """生成示例预测（用于测试）"""
        print("使用示例预测结果")

        predictions = []
        for i, sample in enumerate(dataset):
            prediction = {
                'id': sample.get('id', i),
                'answer': sample.get('golden_answers', [''])[0] if sample.get('golden_answers') else 'Sample answer',
                'retrieved_docs': ['Sample document 1', 'Sample document 2'],
                'retrieval_result': [{
                    'retrieved_docs': ['Sample document 1', 'Sample document 2'],
                    'retrieval_scores': [0.9, 0.8],
                    'retrieval_used': True
                }],
                'attributions': {
                    'visual': [0, 1],
                    'text': [0, 1]
                },
                'position_bias_results': {
                    'average_bias': 0.3,
                    'individual_scores': [0.3, 0.3],
                    'position_weights': [0.4, 0.3, 0.2, 0.07, 0.03]
                },
                'used_retrieval': True
            }
            predictions.append(prediction)

        return predictions

    def _prepare_references(self, dataset) -> List[Dict]:
        """准备参考答案"""
        references = []
        for sample in dataset:
            ref = {
                'id': sample.get('id', ''),
                'question': sample.get('question', ''),
                'golden_answers': sample.get('golden_answers', []),
                'dataset': sample.get('dataset', dataset.name)
            }
            references.append(ref)
        return references

    def _save_results(self, all_results: Dict[str, Any], timestamp: str):
        """保存评测结果"""
        output_dir = Path(self.config['output_dir'])

        # 保存完整结果
        results_file = output_dir / f"unified_evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ 结果已保存: {results_file}")

        # 保存每个数据集的结果
        for dataset_name, result in all_results.items():
            if 'metrics' in result:
                metrics_file = output_dir / f"{dataset_name}_metrics_{timestamp}.json"
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(result['metrics'], f, indent=2, ensure_ascii=False)

    def _generate_report(self, all_results: Dict[str, Any], timestamp: str):
        """生成评测报告"""
        output_dir = Path(self.config['output_dir'])
        report_file = output_dir / f"unified_evaluation_report_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 统一数据集评测报告\n\n")
            f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**样本限制**: {self.config['max_samples']} 样本/数据集\n\n")

            # 核心指标对比表
            f.write("## 核心指标对比\n\n")
            f.write("| 数据集 | 准确率 | F1 | 检索率 | Recall@5 | Faithfulness | Attribution | Position Bias |\n")
            f.write("|--------|--------|----|--------|----------|-------------|------------|-------------|\n")

            for dataset_name, result in all_results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    f.write(f"| {dataset_name.upper()} | ")
                    f.write(f"{metrics.get('accuracy', 0)*100:.2f}% | ")
                    f.write(f"{metrics.get('avg_F1', 0):.4f} | ")
                    f.write(f"{metrics.get('retrieval_rate', 0)*100:.1f}% | ")
                    f.write(f"{metrics.get('avg_Recall@5', 0):.4f} | ")
                    f.write(f"{metrics.get('avg_Faithfulness', 0):.4f} | ")
                    f.write(f"{metrics.get('avg_Attribution_Precision', 0):.4f} | ")
                    f.write(f"{metrics.get('avg_Position_Bias_Score', 0):.4f} |\n")

            # MRAG-Bench场景分析
            if 'mrag-bench' in all_results:
                f.write("\n## MRAG-Bench 场景准确率\n\n")
                mrag_metrics = all_results['mrag-bench']['metrics']

                scenarios = [
                    ('Overall', '总体'),
                    ('Angle', '角度'),
                    ('Partial', '部分'),
                    ('Scope', '范围'),
                    ('Occlusion', '遮挡'),
                    ('Temporal', '时序'),
                    ('Deformation', '变形'),
                    ('Incomplete', '不完整'),
                    ('Biological', '生物'),
                    ('Others', '其他')
                ]

                f.write("| 场景 | 准确率 |\n")
                f.write("|------|--------|\n")

                for scenario, scenario_cn in scenarios:
                    key = f'{scenario.lower()}_accuracy'
                    if key in mrag_metrics:
                        f.write(f"| {scenario_cn} ({scenario}) | {mrag_metrics[key]:.2f}% |\n")

            # 详细指标
            f.write("\n## 详细指标\n\n")
            for dataset_name, result in all_results.items():
                if 'metrics' in result:
                    f.write(f"\n### {dataset_name.upper()}\n\n")
                    metrics = result['metrics']

                    # 核心指标
                    f.write("- 样本数: " + str(result.get('num_samples', 0)) + "\n")
                    f.write(f"- 准确率: {metrics.get('accuracy', 0)*100:.2f}%\n")
                    f.write(f"- F1 Score: {metrics.get('avg_F1', 0):.4f}\n")
                    f.write(f"- 检索率: {metrics.get('retrieval_rate', 0)*100:.1f}%\n")
                    f.write(f"- Recall@5: {metrics.get('avg_Recall@5', 0):.4f}\n")
                    f.write(f"- Faithfulness: {metrics.get('avg_Faithfulness', 0):.4f}\n")
                    f.write(f"- Attribution Precision: {metrics.get('avg_Attribution_Precision', 0):.4f}\n")
                    f.write(f"- Position Bias Score: {metrics.get('avg_Position_Bias_Score', 0):.4f}\n")

        print(f"✅ 报告已生成: {report_file}")


# 命令行接口
def main():
    parser = argparse.ArgumentParser(description='统一数据集评测')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='要评测的数据集列表')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='每个数据集的最大样本数')
    parser.add_argument('--output-dir', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/experiments/results_unified_evaluation',
                       help='输出目录')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='已有结果目录')
    parser.add_argument('--run-inference', action='store_true',
                       help='运行推理')

    args = parser.parse_args()

    # 配置
    config = {
        'max_samples': args.max_samples,
        'output_dir': args.output_dir,
        'run_inference': args.run_inference
    }

    # 创建管理器
    manager = DatasetEvaluationManager(config)

    # 运行评测
    results = manager.run_evaluation(
        datasets=args.datasets,
        results_dir=args.results_dir
    )

    print("\n评测完成!")


if __name__ == '__main__':
    main()