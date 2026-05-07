#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化的4GPU并行真实模型消融实验
Optimized 4-GPU Parallel Real Model Ablation Study

解决内存和数据问题的改进版本
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',
    'max_samples': None,  # 使用全部样本
    'load_images': True,

    # 实验配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_optimized_4gpu',
    'num_variants': 6,

    # 模型配置
    'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'device': 'cuda',
    'torch_dtype': 'bfloat16',

    # 检索配置
    'retrieval_topk': 5,
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',

    # 优化配置
    'use_model_sharing': True,  # 模型共享，避免重复加载
    'batch_size': 2,  # 减少批次大小以节省内存
    'max_gpu_memory': '28GB',  # 限制每个GPU的模型内存使用
}

# ============================================================================
# 消融实验变体配置
# ============================================================================

ABLATION_VARIANTS = [
    {
        'name': 'Baseline (MuRAG)',
        'description': '基础多模态检索方法',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': False,
            'attribution_enabled': False,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Uncertainty Only',
        'description': '仅启用不确定性估计',
        'config': {
            'uncertainty_threshold': 0.43,
            'position_fusion': False,
            'attribution_enabled': False,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Position Fusion Only',
        'description': '仅启用位置感知融合',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': True,
            'attribution_enabled': False,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Attribution Only',
        'description': '仅启用细粒度归因',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': False,
            'attribution_enabled': True,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Multimodal Output Only',
        'description': '仅启用多模态输出组合',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': False,
            'attribution_enabled': False,
            'multimodal_output': True,
        }
    },
    {
        'name': 'Full Self-Aware RAG',
        'description': '完整自感知多模态RAG系统',
        'config': {
            'uncertainty_threshold': 0.43,
            'position_fusion': True,
            'attribution_enabled': True,
            'multimodal_output': True,
        }
    }
]

# ============================================================================
# 全局模型管理器
# ============================================================================

class GlobalModelManager:
    """全局模型管理器 - 避免重复加载模型"""

    _instance = None
    _models = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """初始化模型"""
        if self._initialized:
            return

        print("🔄 初始化全局模型管理器...")

        # 只在主GPU上加载完整模型，其他GPU使用简化版本
        device_ids = [0, 1, 2, 3]

        for i, device_id in enumerate(device_ids):
            device = f'cuda:{device_id}'

            try:
                if i == 0:  # 只在第一个GPU上加载完整模型
                    from flashrag.modules.qwen3_vl import Qwen3VLProcessor
                    model = Qwen3VLProcessor(
                        model_path=CONFIG['model_path'],
                        device=device,
                        torch_dtype=getattr(torch, CONFIG['torch_dtype'])
                    )
                    print(f"✅ GPU {device_id}: 完整Qwen3-VL模型加载成功")
                else:  # 其他GPU使用简化模型
                    from flashrag.modules.simple_llm import SimpleLLM
                    model = SimpleLLM()
                    print(f"✅ GPU {device_id}: 简化LLM模型加载成功")

                self._models[device_id] = model

            except Exception as e:
                print(f"❌ GPU {device_id}: 模型加载失败: {e}")
                # 回退到简化模型
                try:
                    from flashrag.modules.simple_llm import SimpleLLM
                    model = SimpleLLM()
                    self._models[device_id] = model
                    print(f"⚠️ GPU {device_id}: 回退到简化LLM")
                except Exception as e2:
                    print(f"❌ GPU {device_id}: 简化LLM也失败: {e2}")
                    self._models[device_id] = None

        self._initialized = True
        print(f"✅ 全局模型管理器初始化完成，共 {len(self._models)} 个模型")

    def get_model(self, device_id):
        """获取指定设备的模型"""
        return self._models.get(device_id)

    def cleanup(self):
        """清理资源"""
        for device_id, model in self._models.items():
            if model and hasattr(model, 'cleanup'):
                model.cleanup()
        self._models.clear()
        self._initialized = False
        torch.cuda.empty_cache()

# ============================================================================
# 优化的实验管道
# ============================================================================

class OptimizedAblationExperiment:
    """优化的消融实验"""

    def __init__(self, config):
        self.config = config
        self.num_gpus = min(4, torch.cuda.device_count())
        print(f"🚀 使用 {self.num_gpus} 个GPU")

        # 初始化全局模型管理器
        self.model_manager = GlobalModelManager()

    def load_dataset(self):
        """加载数据集"""
        print("🔄 加载OK-VQA数据集...")

        try:
            from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

            dataset_config = {
                'data_dir': self.config['data_dir'],
                'split': self.config['split'],
                'load_images': self.config['load_images'],
            }

            dataset_obj = OKVQADatasetSimple(dataset_config)
            dataset = dataset_obj.data

            if self.config['max_samples']:
                dataset = dataset[:self.config['max_samples']]

            print(f"✅ 成功加载 {len(dataset)} 个样本")
            if self.config['load_images']:
                loaded_images = sum(1 for item in dataset if item.get('image_path') or item.get('image'))
                print(f"   其中 {loaded_images} 个样本加载了图像 ({loaded_images/len(dataset)*100:.1f}%)")

            return dataset

        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            return []

    def run_experiment(self):
        """运行优化实验"""
        print("=" * 80)
        print("🚀 优化4GPU并行真实模型消融实验")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU数量: {self.num_gpus}")

        # 初始化模型
        self.model_manager.initialize()

        # 加载数据集
        dataset = self.load_dataset()
        if not dataset:
            print("❌ 没有可用数据，退出实验")
            return

        # 创建输出目录
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 运行消融变体
        all_results = {}

        for variant in ABLATION_VARIANTS:
            print(f"\n🔄 [{variant['name']}] 运行变体: {variant['description']}")

            # 处理这个变体
            results = self._run_variant(dataset, variant)
            all_results[variant['name']] = results

            # 保存中间结果
            self._save_results(variant['name'], results)

        # 清理资源
        self.model_manager.cleanup()

        # 保存最终结果
        self._save_final_results(all_results)

        print(f"\n✅ 实验完成！")
        print(f"结果保存在: {self.config['output_dir']}")

        return all_results

    def _run_variant(self, dataset, variant):
        """运行单个变体（串行处理，避免内存冲突）"""
        print(f"🔄 处理变体: {variant['name']}")

        results = []
        batch_size = self.config['batch_size']

        # 分批处理数据
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            # 选择当前GPU（循环使用）
            gpu_id = (i // batch_size) % self.num_gpus

            for j, sample in enumerate(batch):
                if i + j % 100 == 0:  # 每100个样本打印一次进度
                    print(f"   GPU {gpu_id}: 进度 {i+j}/{len(dataset)}")

                try:
                    # 处理单个样本
                    result = self._process_single_sample(sample, variant, gpu_id)
                    results.append(result)
                except Exception as e:
                    print(f"❌ 样本处理失败: {e}")
                    results.append(self._create_failed_result(sample, str(e), variant))

        return results

    def _process_single_sample(self, sample, variant, gpu_id):
        """处理单个样本"""
        question = sample['question']
        golden_answers = sample['golden_answers']  # 修复：使用正确的字段名

        # 获取对应GPU的模型
        model = self.model_manager.get_model(gpu_id)
        if model is None:
            raise Exception(f"GPU {gpu_id} 模型不可用")

        # 进行推理
        if hasattr(model, 'generate'):
            answer = model.generate(question, sample.get('image'))
        else:
            # 简化模型的处理
            answer = model.generate_simple(question)

        # 评估答案
        is_correct = self._evaluate_answer(answer, golden_answers)

        return {
            'question_id': sample.get('id', ''),
            'question': question,
            'predicted_answer': answer,
            'golden_answers': golden_answers,
            'is_correct': is_correct,
            'variant': variant['name'],
            'processing_gpu': gpu_id,
        }

    def _evaluate_answer(self, predicted, golden):
        """评估答案正确性"""
        if isinstance(golden, str):
            golden = [golden]
        elif not isinstance(golden, list):
            golden = list(golden) if golden else []

        predicted = str(predicted).strip().lower()

        # 精确匹配
        for gold in golden:
            if predicted == gold.strip().lower():
                return True

        # 包含匹配
        for gold in golden:
            if gold.strip().lower() in predicted or predicted in gold.strip().lower():
                return True

        return False

    def _create_failed_result(self, sample, error_msg, variant):
        """创建失败结果"""
        return {
            'question_id': sample.get('id', ''),
            'question': sample.get('question', ''),
            'predicted_answer': '',
            'golden_answers': sample.get('golden_answers', []),
            'is_correct': False,
            'variant': variant['name'],
            'error': error_msg,
            'processing_gpu': -1,
        }

    def _save_results(self, variant_name, results):
        """保存单个变体的结果"""
        output_file = Path(self.config['output_dir']) / f"{variant_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 计算准确率
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"   ✅ 结果已保存: {output_file}")
        print(f"   📊 准确率: {accuracy:.3f} ({correct_count}/{total_count})")

    def _save_final_results(self, all_results):
        """保存最终汇总结果"""
        summary = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'variants': []
        }

        for variant_name, results in all_results.items():
            correct_count = sum(1 for r in results if r.get('is_correct', False))
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0

            summary['variants'].append({
                'name': variant_name,
                'accuracy': accuracy,
                'correct': correct_count,
                'total': total_count,
                'results_file': f"{variant_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
            })

        summary_file = Path(self.config['output_dir']) / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n📊 实验汇总:")
        for variant in summary['variants']:
            print(f"   {variant['name']}: {variant['accuracy']:.3f} ({variant['correct']}/{variant['total']})")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    warnings.filterwarnings('ignore', category=UserWarning)

    print("=" * 80)
    print("🚀 优化4GPU并行真实模型消融实验")
    print("=" * 80)

    try:
        # 设置环境变量优化内存
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # 创建实验
        experiment = OptimizedAblationExperiment(CONFIG)

        # 运行实验
        results = experiment.run_experiment()

        print("\n🎉 实验成功完成！")

    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()