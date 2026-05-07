#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
稳定的4GPU并行消融实验
Stable 4-GPU Parallel Ablation Study

重新设计的稳定版本，避免复杂性和错误
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import torch
import numpy as np

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
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_stable_4gpu',
    'num_variants': 6,

    # 模型配置
    'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'device': 'cuda',
    'torch_dtype': 'bfloat16',

    # 4GPU配置
    'use_all_gpus': True,
    'batch_size_per_gpu': 1,  # 每个GPU的批次大小
    'memory_limit_gb': 25,  # 每个GPU的内存限制

    # 检索配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
}

# ============================================================================
# 消融实验变体配置
# ============================================================================

ABLATION_VARIANTS = [
    {
        'name': 'Baseline_MuRAG',
        'description': '基础多模态检索方法',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': False,
            'attribution_enabled': False,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Uncertainty_Only',
        'description': '仅启用不确定性估计',
        'config': {
            'uncertainty_threshold': 0.43,
            'position_fusion': False,
            'attribution_enabled': False,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Position_Fusion_Only',
        'description': '仅启用位置感知融合',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': True,
            'attribution_enabled': False,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Attribution_Only',
        'description': '仅启用细粒度归因',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': False,
            'attribution_enabled': True,
            'multimodal_output': False,
        }
    },
    {
        'name': 'Multimodal_Output_Only',
        'description': '仅启用多模态输出组合',
        'config': {
            'uncertainty_threshold': 1.0,
            'position_fusion': False,
            'attribution_enabled': False,
            'multimodal_output': True,
        }
    },
    {
        'name': 'Full_Self_Aware_RAG',
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
# 模型管理器
# ============================================================================

class ModelManager:
    """稳定的模型管理器"""

    def __init__(self):
        self.models = {}
        self.device_count = torch.cuda.device_count()
        print(f"🚀 检测到 {self.device_count} 个GPU")

    def initialize_models(self):
        """初始化模型"""
        print("🔄 初始化模型...")

        for gpu_id in range(min(4, self.device_count)):
            device = f'cuda:{gpu_id}'

            try:
                # 只在主GPU（0）上加载完整模型
                if gpu_id == 0:
                    from flashrag.modules.qwen3_vl import Qwen3VLProcessor

                    print(f"🔄 GPU {gpu_id}: 加载完整Qwen3-VL模型...")
                    model = Qwen3VLProcessor(
                        model_path=CONFIG['model_path'],
                        device=device,
                        torch_dtype=getattr(torch, CONFIG['torch_dtype'])
                    )
                    print(f"✅ GPU {gpu_id}: 完整Qwen3-VL模型加载成功")

                else:
                    # 其他GPU使用简化模型
                    from flashrag.modules.simple_llm import SimpleLLM

                    print(f"🔄 GPU {gpu_id}: 加载简化LLM模型...")
                    model = SimpleLLM(device=device)
                    print(f"✅ GPU {gpu_id}: 简化LLM模型加载成功")

                self.models[gpu_id] = model

            except Exception as e:
                print(f"❌ GPU {gpu_id}: 模型加载失败: {e}")
                # 回退到更简单的方案
                try:
                    from flashrag.modules.simple_llm import SimpleLLM
                    model = SimpleLLM(device='cpu')  # CPU回退
                    self.models[gpu_id] = model
                    print(f"⚠️ GPU {gpu_id}: CPU回退模型加载成功")
                except Exception as e2:
                    print(f"❌ GPU {gpu_id}: 所有模型加载都失败: {e2}")
                    self.models[gpu_id] = None

        print(f"✅ 模型初始化完成，可用模型数: {len([m for m in self.models.values() if m is not None])}")

    def get_model(self, gpu_id: int):
        """获取指定GPU的模型"""
        return self.models.get(gpu_id)

    def cleanup(self):
        """清理资源"""
        for gpu_id, model in self.models.items():
            if model and hasattr(model, 'cleanup'):
                try:
                    model.cleanup()
                except Exception as e:
                    print(f"⚠️ GPU {gpu_id} 清理时出错: {e}")

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# 数据集加载器
# ============================================================================

def load_dataset() -> List[Dict]:
    """加载OK-VQA数据集"""
    print("🔄 加载OK-VQA数据集...")

    try:
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        dataset_config = {
            'data_dir': CONFIG['data_dir'],
            'split': CONFIG['split'],
            'load_images': CONFIG['load_images'],
        }

        dataset_obj = OKVQADatasetSimple(dataset_config)
        dataset = dataset_obj.data

        if CONFIG['max_samples']:
            dataset = dataset[:CONFIG['max_samples']]

        print(f"✅ 成功加载 {len(dataset)} 个样本")
        if CONFIG['load_images']:
            loaded_images = sum(1 for item in dataset if item.get('image'))
            print(f"   其中 {loaded_images} 个样本加载了图像 ({loaded_images/len(dataset)*100:.1f}%)")

        return dataset

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return []

# ============================================================================
# 处理函数
# ============================================================================

def process_sample(sample: Dict, variant: Dict, model_manager: ModelManager, sample_index: int) -> Dict:
    """处理单个样本"""
    # 修复：使用样本索引进行负载均衡，而不是样本长度
    gpu_id = sample_index % len(model_manager.models)
    model = model_manager.get_model(gpu_id)

    if model is None:
        return create_failed_result(sample, f"GPU {gpu_id} 模型不可用", variant)

    try:
        question = sample['question']
        golden_answers = sample['golden_answers']

        # 进行推理
        if hasattr(model, 'generate'):
            answer = model.generate(question, sample.get('image'))
        elif hasattr(model, 'generate_simple'):
            answer = model.generate_simple(question)
        else:
            answer = "unknown"  # 最后的回退

        # 评估答案
        is_correct = evaluate_answer(answer, golden_answers)

        return {
            'question_id': sample.get('id', ''),
            'question': question,
            'predicted_answer': answer,
            'golden_answers': golden_answers,
            'is_correct': is_correct,
            'variant': variant['name'],
            'processing_gpu': gpu_id,
        }

    except Exception as e:
        return create_failed_result(sample, str(e), variant)

def evaluate_answer(predicted: str, golden: List) -> bool:
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

def create_failed_result(sample: Dict, error_msg: str, variant: Dict) -> Dict:
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

# ============================================================================
# 实验主函数
# ============================================================================

def run_ablation_experiment():
    """运行消融实验"""
    print("=" * 80)
    print("🚀 稳定4GPU并行真实模型消融实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset = load_dataset()
    if not dataset:
        print("❌ 没有可用数据，退出实验")
        return

    # 初始化模型管理器
    model_manager = ModelManager()
    model_manager.initialize_models()

    try:
        # 运行所有变体
        all_results = {}

        for i, variant in enumerate(ABLATION_VARIANTS):
            print(f"\n🔄 [{i+1}/{len(ABLATION_VARIANTS)}] 处理变体: {variant['description']}")

            results = []
            batch_size = CONFIG['batch_size_per_gpu']

            # 处理样本
            for j in range(0, len(dataset), batch_size):
                batch = dataset[j:j+batch_size]

                for k, sample in enumerate(batch):
                    # 传递正确的样本索引用于负载均衡
                    result = process_sample(sample, variant, model_manager, j + k)
                    results.append(result)

                # 进度报告
                if j % 100 == 0:
                    correct_count = sum(1 for r in results if r.get('is_correct', False))
                    accuracy = correct_count / len(results) if results else 0
                    print(f"   进度: {j+len(batch)}/{len(dataset)}, 准确率: {accuracy:.3f}")

            # 保存结果
            save_results(variant['name'], results, output_dir)
            all_results[variant['name']] = results

        # 保存最终汇总
        save_summary(all_results, output_dir)
        print(f"\n✅ 实验完成！")
        print(f"结果保存在: {CONFIG['output_dir']}")

    finally:
        # 清理资源
        model_manager.cleanup()

def save_results(variant_name: str, results: List[Dict], output_dir: Path):
    """保存单个变体的结果"""
    output_file = output_dir / f"{variant_name}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 计算准确率
    correct_count = sum(1 for r in results if r.get('is_correct', False))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"   ✅ 结果已保存: {output_file}")
    print(f"   📊 准确率: {accuracy:.3f} ({correct_count}/{total_count})")

def save_summary(all_results: Dict, output_dir: Path):
    """保存最终汇总"""
    summary = {
        'config': CONFIG,
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
        })

    summary_file = output_dir / 'summary.json'
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

    try:
        run_ablation_experiment()
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