#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化2GPU全真实Qwen3-VL并行实验
Optimized 2GPU All Real Qwen3-VL Parallel Experiment

使用2个GPU，确保充足的内存空间进行高质量推理
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
import gc

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
    'max_samples': 2000,  # 2GPU可以处理更多样本
    'load_images': True,

    # 实验配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_2gpu_real_qwen3vl',
    'num_variants': 6,

    # 模型配置 - 2个GPU使用相同模型
    'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'device': 'cuda',
    'torch_dtype': 'bfloat16',

    # 2GPU配置 - 确保充足内存
    'use_all_gpus': True,
    'num_gpus': 2,  # 明确指定使用2个GPU
    'batch_size_per_gpu': 1,
    'memory_limit_gb': 40,  # 预留更多内存
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
# 2GPU真实模型管理器
# ============================================================================

class TwoGPURealModelManager:
    """2GPU真实模型管理器 - 2个GPU都使用Qwen3-VL"""

    def __init__(self):
        self.models = {}
        self.device_count = torch.cuda.device_count()
        self.num_gpus = min(2, self.device_count)  # 确保只使用2个GPU
        print(f"🚀 检测到 {self.device_count} 个GPU，使用 {self.num_gpus} 个GPU进行2GPU并行")

    def initialize_models(self):
        """初始化2个GPU的Qwen3-VL模型"""
        print("🔄 初始化2GPU全真实Qwen3-VL模型...")

        # 清理所有GPU缓存
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        gc.collect()

        for gpu_id in range(self.num_gpus):
            device = f'cuda:{gpu_id}'

            try:
                from flashrag.modules.qwen3_vl import Qwen3VLProcessor

                print(f"🔄 GPU {gpu_id}: 加载Qwen3-VL模型到 {device}...")
                model = Qwen3VLProcessor(
                    model_path=CONFIG['model_path'],
                    device=device,
                    torch_dtype=getattr(torch, CONFIG['torch_dtype'])
                )
                print(f"✅ GPU {gpu_id}: Qwen3-VL模型加载成功")

                self.models[gpu_id] = model

                # 清理缓存，为下一个GPU准备
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ GPU {gpu_id}: 模型加载失败: {e}")
                if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                    print(f"⚠️ GPU {gpu_id}: 内存不足，停止加载更多模型")
                    self.num_gpus = gpu_id  # 只使用前面成功的GPU
                    break
                else:
                    self.models[gpu_id] = None

        # 确保至少有一个模型可用
        working_models = [m for m in self.models.values() if m is not None]
        if not working_models:
            raise RuntimeError("没有可用的模型！请检查GPU内存和模型路径")

        print(f"✅ 模型初始化完成，可用模型数: {len(working_models)}")

    def get_model(self, sample_index: int):
        """获取轮询分配的模型，只在2个GPU间轮询"""
        working_gpus = [gid for gid, model in self.models.items() if model is not None]
        if not working_gpus:
            return None

        # 在可用的真实模型GPU间轮询
        gpu_id = working_gpus[sample_index % len(working_gpus)]
        return self.models[gpu_id]

    def cleanup(self):
        """清理资源"""
        print("🔄 清理GPU资源...")
        for gpu_id, model in self.models.items():
            if model and hasattr(model, 'cleanup'):
                try:
                    model.cleanup()
                except Exception as e:
                    print(f"⚠️ GPU {gpu_id} 清理时出错: {e}")

        # 清理所有GPU
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU资源清理完成")

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

def process_sample(sample: Dict, variant: Dict, model_manager: TwoGPURealModelManager, sample_index: int) -> Dict:
    """使用真实Qwen3-VL处理单个样本"""

    # 获取真实的Qwen3-VL模型
    model = model_manager.get_model(sample_index)
    if model is None:
        return create_failed_result(sample, "无可用模型", variant)

    try:
        question = sample['question']
        golden_answers = sample['golden_answers']

        # 使用真实Qwen3-VL模型进行推理
        answer = model.generate(question, sample.get('image'))

        # 评估答案
        is_correct = evaluate_answer(answer, golden_answers)

        return {
            'question_id': sample.get('id', ''),
            'question': question,
            'predicted_answer': answer,
            'golden_answers': golden_answers,
            'is_correct': is_correct,
            'variant': variant['name'],
            'model_type': 'Qwen3-VL-Real-2GPU',  # 标记使用2GPU真实模型
        }

    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            error_msg = "GPU内存不足"
            # 尝试清理缓存
            torch.cuda.empty_cache()
            gc.collect()

        return create_failed_result(sample, error_msg, variant)

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
        'model_type': 'Failed',
    }

# ============================================================================
# 实验主函数
# ============================================================================

def run_2gpu_real_ablation_experiment():
    """运行2GPU全真实模型消融实验"""
    print("=" * 80)
    print("🚀 优化2GPU全真实Qwen3-VL并行消融实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("架构：2个GPU全部使用真实Qwen3-VL模型，确保充足内存")

    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset = load_dataset()
    if not dataset:
        print("❌ 没有可用数据，退出实验")
        return

    # 初始化2GPU真实模型管理器
    model_manager = TwoGPURealModelManager()
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
                    result = process_sample(sample, variant, model_manager, j + k)
                    results.append(result)

                # 进度报告
                if j % 100 == 0:
                    correct_count = sum(1 for r in results if r.get('is_correct', False))
                    accuracy = correct_count / len(results) if results else 0
                    real_model_count = sum(1 for r in results if 'Qwen3-VL-Real' in r.get('model_type', ''))
                    failed_count = sum(1 for r in results if r.get('error'))
                    print(f"   进度: {j+len(batch)}/{len(dataset)}, 准确率: {accuracy:.3f}, 真实模型: {real_model_count}/{len(results)}, 失败: {failed_count}")

                # 每50个样本清理一次缓存
                if j % 50 == 0 and j > 0:
                    torch.cuda.empty_cache()
                    gc.collect()

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
    real_model_count = sum(1 for r in results if 'Qwen3-VL-Real' in r.get('model_type', ''))
    failed_count = sum(1 for r in results if r.get('error'))
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"   ✅ 结果已保存: {output_file}")
    print(f"   📊 准确率: {accuracy:.3f} ({correct_count}/{total_count})")
    print(f"   🎯 真实模型使用率: {real_model_count/total_count*100:.1f}% ({real_model_count}/{total_count})")
    print(f"   ❌ 失败率: {failed_count/total_count*100:.1f}% ({failed_count}/{total_count})")

def save_summary(all_results: Dict, output_dir: Path):
    """保存最终汇总"""
    summary = {
        'config': CONFIG,
        'architecture': '2GPU All Real Qwen3-VL Models',
        'timestamp': datetime.now().isoformat(),
        'variants': []
    }

    for variant_name, results in all_results.items():
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        real_model_count = sum(1 for r in results if 'Qwen3-VL-Real' in r.get('model_type', ''))
        failed_count = sum(1 for r in results if r.get('error'))
        accuracy = correct_count / total_count if total_count > 0 else 0

        summary['variants'].append({
            'name': variant_name,
            'accuracy': accuracy,
            'correct': correct_count,
            'total': total_count,
            'real_model_usage': real_model_count / total_count if total_count > 0 else 0,
            'failed_rate': failed_count / total_count if total_count > 0 else 0,
            'results_file': f"{variant_name}.json"
        })

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📊 实验汇总 (2GPU全真实Qwen3-VL):")
    for variant in summary['variants']:
        print(f"   {variant['name']}: {variant['accuracy']:.3f} ({variant['correct']}/{variant['total']}, 真实模型: {variant['real_model_usage']*100:.1f}%, 失败: {variant['failed_rate']*100:.1f}%)")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    warnings.filterwarnings('ignore', category=UserWarning)

    try:
        run_2gpu_real_ablation_experiment()
        print("\n🎉 2GPU全真实Qwen3-VL实验成功完成！")
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