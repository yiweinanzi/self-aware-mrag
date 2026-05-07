#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单GPU 100样本真实模型测试
Single GPU 100 Samples Real Model Test

验证修复后的准确率提升效果
"""

import os
import sys
import json
import time
import torch
from datetime import datetime
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_single_gpu_real_model():
    """测试单GPU 100样本的真实Qwen3-VL模型"""
    print("=" * 80)
    print("🚀 单GPU 100样本真���Qwen3-VL模型测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标：验证修复后的准确率提升效果")
    print()

    try:
        from flashrag.modules.qwen3_vl import Qwen3VLProcessor
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        # 初始化模型
        print("🔄 加载Qwen3-VL-8B模型...")
        model = Qwen3VLProcessor(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device='cuda:0',
            torch_dtype=torch.bfloat16
        )
        print("✅ Qwen3-VL模型加载成功")

        # 加载100个样本
        print("\n🔄 加载100个测试样本...")
        dataset_obj = OKVQADatasetSimple({
            'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
            'split': 'val',
            'load_images': True,
        })

        test_samples = dataset_obj.data[:100]
        print(f"✅ 成功加载 {len(test_samples)} 个测试样本")

        # 运行消融测试
        results = []

        # 测试配置
        variants = [
            {
                'name': 'Baseline',
                'config': {'uncertainty_threshold': 1.0, 'position_fusion': False}
            },
            {
                'name': 'Self_Aware_RAG',
                'config': {'uncertainty_threshold': 0.43, 'position_fusion': True}
            }
        ]

        for variant in variants:
            print(f"\n🔄 测试变体: {variant['name']}")
            variant_results = []

            for i, sample in enumerate(test_samples):
                question = sample['question']
                golden_answers = sample['golden_answers']

                print(f"[{i+1}/100] {question[:50]}...")

                try:
                    # 使用真实Qwen3-VL模型推理
                    start_time = time.time()
                    answer = model.generate(question, sample.get('image'))
                    inference_time = time.time() - start_time

                    # 评估答案
                    is_correct = evaluate_answer(answer, golden_answers)

                    result = {
                        'sample_id': i + 1,
                        'question': question,
                        'predicted_answer': answer,
                        'golden_answers': golden_answers,
                        'is_correct': is_correct,
                        'inference_time': inference_time,
                        'variant': variant['name']
                    }

                    variant_results.append(result)

                    # 实时准确率
                    if (i + 1) % 10 == 0:
                        correct_count = sum(1 for r in variant_results if r['is_correct'])
                        accuracy = correct_count / len(variant_results)
                        avg_time = sum(r['inference_time'] for r in variant_results) / len(variant_results)
                        print(f"   前{i+1}个样本: 准确率 {accuracy:.3f} ({correct_count}/{i+1}), 平均推理时间 {avg_time:.2f}s")

                except Exception as e:
                    print(f"   ❌ 推理失败: {e}")
                    variant_results.append({
                        'sample_id': i + 1,
                        'question': question,
                        'predicted_answer': '',
                        'golden_answers': golden_answers,
                        'is_correct': False,
                        'inference_time': 0,
                        'variant': variant['name'],
                        'error': str(e)
                    })

            results.extend(variant_results)

        # 分析结果
        print("\n📊 测试结果分析:")
        print("=" * 60)

        for variant in variants:
            variant_results = [r for r in results if r['variant'] == variant['name']]
            correct_count = sum(1 for r in variant_results if r['is_correct'])
            total_count = len(variant_results)
            accuracy = correct_count / total_count if total_count > 0 else 0
            failed_count = sum(1 for r in variant_results if r.get('error'))
            avg_time = sum(r['inference_time'] for r in variant_results if r['inference_time'] > 0) / len([r for r in variant_results if r['inference_time'] > 0]) if variant_results else 0

            print(f"\n{variant['name']}:")
            print(f"   准确率: {accuracy:.3f} ({correct_count}/{total_count})")
            print(f"   失败率: {failed_count/total_count*100:.1f}% ({failed_count}/{total_count})")
            print(f"   平均推理时间: {avg_time:.2f}秒")

        # 保存结果
        output_file = '/data0/home/zqwang/ACL/test_single_gpu_100samples_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_info': {
                    'timestamp': datetime.now().isoformat(),
                    'samples_count': len(test_samples),
                    'model': 'Qwen3-VL-8B-Instruct',
                    'device': 'cuda:0'
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 测试结果已保存: {output_file}")

        # 准确率对比分析
        baseline_results = [r for r in results if r['variant'] == 'Baseline']
        aware_results = [r for r in results if r['variant'] == 'Self_Aware_RAG']

        baseline_accuracy = sum(1 for r in baseline_results if r['is_correct']) / len(baseline_results) if baseline_results else 0
        aware_accuracy = sum(1 for r in aware_results if r['is_correct']) / len(aware_results) if aware_results else 0

        print(f"\n🎯 关键发现:")
        print(f"   Baseline准确率: {baseline_accuracy:.3f}")
        print(f"   Self_Aware_RAG准确率: {aware_accuracy:.3f}")
        print(f"   准确率提升: {(aware_accuracy - baseline_accuracy)*100:.1f}%")

        if baseline_accuracy > 0.1:  # 如果准确率超过10%，说明修复成功
            print(f"\n🎉 修复成功！准确率从之前的1.3%提升到{baseline_accuracy:.1%}+")
        else:
            print(f"\n⚠️ 准确率仍然较低，需要进一步调试")

        return accuracy

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def evaluate_answer(predicted, golden):
    """评估答案"""
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

if __name__ == '__main__':
    test_single_gpu_real_model()