#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试真实Qwen3-VL模型性能
Test Real Qwen3-VL Model Performance

只用真实模型测试，验证基础性能
"""

import sys
import json
import torch
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_real_qwen3vl():
    """测试真实Qwen3-VL模型"""
    print("🔄 测试真实Qwen3-VL模型性能...")

    try:
        from flashrag.modules.qwen3_vl import Qwen3VLProcessor
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        # 初始化模型
        print("🔄 加载Qwen3-VL模型...")
        model = Qwen3VLProcessor(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device='cuda:0',
            torch_dtype=torch.bfloat16
        )
        print("✅ Qwen3-VL模型加载成功")

        # 加载少量样本测试
        print("🔄 加载测试数据...")
        dataset_obj = OKVQADatasetSimple({
            'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
            'split': 'val',
            'load_images': True,
        })

        # 测试前100个样本
        test_samples = dataset_obj.data[:100]
        print(f"✅ 加载测试样本: {len(test_samples)}")

        # 测试推理
        correct_count = 0
        results = []

        for i, sample in enumerate(test_samples):
            question = sample['question']
            golden_answers = sample['golden_answers']

            print(f"\n[{i+1}/20] 问题: {question}")
            print(f"   答案: {golden_answers[:3]}")  # 只显示前3个答案

            try:
                # 使用真实模型推理
                answer = model.generate(question, sample.get('image'))
                print(f"   预测: {answer}")

                # 评估
                is_correct = evaluate_answer(answer, golden_answers)
                if is_correct:
                    correct_count += 1
                    print("   ✅ 正确!")
                else:
                    print("   ❌ 错误")

                results.append({
                    'question': question,
                    'predicted': answer,
                    'golden': golden_answers,
                    'correct': is_correct
                })

            except Exception as e:
                print(f"   ❌ 推理失败: {e}")
                results.append({
                    'question': question,
                    'predicted': '',
                    'golden': golden_answers,
                    'correct': False,
                    'error': str(e)
                })

        # 计算准确率
        accuracy = correct_count / len(test_samples)
        print(f"\n📊 测试结果:")
        print(f"   准确率: {accuracy:.3f} ({correct_count}/{len(test_samples)})")
        print(f"   成功推理: {len([r for r in results if not r.get('error')])}")
        print(f"   推理失败: {len([r for r in results if r.get('error')])}")

        # 保存结果
        with open('/data0/home/zqwang/ACL/test_real_model_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

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
    test_real_qwen3vl()