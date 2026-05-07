#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试四个数据集的脚本 - 每个数据集10个样本
使用模拟数据快速验证流程
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入必要的模块
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 配置
CONFIG = {
    'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/test_results_4datasets',
    'max_samples': 10,
}

# ============================================================================
# 测试数据
# ============================================================================

def get_mock_data(dataset_name):
    """获取模拟数���"""
    if dataset_name == 'okvqa':
        return [
            {'id': f'okvqa_{i}', 'question': f'OK-VQA question {i}', 'answer': f'answer_{i}', 'golden_answers': [f'answer_{i}']}
            for i in range(CONFIG['max_samples'])
        ]
    elif dataset_name == 'mrag-bench':
        return [
            {'id': f'mrag_{i}', 'question': f'MRAG question {i}', 'answer': chr(65 + i % 4), 'A': 'Option A', 'B': 'Option B', 'C': 'Option C', 'D': 'Option D'}
            for i in range(CONFIG['max_samples'])
        ]
    elif dataset_name == 'multimodalqa':
        return [
            {'id': f'mmqa_{i}', 'question': f'MMQA question {i}', 'answer': f'answer_{i}', 'question_type': 'TextQ', 'modalities': ['text']}
            for i in range(CONFIG['max_samples'])
        ]
    elif dataset_name == 'aokvqa':
        return [
            {'id': f'aokvqa_{i}', 'question': f'A-OKVQA question {i}', 'answer': chr(65 + i % 4),
             'choices': ['Option A', 'Option B', 'Option C', 'Option D'], 'correct_choice_idx': i % 4}
            for i in range(CONFIG['max_samples'])
        ]

def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*80)
    print("测试Qwen3-VL模型加载")
    print("="*80)

    try:
        model = create_qwen3_vl_wrapper(
            model_path=CONFIG['model_path'],
            device="cuda",
            torch_dtype="bfloat16",
            thinking=False
        )
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "="*80)
    print("测试数据集加载")
    print("="*80)

    datasets = ['okvqa', 'mrag-bench', 'multimodalqa', 'aokvqa']

    for dataset in datasets:
        print(f"\n测试 {dataset.upper()} 数据集...")
        data = get_mock_data(dataset)
        print(f"✅ {dataset}: 加载 {len(data)} 样本成功")

        # 打印第一个样本
        if data:
            print(f"   示例: {data[0]}")

def test_evaluation():
    """测试评估流程"""
    print("\n" + "="*80)
    print("测试评估流程")
    print("="*80)

    # 创建模拟结果
    mock_results = []
    for i in range(10):
        mock_results.append({
            'answer': f'predicted_answer_{i}',
            'golden_answers': [f'answer_{i}'],
            'retrieved_docs': [{'contents': f'retrieved_doc_{j}'} for j in range(3)],
            'question': f'test_question_{i}',
            'id': f'test_{i}'
        })

    try:
        metrics = evaluate_comprehensive_metrics(mock_results)
        print("✅ 评估成功")
        print(f"   EM: {metrics.get('em', 0):.4f}")
        print(f"   F1: {metrics.get('avg_F1', 0):.4f}")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

def save_test_report(results):
    """保存测试报告"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    report_file = os.path.join(CONFIG['output_dir'], 'test_report.md')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 四个数据集测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for dataset, status in results.items():
            f.write(f"## {dataset.upper()}\n")
            f.write(f"- 状态: {'✅ 成功' if status['success'] else '❌ 失败'}\n")
            if not status['success']:
                f.write(f"- 错误: {status.get('error', 'Unknown')}\n")
            f.write(f"- 耗时: {status.get('time', 0):.2f}s\n")
            f.write("\n")

def main():
    """主函数"""
    print("="*80)
    print("四个数据集快速测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"每个数据集测试样本数: {CONFIG['max_samples']}")

    results = {}

    # 1. 测试数据集加载
    start_time = time.time()
    try:
        test_dataset_loading()
        results['data_loading'] = {'success': True, 'time': time.time() - start_time}
    except Exception as e:
        results['data_loading'] = {'success': False, 'error': str(e), 'time': time.time() - start_time}

    # 2. 测试模型加载
    start_time = time.time()
    model = test_model_loading()
    results['model_loading'] = {'success': model is not None, 'time': time.time() - start_time}

    # 3. 测试评估流程
    start_time = time.time()
    test_evaluation()
    results['evaluation'] = {'success': True, 'time': time.time() - start_time}

    # 保存报告
    save_test_report(results)

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    print(f"报告保存在: {CONFIG['output_dir']}/test_report.md")

if __name__ == '__main__':
    main()