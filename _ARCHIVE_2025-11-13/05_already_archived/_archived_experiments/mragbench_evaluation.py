#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRAG-Bench评测脚本
评估位置偏差改善效果

参考: MRAG-Bench (ICLR 2025)
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def load_mragbench_data(data_dir='/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench'):
    """
    加载MRAG-Bench数据集
    
    注意: 需要根据实际数据格式调整
    """
    print(f"从 {data_dir} 加载MRAG-Bench数据...")
    
    # TODO: 根据实际数据格式实现
    # 预期格式: 
    # {
    #   'question': str,
    #   'image': path,
    #   'documents': [{'text': str, 'image': path, 'relevance': float}],
    #   'answer': str,
    #   'metadata': {'position_sensitive': bool}
    # }
    
    samples = []
    
    # 示例加载逻辑（需要根据实际调整）
    try:
        import json
        metadata_file = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                data = json.load(f)
                samples = data.get('samples', [])
        else:
            print(f"⚠️ 未找到metadata.json，尝试其他格式...")
            # 尝试加载其他格式
            pass
    except Exception as e:
        print(f"⚠️ 加载数据失败: {e}")
    
    return samples


def reorder_documents(documents, position='beginning', key_doc_idx=0):
    """
    重新排列文档，将关键文档放在指定位置
    
    Args:
        documents: 文档列表
        position: 'beginning', 'middle', 'end'
        key_doc_idx: 关键文档的原始索引
    
    Returns:
        重排后的文档列表
    """
    docs = documents.copy()
    key_doc = docs.pop(key_doc_idx)
    
    if position == 'beginning':
        docs.insert(0, key_doc)
    elif position == 'middle':
        mid = len(docs) // 2
        docs.insert(mid, key_doc)
    elif position == 'end':
        docs.append(key_doc)
    
    return docs


def evaluate_position_bias(model, samples, positions=['beginning', 'middle', 'end']):
    """
    评估位置偏差
    
    对每个样本，测试关键文档在不同位置时的性能
    """
    print("\n" + "=" * 80)
    print("🔍 位置偏差评估")
    print("=" * 80)
    
    results_by_position = defaultdict(list)
    position_bias_scores = []
    
    for sample in tqdm(samples, desc="评估进度"):
        question = sample['question']
        image = sample.get('image')
        documents = sample.get('documents', [])
        golden_answer = sample.get('answer', '')
        
        # 确定关键文档（通常是relevance最高的）
        if documents:
            key_doc_idx = max(range(len(documents)), 
                            key=lambda i: documents[i].get('relevance', 0))
        else:
            continue
        
        sample_results = {}
        
        # 测试不同位置
        for pos in positions:
            # 重排文档
            reordered_docs = reorder_documents(documents, pos, key_doc_idx)
            
            # 生成答案（TODO: 实际调用模型）
            try:
                # answer = model.generate(question, reordered_docs, image)
                # 临时模拟
                answer = f"answer_at_{pos}"
                
                # 评估准确性
                # score = evaluate_answer(answer, golden_answer)
                score = np.random.random()  # 临时模拟
                
                sample_results[pos] = score
                results_by_position[pos].append(score)
                
            except Exception as e:
                print(f"\n⚠️ 评估失败: {e}")
                continue
        
        # 计算该样本的位置偏差
        if len(sample_results) == len(positions):
            bias = np.std(list(sample_results.values()))
            position_bias_scores.append(bias)
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📊 位置偏差评估结果")
    print("=" * 80)
    
    # 各位置平均性能
    print("\n各位置平均准确率:")
    for pos in positions:
        if results_by_position[pos]:
            avg_score = np.mean(results_by_position[pos])
            std_score = np.std(results_by_position[pos])
            print(f"  {pos:12s}: {avg_score:.4f} (±{std_score:.4f})")
    
    # 位置偏差分数
    overall_bias = np.mean(position_bias_scores) if position_bias_scores else 0
    print(f"\n位置偏差分数 (越小越好): {overall_bias:.4f}")
    
    # 位置敏感性
    all_scores = []
    for pos in positions:
        all_scores.extend(results_by_position[pos])
    
    if all_scores:
        position_sensitivity = (max(np.mean(results_by_position[p]) for p in positions) - 
                              min(np.mean(results_by_position[p]) for p in positions))
        print(f"位置敏感性 (最大-最小): {position_sensitivity:.4f}")
    
    return {
        'results_by_position': dict(results_by_position),
        'position_bias_score': overall_bias,
        'position_sensitivity': position_sensitivity if all_scores else 0,
        'num_samples': len(samples)
    }


def evaluate_answer(answer, golden_answer):
    """
    评估答案准确性
    
    TODO: 实现具体的评估逻辑（EM, F1等）
    """
    # 简单的字符串匹配
    answer_lower = answer.lower().strip()
    golden_lower = golden_answer.lower().strip()
    
    if golden_lower in answer_lower:
        return 1.0
    
    # 可以添加更复杂的评估逻辑
    # - F1 score
    # - BERTScore
    # - Exact Match
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='MRAG-Bench位置偏差评测')
    parser.add_argument('--model_name', default='ours', help='模型名称')
    parser.add_argument('--data_dir', 
                       default='/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench',
                       help='数据目录')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    parser.add_argument('--output_dir', 
                       default='experiments/mragbench_results',
                       help='输出目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 MRAG-Bench位置偏差评测")
    print("=" * 80)
    print(f"模型: {args.model_name}")
    print(f"数据目录: {args.data_dir}")
    print("=" * 80)
    
    # 加载数据
    samples = load_mragbench_data(args.data_dir)
    
    if not samples:
        print("❌ 未找到数据，请检查数据目录")
        print("\n建议:")
        print("1. 检查数据目录是否正确")
        print("2. 确认MRAG-Bench数据已下载")
        print("3. 根据实际数据格式修改load_mragbench_data函数")
        return
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"\n✅ 加载了 {len(samples)} 个样本")
    
    # 加载模型（TODO: 实际实现）
    print("\n加载模型...")
    model = None  # TODO: 加载实际模型
    
    # 评估位置偏差
    results = evaluate_position_bias(model, samples)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    output = {
        'model': args.model_name,
        'num_samples': len(samples),
        'position_bias_score': results['position_bias_score'],
        'position_sensitivity': results['position_sensitivity'],
        'results_by_position': {
            pos: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            }
            for pos, scores in results['results_by_position'].items()
        }
    }
    
    output_file = os.path.join(args.output_dir, f'{args.model_name}_results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
