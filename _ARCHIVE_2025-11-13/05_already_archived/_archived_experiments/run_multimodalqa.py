#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiModalQA评测脚本
评估跨模态推理能力

数据集: MultiModalQA (29,918 questions, 52,274 images)
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='MultiModalQA评测')
    parser.add_argument('--model_name', default='ours', help='模型名称')
    parser.add_argument('--max_samples', type=int, default=5000, 
                       help='最大样本数（默认5000）')
    parser.add_argument('--data_dir',
                       default='/root/autodl-tmp/FlashRAG/flashrag/data/MultiModalQA',
                       help='数据目录')
    parser.add_argument('--output_dir',
                       default='experiments/multimodalqa_results',
                       help='输出目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 MultiModalQA评测")
    print("=" * 80)
    print(f"模型: {args.model_name}")
    print(f"最大样本数: {args.max_samples}")
    print("=" * 80)
    
    # 加载数据集
    print("\n加载MultiModalQA数据集...")
    try:
        from flashrag.dataset.multimodalqa_dataset import MultiModalQADataset
        
        config = {
            'data_dir': args.data_dir,
            'max_samples': args.max_samples
        }
        
        dataset = MultiModalQADataset(config)
        print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        print("\n可能的原因:")
        print("1. MultiModalQA数据集未下载")
        print("2. multimodalqa_dataset.py加载器未实现")
        print("\n建议:")
        print("1. 确认数据已下载到:", args.data_dir)
        print("2. 检查加载器实现")
        return
    
    # 加载模型
    print("\n加载模型...")
    try:
        from flashrag.modules.mllm_wrapper import LLaVAWrapper
        from flashrag.pipeline.self_aware_pipeline_fixed import SelfAwareMultimodalPipeline
        
        # 模型配置
        model_config = {
            'llava_model_path': '/root/autodl-tmp/models/llava-v1.5-7b',
            'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
            'bge_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
            'uncertainty_threshold': 0.5,
            'use_position_fusion': True,
            'use_attribution': True,
            'use_multimodal_output': False  # MultiModalQA主要测准确率
        }
        
        llava = LLaVAWrapper(model_config['llava_model_path'])
        retriever = None  # TODO: 实现检索器
        
        pipeline = SelfAwareMultimodalPipeline(llava, retriever, model_config)
        print("✅ 模型加载完成")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 运行评测
    print("\n开始评测...")
    print("=" * 80)
    
    results = []
    correct_count = 0
    
    # 统计不同类型问题的性能
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, sample in enumerate(tqdm(dataset[:args.max_samples], desc="评测进度")):
        try:
            # 运行模型
            result = pipeline.run_single(sample)
            
            # 评估
            answer = result['answer']
            if isinstance(answer, dict):
                answer = answer.get('text', '')
            
            answer_lower = answer.lower().strip()
            golden_answers = sample.get('golden_answers', [])
            
            # 检查是否正确
            correct = any(g.lower().strip() in answer_lower 
                         for g in golden_answers)
            
            if correct:
                correct_count += 1
            
            # 按类型统计
            question_type = sample.get('type', 'unknown')
            type_stats[question_type]['total'] += 1
            if correct:
                type_stats[question_type]['correct'] += 1
            
            # 记录结果
            results.append({
                'question_id': sample.get('question_id', i),
                'question': sample['question'],
                'answer': answer,
                'golden_answers': golden_answers,
                'correct': correct,
                'type': question_type,
                'retrieved': result.get('retrieved', False)
            })
            
            # 定期显示进度
            if (i + 1) % 100 == 0:
                current_acc = correct_count / (i + 1)
                print(f"\n当前准确率: {current_acc*100:.2f}% ({correct_count}/{i+1})")
        
        except Exception as e:
            print(f"\n⚠️ 样本{i}处理失败: {e}")
            continue
    
    # 计算总体指标
    total_samples = len(results)
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📊 MultiModalQA评测结果")
    print("=" * 80)
    print(f"\n总样本数: {total_samples}")
    print(f"正确数量: {correct_count}")
    print(f"准确率: {accuracy*100:.2f}%")
    
    # 按类型统计
    if type_stats:
        print("\n各类型准确率:")
        for qtype, stats in sorted(type_stats.items()):
            type_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {qtype:20s}: {type_acc*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    # 检索率
    retrieved_count = sum(1 for r in results if r.get('retrieved', False))
    retrieval_rate = retrieved_count / total_samples if total_samples > 0 else 0
    print(f"\n检索率: {retrieval_rate*100:.1f}% ({retrieved_count}/{total_samples})")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 详细结果
    output_file = os.path.join(args.output_dir, f'{args.model_name}_detailed.json')
    with open(output_file, 'w') as f:
        json.dump({
            'model': args.model_name,
            'total_samples': total_samples,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'retrieval_rate': retrieval_rate,
            'type_stats': dict(type_stats),
            'samples': results[:100]  # 只保存前100个样本示例
        }, f, indent=2, default=str)
    
    # 摘要
    summary_file = os.path.join(args.output_dir, f'{args.model_name}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"MultiModalQA评测结果 - {args.model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"正确数量: {correct_count}\n")
        f.write(f"准确率: {accuracy*100:.2f}%\n")
        f.write(f"检索率: {retrieval_rate*100:.1f}%\n\n")
        
        if type_stats:
            f.write("各类型准确率:\n")
            for qtype, stats in sorted(type_stats.items()):
                type_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"  {qtype}: {type_acc*100:.2f}% ({stats['correct']}/{stats['total']})\n")
    
    print(f"\n✅ 详细结果已保存: {output_file}")
    print(f"✅ 摘要已保存: {summary_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()

