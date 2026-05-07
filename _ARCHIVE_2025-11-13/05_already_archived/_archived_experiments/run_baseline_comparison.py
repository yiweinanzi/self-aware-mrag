#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline对比实验
对比5个SOTA方法: MuRAG, mR²AG, VisRAG, REVEAL, RagVL

生成主结果表（Table 1）
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import argparse
from tqdm import tqdm
import time

def run_single_baseline(baseline_name, dataset, max_samples=500):
    """
    运行单个baseline
    
    Args:
        baseline_name: baseline名称
        dataset: 数据集
        max_samples: 最大样本数
    
    Returns:
        dict: 结果字典
    """
    print(f"\n{'='*80}")
    print(f"运行 {baseline_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 导入对应的baseline
        if baseline_name == 'murag':
            from experiments.baselines.murag_baseline import MuRAGBaseline
            model = MuRAGBaseline()
            
        elif baseline_name == 'mr2ag':
            from experiments.baselines.mr2ag_baseline import MR2AGBaseline
            model = MR2AGBaseline()
            
        elif baseline_name == 'visrag':
            from experiments.baselines.visrag_baseline import VisRAGBaseline
            model = VisRAGBaseline()
            
        elif baseline_name == 'reveal':
            from experiments.baselines.reveal_baseline import REVEALBaseline
            model = REVEALBaseline()
            
        elif baseline_name == 'ragvl':
            from experiments.baselines.ragvl_baseline import RagVLBaseline
            model = RagVLBaseline()
            
        else:
            raise ValueError(f"未知的baseline: {baseline_name}")
        
        print(f"✅ {baseline_name}模型加载完成")
        
    except Exception as e:
        print(f"❌ 加载{baseline_name}失败: {e}")
        print(f"   可能原因: baseline实现文件不存在")
        print(f"   跳过{baseline_name}...")
        return None
    
    # 运行评测
    results = []
    correct_count = 0
    
    for i, sample in enumerate(tqdm(dataset[:max_samples], 
                                   desc=f"{baseline_name}评测")):
        try:
            # 生成答案
            answer = model.generate(sample)
            
            # 评估
            answer_lower = answer.lower().strip()
            golden_answers = sample.get('golden_answers', [])
            correct = any(g.lower().strip() in answer_lower 
                         for g in golden_answers)
            
            if correct:
                correct_count += 1
            
            results.append({
                'question': sample['question'],
                'answer': answer,
                'golden_answers': golden_answers,
                'correct': correct
            })
            
        except Exception as e:
            if i < 5:  # 只显示前5个错误
                print(f"\n⚠️ 样本{i}处理失败: {e}")
            results.append({
                'question': sample.get('question', ''),
                'answer': '',
                'golden_answers': sample.get('golden_answers', []),
                'correct': False
            })
    
    # 计算指标
    total_samples = len(results)
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{baseline_name}结果:")
    print(f"  准确率: {accuracy*100:.2f}% ({correct_count}/{total_samples})")
    print(f"  用时: {elapsed_time/60:.1f}分钟")
    
    return {
        'baseline': baseline_name,
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_samples': total_samples,
        'time_minutes': elapsed_time / 60,
        'samples': results[:20]  # 保存前20个样本示例
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline对比实验')
    parser.add_argument('--baselines', nargs='+',
                       default=['murag', 'mr2ag', 'visrag', 'reveal', 'ragvl'],
                       help='要运行的baseline列表')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='每个baseline的最大样本数')
    parser.add_argument('--dataset', default='okvqa',
                       help='数据集名称')
    parser.add_argument('--output_dir',
                       default='experiments/baseline_comparison',
                       help='输出目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🏆 Baseline对比实验")
    print("=" * 80)
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"每个baseline样本数: {args.max_samples}")
    print(f"数据集: {args.dataset}")
    print("=" * 80)
    
    # 加载数据集
    print("\n加载数据集...")
    try:
        if args.dataset == 'okvqa':
            from flashrag.dataset.okvqa_dataset_lazy import OKVQADataset
            config = {'data_dir': 'flashrag/data/VQA'}
            dataset = OKVQADataset(config)
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")
        
        print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 运行所有baseline
    all_results = []
    
    for baseline in args.baselines:
        result = run_single_baseline(baseline, dataset, args.max_samples)
        
        if result is not None:
            all_results.append(result)
        else:
            print(f"⚠️ {baseline}跳过")
    
    # 添加我们的方法（从已有结果）
    # 注意: 这里使用已经运行的实验结果
    our_result = {
        'baseline': 'ours_full',
        'accuracy': 0.6890,  # 68.90% from ablation_500_5M
        'correct_count': 319,
        'total_samples': 463,
        'time_minutes': None,
        'note': '来自ablation_500_5M实验'
    }
    all_results.append(our_result)
    
    # 按准确率排序
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # 生成对比表
    print("\n" + "=" * 80)
    print("📊 主结果表（Table 1）")
    print("=" * 80)
    print(f"{'Method':<20} {'Accuracy':<12} {'Correct/Total':<15} {'Time(min)'}")
    print("-" * 80)
    
    for r in all_results:
        method_name = r['baseline'].upper().replace('_', ' ')
        accuracy_str = f"{r['accuracy']*100:.2f}%"
        count_str = f"{r['correct_count']}/{r['total_samples']}"
        time_str = f"{r['time_minutes']:.1f}" if r.get('time_minutes') else "N/A"
        
        print(f"{method_name:<20} {accuracy_str:<12} {count_str:<15} {time_str}")
    
    print("=" * 80)
    
    # 计算相对提升
    if len(all_results) > 1:
        best_baseline = max([r for r in all_results if r['baseline'] != 'ours_full'],
                           key=lambda x: x['accuracy'])
        our_method = [r for r in all_results if r['baseline'] == 'ours_full'][0]
        
        improvement = (our_method['accuracy'] - best_baseline['accuracy']) * 100
        relative_improvement = (our_method['accuracy'] / best_baseline['accuracy'] - 1) * 100
        
        print(f"\n💡 关键发现:")
        print(f"  最佳Baseline: {best_baseline['baseline'].upper()} ({best_baseline['accuracy']*100:.2f}%)")
        print(f"  我们的方法: {our_method['accuracy']*100:.2f}%")
        print(f"  绝对提升: +{improvement:.2f}%")
        print(f"  相对提升: +{relative_improvement:.1f}%")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # JSON格式
    output_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'max_samples_per_baseline': args.max_samples,
            'results': all_results
        }, f, indent=2, default=str)
    
    # Markdown表格
    md_file = os.path.join(args.output_dir, 'comparison_table.md')
    with open(md_file, 'w') as f:
        f.write("# Baseline对比结果\n\n")
        f.write(f"**数据集**: {args.dataset.upper()}\n")
        f.write(f"**样本数**: {args.max_samples}\n\n")
        f.write("| Method | Accuracy | Correct/Total | Time(min) |\n")
        f.write("|--------|----------|---------------|----------|\n")
        
        for r in all_results:
            method = r['baseline'].upper().replace('_', ' ')
            acc = f"{r['accuracy']*100:.2f}%"
            count = f"{r['correct_count']}/{r['total_samples']}"
            time_val = f"{r['time_minutes']:.1f}" if r.get('time_minutes') else "N/A"
            f.write(f"| {method} | {acc} | {count} | {time_val} |\n")
    
    print(f"\n✅ 结果已保存:")
    print(f"   JSON: {output_file}")
    print(f"   Markdown: {md_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
