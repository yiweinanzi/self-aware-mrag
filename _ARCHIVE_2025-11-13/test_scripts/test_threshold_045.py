#!/usr/bin/env python3
"""
测试优化后的Self-Aware-MRAG (threshold=0.45)
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from FlashRAG.experiments.run_all_baselines_100samples import *

if __name__ == '__main__':
    print("="*80)
    print("测试 Self-Aware-MRAG (threshold=0.45)")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: 全部(1353)")
    print(f"Threshold: 0.45 (优化后)")
    print("="*80)
    
    # 加载数据
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    print(f"\n✅ 加载数据: {len(samples)} 样本")
    
    # 初始化模型
    print("\n初始化模型...")
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])
    multimodal_retriever = init_retriever(CONFIG, use_multimodal=True)
    
    # 创建pipeline（threshold=0.45）
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=multimodal_retriever,
        config={
            'uncertainty_threshold': 0.45,  # ✅ 优化后
            'use_improved_estimator': False,
            'use_position_fusion': True,
            'use_attribution': True,
            'enable_multimodal_output': False,
            'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
            'retrieval_topk': 5,
            'thinking': False,
            'max_images': 20,
        }
    )
    
    # 运行测试
    print("\n开始测试...")
    results, elapsed = run_method("Self-Aware-MRAG (threshold=0.45)", pipeline, samples)
    
    # 计算指标
    print("\n计算指标...")
    metrics = calculate_metrics("Self-Aware-MRAG", results, samples)
    
    # 输出结果
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    print(f"EM: {metrics['em']:.4f} ({metrics['em']*100:.2f}%)")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"VQA-Score: {metrics.get('vqa_score', 0):.4f}")
    print(f"时间: {elapsed/len(samples):.2f}秒/样本")
    
    # 统计检索
    retrieve_count = sum(1 for r in results if r.get('retrieved', False))
    print(f"检索率: {retrieve_count}/{len(samples)} ({retrieve_count/len(samples)*100:.2f}%)")
    
    # 对比之前
    print("\n" + "="*80)
    print("对比（threshold=0.35 → 0.45）")
    print("="*80)
    print(f"EM: 47.97% → {metrics['em']*100:.2f}% (差异: {(metrics['em']-0.4797)*100:+.2f}%)")
    print(f"检索率: 93.57% → {retrieve_count/len(samples)*100:.2f}% (差异: {(retrieve_count/len(samples)-0.9357)*100:+.2f}%)")
    print("="*80)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
