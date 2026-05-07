#!/usr/bin/env python3
"""
MRAG-Bench 基线方法对比实验 - 优化版
主要优化：
1. Self-Aware-MRAG使用更低的不确定性阈值(0.35)
2. RagVL限制文档长度以提高速度
3. 修复了answer_choice字段问题
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
sys.path.insert(0, '/data0/home/zqwang/ACL/MRAG-Bench-main/eval/utils')

# 导入FlashRAG模块
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 导入基线方法
from experiments.baselines.samrag_adapted import SAMRAGAdapted
from experiments.baselines.mr2ag_enhanced import MR2AGEnhanced
from experiments.baselines.vidorag_pipeline import ViDoRAGPipeline
from experiments.baselines.ragvl_enhanced import RagVLEnhanced  # 使用修改后的版本
from experiments.baselines.visrag_pipeline import VisRAGPipeline
from experiments.baselines.murag_pipeline import MuRAGPipeline
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL


# 配置
CONFIG = {
    # 模型配置
    'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'device': 'cuda',
    'temperature': 0.01,
    'max_new_tokens': 10,  # MRAG只需要输出A/B/C/D

    # 不确定性估计器配置 - 降低阈值！
    'use_improved_estimator': True,
    'uncertainty_threshold': 0.35,  # 降低到0.35以鼓励更多检索

    # 不确定性权重配置
    'text_weight': 0.4,
    'visual_weight': 0.4,
    'alignment_weight': 0.2,
}

# 检索器配置
RETRIEVER_CONFIG = {
    'retriever_type': 'bm25',
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/wiki_2023_3m',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-zh-v1.5',
    'top_k': 5,  # 减少检索文档数
}


def load_dataset(dataset_path, max_samples=None):
    """加载MRAG-Bench数据集"""
    print(f"加载数据集: MRAG-Bench")
    print(f"数据路径: {dataset_path}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    try:
        import datasets
        dataset_dict = datasets.load_from_disk(dataset_path)
        test_data = dataset_dict['test']

        if max_samples:
            test_data = test_data.select(range(min(max_samples, len(test_data))))

        # 转换为列表，包含所有字段（重要：保留answer_choice）
        samples = []
        for i, item in enumerate(test_data):
            sample = {
                'id': f"mrag_bench_{i}",
                'question': item['question'],
                'image': item['image'],
                'answer': item['answer'],  # ground truth text
                'answer_choice': item['answer_choice'],  # ground truth letter (important!)
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                # MRAG-Bench特有字段
                'scenario': item.get('scenario', 'Unknown'),
                'choices': ['A', 'B', 'C', 'D'],
            }
            samples.append(sample)

        print(f"✅ Arrow格式加载成功: {len(samples)} 样本")

        # 打印场景分布
        from collections import Counter
        scenarios = Counter([s['scenario'] for s in samples])
        print(f"场景分布: {dict(scenarios)}")

        return samples

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    from flashrag.model.model_loader import create_qwen3_vl_wrapper
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper


def init_retriever(config, use_multimodal=False):
    """初始化检索器"""
    print("初始化检索器...")
    print(f"  模式: {'多模态融合' if use_multimodal else '纯文本'}")

    # 使用现有索引
    index_path = config['index_path']
    if os.path.exists(index_path):
        print(f"✅ 使用现有索引: {index_path}")
    else:
        print(f"⚠️ 索引不存在，将在运行时创建")

    # 初始化BGE文本检索器
    from flashrag.retriever import E5Retriever
    retriever = E5Retriever(
        index_path=index_path,
        corpus_path=config['corpus_path'],
        retrieval_method='e5',
        model_path=config['retrieval_model_path'],
        retrieval_query_max_length=512,
        retrieval_pooling_method='mean',
        retrieval_use_fp16=True,
        retrieval_batch_size=128,
        retrieval_topk=config['top_k'],
        save_retrieval_cache=False,
        use_retrieval_cache=False,
        retrieval_cache_path=None,
    )

    print("✅ 文本检索器加载成功")
    return retriever


def parse_multi_choice_response(response, choices_list, gt_idx=None, index2ans=None, sample=None):
    """
    改进的多选题答案解析，支持内容匹配
    """
    response = response.strip()

    # 1. 首先尝试直接匹配选项
    for choice in choices_list:
        if response == choice:
            return choice

    # 2. 如果有sample，检查是否直接回答了选项内容（重要！）
    if sample:
        for choice in choices_list:
            if choice in sample:
                option_text = sample[choice]
                if option_text and option_text.lower() in response.lower():
                    print(f"[DEBUG] Content-based match: '{option_text}' in response -> {choice}")
                    return choice

    # 3. 尝试各种格式模式
    import re
    patterns = [
        r'答案[是]?[:]*\s*([ABCD])',
        r'选择[是]?[:]*\s*([ABCD])',
        r'Answer is[:]*\s*([ABCD])',
        r'The answer is[:]*\s*([ABCD])',
        r'([ABCD])\.',
        r'^([ABCD])\s*$',
        r'\(([ABCD])\)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return response.strip()


def calculate_mrag_metrics(method_name, results, samples):
    """计算MRAG指标"""
    print(f"\n计算 {method_name} 的指标...")
    print("-" * 60)

    # 统计准确率
    correct = 0
    total = 0
    scenario_correct = {}
    scenario_total = {}

    for i, (result, sample) in enumerate(zip(results, samples)):
        # MRAG-Bench使用answer_choice字段存储选项字母
        gt = sample.get('answer_choice', sample['answer']).upper()
        pred = result.get('answer', '').strip()

        # 使用改进的解析函数
        pred_parsed = parse_multi_choice_response(pred, ['A', 'B', 'C', 'D'], sample=sample)

        # 统计总准确率
        if gt and pred_parsed and gt.upper() == pred_parsed.upper():
            correct += 1
        else:
            # 记录错误案例以便调试
            if i < 3:  # 只打印前3个错误
                print(f"[DEBUG] Sample {i}:")
                print(f"  Question: {sample.get('question', '')[:50]}...")
                print(f"  GT choice: {gt}")
                print(f"  GT text: {sample.get(gt, '')}")
                print(f"  Pred raw: {pred}")
                print(f"  Pred parsed: {pred_parsed}")

        total += 1

        # 分场景统计
        scenario = sample.get('scenario', 'Unknown')
        if scenario not in scenario_correct:
            scenario_correct[scenario] = 0
            scenario_total[scenario] = 0
        scenario_total[scenario] += 1
        if gt and pred_parsed and gt.upper() == pred_parsed.upper():
            scenario_correct[scenario] += 1

    overall_accuracy = correct / total * 100 if total > 0 else 0

    # 使用FlashRAG评估器计算其他指标
    formatted_results = []
    for i, (r, s) in enumerate(zip(results, samples)):
        formatted_result = {
            'answer': r.get('answer', ''),
            'golden_answers': [s.get('answer_choice', s['answer'])],  # MRAG-Bench使用answer_choice
            'retrieved_docs': r.get('retrieved_docs', []),
            'question': s.get('question', ''),
            'id': s.get('id', f'sample_{i}'),
            'choices': s.get('choices', ['A', 'B', 'C', 'D']),
        }

        # 转换retrieved_docs为标准格式
        docs = r.get('retrieved_docs', [])
        if docs:
            formatted_result['retrieved_docs'] = [
                {'contents': doc} if isinstance(doc, str) else {'contents': str(doc)}
                for doc in docs
            ]
        else:
            formatted_result['retrieved_docs'] = []

        formatted_results.append(formatted_result)

    try:
        from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics
        metrics = evaluate_comprehensive_metrics(formatted_results)
    except:
        metrics = {
            'em': correct / total if total > 0 else 0,
            'avg_F1': 0,
            'retrieval_rate': 0,
            'avg_Recall@5': 0,
            'vqa_score': 0,
            'scenario_accuracy': {}
        }

    # 打印指标
    print(f"  ✅ MRAG-Bench Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"  ✅ EM: {metrics.get('em', correct/total):.4f}")
    print(f"  ✅ F1: {metrics.get('avg_F1', 0):.4f}")
    print(f"  ✅ Retrieval Rate: {metrics.get('retrieval_rate', 0):.4f}")
    print(f"  ✅ Recall@5: {metrics.get('avg_Recall@5', 0):.4f}")

    # 打印主要场景的准确率
    print(f"  📊 分场景准确率:")
    for scenario, acc in sorted(metrics.get('scenario_accuracy', {}).items()):
        print(f"    - {scenario}: {acc:.2f}%")

    # 返回所有指标
    unified_metrics = {
        'method': method_name,
        # MRAG-Bench准确率
        'accuracy': overall_accuracy / 100,  # 转换为0-1范围
        'em': metrics.get('em', overall_accuracy / 100),
        'f1': metrics.get('avg_F1', 0),
        'retrieval_rate': metrics.get('retrieval_rate', 0),
        'retrieval_recall_top5': metrics.get('avg_Recall@5', 0),
        'vqa_score': metrics.get('vqa_score', 0),
        'faithfulness': metrics.get('avg_Faithfulness', 0),
        'attribution_precision': metrics.get('avg_Attribution_Precision', 0),
        'position_bias_score': metrics.get('avg_Position_Bias_Score', 0),

        # MRAG-Bench特有：分场景准确率
        'scenario_accuracy': {
            scenario: (scenario_correct[scenario] / scenario_total[scenario] * 100)
            for scenario in scenario_correct
        },
    }

    return unified_metrics


def run_method(method_name, pipeline, samples):
    """运行单个方法"""
    print(f"\n{'='*80}")
    print(f"评测方法: {method_name}")
    print(f"{'='*80}")

    start_time = time.time()
    results = []

    for i, sample in enumerate(samples):
        print(f"\r进度: {i+1}/{len(samples)}", end='', flush=True)
        try:
            result = pipeline.run_single(sample)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] 样本 {i} 失败: {e}")
            results.append({
                'answer': '',
                'retrieved_docs': []
            })

    print()  # 换行
    end_time = time.time()

    # 计算指标
    metrics = calculate_mrag_metrics(method_name, results, samples)
    metrics['execution_time'] = end_time - start_time
    metrics['samples_per_second'] = len(samples) / (end_time - start_time)

    return results, metrics


def main():
    """主函数"""
    print("="*80)
    print("MRAG-Bench 基线方法对比实验 - 优化版")
    print("="*80)

    # 1. 加载数据集
    print("\n1. 加载数据集")
    print("-"*40)
    dataset_path = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw"
    max_samples = 10
    samples = load_dataset(dataset_path, max_samples=max_samples)

    if not samples:
        print("❌ 无法加载数据集，退出")
        return

    # 2. 初始化模型和检索器
    print("\n2. 初始化模型和检索器")
    print("-"*40)

    # 初始化Qwen3-VL
    qwen3_vl = init_qwen3_vl(CONFIG['model_path'])

    # 初始化检索器
    bge_retriever = init_retriever(RETRIEVER_CONFIG, use_multimodal=False)

    # 3. 运行所有方法
    print("\n3. 运行所有方法")
    print("-"*40)

    methods = {
        'Self-Aware-MRAG': lambda: SelfAwarePipelineQwen3VL(
            qwen3_vl, bge_retriever, CONFIG
        ),
        'SAM-RAG': lambda: SAMRAGBaseline(
            qwen3_vl, bge_retriever, {**CONFIG, **{'sam_max_batches': 4}}
        ),
        'MuRAG': lambda: MuRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'VisRAG': lambda: VisRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'ViDoRAG': lambda: ViDoRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'RagVL': lambda: RagVLEnhanced(qwen3_vl, bge_retriever, {
            **CONFIG,
            'use_reranking': False,  # 禁用MLLM reranking
            'rerank_topk': 3,  # 最多保留3篇文档
            'clip_topk': 10,  # 限制检索到10篇
        }),
        'mR2AG': lambda: MR2AGEnhancedBaseline(qwen3_vl, bge_retriever, CONFIG),
    }

    all_results = {}
    all_metrics = {}

    for method_name, pipeline_factory in methods.items():
        try:
            pipeline = pipeline_factory()
            results, metrics = run_method(method_name, pipeline, samples)
            all_results[method_name] = results
            all_metrics[method_name] = metrics

            # 打印结果
            print(f"\n✅ {method_name} 完成:")
            print(f"   准确率: {metrics['accuracy']*100:.2f}%")
            print(f"   EM: {metrics['em']:.4f}")
            print(f"   F1: {metrics['f1']:.4f}")
            print(f"   耗时: {metrics['execution_time']:.1f}秒 ({metrics['samples_per_second']:.2f}样本/秒)")

        except Exception as e:
            print(f"\n❌ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 4. 保存结果
    print("\n4. 保存结果")
    print("-"*40)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_optimized")
    output_dir.mkdir(exist_ok=True)

    # 保存指标
    import json
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 指标已保存: {metrics_file}")

    # 打印汇总
    print(f"\n{'='*80}")
    print("📊 实验结果汇总（优化版）")
    print(f"{'='*80}")
    print(f"{'方法':<20} {'准确率':<10} {'EM':<10} {'F1':<10} {'耗时(秒)':<10}")
    print("-"*80)

    for method, metrics in all_metrics.items():
        print(f"{method:<20} {metrics['accuracy']*100:>8.2f}% "
              f"{metrics['em']:>8.4f} {metrics['f1']:>8.4f} "
              f"{metrics.get('execution_time', 0):>8.1f}")

    print(f"\n🎯 最佳方法: {max(all_metrics.items(), key=lambda x: x[1]['accuracy'])[0]}")
    print(f"   准确率: {max(all_metrics.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="运行优化版MRAG实验")
    parser.add_argument('--max_samples', type=int, default=10, help='最大样本数')
    args = parser.parse_args()
    main()