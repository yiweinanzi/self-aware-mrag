#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行OK-VQA七个Baseline方法对比测试 - 修复版
基于消融实验的成功代码

七个方法：
1. Self-Aware-MRAG (Ours)
2. MuRAG
3. VisRAG
4. ViDoRAG
5. RagVL
6. SAM-RAG
7. mR²AG

七个核心指标：
1. EM (Exact Match)
2. F1 Score
3. Recall@5
4. VQA-Score
5. Faithfulness
6. Attribution Precision
7. Position Bias Score
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入基于消融实验的成功模块
from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# ============================================================================
# 配置参数 - 基于消融实验的最佳配置
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',
    'max_samples': 10,  # 10个样本测试
    'load_images': True,  # 加载图像

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.01,

    # 检索器配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # 不确定性配置
    'uncertainty_threshold': 0.43,
    'use_improved_estimator': True,
    'text_weight': 0.4,
    'visual_weight': 0.3,
    'alignment_weight': 0.3,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines',
}

# ============================================================================
# 数据加载 - 使用消融实验的成功方法
# ============================================================================

def load_dataset(dataset_path, max_samples=None):
    """加载OK-VQA数据集 - 使用消融实验的成功方法"""
    print(f"加载数据集: OK-VQA")
    print(f"数据路径: {dataset_path}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    # 使用消融实验成功的数据加载方式
    try:
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        dataset = OKVQADatasetSimple({
            'data_dir': dataset_path,
            'split': 'val',  # 使用val split
            'load_images': True,  # 加载图像
        })

        # 转换为样本列表
        samples = []
        for i in range(min(max_samples if max_samples else len(dataset), len(dataset))):
            item = dataset[i]
            sample = {
                'id': item['id'],
                'question': item['question'],
                'image': item.get('image'),
                'answer': item.get('answer', ''),
                'golden_answers': item['golden_answers']
            }
            samples.append(sample)

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('image') is not None for s in samples)}")
        return samples

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# 模型和检索器初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper


def init_retriever(config):
    """初始化检索器 - 基于消融实验的成功方法"""
    print("初始化检索器...")

    # 使用消融实验的检索器配置
    retriever_config = {
        'index_path': config['faiss_index_path'],
        'corpus_path': config['corpus_path'],
        'retrieval_method': 'e5',
        'retrieval_model_path': config['retrieval_model_path'],
        'retrieval_query_max_length': 512,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 128,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': False,
        'use_sentence_transformer': False,
        'faiss_gpu': False,
        'instruction': '',
    }

    try:
        from flashrag.retriever import DenseRetriever
        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")
        return retriever
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Baseline方法实现
# ============================================================================

class MuRAGPipeline:
    """MuRAG Baseline"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')

        # 简单检索
        retrieved_docs = []
        if self.retriever:
            try:
                search_results = self.retriever.search(question, num=5, return_score=True)
                if isinstance(search_results, tuple):
                    retrieved_docs, _ = search_results
                else:
                    retrieved_docs = search_results if search_results else []
            except Exception as e:
                print(f"检索失败: {e}")
                retrieved_docs = []

        # 构建上下文
        context = ""
        if retrieved_docs:
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:3]):
                doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
                context_parts.append(f"Document {i+1}: {doc_text[:200]}")
            context = "\n\n".join(context_parts)

        # 生成答案
        if context:
            prompt = f"Based on the following context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=20,
                temperature=0.01
            ).strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved': len(retrieved_docs) > 0
        }

class VisRAGPipeline:
    """VisRAG Baseline - 纯视觉检索"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.config = config
        # 模拟视觉检索（实际应该使用CLIP等）

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')

        # VisRAG通常只使用图像信息
        # 这里简化为基于图像的问答
        prompt = f"Look at the image and answer: {question}\n\nAnswer:"

        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=20,
                temperature=0.01
            ).strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': [],  # VisRAG不使用文本检索
            'retrieved': False
        }

class ViDoRAGPipeline:
    """ViDoRAG Baseline - 替代REVEAL"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')

        # 增强的多模态检索
        retrieved_docs = []
        if self.retriever:
            # 多轮查询扩展
            queries = [question, f"visual {question}", f"{question} in image"]
            all_docs = []

            for query in queries:
                try:
                    search_results = self.retriever.search(query, num=3, return_score=True)
                    if isinstance(search_results, tuple):
                        docs, _ = search_results
                        all_docs.extend(docs)
                except:
                    pass

            # 去重
            retrieved_docs = []
            seen_ids = set()
            for doc in all_docs:
                doc_id = doc.get('id', str(doc))
                if doc_id not in seen_ids:
                    retrieved_docs.append(doc)
                    seen_ids.add(doc_id)

        # 增强的prompt
        if retrieved_docs:
            context_parts = ["Visual and textual evidence:"]
            for i, doc in enumerate(retrieved_docs[:5]):
                doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
                context_parts.append(f"Evidence {i+1}: {doc_text[:150]}...")
            context = "\n\n".join(context_parts)

            prompt = f"""Based on the visual evidence in the image and the textual evidence below:

{context}

Question: {question}

Provide a concise answer:"""
        else:
            prompt = f"Based on the image, answer: {question}"

        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=25,
                temperature=0.01
            ).strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved': len(retrieved_docs) > 0
        }

class RagVLPipeline:
    """RagVL Baseline"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')

        # RagVL特定的检索策略
        retrieved_docs = []
        if self.retriever:
            try:
                # 使用问题关键词进行检索
                keywords = question.split()[:3]  # 取前3个词作为关键词
                query = " ".join(keywords)
                search_results = self.retriever.search(query, num=5, return_score=True)
                if isinstance(search_results, tuple):
                    retrieved_docs, _ = search_results
                else:
                    retrieved_docs = search_results if search_results else []
            except Exception as e:
                print(f"检索失败: {e}")
                retrieved_docs = []

        # RagVL的prompt策略
        if retrieved_docs:
            context = "\n".join([
                doc.get('contents', '')[:100] if isinstance(doc, dict) else str(doc)[:100]
                for doc in retrieved_docs[:3]
            ])

            prompt = f"""Context: {context}

Image Analysis: [Analyze the image content]

Based on both the context and image, answer: {question}

Answer:"""
        else:
            prompt = f"Analyze the image and answer: {question}\nAnswer:"

        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=20,
                temperature=0.01
            ).strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved': len(retrieved_docs) > 0
        }

class SAMRAGPipeline:
    """SAM-RAG Baseline - 替代Self-RAG"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')

        # SAM-RAG的检索步骤
        retrieved_docs = []
        if self.retriever:
            try:
                search_results = self.retriever.search(question, num=5, return_score=True)
                if isinstance(search_results, tuple):
                    retrieved_docs, scores = search_results
                else:
                    retrieved_docs = search_results if search_results else []
                    scores = [1.0] * len(retrieved_docs)
            except Exception as e:
                print(f"检索失败: {e}")
                retrieved_docs = []

        # SAM-RAG的critique-revise循环（简化版）
        if retrieved_docs:
            context = "\n".join([
                f"[{i+1}] {doc.get('contents', '')[:150] if isinstance(doc, dict) else str(doc)[:150]}"
                for i, doc in enumerate(retrieved_docs[:3])
            ])

            # 第一版答案
            prompt1 = f"""Based on the evidence:
{context}

Question: {question}

Answer:"""

            try:
                answer1 = self.qwen3_vl.generate(
                    text=prompt1,
                    image=image,
                    max_new_tokens=15,
                    temperature=0.1
                ).strip()

                # 简单的critique
                critique_prompt = f"""Is "{answer1}" a good answer to "{question}" based on the evidence? (yes/no)"""

                critique = self.qwen3_vl.generate(
                    text=critique_prompt,
                    max_new_tokens=5,
                    temperature=0.0
                ).strip().lower()

                # 如果critique为no，尝试改进
                if 'no' in critique:
                    revise_prompt = f"""Based on the evidence:
{context}

Improve this answer to "{question}". Previous answer: "{answer1}" was not good.

Better answer:"""

                    answer = self.qwen3_vl.generate(
                        text=revise_prompt,
                        image=image,
                        max_new_tokens=15,
                        temperature=0.01
                    ).strip()
                else:
                    answer = answer1

            except Exception as e:
                print(f"生成失败: {e}")
                answer = answer1 if 'answer1' in locals() else ""
        else:
            prompt = f"Answer: {question}"
            try:
                answer = self.qwen3_vl.generate(
                    text=prompt,
                    image=image,
                    max_new_tokens=15,
                    temperature=0.01
                ).strip()
            except Exception as e:
                print(f"生成失败: {e}")
                answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved': len(retrieved_docs) > 0
        }

class MR2AGPipeline:
    """mR²AG Baseline"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')

        # mR²AG的多轮检索
        retrieved_docs = []
        if self.retriever:
            # 第一轮：原始问题
            try:
                search_results = self.retriever.search(question, num=3, return_score=True)
                if isinstance(search_results, tuple):
                    docs1, _ = search_results
                else:
                    docs1 = search_results if search_results else []

                # 第二轮：基于第一轮结果的查询
                if docs1:
                    refined_query = f"{question} related to {docs1[0].get('contents', '')[:50]}"
                    search_results2 = self.retriever.search(refined_query, num=2, return_score=True)
                    if isinstance(search_results2, tuple):
                        docs2, _ = search_results2
                    else:
                        docs2 = search_results2 if search_results2 else []

                    retrieved_docs = docs1 + docs2
                else:
                    retrieved_docs = docs1

            except Exception as e:
                print(f"检索失败: {e}")
                retrieved_docs = []

        # mR²AG的推理链prompt
        if retrieved_docs:
            context_parts = ["Evidence:"]
            for i, doc in enumerate(retrieved_docs[:5]):
                doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
                context_parts.append(f"Step {i+1}: {doc_text[:100]}")

            context = "\n".join(context_parts)

            prompt = f"""Let's think step by step to answer this question.

{context}

Question: {question}

Step 1: Analyze the question.
Step 2: Consider the evidence.
Step 3: Formulate the answer.

Final Answer:"""
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=30,
                temperature=0.01
            ).strip()

            # 提取最终答案（去除推理步骤）
            if "Final Answer:" in answer:
                answer = answer.split("Final Answer:")[-1].strip()

        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved': len(retrieved_docs) > 0
        }

# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有Baseline方法对比"""
    print("="*80)
    print("OK-VQA 七个Baseline方法对比测试")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")

    # 1. 加载数据集（基于消融实验的方式）
    print("\n1. 加载数据集")
    print("-" * 40)
    try:
        dataset = OKVQADatasetSimple({
            'data_dir': CONFIG['data_dir'],
            'split': CONFIG['split'],
            'load_images': CONFIG['load_images'],
        })

        # 限制样本数
        samples = []
        for i in range(min(CONFIG['max_samples'], len(dataset))):
            sample = dataset[i]
            samples.append({
                'id': sample['id'],
                'question': sample['question'],
                'image': sample['image'],
                'answer': sample.get('answer', ''),
                'golden_answers': sample['golden_answers']
            })

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('image') is not None for s in samples)}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 初始化模型和检索器
    print("\n2. 初始化模型和检索器")
    print("-" * 40)

    # Qwen3-VL
    try:
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=CONFIG['qwen3_vl_path'],
            device='cuda',
            torch_dtype=CONFIG['torch_dtype']
        )
        print("✅ Qwen3-VL加载成功")
    except Exception as e:
        print(f"❌ Qwen3-VL加载失败: {e}")
        return

    # 检索器
    try:
        retriever_config = {
            'index_path': CONFIG['faiss_index_path'],
            'corpus_path': CONFIG['corpus_path'],
            'retrieval_method': 'e5',
            'retrieval_model_path': CONFIG['retrieval_model_path'],
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 128,
            'retrieval_topk': CONFIG['retrieval_topk'],
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            'use_reranker': False,
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
        }

        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        retriever = None

    # 3. 定义七个Baseline方法
    print("\n3. 初始化七个Baseline方法")
    print("-" * 40)

    baselines = {
        'Self-Aware-MRAG (Ours)': lambda: SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config={
                'uncertainty_threshold': CONFIG['uncertainty_threshold'],
                'use_improved_estimator': CONFIG['use_improved_estimator'],
                'use_position_fusion': True,
                'use_attribution': True,
            }
        ),
        'MuRAG': lambda: MuRAGPipeline(qwen3_vl, retriever, CONFIG),
        'VisRAG': lambda: VisRAGPipeline(qwen3_vl, retriever, CONFIG),
        'ViDoRAG': lambda: ViDoRAGPipeline(qwen3_vl, retriever, CONFIG),
        'RagVL': lambda: RagVLPipeline(qwen3_vl, retriever, CONFIG),
        'SAM-RAG': lambda: SAMRAGPipeline(qwen3_vl, retriever, CONFIG),
        'mR²AG': lambda: MR2AGPipeline(qwen3_vl, retriever, CONFIG),
    }

    print(f"✅ 初始化了 {len(baselines)} 个Baseline方法")

    # 4. 运行所有方法
    print("\n4. 运行所有Baseline方法")
    print("-" * 40)

    all_results = {}
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    for method_name, method_factory in baselines.items():
        print(f"\n{'='*60}")
        print(f"运行方法: {method_name}")
        print(f"{'='*60}")

        try:
            # 初始化方法
            pipeline = method_factory()
            print(f"✅ {method_name} 初始化成功")

            # 运行测试
            start_time = time.time()
            results = []

            for i, sample in enumerate(samples):
                print(f"\r进度: {i+1}/{len(samples)}", end='', flush=True)
                result = pipeline.run_single(sample)

                # 确保结果包含必要字段
                if 'golden_answers' not in result:
                    result['golden_answers'] = sample['golden_answers']
                results.append(result)

            elapsed_time = time.time() - start_time

            # 评估
            print(f"\n\n评估 {method_name}...")
            metrics = evaluate_comprehensive_metrics(results)

            # 保存结果
            method_result = {
                'method': method_name,
                'config': CONFIG,
                'metrics': metrics,
                'results': results,
                'elapsed_time': elapsed_time,
                'samples_per_second': len(results) / elapsed_time if elapsed_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }

            all_results[method_name] = method_result

            # 保存单个方法结果
            output_file = os.path.join(
                CONFIG['output_dir'],
                f"{method_name.replace(' ', '_').replace('(Ours)', 'Self_Aware_MRAG')}_results.json"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(method_result, f, indent=2, ensure_ascii=False)

            print(f"\n✅ {method_name} 完成")
            print(f"   耗时: {elapsed_time:.1f}秒")
            print(f"   速度: {method_result['samples_per_second']:.2f} 样本/秒")

        except Exception as e:
            print(f"\n❌ {method_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 保存汇总结果
    print("\n\n5. 保存汇总结果")
    print("-" * 40)

    # 创建汇总报告
    summary = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(samples),
        'methods': {}
    }

    # 提取关键指标
    print("\n关键指标汇总:")
    print("-" * 80)
    print(f"{'方法':<20} {'EM':<8} {'F1':<8} {'Recall@5':<10} {'VQA-Score':<11} {'Faithfulness':<12} {'Attribution':<12} {'PositionBias':<12}")
    print("-" * 80)

    for method_name, result in all_results.items():
        metrics = result.get('metrics', {})

        summary['methods'][method_name] = {
            'EM': metrics.get('em', 0),
            'F1': metrics.get('avg_F1', 0),
            'Recall@5': metrics.get('retrieval_recall_top5', 0),
            'VQA-Score': metrics.get('avg_VQA_Score', 0),
            'Faithfulness': metrics.get('avg_Faithfulness', 0),
            'Attribution Precision': metrics.get('avg_Attribution_Precision', 0),
            'Position Bias Score': metrics.get('avg_Position_Bias_Score', 0),
        }

        print(f"{method_name:<20} "
              f"{metrics.get('em', 0):<8.4f} "
              f"{metrics.get('avg_F1', 0):<8.4f} "
              f"{metrics.get('retrieval_recall_top5', 0):<10.4f} "
              f"{metrics.get('avg_VQA_Score', 0):<11.4f} "
              f"{metrics.get('avg_Faithfulness', 0):<12.4f} "
              f"{metrics.get('avg_Attribution_Precision', 0):<12.4f} "
              f"{metrics.get('avg_Position_Bias_Score', 0):<12.4f}")

    # 保存汇总
    summary_file = os.path.join(CONFIG['output_dir'], 'all_baselines_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 汇总结果已保存到: {summary_file}")

    # 6. 完成
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"完成方法数: {len(all_results)}")
    print(f"输出目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()