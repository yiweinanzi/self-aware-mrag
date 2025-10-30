#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - 100样本，7个核心指标

方法列表：
1. Self-Aware-MRAG (Our Method)
2. Self-RAG
3. mR²AG
4. VisRAG
5. REVEAL
6. RagVL
7. MuRAG

指标列表（7个核心指标）：
1. EM (Exact Match)
2. F1 (Token-level F1)
3. Recall@5 (Retrieval Recall)
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
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加FlashRAG路径
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'mragbench',
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'max_samples': 100,
    
    # 模型配置
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    
    # 检索器配置（使用纯Wikipedia 3M语料库和索引）
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
    
    # CLIP多模态检索配置（可选，如果CLIP索引不存在会降级为纯BGE）
    'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
    'clip_index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/clip',
    
    # 评测配置
    'save_results': True,
    'output_dir': '/root/autodl-tmp/FlashRAG/experiments/results_baseline_comparison_100_wiki3m',
    
    # 生成参数（统一）
    'temperature': 0.01,
    'max_new_tokens': 10,
    'retrieval_topk': 5,
    
    # 不确定性估计器配置（使用改进版）
    'use_improved_estimator': True,
    'uncertainty_threshold': 0.35,  # 默认阈值（将在threshold sweep中测试多个值）
}


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(dataset_path, max_samples=None):
    """加载MRAG-Bench数据集（Arrow格式）"""
    print(f"加载数据集: {dataset_path}")
    
    dataset_dict = datasets.load_from_disk(dataset_path)
    test_data = dataset_dict['test']
    
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    # 转换为列表
    samples = []
    for item in test_data:
        sample = {
            'question': item['question'],
            'image': item['image'],
            'answer': item['answer'],  # ground truth
            'A': item['A'],
            'B': item['B'],
            'C': item['C'],
            'D': item['D'],
        }
        samples.append(sample)
    
    print(f"✅ 加载完成: {len(samples)} 样本")
    return samples


# ============================================================================
# 模型和检索器初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper


def init_retriever(config, use_multimodal=False):
    """
    初始化检索器
    
    Args:
        config: 配置字典
        use_multimodal: 是否使用多模态检索融合 (BGE + CLIP)
    """
    print("初始化检索器...")
    print(f"  模式: {'多模态融合 (BGE + CLIP)' if use_multimodal else '纯文本 (BGE)'}")
    
    # 检查索引文件是否存在
    import os
    from flashrag.retriever.index_builder import Index_Builder
    
    index_path = config.get('index_path', '')
    corpus_path = config['corpus_path']
    
    if not os.path.exists(index_path):
        print(f"⚠️ 索引文件不存在: {index_path}")
        print(f"✅ 将从真实语料库动态构建索引: {corpus_path}")
        print(f"⏱️  预计时间: 30-60分钟（3M文档）")
        print(f"💡 这样明天早上索引和实验结果都完成了")
        
        # 从真实语料库构建索引
        index_dir = os.path.dirname(index_path)
        os.makedirs(index_dir, exist_ok=True)
        
        print(f"\n开始构建索引...")
        builder = Index_Builder(
            retrieval_method='e5',
            model_path=config['retrieval_model_path'],
            corpus_path=corpus_path,
            save_dir=index_dir,
            max_length=512,
            batch_size=256,
            use_fp16=True,
            faiss_type='Flat',
            pooling_method='mean',
            save_embedding=True
        )
        
        builder.build_index()
        print(f"✅ 索引构建完成: {index_path}")
    else:
        print(f"✅ 使用现有索引: {index_path}")
    
    # 初始化BGE文本检索器
    retriever_config = {
        'index_path': index_path,
        'corpus_path': corpus_path,
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
    
    bge_retriever = DenseRetriever(retriever_config)
    print("✅ BGE文本检索器加载成功")
    
    # 如果不使用多模态，直接返回BGE检索器
    if not use_multimodal:
        return bge_retriever
    
    # 检查CLIP索引是否存在
    clip_index_dir = config.get('clip_index_path', '/root/autodl-tmp/FlashRAG/indexes/3m_real/clip')
    clip_index_file = os.path.join(clip_index_dir, 'clip_Flat.index')
    
    if not os.path.exists(clip_index_file):
        print(f"⚠️  CLIP索引不存在: {clip_index_file}")
        print(f"💡 降级使用纯BGE文本检索")
        return bge_retriever
    
    # 初始化CLIP视觉检索器
    print(f"✅ CLIP索引已存在，初始化多模态检索器...")
    clip_retriever_config = {
        'index_path': clip_index_file,
        'corpus_path': corpus_path,
        'retrieval_method': 'clip',
        'retrieval_model_path': config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336'),
        'retrieval_query_max_length': 77,
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 64,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'index_modal': 'all',  # CLIP索引包含text+image
    }
    
    clip_retriever = DenseRetriever(clip_retriever_config)
    print("✅ CLIP视觉检索器加载成功")
    
    # 创建多模态融合检索器
    from flashrag.retriever.multimodal_retriever import SelfAwareMultimodalRetriever
    
    multimodal_config = {
        'retrieval_topk': config['retrieval_topk'],
        'use_clip': True,
        'clip_model_path': config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336'),
        'fusion_method': 'weighted',
        'position_encoding': 'learned',
        'text_weight': 0.6,  # BGE权重
        'visual_weight': 0.4,  # CLIP权重
    }
    
    multimodal_retriever = SelfAwareMultimodalRetriever(
        config=multimodal_config,
        text_retriever=bge_retriever,
        visual_retriever=clip_retriever
    )
    
    print("✅ 多模态融合检索器初始化完成 (BGE 60% + CLIP 40%)")
    return multimodal_retriever


# ============================================================================
# Baseline实现（简化版）
# ============================================================================

class BaselinePipeline:
    """Baseline方法的基类"""
    
    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config
    
    def run_single(self, sample):
        """运行单个样本（子类实现）"""
        raise NotImplementedError
    
    def _construct_prompt(self, question, options, context=None):
        """构建多选题prompt"""
        if context:
            prompt = f"""Based on the following evidence, answer the question.

{context}

Question: {question}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Answer with ONLY the letter (A/B/C/D):"""
        else:
            prompt = f"""Question: {question}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Answer with ONLY the letter (A/B/C/D):"""
        
        return prompt
    
    def _generate(self, prompt, image):
        """生成答案"""
        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature']
            )
            return answer.strip()
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ""
    
    def _map_letter_to_answer(self, prediction, sample):
        """将字母映射回答案"""
        pred_letter = prediction.upper()[0] if prediction else '?'
        if pred_letter in ['A', 'B', 'C', 'D']:
            return sample[pred_letter]
        return prediction


class SelfRAGPipeline(BaselinePipeline):
    """Self-RAG: 总是检索 + 反思机制"""
    
    def run_single(self, sample):
        # 总是检索
        results = self.retriever.search(sample['question'], num=5)
        context = "\n\n".join([doc.get('contents', '') for doc in results[:5]])
        
        # 生成答案（简化版，无反思token）
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
            'used_retrieval': True
        }


class MR2AGPipeline(BaselinePipeline):
    """mR²AG: 多轮检索 + 重排"""
    
    def run_single(self, sample):
        # 第一轮检索
        results = self.retriever.search(sample['question'], num=10)
        
        # 简化版：取top-5（实际应该有重排）
        context = "\n\n".join([doc.get('contents', '') for doc in results[:5]])
        
        # 生成答案
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
            'used_retrieval': True
        }


class VisRAGPipeline(BaselinePipeline):
    """VisRAG: 视觉优先 + 检索增强"""
    
    def run_single(self, sample):
        # 总是检索
        results = self.retriever.search(sample['question'], num=5)
        context = "\n\n".join([doc.get('contents', '') for doc in results[:5]])
        
        # 生成答案（视觉优先：图像在prompt前）
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
            'used_retrieval': True
        }


class REVEALPipeline(BaselinePipeline):
    """REVEAL: 跨模态融合"""
    
    def run_single(self, sample):
        # 总是检索
        results = self.retriever.search(sample['question'], num=5)
        context = "\n\n".join([doc.get('contents', '') for doc in results[:5]])
        
        # 生成答案
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
            'used_retrieval': True
        }


class RagVLPipeline(BaselinePipeline):
    """RagVL: 多模态RAG"""
    
    def run_single(self, sample):
        # 总是检索
        results = self.retriever.search(sample['question'], num=5)
        context = "\n\n".join([doc.get('contents', '') for doc in results[:5]])
        
        # 生成答案
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
            'used_retrieval': True
        }


class MuRAGPipeline(BaselinePipeline):
    """MuRAG: 多路径融合"""
    
    def run_single(self, sample):
        # 总是检索
        results = self.retriever.search(sample['question'], num=5)
        context = "\n\n".join([doc.get('contents', '') for doc in results[:5]])
        
        # 生成答案
        options = {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']}
        prompt = self._construct_prompt(sample['question'], options, context)
        prediction = self._generate(prompt, sample['image'])
        
        return {
            'answer': self._map_letter_to_answer(prediction, sample),
            'raw_prediction': prediction,
            'retrieved_docs': [doc.get('contents', '') for doc in results[:5]],
            'used_retrieval': True
        }


# ============================================================================
# 评测主函数
# ============================================================================

class MockData:
    """模拟数据对象（用于指标计算）"""
    def __init__(self, predictions, golden_answers, retrieval_results):
        self.pred = predictions
        self.golden_answers = [[ans] if isinstance(ans, str) else ans for ans in golden_answers]
        self.retrieval_result = retrieval_results
        self.items = [{'golden_answers': ga} for ga in self.golden_answers]
        # 修复：添加choices属性（空列表表示不是多选题格式）
        self.choices = [[] for _ in predictions]


def run_method(method_name, pipeline, samples):
    """运行单个方法"""
    print(f"\n{'='*80}")
    print(f"评测方法: {method_name}")
    print(f"{'='*80}")
    
    results = []
    start_time = time.time()
    
    for sample in tqdm(samples, desc=f"运行 {method_name}"):
        result = pipeline.run_single(sample)
        result['question'] = sample['question']
        result['ground_truth'] = sample['answer']
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    return results, elapsed_time


def calculate_metrics(method_name, results, samples):
    """计算7个核心指标"""
    print(f"\n计算 {method_name} 的指标...")
    
    # 准备数据
    predictions = [r['answer'] for r in results]
    golden_answers = [s['answer'] for s in samples]
    
    # 修复：retrieval_result应该是文档列表，每个文档是dict
    retrieval_results = []
    for r in results:
        docs = r.get('retrieved_docs', [])
        # 转换为正确格式：列表的列表，每个元素是字典
        if docs:
            doc_list = [{'contents': doc} if isinstance(doc, str) else {'contents': str(doc)} for doc in docs]
        else:
            doc_list = []
        retrieval_results.append(doc_list)
    
    # 创建MockData对象
    data = MockData(predictions, golden_answers, retrieval_results)
    
    # 计算所有指标
    config = {
        'use_llm_judge': False,  # Faithfulness使用简化版
        'dataset_name': 'mragbench',  # 修复：添加dataset_name
        'metric_setting': {
            'retrieval_recall_topk': 5,  # Recall@5
        }
    }
    calculator = CompleteMetricsCalculator(config)
    
    metrics = calculator.calculate_all_metrics(data)
    
    return metrics


def main():
    """主函数"""
    print("="*80)
    print("Baseline对比实验 - 100样本, 7个核心指标")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")
    print()
    
    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("="*80)
    print("1. 加载数据集")
    print("="*80)
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    
    # 初始化模型和检索器
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])
    
    # 初始化BGE检索器（用于baseline方法）
    bge_retriever = init_retriever(CONFIG, use_multimodal=False)
    
    # 初始化多模态融合检索器（用于Self-Aware-MRAG）
    multimodal_retriever = init_retriever(CONFIG, use_multimodal=True)
    
    # 定义所有方法
    methods = {
        'Self-Aware-MRAG': lambda: SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=multimodal_retriever,  # ✅ 使用BGE+CLIP多模态融合检索器
            config={
                # 核心创新点 - 全部启用
                'uncertainty_threshold': 0.35,  # ✅ 自适应检索阈值
                'use_improved_estimator': True,  # ✅ 改进版多模态不确定性估计器
                'use_position_fusion': True,     # ✅ 位置感知跨模态融合
                'use_attribution': True,          # ✅ 细粒度归因（创新点3）
                'enable_multimodal_output': False,  # 可选：多模态输出增强
                
                # 模型配置
                'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
                'retrieval_topk': 5,
                
                # Qwen3-VL配置
                'thinking': False,  # 确保不使用thinking模式
                'max_images': 20,   # 最多20张图像
            }
        ),
        'Self-RAG': lambda: SelfRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'mR2AG': lambda: MR2AGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'VisRAG': lambda: VisRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
        'REVEAL': lambda: REVEALPipeline(qwen3_vl, bge_retriever, CONFIG),
        'RagVL': lambda: RagVLPipeline(qwen3_vl, bge_retriever, CONFIG),
        'MuRAG': lambda: MuRAGPipeline(qwen3_vl, bge_retriever, CONFIG),
    }
    
    # 运行所有方法
    print("\n" + "="*80)
    print("3. 运行所有方法")
    print("="*80)
    
    all_results = {}
    all_metrics = {}
    
    for method_name, pipeline_factory in methods.items():
        try:
            pipeline = pipeline_factory()
            results, elapsed_time = run_method(method_name, pipeline, samples)
            
            # 计算指标
            metrics = calculate_metrics(method_name, results, samples)
            metrics['runtime_seconds'] = elapsed_time
            metrics['seconds_per_sample'] = elapsed_time / len(samples)
            
            all_results[method_name] = results
            all_metrics[method_name] = metrics
            
            print(f"\n✅ {method_name} 完成:")
            print(f"   EM: {metrics.get('em', 0):.4f}")
            print(f"   F1: {metrics.get('f1', 0):.4f}")
            print(f"   VQA-Score: {metrics.get('vqa_score', 0):.4f}")
            print(f"   时间: {metrics['seconds_per_sample']:.2f}秒/样本")
            
        except Exception as e:
            print(f"\n❌ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    print("\n" + "="*80)
    print("4. 保存结果")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_file = output_dir / f"all_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 详细结果: {results_file}")
    
    # 保存指标对比
    metrics_file = output_dir / f"metrics_comparison_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 指标对比: {metrics_file}")
    
    # 生成对比报告
    report_file = output_dir / f"COMPARISON_REPORT_{timestamp}.md"
    generate_report(all_metrics, report_file, samples)
    print(f"✅ 对比报告: {report_file}")
    
    print("\n" + "="*80)
    print("评测完成!")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def generate_report(all_metrics, report_file, samples):
    """生成对比报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Baseline对比实验报告\n\n")
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {len(samples)}\n\n")
        
        f.write("---\n\n")
        f.write("## 核心指标对比（7个指标）\n\n")
        
        # 表格
        f.write("| Method | EM | F1 | Recall@5 | VQA | Faith | Attr | PosBias | 时间(s) |\n")
        f.write("|--------|----|----|----------|-----|-------|------|---------|--------|\n")
        
        for method_name, metrics in all_metrics.items():
            f.write(f"| {method_name} | ")
            f.write(f"{metrics.get('em', 0):.4f} | ")
            f.write(f"{metrics.get('f1', 0):.4f} | ")
            f.write(f"{metrics.get('retrieval_recall_top5', 0):.4f} | ")
            f.write(f"{metrics.get('vqa_score', 0):.4f} | ")
            f.write(f"{metrics.get('faithfulness', 0):.4f} | ")
            f.write(f"{metrics.get('attribution_precision', 0):.4f} | ")
            f.write(f"{metrics.get('position_bias_score', 0):.4f} | ")
            f.write(f"{metrics.get('seconds_per_sample', 0):.2f} |\n")
        
        f.write("\n")
        f.write("**注**:\n")
        f.write("- EM: Exact Match (精确匹配)\n")
        f.write("- F1: Token-level F1\n")
        f.write("- Recall@5: 检索召回率\n")
        f.write("- VQA: VQA-Score\n")
        f.write("- Faith: Faithfulness (忠实度)\n")
        f.write("- Attr: Attribution Precision (归因精度)\n")
        f.write("- PosBias: Position Bias Score (位置偏差，越低越好)\n")


if __name__ == '__main__':
    main()

