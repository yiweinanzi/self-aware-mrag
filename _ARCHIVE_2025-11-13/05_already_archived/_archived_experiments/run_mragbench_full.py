#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRAG-Bench完整评测脚本
使用500万混合语料库（Wikipedia + Conceptual Captions）
计算所有指标
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import argparse
import time
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='MRAG-Bench评测')
    parser.add_argument('--max_samples', type=int, default=1353, help='最大样本数')
    parser.add_argument('--max_wiki', type=int, default=5000000, help='Wikipedia条数')
    parser.add_argument('--max_cc', type=int, default=0, help='Conceptual Captions条数（0表示用剩余配额）')
    parser.add_argument('--topk', type=int, default=5, help='检索Top-K')
    parser.add_argument('--output_dir', type=str, default='experiments/mragbench_500w_results')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.43)
    return parser.parse_args()


def load_mragbench_dataset(max_samples=None):
    """加载MRAG-Bench数据集"""
    print("\n" + "="*80)
    print("📂 加载MRAG-Bench数据集")
    print("="*80)
    
    data_file = '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/test.json'
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # 尝试加载图像
    try:
        from datasets import load_from_disk
        raw_dataset = load_from_disk('/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw')
        
        # 匹配图像到样本
        for i, sample in enumerate(data):
            if i < len(raw_dataset['test']):
                raw_sample = raw_dataset['test'][i]
                if 'image' in raw_sample:
                    sample['image'] = raw_sample['image']
        
        print(f"✅ 成功加载图像")
    except Exception as e:
        print(f"⚠️ 图像加载失败: {e}")
        print("   将使用纯文本模式")
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"✅ 数据集加载完成: {len(data)} 个样本")
    
    # 统计场景分布
    scenarios = defaultdict(int)
    aspects = defaultdict(int)
    for sample in data:
        scenarios[sample.get('scenario', 'unknown')] += 1
        aspects[sample.get('aspect', 'unknown')] += 1
    
    print(f"\n场景分布:")
    for scenario, count in sorted(scenarios.items()):
        print(f"  {scenario}: {count}")
    
    print(f"\n方面分布:")
    for aspect, count in sorted(aspects.items()):
        print(f"  {aspect}: {count}")
    
    return data


def build_mixed_corpus(args):
    """构建混合检索库（Wikipedia + Conceptual Captions）"""
    print("\n" + "="*80)
    print("📚 构建混合检索库")
    print("="*80)
    
    from transformers import AutoTokenizer, AutoModel
    import faiss
    
    # 加载BGE模型
    print("加载BGE模型...")
    bge_path = '/root/autodl-tmp/models/bge-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(bge_path)
    bge_model = AutoModel.from_pretrained(bge_path).cuda().eval()
    print("✅ BGE模型加载完成")
    
    # 1. 加载Wikipedia
    print(f"\n📖 加载Wikipedia（目标: {args.max_wiki:,} 条）...")
    wiki_docs = []
    wiki_texts = []
    wiki_file = '/root/autodl-tmp/data/wikipedia/psgs_w100.tsv'
    
    with open(wiki_file, 'r', encoding='utf-8') as f:
        f.readline()  # 跳过header
        for i, line in enumerate(tqdm(f, desc="读取Wikipedia", total=args.max_wiki)):
            if i >= args.max_wiki:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                doc_id, text = parts[0], parts[1]
                title = parts[2] if len(parts) > 2 else ""
                wiki_docs.append({
                    'id': f'wiki_{doc_id}',
                    'text': text,
                    'title': title,
                    'source': 'wikipedia'
                })
                wiki_texts.append(text)
    
    print(f"✅ Wikipedia加载完成: {len(wiki_docs):,} 条")
    
    # 2. 加载Conceptual Captions
    cc_docs = []
    cc_texts = []
    
    # 确定加载数量：如果指定了max_cc则用指定值，否则加载全部（不限制）
    if args.max_cc > 0:
        max_cc = args.max_cc
    else:
        # 加载全部CC（约330万条）
        max_cc = float('inf')
    
    print(f"\n🖼️ 加载Conceptual Captions（{'全部' if max_cc == float('inf') else f'目标: {max_cc:,} 条'}）...")
    cc_file = '/root/autodl-tmp/data/conceptual_captions/Train_GCC-training.tsv'
    
    try:
        
        with open(cc_file, 'r', encoding='utf-8') as f:
            total_hint = int(max_cc) if max_cc != float('inf') else None
            for i, line in enumerate(tqdm(f, desc="读取CC", total=total_hint)):
                if max_cc != float('inf') and i >= max_cc:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    caption, url = parts[0], parts[1]
                    cc_docs.append({
                        'id': f'cc_{i}',
                        'text': caption,
                        'title': 'Image Caption',
                        'source': 'conceptual_captions',
                        'url': url
                    })
                    cc_texts.append(caption)
        
        print(f"✅ Conceptual Captions加载完成: {len(cc_docs):,} 条")
    except Exception as e:
        print(f"⚠️ Conceptual Captions加载失败: {e}")
        print("   将只使用Wikipedia")
    
    # 3. 合并
    all_docs = wiki_docs + cc_docs
    all_texts = wiki_texts + cc_texts
    
    print(f"\n📊 语料库统计:")
    print(f"  Wikipedia: {len(wiki_docs):,} 条 ({len(wiki_docs)/len(all_docs)*100:.1f}%)")
    print(f"  Conceptual Captions: {len(cc_docs):,} 条 ({len(cc_docs)/len(all_docs)*100:.1f}%)")
    print(f"  总计: {len(all_docs):,} 条")
    
    # 4. 编码
    print(f"\n🔄 编码文档（预计时间: ~{len(all_docs)//5000:.0f}分钟）...")
    all_embeddings = []
    batch_size = 128
    
    start_time = time.time()
    for i in tqdm(range(0, len(all_texts), batch_size), desc="编码进度"):
        batch_texts = all_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors='pt').to('cuda')
        with torch.no_grad():
            embeddings = bge_model(**inputs).last_hidden_state[:, 0].cpu().numpy()
        all_embeddings.append(embeddings)
        
        # 进度报告
        if (i // batch_size) % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            progress = i / len(all_texts)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  已编码: {i:,}/{len(all_texts):,} ({progress*100:.1f}%), "
                  f"耗时: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
    
    embeddings = np.vstack(all_embeddings)
    print(f"✅ 编码完成，耗时: {(time.time()-start_time)/60:.1f}分钟")
    
    # 5. 构建FAISS索引
    print("\n🔍 构建FAISS索引...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"✅ FAISS索引构建完成: {index.ntotal:,} 条")
    
    # 创建检索函数
    def retrieve_fn(query_text, query_image=None, top_k=5):
        inputs = tokenizer([query_text], padding=True, truncation=True, 
                          max_length=512, return_tensors='pt').to('cuda')
        with torch.no_grad():
            query_emb = bge_model(**inputs).last_hidden_state[:, 0].cpu().numpy()
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, top_k)
        return [all_docs[idx] for idx in indices[0]], scores[0].tolist()
    
    return retrieve_fn, {'bge': bge_model, 'tokenizer': tokenizer}, all_docs


def load_models():
    """加载LLaVA和CLIP模型"""
    print("\n" + "="*80)
    print("🤖 加载模型")
    print("="*80)
    
    from flashrag.modules.mllm_wrapper import LLaVAWrapper
    
    print("加载LLaVA模型...")
    llava = LLaVAWrapper('/root/autodl-tmp/models/llava-v1.5-7b', device='cuda')
    print("✅ LLaVA加载完成")
    
    return {'llava': llava}


def run_evaluation(models, samples, retrieve_fn, args):
    """运行评测（使用最佳配置）"""
    print("\n" + "="*80)
    print("🚀 开始评测")
    print("="*80)
    
    print(f"\n配置:")
    print(f"  - 不确定性阈值: {args.uncertainty_threshold}")
    print(f"  - Top-K: {args.topk}")
    print(f"  - 文本权重: 0.4")
    print(f"  - 视觉权重: 0.3")
    print(f"  - 对齐权重: 0.3")
    
    # 初始化不确定性估计器
    from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
    from flashrag.modules.attribution import FineGrainedMultimodalAttribution
    from flashrag.modules.position_aware_fusion import PositionAwareCrossModalFusion
    
    uncertainty_estimator = CrossModalUncertaintyEstimator(
        mllm_model=models['llava'],
        config={
            'eigen_threshold': -6.0,
            'use_clip_for_alignment': True,
            'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
            'text_weight': 0.4,
            'visual_weight': 0.3,
            'alignment_weight': 0.3
        }
    )
    
    attribution_module = FineGrainedMultimodalAttribution(mllm_model=None)
    position_fusion = PositionAwareCrossModalFusion(
        mllm_model=models['llava'],
        config={'device': 'cpu'}
    )
    
    results = []
    metrics = {
        'total': 0,
        'correct': 0,
        'retrieved': 0,
        'uncertainties': [],
        'retrieval_triggered': [],
        'by_scenario': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_aspect': defaultdict(lambda: {'total': 0, 'correct': 0})
    }
    
    for sample in tqdm(samples, desc="评测进度"):
        question = sample['question']
        image = sample.get('image')
        scenario = sample.get('scenario', 'unknown')
        aspect = sample.get('aspect', 'unknown')
        
        # 1. 不确定性估计
        unc_info = uncertainty_estimator.estimate(question, image)
        total_unc = unc_info.get('total', 0.5)
        metrics['uncertainties'].append(total_unc)
        
        # 2. 自适应检索
        should_retrieve = total_unc > args.uncertainty_threshold
        metrics['retrieval_triggered'].append(should_retrieve)
        
        if should_retrieve:
            # 检索
            docs, scores = retrieve_fn(question, image, args.topk)
            metrics['retrieved'] += 1
            
            # 位置融合
            if docs:
                fused_docs, fused_scores = position_fusion.rerank_with_position_weighting(
                    docs, scores, question
                )
            else:
                fused_docs, fused_scores = [], []
            
            # 构建上下文
            if fused_docs:
                context = "\n\n".join([
                    f"[{i+1}] {doc.get('title', 'Document')}: {doc.get('text', '')}"
                    for i, doc in enumerate(fused_docs[:3])
                ])
            else:
                context = ""
        else:
            docs, fused_docs, fused_scores = [], [], []
            context = ""
        
        # 3. 生成答案
        prompt = f"Question: {question}\n"
        if context:
            prompt += f"\nContext:\n{context}\n\n"
        prompt += "Answer:"
        
        answer = models['llava'].generate(prompt, image)
        
        # 4. 归因（如果有检索）
        attributions = None
        if fused_docs:
            try:
                retrieved_texts = [doc.get('text', '') for doc in fused_docs]
                attributions = attribution_module.attribute_text_evidence(
                    generated_text=answer,
                    retrieved_texts=retrieved_texts
                )
            except:
                attributions = None
        
        # 5. 评估（简化版，实际需要答案标注）
        # MRAG-Bench的评估需要人工标注或特定评估脚本
        # 这里我们只记录结果
        
        result = {
            'question': question,
            'answer': answer,
            'scenario': scenario,
            'aspect': aspect,
            'uncertainty': total_unc,
            'retrieved': should_retrieve,
            'n_docs': len(fused_docs),
            'attributions': attributions
        }
        results.append(result)
        
        metrics['total'] += 1
        metrics['by_scenario'][scenario]['total'] += 1
        metrics['by_aspect'][aspect]['total'] += 1
    
    # 计算最终指标
    final_metrics = {
        'total_samples': metrics['total'],
        'retrieval_rate': metrics['retrieved'] / metrics['total'] if metrics['total'] > 0 else 0,
        'avg_uncertainty': np.mean(metrics['uncertainties']) if metrics['uncertainties'] else 0,
        'uncertainty_std': np.std(metrics['uncertainties']) if metrics['uncertainties'] else 0,
        'by_scenario': dict(metrics['by_scenario']),
        'by_aspect': dict(metrics['by_aspect'])
    }
    
    return results, final_metrics


def main():
    args = parse_args()
    
    print("="*80)
    print("🔬 MRAG-Bench完整评测")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {args.max_samples}")
    print(f"语料库: Wikipedia({args.max_wiki:,}) + CC(自动补充到500万)")
    print(f"阈值: {args.uncertainty_threshold}")
    
    start_time = time.time()
    
    # 1. 加载数据集
    samples = load_mragbench_dataset(args.max_samples)
    
    # 2. 构建检索库
    retrieve_fn, retrieval_models, corpus = build_mixed_corpus(args)
    
    # 3. 加载模型
    models = load_models()
    models.update(retrieval_models)
    
    # 4. 运行评测
    results, metrics = run_evaluation(models, samples, retrieve_fn, args)
    
    # 5. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'metrics': metrics,
            'results': results,
            'corpus_stats': {
                'total_docs': len(corpus),
                'sources': {
                    'wikipedia': sum(1 for d in corpus if d['source'] == 'wikipedia'),
                    'conceptual_captions': sum(1 for d in corpus if d['source'] == 'conceptual_captions')
                }
            }
        }, f, ensure_ascii=False, indent=2)
    
    # 6. 生成报告
    elapsed = time.time() - start_time
    
    report = f"""
{'='*80}
📊 MRAG-Bench评测结果
{'='*80}

⏱️ 时间统计:
  - 总耗时: {elapsed/3600:.2f} 小时
  - 平均每样本: {elapsed/len(samples):.2f} 秒

📈 整体指标:
  - 总样本数: {metrics['total_samples']}
  - 检索率: {metrics['retrieval_rate']*100:.1f}%
  - 平均不确定性: {metrics['avg_uncertainty']:.4f} ± {metrics['uncertainty_std']:.4f}

📚 语料库统计:
  - 总文档数: {len(corpus):,}
  - Wikipedia: {sum(1 for d in corpus if d['source'] == 'wikipedia'):,}
  - Conceptual Captions: {sum(1 for d in corpus if d['source'] == 'conceptual_captions'):,}

📋 场景分布:
"""
    
    for scenario, stats in metrics['by_scenario'].items():
        report += f"  {scenario}: {stats['total']} 样本\n"
    
    report += "\n📋 方面分布:\n"
    for aspect, stats in metrics['by_aspect'].items():
        report += f"  {aspect}: {stats['total']} 样本\n"
    
    report += f"\n✅ 结果已保存: {output_file}\n"
    report += "="*80 + "\n"
    
    print(report)
    
    # 保存报告
    report_file = os.path.join(args.output_dir, 'REPORT.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 报告已保存: {report_file}")


if __name__ == '__main__':
    main()

