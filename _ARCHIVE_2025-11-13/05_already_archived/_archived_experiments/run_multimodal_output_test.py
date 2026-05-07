#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行启用多模态输出的Pipeline测试
测试MRAG 3.0的End-to-End Multimodality功能
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import yaml
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='测试多模态输出Pipeline')
    parser.add_argument('--max_samples', type=int, default=100, help='测试样本数')
    parser.add_argument('--config', default='experiments/configs/enable_multimodal_output.yaml')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 多模态输出Pipeline测试")
    print("=" * 80)
    
    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 确认多模态输出已启用
    if not config.get('use_multimodal_output'):
        print("⚠️ 警告: use_multimodal_output未启用！")
        config['use_multimodal_output'] = True
        print("✅ 已强制启用多模态输出")
    
    config['max_samples'] = args.max_samples
    
    print(f"\n配置:")
    print(f"  - 多模态输出: {config['use_multimodal_output']} ⭐")
    print(f"  - 测试样本数: {args.max_samples}")
    print(f"  - 置信度阈值: {config.get('insertion_confidence_threshold', 0.7)}")
    print(f"  - 最大图像数: {config.get('max_images_per_answer', 3)}")
    
    # 加载数据集
    print("\n加载数据集...")
    from flashrag.dataset.okvqa_dataset_lazy import OKVQADatasetLazy
    dataset = OKVQADatasetLazy(config)
    print(f"✅ 数据集加载完成: {len(dataset)}个样本")
    
    # 加载模型
    print("\n加载模型...")
    from flashrag.modules.mllm_wrapper import LLaVAWrapper
    from transformers import AutoTokenizer, AutoModel
    import faiss
    import torch
    
    llava = LLaVAWrapper(config.get('llava_model_path'))
    
    # 构建简化的BGE检索器
    print("\n构建检索器...")
    bge_path = config.get('bge_model_path', '/root/autodl-tmp/models/bge-large-en-v1.5')
    tokenizer = AutoTokenizer.from_pretrained(bge_path)
    bge_model = AutoModel.from_pretrained(bge_path).cuda().eval()
    
    # 读取少量Wikipedia数据
    docs = []
    texts = []
    wiki_file = '/root/autodl-tmp/data/wikipedia/psgs_w100.tsv'
    max_wiki = 10000  # 使用1万条数据快速测试
    
    with open(wiki_file, 'r', encoding='utf-8') as f:
        f.readline()  # 跳过header
        for i, line in enumerate(f):
            if i >= max_wiki:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                doc_id, text, title = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
                docs.append({'id': doc_id, 'text': text, 'title': title})
                texts.append(text)
    
    print(f"✅ 加载了 {len(docs)} 条Wikipedia数据")
    
    # 编码并构建FAISS索引
    print("编码文档...")
    all_embeddings = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        with torch.no_grad():
            embeddings = bge_model(**inputs).last_hidden_state[:, 0].cpu().numpy()
        all_embeddings.append(embeddings)
    
    import numpy as np
    embeddings = np.vstack(all_embeddings)
    
    # 构建FAISS索引
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"✅ FAISS索引构建完成，共 {index.ntotal} 条")
    
    # 创建检索函数
    def retrieve_fn(query_text, query_image=None, top_k=5):
        inputs = tokenizer([query_text], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        with torch.no_grad():
            query_emb = bge_model(**inputs).last_hidden_state[:, 0].cpu().numpy()
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, top_k)
        return [docs[idx] for idx in indices[0]], scores[0].tolist()
    
    class SimpleRetriever:
        def retrieve(self, query_text, query_image=None, top_k=5):
            return retrieve_fn(query_text, query_image, top_k)
    
    retriever = SimpleRetriever()
    
    # 创建Pipeline
    print("\n创建Pipeline...")
    from flashrag.pipeline.self_aware_pipeline_fixed import SelfAwareMultimodalPipeline
    pipeline = SelfAwareMultimodalPipeline(llava, retriever, config)
    
    # 运行测试
    print("\n开始测试...")
    print("=" * 80)
    
    results = []
    multimodal_count = 0
    scenarios = {}
    
    for i in tqdm(range(min(args.max_samples, len(dataset))), desc="测试进度"):
        try:
            sample = dataset[i]
            result = pipeline.run_single(sample)
            results.append(result)
            
            # 统计多模态答案
            answer = result.get('answer', '')
            if isinstance(answer, dict) and answer.get('images'):
                multimodal_count += 1
                scenario = answer.get('scenario', 'unknown')
                scenarios[scenario] = scenarios.get(scenario, 0) + 1
            
        except Exception as e:
            print(f"\n⚠️ 样本{i}处理失败: {e}")
            continue
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📊 测试结果")
    print("=" * 80)
    print(f"总样本: {len(results)}")
    print(f"多模态答案: {multimodal_count} ({multimodal_count/len(results)*100:.1f}%)")
    print(f"纯文本答案: {len(results)-multimodal_count} ({(len(results)-multimodal_count)/len(results)*100:.1f}%)")
    
    if scenarios:
        print("\n场景分布:")
        for scenario, count in sorted(scenarios.items()):
            print(f"  {scenario}: {count} ({count/multimodal_count*100:.1f}%)")
    
    # 保存结果
    output_dir = config.get('output_dir', 'experiments/multimodal_output_test')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump({
            'total_samples': len(results),
            'multimodal_count': multimodal_count,
            'multimodal_ratio': multimodal_count/len(results) if results else 0,
            'scenarios': scenarios,
            'config': config,
            'samples': results[:10]  # 只保存前10个样本示例
        }, f, indent=2, default=str)
    
    print(f"\n✅ 结果已保存: {output_dir}/results.json")
    print("=" * 80)

if __name__ == '__main__':
    main()

