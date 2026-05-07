#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试Our Method（修正阈值后）"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from flashrag.dataset.okvqa_dataset_lazy import OKVQADatasetLazy
from flashrag.modules.mllm_wrapper import LLaVAWrapper
from flashrag.modules.uncertainty_estimator import CrossModalUncertaintyEstimator
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import faiss

print("="*70)
print("🔬 快速测试 Our Method（修正阈值=0.43）")
print("="*70)

# 加载模型
print("\n📦 加载模型...")
llava = LLaVAWrapper('/root/autodl-tmp/models/llava-v1.5-7b', device='cuda')

bge_path = '/root/autodl-tmp/models/bge-large-en-v1.5'
bge_tokenizer = AutoTokenizer.from_pretrained(bge_path)
bge_model = AutoModel.from_pretrained(bge_path).to('cuda').eval()

# 构建检索器（简化版，只用10万条）
print("\n📚 构建检索器（100,000条）...")
docs = []
texts = []
wiki_file = '/root/autodl-tmp/data/wikipedia/psgs_w100.tsv'

with open(wiki_file, 'r', encoding='utf-8') as f:
    f.readline()
    for i, line in enumerate(tqdm(f, total=100000, desc="读取")):
        if i >= 100000:
            break
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            docs.append({'id': f'wiki_{i}', 'text': parts[1]})
            texts.append(parts[1])

all_embs = []
for i in tqdm(range(0, len(texts), 256), desc="BGE编码"):
    batch = texts[i:i+256]
    inputs = bge_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bge_model(**inputs)
        all_embs.append(outputs.last_hidden_state[:, 0, :].cpu())

all_embs = torch.cat(all_embs, 0).numpy().astype('float32')
index = faiss.IndexFlatIP(all_embs.shape[1])
faiss.normalize_L2(all_embs)
index.add(all_embs)
print(f"✅ 索引完成: {index.ntotal:,} 条")

def retrieve_fn(question, topk=5):
    inputs = bge_tokenizer([question], padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bge_model(**inputs)
        q_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, topk)
    return [docs[idx]['text'] for idx in indices[0]], scores[0].tolist()

# 加载数据集
print("\n📂 加载数据集...")
dataset = OKVQADatasetLazy({
    'data_dir': 'flashrag/data/VQA',
    'split': 'val',
    'load_images': True
})
samples = [dataset[i] for i in range(100)]
print(f"✅ 数据集: {len(samples)} 样本")

# Our Method配置
print("\n🔧 配置Our Method（68.90%配置）")
print(f"  阈值: 0.43 (对应8%检索率)")

uncertainty = CrossModalUncertaintyEstimator(
    mllm_model=None,
    config={
        'eigen_threshold': -6.0,
        'use_clip_for_alignment': True,
        'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
        'text_weight': 0.4,
        'visual_weight': 0.3,
        'alignment_weight': 0.3
    }
)

# 运行评测
print("\n"+"="*70)
print("开始评测...")
print("="*70)

correct = 0
retrieved_count = 0

for sample in tqdm(samples, desc="Our Method"):
    question = sample['question']
    image = sample.get('image')
    
    # 不确定性估计
    unc = uncertainty.estimate(question, image)
    should_retrieve = unc.get('total', 0.5) > 0.43
    
    if should_retrieve:
        retrieved_count += 1
        retrieved_docs, scores = retrieve_fn(question, 5)
        retrieved_docs = retrieved_docs[:3]  # Position Fusion
        context = "\n\n".join([f"[Doc {i+1}] {doc[:200]}" for i, doc in enumerate(retrieved_docs)])
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    
    # 生成
    answer = llava.generate(text=prompt, image=image, max_new_tokens=20, temperature=0.2)
    
    # 评估
    if any(g.lower().strip() in answer.lower().strip() for g in sample.get('golden_answers', [])):
        correct += 1

acc = correct / len(samples)
ret_rate = retrieved_count / len(samples)

print("\n"+"="*70)
print("📊 结果")
print("="*70)
print(f"准确率: {acc*100:.2f}% ({correct}/{len(samples)})")
print(f"检索率: {ret_rate*100:.1f}% ({retrieved_count}/{len(samples)})")

if acc >= 0.65:
    print(f"\n✅ 成功！准确率恢复到 {acc*100:.1f}% (接近68%)")
else:
    print(f"\n⚠️  准确率 {acc*100:.1f}% 仍然偏低，可能需要进一步调整")

print("="*70)
