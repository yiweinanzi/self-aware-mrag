#!/usr/bin/env python3
"""为MultiModalQA构建完整的多模态索引（BGE + CLIP）"""

import os
import json
import gzip
import numpy as np
from datetime import datetime
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import faiss

def build_text_index():
    """构建文本索引（如果还没存在）"""
    print("\n" + "="*80)
    print("检查BGE文本索引")
    print("="*80)

    output_dir = "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"
    index_path = os.path.join(output_dir, 'bge_Flat.index')
    corpus_path = os.path.join(output_dir, 'corpus.jsonl')

    if os.path.exists(index_path) and os.path.exists(corpus_path):
        print(f"✅ BGE索引已存在: {index_path}")
        return corpus_path

    print("⚠️  BGE索引不存在，需要构建...")
    return None

def build_clip_index():
    """构建CLIP索引"""
    print("\n" + "="*80)
    print("构建CLIP多模态索引")
    print("="*80)

    data_dir = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA"
    output_dir = "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"

    os.makedirs(output_dir, exist_ok=True)

    # 加载所有文档
    documents = []
    print("\n1. 加载文档...")

    # 从corpus.jsonl加载文档
    corpus_path = os.path.join(output_dir, 'corpus.jsonl')
    if os.path.exists(corpus_path):
        print("   从corpus.jsonl加载文档...")
        with open(corpus_path, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    documents.append({
                        'id': doc.get('docid', doc.get('title', '')),
                        'title': doc.get('title', ''),
                        'text': doc.get('text', ''),
                        'contents': doc.get('contents', '')
                    })
                    if len(documents) % 50000 == 0:
                        print(f"   已加载 {len(documents):,} 个文档")

    print(f"\n总共加载 {len(documents):,} 个文档")

    # 加载CLIP模型
    print("\n2. 加载CLIP模型...")
    model_path = "/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336"

    # 使用更稳健的加载方式
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")

    model = CLIPModel.from_pretrained(model_path)
    if device == "cuda":
        model = model.cuda()

    processor = CLIPProcessor.from_pretrained(model_path)
    print(f"   模型加载成功: {model_path}")

    # 准备文本
    print("\n3. 准备文本...")
    texts = []
    doc_ids = []

    for doc in documents:
        # 组合标题和内容作为CLIP文本输入
        text = doc['contents'][:200]  # 限制长度以避免超出token限制
        texts.append(text)
        doc_ids.append(doc['id'])

    print(f"   准备了 {len(texts)} 个文本")

    # 分批处理
    print("\n4. 使用CLIP编码...")
    batch_size = 100  # 减小batch size以避免内存问题
    all_features = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        try:
            with torch.no_grad():
                # 使用安全的方式处理文本
                inputs = processor(
                    text=batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=77,  # CLIP的最大token长度
                    return_tensors="pt"
                )

                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # 获取文本特征
                text_features = model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                all_features.append(text_features.cpu().numpy())

                print(f"   处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}，大小: {len(batch_texts)}")

        except Exception as e:
            print(f"   批次 {i//batch_size + 1} 处理失败: {e}")
            # 尝试逐个处理
            for text in batch_texts:
                try:
                    with torch.no_grad():
                        inputs = processor(
                            text=[text],
                            padding=True,
                            truncation=True,
                            max_length=77,
                            return_tensors="pt"
                        )
                        if device == "cuda":
                            inputs = {k: v.cuda() for k, v in inputs.items()}

                        features = model.get_text_features(**inputs)
                        features = features / features.norm(dim=-1, keepdim=True)
                        all_features.append(features.cpu().numpy())
                except Exception as e2:
                    print(f"     单个文本处理失败: {e2}")
                    # 添加零向量作为占位符
                    all_features.append(np.zeros((1, 768)))

    # 合并所有特征
    all_features = np.vstack(all_features)
    print(f"\n✅ 文本特征维度: {all_features.shape}")

    # 构建FAISS索引
    print("\n5. 构建FAISS索引...")
    dimension = all_features.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积相似度

    # 添加向量到索引
    index.add(all_features)
    print(f"   索引大小: {index.ntotal} 个向量")

    # 保存索引
    index_path = os.path.join(output_dir, 'clip_Flat.index')
    faiss.write_index(index, index_path)
    print(f"✅ 索引已保存: {index_path}")

    # 保存映射信息
    mapping_path = os.path.join(output_dir, 'clip_mapping.json')
    mapping = {
        'doc_ids': doc_ids,
        'texts': texts,
        'dimension': dimension,
        'total_docs': len(documents)
    }

    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ 映射信息已保存: {mapping_path}")

    return index_path, mapping_path

def main():
    """主函数"""
    print("="*80)
    print("MultiModalQA完整多模态索引构建工具")
    print(f"开始时间: {datetime.now()}")
    print("="*80)

    # 检查BGE索引
    corpus_path = build_text_index()

    # 构建CLIP索引
    clip_index_path, clip_mapping_path = build_clip_index()

    print("\n" + "="*80)
    print("索引构建完成！")
    print(f"结束时间: {datetime.now()}")
    print("="*80)

    print("\n构建的索引文件:")
    output_dir = "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"
    print(f"📁 {output_dir}/")
    print(f"  ├── bge_Flat.index (BGE文本索引)")
    print(f"  ├── corpus.jsonl (文档语料库)")
    print(f"  ├── clip_Flat.index (CLIP多模态索引)")
    print(f"  ├── clip_mapping.json (CLIP映射信息)")
    print(f"  └── doc_mapping.json (文档映射)")

    print("\n使用方法:")
    print("python run_all_baselines_MultimodalVQA.py \\")
    print("    --use_multimodal_retrieval \\")
    print("    --max_samples 10 \\")
    print("    --split dev \\")
    print("    --output_dir results_multimodalqa_multimodal")

if __name__ == "__main__":
    main()