#!/usr/bin/env python3
"""为MultiModalQA构建专门的索引"""

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
    """构建文本索引（包含MultiModalQA的文本和表格内容）"""
    print("\n" + "="*80)
    print("构建MultiModalQA文本索引")
    print("="*80)

    data_dir = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA"
    output_dir = "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"

    os.makedirs(output_dir, exist_ok=True)

    # 加载所有文档
    documents = []

    # 1. 加载文本文档
    print("\n1. 加载文本文档...")
    with gzip.open(os.path.join(data_dir, "MMQA_texts.jsonl.gz"), 'rt') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            doc = {
                'id': item['id'],
                'title': item.get('title', ''),
                'text': item['text'],
                'source': 'text'
            }
            documents.append(doc)
            if (i + 1) % 1000 == 0:
                print(f"   已加载 {i+1:,} 个文本文档")

    # 2. 加载表格文档（将表格转换为文本）
    print("\n2. 加载表格文档...")
    with gzip.open(os.path.join(data_dir, "MMQA_tables.jsonl.gz"), 'rt') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            table = item.get('table', {})

            # 构建表格文本描述
            table_text = []
            table_text.append(f"标题: {item.get('title', '')}")

            # 添加表头
            headers = table.get('header', [])
            if headers:
                header_names = [h.get('column_name', '') for h in headers]
                table_text.append(" | ".join(header_names))
                table_text.append("-" * 50)

            # 添加表格行
            for row in table.get('table_rows', []):
                row_text = [cell.get('text', '') if isinstance(cell, dict) else str(cell) for cell in row]
                table_text.append(" | ".join(row_text))

            doc = {
                'id': item['id'],
                'title': item.get('title', ''),
                'text': '\n'.join(table_text),
                'source': 'table'
            }
            documents.append(doc)
            if (i + 1) % 100 == 0:
                print(f"   已加载 {i+1:,} 个表格文档")

    print(f"\n总共加载 {len(documents):,} 个文档（文本 + 表格）")

    # 使用BGE编码文本
    print("\n3. 使用BGE编码文档...")
    model_path = '/data0/home/zqwang/ACL/models/bge-large-en-v1.5'
    encoder = SentenceTransformer(model_path, device='cuda')

    batch_size = 256
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [f"{doc['title']}\n\n{doc['text']}" for doc in batch]

        print(f"   编批 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}, 大小: {len(texts)}")

        with torch.no_grad():
            embeddings = encoder.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_tensor=True
            )
            all_embeddings.append(embeddings.cpu().numpy())

    # 合并所有嵌入
    all_embeddings = np.vstack(all_embeddings)
    print(f"\n✅ 文本嵌入维度: {all_embeddings.shape}")

    # 构建FAISS索引
    print("\n4. 构建FAISS索引...")
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)

    # 保存索引
    index_path = os.path.join(output_dir, 'bge_Flat.index')
    faiss.write_index(index, index_path)

    # 保存文档映射（FlashRAG格式）
    corpus_data = []
    for doc in documents:
        corpus_item = {
            'docid': doc['id'],
            'title': doc['title'],
            'text': doc['text']
        }
        corpus_data.append(corpus_item)

    # 保存为corpus格式
    corpus_path = os.path.join(output_dir, 'documents.json')
    with open(corpus_path, 'w') as f:
        for item in corpus_data:
            f.write(json.dumps(item) + '\n')

    # 同时保存映射信息
    doc_mapping = {
        'ids': [doc['id'] for doc in documents],
        'titles': [doc['title'] for doc in documents],
        'sources': [doc['source'] for doc in documents]
    }
    with open(os.path.join(output_dir, 'doc_mapping.json'), 'w') as f:
        json.dump(doc_mapping, f, indent=2)

    print(f"\n✅ 文本索引构建完成:")
    print(f"   - 索引文件: {index_path}")
    print(f"   - 文档数: {len(documents)}")
    print(f"   - 维度: {dimension}")

    return documents


def build_clip_index(documents):
    """构建CLIP索引（包含文本和图像描述）"""
    print("\n" + "="*80)
    print("构建MultiModalQA CLIP索引")
    print("="*80)

    output_dir = "/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"

    # 加载CLIP模型
    print("\n1. 加载CLIP模型...")
    model_path = "/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(model_path).cuda()
    processor = CLIPProcessor.from_pretrained(model_path)  # CLIPProcessor不需要.cuda()
    print(f"   模型: {model_path}")

    # 准备文本
    print("\n2. 准备文本和图像路径...")
    image_dir = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/images/final_dataset_images"

    texts = []
    image_paths = []
    valid_doc_ids = []

    for doc in documents:
        # 文本：标题 + 内容的精简版本
        text = f"Document: {doc['title']}"
        if doc['source'] == 'text':
            text = f"{text}\n{doc['text'][:200]}..."  # 文本只保留前200字符
        elif doc['source'] == 'table':
            # 表格提取关键信息
            lines = doc['text'].split('\n')
            text = f"{text}\n" + "\n".join(lines[1:5])  # 保留表头和前几行

        texts.append(text)

        # 检查是否有相关图像
        if 'image_doc_ids' in doc['id'] or doc['source'] == 'table':
            # 对于有图像的文档，我们需要找到对应的图像
            # 这里简化处理，使用一个占位符或描述
            image_path = f"no_image_{doc['id']}.jpg"
        else:
            image_path = None

        image_paths.append(image_path)
        valid_doc_ids.append(doc['id'])

    print(f"   文本数: {len(texts)}")
    print(f"   图像数: {len([p for p in image_paths if p and not p.startswith('no_image')])}")

    # 创建图像描述（占位符）
    image_descriptions = []
    for i, (text, image_path) in enumerate(zip(texts, image_paths)):
        if image_path and image_path.startswith('no_image'):
            # 为无图像文档创建描述
            desc = f"This document contains information about {text[:50]}..."
        else:
            desc = text
        image_descriptions.append(desc)

    # 使用CLIP编码
    print("\n3. 使用CLIP编码...")
    batch_size = 64

    with torch.no_grad():
        # 编码文本
        text_inputs = processor(
            texts=texts[:len(image_descriptions)],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(model.device)

        text_features = model.get_text_features(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    print(f"✅ 文本特征维度: {text_features.shape}")

    # 构建FAISS索引
    print("\n4. 构建FAISS索引...")
    dimension = text_features.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(text_features)

    # 保存索引
    index_path = os.path.join(output_dir, 'clip_Flat.index')
    faiss.write_index(index, index_path)

    # 保存映射信息
    mapping = {
        'doc_ids': valid_doc_ids,
        'texts': texts,
        'image_paths': image_paths,
        'image_descriptions': image_descriptions
    }

    with open(os.path.join(output_dir, 'clip_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✅ CLIP索引构建完成:")
    print(f"   - 索引文件: {index_path}")
    print(f"   - 文本数: {len(texts)}")
    print(f"   - 维度: {dimension}")


def main():
    """主函数"""
    print("="*80)
    print("MultiModalQA索引构建工具")
    print(f"开始时间: {datetime.now()}")
    print("="*80)

    # 构建文本索引
    documents = build_text_index()

    # 构建CLIP索引
    build_clip_index(documents)

    print("\n" + "="*80)
    print("索引构建完成！")
    print(f"结束时间: {datetime.now()}")
    print("="*80)
    print("\n使用方法：")
    print("python run_all_baselines_MultimodalVQA_multimodal.py \\")
    print("    --use_multimodal_retrieval \\")
    print("    --index_path /data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index \\")
    print("    --clip_index_path /data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/clip_Flat.index")
    print("="*80)


if __name__ == "__main__":
    main()