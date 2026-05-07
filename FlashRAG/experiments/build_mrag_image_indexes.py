#!/usr/bin/env python3
"""为MRAG-Bench构建图像索引"""

import os
import json
import torch
from PIL import Image
import numpy as np
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
import faiss
from tqdm import tqdm

def build_clip_image_index():
    """构建MRAG-Bench图像的CLIP索引"""
    print("="*80)
    print("构建MRAG-Bench图像CLIP索引")
    print("="*80)
    print(f"开始时间: {datetime.now()}")

    # 路径配置
    image_corpus_dir = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/image_corpus"
    output_dir = "/data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench"
    os.makedirs(output_dir, exist_ok=True)

    # 加载CLIP模型
    print("\n1. 加载CLIP模型...")
    model_path = "/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    print(f"   模型: {model_path}")
    print(f"   设备: {device}")

    # 收集所有图像文件
    print("\n2. 收集图像文件...")
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for filename in os.listdir(image_corpus_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_files.append(filename)

    print(f"   找到 {len(image_files)} 个图像文件")

    # 处理图像并提取特征
    print("\n3. 提取图像特征...")
    batch_size = 32
    all_features = []
    image_metadata = []

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        images = []

        # 加载图像
        for filename in batch_files:
            try:
                image_path = os.path.join(image_corpus_dir, filename)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"   警告: 无法加载图像 {filename}: {e}")
                images.append(None)

        # 过滤有效图像
        valid_images = [img for img in images if img is not None]
        valid_filenames = [f for img, f in zip(images, batch_files) if img is not None]

        if not valid_images:
            continue

        try:
            with torch.no_grad():
                # 处理图像
                inputs = processor(
                    images=valid_images,
                    return_tensors="pt",
                    padding=True
                )

                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # 获取图像特征
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 移到CPU
                all_features.append(image_features.cpu().numpy())

                # 保存元数据
                for filename in valid_filenames:
                    image_metadata.append({
                        'filename': filename,
                        'image_id': os.path.splitext(filename)[0],  # 去掉扩展名
                        'scenario': filename.split('_')[0] if '_' in filename else 'Unknown'
                    })

        except Exception as e:
            print(f"   批次处理失败: {e}")
            continue

        print(f"   处理进度: {i+len(batch_files)}/{len(image_files)} ({(i+len(batch_files))/len(image_files)*100:.1f}%)")

    # 合并所有特征
    if all_features:
        all_features = np.vstack(all_features)
        print(f"\n✅ 图像特征维度: {all_features.shape}")
    else:
        print("\n❌ 没有成功提取任何特征")
        return

    # 构建FAISS索引
    print("\n4. 构建FAISS索引...")
    dimension = all_features.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
    index.add(all_features)
    print(f"   索引大小: {index.ntotal} 个向量")

    # 保存索引
    index_path = os.path.join(output_dir, 'clip_image_Flat.index')
    faiss.write_index(index, index_path)
    print(f"✅ 索引已保存: {index_path}")

    # 保存元数据
    metadata_path = os.path.join(output_dir, 'image_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(image_metadata, f, indent=2)
    print(f"✅ 元数据已保存: {metadata_path}")

    # 保存图像映射文件（FlashRAG格式）
    corpus_path = os.path.join(output_dir, 'image_corpus.jsonl')
    with open(corpus_path, 'w') as f:
        for meta in image_metadata:
            # 每行一个图像文档
            doc = {
                'docid': meta['image_id'],
                'image_path': os.path.join(image_corpus_dir, meta['filename']),
                'title': meta['image_id'],
                'text': f"Image from MRAG-Bench: {meta['image_id']} (Scenario: {meta['scenario']})",
                'scenario': meta['scenario']
            }
            f.write(json.dumps(doc) + '\n')
    print(f"✅ 语料库已保存: {corpus_path}")

    print("\n" + "="*80)
    print("索引构建完成！")
    print(f"结束时间: {datetime.now()}")
    print("="*80)

    return index_path, metadata_path, corpus_path

if __name__ == "__main__":
    build_clip_image_index()