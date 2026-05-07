#!/usr/bin/env python3
"""
为纯Wikipedia 3M语料库重建BGE索引
"""

import sys
from pathlib import Path

# 添加FlashRAG路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashrag.retriever.index_builder import Index_Builder
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='corpus/corpus_wiki_3m.jsonl', help='语料库路径')
    parser.add_argument('--output', default='indexes/wiki_3m/bge', help='输出目录')
    parser.add_argument('--faiss-type', default='Flat', choices=['Flat', 'HNSW'], help='FAISS索引类型')
    parser.add_argument('--batch-size', type=int, default=512, help='批处理大小')
    args = parser.parse_args()
    
    # 路径设置
    corpus_path = Path(__file__).parent.parent / args.corpus
    output_dir = Path(__file__).parent.parent / args.output
    
    print("=" * 80)
    print("为纯Wikipedia 3M语料库构建BGE索引")
    print("=" * 80)
    print(f"语料库: {corpus_path}")
    print(f"输出目录: {output_dir}")
    print(f"索引类型: {args.faiss_type}")
    print("=" * 80)
    print()
    
    # 检查语料库
    if not corpus_path.exists():
        print(f"❌ 错误: 语料库文件不存在: {corpus_path}")
        sys.exit(1)
    
    # 加载语料库
    print("📖 加载语料库...")
    corpus_dataset = load_dataset('json', data_files=str(corpus_path), split='train')
    print(f"✅ 语料库加载成功: {len(corpus_dataset):,} 条")
    print()
    
    # BGE配置
    model_path = "/root/autodl-tmp/models/bge-large-en-v1.5"
    bge_save_dir = output_dir
    bge_save_dir.mkdir(parents=True, exist_ok=True)
    
    bge_index_file = bge_save_dir / f"e5_{args.faiss_type}.index"
    bge_emb_file = bge_save_dir / f"emb_e5.memmap"
    
    # 检查是否已存在
    if bge_index_file.exists() and bge_emb_file.exists():
        print(f"⚠️  BGE索引已存在，是否覆盖？")
        print(f"   - 索引文件: {bge_index_file}")
        print(f"   - 嵌入文件: {bge_emb_file}")
        response = input("输入 'yes' 继续覆盖，或按Enter跳过: ")
        if response.lower() != 'yes':
            print("✅ 跳过索引构建")
            return
        print("⚠️  将覆盖现有索引")
        print()
    
    # 构建BGE索引
    print("=" * 80)
    print("🔨 构建BGE文本索引")
    print("=" * 80)
    print(f"模型: {model_path}")
    print(f"语料库大小: {len(corpus_dataset):,} 条")
    print(f"批处理大小: {args.batch_size}")
    print()
    
    builder = Index_Builder(
        retrieval_method="e5",
        model_path=model_path,
        corpus_path=str(corpus_path),
        save_dir=str(bge_save_dir),
        max_length=512,
        batch_size=args.batch_size,
        use_fp16=True,
        pooling_method='cls',
        faiss_type=args.faiss_type
    )
    
    # 开始构建
    print("开始构建索引...")
    builder.build_index()
    
    # 验证输出
    print()
    print("=" * 80)
    print("✅ 索引构建完成！")
    print("=" * 80)
    print(f"索引文件: {bge_index_file}")
    print(f"  大小: {bge_index_file.stat().st_size / (1024**3):.2f} GB")
    print(f"嵌入文件: {bge_emb_file}")
    print(f"  大小: {bge_emb_file.stat().st_size / (1024**3):.2f} GB")
    print()

if __name__ == '__main__':
    main()
