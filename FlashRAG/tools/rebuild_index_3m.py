#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ P1-2: 基于3M语料重建索引

功能：
1. 加载corpus_3m.jsonl语料
2. 使用BGE-large-en-v1.5构建文本索引
3. 使用CLIP-ViT-L/14-336构建图像索引
4. 生成meta.json元数据

使用方法：
    python tools/rebuild_index_3m.py --corpus corpus/corpus_3m.jsonl --output indexes/3m/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flashrag.retriever.index_builder import Index_Builder


def main():
    parser = argparse.ArgumentParser(description='重建3M语料索引')
    parser.add_argument('--corpus', type=str,
                       default='/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
                       help='语料文件路径')
    parser.add_argument('--output', type=str,
                       default='/root/autodl-tmp/FlashRAG/indexes/3m/',
                       help='索引输出目录')
    parser.add_argument('--bge_model', type=str,
                       default='/root/autodl-tmp/models/bge-large-en-v1.5',
                       help='BGE模型路径')
    parser.add_argument('--clip_model', type=str,
                       default='/root/autodl-tmp/models/clip-vit-large-patch14-336',
                       help='CLIP模型路径')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批处理大小')
    parser.add_argument('--faiss_type', type=str, default='Flat',
                       choices=['Flat', 'IVF', 'IVFPQ'],
                       help='FAISS索引类型')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔨 3M语料索引重建")
    print("=" * 80)
    print(f"语料文件: {args.corpus}")
    print(f"输出目录: {args.output}")
    print(f"BGE模型: {args.bge_model}")
    print(f"CLIP模型: {args.clip_model}")
    print(f"FAISS类型: {args.faiss_type}")
    print()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查语料文件
    if not os.path.exists(args.corpus):
        print(f"❌ 语料文件不存在: {args.corpus}")
        return
    
    # 加载语料统计
    stats_file = Path(args.corpus).with_suffix('.stats.json')
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"✅ 语料统计:")
        print(f"  - 总文档数: {stats['total_docs']:,}")
        print(f"  - Wikipedia: {stats['wiki_docs']:,}")
        print(f"  - CC3M: {stats['cc3m_docs']:,}")
        print()
    
    # ========================================================================
    # 1. 构建BGE文本索引
    # ========================================================================
    print("\n" + "=" * 80)
    print("📝 构建BGE文本索引")
    print("=" * 80)
    
    bge_save_dir = output_dir / "bge"
    bge_save_dir.mkdir(exist_ok=True)
    
    # 检查索引是否已存在
    bge_index_file = bge_save_dir / f"e5_{args.faiss_type}.index"
    bge_emb_file = bge_save_dir / "emb_e5.memmap"
    
    if bge_index_file.exists() and bge_emb_file.exists():
        print(f"✅ BGE索引已存在，跳过构建")
        print(f"   - 索引文件: {bge_index_file} ({bge_index_file.stat().st_size / 1e9:.1f} GB)")
        print(f"   - 嵌入文件: {bge_emb_file} ({bge_emb_file.stat().st_size / 1e9:.1f} GB)")
    else:
        try:
            bge_builder = Index_Builder(
                retrieval_method='e5',  # BGE使用e5方法
                model_path=args.bge_model,
                corpus_path=args.corpus,
                save_dir=str(bge_save_dir),
                max_length=512,
                batch_size=args.batch_size,
                use_fp16=True,
                faiss_type=args.faiss_type,
                pooling_method='mean',
                save_embedding=True
            )
            
            print("开始构建BGE索引...")
            bge_builder.build_index()
            print("✅ BGE索引构建完成")
            
        except Exception as e:
            print(f"❌ BGE索引构建失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 2. 构建CLIP图像索引
    # ========================================================================
    print("\n" + "=" * 80)
    print("🖼️  构建CLIP图像索引")
    print("=" * 80)
    
    clip_save_dir = output_dir / "clip"
    clip_save_dir.mkdir(exist_ok=True)
    
    # 检查索引是否已存在
    clip_index_file = clip_save_dir / f"clip_{args.faiss_type}.index"
    clip_emb_file = clip_save_dir / "emb_clip.memmap"
    
    if clip_index_file.exists() and clip_emb_file.exists():
        print(f"✅ CLIP索引已存在，跳过构建")
        print(f"   - 索引文件: {clip_index_file} ({clip_index_file.stat().st_size / 1e9:.1f} GB)")
        print(f"   - 嵌入文件: {clip_emb_file} ({clip_emb_file.stat().st_size / 1e9:.1f} GB)")
    else:
        try:
            clip_builder = Index_Builder(
                retrieval_method='clip',
                model_path=args.clip_model,
                corpus_path=args.corpus,
                save_dir=str(clip_save_dir),
                max_length=77,  # CLIP文本长度
                batch_size=args.batch_size // 2,  # CLIP显存需求更大
                use_fp16=True,
                faiss_type=args.faiss_type,
                save_embedding=True,
                index_modal='all'  # 索引图像和文本
            )
            
            print("开始构建CLIP索引...")
            clip_builder.build_index()
            print("✅ CLIP索引构建完成")
            
        except Exception as e:
            print(f"❌ CLIP索引构建失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 3. 生成元数据
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 生成元数据")
    print("=" * 80)
    
    meta = {
        'index_name': '3M Mixed Corpus Index',
        'version': '1.0',
        'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'corpus_file': str(args.corpus),
        'corpus_stats': stats if stats_file.exists() else {},
        'indexes': {
            'bge': {
                'model': args.bge_model,
                'path': str(bge_save_dir),
                'faiss_type': args.faiss_type,
                'dimension': 1024,  # BGE-large dimension
                'description': 'Text retrieval using BGE-large-en-v1.5'
            },
            'clip': {
                'model': args.clip_model,
                'path': str(clip_save_dir),
                'faiss_type': args.faiss_type,
                'dimension': 768,  # CLIP ViT-L/14 dimension
                'description': 'Multimodal retrieval using CLIP-ViT-L/14-336'
            }
        }
    }
    
    meta_file = output_dir / 'meta.json'
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 元数据文件: {meta_file}")
    
    # ========================================================================
    # 4. 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 索引重建完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - BGE索引: {bge_save_dir}")
    print(f"  - CLIP索引: {clip_save_dir}")
    print(f"  - 元数据: {meta_file}")
    print()


if __name__ == '__main__':
    main()

