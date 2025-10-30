#!/usr/bin/env python3
"""
构建纯Wikipedia 3M语料库
从 psgs_w100.tsv 中选取前300万条Wikipedia文档
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm
import sys

def load_wikipedia(tsv_path, max_docs=3000000):
    """从TSV文件加载Wikipedia文档"""
    docs = []
    print(f"📖 加载Wikipedia数据: {tsv_path}")
    print(f"   目标数量: {max_docs:,} 条")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(tqdm(reader, desc="加载Wikipedia", total=max_docs)):
            if idx >= max_docs:
                break
            
            title = row.get('title', '').strip()
            text = row.get('text', '').strip()
            
            if not title or not text:
                continue
            
            doc = {
                'id': idx,
                'source': 'wikipedia',
                'title': title,
                'contents': f"{title}\n{text}",
                'image_url': ""  # 保持一致的schema（虽然是纯文本）
            }
            docs.append(doc)
    
    print(f"✅ Wikipedia加载完成: {len(docs):,} 条")
    return docs

def save_corpus(docs, output_path):
    """保存为JSONL格式"""
    print(f"\n💾 保存语料库到: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in tqdm(docs, desc="写入文件"):
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ 语料库保存成功！")
    print(f"   文件: {output_path}")
    print(f"   大小: {output_path.stat().st_size / (1024**3):.2f} GB")
    print(f"   条目数: {len(docs):,}")

def main():
    # 路径配置
    wiki_tsv = Path("/root/autodl-tmp/data/wikipedia/psgs_w100.tsv")
    output_file = Path("/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl")
    
    print("=" * 80)
    print("构建纯Wikipedia 3M语料库")
    print("=" * 80)
    print()
    
    # 检查输入文件
    if not wiki_tsv.exists():
        print(f"❌ 错误: Wikipedia TSV文件不存在: {wiki_tsv}")
        sys.exit(1)
    
    print(f"📋 配置信息:")
    print(f"   Wikipedia TSV: {wiki_tsv}")
    print(f"   输出文件: {output_file}")
    print(f"   目标数量: 3,000,000 条")
    print()
    
    # 加载Wikipedia数据
    docs = load_wikipedia(wiki_tsv, max_docs=3000000)
    
    # 保存语料库
    save_corpus(docs, output_file)
    
    print()
    print("=" * 80)
    print("✅ 语料库构建完成！")
    print("=" * 80)
    print()
    print("📋 后续步骤:")
    print("1. 重建BGE索引:")
    print("   python tools/rebuild_index_wiki_3m.py")
    print()
    print("2. 更新实验配置中的corpus_path和index_path")
    print()

if __name__ == '__main__':
    main()
