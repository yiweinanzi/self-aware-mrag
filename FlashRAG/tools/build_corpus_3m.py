#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ P1-1: 3M语料构建工具

构建3M规模的混合语料：
- Wikipedia: 1.5M文档
- Conceptual Captions 3M (CC3M): 1.5M图文对

功能：
1. 下载/加载Wikipedia语料
2. 下载/加载CC3M数据集
3. 去重（基于内容哈希）
4. 统一格式（JSONL）
5. 生成统计信息

输出格式（JSONL）：
{
    "id": <int>,
    "source": "wikipedia" | "cc3m",
    "title": <str>,
    "contents": <str>,
    "image_url": <str> (仅CC3M),
    "image_path": <str> (仅CC3M，如果下载了图片)
}

使用方法：
    python tools/build_corpus_3m.py --output corpus/corpus_3m.jsonl
    python tools/build_corpus_3m.py --quick  # 快速模式（10k样本）
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from tqdm import tqdm
from collections import Counter

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import datasets
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  datasets库未安装，将使用备用方法")


class CorpusBuilder:
    """3M语料构建器"""
    
    def __init__(self, output_path: str, quick_mode: bool = False, wiki_samples: int = 1500000, cc3m_samples: int = 1500000):
        """
        初始化构建器
        
        Args:
            output_path: 输出JSONL文件路径
            quick_mode: 快速模式（减少样本数）
            wiki_samples: Wikipedia样本数
            cc3m_samples: CC3M样本数
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.quick_mode = quick_mode
        if quick_mode:
            self.wiki_samples = 5000
            self.cc3m_samples = 5000
            print("🚀 快速模式：Wiki 5k + CC3M 5k")
        else:
            self.wiki_samples = wiki_samples
            self.cc3m_samples = cc3m_samples
            print(f"📊 标准模式：Wiki {wiki_samples//1000}k + CC3M {cc3m_samples//1000}k")
        
        # 去重哈希集合
        self.content_hashes: Set[str] = set()
        
        # 统计信息
        self.stats = {
            'total_docs': 0,
            'wiki_docs': 0,
            'cc3m_docs': 0,
            'duplicates': 0,
            'total_tokens': 0,
            'avg_length': 0,
            'source_distribution': Counter(),
        }
    
    def compute_hash(self, text: str) -> str:
        """计算文本的MD5哈希"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """检查是否重复"""
        text_hash = self.compute_hash(text)
        if text_hash in self.content_hashes:
            return True
        self.content_hashes.add(text_hash)
        return False
    
    def load_wikipedia(self) -> List[Dict]:
        """
        加载Wikipedia语料（从本地TSV文件）
        
        Returns:
            list: Wikipedia文档列表
        """
        print("\n" + "="*80)
        print("📚 加载Wikipedia语料")
        print("="*80)
        
        docs = []
        
        # 本地TSV文件路径
        wiki_file = '/root/autodl-tmp/data/wikipedia/psgs_w100.tsv'
        
        if os.path.exists(wiki_file):
            print(f"从本地文件加载: {wiki_file}")
            print(f"  目标样本数: {self.wiki_samples:,}")
            
            # 读取TSV文件
            with open(wiki_file, 'r', encoding='utf-8') as f:
                # 跳过表头
                header = f.readline()
                
                # 流式处理
                count = 0
                for line in tqdm(f, total=self.wiki_samples, desc="Wikipedia"):
                    if count >= self.wiki_samples:
                        break
                    
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            continue
                        
                        doc_id = parts[0]
                        text = parts[1]
                        title = parts[2]
                        
                        if not text or len(text) < 100:
                            continue
                        
                        # 去重
                        if self.is_duplicate(text):
                            self.stats['duplicates'] += 1
                            continue
                        
                        doc = {
                            'id': len(docs),
                            'source': 'wikipedia',
                            'title': title,
                            'contents': f"{title}\n{text}",
                            'image_url': ""  # Wikipedia没有图片URL（使用空字符串保证类型一致）
                        }
                        docs.append(doc)
                        count += 1
                        
                    except Exception as e:
                        continue
            
            print(f"✅ Wikipedia: {len(docs):,} 文档")
            return docs
        
        else:
            print(f"❌ 文件不存在: {wiki_file}")
            print("  使用模拟数据（测试用）")
            for i in tqdm(range(self.wiki_samples), desc="Wikipedia (模拟)"):
                doc = {
                    'id': i,
                    'source': 'wikipedia',
                    'title': f'Wikipedia Article {i}',
                    'contents': f'Wikipedia Article {i}\nThis is a simulated Wikipedia article about topic {i}. ' * 10,
                    'image_url': ""
                }
                docs.append(doc)
            
            return docs
    
    def load_cc3m(self) -> List[Dict]:
        """
        加载Conceptual Captions 3M语料（从本地TSV文件）
        
        Returns:
            list: CC3M文档列表
        """
        print("\n" + "="*80)
        print("🖼️  加载CC3M语料")
        print("="*80)
        
        docs = []
        
        # 本地TSV文件路径
        cc3m_file = '/root/autodl-tmp/data/conceptual_captions/Train_GCC-training.tsv'
        
        if os.path.exists(cc3m_file):
            print(f"从本地文件加载: {cc3m_file}")
            print(f"  目标样本数: {self.cc3m_samples:,}")
            
            # 读取TSV文件
            with open(cc3m_file, 'r', encoding='utf-8') as f:
                # 流式处理（CC3M的TSV格式：caption \t image_url）
                count = 0
                for line in tqdm(f, total=self.cc3m_samples, desc="CC3M"):
                    if count >= self.cc3m_samples:
                        break
                    
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) < 2:
                            continue
                        
                        caption = parts[0]
                        image_url = parts[1]
                        
                        if not caption or len(caption) < 10:
                            continue
                        
                        # 去重
                        if self.is_duplicate(caption):
                            self.stats['duplicates'] += 1
                            continue
                        
                        # CC3M的title就是caption的前50字符
                        title = caption[:50] + "..." if len(caption) > 50 else caption
                        
                        doc = {
                            'id': len(docs),
                            'source': 'cc3m',
                            'title': title,
                            'contents': f"{title}\n{caption}",
                            'image_url': image_url
                        }
                        docs.append(doc)
                        count += 1
                        
                    except Exception as e:
                        continue
            
            print(f"✅ CC3M: {len(docs):,} 文档")
            return docs
        
        else:
            print(f"❌ 文件不存在: {cc3m_file}")
            print("  使用模拟数据（测试用）")
            for i in tqdm(range(self.cc3m_samples), desc="CC3M (模拟)"):
                caption = f"A photo of object {i} in setting {i % 100}"
                doc = {
                    'id': i,
                    'source': 'cc3m',
                    'title': caption,
                    'contents': f"{caption}\n{caption}",
                    'image_url': f'http://example.com/image_{i}.jpg'
                }
                docs.append(doc)
            
            return docs
    
    def merge_and_save(self, wiki_docs: List[Dict], cc3m_docs: List[Dict]):
        """
        合并并保存语料
        
        Args:
            wiki_docs: Wikipedia文档
            cc3m_docs: CC3M文档
        """
        print("\n" + "="*80)
        print("💾 合并并保存语料")
        print("="*80)
        
        # 重新分配ID
        all_docs = []
        doc_id = 0
        
        # Wikipedia
        for doc in wiki_docs:
            doc['id'] = doc_id
            all_docs.append(doc)
            doc_id += 1
        
        # CC3M
        for doc in cc3m_docs:
            doc['id'] = doc_id
            all_docs.append(doc)
            doc_id += 1
        
        # 保存为JSONL
        print(f"保存到: {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for doc in tqdm(all_docs, desc="保存"):
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # 更新统计
        self.stats['total_docs'] = len(all_docs)
        self.stats['wiki_docs'] = len(wiki_docs)
        self.stats['cc3m_docs'] = len(cc3m_docs)
        self.stats['source_distribution']['wikipedia'] = len(wiki_docs)
        self.stats['source_distribution']['cc3m'] = len(cc3m_docs)
        
        # 计算平均长度
        total_length = sum(len(doc['contents']) for doc in all_docs)
        self.stats['avg_length'] = total_length // len(all_docs) if all_docs else 0
        self.stats['total_tokens'] = total_length // 4  # 粗略估计（1 token ≈ 4 chars）
        
        print(f"✅ 保存完成: {len(all_docs):,} 文档")
    
    def generate_statistics(self):
        """生成统计报告"""
        print("\n" + "="*80)
        print("📊 语料统计")
        print("="*80)
        
        print(f"\n总文档数: {self.stats['total_docs']:,}")
        print(f"  - Wikipedia: {self.stats['wiki_docs']:,} ({self.stats['wiki_docs']/self.stats['total_docs']*100:.1f}%)")
        print(f"  - CC3M: {self.stats['cc3m_docs']:,} ({self.stats['cc3m_docs']/self.stats['total_docs']*100:.1f}%)")
        print(f"\n去重统计:")
        print(f"  - 重复文档: {self.stats['duplicates']:,}")
        print(f"\n文本统计:")
        print(f"  - 平均长度: {self.stats['avg_length']:,} 字符")
        print(f"  - 估计Token数: {self.stats['total_tokens']:,}")
        
        # 保存统计到JSON
        stats_path = self.output_path.with_suffix('.stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                **self.stats,
                'source_distribution': dict(self.stats['source_distribution']),
                'build_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'output_path': str(self.output_path)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 统计文件: {stats_path}")
        
        # 保存元数据（meta.json）
        meta_path = self.output_path.parent / 'meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'corpus_name': '3M Mixed Corpus',
                'version': '1.0',
                'total_documents': self.stats['total_docs'],
                'sources': {
                    'wikipedia': self.stats['wiki_docs'],
                    'cc3m': self.stats['cc3m_docs']
                },
                'build_date': datetime.now().strftime('%Y-%m-%d'),
                'corpus_file': self.output_path.name,
                'stats_file': stats_path.name,
                'avg_document_length': self.stats['avg_length'],
                'estimated_tokens': self.stats['total_tokens'],
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 元数据文件: {meta_path}")
    
    def build(self):
        """执行构建流程"""
        print("="*80)
        print("🏗️  3M语料构建器")
        print("="*80)
        print(f"输出路径: {self.output_path}")
        print(f"模式: {'快速' if self.quick_mode else '标准'}")
        print()
        
        # 1. 加载Wikipedia
        wiki_docs = self.load_wikipedia()
        
        # 2. 加载CC3M
        cc3m_docs = self.load_cc3m()
        
        # 3. 合并并保存
        self.merge_and_save(wiki_docs, cc3m_docs)
        
        # 4. 生成统计
        self.generate_statistics()
        
        print("\n" + "="*80)
        print("✅ 语料构建完成！")
        print("="*80)
        print(f"\n输出文件:")
        print(f"  - 语料: {self.output_path}")
        print(f"  - 统计: {self.output_path.with_suffix('.stats.json')}")
        print(f"  - 元数据: {self.output_path.parent / 'meta.json'}")
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3M语料构建工具')
    parser.add_argument('--output', type=str, 
                       default='/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
                       help='输出JSONL文件路径')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（Wiki 5k + CC3M 5k）')
    parser.add_argument('--wiki-samples', type=int, default=1500000,
                       help='Wikipedia样本数（默认1.5M）')
    parser.add_argument('--cc3m-samples', type=int, default=1500000,
                       help='CC3M样本数（默认1.5M）')
    
    args = parser.parse_args()
    
    # 检查datasets库
    if not DATASETS_AVAILABLE:
        print("⚠️  警告: datasets库未安装")
        print("   将使用模拟数据（仅用于测试）")
        print("   安装: pip install datasets")
        print()
    
    # 构建语料
    builder = CorpusBuilder(
        output_path=args.output,
        quick_mode=args.quick,
        wiki_samples=args.wiki_samples,
        cc3m_samples=args.cc3m_samples
    )
    builder.build()


if __name__ == '__main__':
    main()

