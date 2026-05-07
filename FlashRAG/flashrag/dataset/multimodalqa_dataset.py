#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiModalQA数据集加载器

数据集：MultiModalQA (Talmor et al., 2021)
规模：29,918 questions
特点：35.7% require cross-modality reasoning

数据位置：/root/autodl-tmp/FlashRAG/flashrag/data/MultiModalQA/
"""

import gzip
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

class MultiModalQADataset:
    """
    MultiModalQA数据集加载器
    
    使用示例：
    ```python
    dataset = MultiModalQADataset(
        data_dir='/root/autodl-tmp/FlashRAG/flashrag/data/MultiModalQA',
        split='dev'
    )
    
    samples = dataset.load()
    print(f"加载了 {len(samples)} 个样本")
    ```
    """
    
    def __init__(self, data_dir: str, split: str = 'dev'):
        """
        初始化
        
        Args:
            data_dir: 数据目录
            split: 'train', 'dev', 'test'
        """
        self.data_dir = data_dir
        self.split = split
        
        # 数据文件路径
        self.qa_file = os.path.join(data_dir, f'MMQA_{split}.jsonl.gz')
        self.images_file = os.path.join(data_dir, 'MMQA_images.jsonl.gz')
        self.texts_file = os.path.join(data_dir, 'MMQA_texts.jsonl.gz')
        
        print(f"✅ MultiModalQA数据集初始化")
        print(f"  Split: {split}")
        print(f"  数据目录: {data_dir}")
    
    def load(self, max_samples: int = None) -> List[Dict]:
        """
        加载数据集
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            List[Dict]: 样本列表
        """
        print(f"\n加载MultiModalQA {self.split} split...")
        
        # 1. 加载QA数据
        samples = []
        
        with gzip.open(self.qa_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="加载QA")):
                if max_samples and i >= max_samples:
                    break
                
                data = json.loads(line)
                
                # 提取基本信息
                sample = {
                    'id': data.get('id', i),
                    'question': data.get('question', ''),
                    'answers': data.get('answers', []),
                    'metadata': data.get('metadata', {}),
                    # 图像和文本信息（可能需要join）
                    'image_ids': data.get('image_ids', []),
                    'text_ids': data.get('text_ids', []),
                }
                
                samples.append(sample)
        
        print(f"✅ 加载了 {len(samples)} 个样本")
        
        return samples
    
    def __len__(self):
        """返回数据集大小"""
        # 简化版：返回估计值
        if self.split == 'train':
            return 20000
        elif self.split == 'dev':
            return 6000
        else:
            return 3918  # test
    
    def __getitem__(self, idx):
        """获取单个样本"""
        samples = self.load(max_samples=idx+1)
        return samples[idx] if idx < len(samples) else None


# 便捷函数
def load_multimodalqa(data_dir: str, split: str = 'dev', max_samples: int = None):
    """快速加载MultiModalQA"""
    dataset = MultiModalQADataset(data_dir, split)
    return dataset.load(max_samples)


if __name__ == '__main__':
    print("MultiModalQA数据集加载器")
    print("=" * 70)
    
    # 测试加载
    data_dir = '/root/autodl-tmp/FlashRAG/flashrag/data/MultiModalQA'
    
    if os.path.exists(data_dir):
        dataset = MultiModalQADataset(data_dir, 'dev')
        
        print("\n测试加载10个样本...")
        samples = dataset.load(max_samples=10)
        
        print(f"\n✅ 加载成功: {len(samples)} 个样本")
        print(f"\n示例样本:")
        if samples:
            sample = samples[0]
            print(f"  ID: {sample['id']}")
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  Answers: {sample['answers']}")
            print(f"  Image IDs: {sample.get('image_ids', [])}")
    else:
        print(f"⚠️  数据目录不存在: {data_dir}")
    
    print("=" * 70)


